"""
Simulation script to generate the Defense-Service Tradeoff Curve.
Sweeps static IDS CPU allocation eta and compares against an adaptive RL policy.
"""
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from environment import build_env_base
from eval_common import BaseEvaluator, run_episode
from method_policy import ConstantPolicy, make_baseline_policy

import copy
import os
import tempfile
from eval_common import PROPOSED_CONFIGS

class TradeoffEvaluator(BaseEvaluator):
    def mutate_cfg(self, cfg: dict, offload_mode: str = "delay_workload", acc_model: str = "gm") -> dict:
        cfg_mut = copy.deepcopy(cfg)
        cfg_mut["globals"]["offload_mode"] = offload_mode
        if "accuracy_matrix" in cfg_mut.get("globals", {}):
            cfg_mut["globals"]["accuracy_matrix"]["model"] = acc_model
        return cfg_mut

    def run_sweep(self, eta_steps=11):
        etas = np.linspace(0.0, 0.6, eta_steps)
        results = []

        # We assume all edge areas have the same budget for simplicity in the sweep label,
        # but the policy handles per-edge budgets correctly.
        env = build_env_base(self.args.cfg)
        max_cpu = env.edge_areas[0].budget.cpu
        
        for eta in tqdm(etas, desc="Sweeping eta"):
            cpu_val = eta * max_cpu
            policy = ConstantPolicy(cpu_val)
            label = f"static_{eta:.2f}"
            
            res_dict, _ = self.run_simulation(env, self.cfg_original, policy, label)
            metrics = self.extract_metrics(res_dict)
            
            results.append({
                "eta": eta,
                "residual_atk": metrics["atk_leak"],
                "slo_vio": metrics["slo_vio"],
                "bcd": metrics["bcd"]
            })
            
        return results

    def run_adaptive(self):
        env = build_env_base(self.args.cfg)
        # Use MA LSTM RL as the proposed method
        policy = make_baseline_policy(
            "ma_lstm_rl", 
            ckpt_path=self.args.adaptive_ckpt,
            device=self.args.device
        )
        label = "adaptive_rl"
        res_dict, _ = self.run_simulation(env, self.cfg_original, policy, label)
        return self.extract_metrics(res_dict)

def plot_tradeoff(sweep_results, adaptive_metrics, outpath):
    etas = [r["eta"] for r in sweep_results]
    res_atk = [r["residual_atk"] for r in sweep_results]
    slo_vio = [r["slo_vio"] for r in sweep_results]
    bcd = [r["bcd"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot static curves
    line1, = ax.plot(etas, res_atk, 'r-o', label='Residual Malicious Traffic')
    line2, = ax.plot(etas, slo_vio, 'b-s', label='SLO Violation Rate')
    line3, = ax.plot(etas, bcd, 'm-^', label='Benign Collateral Damage (BCD)')

    # Plot adaptive point
    # We don't have a single "eta" for adaptive, so we might plot it as a horizontal line or just dots.
    # The requirement says "scattered dots that fall below both curves simultaneously".
    # This implies we plot them at their respective metric values.
    # But what is the X-axis for the adaptive dot? 
    # Usually, we'd plot (residual, slo_vio) as coordinates if it's a Pareto plot.
    # However, the user said "X-axis: static IDS allocation ratio eta".
    # So for the adaptive method, we can plot it at its *average* IDS allocation ratio.
    
    # Wait, if X is eta, and we have multiple metrics on Y, then adaptive is a set of dots at X=avg_eta_adaptive.
    # Let's see if we can get avg_eta for adaptive.
    # We need to extract it from the simulation results.
    
    # Actually, the user says: "Overlay the proposed adaptive method as scattered dots that fall below both curves simultaneously"
    # This suggests that at some equivalent X, the Y values are lower.
    # But "X-axis: static IDS allocation ratio eta" makes it tricky for adaptive.
    # Maybe we just plot the adaptive dots at their observed average CPU ratio.

    # Let's adjust the run_adaptive to return the avg_eta too.
    
    ax.set_xlabel('Static IDS Allocation Ratio ($\eta$)')
    ax.set_ylabel('Metric Value')
    ax.set_title('Defense-Service Tradeoff Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"Plot saved to {outpath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/simulation_0.yaml")
    parser.add_argument("--outdir", default="eval_tradeoff_out/test")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--ckpt", default="checkpoints/_phase2/phase2_act3/ckpt_best.pt", help="Path to the adaptive method checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eta_steps", type=int, default=11)
    # Re-use some defaults from eval_common/BaseEvaluator
    parser.add_argument("--decision_interval", type=int, default=None)
    parser.add_argument("--scale_step", type=float, default=0.5)
    parser.add_argument("--ids_cpu_min", type=float, default=0.5)
    parser.add_argument("--tbsa_table", default="tbsa_table_15.npz")
    args = parser.parse_args()

    evaluator = TradeoffEvaluator(args)
    
    print("Running static sweep...")
    sweep_results = evaluator.run_sweep(eta_steps=args.eta_steps)
    
    print("\nSweep Results:")
    for r in sweep_results:
        print(f"  eta={r['eta']:.2f}: Leak={r['residual_atk']:.3f}, SLO={r['slo_vio']:.3f}, BCD={r['bcd']:.3f}")

    adaptive_points = []
    for acc_model, offload_mode in PROPOSED_CONFIGS:
        label = f"adaptive_{acc_model}_{offload_mode}"
        print(f"\nRunning adaptive method: {acc_model}, {offload_mode} (ckpt={args.ckpt})...")
        
        cfg_mut = evaluator.mutate_cfg(evaluator.cfg_original, offload_mode, acc_model)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(cfg_mut, tmp)
            tmp_path = tmp.name
        
        try:
            env = build_env_base(tmp_path)
            policy = make_baseline_policy("ma_lstm_rl", ckpt_path=args.ckpt, device=args.device)
            res_dict, _ = evaluator.run_simulation(env, cfg_mut, policy, label)
            metrics = evaluator.extract_metrics(res_dict)
            avg_eta = np.mean(res_dict["cpu_to_ids_ratio"])
            
            adaptive_points.append({
                "label": f"{acc_model}, {offload_mode}",
                "avg_eta": avg_eta,
                "metrics": metrics
            })
            
            print(f"  Metrics (avg_eta={avg_eta:.3f}):")
            print(f"    Leak={metrics['atk_leak']:.3f}, SLO={metrics['slo_vio']:.3f}, BCD={metrics['bcd']:.3f}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Evaluation of other adaptive baselines
    other_adaptive = []
    for bname in ["reactive", "offline_optimal"]:
        print(f"\nRunning adaptive baseline: {bname}...")
        env = build_env_base(args.cfg)
        try:
            policy = make_baseline_policy(bname, tbsa_table_path=args.tbsa_table, device=args.device)
            res_dict, _ = evaluator.run_simulation(env, evaluator.cfg_original, policy, f"baseline_{bname}")
            metrics = evaluator.extract_metrics(res_dict)
            avg_eta = np.mean(res_dict["cpu_to_ids_ratio"])
            other_adaptive.append({
                "label": bname,
                "avg_eta": avg_eta,
                "metrics": metrics
            })
            print(f"  Metrics (avg_eta={avg_eta:.3f}):")
            print(f"    Leak={metrics['atk_leak']:.3f}, SLO={metrics['slo_vio']:.3f}, BCD={metrics['bcd']:.3f}")
        except Exception as e:
            print(f"  [warn] Failed to run {bname}: {e}")

    # Plotting
    etas = [r["eta"] for r in sweep_results]
    res_atk = [r["residual_atk"] for r in sweep_results]
    slo_vio = [r["slo_vio"] for r in sweep_results]
    bcd = [r["bcd"] for r in sweep_results]

    plt.figure(figsize=(14, 8))
    plt.plot(etas, res_atk, 'r-o', alpha=0.4, label='Static: Residual Malicious Traffic')
    plt.plot(etas, slo_vio, 'b-s', alpha=0.4, label='Static: SLO Violation Rate')
    plt.plot(etas, bcd, 'm-^', alpha=0.4, label='Static: Benign Collateral Damage (BCD)')

    # Adaptive points (Proposed)
    markers_prop = ['*', 'P', 'X', 'D']
    for i, pt in enumerate(adaptive_points):
        m = markers_prop[i % len(markers_prop)]
        ae = pt["avg_eta"]
        met = pt["metrics"]
        lbl = pt["label"]
        plt.axvline(x=ae, color='red', linestyle=':', alpha=0.1)
        plt.scatter([ae], [met["atk_leak"]], color='red', marker=m, s=200, edgecolors='black', label=f'Prop {lbl} (Leak)', zorder=10)
        plt.scatter([ae], [met["slo_vio"]], color='blue', marker=m, s=200, edgecolors='black', label=f'Prop {lbl} (SLO)', zorder=10)
        plt.scatter([ae], [met["bcd"]], color='purple', marker=m, s=200, edgecolors='black', label=f'Prop {lbl} (BCD)', zorder=10)

    # Other Adaptive points
    markers_other = ['d', 'h', 'p', '8']
    for i, pt in enumerate(other_adaptive):
        m = markers_other[i % len(markers_other)]
        ae = pt["avg_eta"]
        met = pt["metrics"]
        lbl = pt["label"]
        plt.axvline(x=ae, color='gray', linestyle=':', alpha=0.1)
        plt.scatter([ae], [met["atk_leak"]], color='red', marker=m, s=150, alpha=0.7, edgecolors='gray', label=f'Base {lbl} (Leak)', zorder=5)
        plt.scatter([ae], [met["slo_vio"]], color='blue', marker=m, s=150, alpha=0.7, edgecolors='gray', label=f'Base {lbl} (SLO)', zorder=5)
        plt.scatter([ae], [met["bcd"]], color='purple', marker=m, s=150, alpha=0.7, edgecolors='gray', label=f'Base {lbl} (BCD)', zorder=5)

    plt.xlabel('IDS Allocation Ratio ($\eta$)')
    plt.ylabel('Metric Value')
    plt.title('Defense-Service Tradeoff Curve (Proposed vs Baselines)')
    plt.grid(True, alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "tradeoff_curve.png", dpi=200)
    print(f"Plot saved to {outdir / 'tradeoff_curve.png'}")

if __name__ == "__main__":
    main()
