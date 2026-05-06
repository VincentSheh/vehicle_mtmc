"""
Evaluate baseline policies and plot results (no wandb).
"""
from __future__ import annotations

import argparse
import copy
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from tqdm import tqdm
from matplotlib import pyplot as plt

from environment import build_env_base
from eval_common import (
    BaseEvaluator, 
    PROPOSED_CONFIGS, 
    DISPLAY_NAMES,
    MODEL_DISPLAY_NAMES,
    OFFLOAD_DISPLAY_NAMES
)

class BaselineEvaluator(BaseEvaluator):
    def mutate_cfg(self, cfg: dict, atk_lvl: str = "default", user_lvl: str = "default", 
                   offload_mode: str = "cto", acc_model: str = "gm") -> dict:
        cfg_mut = copy.deepcopy(cfg)
        if "attack_sampler" in cfg_mut.get("globals", {}):
            cfg_mut["globals"]["attack_sampler"]["level"] = atk_lvl
        if "user_sampler" in cfg_mut.get("globals", {}) and "synthetic" in cfg_mut["globals"]["user_sampler"]:
            cfg_mut["globals"]["user_sampler"]["synthetic"]["level"] = user_lvl
        
        cfg_mut["globals"]["offload_mode"] = offload_mode
        if "accuracy_matrix" in cfg_mut.get("globals", {}):
            cfg_mut["globals"]["accuracy_matrix"]["model"] = acc_model
        return cfg_mut

    def run(self):
        methods = self.get_methods()
        policies = self.build_policies(methods)
        
        # atk_lvls = ["low", "mid", "high"]
        # user_lvls = ["low", "mid", "high"]
        atk_lvls = ["default"]
        user_lvls = ["default"]

        cfg_offload_mode = self.cfg_original["globals"].get("offload_mode", "balance")
        cfg_acc_model = self.cfg_original["globals"].get("accuracy_matrix", {}).get("model", "gm")
        proposed_method = self.args.proposed_method
        BASELINE_OFFLOAD = "cto"

        offload_modes = self.args.offload_modes if self.args.offload_modes else [cfg_offload_mode]
        multi_offload = len(offload_modes) > 1

        for atk_lvl in atk_lvls:
            for user_lvl in user_lvls:
                print(f"\n>>> Evaluating Levels: Attack={atk_lvl}, User={user_lvl}")
                
                # Build runs: (policy_key, result_key, offload_mode, acc_model)
                runs = []
                for mname in methods:
                    if mname not in policies: continue
                    if proposed_method and mname == proposed_method:
                        for acc_model, om in PROPOSED_CONFIGS:
                            rkey = f"{mname}[{acc_model},{om}]"
                            runs.append((mname, rkey, om, acc_model))
                    else:
                        for om in offload_modes:
                            rkey = f"{mname}[{om}]" if multi_offload else mname
                            runs.append((mname, rkey, om, cfg_acc_model))

                # Group by (om, acc_model) to reuse environments
                groups = {}
                for pkey, rkey, om, amod in runs:
                    groups.setdefault((om, amod), []).append((pkey, rkey))

                results = {}
                lvl_outdir = self.outdir / f"atk_{atk_lvl}_user_{user_lvl}"
                lvl_outdir.mkdir(parents=True, exist_ok=True)
                csv_path = lvl_outdir / "results.csv"

                if self.args.replot:
                    print(f"[replot] Loading data from {csv_path}...")
                    if csv_path.exists():
                        # Minimal CSV loader for baselines (format: Method,Metric,Value)
                        try:
                            import csv
                            with open(csv_path, newline="") as f:
                                for row in csv.DictReader(f):
                                    m_label, mk, val = row["Method"], row["MetricKey"], float(row["Value"])
                                    if m_label not in results: results[m_label] = {}
                                    # Note: this only loads scalar summary metrics, ts data is lost in replot for baselines
                                    # unless we save/load full timeseries, which is complex. 
                                    # For now, we only support summary bar plot regeneration.
                                    results[m_label][mk] = val
                            
                            # Transform to match plot_qoe_vio_bars expectations
                            plot_results = {k: {"qoe": np.array([v.get("slo_vio", 0.0)]), "qoe_vio_rate": np.array([v.get("slo_vio", 0.0)]), 
                                               "benign_col_dmg": np.array([v.get("bcd", 0.0)]), "attack_in_rate": np.array([1.0]), "attack_drop_rate": np.array([v.get("atk_drop", 0.0)])} 
                                           for k, v in results.items()}
                            plot_qoe_vio_bars(plot_results, lvl_outdir / "summary.png", self.reward_q_th)
                        except Exception as e: print(f"[warn] Failed to replot: {e}")
                    else:
                        print(f"[warn] {csv_path} not found, skipping replot")
                    continue

                for (om, amod), pkey_rkey_pairs in groups.items():
                    cfg = self.mutate_cfg(self.cfg_original, atk_lvl, user_lvl, om, amod)
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                        yaml.dump(cfg, tmp)
                        tmp_path = tmp.name
                    try:
                        env = build_env_base(tmp_path)
                        for pkey, rkey in pkey_rkey_pairs:
                            results[rkey], _ = self.run_simulation(env, cfg, policies[pkey], rkey, cache_key=f"{rkey}_{atk_lvl}_{user_lvl}", cache_dir=lvl_outdir / "cache")
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)

                lvl_outdir = self.outdir / f"atk_{atk_lvl}_user_{user_lvl}"
                lvl_outdir.mkdir(parents=True, exist_ok=True)

                display_results = {self._display_label(rk): v for rk, v in results.items()}
                area_ids = [e.area_id for e in env.edge_areas]
                
                plot_ts_continuous(display_results, lvl_outdir / "qoe_ts.png", area_ids, self.reward_q_th)
                plot_qoe_vio_bars(display_results, lvl_outdir / "summary.png", self.reward_q_th)

                self._print_table(results, csv_path)

    def _display_label(self, rkey: str) -> str:
        if "[" in rkey:
            inner = rkey[rkey.index("[")+1:].rstrip("]")
            mname = rkey[:rkey.index("[")]
            if "," in inner:
                model_key, om = inner.split(",", 1)
                return f"{mname} {om}_{model_key}"
            return f"{DISPLAY_NAMES.get(mname, mname)} ({OFFLOAD_DISPLAY_NAMES.get(inner, inner)})"
        return DISPLAY_NAMES.get(rkey, rkey)

    def _print_table(self, results, csv_path: Path):
        col_w = 32
        header = f"{'Method':<{col_w}} {'qoe_vio_rate':>12} {'reward/mean':>12} {'qoe_penalty':>12} {'atk_drop_pct':>12} {'n_eps':>8}"
        print("\n" + header + "\n" + "-" * len(header))
        
        rows = []
        for rkey, r in results.items():
            m = self.extract_metrics(r)
            atk_in, atk_drp = r['attack_in_rate'].sum(), r['attack_drop_rate'].sum()
            atk_drop_pct = atk_drp / atk_in if atk_in > 1e-6 else 0.0
            label = self._display_label(rkey)
            print(f"{label:<{col_w}} {m['slo_vio']:>12.1%} {m['reward']:>12.4f} "
                  f"{float(np.mean(r['reward_qoe_penalty'])):>12.4f} {atk_drop_pct:>12.1%} {int(m['n_episodes']):>8}")
            
            # Save for replot
            rows.append({"Method": label, "Metric": "SLO Violation Rate", "MetricKey": "slo_vio", "Value": m["slo_vio"]})
            rows.append({"Method": label, "Metric": "Benign Collateral Damage", "MetricKey": "bcd", "Value": m["bcd"]})
            rows.append({"Method": label, "Metric": "Attack Drop %", "MetricKey": "atk_drop", "Value": atk_drop_pct})
            rows.append({"Method": label, "Metric": "Reward", "MetricKey": "reward", "Value": m["reward"]})
            rows.append({"Method": label, "Metric": "Num Episodes", "MetricKey": "n_episodes", "Value": m["n_episodes"]})

        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Method", "Metric", "MetricKey", "Value"])
            writer.writeheader()
            writer.writerows(rows)

def plot_ts_continuous(results, outpath, area_ids, slo_qoe_min=0.2):
    static_panels = [
        ("benign_col_dmg",    "Benign Collateral Damage"),
        ("local_num_req",     "Local #Req"),
        ("attack_in_rate",    "Attack in rate"),
        ("attack_drop_rate",  "Attack drop rate"),
        ("reward_lambda_res", "λ_res (raw attack pass-through)"),
        ("cpu_util",          "CPU Utilization"),
        ("cpu_to_ids_ratio",  "CPU→IDS Ratio"),
    ]
    n_edges = len(area_ids)
    n_panels = n_edges + len(static_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(15, 3 * n_panels), sharex=True)

    for ei, area_id in enumerate(area_ids):
        ax = axes[ei]
        for method, series in results.items():
            y = series.get("qoe_per_edge")[:, ei] if "qoe_per_edge" in series else series.get("qoe")
            if y is None or y.size == 0: continue
            avg_qoe, vio_rate = np.nanmean(y), np.nanmean(y < slo_qoe_min)
            ax.plot(np.arange(len(y)), y, label=f"{method} (avg={avg_qoe:.3f}, vio={vio_rate:.2%})")
        ax.axhline(slo_qoe_min, color="red", linestyle="--", alpha=0.4)
        ax.set_ylabel(f"QoE [{area_id}]")
        ax.grid(True, alpha=0.3)
        if ei == 0: ax.legend(loc="upper right")

    for ax, (k, ylabel) in zip(axes[n_edges:], static_panels):
        for method, series in results.items():
            y_raw = series.get(k)
            if y_raw is None or y_raw.size == 0: continue
            y = np.mean(y_raw, axis=1) if y_raw.ndim == 2 else y_raw
            line = ax.plot(np.arange(len(y)), y, label=method)[0]
            if y_raw.ndim == 2:
                ax.fill_between(np.arange(len(y)), np.min(y_raw, axis=1), np.max(y_raw, axis=1), color=line.get_color(), alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("decision step")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_qoe_vio_bars(results, outpath, qoe_slo_min=0.2):
    methods, avg_qoe, vio_rate, avg_bcd, avg_drop = [], [], [], [], []
    for method, series in results.items():
        q = series.get("qoe")
        if q is None or q.size == 0: continue
        methods.append(method)
        avg_qoe.append(float(np.mean(q)))
        vio_rate.append(float(np.mean(series.get("qoe_vio_rate", q < qoe_slo_min))))
        avg_bcd.append(float(np.mean(series.get("benign_col_dmg", 0.0))))
        
        atk_in, atk_drp = series.get("attack_in_rate"), series.get("attack_drop_rate")
        avg_drop.append(float(atk_drp.sum() / atk_in.sum()) if atk_in is not None and atk_in.sum() > 1e-6 else 0.0)

    x, width = np.arange(len(methods)), 0.7
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    titles = ["Average QoE", f"SLO Violations (<{qoe_slo_min})", "Avg Benign Col. Damage", "Avg Attack Drop %"]
    data_list = [avg_qoe, vio_rate, avg_bcd, avg_drop]
    formats = [".3f", ".1%", ".3f", ".1%"]
    
    for i, (title, data, fmt) in enumerate(zip(titles, data_list, formats)):
        bars = axes[i].bar(x, data, width)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(methods, rotation=20, ha="right")
        axes[i].set_title(title)
        axes[i].grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, h, f"{h:{fmt}}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="./configs/simulation_ma_0.yaml")
    ap.add_argument("--outdir", default="eval_out")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--decision_interval", type=int, default=None)
    ap.add_argument("--scale_step", type=float, default=0.5)
    ap.add_argument("--ids_cpu_min", type=float, default=0.5)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--tbsa_table", default="tbsa_table_15.npz")
    ap.add_argument("--ckpt", default="checkpoints/singleedge/rew_32_netting_a4_rew/ckpt_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--offload_modes", nargs="+", default=None)
    ap.add_argument("--proposed_method", default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    evaluator = BaselineEvaluator(args)
    evaluator.run()

if __name__ == "__main__":
    main()
