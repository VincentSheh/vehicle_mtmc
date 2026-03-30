"""
Compare system behaviour under yoyo vs static attack patterns.
Plots attack_in_rate and machine count (cpu_to_ids_ratio * n_cores)
for each pattern using a reactive IDS-CPU policy.

Usage:
    python compare_attacks.py
    python compare_attacks.py --attacker atk_02 --episodes 3 --outdir eval_out
"""
import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from environment import build_env_from_cfg, _reactive_ids_cpu

PATTERNS = ["yoyo", "static"]
N_CORES = 8  # matches budget.cpu in simulation_0.yaml


def run_episode(cfg: dict, seed: int, decision_interval: int) -> pd.DataFrame:
    """Build env directly from a config dict and run one reactive episode."""
    env = build_env_from_cfg(cfg)
    env.reset(seed)

    ids_cpu = np.array([e.ids_cpu for e in env.edge_areas], dtype=np.float32)

    t = 0
    while t < env.t_max:
        ids_cpu = _reactive_ids_cpu(env, ids_cpu, decision_interval)
        for _ in range(decision_interval):
            env.step(ids_cpu.tolist())
            t += 1
            if t >= env.t_max:
                break

    return pd.DataFrame([m.__dict__ for m in env.history])


def active_attacker_keys(cfg: dict) -> list:
    """Return deduplicated list of attacker keys that are active in the edge areas."""
    keys = []
    for area_cfg in cfg["edge_areas"]:
        for atk_ref in area_cfg.get("attackers", []):
            k = atk_ref["attacker_type"]
            if k not in keys:
                keys.append(k)
    return keys


def collect_results(
    base_cfg: dict,
    patterns: list,
    n_episodes: int,
    decision_interval: int,
    base_seed: int,
) -> dict:
    keys = active_attacker_keys(base_cfg)
    print(f"Active attackers: {keys}")

    results = {}
    for pattern in patterns:
        cfg = copy.deepcopy(base_cfg)
        for key in keys:
            cfg["globals"]["attack"][key]["pattern_type"] = pattern

        dfs = []
        for ep in range(n_episodes):
            df = run_episode(cfg, seed=base_seed + ep * 1000, decision_interval=decision_interval)
            df["episode"] = ep
            df["t_global"] = ep * base_cfg["run"]["t_max"] + df["t"]
            dfs.append(df)

        results[pattern] = pd.concat(dfs, ignore_index=True)
        print(f"  [{pattern}] {n_episodes} episode(s) done")

    return results


def plot_comparison(results: dict, outdir: str, smooth: int = 200):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out_path = Path(outdir) / "compare_yoyo_vs_static.png"

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    colors = {"yoyo": "#e05c3a", "static": "#3a7be0"}

    for pattern, df in results.items():
        color = colors.get(pattern)
        for area_id in df["area_id"].unique():
            sub = df[df["area_id"] == area_id].sort_values("t_global")
            t = sub["t_global"].values
            label = f"{pattern} ({area_id})"

            atk_rate = pd.Series(sub["attack_in_rate"].values).rolling(smooth, min_periods=1).mean().values
            axes[0].plot(t, atk_rate, label=label, color=color, alpha=0.85)

            machine_count = sub["cpu_to_ids_ratio"].values * N_CORES
            mc_smooth = pd.Series(machine_count).rolling(smooth, min_periods=1).mean().values
            axes[1].plot(t, mc_smooth, label=label, color=color, alpha=0.85)

            qoe_smooth = pd.Series(sub["qoe_mean"].values).rolling(smooth, min_periods=1).mean().values
            axes[2].plot(t, qoe_smooth, label=label, color=color, alpha=0.85)

    axes[0].set_ylabel("Attack In Rate (flows/step)")
    axes[0].set_title("Attack In Rate — Yoyo vs Static")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel(f"IDS Machine Count (out of {N_CORES})")
    axes[1].set_title("IDS Machine Count — Yoyo vs Static")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("QoE")
    axes[2].set_title("QoE — Yoyo vs Static")
    axes[2].set_xlabel("Simulation step")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(0, 20000)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="./configs/simulation_0.yaml")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--decision_interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smooth", type=int, default=200,
                    help="Rolling-mean window for plots (steps)")
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    with open(args.cfg) as f:
        base_cfg = yaml.safe_load(f)

    print(f"Running {args.episodes} episode(s) per pattern "
          f"[decision_interval={args.decision_interval}]")

    results = collect_results(
        base_cfg=base_cfg,
        patterns=PATTERNS,
        n_episodes=args.episodes,
        decision_interval=args.decision_interval,
        base_seed=args.seed,
    )

    plot_comparison(results, outdir=args.outdir, smooth=args.smooth)

    print("\n--- Summary ---")
    for pattern, df in results.items():
        avg_atk = df["attack_in_rate"].mean()
        avg_mc = df["cpu_to_ids_ratio"].mean() * N_CORES
        print(f"  {pattern:10s}  avg_attack_in_rate={avg_atk:.4f}  avg_machine_count={avg_mc:.2f}")


if __name__ == "__main__":
    main()
