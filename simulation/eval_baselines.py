"""
Evaluate baseline policies and upload results to wandb.

Usage:
    python eval_baselines.py --cfg configs/simulation_0.yaml --episodes 30
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
import wandb
from tqdm import tqdm

from environment import build_env_base
from method_policy import make_baseline_policy, BaselinePolicy
from policy import (
    run_episode,
    build_observation_from_history,
    decision_cpu_util,
    apply_delta,
)

BASELINE_METHODS = ["random", "constant_0.5", "constant_1.5", "reactive", "tbsa"]


def _empty_arrays() -> Dict[str, np.ndarray]:
    return {k: np.array([], dtype=np.float32) for k in [
        "qoe", "benign_col_dmg", "cpu_util", "local_num_req",
        "attack_in_rate", "attack_drop_rate",
        "reward_lambda_res", "reward_benign_col_dmg", "reward_qoe_penalty",
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",               type=str,   default="./configs/simulation_0.yaml")
    ap.add_argument("--episodes",          type=int,   default=30)
    ap.add_argument("--decision_interval", type=int,   default=300)
    ap.add_argument("--scale_step",        type=float, default=0.5)
    ap.add_argument("--ids_cpu_min",       type=float, default=0.5)
    ap.add_argument("--tbsa_table",        type=str,   default="tbsa_table.npz")
    ap.add_argument("--wandb_project",     type=str,   default="edgeids")
    ap.add_argument("--wandb_entity",      type=str,   default="asture123-national-taiwan-university")
    ap.add_argument("--wandb_run_name",    type=str,   default="baselines_eval")
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg["run"]["seed"])

    obs_keys = [
        "local_num_req",
        "attack_in_rate",
        "ema_mom",
        "cpu_to_ids_ratio",
        "ids_cpu_utilization",
    ]

    env = build_env_base(args.cfg)

    # Build baseline policy objects
    tbsa_table_path = Path(args.tbsa_table)
    tbsa_exists = tbsa_table_path.exists()
    if not tbsa_exists:
        print(f"[warn] TBSA table not found at {tbsa_table_path}; skipping 'tbsa' method.")

    methods = [m for m in BASELINE_METHODS if m != "tbsa" or tbsa_exists]
    policies: Dict[str, BaselinePolicy] = {}
    for name in methods:
        tbsa_path = str(tbsa_table_path) if name == "tbsa" else None
        policies[name] = make_baseline_policy(name, tbsa_table_path=tbsa_path)

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "cfg_path":          args.cfg,
            "episodes":          args.episodes,
            "decision_interval": args.decision_interval,
            "scale_step":        args.scale_step,
            "ids_cpu_min":       args.ids_cpu_min,
            "methods":           methods,
            "env/t_max":         cfg["run"]["t_max"],
            "env/seed":          cfg["run"]["seed"],
        },
    )

    wandb.define_metric("episode")
    wandb.define_metric("ep/*", step_metric="episode")
    wandb.define_metric("decision_step")
    wandb.define_metric("ts/*",  step_metric="decision_step")

    # Per-episode accumulators for aggregate stats
    ep_stats: Dict[str, Dict[str, List[float]]] = {m: {} for m in methods}

    # Decision-step time series table (one row per method × episode × decision-step)
    ts_table = wandb.Table(columns=[
        "method", "episode", "decision_step",
        "qoe", "benign_col_dmg",
        "attack_in_rate", "attack_drop_rate", "lambda_res_frac",
        "cpu_to_ids_ratio", "reward_total",
    ])

    for ep in tqdm(range(args.episodes), desc="episodes"):
        ep_seed = base_seed + ep * 1000

        for mname in methods:
            result = run_episode(
                env=env,
                cfg=cfg,
                policy=policies[mname],
                decision_interval=args.decision_interval,
                obs_keys=obs_keys,
                scale_step=args.scale_step,
                ids_cpu_min=args.ids_cpu_min,
                seed=ep_seed,
            )

            qoe       = result["qoe"]
            bcd       = result["benign_col_dmg"]
            atk_in    = result["attack_in_rate"]
            atk_drop  = result["attack_drop_rate"]
            r_lres    = result["reward_lambda_res"]
            r_bcd     = result["reward_benign_col_dmg"]
            r_qoe     = result["reward_qoe_penalty"]
            cpu_ratio = result["cpu_to_ids_ratio"]

            # QoE violation rate (fraction below 0.2 SLO)
            q_th = float(cfg.get("globals", {}).get("reward", {}).get("q_th", 0.20))
            vio_rate  = float(np.mean((qoe < q_th).astype(np.float32))) if qoe.size else 0.0
            avg_qoe   = float(np.nanmean(qoe))   if qoe.size   else 0.0
            avg_bcd   = float(np.nanmean(bcd))   if bcd.size   else 0.0
            avg_lres  = float(np.nanmean(r_lres)) if r_lres.size else 0.0

            # Total reward per decision step (sum of components, already weighted)
            reward_ts = -(r_lres + r_bcd + r_qoe)

            # Lambda-res fraction (unweighted) per step
            with np.errstate(divide="ignore", invalid="ignore"):
                lres_frac = np.where(atk_in > 1e-6,
                                     np.maximum(0.0, atk_in - atk_drop) / atk_in,
                                     0.0)

            # --- per-episode scalar logging ---
            global_step = ep * len(methods) + methods.index(mname)
            wandb.log({
                "episode": ep,
                f"ep/{mname}/qoe_mean":    avg_qoe,
                f"ep/{mname}/vio_rate":    vio_rate,
                f"ep/{mname}/bcd_mean":    avg_bcd,
                f"ep/{mname}/lambda_res":  avg_lres,
                f"ep/{mname}/reward_mean": float(np.nanmean(reward_ts)),
            }, step=global_step)

            # --- decision-step time-series table rows ---
            for t in range(len(qoe)):
                r_total = float(-(r_lres[t] + r_bcd[t] + r_qoe[t])) if t < len(r_lres) else 0.0
                ts_table.add_data(
                    mname, ep, t,
                    float(qoe[t]),
                    float(bcd[t]) if t < len(bcd) else 0.0,
                    float(atk_in[t]) if t < len(atk_in) else 0.0,
                    float(atk_drop[t]) if t < len(atk_drop) else 0.0,
                    float(lres_frac[t]) if t < len(lres_frac) else 0.0,
                    float(cpu_ratio[t]) if t < len(cpu_ratio) else 0.0,
                    r_total,
                )

            # Accumulate for summary
            for k, v in [
                ("qoe_mean",    avg_qoe),
                ("vio_rate",    vio_rate),
                ("bcd_mean",    avg_bcd),
                ("lambda_res",  avg_lres),
                ("reward_mean", float(np.nanmean(reward_ts))),
            ]:
                ep_stats[mname].setdefault(k, []).append(v)

    # --- Upload decision-step time series table ---
    wandb.log({"timeseries/decision_steps": ts_table})

    # --- Summary comparison table ---
    summary_cols = ["method", "qoe_mean", "vio_rate", "bcd_mean", "lambda_res", "reward_mean"]
    summary_table = wandb.Table(columns=summary_cols)

    bar_data = {k: [] for k in ["method", "qoe_mean", "vio_rate", "bcd_mean", "lambda_res", "reward_mean"]}

    for mname in methods:
        stats = ep_stats[mname]
        row_vals = {
            "method":      mname,
            "qoe_mean":    float(np.mean(stats.get("qoe_mean",   [0.0]))),
            "vio_rate":    float(np.mean(stats.get("vio_rate",   [0.0]))),
            "bcd_mean":    float(np.mean(stats.get("bcd_mean",   [0.0]))),
            "lambda_res":  float(np.mean(stats.get("lambda_res", [0.0]))),
            "reward_mean": float(np.mean(stats.get("reward_mean",[0.0]))),
        }
        summary_table.add_data(*[row_vals[c] for c in summary_cols])
        for k in bar_data:
            bar_data[k].append(row_vals[k])

        # Also log as flat summary scalars
        wandb.summary[f"{mname}/qoe_mean"]    = row_vals["qoe_mean"]
        wandb.summary[f"{mname}/vio_rate"]    = row_vals["vio_rate"]
        wandb.summary[f"{mname}/bcd_mean"]    = row_vals["bcd_mean"]
        wandb.summary[f"{mname}/lambda_res"]  = row_vals["lambda_res"]
        wandb.summary[f"{mname}/reward_mean"] = row_vals["reward_mean"]

    wandb.log({"summary/comparison_table": summary_table})

    # Bar charts for key metrics
    for metric in ["qoe_mean", "vio_rate", "bcd_mean", "lambda_res"]:
        data = [[m, v] for m, v in zip(bar_data["method"], bar_data[metric])]
        table = wandb.Table(data=data, columns=["method", metric])
        wandb.log({
            f"summary/bar_{metric}": wandb.plot.bar(table, "method", metric, title=metric)
        })

    run.finish()
    print(f"\nDone. Results uploaded to wandb run: {run.url}")


if __name__ == "__main__":
    main()
