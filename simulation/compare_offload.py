"""
Compare offload modes on a fixed policy using configs/simulation_ma_0.yaml.

Each offload mode runs --episodes episodes with the same seeds and policy.
Results are plotted and a summary table is printed.

Usage:
    python compare_offload.py
    python compare_offload.py --cfg configs/simulation_ma_0.yaml --episodes 5
    python compare_offload.py --modes none balance delay_workload cto cto_acc full
    python compare_offload.py --policy autoscale_def --atk_level high --user_level high
    python compare_offload.py --policy tbsa --tbsa_table_path tbsa_table.pkl
    python compare_offload.py --policy lstm_rl --ckpt_path checkpoints/run/ckpt_iter_000100.pt
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

from environment import build_env_base
from train_sa_lstm import TorchRLEnvWrapper
from method_policy import OFFLOAD_DISPLAY_NAMES, make_baseline_policy
from eval_baselines import (
    run_episode,
    plot_ts_continuous,
    plot_qoe_vio_bars,
)

ALL_MODES = ["none", "balance", "delay_workload", "cto", "cto_acc", "full"]


def _build_env(cfg: dict, offload_mode: str, acc_model: Optional[str]) -> object:
    patched = copy.deepcopy(cfg)
    patched["globals"]["offload_mode"] = offload_mode
    if acc_model is not None and "accuracy_matrix" in patched.get("globals", {}):
        patched["globals"]["accuracy_matrix"]["model"] = acc_model
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(patched, f)
        path = f.name
    try:
        return build_env_base(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def main():
    ap = argparse.ArgumentParser(description="Compare offload modes")
    ap.add_argument("--cfg",      default="configs/simulation_ma_0.yaml")
    ap.add_argument("--outdir",   default="eval_out/offload_compare")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--policy",   default="autoscale_def",
                    help="Baseline policy to use (same for all modes)")
    ap.add_argument("--modes",    nargs="+", default=ALL_MODES,
                    help="Offload modes to compare")
    ap.add_argument("--acc_model", default=None,
                    help="Override accuracy_matrix.model (gm|lm). Defaults to config value.")
    ap.add_argument("--atk_level",  default=None,
                    help="Override attack_sampler.level (low|mid|high|default)")
    ap.add_argument("--user_level", default=None,
                    help="Override user_sampler.synthetic.level (low|mid|high|default)")
    ap.add_argument("--scale_step",  type=float, default=0.5)
    ap.add_argument("--ids_cpu_min", type=float, default=0.5)
    ap.add_argument("--decision_interval", type=int, default=None)
    ap.add_argument("--tbsa_table_path", default="tbsa_table_15.npz",
                    help="Path to TBSA lookup table (required for --policy tbsa)")
    ap.add_argument("--ckpt_path", default="checkpoints/_singleedge/a4_sf20_atk3_a18_default_default/ckpt_best.pt",
                    help="Checkpoint path (required for --policy lstm_rl / ma_lstm_rl)")
    args = ap.parse_args()

    with open(args.cfg) as f:
        cfg_original = yaml.safe_load(f)

    base_seed = int(cfg_original["run"]["seed"])
    np.random.seed(base_seed)

    decision_interval = args.decision_interval or int(cfg_original["globals"]["decision_interval"])
    acc_model = args.acc_model or cfg_original["globals"].get("accuracy_matrix", {}).get("model", "gm")

    _wrapper     = TorchRLEnvWrapper(cfg_path=args.cfg, decision_interval=decision_interval, device="cpu")
    obs_keys     = _wrapper.obs_keys
    reward_alpha = _wrapper.reward_alpha
    reward_beta  = _wrapper.reward_beta
    reward_gamma = _wrapper.reward_gamma
    reward_q_th  = _wrapper.reward_q_th
    n_actions    = _wrapper.n_actions
    del _wrapper

    cfg_base = copy.deepcopy(cfg_original)
    if args.atk_level and "attack_sampler" in cfg_base.get("globals", {}):
        cfg_base["globals"]["attack_sampler"]["level"] = args.atk_level
    if args.user_level and "user_sampler" in cfg_base.get("globals", {}) and \
            "synthetic" in cfg_base["globals"]["user_sampler"]:
        cfg_base["globals"]["user_sampler"]["synthetic"]["level"] = args.user_level

    atk_lvl  = args.atk_level  or cfg_base["globals"].get("attack_sampler",  {}).get("level",  "default")
    user_lvl = args.user_level or cfg_base["globals"].get("user_sampler", {}).get("synthetic", {}).get("level", "default")
    print(f"Attack level: {atk_lvl}  |  User level: {user_lvl}  |  Acc model: {acc_model}")
    print(f"Policy: {args.policy}  |  Modes: {args.modes}\n")

    try:
        policy = make_baseline_policy(
            args.policy,
            tbsa_table_path=args.tbsa_table_path,
            ckpt_path=args.ckpt_path,
            obs_keys=obs_keys,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"[error] Could not build policy '{args.policy}': {e}")
        return

    results: Dict[str, Dict[str, np.ndarray]] = {}
    area_ids: List[str] = []

    for mode in args.modes:
        env = _build_env(cfg_base, mode, acc_model)
        if not area_ids:
            area_ids = [e.area_id for e in env.edge_areas]

        label = OFFLOAD_DISPLAY_NAMES.get(mode, mode)
        print(f"Running mode: {mode} ({label})")
        mode_results: Dict[str, np.ndarray] = {}
        last_ids_cpu = None

        for ep in tqdm(range(args.episodes), desc=f"  episodes [{mode}]"):
            ep_seed = base_seed + (ep + 1) * 1000
            ep_result, last_ids_cpu = run_episode(
                env=env,
                cfg=cfg_base,
                policy=policy,
                obs_keys=obs_keys,
                decision_interval=decision_interval,
                scale_step=args.scale_step,
                n_actions=n_actions,
                ids_cpu_min=args.ids_cpu_min,
                seed=ep_seed,
                reward_alpha=reward_alpha,
                reward_beta=reward_beta,
                reward_gamma=reward_gamma,
                reward_q_th=reward_q_th,
                initial_ids_cpu=last_ids_cpu,
            )
            for k, v in ep_result.items():
                existing = mode_results.get(k)
                mode_results[k] = v if existing is None or existing.size == 0 \
                    else np.concatenate([existing, v], axis=0)

        results[label] = mode_results

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_ts_continuous(results, outdir / "qoe_ts.png",
                       area_ids=area_ids, slo_qoe_min=reward_q_th)
    plot_qoe_vio_bars(results, outdir / "summary.png",
                      qoe_slo_min=reward_q_th)
    print(f"\nPlots saved to {outdir}/")

    col_w = 20
    header = (f"{'Mode':<{col_w}} {'QoE (pen.)':>10} {'Vio Rate':>10} "
              f"{'Atk Drop%':>10} {'λ_res':>10} {'BCD':>10} {'Reward':>10}")
    print("\n" + header)
    print("-" * len(header))
    for mode, label in [(m, OFFLOAD_DISPLAY_NAMES.get(m, m)) for m in args.modes]:
        r = results.get(label)
        if r is None:
            continue
        qoe_raw = r["qoe"]
        vio_rate = float(np.mean(r["qoe_vio_rate"]))
        penalty  = float(np.exp(-3.0 * vio_rate))
        avg_qoe  = float(np.mean(qoe_raw)) * penalty
        atk_in   = float(r["attack_in_rate"].sum())
        atk_drp  = float(r["attack_drop_rate"].sum())
        atk_pct  = atk_drp / atk_in if atk_in > 1e-6 else 0.0
        lres     = 1.0 - atk_pct
        bcd      = float(np.mean(r["benign_col_dmg"]))
        reward   = float(np.mean(r["reward"]))
        print(f"{label:<{col_w}} {avg_qoe:>10.4f} {vio_rate:>10.1%} "
              f"{atk_pct:>10.1%} {lres:>10.4f} {bcd:>10.4f} {reward:>10.4f}")


if __name__ == "__main__":
    main()
