"""
Evaluate baseline policies and plot results (no wandb).

Each method runs for --episodes episodes; per-decision metrics are
concatenated across episodes and fed to the same plotting functions
used by old_policy.py.

Usage:
    python eval_baselines.py --cfg configs/simulation_0.yaml --episodes 10
    python eval_baselines.py --methods reactive random constant_4.0 lstm_rl \
        --ckpt checkpoints/<run>/ckpt_best.pt --episodes 5
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from environment import build_env_base, TorchRLEnvWrapper
from method_policy import ActContext, BaselinePolicy, make_baseline_policy
from old_policy import plot_ts_continuous, plot_qoe_vio_bars

SCALING_K      = 2.0
SCALING_QUANTA = [0.5, 1.0, 1.5, 2.0]

DEFAULT_METHODS = ["random", "constant_0.5", "constant_1.5", "tbsa", "reactive"]
DEFAULT_METHODS = ["tbsa", "reactive", "lstm_rl"]


# ---------------------------------------------------------------------------
# Helpers (mirror TorchRLEnvWrapper internals)
# ---------------------------------------------------------------------------

def _overhead_rate(pending: float, scaling_time_steps: List[int]) -> float:
    abs_p = abs(pending)
    if abs_p < 1e-9:
        return 0.0
    for i, q in enumerate(SCALING_QUANTA):
        if abs_p <= q + 1e-9:
            return abs_p / scaling_time_steps[i]
    return abs_p / scaling_time_steps[-1]


def _build_obs_flat(
    env,
    decision_interval: int,
    obs_keys: List[str],
    scaling_pending: float,
) -> np.ndarray:
    n_edges = len(env.edge_areas)
    obs_dim = len(obs_keys) + 1
    obs = np.zeros((n_edges, obs_dim), dtype=np.float32)

    if env.history:
        records = env.history[-decision_interval * n_edges:]
        df = pd.DataFrame([m.__dict__ for m in records])
        for i, area_id in enumerate([e.area_id for e in env.edge_areas]):
            g = df[df["area_id"] == area_id]
            if g.empty:
                continue
            for j, k in enumerate(obs_keys):
                if k not in g.columns:
                    continue
                vals = g[k].values
                if k == "I_net":
                    obs[i, j] = float(np.sum(vals))
                elif k == "cpu_to_ids_ratio":
                    obs[i, j] = float(vals[-1])
                elif k == "ema_mom":
                    vals_nz = vals[vals != 0.0]
                    obs[i, j] = float(np.mean(vals_nz)) if len(vals_nz) > 0 else 0.0
                else:
                    obs[i, j] = float(np.mean(vals))

    obs[:, -1] = float(np.clip(scaling_pending / SCALING_K, -1.0, 1.0))
    return obs.reshape(-1).astype(np.float32)


def _cpu_util(env, decision_interval: int) -> float:
    n_edges = len(env.edge_areas)
    if len(env.history) < decision_interval * n_edges:
        return 0.0
    records = env.history[-decision_interval * n_edges:]
    df = pd.DataFrame([m.__dict__ for m in records])
    utils = []
    for edge in env.edge_areas:
        g = df[df["area_id"] == edge.area_id]
        if g.empty or "ids_cpu_utilization" not in g.columns:
            continue
        utils.append(float(np.clip(np.mean(g["ids_cpu_utilization"].values), 0.0, 1.0)))
    return float(max(utils)) if utils else 0.0


# ---------------------------------------------------------------------------
# Episode runner — returns the same dict schema as old_policy.run_episode
# ---------------------------------------------------------------------------

def run_episode(
    env,
    cfg: dict,
    policy: BaselinePolicy,
    obs_keys: List[str],
    decision_interval: int,
    scale_step: float,
    ids_cpu_min: float,
    seed: int,
    reward_alpha: float,
    reward_beta: float,
    reward_gamma: float,
    reward_q_th: float,
) -> Dict[str, np.ndarray]:
    """
    Run one episode with *policy* and return a dict of per-decision arrays
    (same keys as old_policy.run_episode so the same plotting code works).
    """
    env.reset(seed)
    policy.reset()

    n_edges     = len(env.edge_areas)
    ids_cpu_max = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)
    ids_cpu     = np.array([e.ids_cpu for e in env.edge_areas], dtype=np.float32)
    ids_cpu     = np.clip(ids_cpu, ids_cpu_min, ids_cpu_max)

    scaling_time_steps: List[int] = list(
        cfg["globals"].get("scaling_time_step", [300, 450, 498, 544])
    )
    scaling_pending = 0.0
    overhead_rate   = 0.0
    rng = np.random.default_rng(seed)

    decisions = math.ceil(int(cfg["run"]["t_max"]) / decision_interval)

    qoe_ts                = []
    benign_col_dmg_ts     = []
    cpu_util_ts           = []
    local_num_req_ts      = []
    attack_in_rate_ts     = []
    attack_in_rate_std_ts = []
    attack_drop_rate_ts   = []
    ema_mom_ts            = []
    cpu_to_ids_ratio_ts   = []
    reward_lambda_res_ts  = []
    reward_bcd_ts         = []
    reward_qoe_penalty_ts = []

    for _ in range(decisions):
        if env.t >= env.t_max:
            break

        obs_flat = _build_obs_flat(env, decision_interval, obs_keys, scaling_pending)
        cpu_util = _cpu_util(env, decision_interval)

        ctx = ActContext(
            env=env,
            ids_cpu=ids_cpu.copy(),
            ids_cpu_min=ids_cpu_min,
            ids_cpu_max=ids_cpu_max,
            cpu_util=cpu_util,
            decision_interval=decision_interval,
            rng=rng,
            scaling_pending=scaling_pending,
            obs_flat=obs_flat,
        )

        ids_cpu_abs, delta = policy.act(ctx)

        prev_ids_cpu = ids_cpu.copy()
        if ids_cpu_abs is not None:
            ids_cpu = np.clip(ids_cpu_abs, ids_cpu_min, ids_cpu_max).astype(np.float32)
        else:
            ids_cpu = np.clip(
                ids_cpu + delta.astype(np.float32) * scale_step,
                ids_cpu_min,
                ids_cpu_max,
            ).astype(np.float32)

        delta_eff = float(ids_cpu[0] - prev_ids_cpu[0])
        if delta_eff > 0.0:
            scaling_pending = min(scaling_pending + delta_eff, SCALING_K)
        else:
            scaling_pending = max(0.0, scaling_pending + delta_eff)
        overhead_rate = _overhead_rate(scaling_pending, scaling_time_steps)

        ids_cpu_eff = ids_cpu.copy()
        if scaling_pending > 1e-9:
            ids_cpu_eff[0] = ids_cpu[0] - scaling_pending

        for _ in range(decision_interval):
            if scaling_pending > 1e-9:
                consumed         = min(scaling_pending, overhead_rate)
                scaling_pending -= consumed
                if scaling_pending < 1e-9:
                    scaling_pending = 0.0
                    ids_cpu_eff[0]  = ids_cpu[0]
            env.step(ids_cpu_eff, 0.0)
            if env.t >= env.t_max:
                break

        # --- aggregate metrics over the just-completed decision window ---
        window = env.history[-decision_interval * n_edges:]
        df = pd.DataFrame([m.__dict__ for m in window])

        def _col_mean(col):
            return float(df[col].mean()) if col in df.columns else 0.0

        qoe_mean = (
            float(np.mean(df["qoe_mean"].values)) if "qoe_mean" in df.columns else 0.0
        )
        bcd_mean = (
            float(np.mean(df["benign_col_dmg"].values)) if "benign_col_dmg" in df.columns else 0.0
        )

        qoe_ts.append(qoe_mean)
        benign_col_dmg_ts.append(bcd_mean)
        cpu_util_ts.append(cpu_util)
        local_num_req_ts.append(_col_mean("local_num_req"))
        attack_in_rate_ts.append(_col_mean("attack_in_rate"))
        attack_in_rate_std_ts.append(
            float(df["attack_in_rate"].std()) if "attack_in_rate" in df.columns else 0.0
        )
        attack_drop_rate_ts.append(_col_mean("attack_drop_rate"))
        ema_mom_ts.append(_col_mean("ema_mom"))
        ratios = ids_cpu / np.array([e.budget.cpu for e in env.edge_areas], dtype=np.float32)
        cpu_to_ids_ratio_ts.append(float(ratios.mean()))

        # Reward components (same formula as TorchRLEnvWrapper._build_reward)
        if "attack_in_rate" in df.columns and "attack_drop_rate" in df.columns:
            atk_in   = df["attack_in_rate"].values.astype(np.float32)
            atk_drop = df["attack_drop_rate"].values.astype(np.float32)
            atk_pass = np.maximum(0.0, atk_in - atk_drop)
            lres     = np.divide(atk_pass, atk_in, out=np.zeros_like(atk_pass), where=atk_in > 1e-6)
            qoes     = df["qoe_mean"].values.astype(np.float32) if "qoe_mean" in df.columns else np.zeros(len(atk_in))
            sf       = np.maximum(0.0, reward_q_th - qoes) / max(reward_q_th, 1e-6)
            bcd_vals = df["benign_col_dmg"].values.astype(np.float32) if "benign_col_dmg" in df.columns else np.zeros(len(atk_in))

            reward_lambda_res_ts.append(float(reward_beta  * np.mean(lres)))
            reward_bcd_ts.append(       float(reward_gamma * np.mean(bcd_vals)))
            reward_qoe_penalty_ts.append(float(reward_alpha * np.mean(sf)))
        else:
            reward_lambda_res_ts.append(0.0)
            reward_bcd_ts.append(0.0)
            reward_qoe_penalty_ts.append(0.0)

    return {
        "qoe":                  np.asarray(qoe_ts,                dtype=np.float32),
        "benign_col_dmg":       np.asarray(benign_col_dmg_ts,     dtype=np.float32),
        "cpu_util":             np.asarray(cpu_util_ts,           dtype=np.float32),
        "local_num_req":        np.asarray(local_num_req_ts,      dtype=np.float32),
        "attack_in_rate":       np.asarray(attack_in_rate_ts,     dtype=np.float32),
        "attack_in_rate_std":   np.asarray(attack_in_rate_std_ts, dtype=np.float32),
        "attack_drop_rate":     np.asarray(attack_drop_rate_ts,   dtype=np.float32),
        "ema_mom":              np.asarray(ema_mom_ts,            dtype=np.float32),
        "cpu_to_ids_ratio":     np.asarray(cpu_to_ids_ratio_ts,   dtype=np.float32),
        "reward_lambda_res":    np.asarray(reward_lambda_res_ts,  dtype=np.float32),
        "reward_benign_col_dmg": np.asarray(reward_bcd_ts,        dtype=np.float32),
        "reward_qoe_penalty":   np.asarray(reward_qoe_penalty_ts, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate baseline policies and plot results")
    ap.add_argument("--cfg",               default="./configs/simulation_0.yaml")
    ap.add_argument("--outdir",            default="eval_out")
    ap.add_argument("--episodes",          type=int,   default=10)
    ap.add_argument("--decision_interval", type=int,   default=None,
                    help="Overrides globals.decision_interval from cfg if set")
    ap.add_argument("--scale_step",        type=float, default=0.5)
    ap.add_argument("--ids_cpu_min",       type=float, default=0.5)
    ap.add_argument("--methods",           nargs="+",  default=None,
                    help="Methods to evaluate. Defaults: random constant_0.5 constant_4.0 reactive")
    ap.add_argument("--tbsa_table",        default="tbsa_table.npz",
                    help="TBSA lookup-table path (needed when 'tbsa' is in --methods)")
    ap.add_argument("--ckpt",              default="checkpoints/tdsc_so_rew/ckpt_iter_000150.pt",
                    help="Checkpoint path (needed when 'lstm_rl' is in --methods)")
    ap.add_argument("--device",            default="cpu")
    args = ap.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    base_seed         = int(cfg["run"]["seed"])
    decision_interval = args.decision_interval or int(cfg["globals"]["decision_interval"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve methods list
    methods: List[str] = args.methods if args.methods else list(DEFAULT_METHODS)

    # Read obs_keys and reward weights from a throw-away wrapper instance
    _wrapper      = TorchRLEnvWrapper(cfg_path=args.cfg, decision_interval=decision_interval, device="cpu")
    obs_keys      = _wrapper.obs_keys
    reward_alpha  = _wrapper.reward_alpha
    reward_beta   = _wrapper.reward_beta
    reward_gamma  = _wrapper.reward_gamma
    reward_q_th   = _wrapper.reward_q_th
    del _wrapper

    env = build_env_base(args.cfg)

    # Build one policy per method
    tbsa_table_path = Path(args.tbsa_table)
    policies: Dict[str, BaselinePolicy] = {}
    for name in methods:
        tbsa_path = str(tbsa_table_path) if name == "tbsa" else None
        ckpt      = args.ckpt             if name == "lstm_rl" else None
        ok_keys   = obs_keys              if name == "lstm_rl" else None
        try:
            policies[name] = make_baseline_policy(
                name,
                tbsa_table_path=tbsa_path,
                ckpt_path=ckpt,
                obs_keys=ok_keys,
                device=args.device,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"[warn] Skipping '{name}': {exc}")

    methods = [m for m in methods if m in policies]
    if not methods:
        raise SystemExit("No valid methods to evaluate.")

    # Accumulate per-decision results across episodes
    results: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}

    for ep in tqdm(range(args.episodes), desc="episodes"):
        ep_seed = base_seed + ep * 1000
        for mname in methods:
            ep_result = run_episode(
                env=env,
                cfg=cfg,
                policy=policies[mname],
                obs_keys=obs_keys,
                decision_interval=decision_interval,
                scale_step=args.scale_step,
                ids_cpu_min=args.ids_cpu_min,
                seed=ep_seed,
                reward_alpha=reward_alpha,
                reward_beta=reward_beta,
                reward_gamma=reward_gamma,
                reward_q_th=reward_q_th,
            )
            for k, v in ep_result.items():
                results[mname][k] = np.concatenate(
                    [results[mname].get(k, np.array([], dtype=np.float32)), v]
                )

    # Plot — reuse the plotting functions from old_policy.py
    ts_path      = outdir / "qoe_ts.png"
    summary_path = outdir / "summary.png"
    plot_ts_continuous(results, ts_path,      slo_qoe_min=reward_q_th, beta=3)
    plot_qoe_vio_bars( results, summary_path, qoe_slo_min=reward_q_th, beta=3)

    print(f"Plots saved to {outdir}/")


if __name__ == "__main__":
    main()
