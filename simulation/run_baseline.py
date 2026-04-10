"""
Evaluate a single baseline policy (from method_policy.py) and log to wandb
with the same metric schema as train_lstm.py, enabling side-by-side comparison.

Usage examples:
    python run_baseline.py --method reactive
    python run_baseline.py --method constant_4.0
    python run_baseline.py --method random
    python run_baseline.py --method tbsa     --tbsa_table path/to/table.csv
    python run_baseline.py --method lstm_rl  --ckpt checkpoints/<run>/ckpt_best.pt
"""
from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

from environment import build_env_base, TorchRLEnvWrapper
from logger import wandb_init
from method_policy import ActContext, BaselinePolicy, make_baseline_policy

# ---------------------------------------------------------------------------
# Constants (mirror TorchRLEnvWrapper defaults)
# ---------------------------------------------------------------------------
SCALING_K      = 2.0
SCALING_QUANTA = [0.5, 1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------
# Helpers
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
    """Build the flat observation array — mirrors TorchRLEnvWrapper._build_observation."""
    n_edges = len(env.edge_areas)
    obs_dim = len(obs_keys) + 1          # +1 for scaling_pending
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
    """Max mean IDS CPU utilisation across edges over the last decision window."""
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


def _reward_components(
    env,
    n_edges: int,
    alpha: float,
    beta: float,
    gamma: float,
    q_th: float,
) -> dict:
    """Per-tick reward — mirrors TorchRLEnvWrapper._build_reward."""
    if not env.history:
        return {"reward": 0.0, "lambda_res": 0.0, "benign_col_dmg": 0.0, "qoe_penalty": 0.0}

    last = env.history[-n_edges:]
    qoe        = np.array([float(m.qoe_mean)         for m in last], dtype=np.float32)
    bcd        = np.array([float(m.benign_col_dmg)   for m in last], dtype=np.float32)
    atk_in     = np.array([float(m.attack_in_rate)   for m in last], dtype=np.float32)
    atk_drop   = np.array([float(m.attack_drop_rate) for m in last], dtype=np.float32)

    atk_pass  = np.maximum(0.0, atk_in - atk_drop)
    lres      = np.divide(atk_pass, atk_in, out=np.zeros_like(atk_pass), where=atk_in > 1e-6)
    shortfall = np.maximum(0.0, q_th - qoe) / max(q_th, 1e-6)

    r_lres = float(np.mean(lres))
    r_bcd  = float(np.mean(bcd))
    r_qoe  = float(np.mean(shortfall))

    return {
        "reward":        -(alpha * r_qoe + beta * r_lres + gamma * r_bcd),
        "lambda_res":    r_lres,
        "benign_col_dmg": r_bcd,
        "qoe_penalty":   r_qoe,
    }


# ---------------------------------------------------------------------------
# Episode runner
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
    global_decision_step: int,
) -> Tuple[dict, int]:
    """
    Run one episode and log per-decision obs/* to wandb.
    Returns (episode_metrics_dict, n_decisions_taken).
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

    total_reward = total_lres = total_bcd = total_qoe_penalty = 0.0
    total_ticks  = 0
    n_decisions  = 0

    obs_keys_full = obs_keys + ["scaling_pending"]

    for _ in range(decisions):
        if env.t >= env.t_max:
            break

        # Build observation (same logic as TorchRLEnvWrapper._build_observation)
        obs_flat = _build_obs_flat(env, decision_interval, obs_keys, scaling_pending)
        cpu_util = _cpu_util(env, decision_interval)

        # Log per-decision obs to wandb (mirrors train_lstm.py obs/* logging)
        step_log = {"decision_step": global_decision_step + n_decisions}
        obs_2d = obs_flat.reshape(n_edges, -1)
        for j, name in enumerate(obs_keys_full):
            step_log[f"obs/{name}"] = float(
                obs_2d[0, j] if n_edges == 1 else np.mean(obs_2d[:, j])
            )
        wandb.log(step_log)

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

        delta_eff       = float(ids_cpu[0] - prev_ids_cpu[0])
        scaling_pending = float(np.clip(scaling_pending + delta_eff, -SCALING_K, SCALING_K))
        overhead_rate   = _overhead_rate(scaling_pending, scaling_time_steps)

        # Simulate decision_interval ticks
        for _ in range(decision_interval):
            if abs(scaling_pending) > 1e-9:
                consumed        = float(np.sign(scaling_pending)) * min(abs(scaling_pending), overhead_rate)
                scaling_pending -= consumed
                step_overhead   = consumed
            else:
                step_overhead = 0.0

            env.step(ids_cpu, step_overhead)

            r = _reward_components(env, n_edges, reward_alpha, reward_beta, reward_gamma, reward_q_th)
            total_reward      += r["reward"]
            total_lres        += r["lambda_res"]
            total_bcd         += r["benign_col_dmg"]
            total_qoe_penalty += r["qoe_penalty"]
            total_ticks       += 1

            if env.t >= env.t_max:
                break

        n_decisions += 1

    n = max(1, total_ticks)

    # QoE violation rate over the full episode
    qoe_vio_rate = 0.0
    if env.history:
        qoes = np.array([float(m.qoe_mean) for m in env.history], dtype=np.float32)
        qoe_vio_rate = float(np.mean(qoes < reward_q_th))

    return {
        "reward_mean":    total_reward      / n,
        "lambda_res":     total_lres        / n,
        "benign_col_dmg": total_bcd         / n,
        "qoe_penalty":    total_qoe_penalty / n,
        "final_qoe":      float(env.final_qoe),
        "qoe_vio_rate":   qoe_vio_rate,
    }, n_decisions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    env_cfg_path: str  = "./configs/simulation_0.yaml",
    train_cfg_path: str = "./configs/train.yaml",
    method: str        = "reactive",
    ckpt_path: Optional[str] = None,
    tbsa_table_path: Optional[str] = None,
    episodes: int      = 100,
    device: str        = "cpu",
):
    with open(env_cfg_path) as f:
        env_cfg = yaml.safe_load(f)
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)

    # Tag the wandb run with the method name so it sits alongside training runs
    if "logger" not in train_cfg or train_cfg["logger"] is None:
        train_cfg["logger"] = {}
    train_cfg["logger"]["exp_name"] = f"baseline_{method}"
    train_cfg["cfg_path"] = env_cfg_path

    run_wb = wandb_init(env_cfg, train_cfg)
    wandb.config.update({"baseline/method": method, "baseline/episodes": episodes}, allow_val_change=True)

    np.random.seed(env_cfg["run"]["seed"])

    decision_interval = env_cfg["globals"]["decision_interval"]

    # Instantiate a wrapper only to read obs_keys and reward weights, then discard
    _wrapper = TorchRLEnvWrapper(
        cfg_path=env_cfg_path,
        decision_interval=decision_interval,
        device="cpu",
    )
    obs_keys      = _wrapper.obs_keys
    reward_alpha  = _wrapper.reward_alpha
    reward_beta   = _wrapper.reward_beta
    reward_gamma  = _wrapper.reward_gamma
    reward_q_th   = _wrapper.reward_q_th
    del _wrapper

    # Build the raw Environment (no TorchRL wrapper needed for baseline eval)
    env = build_env_base(env_cfg_path)

    # Build policy
    policy = make_baseline_policy(
        name=method,
        tbsa_table_path=tbsa_table_path,
        ckpt_path=ckpt_path,
        obs_keys=obs_keys,
        device=device,
    )

    scale_step   = 0.5
    ids_cpu_min  = 0.5

    global_decision_step = 0
    best_qoe = -1e9

    for it in range(episodes):
        seed = env_cfg["run"]["seed"] + it * 1000

        metrics, n_dec = run_episode(
            env=env,
            cfg=env_cfg,
            policy=policy,
            obs_keys=obs_keys,
            decision_interval=decision_interval,
            scale_step=scale_step,
            ids_cpu_min=ids_cpu_min,
            seed=seed,
            reward_alpha=reward_alpha,
            reward_beta=reward_beta,
            reward_gamma=reward_gamma,
            reward_q_th=reward_q_th,
            global_decision_step=global_decision_step,
        )

        global_decision_step += n_dec

        print(
            f"Iter={it:4d} | rew={metrics['reward_mean']:+.4f} "
            f"| atk_pass={metrics['lambda_res']:.3f} "
            f"bcd={metrics['benign_col_dmg']:.3f} "
            f"qoe_sf={metrics['qoe_penalty']:.3f} "
            f"| qoe_vio={metrics['qoe_vio_rate']:.1%}"
        )

        if metrics["final_qoe"] > best_qoe:
            best_qoe = metrics["final_qoe"]

        # Log with same keys as train_lstm.py so wandb panels overlap
        wandb.log(
            {
                "iter":  it,
                # QoE
                "qoe/mean":     metrics["final_qoe"],
                "qoe/vio_rate": metrics["qoe_vio_rate"],
                # Reward
                "reward/mean":           metrics["reward_mean"],
                "reward/lambda_res":     metrics["lambda_res"],
                "reward/benign_col_dmg": metrics["benign_col_dmg"],
                "reward/qoe_penalty":    metrics["qoe_penalty"],
            }
        )

    print(f"\nBest QoE over {episodes} episodes: {best_qoe:.4f}")
    wandb.finish()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a baseline policy with wandb logging")
    parser.add_argument("--cfg",        default="./configs/simulation_0.yaml",
                        help="Environment config YAML")
    parser.add_argument("--train_cfg",  default="./configs/train.yaml",
                        help="Train config YAML (for wandb/logger settings)")
    parser.add_argument("--method",     default="reactive",
                        help="Policy: random | constant_<X> | reactive | tbsa | lstm_rl")
    parser.add_argument("--ckpt",       default=None,
                        help="Checkpoint path (required for lstm_rl)")
    parser.add_argument("--tbsa_table", default=None,
                        help="TBSA lookup-table path (required for tbsa)")
    parser.add_argument("--episodes",   type=int, default=100,
                        help="Number of episodes to run")
    parser.add_argument("--device",     default="cpu",
                        help="Torch device (cpu | cuda)")
    args = parser.parse_args()

    run(
        env_cfg_path=args.cfg,
        train_cfg_path=args.train_cfg,
        method=args.method,
        ckpt_path=args.ckpt,
        tbsa_table_path=args.tbsa_table,
        episodes=args.episodes,
        device=args.device,
    )
