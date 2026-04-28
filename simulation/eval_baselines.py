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
import copy
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from environment import build_env_base
from environment import TorchRLEnvWrapper
from method_policy import ActContext, BaselinePolicy, make_baseline_policy, OFFLOAD_DISPLAY_NAMES
from matplotlib import pyplot as plt

SCALING_QUANTA = [0.5, 1.0, 1.5, 2.0]

DEFAULT_METHODS = [
    "no_ids",
    "static_low", 
    # "static_balanced",
    "static_high",
    # "autoscale_app", 
    "autoscale_def",
    "offline_optimal",
    "lstm_rl",
]
DEFAULT_METHODS = [
    "autoscale_def",
    "offline_optimal",
    "lstm_rl",
]

# Human-readable labels used in plots and summary table
DISPLAY_NAMES: Dict[str, str] = {
    "no_ids":          "No IDS",
    "static_low":      "Static Low",
    "static_balanced": "Static Balanced ",
    "static_high":     "Static High",
    "autoscale_app":   "Autoscale (App load)",
    "autoscale_def":   "Autoscale (Defense load)",
    "offline_optimal": "Offline Optimal (TBSA)",
    "lstm_rl":         "LSTM RL",
    # legacy / custom names fall through to raw name
}

def plot_ts_continuous(
    results: Dict[str, Dict[str, np.ndarray]],
    outpath: Path,
    area_ids: List[str],
    slo_qoe_min: float = 0.2,
    beta: int = 3,
):
    static_panels = [
        ("benign_col_dmg",    "Benign Collateral Damage"),
        ("local_num_req",     "Local #Req"),
        ("attack_in_rate",    "Attack in rate"),
        ("attack_drop_rate",  "Attack drop rate"),
        ("reward_lambda_res", "λ_res (raw attack pass-through)"),
        ("cpu_util",          "CPU Utilization"),
        ("cpu_to_ids_ratio",  "CPU→IDS Ratio"),
    ]

    n_edges  = len(area_ids)
    n_panels = n_edges + len(static_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(15, 3 * n_panels), sharex=True)

    # --- per-edge QoE panels ---
    for ei, area_id in enumerate(area_ids):
        ax = axes[ei]
        for method, series in results.items():
            qoe_pe = series.get("qoe_per_edge", None)
            if qoe_pe is not None and qoe_pe.ndim == 2 and qoe_pe.shape[1] > ei:
                y = qoe_pe[:, ei]
            else:
                y = series.get("qoe", None)
            if y is None or y.size == 0:
                continue

            x = np.arange(len(y))
            y_valid = y[np.isfinite(y)]
            avg_qoe  = float(np.nanmean(y_valid)) if y_valid.size else 0.0
            vio_rate = float(np.nanmean((y_valid < slo_qoe_min).astype(np.float32))) if y_valid.size else 0.0
            ax.plot(x, y, label=f"{method} (avg={avg_qoe:.3f}, vio={vio_rate:.2%})")

        ax.axhline(slo_qoe_min, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_ylabel(f"QoE [{area_id}]")
        ax.grid(True, alpha=0.3)
        if ei == 0:
            ax.legend(loc="upper right")

    # --- static panels ---
    for ax, (k, ylabel) in zip(axes[n_edges:], static_panels):
        for method, series in results.items():
            y_raw = series.get(k, None)
            if y_raw is None or y_raw.size == 0:
                continue
            if y_raw.ndim == 2:
                y     = np.mean(y_raw, axis=1)
                y_min = np.min(y_raw,  axis=1)
                y_max = np.max(y_raw,  axis=1)
            else:
                y = y_raw
                y_min = y_max = None
            x    = np.arange(len(y))
            line = ax.plot(x, y, label=method)[0]
            if y_min is not None:
                ax.fill_between(x, y_min, y_max, color=line.get_color(), alpha=0.15)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("decision step")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_qoe_vio_bars(results: Dict[str, Dict[str, np.ndarray]],
                      outpath: Path,
                      qoe_slo_min: float = 0.2,
                      beta: float = 3.0):
    methods, avg_qoe, vio_rate = [], [], []
    avg_benign_col_dmg, avg_attack_drop_pct = [], []

    for method, series in results.items():
        qoe_raw = series.get("qoe", None)
        if qoe_raw is None or qoe_raw.size == 0:
            continue

        # qoe_raw shape: (T,) or (T, E)
        if qoe_raw.ndim == 2:
            q = np.mean(qoe_raw, axis=1)  # average across edges for QoE bar
            vr = float(np.mean(qoe_raw < qoe_slo_min))  # per-edge violation across all steps
        else:
            q = qoe_raw
            # Prefer pre-calculated per-step violation rate if available
            vio_series = series.get("qoe_vio_rate", None)
            if vio_series is not None and vio_series.size > 0:
                vr = float(np.mean(vio_series))
            else:
                vr = float(np.nanmean((q < qoe_slo_min).astype(np.float32)))
        
        v = float(np.exp(-beta * vr))
        methods.append(method)
        vio_rate.append(vr)
        avg_qoe.append(float(np.mean(q)) * v)

        # Benign Collateral Damage
        bcd_raw = series.get("benign_col_dmg", None)
        if bcd_raw is None:
            avg_benign_col_dmg.append(np.nan)
        else:
            avg_benign_col_dmg.append(float(np.mean(bcd_raw)))

        # Attack Drop % — computed from raw rates so no-attack windows don't inflate the bar
        # (reward_lambda_res = 0 on no-attack windows, which reads as "100% dropped" when inverted)
        atk_in_raw  = series.get("attack_in_rate",  None)
        atk_drp_raw = series.get("attack_drop_rate", None)
        if atk_in_raw is not None and atk_drp_raw is not None:
            total_in  = float(atk_in_raw.sum())
            total_drp = float(atk_drp_raw.sum())
            avg_attack_drop_pct.append(total_drp / total_in if total_in > 1e-6 else 0.0)
        else:
            avg_attack_drop_pct.append(np.nan)

    x = np.arange(len(methods), dtype=np.int32)
    width = 0.7

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    bars_qoe = axes[0].bar(x, avg_qoe, width)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylabel("Average QoE")
    axes[0].set_title("Average QoE")
    axes[0].grid(axis="y", alpha=0.3)
    for bar in bars_qoe:
        h = float(bar.get_height())
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars_vio = axes[1].bar(x, vio_rate, width)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_ylabel("Violation Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(f"SLO Violations (QoE < {qoe_slo_min})")
    axes[1].grid(axis="y", alpha=0.3)
    for bar in bars_vio:
        h = float(bar.get_height())
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars_dmg = axes[2].bar(x, avg_benign_col_dmg, width)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=20, ha="right")
    axes[2].set_ylabel("Benign Collateral Damage")
    axes[2].set_title("Average Benign Collateral Damage")
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars_dmg, avg_benign_col_dmg):
        if np.isfinite(val):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                float(bar.get_height()),
                f"{float(val):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    bars_drop = axes[3].bar(x, avg_attack_drop_pct, width)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(methods, rotation=20, ha="right")
    axes[3].set_ylabel("Attack Drop %")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_title("Average Attack Drop Percentage")
    axes[3].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars_drop, avg_attack_drop_pct):
        if np.isfinite(val):
            axes[3].text(
                bar.get_x() + bar.get_width() / 2,
                float(bar.get_height()),
                f"{float(val):.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()



# ---------------------------------------------------------------------------
# Helpers (mirror TorchRLEnvWrapper internals)
# ---------------------------------------------------------------------------

def _lookup_scaling_duration(magnitude: float, scaling_time_steps: List[int]) -> int:
    for i, q in enumerate(SCALING_QUANTA):
        if magnitude <= q + 1e-9:
            return scaling_time_steps[i]
    return scaling_time_steps[-1]


def _build_obs_flat(
    env,
    decision_interval: int,
    obs_keys: List[str],
    ids_cpu_target: np.ndarray,
    ids_cpu_settled: np.ndarray,
    transition_ticks_remaining: np.ndarray,
    scaling_time_steps: List[int],
    scale_step: float,
    n_actions: int,
) -> np.ndarray:
    n_edges = len(env.edge_areas)
    area_ids = [e.area_id for e in env.edge_areas]
    # obs layout: [obs_keys(5)] [neighbor_ids_util, neighbor_delta, neighbor_atk_rate(3, multi-edge only)]
    #             [prev_slo_vio(1)] [transition_ticks_norm, delta_in_flight_norm(2)]
    # Single-edge omits neighbor block (8 dims) to match single-agent training layout.
    n_nbr = 3 if n_edges > 1 else 0
    obs_dim = len(obs_keys) + n_nbr + 1 + 2
    obs = np.zeros((n_edges, obs_dim), dtype=np.float32)

    if not env.history:
        return obs.reshape(-1).astype(np.float32)

    records = env.history[-decision_interval * n_edges:]
    df = pd.DataFrame([m.__dict__ for m in records])

    n_base = len(obs_keys)

    # 1. Per-edge base features
    for i, area_id in enumerate(area_ids):
        g = df[df["area_id"] == area_id]
        if g.empty:
            continue
        for j, k in enumerate(obs_keys):
            if k == "cpu_to_ids_ratio":
                obs[i, j] = float(g[k].values[-1])
            else:
                obs[i, j] = float(np.mean(g[k].values))

    # 2. Neighbor state features
    edge_ids_util: Dict[str, float] = {}
    edge_atk_rate: Dict[str, float] = {}
    for i, area_id in enumerate(area_ids):
        g = df[df["area_id"] == area_id]
        if g.empty:
            edge_ids_util[area_id] = 0.0
            edge_atk_rate[area_id] = 0.0
        else:
            edge_ids_util[area_id] = float(np.clip(np.mean(g["ids_cpu_utilization"].values), 0.0, 1.0))
            edge_atk_rate[area_id] = float(np.mean(g["attack_in_rate"].values))

    max_delta = scale_step * (n_actions - 1) / 2.0
    if n_nbr > 0:
        for i, area_id in enumerate(area_ids):
            nbr_utils: List[float] = []
            nbr_deltas: List[float] = []
            nbr_atk: List[float] = []
            for j in range(n_edges):
                if j == i:
                    continue
                other_id = area_ids[j]
                nbr_utils.append(edge_ids_util[other_id])
                nbr_atk.append(edge_atk_rate[other_id])
                delta = float(ids_cpu_target[j]) - float(ids_cpu_settled[j])
                nbr_deltas.append(float(np.clip(delta / max(max_delta, 1e-6), -1.0, 1.0)))

            obs[i, n_base]     = float(np.mean(nbr_utils))  if nbr_utils  else 0.0
            obs[i, n_base + 1] = float(np.mean(nbr_deltas)) if nbr_deltas else 0.0
            obs[i, n_base + 2] = float(np.mean(nbr_atk))    if nbr_atk    else 0.0

    # 3. Previous SLO violation flag
    for i, area_id in enumerate(area_ids):
        g = df[df["area_id"] == area_id]
        if g.empty:
            obs[i, n_base + n_nbr] = 0.0
            continue
        last_qoe = float(g["qoe_mean"].values[-1])
        threshold = float(env.edge_areas[i].slo_threshold)
        obs[i, n_base + n_nbr] = 1.0 if last_qoe < threshold else 0.0

    # 4. Transition state features
    max_dur = float(scaling_time_steps[-1])
    for i in range(n_edges):
        obs[i, -2] = float(transition_ticks_remaining[i]) / max(max_dur, 1.0)
        delta_in_flight = float(ids_cpu_target[i]) - float(ids_cpu_settled[i])
        obs[i, -1] = float(np.clip(delta_in_flight / max(max_delta, 1e-6), -1.0, 1.0))

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
    n_actions: int,
    ids_cpu_min: float,
    seed: int,
    reward_alpha: float,
    reward_beta: float,
    reward_gamma: float,
    reward_q_th: float,
    initial_ids_cpu: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Run one episode with *policy* and return a dict of per-decision arrays
    and the final ids_cpu state for carry-over.
    """
    env.reset(seed)
    policy.reset()

    n_edges      = len(env.edge_areas)
    area_ids_run = [e.area_id for e in env.edge_areas]
    ids_cpu_max  = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)
    effective_min = float(policy.min_cpu_override) if policy.min_cpu_override is not None else ids_cpu_min

    if initial_ids_cpu is not None:
        ids_cpu = initial_ids_cpu.copy()
    else:
        ids_cpu = np.array([e.ids_cpu for e in env.edge_areas], dtype=np.float32)

    ids_cpu = np.clip(ids_cpu, effective_min, ids_cpu_max)

    scaling_time_steps: List[int] = list(
        cfg["globals"].get("scaling_time_step", [300, 450, 498, 544])
    )
    ids_cpu_settled            = ids_cpu.copy()
    ids_cpu_target             = ids_cpu.copy()
    transition_ticks_remaining = np.zeros(n_edges, dtype=np.int32)
    transition_ticks_total     = np.ones(n_edges, dtype=np.int32)
    max_scaling_duration       = float(scaling_time_steps[-1])
    max_delta                  = scale_step * (n_actions - 1) / 2.0   # largest single command
    rng = np.random.default_rng(seed)

    decisions = math.ceil(int(cfg["run"]["t_max"]) / decision_interval)

    qoe_ts                = []
    qoe_per_edge_ts       = []
    benign_col_dmg_ts     = []
    cpu_util_ts           = []
    local_num_req_ts      = []
    attack_in_rate_ts     = []
    attack_in_rate_std_ts = []
    attack_drop_rate_ts   = []
    ema_mom_ts            = []
    cpu_to_ids_ratio_ts   = []
    qoe_vio_rate_ts       = []
    reward_lambda_res_ts  = []
    reward_bcd_ts         = []
    reward_qoe_penalty_ts = []
    reward_ts             = []
    for _ in range(decisions):
        if env.t >= env.t_max:
            break

        ticks_norm = transition_ticks_remaining.astype(np.float32) / max(max_scaling_duration, 1.0)
        delta_if   = ids_cpu_target - ids_cpu_settled
        dif_norm   = np.clip(delta_if / max(max_delta, 1e-6), -1.0, 1.0)

        obs_flat = _build_obs_flat(
            env, decision_interval, obs_keys,
            ids_cpu_target, ids_cpu_settled, transition_ticks_remaining,
            scaling_time_steps, scale_step, n_actions
        )
        cpu_util = _cpu_util(env, decision_interval)

        ctx = ActContext(
            env=env,
            ids_cpu=ids_cpu.copy(),   # current queue position
            ids_cpu_min=effective_min,
            ids_cpu_max=ids_cpu_max,
            cpu_util=cpu_util,
            decision_interval=decision_interval,
            rng=rng,
            transition_ticks_norm=ticks_norm,
            delta_in_flight_norm=dif_norm,
            obs_flat=obs_flat,
        )

        ids_cpu_abs, delta = policy.act(ctx)

        prev_ids_cpu = ids_cpu.copy()
        _max_q = 2.0  # SCALING_QUANTA[-1]

        for i in range(n_edges):
            if ids_cpu_abs is not None:
                # Absolute policy (clamped to settled ± max_quantum to match netting)
                ids_cpu[i] = np.clip(
                    ids_cpu_abs[i],
                    np.maximum(effective_min, ids_cpu_settled[i] - _max_q),
                    np.minimum(ids_cpu_max[i], ids_cpu_settled[i] + _max_q),
                )
            else:
                # Delta policy: net onto current queue
                ids_cpu[i] = np.clip(
                    ids_cpu[i] + float(delta[i]) * scale_step,
                    np.maximum(effective_min, ids_cpu_settled[i] - _max_q),
                    np.minimum(ids_cpu_max[i], ids_cpu_settled[i] + _max_q),
                )

            # Start transition if settled and desired allocation changed
            delta_eff = float(ids_cpu[i] - prev_ids_cpu[i])
            if transition_ticks_remaining[i] <= 0 and abs(delta_eff) > 1e-9:
                ids_cpu_target[i] = ids_cpu[i]
                gap = abs(float(ids_cpu_target[i]) - float(ids_cpu_settled[i]))
                transition_ticks_total[i]     = _lookup_scaling_duration(gap, scaling_time_steps)
                transition_ticks_remaining[i] = transition_ticks_total[i]

        for _ in range(decision_interval):
            ids_cpu_eff = ids_cpu_settled.copy()
            step_overhead = np.zeros(n_edges, dtype=np.float32)
            for i in range(n_edges):
                if transition_ticks_remaining[i] > 0:
                    delta_to_settled = float(ids_cpu_target[i]) - float(ids_cpu_settled[i])
                    if abs(delta_to_settled) > 1e-9:
                        step_overhead[i] = -delta_to_settled

                    transition_ticks_remaining[i] -= 1
                    if transition_ticks_remaining[i] == 0:
                        ids_cpu_settled[i] = ids_cpu_target[i]
                        queued_delta = float(ids_cpu[i]) - float(ids_cpu_settled[i])
                        if abs(queued_delta) > 1e-9:
                            ids_cpu_target[i] = ids_cpu[i]
                            gap = abs(queued_delta)
                            transition_ticks_total[i]     = _lookup_scaling_duration(gap, scaling_time_steps)
                            transition_ticks_remaining[i] = transition_ticks_total[i]
                            new_d = float(ids_cpu_target[i]) - float(ids_cpu_settled[i])
                            step_overhead[i] = -new_d if abs(new_d) > 1e-9 else 0.0
                            ids_cpu_eff[i] = ids_cpu_settled[i]
                        else:
                            ids_cpu_eff[i] = ids_cpu_settled[i]
                            step_overhead[i] = 0.0
            
            env.step(ids_cpu_eff, step_overhead)
            if env.t >= env.t_max:
                break

        # --- aggregate metrics over the just-completed decision window ---
        window = env.history[-decision_interval * n_edges:]
        df = pd.DataFrame([m.__dict__ for m in window])

        def _col_mean(col):
            return float(df[col].mean()) if col in df.columns else 0.0

        qoe_vals = df["qoe_mean"].values.astype(np.float32) if "qoe_mean" in df.columns else np.array([])
        qoe_mean = float(np.mean(qoe_vals)) if qoe_vals.size > 0 else 0.0
        
        # PER-STEP VIOLATION RATE (Matches TorchRLEnvWrapper)
        v_rate = float(np.mean(qoe_vals < reward_q_th)) if qoe_vals.size > 0 else 0.0
        
        bcd_mean = (
            float(np.mean(df["benign_col_dmg"].values)) if "benign_col_dmg" in df.columns else 0.0
        )

        qoe_ts.append(qoe_mean)
        qoe_vio_rate_ts.append(v_rate)
        per_edge_qoes = []
        for aid in area_ids_run:
            g = df[df["area_id"] == aid] if "area_id" in df.columns else pd.DataFrame()
            per_edge_qoes.append(float(np.mean(g["qoe_mean"].values)) if not g.empty and "qoe_mean" in g.columns else 0.0)
        qoe_per_edge_ts.append(per_edge_qoes)
        benign_col_dmg_ts.append(bcd_mean)
        cpu_util_ts.append(cpu_util)
        local_num_req_ts.append(_col_mean("local_num_req"))
        attack_in_rate_ts.append(_col_mean("attack_in_rate"))
        attack_in_rate_std_ts.append(
            float(df["attack_in_rate"].std()) if "attack_in_rate" in df.columns else 0.0
        )
        attack_drop_rate_ts.append(_col_mean("attack_drop_rate"))
        ema_mom_ts.append(_col_mean("ema_mom"))
        ratios = ids_cpu_settled / np.array([e.budget.cpu for e in env.edge_areas], dtype=np.float32)
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

            attack_mask   = atk_in > 1e-6
            r_lres_scalar = float(np.mean(lres[attack_mask])) if attack_mask.any() else 0.0
            r_bcd  = float(np.mean(bcd_vals))
            r_sf   = float(np.mean(sf))
            reward_lambda_res_ts.append(r_lres_scalar)
            reward_bcd_ts.append(r_bcd)
            reward_qoe_penalty_ts.append(r_sf)
            reward_ts.append(-(reward_alpha * r_sf + reward_beta * r_lres_scalar + reward_gamma * r_bcd))
        else:
            reward_lambda_res_ts.append(0.0)
            reward_bcd_ts.append(0.0)
            reward_qoe_penalty_ts.append(0.0)
            reward_ts.append(0.0)

    res = {
        "qoe":                  np.asarray(qoe_ts,                dtype=np.float32),
        "qoe_per_edge":         np.asarray(qoe_per_edge_ts,       dtype=np.float32),
        "qoe_vio_rate":         np.asarray(qoe_vio_rate_ts,       dtype=np.float32),
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
        "reward":               np.asarray(reward_ts,             dtype=np.float32),
    }
    return res, ids_cpu.copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate baseline policies and plot results")
    ap.add_argument("--cfg",               default="./configs/simulation_ma_0.yaml")
    ap.add_argument("--outdir",            default="eval_out")
    ap.add_argument("--episodes",          type=int,   default=10)
    ap.add_argument("--decision_interval", type=int,   default=None,
                    help="Overrides globals.decision_interval from cfg if set")
    ap.add_argument("--scale_step",        type=float, default=0.5)
    ap.add_argument("--ids_cpu_min",       type=float, default=0.5)
    ap.add_argument("--methods",           nargs="+",  default=None,
                    help="Methods to evaluate. Defaults: random constant_0.5 constant_4.0 reactive")
    ap.add_argument("--tbsa_table",        default="tbsa_table_15.npz",
                    help="TBSA lookup-table path (needed when 'tbsa' is in --methods)")
    ap.add_argument("--ckpt",              default="checkpoints/singleedge/rew_32_netting_a4_rew/ckpt_best.pt",
                    help="Checkpoint path (needed when 'lstm_rl' is in --methods)")
    ap.add_argument("--device",            default="cpu")
    ap.add_argument("--offload_modes",     nargs="+", default=None,
                    help="Offload modes to compare (e.g. none balance delay_workload full). "
                         "Defaults to the value in the config file.")
    args = ap.parse_args()

    with open(args.cfg) as f:
        cfg_original = yaml.safe_load(f)

    base_seed = int(cfg_original["run"]["seed"])
    np.random.seed(base_seed)

    decision_interval = args.decision_interval or int(cfg_original["globals"]["decision_interval"])

    outdir_base = Path(args.outdir)
    outdir_base.mkdir(parents=True, exist_ok=True)

    methods: List[str] = args.methods if args.methods else list(DEFAULT_METHODS)

    _wrapper      = TorchRLEnvWrapper(cfg_path=args.cfg, decision_interval=decision_interval, device="cpu")
    obs_keys      = _wrapper.obs_keys
    reward_alpha  = _wrapper.reward_alpha
    reward_beta   = _wrapper.reward_beta
    reward_gamma  = _wrapper.reward_gamma
    reward_q_th   = _wrapper.reward_q_th
    n_actions     = _wrapper.n_actions
    del _wrapper

    atk_lvls = ["low", "mid", "high"]
    user_lvls = ["low", "mid", "high"]

    # Resolve offload modes: explicit list or fall back to config value
    cfg_offload_mode = cfg_original["globals"].get("offload_mode", "balance")
    offload_modes: List[str] = args.offload_modes if args.offload_modes else [cfg_offload_mode]
    multi_offload = len(offload_modes) > 1

    for atk_lvl in atk_lvls:
        for user_lvl in user_lvls:
            print(f"\n>>> Evaluating Levels: Attack={atk_lvl}, User={user_lvl}")

            cfg_base = copy.deepcopy(cfg_original)
            if "attack_sampler" in cfg_base.get("globals", {}):
                cfg_base["globals"]["attack_sampler"]["level"] = atk_lvl
            if "user_sampler" in cfg_base.get("globals", {}) and "synthetic" in cfg_base["globals"]["user_sampler"]:
                cfg_base["globals"]["user_sampler"]["synthetic"]["level"] = user_lvl

            tbsa_table_path = Path(args.tbsa_table)
            policies: Dict[str, BaselinePolicy] = {}
            for name in methods:
                tbsa_path = str(tbsa_table_path) if name in ("tbsa", "offline_optimal") else None
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

            valid_methods = [m for m in methods if m in policies]
            if not valid_methods:
                print(f"[skip] No valid methods for Attack={atk_lvl}, User={user_lvl}")
                continue

            # results keyed by "{method}[{offload_mode}]" when comparing multiple modes,
            # or just "{method}" when a single mode is evaluated.
            results: Dict[str, Dict[str, np.ndarray]] = {}
            last_ids_cpu: Dict[str, Optional[np.ndarray]] = {}

            for offload_mode in offload_modes:
                cfg = copy.deepcopy(cfg_base)
                cfg["globals"]["offload_mode"] = offload_mode

                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                    yaml.dump(cfg, tmp)
                    tmp_cfg_path = tmp.name

                try:
                    env = build_env_base(tmp_cfg_path)
                finally:
                    if os.path.exists(tmp_cfg_path):
                        os.remove(tmp_cfg_path)

                for mname in valid_methods:
                    rkey = f"{mname}[{offload_mode}]" if multi_offload else mname
                    results[rkey] = {}
                    last_ids_cpu[rkey] = None

                for ep in tqdm(range(args.episodes),
                               desc=f"episodes ({atk_lvl}/{user_lvl}/offload={offload_mode})"):
                    ep_seed = base_seed + (ep + 1) * 1000
                    for mname in valid_methods:
                        rkey = f"{mname}[{offload_mode}]" if multi_offload else mname
                        ep_result, final_ids = run_episode(
                            env=env,
                            cfg=cfg,
                            policy=policies[mname],
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
                            initial_ids_cpu=last_ids_cpu[rkey],
                        )
                        last_ids_cpu[rkey] = final_ids
                        for k, v in ep_result.items():
                            existing = results[rkey].get(k)
                            if existing is None or existing.size == 0:
                                results[rkey][k] = v
                            else:
                                results[rkey][k] = np.concatenate([existing, v], axis=0)

            lvl_outdir = outdir_base / f"atk_{atk_lvl}_user_{user_lvl}"
            lvl_outdir.mkdir(parents=True, exist_ok=True)

            def _display_label(rkey: str) -> str:
                if multi_offload and "[" in rkey:
                    mname, om = rkey.rsplit("[", 1)
                    om = om.rstrip("]")
                    base = DISPLAY_NAMES.get(mname, mname)
                    ol   = OFFLOAD_DISPLAY_NAMES.get(om, om)
                    return f"{base} ({ol})"
                return DISPLAY_NAMES.get(rkey, rkey)

            display_results = {_display_label(rk): results[rk] for rk in results}
            area_ids = [e.area_id for e in env.edge_areas]
            ts_path      = lvl_outdir / "qoe_ts.png"
            summary_path = lvl_outdir / "summary.png"
            plot_ts_continuous(display_results, ts_path, area_ids=area_ids, slo_qoe_min=reward_q_th, beta=3)
            plot_qoe_vio_bars( display_results, summary_path, qoe_slo_min=reward_q_th, beta=3)

            print(f"Plots saved to {lvl_outdir}/")

            col_w = 32
            header = f"{'Method':<{col_w}} {'qoe_vio_rate':>12} {'reward/mean':>12} {'qoe_penalty':>12} {'atk_drop_pct':>12} {'lambda_res':>12}"
            print("\n" + header)
            print("-" * len(header))
            for rkey in results:
                r = results[rkey]
                atk_in_sum  = float(r['attack_in_rate'].sum())
                atk_drp_sum = float(r['attack_drop_rate'].sum())
                atk_drop_pct = atk_drp_sum / atk_in_sum if atk_in_sum > 1e-6 else 0.0
                label = _display_label(rkey)
                print(
                    f"{label:<{col_w}} "
                    f"{float(np.mean(r['qoe_vio_rate'])):>12.1%} "
                    f"{float(np.mean(r['reward'])):>12.4f} "
                    f"{float(np.mean(r['reward_qoe_penalty'])):>12.4f} "
                    f"{atk_drop_pct:>12.1%} "
                    f"{1.0 - atk_drop_pct:>12.4f}"
                )


if __name__ == "__main__":
    main()
