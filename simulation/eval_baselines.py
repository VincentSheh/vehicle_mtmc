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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from environment import build_env_base, TorchRLEnvWrapper
from method_policy import ActContext, BaselinePolicy, make_baseline_policy
from old_policy import plot_ts_continuous, plot_qoe_vio_bars

SCALING_QUANTA = [0.5, 1.0, 1.5, 2.0]

DEFAULT_METHODS = [
    # "no_ids",
    "static_low", 
    "static_balanced",
    "static_high",
    # "autoscale_app", 
    # "autoscale_def",
    # "offline_optimal",
    "lstm_rl",
]
# DEFAULT_METHODS = [
#     "autoscale_def",
#     "offline_optimal",
#     "lstm_rl",
# ]

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
    transition_ticks_norm: float,
    delta_in_flight_norm: float,
    queue_ahead_norm: float,
) -> np.ndarray:
    n_edges = len(env.edge_areas)
    obs_dim = len(obs_keys) + 3   # +3: transition_ticks_norm, delta_in_flight_norm, queue_ahead_norm
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

    obs[:, -3] = float(np.clip(transition_ticks_norm, 0.0, 1.0))
    obs[:, -2] = float(np.clip(delta_in_flight_norm, -1.0, 1.0))
    obs[:, -1] = float(np.clip(queue_ahead_norm, -1.0, 1.0))
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

    n_edges     = len(env.edge_areas)
    ids_cpu_max = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)
    effective_min = float(policy.min_cpu_override) if policy.min_cpu_override is not None else ids_cpu_min

    if initial_ids_cpu is not None:
        ids_cpu = initial_ids_cpu.copy()
    else:
        ids_cpu = np.array([e.ids_cpu for e in env.edge_areas], dtype=np.float32)

    ids_cpu     = np.clip(ids_cpu, effective_min, ids_cpu_max)

    scaling_time_steps: List[int] = list(
        cfg["globals"].get("scaling_time_step", [300, 450, 498, 544])
    )
    ids_cpu_settled            = ids_cpu.copy()
    ids_cpu_target             = ids_cpu.copy()
    transition_ticks_remaining = 0
    transition_ticks_total     = 1
    max_scaling_duration       = float(scaling_time_steps[-1])
    max_delta                  = scale_step * (n_actions - 1) / 2.0   # largest single command
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
    qoe_vio_rate_ts       = []
    reward_lambda_res_ts  = []
    reward_bcd_ts         = []
    reward_qoe_penalty_ts = []
    reward_ts             = []
    for _ in range(decisions):
        if env.t >= env.t_max:
            break

        ticks_norm      = float(transition_ticks_remaining) / max(max_scaling_duration, 1.0)
        delta_in_flight = float(ids_cpu_target[0]) - float(ids_cpu_settled[0])
        d_flight_norm   = float(np.clip(delta_in_flight / max(max_delta, 1e-6), -1.0, 1.0))
        queue_ahead     = float(ids_cpu[0]) - float(ids_cpu_target[0])
        q_ahead_norm    = float(np.clip(queue_ahead / max(max_delta, 1e-6), -1.0, 1.0))
        obs_flat = _build_obs_flat(env, decision_interval, obs_keys, ticks_norm, d_flight_norm, q_ahead_norm)
        cpu_util = _cpu_util(env, decision_interval)

        ctx = ActContext(
            env=env,
            ids_cpu=ids_cpu.copy(),   # netting: policies base delta on current queue
            ids_cpu_min=effective_min,
            ids_cpu_max=ids_cpu_max,
            cpu_util=cpu_util,
            decision_interval=decision_interval,
            rng=rng,
            transition_ticks_norm=ticks_norm,
            delta_in_flight_norm=d_flight_norm,
            queue_ahead_norm=q_ahead_norm,
            obs_flat=obs_flat,
        )

        ids_cpu_abs, delta = policy.act(ctx)

        prev_ids_cpu = ids_cpu.copy()
        _max_q = SCALING_QUANTA[-1]   # 2.0 — max quantum the transition system supports
        if ids_cpu_abs is not None:
            # Abs-value policies (constant, tbsa, lstm_rl) return an absolute position;
            # treat it as the new queue value directly, but still apply the relative clamp
            # to settled state to match TorchRLEnvWrapper's netting behavior.
            ids_cpu = np.clip(
                ids_cpu_abs,
                np.maximum(effective_min, ids_cpu_settled - _max_q),
                np.minimum(ids_cpu_max, ids_cpu_settled + _max_q),
            ).astype(np.float32)
        else:
            # Delta policies: net delta onto current queue, clamped to settled ± max_quantum.
            ids_cpu = np.clip(
                ids_cpu + delta.astype(np.float32) * scale_step,
                np.maximum(effective_min, ids_cpu_settled - _max_q),
                np.minimum(ids_cpu_max, ids_cpu_settled + _max_q),
            ).astype(np.float32)

        delta_eff = float(ids_cpu[0] - prev_ids_cpu[0])

        # Settled: apply delta immediately (start transition).
        # In transition: net delta onto queue; ids_cpu_target unchanged until current
        # transition completes, then the accumulated queue fires as the next command.
        if transition_ticks_remaining <= 0:
            if abs(delta_eff) > 1e-9:
                ids_cpu_target[0] = ids_cpu[0]
                gap = abs(float(ids_cpu_target[0]) - float(ids_cpu_settled[0]))
                transition_ticks_total     = _lookup_scaling_duration(gap, scaling_time_steps)
                transition_ticks_remaining = transition_ticks_total
        # else: in transition — ids_cpu updated as queue; ids_cpu_target unchanged

        # Effective allocation during a transition.
        # Both directions: IDS holds at settled for the full duration.
        # scale-up:   VA drops immediately (overhead = -(target-settled) < 0)
        # scale-down: VA holds at settled, step_overhead = 0 (no VA penalty; cores
        #             are freed only when the timer expires and settled commits to target)
        delta_to_settled = float(ids_cpu_target[0]) - float(ids_cpu_settled[0])
        ids_cpu_eff = ids_cpu_settled.copy()      # IDS holds at settled in all cases
        step_overhead = -delta_to_settled if (transition_ticks_remaining > 0 and delta_to_settled > 1e-9) else 0.0

        for _ in range(decision_interval):
            if transition_ticks_remaining > 0:
                transition_ticks_remaining -= 1
                if transition_ticks_remaining == 0:
                    ids_cpu_settled[0] = ids_cpu_target[0]
                    queued_delta = float(ids_cpu[0]) - float(ids_cpu_settled[0])
                    if abs(queued_delta) > 1e-9:
                        ids_cpu_target[0] = ids_cpu[0]
                        gap = abs(queued_delta)
                        transition_ticks_total     = _lookup_scaling_duration(gap, scaling_time_steps)
                        transition_ticks_remaining = transition_ticks_total
                        new_d = float(ids_cpu_target[0]) - float(ids_cpu_settled[0])
                        ids_cpu_eff[0] = ids_cpu_settled[0]   # hold at settled (both directions)
                        step_overhead = -new_d if new_d > 1e-9 else 0.0
                    else:
                        ids_cpu_eff[0] = ids_cpu_settled[0]
                        step_overhead = 0.0
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
    # ap.add_argument("--ckpt",              default="checkpoints/rew_32_netting_a4_rew/ckpt_iter_000600.pt",
    # ap.add_argument("--ckpt",              default="checkpoints/tdsc/ckpt_iter_000500.pt",
    ap.add_argument("--ckpt",              default="checkpoints/rew_32_netting_a4_rew/ckpt_best.pt",
                    help="Checkpoint path (needed when 'lstm_rl' is in --methods)")
    ap.add_argument("--device",            default="cpu")
    args = ap.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # Sync global random state with training for AttackTypeLibrary (identical to train_lstm.py)
    base_seed = int(cfg["run"]["seed"])
    np.random.seed(base_seed)

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
    n_actions     = _wrapper.n_actions
    del _wrapper

    env = build_env_base(args.cfg)

    # Build one policy per method
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

    methods = [m for m in methods if m in policies]
    if not methods:
        raise SystemExit("No valid methods to evaluate.")

    # Accumulate per-decision results across episodes
    results: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
    # Track carry-over state per method (matches training SyncDataCollector behavior)
    last_ids_cpu: Dict[str, Optional[np.ndarray]] = {m: None for m in methods}

    for ep in tqdm(range(args.episodes), desc="episodes"):
        ep_seed = base_seed + (ep + 1) * 1000   # matches training: base_seed + episode_id*1000
        for mname in methods:
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
                initial_ids_cpu=last_ids_cpu[mname],
                # initial_ids_cpu=None,
            )
            last_ids_cpu[mname] = final_ids
            for k, v in ep_result.items():
                results[mname][k] = np.concatenate(
                    [results[mname].get(k, np.array([], dtype=np.float32)), v]
                )

    # Remap to display names for plots and summary
    display_results = {DISPLAY_NAMES.get(m, m): results[m] for m in methods}

    # Plot — reuse the plotting functions from old_policy.py
    ts_path      = outdir / "qoe_ts.png"
    summary_path = outdir / "summary.png"
    plot_ts_continuous(display_results, ts_path,      slo_qoe_min=reward_q_th, beta=3)
    plot_qoe_vio_bars( display_results, summary_path, qoe_slo_min=reward_q_th, beta=3)

    print(f"Plots saved to {outdir}/")

    # Summary table — comparable to wandb training metrics
    col_w = 26
    header = f"{'Method':<{col_w}} {'qoe_vio_rate':>12} {'reward/mean':>12} {'qoe_penalty':>12} {'atk_drop_pct':>12} {'lambda_res':>12}"
    print("\n" + header)
    print("-" * len(header))
    for mname in methods:
        r = results[mname]
        lres  = float(np.mean(r['reward_lambda_res']))
        label = DISPLAY_NAMES.get(mname, mname)
        print(
            f"{label:<{col_w}} "
            f"{float(np.mean(r['qoe_vio_rate'])):>12.1%} "
            f"{float(np.mean(r['reward'])):>12.4f} "
            f"{float(np.mean(r['reward_qoe_penalty'])):>12.4f} "
            f"{1.0 - lres:>12.1%} "   # attack drop % — matches summary.png bar
            f"{lres:>12.4f}"           # raw pass-through fraction (all ticks, matches training)
        )


if __name__ == "__main__":
    main()
