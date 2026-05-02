from __future__ import annotations

import argparse
import copy
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from environment import build_env_base
from train_sa_lstm import TorchRLEnvWrapper
from method_policy import (
    ActContext, 
    BaselinePolicy, 
    make_baseline_policy, 
    OFFLOAD_DISPLAY_NAMES, 
    MODEL_DISPLAY_NAMES
)

# --- Shared Constants ---

SCALING_QUANTA = [0.5, 1.0, 1.5, 2.0]

DEFAULT_METHODS = [
    # "no_ids",
    # "static_low",
    # "static_high",
    # "autoscale_def",
    # "offline_optimal",
]

PROPOSED_CONFIGS = [
    ("gm", "delay_workload"),
    # ("gm", "cto"),
    ("lm", "delay_workload"),
    # ("lm", "cto_acc_inv"),
    ("lm", "cto"),
    ("lm", "cto_acc"),
]

DISPLAY_NAMES: Dict[str, str] = {
    "no_ids":          "No IDS",
    "static_low":      "Static Low",
    "static_balanced": "Static Balanced",
    "static_high":     "Static High",
    "autoscale_app":   "Autoscale (App)",
    "autoscale_def":   "Autoscale (Def)",
    "offline_optimal": "TBSA Optimal",
    "lstm_rl":         "LSTM RL",
    "ma_lstm_rl":      "MA LSTM RL",
    "gm":              "Global Model",
    "lm":              "Local Model",
}

METRICS = [
    ("slo_vio",  "SLO Violation Rate"),
    ("bcd",      "Benign Collateral Damage"),
    ("atk_leak", "Attack Leakage %"),
    ("realloc",  "Avg. Reallocations"),
]

# --- Core Episode Logic (Extracted from eval_baselines.py) ---

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
    n_nbr = 3 if n_edges > 1 else 0
    obs_dim = len(obs_keys) + n_nbr + 1 + 2
    obs = np.zeros((n_edges, obs_dim), dtype=np.float32)

    if not env.history:
        return obs.reshape(-1).astype(np.float32)

    records = env.history[-decision_interval * n_edges:]
    df = pd.DataFrame([m.__dict__ for m in records])
    n_base = len(obs_keys)

    for i, area_id in enumerate(area_ids):
        g = df[df["area_id"] == area_id]
        if g.empty: continue
        for j, k in enumerate(obs_keys):
            if k == "cpu_to_ids_ratio":
                obs[i, j] = float(g[k].values[-1])
            else:
                obs[i, j] = float(np.mean(g[k].values))

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
            nbr_utils, nbr_deltas, nbr_atk = [], [], []
            for j in range(n_edges):
                if j == i: continue
                other_id = area_ids[j]
                nbr_utils.append(edge_ids_util[other_id])
                nbr_atk.append(edge_atk_rate[other_id])
                delta = float(ids_cpu_target[j]) - float(ids_cpu_settled[j])
                nbr_deltas.append(float(np.clip(delta / max(max_delta, 1e-6), -1.0, 1.0)))
            obs[i, n_base]     = float(np.mean(nbr_utils))  if nbr_utils  else 0.0
            obs[i, n_base + 1] = float(np.mean(nbr_deltas)) if nbr_deltas else 0.0
            obs[i, n_base + 2] = float(np.mean(nbr_atk))    if nbr_atk    else 0.0

    for i, area_id in enumerate(area_ids):
        g = df[df["area_id"] == area_id]
        if g.empty:
            obs[i, n_base + n_nbr] = 0.0
            continue
        last_qoe = float(g["qoe_mean"].values[-1])
        threshold = float(env.edge_areas[i].slo_threshold)
        obs[i, n_base + n_nbr] = 1.0 if last_qoe < threshold else 0.0

    max_dur = float(scaling_time_steps[-1])
    for i in range(n_edges):
        obs[i, -2] = float(transition_ticks_remaining[i]) / max(max_dur, 1.0)
        delta_in_flight = float(ids_cpu_target[i]) - float(ids_cpu_settled[i])
        obs[i, -1] = float(np.clip(delta_in_flight / max(max_delta, 1e-6), -1.0, 1.0))

    return obs.reshape(-1).astype(np.float32)

def _get_cpu_utils(env, decision_interval: int) -> np.ndarray:
    n_edges = len(env.edge_areas)
    if len(env.history) < decision_interval * n_edges:
        return np.zeros(n_edges, dtype=np.float32)
    records = env.history[-decision_interval * n_edges:]
    df = pd.DataFrame([m.__dict__ for m in records])
    utils = np.zeros(n_edges, dtype=np.float32)
    for i, edge in enumerate(env.edge_areas):
        g = df[df["area_id"] == edge.area_id]
        if g.empty or "ids_cpu_utilization" not in g.columns:
            continue
        utils[i] = float(np.clip(np.mean(g["ids_cpu_utilization"].values), 0.0, 1.0))
    return utils

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
    max_delta                  = scale_step * (n_actions - 1) / 2.0
    rng = np.random.default_rng(seed)

    decisions = math.ceil(int(cfg["run"]["t_max"]) / decision_interval)

    metrics_ts: Dict[str, List] = {
        "qoe": [], "qoe_per_edge": [], "qoe_vio_rate": [], "benign_col_dmg": [],
        "cpu_util": [], "local_num_req": [], "attack_in_rate": [], "attack_in_rate_std": [],
        "attack_drop_rate": [], "ema_mom": [], "cpu_to_ids_ratio": [],
        "reward_lambda_res": [], "reward_benign_col_dmg": [], "reward_qoe_penalty": [], "reward": []
    }

    for _ in range(decisions):
        if env.t >= env.t_max: break

        ticks_norm = transition_ticks_remaining.astype(np.float32) / max(max_scaling_duration, 1.0)
        dif_norm   = np.clip((ids_cpu_target - ids_cpu_settled) / max(max_delta, 1e-6), -1.0, 1.0)

        obs_flat = _build_obs_flat(
            env, decision_interval, obs_keys,
            ids_cpu_target, ids_cpu_settled, transition_ticks_remaining,
            scaling_time_steps, scale_step, n_actions
        )
        cpu_utils = _get_cpu_utils(env, decision_interval)
        cpu_util  = float(np.max(cpu_utils))

        queue_ahead_norm = np.clip((ids_cpu - ids_cpu_target) / max(max_delta, 1e-6), -1.0, 1.0)
        ctx = ActContext(
            env=env, ids_cpu=ids_cpu.copy(), ids_cpu_min=effective_min, ids_cpu_max=ids_cpu_max,
            cpu_util=cpu_util, decision_interval=decision_interval, rng=rng,
            transition_ticks_norm=ticks_norm, delta_in_flight_norm=dif_norm,
            obs_flat=obs_flat, queue_ahead_norm=queue_ahead_norm,
            cpu_utils=cpu_utils,
        )

        ids_cpu_abs, delta = policy.act(ctx)
        prev_ids_cpu = ids_cpu.copy()
        _max_q = 2.0

        for i in range(n_edges):
            if ids_cpu_abs is not None:
                ids_cpu[i] = np.clip(ids_cpu_abs[i], np.maximum(effective_min, ids_cpu_settled[i] - _max_q), np.minimum(ids_cpu_max[i], ids_cpu_settled[i] + _max_q))
            else:
                ids_cpu[i] = np.clip(ids_cpu[i] + float(delta[i]) * scale_step, np.maximum(effective_min, ids_cpu_settled[i] - _max_q), np.minimum(ids_cpu_max[i], ids_cpu_settled[i] + _max_q))

            if transition_ticks_remaining[i] <= 0 and abs(float(ids_cpu[i] - prev_ids_cpu[i])) > 1e-9:
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
                    if abs(delta_to_settled) > 1e-9: step_overhead[i] = -delta_to_settled
                    transition_ticks_remaining[i] -= 1
                    if transition_ticks_remaining[i] == 0:
                        ids_cpu_settled[i] = ids_cpu_target[i]
                        queued_delta = float(ids_cpu[i]) - float(ids_cpu_settled[i])
                        if abs(queued_delta) > 1e-9:
                            ids_cpu_target[i] = ids_cpu[i]
                            transition_ticks_total[i]     = _lookup_scaling_duration(abs(queued_delta), scaling_time_steps)
                            transition_ticks_remaining[i] = transition_ticks_total[i]
                            new_d = float(ids_cpu_target[i]) - float(ids_cpu_settled[i])
                            step_overhead[i] = -new_d if abs(new_d) > 1e-9 else 0.0
            env.step(ids_cpu_eff, step_overhead)
            if env.t >= env.t_max: break

        window = env.history[-decision_interval * n_edges:]
        df = pd.DataFrame([m.__dict__ for m in window])
        qoe_vals = df["qoe_mean"].values.astype(np.float32) if "qoe_mean" in df.columns else np.array([])
        
        metrics_ts["qoe"].append(float(np.mean(qoe_vals)) if qoe_vals.size > 0 else 0.0)
        metrics_ts["qoe_vio_rate"].append(float(np.mean(qoe_vals < reward_q_th)) if qoe_vals.size > 0 else 0.0)
        metrics_ts["benign_col_dmg"].append(float(np.mean(df["benign_col_dmg"].values)) if "benign_col_dmg" in df.columns else 0.0)
        metrics_ts["cpu_util"].append(cpu_util)
        
        per_edge_qoes = []
        for aid in area_ids_run:
            g = df[df["area_id"] == aid] if "area_id" in df.columns else pd.DataFrame()
            per_edge_qoes.append(float(np.mean(g["qoe_mean"].values)) if not g.empty and "qoe_mean" in g.columns else 0.0)
        metrics_ts["qoe_per_edge"].append(per_edge_qoes)
        
        for k in ["local_num_req", "attack_in_rate", "attack_drop_rate", "ema_mom"]:
            metrics_ts[k].append(float(df[k].mean()) if k in df.columns else 0.0)
        metrics_ts["attack_in_rate_std"].append(float(df["attack_in_rate"].std()) if "attack_in_rate" in df.columns else 0.0)
        
        ratios = ids_cpu_settled / np.array([e.budget.cpu for e in env.edge_areas], dtype=np.float32)
        metrics_ts["cpu_to_ids_ratio"].append(float(ratios.mean()))

        if "attack_in_rate" in df.columns and "attack_drop_rate" in df.columns:
            atk_in, atk_drop = df["attack_in_rate"].values.astype(np.float32), df["attack_drop_rate"].values.astype(np.float32)
            atk_pass = np.maximum(0.0, atk_in - atk_drop)
            lres = np.divide(atk_pass, atk_in, out=np.zeros_like(atk_pass), where=atk_in > 1e-6)
            sf = np.maximum(0.0, reward_q_th - (df["qoe_mean"].values.astype(np.float32) if "qoe_mean" in df.columns else np.zeros(len(atk_in)))) / max(reward_q_th, 1e-6)
            bcd_vals = df["benign_col_dmg"].values.astype(np.float32) if "benign_col_dmg" in df.columns else np.zeros(len(atk_in))
            mask = atk_in > 1e-6
            r_lres = float(np.mean(lres[mask])) if mask.any() else 0.0
            r_bcd, r_sf = float(np.mean(bcd_vals)), float(np.mean(sf))
            metrics_ts["reward_lambda_res"].append(r_lres)
            metrics_ts["reward_benign_col_dmg"].append(r_bcd)
            metrics_ts["reward_qoe_penalty"].append(r_sf)
            metrics_ts["reward"].append(-(reward_alpha * r_sf + reward_beta * r_lres + reward_gamma * r_bcd))
        else:
            for k in ["reward_lambda_res", "reward_benign_col_dmg", "reward_qoe_penalty", "reward"]: metrics_ts[k].append(0.0)

    return {k: np.asarray(v, dtype=np.float32) for k, v in metrics_ts.items()}, ids_cpu.copy()

# --- BaseEvaluator Class ---

class BaseEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        with open(args.cfg) as f:
            self.cfg_original = yaml.safe_load(f)
        
        self.base_seed = int(self.cfg_original["run"]["seed"])
        np.random.seed(self.base_seed)
        
        self.decision_interval = args.decision_interval or int(self.cfg_original["globals"]["decision_interval"])
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        
        # Setup obs/reward params from a temporary env wrapper
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(self.cfg_original, tmp)
            tmp_path = tmp.name
        try:
            wrapper = TorchRLEnvWrapper(cfg_path=tmp_path, decision_interval=self.decision_interval, device="cpu")
            self.obs_keys = wrapper.obs_keys
            self.reward_alpha = wrapper.reward_alpha
            self.reward_beta = wrapper.reward_beta
            self.reward_gamma = wrapper.reward_gamma
            self.reward_q_th = wrapper.reward_q_th
            self.n_actions = wrapper.n_actions
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

    def get_methods(self) -> List[str]:
        methods = list(self.args.methods) if self.args.methods else list(DEFAULT_METHODS)
        if hasattr(self.args, "proposed_method") and self.args.proposed_method and self.args.proposed_method not in methods:
            methods.append(self.args.proposed_method)
        return methods

    def build_policies(self, methods: List[str]) -> Dict[str, BaselinePolicy]:
        policies = {}
        for name in methods:
            tbsa_path = str(self.args.tbsa_table) if name in ("tbsa", "offline_optimal") else None
            ckpt = self.args.ckpt if name in ("lstm_rl", "ma_lstm_rl") else None
            ok_keys = self.obs_keys if name == "lstm_rl" else None
            try:
                policies[name] = make_baseline_policy(name, tbsa_table_path=tbsa_path, ckpt_path=ckpt, obs_keys=ok_keys, device=self.args.device)
            except (ValueError, FileNotFoundError) as exc:
                print(f"[warn] Skipping '{name}': {exc}")
        return policies

    def mutate_cfg(self, cfg: dict, *args, **kwargs) -> dict:
        """Override in subclasses to modify config for each simulation run."""
        return cfg

    def run_simulation(self, env, cfg, policy, label, initial_ids_cpu=None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        accumulated = {}
        last_ids = initial_ids_cpu
        for ep in tqdm(range(self.args.episodes), desc=f"episodes ({label})", leave=False):
            ep_seed = self.base_seed + (ep + 1) * 1000
            res, last_ids = run_episode(
                env=env, cfg=cfg, policy=policy, obs_keys=self.obs_keys,
                decision_interval=self.decision_interval, scale_step=self.args.scale_step,
                n_actions=self.n_actions, ids_cpu_min=self.args.ids_cpu_min, seed=ep_seed,
                reward_alpha=self.reward_alpha, reward_beta=self.reward_beta,
                reward_gamma=self.reward_gamma, reward_q_th=self.reward_q_th,
                initial_ids_cpu=last_ids,
            )
            # Custom metric: reallocations
            ratios = res["cpu_to_ids_ratio"]
            res["reallocations"] = np.array([np.sum(np.abs(np.diff(ratios)) > 1e-6)] if ratios.size > 1 else [0], dtype=np.float32)
            
            for k, v in res.items():
                accumulated[k] = np.concatenate([accumulated[k], v]) if k in accumulated else v
        return accumulated, last_ids

    @staticmethod
    def extract_metrics(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        atk_in, atk_drp = arrays.get("attack_in_rate", np.array([])), arrays.get("attack_drop_rate", np.array([]))
        if atk_in.size > 0 and atk_drp.size == atk_in.size:
            atk_pass = np.maximum(0.0, atk_in - atk_drp)
            lres = np.divide(atk_pass, atk_in, out=np.zeros_like(atk_pass), where=atk_in > 1e-6)
            mask = atk_in > 1e-6
            atk_leak = float(np.mean(lres[mask])) if mask.any() else 0.0
        else: atk_leak = np.nan

        return {
            "slo_vio":  float(np.mean(arrays["qoe_vio_rate"])) if "qoe_vio_rate" in arrays else np.nan,
            "bcd":      float(np.mean(arrays["reward_benign_col_dmg"])) if "reward_benign_col_dmg" in arrays else np.nan,
            "atk_leak": atk_leak,
            "atk_drop": 1.0 - atk_leak if not np.isnan(atk_leak) else np.nan,
            "realloc":  float(np.mean(arrays["reallocations"])) if "reallocations" in arrays else np.nan,
            "reward":   float(np.mean(arrays["reward"])) if "reward" in arrays else np.nan,
        }

    @staticmethod
    def make_display_label(mname: str, offload_mode: str, show_offload: bool) -> str:
        base = DISPLAY_NAMES.get(mname, mname)
        if show_offload:
            ol = OFFLOAD_DISPLAY_NAMES.get(offload_mode, offload_mode)
            return f"{base} ({ol})"
        return base

    @staticmethod
    def make_proposed_display_label(mname: str, model_key: str, offload_mode: str) -> str:
        return f"{mname} {offload_mode}_{model_key}"
