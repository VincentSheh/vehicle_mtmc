# evaluate.py
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from environment import build_env_base  # your project
from train import ActorNet
from tqdm import tqdm


class RLPolicy:
    def __init__(self, ckpt_path: str, obs_dim: int, device: str = "cpu", greedy: bool = True):
        self.device = torch.device(device)
        self.greedy = greedy
        self.net = ActorNet(obs_dim=obs_dim, n_actions=3).to(self.device)
        

        state = torch.load(ckpt_path, map_location=self.device)
        self.obsnorm = state["obsnorm"]
        if isinstance(state, dict) and "actor_net" in state:
            self.net.load_state_dict(state["actor_net"])
        elif isinstance(state, dict) and "state_dict" in state:
            self.net.load_state_dict(state["state_dict"])
        elif isinstance(state, dict):
            self.net.load_state_dict(state)
        else:
            raise ValueError("Unsupported checkpoint format for RL actor.")

        self.net.eval()
        
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obsnorm is None:
            return obs
        loc = self.obsnorm["loc"].detach().cpu().numpy()
        scale = self.obsnorm["scale"].detach().cpu().numpy()
        return (obs - loc) / (scale + 1e-8)        

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (n_edges, obs_dim)
        returns action per edge in {-1,0,+1} as np.int64
        """
        x = torch.from_numpy(obs).to(self.device)
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        if self.greedy:
            a = torch.argmax(probs, dim=-1)
        else:
            a = torch.multinomial(probs, num_samples=1).squeeze(-1)
        delta = (a.to(torch.int64) - 1).detach().cpu().numpy()
        return delta


# ----------------------------
# Observation builder (match your wrapper logic)
# ----------------------------
def build_observation_from_history(env, decision_interval: int, obs_keys: List[str]) -> np.ndarray:
    n_edges = len(env.edge_areas)
    obs_dim = len(obs_keys)
    obs = np.zeros((n_edges, obs_dim), dtype=np.float32)

    if not env.history:
        return obs

    records = env.history[-decision_interval * n_edges :]
    df = pd.DataFrame([m.__dict__ for m in records])

    area_ids = [e.area_id for e in env.edge_areas]
    for i, area_id in enumerate(area_ids):
        g = df[df["area_id"] == area_id]
        if g.empty:
            continue
        for j, k in enumerate(obs_keys):
            vals = g[k].values
            if k == "I_net":
                obs[i, j] = float(np.sum(vals))
            elif k == "cpu_to_ids_ratio":
                obs[i, j] = float(vals[-1])
            else:
                obs[i, j] = float(np.mean(vals))
    return obs


def decision_qoe_mean(env, decision_interval: int) -> float:
    n_edges = len(env.edge_areas)
    if len(env.history) < decision_interval * n_edges:
        return 0.0
    block = env.history[-decision_interval * n_edges :]
    return float(np.mean([m.qoe_mean for m in block]))


def decision_cpu_util(env, decision_interval: int) -> float:
    """
    Reactive trigger signal based ONLY on IDS CPU utilization.
    Take max across edges (worst-case).
    """
    n_edges = len(env.edge_areas)
    if len(env.history) < decision_interval * n_edges:
        return 0.0

    block = env.history[-decision_interval * n_edges :]
    df = pd.DataFrame([m.__dict__ for m in block])

    utils = []
    attack_in_rate_ts = []
    local_gt_num_objects_ts = []    
    for area_id in [e.area_id for e in env.edge_areas]:
        g = df[df["area_id"] == area_id]
        if g.empty or "ids_cpu_utilization" not in g:
            continue

        ids_util = float(np.mean(g["ids_cpu_utilization"].values))
        utils.append(float(np.clip(ids_util, 0.0, 1.0)))
        attack_in_rate_ts.append(float(df["attack_in_rate"].mean()) if "attack_in_rate" in df else 0.0)
        local_gt_num_objects_ts.append(float(df["local_gt_num_objects"].mean()) if "local_gt_num_objects" in df else 0.0)        

    return float(max(utils)) if utils else 0.0


# ----------------------------
# Methods
# ----------------------------
def apply_delta(ids_cpu: np.ndarray, delta: np.ndarray, scale_step: float, ids_cpu_min: float, ids_cpu_max: np.ndarray) -> np.ndarray:
    out = ids_cpu + delta.astype(np.float32) * float(scale_step)
    out = np.maximum(out, ids_cpu_min)
    out = np.minimum(out, ids_cpu_max)
    return out


def run_episode(
    cfg: dict,
    cfg_path: str,
    method: str,
    decision_interval: int,
    obs_keys: List[str],
    scale_step: float,
    ids_cpu_min: float,
    constant_ids_cpu: Optional[float],
    seed: int,
    rl_policy: Optional[RLPolicy],
) -> np.ndarray:
    env = build_env_base(cfg_path)

    # per-episode randomness
    env.reset()
    for i, edge in enumerate(env.edge_areas):
        edge.cpu_to_ids_ratio = 0.5
        if hasattr(edge, "reset"):
            edge.reset(seed=seed + 100 * i)

    rng = np.random.default_rng(seed)

    t_max = int(cfg["run"]["t_max"])
    n_edges = len(env.edge_areas)
    ids_cpu_max = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)

    ids_cpu = np.array([e.cpu_to_ids_ratio * e.budget.cpu for e in env.edge_areas], dtype=np.float32)
    ids_cpu = np.clip(ids_cpu, ids_cpu_min, ids_cpu_max)

    if constant_ids_cpu is not None:
        ids_cpu = np.clip(np.full(n_edges, float(constant_ids_cpu), dtype=np.float32), ids_cpu_min, ids_cpu_max)

    decisions = math.ceil(t_max / decision_interval)
    qoes: List[float] = []
    qoe_ts = []
    cpu_util_ts = []
    local_num_obj_ts = []
    cpu_to_ids_ratio_ts = []

    local_gt_num_obj_ts = []
    attack_in_rate_ts = []

    for _k in tqdm(range(decisions)):
        if env.t >= env.t_max:
            break

        obs = build_observation_from_history(env, decision_interval, obs_keys)
        cpu_util = decision_cpu_util(env, decision_interval)

        # ---------- policy ----------
        if method == "constant":
            delta = np.zeros(n_edges, dtype=np.int64)
        elif method == "random":
            delta = rng.integers(-1, 2, size=n_edges, dtype=np.int64)
        elif method == "reactive":
            if cpu_util >= 0.80:
                delta = np.ones(n_edges, dtype=np.int64)
            elif cpu_util <= 0.20:
                delta = -np.ones(n_edges, dtype=np.int64)
            else:
                delta = np.zeros(n_edges, dtype=np.int64)
        elif method == "rl":
            if rl_policy.obsnorm is not None:
                obs = rl_policy.normalize_obs(obs)            
            delta = rl_policy.act(obs)
        else:
            raise ValueError(method)

        ids_cpu = apply_delta(ids_cpu, delta, scale_step, ids_cpu_min, ids_cpu_max)

        for _ in range(decision_interval):
            env.step(ids_cpu)
            if env.t >= env.t_max:
                break

        # ---------- metrics ----------
        qoe = decision_qoe_mean(env, decision_interval)

        block = env.history[-decision_interval * n_edges :]
        df = pd.DataFrame([m.__dict__ for m in block])

        qoe_ts.append(qoe)
        cpu_util_ts.append(cpu_util)
        local_num_obj_ts.append(df["local_num_objects"].mean())
        cpu_to_ids_ratio_ts.append(df["cpu_to_ids_ratio"].mean())

        # new metrics
        if "local_gt_num_objects" in df.columns:
            local_gt_num_obj_ts.append(df["local_gt_num_objects"].mean())
        else:
            local_gt_num_obj_ts.append(0.0)

        if "attack_in_rate" in df.columns:
            attack_in_rate_ts.append(df["attack_in_rate"].mean())
        else:
            attack_in_rate_ts.append(0.0)

    return {
        "qoe": np.array(qoe_ts),
        "cpu_util": np.array(cpu_util_ts),
        "local_num_objects": np.array(local_num_obj_ts),
        "local_gt_num_objects": np.array(local_gt_num_obj_ts),
        "attack_in_rate": np.array(attack_in_rate_ts),
        "cpu_to_ids_ratio": np.array(cpu_to_ids_ratio_ts),
    }
        
def plot_ts_continuous(results, outpath, slo_qoe_min: float = 0.0):
    """
    Plots 4 panels and annotates legend (QoE panel) with:
      - avg QoE
      - avg SLO violation rate (QoE < slo_qoe_min)
    """
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)

    panels = [
        ("qoe", "QoE"),
        ("local_gt_num_objects", "Local GT #Objects"),
        ("attack_in_rate", "Attack in rate"),
        ("cpu_to_ids_ratio", "CPU→IDS Ratio"),
    ]

    for ax, (k, ylabel) in zip(axes, panels):
        for method, series in results.items():
            y = series.get(k, None)
            if y is None or y.size == 0:
                continue

            x = np.arange(len(y))

            # annotate only in QoE panel
            if k == "qoe":
                y_valid = y[np.isfinite(y)]
                avg_qoe = float(np.nanmean(y_valid)) if y_valid.size else 0.0
                vio = (y_valid < float(slo_qoe_min)).astype(np.float32)
                vio_rate = float(np.nanmean(vio)) if y_valid.size else 0.0
                label = f"{method} (avg={avg_qoe:.3f}, vio={vio_rate:.2%})"
            else:
                label = method

            ax.plot(x, y, label=label)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("decision step")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    
# ----------------------------
# Plot helper
# ----------------------------
def pad_to_max(arrs: List[np.ndarray]) -> np.ndarray:
    if not arrs:
        return np.zeros((0, 0), dtype=np.float32)
    L = max(a.shape[0] for a in arrs)
    out = np.full((len(arrs), L), np.nan, dtype=np.float32)
    for i, a in enumerate(arrs):
        out[i, : a.shape[0]] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="eval_out")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--decision_interval", type=int, default=500)
    ap.add_argument("--scale_step", type=float, default=0.5)
    ap.add_argument("--ids_cpu_min", type=float, default=0.5)
    ap.add_argument("--constant_ids_cpu", type=float, default=0.5)

    ap.add_argument("--rl_ckpt", type=str, default="checkpoints/ppo_simulation_0/ckpt_baseline_000800.pt")
    ap.add_argument("--rl_device", type=str, default="cuda")
    ap.add_argument("--rl_greedy", action="store_true")

    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg["run"]["seed"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    obs_keys = [
        "local_num_objects",
        "attack_drop_rate",
        "cpu_to_ids_ratio",
        "bw_utilization",
    ]
    obs_dim = len(obs_keys)

    rl_policy = None
    if args.rl_ckpt is not None:
        rl_policy = RLPolicy(
            ckpt_path=args.rl_ckpt,
            obs_dim=obs_dim,
            device=args.rl_device,
            greedy=args.rl_greedy,
        )

    methods = ["rl", "random", "constant", "reactive"]
    # methods = ["constant", "reactive"]
    results: Dict[str, Dict[str, np.ndarray]] = {
        m: {
            "qoe": np.array([], dtype=np.float32),
            "cpu_util": np.array([], dtype=np.float32),
            "local_num_objects": np.array([], dtype=np.float32),
            "local_gt_num_objects": np.array([], dtype=np.float32),
            "attack_in_rate": np.array([], dtype=np.float32),
            "cpu_to_ids_ratio": np.array([], dtype=np.float32),
        }
        for m in methods
    }

    for ep in range(args.episodes):
        ep_seed = base_seed + ep * 1000
        for m in methods:
            if m == "rl" and rl_policy is None:
                continue
            q = run_episode(
                cfg=cfg,
                cfg_path=args.cfg,
                method=m,
                decision_interval=args.decision_interval,
                obs_keys=obs_keys,
                scale_step=args.scale_step,
                ids_cpu_min=args.ids_cpu_min,
                constant_ids_cpu=args.constant_ids_cpu,
                seed=ep_seed,
                rl_policy=rl_policy,
            )

            for k in results[m].keys():
                results[m][k] = np.concatenate([results[m][k], q[k]])


    plot_ts_continuous(results, "qoe", outdir / "qoe_ts.png", "QoE")
    plot_ts_continuous(results, "cpu_util", outdir / "cpu_util_ts.png", "CPU Utilization")
    plot_ts_continuous(results, "local_num_objects", outdir / "local_num_objects_ts.png", "Local #Objects")
    plot_ts_continuous(results, "cpu_to_ids_ratio", outdir / "cpu_to_ids_ratio_ts.png", "CPU→IDS Ratio")


if __name__ == "__main__":
    main()