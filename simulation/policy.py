from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from environment import build_env_base

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType
from torchrl.modules import ProbabilisticActor
from torch.distributions import Categorical


# =========================================================
# Utilities
# =========================================================

def apply_delta(
    ids_cpu: np.ndarray,
    delta: np.ndarray,
    scale_step: float,
    ids_cpu_min: float,
    ids_cpu_max: np.ndarray,
) -> np.ndarray:
    out = ids_cpu + delta.astype(np.float32) * float(scale_step)
    out = np.maximum(out, ids_cpu_min)
    out = np.minimum(out, ids_cpu_max)
    return out


def add_agent_id(obs_mat: np.ndarray) -> np.ndarray:
    """
    obs_mat: [E, D]
    returns: [E, D+E]
    """
    e = obs_mat.shape[0]
    eye = np.eye(e, dtype=np.float32)
    return np.concatenate([obs_mat.astype(np.float32), eye], axis=-1)


def build_observation_from_history(env, decision_interval: int, obs_keys: List[str]) -> np.ndarray:
    """
    Match training env._build_observation().
    Returns: [E, obs_dim]
    """
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
            if k not in g.columns:
                continue
            vals = g[k].values
            if k == "ema_mom":
                vals_nz = vals[vals != 0.0]
                obs[i, j] = float(np.mean(vals_nz)) if len(vals_nz) else 0.0
            else:
                obs[i, j] = float(np.mean(vals))
    return obs


def decision_qoe_per_edge(env, threshold: float = 0.35, alpha: float = 0.6) -> np.ndarray:
    """
    Match current training env._qoe_vec().
    Uses whole-episode history up to current time.
    Returns: [E]
    """
    n_edges = len(env.edge_areas)

    if not getattr(env, "history", None) or len(env.history) == 0:
        return np.zeros((n_edges,), dtype=np.float32)

    q_vec = np.zeros((n_edges,), dtype=np.float32)

    for i, edge in enumerate(env.edge_areas):
        h = [m for m in env.history if m.area_id == edge.area_id]
        if not h:
            q_vec[i] = 0.0
            continue

        q = np.asarray([float(m.qoe_mean) for m in h], dtype=np.float32)
        if q.size == 0:
            q_vec[i] = 0.0
            continue

        slo_thr = float(getattr(edge, "slo_threshold", threshold))
        slo_beta = float(getattr(edge, "slo_beta", alpha))

        viol_rate = float((q < slo_thr).mean())
        v_edge = float(np.exp(-slo_beta * viol_rate))

        q_vec[i] = float(q.mean()) * v_edge

    return q_vec


def decision_reward_per_edge(env, threshold: float = 0.35, alpha: float = 0.6) -> np.ndarray:
    """
    Match current training env._build_reward_per_agent() on the most recent block.
    Returns: [E]
    """
    n_edges = len(env.edge_areas)
    if len(env.history) < n_edges:
        return np.zeros((n_edges,), dtype=np.float32)

    last_block = env.history[-n_edges:]
    q_local = np.asarray([float(m.qoe_mean) for m in last_block], dtype=np.float32)
    viol = np.maximum(0.0, threshold - q_local)
    penalty = (alpha * (viol ** 2)).astype(np.float32)
    return q_local - penalty


def decision_metrics_per_edge(env, decision_interval: int, metric_keys: List[str]) -> Dict[str, np.ndarray]:
    """
    Returns per-edge window-averaged metrics [E].
    """
    n_edges = len(env.edge_areas)
    out = {k: np.zeros((n_edges,), dtype=np.float32) for k in metric_keys}

    if len(env.history) < decision_interval * n_edges:
        return out

    block = env.history[-decision_interval * n_edges :]
    df = pd.DataFrame([m.__dict__ for m in block])
    area_ids = [e.area_id for e in env.edge_areas]

    for i, aid in enumerate(area_ids):
        g = df[df["area_id"] == aid]
        if g.empty:
            continue
        for k in metric_keys:
            if k not in g.columns:
                continue
            vals = g[k].values
            if k == "ema_mom":
                vals_nz = vals[vals != 0.0]
                out[k][i] = float(np.mean(vals_nz)) if len(vals_nz) else 0.0
            else:
                out[k][i] = float(np.mean(vals))
    return out


def _find_obsnorm_loc_scale_from_env_state(obsnorm_state: dict, obs_dim: int, device: torch.device):
    if not isinstance(obsnorm_state, dict):
        return None, None

    target_numel = obs_dim
    loc = None
    scale = None

    for k, v in obsnorm_state.items():
        if not torch.is_tensor(v):
            continue

        name = str(k).lower()
        if v.numel() != target_numel:
            continue

        if loc is None and "loc" in name:
            loc = v.detach().to(device).reshape(obs_dim)
        if scale is None and ("scale" in name or "std" in name):
            scale = v.detach().to(device).reshape(obs_dim)

    return loc, scale

def debug_obsnorm_state(obsnorm_state):
    print("=== obsnorm keys ===")
    for k, v in obsnorm_state.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, numel={v.numel()}")
        else:
            print(f"{k}: {type(v)}")

# =========================================================
# Policy interfaces
# =========================================================

class MultiEdgePolicy(Protocol):
    def reset_episode(self) -> None: ...
    def act(self, obs_mat: np.ndarray) -> np.ndarray: ...


# =========================================================
# Networks matching current training code
# =========================================================

class AgentMLPCore(nn.Module):
    in_keys = [("agents", "observation", "obs")]
    out_keys = [("agents", "features")]

    def __init__(self, n_edges: int, obs_dim: int, hidden_dim: int, device: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        ).to(device)

    def forward(self, td):
        obs = td.get(("agents", "observation", "obs"))
        td.set(("agents", "features"), self.net(obs))
        return td


# =========================================================
# RL policy compatible with current training code
# =========================================================

class MultiAgentMLPPolicy:
    """
    Compatible with current training code:
    - ObservationNorm on ('agents','observation','obs') before AddAgentID
    - AddAgentID appends one-hot IDs
    - Actor = AgentMLPCore + Linear head
    """

    def __init__(self, ckpt_path: str, n_edges: int, obs_dim_no_id: int, device: str = "cpu", greedy: bool = True):
        self.device = torch.device(device)
        self.greedy = greedy
        self.n_edges = int(n_edges)
        self.obs_dim_no_id = int(obs_dim_no_id)
        self.obs_dim_with_id = self.obs_dim_no_id + self.n_edges

        state = torch.load(ckpt_path, map_location=self.device)
        train_cfg = state.get("train_cfg", {})
        model_cfg = train_cfg.get("model", {})

        self.n_actions = int(model_cfg.get("n_actions", 3))
        self.hidden_dim = int(model_cfg.get("hidden_dim", 64))

        actor_core = AgentMLPCore(
            n_edges=self.n_edges,
            obs_dim=self.obs_dim_with_id,
            hidden_dim=self.hidden_dim,
            device=str(self.device),
        )
        actor_head = TensorDictModule(
            nn.Linear(self.hidden_dim, self.n_actions).to(self.device),
            in_keys=[("agents", "features")],
            out_keys=[("agents", "logits")],
        )

        self.policy = ProbabilisticActor(
            module=TensorDictSequential(actor_core, actor_head),
            in_keys=[("agents", "logits")],
            out_keys=[("agents", "action")],
            distribution_class=Categorical,
            return_log_prob=False,
            default_interaction_type=InteractionType.MODE if greedy else InteractionType.RANDOM,
        ).to(self.device)

        self.policy.load_state_dict(state["policy"], strict=True)
        self.policy.eval()

        obsnorm_state = state.get("obsnorm", None)
        debug_obsnorm_state(state["obsnorm"])            
        self.obs_loc, self.obs_scale = _find_obsnorm_loc_scale_from_env_state(
            obsnorm_state=obsnorm_state,
            obs_dim=self.obs_dim_no_id,
            device=self.device,
        )

        if self.obs_loc is None or self.obs_scale is None:
            print("Warning: could not recover ObservationNorm loc/scale from checkpoint. Using raw observations.")

    def reset_episode(self) -> None:
        return

    def _normalize_obs(self, obs_mat: np.ndarray) -> np.ndarray:
        """
        Normalize BEFORE agent ID append, matching training transform order.
        obs_mat: [E, D]
        """
        if self.obs_loc is None or self.obs_scale is None:
            return obs_mat.astype(np.float32)

        # Keep as 1D arrays of size D
        loc = self.obs_loc.detach().cpu().numpy()
        scale = self.obs_scale.detach().cpu().numpy()

        # Numpy automatically broadcasts (E, D) with (D,)
        obs_n = (obs_mat - loc) / (scale + 1e-8)
        return obs_n.astype(np.float32)

    @torch.no_grad()
    def act(self, obs_mat: np.ndarray) -> np.ndarray:
        obs_mat = self._normalize_obs(obs_mat)
        obs_with_id = add_agent_id(obs_mat)

        obs = torch.tensor(obs_with_id, device=self.device, dtype=torch.float32).unsqueeze(0)
        td = TensorDict(
            {("agents", "observation", "obs"): obs},
            batch_size=[1],
            device=self.device,
        )

        # Let ProbabilisticActor do all the work!
        # This runs the network AND automatically samples/takes the mode
        self.policy(td)

        # Extract the automatically generated action
        a = td.get(("agents", "action")).squeeze(0)
        
        # You can still inspect the logits since the network populated them
        logits = td.get(("agents", "logits")).squeeze(0)   # [E, A]
        print("logits =", logits)
        print("action =", a)

        return (a.to(torch.int64) - 1).detach().cpu().numpy().astype(np.int64)
# =========================================================
# Episode runner
# =========================================================

def run_episode(
    cfg: dict,
    cfg_path: str,
    method: str,
    decision_interval: int,
    obs_keys: List[str],
    scale_step: float,
    ids_cpu_min: float,
    threshold: float,
    alpha: float,
    seed: int,
    rl_policy: Optional[MultiEdgePolicy],
) -> Dict[str, np.ndarray]:
    env = build_env_base(cfg_path)
    env.reset(seed)

    for i, edge in enumerate(env.edge_areas):
        edge.reset(seed=seed + 100 * i)
        edge.ids_cpu = 0.5

    rng = np.random.default_rng(seed)

    t_max = int(cfg["run"]["t_max"])
    n_edges = len(env.edge_areas)
    ids_cpu_max = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)

    ids_cpu = np.array([e.ids_cpu for e in env.edge_areas], dtype=np.float32)
    ids_cpu = np.clip(ids_cpu, ids_cpu_min, ids_cpu_max)

    decisions = math.ceil(t_max / decision_interval)

    qoe_ts: List[np.ndarray] = []
    reward_ts: List[np.ndarray] = []
    obs_ts: List[np.ndarray] = []

    metric_keys = [
        "local_num_req",
        "attack_in_rate",
        "cpu_to_ids_ratio",
        "ids_cpu_utilization",
        "total_cpu_to_ids_ratio",
        "ema_mom",
    ]
    metric_ts = {k: [] for k in metric_keys}

    if method == "rl":
        if rl_policy is None:
            raise ValueError("rl_policy is None but method == 'rl'")
        rl_policy.reset_episode()

    for _ in range(decisions):
        if env.t >= env.t_max:
            break

        obs_mat = build_observation_from_history(env, decision_interval, obs_keys)  # [E,D]
        obs_ts.append(obs_mat.copy())

        ids_util = decision_metrics_per_edge(env, decision_interval, ["ids_cpu_utilization"])["ids_cpu_utilization"]

        if method.startswith("constant_"):
            constant_cpu = float(method.split("_", 1)[1])
            ids_cpu = np.clip(np.full(n_edges, constant_cpu, dtype=np.float32), ids_cpu_min, ids_cpu_max)
            delta = np.zeros(n_edges, dtype=np.int64)

        elif method == "random":
            delta = rng.integers(-1, 2, size=n_edges, dtype=np.int64)

        elif method == "reactive":
            delta = np.zeros(n_edges, dtype=np.int64)
            delta[ids_util >= 0.80] = 1
            delta[ids_util <= 0.20] = -1

        elif method == "rl":
            delta = rl_policy.act(obs_mat)
        else:
            raise ValueError(method)

        ids_cpu = apply_delta(ids_cpu, delta, scale_step, ids_cpu_min, ids_cpu_max)

        overheads = (ids_cpu - np.array([e.ids_cpu for e in env.edge_areas], dtype=np.float32)).astype(np.float32).tolist()

        for i, edge in enumerate(env.edge_areas):
            edge.ids_cpu = float(ids_cpu[i])

        for _ in range(decision_interval):
            env.step(ids_cpu, overheads)
            if env.t >= env.t_max:
                break

        qoe = decision_qoe_per_edge(env, threshold=threshold, alpha=alpha)
        rew = decision_reward_per_edge(env, threshold=threshold, alpha=alpha)
        met = decision_metrics_per_edge(env, decision_interval, metric_keys)

        qoe_ts.append(qoe)
        reward_ts.append(rew)
        for k in metric_keys:
            metric_ts[k].append(met[k])

    out = {
        "qoe": np.asarray(qoe_ts, dtype=np.float32),
        "reward": np.asarray(reward_ts, dtype=np.float32),
        "obs": np.asarray(obs_ts, dtype=np.float32),
    }
    for k in metric_keys:
        out[k] = np.asarray(metric_ts[k], dtype=np.float32)
    return out


# =========================================================
# Plotting
# =========================================================

def plot_ts_per_edge(
    results: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
    edge_names: List[str],
    slo_qoe_min: float = 0.2,
):
    panels = [
        ("qoe", "QoE"),
        ("reward", "Reward"),
        ("ids_cpu_utilization", "IDS CPU Util"),
        ("local_num_req", "Local #Req"),
        ("attack_in_rate", "Attack in rate"),
        ("cpu_to_ids_ratio", "CPU→IDS Ratio"),
        ("total_cpu_to_ids_ratio", "Total CPU→IDS Ratio"),
        ("ema_mom", "EMA Momentum"),
    ]

    for e_idx, _e_name in enumerate(edge_names):
        edge_dir = outdir / f"edge_{e_idx}"
        edge_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 14), sharex=True)

        for ax, (k, ylabel) in zip(axes, panels):
            for method, series in results.items():
                y = series.get(k, None)
                if y is None or y.size == 0:
                    continue

                y_edge = y[:, e_idx] if (isinstance(y, np.ndarray) and y.ndim == 2) else y
                x = np.arange(len(y_edge))

                if k == "qoe":
                    y_valid = y_edge[np.isfinite(y_edge)]
                    avg_qoe = float(np.nanmean(y_valid)) if y_valid.size else 0.0
                    vio_rate = float(np.nanmean((y_valid < float(slo_qoe_min)).astype(np.float32))) if y_valid.size else 0.0
                    label = f"{method} (avg={avg_qoe:.3f}, vio={vio_rate:.2%})"
                else:
                    label = method

                ax.plot(x, y_edge, label=label)

            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("decision step")
        axes[0].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(edge_dir / "ts.png", dpi=200)
        plt.close()


def plot_obs_per_edge(
    results: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
    edge_names: List[str],
    obs_keys: List[str],
):
    for e_idx, _e_name in enumerate(edge_names):
        edge_dir = outdir / f"edge_{e_idx}"
        edge_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(len(obs_keys), 1, figsize=(10, 2.2 * len(obs_keys)), sharex=True)
        if len(obs_keys) == 1:
            axes = [axes]

        for j, key in enumerate(obs_keys):
            ax = axes[j]
            for method, series in results.items():
                obs = series.get("obs", None)
                if obs is None or obs.size == 0:
                    continue
                if not (isinstance(obs, np.ndarray) and obs.ndim == 3):
                    raise ValueError(f'results["{method}"]["obs"] must be [T,E,obs_dim], got {getattr(obs, "shape", None)}')

                y = obs[:, e_idx, j]
                x = np.arange(len(y))
                ax.plot(x, y, label=method)

            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(loc="upper right")

        axes[-1].set_xlabel("decision step")
        plt.tight_layout()
        plt.savefig(edge_dir / "obs.png", dpi=200)
        plt.close()


def plot_qoe_vio_bars_per_edge(
    results: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
    edge_names: List[str],
    qoe_slo_min: float = 0.2,
    beta: float = 3.0,
):
    for e_idx, _e_name in enumerate(edge_names):
        edge_dir = outdir / f"edge_{e_idx}"
        edge_dir.mkdir(parents=True, exist_ok=True)

        methods, avg_qoe, vio_rate = [], [], []

        for method, series in results.items():
            qoe = series.get("qoe", None)
            if qoe is None or qoe.size == 0:
                continue
            q = np.asarray(qoe, dtype=np.float32)
            if q.ndim == 2:
                q = q[:, e_idx]
            q = q[np.isfinite(q)]
            if q.size == 0:
                continue

            vr = float(np.mean((q < qoe_slo_min).astype(np.float32)))
            v = float(np.exp(-beta * vr))
            methods.append(method)
            vio_rate.append(vr)
            avg_qoe.append(float(np.mean(q)) * v)

        x = np.arange(len(methods), dtype=np.int32)
        width = 0.7

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        bars_qoe = axes[0].bar(x, avg_qoe, width)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=20, ha="right")
        axes[0].set_ylabel("Average QoE")
        axes[0].set_title("Average QoE (SLO-adjusted)")
        axes[0].grid(axis="y", alpha=0.3)
        for bar in bars_qoe:
            h = float(bar.get_height())
            axes[0].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

        bars_vio = axes[1].bar(x, vio_rate, width)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods, rotation=20, ha="right")
        axes[1].set_ylabel("Violation Rate")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title(f"SLO Violations (QoE < {qoe_slo_min})")
        axes[1].grid(axis="y", alpha=0.3)
        for bar in bars_vio:
            h = float(bar.get_height())
            axes[1].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1%}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(edge_dir / "summary.png", dpi=200)
        plt.close()

def plot_global_weighted_summary(
    results: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
    qoe_slo_min: float = 0.2,
    beta: float = 3.0,
):
    """
    Plots a system-wide summary where QoE and Violations are weighted 
    by the number of users (local_num_req) in each area.
    """
    methods, weighted_avg_qoe, weighted_vio_rate = [], [], []

    for method, series in results.items():
        qoe = series.get("qoe") 
        users = series.get("local_num_req") 
        
        if qoe is None or users is None or qoe.size == 0:
            continue

        q_flat = qoe.flatten()
        u_flat = users.flatten()
        
        # Filter NaNs and ensure we don't divide by zero
        mask = np.isfinite(q_flat) & (u_flat > 0)
        q_clean = q_flat[mask]
        u_clean = u_flat[mask]
        
        if u_clean.sum() == 0:
            continue

        # Calculate weighted metrics
        is_violating = (q_clean < qoe_slo_min).astype(np.float32)
        v_rate = np.average(is_violating, weights=u_clean)
        
        raw_avg_qoe = np.average(q_clean, weights=u_clean)
        slo_penalty = np.exp(-beta * v_rate)
        final_qoe = raw_avg_qoe * slo_penalty

        methods.append(method)
        weighted_avg_qoe.append(final_qoe)
        weighted_vio_rate.append(v_rate)

    if not methods:
        return

    # --- Plotting ---
    x = np.arange(len(methods))
    width = 0.6
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # 1. Weighted QoE Plot
    bars0 = axes[0].bar(x, weighted_avg_qoe, width, color='skyblue', edgecolor='black')
    axes[0].set_title(f"Global Weighted QoE\n(Threshold={qoe_slo_min}, Beta={beta})", fontweight='bold')
    axes[0].set_ylabel("Weighted QoE Score")
    
    # Add labels to QoE bars
    for bar in bars0:
        val = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width()/2, val,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold'
        )

    # 2. Weighted Violation Rate Plot
    bars1 = axes[1].bar(x, weighted_vio_rate, width, color='salmon', edgecolor='black')
    axes[1].set_title("Global Weighted Violations\n(% of User-Requests below Threshold)", fontweight='bold')
    axes[1].set_ylabel("Weighted Violation Rate")
    axes[1].set_ylim(0, max(max(weighted_vio_rate) * 1.2, 0.2)) # Dynamic height with headroom

    # Add labels to Violation bars as percentages
    for bar in bars1:
        val = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width()/2, val,
            f'{val:.1%}', ha='center', va='bottom', fontweight='bold'
        )

    # General styling
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path = outdir / "global_weighted_summary.png"
    plt.savefig(save_path, dpi=300)
    print(f"Global summary saved to: {save_path}")
    plt.close()
# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="./configs/simulation_ma_0.yaml")
    ap.add_argument("--outdir", type=str, default="eval_out")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--decision_interval", type=int, default=500)
    ap.add_argument("--scale_step", type=float, default=0.5)
    ap.add_argument("--ids_cpu_min", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--rl_device", type=str, default="cuda")
    ap.add_argument("--rl_greedy", action="store_true")
    ap.add_argument(
        "--rl_specs",
        type=str,
        default="",
        help="Comma-separated specs: <name>:<mode>:<ckpt>. mode currently supports mlp",
    )

    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg["run"]["seed"])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    

    # Must match current training obs_keys exactly
    obs_keys = [
        "local_num_req",
        "attack_in_rate",
        "cpu_to_ids_ratio",
        "ids_cpu_utilization",
        "total_cpu_to_ids_ratio",
        "ema_mom",
    ]
    obs_dim = len(obs_keys)

    tmp_env = build_env_base(args.cfg)
    n_edges = len(tmp_env.edge_areas)
    edge_names = [str(e.area_id) for e in tmp_env.edge_areas]

    rl_candidates = [
        ("mlp_best", "mlp", "checkpoints/ima_ppo/ckpt_best.pt"),
    ]

    if args.rl_specs.strip():
        rl_candidates = []
        for item in args.rl_specs.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":", 2)
            if len(parts) != 3:
                raise ValueError(f"Bad --rl_specs entry: {item}")
            name, mode, ckpt = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if mode != "mlp":
                raise ValueError(f"Bad mode '{mode}' for {name}, expected mlp")
            rl_candidates.append((name, mode, ckpt))

    baseline_methods = ["random", "constant_0.0", 
                        "constant_2.0", "constant_3.0", "constant_4.0",
                        "reactive"]
    # baseline_methods = []
    rl_method_keys = [f"rl_{name}" for (name, _mode, _ckpt) in rl_candidates]
    all_methods = baseline_methods + rl_method_keys

    def init_results(methods: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            m: {
                "qoe": np.array([], dtype=np.float32),
                "reward": np.array([], dtype=np.float32),
                "local_num_req": np.array([], dtype=np.float32),
                "attack_in_rate": np.array([], dtype=np.float32),
                "cpu_to_ids_ratio": np.array([], dtype=np.float32),
                "ids_cpu_utilization": np.array([], dtype=np.float32),
                "total_cpu_to_ids_ratio": np.array([], dtype=np.float32),
                "ema_mom": np.array([], dtype=np.float32),
                "obs": np.array([], dtype=np.float32),
            }
            for m in methods
        }

    results_all = init_results(all_methods)

    for ep in tqdm(range(args.episodes), desc="Baselines"):
        ep_seed = base_seed + ep * 1000
        for m in baseline_methods:
            q = run_episode(
                cfg=cfg,
                cfg_path=args.cfg,
                method=m,
                decision_interval=args.decision_interval,
                obs_keys=obs_keys,
                scale_step=args.scale_step,
                ids_cpu_min=args.ids_cpu_min,
                threshold=args.threshold,
                alpha=args.alpha,
                seed=ep_seed,
                rl_policy=None,
            )
            for k, v in q.items():
                results_all[m][k] = np.concatenate([results_all[m][k], v], axis=0) if results_all[m][k].size else v

    for name, mode, ckpt in rl_candidates:
        if mode == "mlp":
            rl_policy = MultiAgentMLPPolicy(
                ckpt_path=ckpt,
                n_edges=n_edges,
                obs_dim_no_id=obs_dim,
                device=args.rl_device,
                greedy=args.rl_greedy,
            )
        else:
            raise ValueError(mode)

        mkey = f"rl_{name}"
        for ep in tqdm(range(args.episodes), desc=f"RL {name} ({mode})"):
            ep_seed = base_seed + ep * 1000
            q = run_episode(
                cfg=cfg,
                cfg_path=args.cfg,
                method="rl",
                decision_interval=args.decision_interval,
                obs_keys=obs_keys,
                scale_step=args.scale_step,
                ids_cpu_min=args.ids_cpu_min,
                threshold=args.threshold,
                alpha=args.alpha,
                seed=ep_seed,
                rl_policy=rl_policy,
            )
            for k, v in q.items():
                results_all[mkey][k] = np.concatenate([results_all[mkey][k], v], axis=0) if results_all[mkey][k].size else v

    out_all = outdir / "all"
    out_all.mkdir(parents=True, exist_ok=True)

    plot_ts_per_edge(results_all, out_all, edge_names=edge_names, slo_qoe_min=0.2)
    plot_obs_per_edge(results_all, out_all, edge_names=edge_names, obs_keys=obs_keys)
    plot_qoe_vio_bars_per_edge(results_all, out_all, edge_names=edge_names, qoe_slo_min=0.2)
    plot_global_weighted_summary(
            results_all, 
            out_all, 
            qoe_slo_min=0.2, # Using the threshold from args
            beta=args.alpha             # Using alpha as the SLO sensitivity
        )

if __name__ == "__main__":
    main()