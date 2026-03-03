# evaluate.py
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from environment import build_env_base  # your project

# --- single-agent feature net used by your LSTM policy ---
from train import FeatureNet

import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType
from torchrl.modules import LSTMModule, MultiAgentMLP, ProbabilisticActor
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


def build_observation_from_history(env, decision_interval: int, obs_keys: List[str]) -> np.ndarray:
    """
    Build per-edge observation from env.history.
    Returns: obs_mat shape [E, obs_dim]
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
            if k == "I_net":
                obs[i, j] = float(np.sum(vals))
            elif k == "cpu_to_ids_ratio":
                obs[i, j] = float(vals[-1])
            else:
                obs[i, j] = float(np.mean(vals))
    return obs


def decision_qoe_per_edge(env, decision_interval: int) -> np.ndarray:
    """
    Returns per-edge mean QoE over the last decision window.
    Shape: [E]
    """
    n_edges = len(env.edge_areas)
    if len(env.history) < decision_interval * n_edges:
        return np.zeros((n_edges,), dtype=np.float32)

    block = env.history[-decision_interval * n_edges :]
    df = pd.DataFrame([m.__dict__ for m in block])

    out = np.zeros((n_edges,), dtype=np.float32)
    area_ids = [e.area_id for e in env.edge_areas]
    for i, aid in enumerate(area_ids):
        g = df[df["area_id"] == aid]
        out[i] = float(g["qoe_mean"].mean()) if ("qoe_mean" in g.columns and not g.empty) else 0.0
    return out


def decision_ids_util_per_edge(env, decision_interval: int) -> np.ndarray:
    """
    Returns per-edge mean IDS CPU utilization over the last decision window.
    Shape: [E], values in [0,1]
    """
    n_edges = len(env.edge_areas)
    if len(env.history) < decision_interval * n_edges:
        return np.zeros((n_edges,), dtype=np.float32)

    block = env.history[-decision_interval * n_edges :]
    df = pd.DataFrame([m.__dict__ for m in block])

    out = np.zeros((n_edges,), dtype=np.float32)
    area_ids = [e.area_id for e in env.edge_areas]
    for i, aid in enumerate(area_ids):
        g = df[df["area_id"] == aid]
        if g.empty or "ids_cpu_utilization" not in g.columns:
            out[i] = 0.0
        else:
            out[i] = float(np.clip(np.mean(g["ids_cpu_utilization"].values), 0.0, 1.0))
    return out


def decision_metrics_per_edge(env, decision_interval: int) -> Dict[str, np.ndarray]:
    """
    Returns per-edge window-averaged metrics as vectors [E].
    Keys: local_num_req, attack_in_rate, cpu_to_ids_ratio, ids_cpu_utilization
    """
    n_edges = len(env.edge_areas)
    out = {
        "local_num_req": np.zeros((n_edges,), dtype=np.float32),
        "attack_in_rate": np.zeros((n_edges,), dtype=np.float32),
        "cpu_to_ids_ratio": np.zeros((n_edges,), dtype=np.float32),
        "ids_cpu_utilization": np.zeros((n_edges,), dtype=np.float32),
    }

    if len(env.history) < decision_interval * n_edges:
        return out

    block = env.history[-decision_interval * n_edges :]
    df = pd.DataFrame([m.__dict__ for m in block])
    area_ids = [e.area_id for e in env.edge_areas]

    for i, aid in enumerate(area_ids):
        g = df[df["area_id"] == aid]
        if g.empty:
            continue
        if "local_num_req" in g.columns:
            out["local_num_req"][i] = float(np.mean(g["local_num_req"].values))
        if "attack_in_rate" in g.columns:
            out["attack_in_rate"][i] = float(np.mean(g["attack_in_rate"].values))
        if "cpu_to_ids_ratio" in g.columns:
            out["cpu_to_ids_ratio"][i] = float(np.mean(g["cpu_to_ids_ratio"].values))
        if "ids_cpu_utilization" in g.columns:
            out["ids_cpu_utilization"][i] = float(np.clip(np.mean(g["ids_cpu_utilization"].values), 0.0, 1.0))

    return out


# =========================================================
# Policy interfaces
# =========================================================

class MultiEdgePolicy(Protocol):
    def reset_episode(self) -> None: ...
    def act(self, obs_mat: np.ndarray) -> np.ndarray: ...


# =========================================================
# RL policies
# =========================================================

class CollaborativeMAPPOPolicy:
    """
    Collaborative multi-agent (shared params):
      input: obs_mat [E, obs_dim]
      output: delta [E] in {-1,0,+1}

    Matches MAPPO-style MultiAgentMLP actor checkpoints.
    """

    def __init__(self, ckpt_path: str, n_edges: int, obs_dim: int, device: str = "cpu", greedy: bool = True):
        self.device = torch.device(device)
        self.greedy = greedy
        self.n_edges = int(n_edges)
        self.obs_dim = int(obs_dim)

        state = torch.load(ckpt_path, map_location=self.device)
        train_cfg = state.get("train_cfg", {})
        model_cfg = train_cfg.get("model", {})

        self.n_actions = int(model_cfg.get("n_actions", 3))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))

        actor_net = MultiAgentMLP(
            n_agent_inputs=self.obs_dim,
            n_agent_outputs=self.n_actions,
            n_agents=self.n_edges,
            centralised=False,
            share_params=True,
            device=self.device,
            depth=3,
            num_cells=hidden_dim,
            activation_class=torch.nn.Tanh,
        )

        actor = TensorDictModule(
            actor_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )

        self.policy = ProbabilisticActor(
            module=actor,
            in_keys=[("agents", "logits")],
            out_keys=[("agents", "action")],
            distribution_class=Categorical,
            return_log_prob=False,
            default_interaction_type=InteractionType.MODE if greedy else InteractionType.RANDOM,
        ).to(self.device)

        self.policy.load_state_dict(state["policy"], strict=True)
        self.policy.eval()

        # Optional: VecNorm / normalization stats
        self.env_state = state.get("env_state", None) or state.get("obsnorm", None)
        self.obs_loc = None
        self.obs_scale = None
        if isinstance(self.env_state, dict):
            loc_key = next((k for k in self.env_state.keys() if str(k).endswith("loc")), None)
            scale_key = next((k for k in self.env_state.keys() if str(k).endswith("scale")), None)
            if loc_key and scale_key:
                self.obs_loc = self.env_state[loc_key].detach().to(self.device).reshape(-1)
                self.obs_scale = self.env_state[scale_key].detach().to(self.device).reshape(-1)

    def reset_episode(self) -> None:
        return

    def _normalize_obs_mat(self, obs_mat: np.ndarray) -> np.ndarray:
        if self.obs_loc is None or self.obs_scale is None:
            return obs_mat
        flat = obs_mat.reshape(-1).astype(np.float32)
        loc = self.obs_loc.detach().cpu().numpy()
        scale = self.obs_scale.detach().cpu().numpy()
        if loc.shape[0] != flat.shape[0]:
            return obs_mat
        flat_n = (flat - loc) / (scale + 1e-8)
        return flat_n.reshape(obs_mat.shape)

    @torch.no_grad()
    def act(self, obs_mat: np.ndarray) -> np.ndarray:
        obs_mat = self._normalize_obs_mat(obs_mat.astype(np.float32))
        obs = torch.tensor(obs_mat, device=self.device, dtype=torch.float32).unsqueeze(0)  # [1,E,obs_dim]
        td = TensorDict({("agents", "observation"): obs}, batch_size=[1], device=self.device)
        td_out = self.policy(td)
        a = td_out.get(("agents", "action")).squeeze(0)  # [E], 0..2
        return (a.to(torch.int64) - 1).detach().cpu().numpy().astype(np.int64)


class SingleEdgeLSTMPolicy:
    """
    Single-edge policy used by IndependentMultiAgentPolicy.
    Sees only one edge obs vector (obs_dim,) and outputs one delta in {-1,0,+1}.
    """

    def __init__(self, ckpt_path: str, obs_dim: int, device: str = "cpu", greedy: bool = True):
        self.device = torch.device(device)
        self.greedy = greedy

        state = torch.load(ckpt_path, map_location=self.device)
        train_cfg = state.get("train_cfg", {})
        feature_dim = int(train_cfg["model"]["hidden_dim"])
        n_actions = int(train_cfg["model"]["n_actions"])

        self.h_size = feature_dim
        self.n_layers = 1

        # feature extractor expects "observation_flat"
        feature_module = TensorDictModule(
            FeatureNet(obs_dim, feature_dim).to(self.device),
            in_keys=["observation_flat"],
            out_keys=["features"],
        )

        self.lstm = LSTMModule(
            input_size=feature_dim,
            hidden_size=feature_dim,
            in_key="features",
            out_key="features",
            device=self.device,
        )

        actor_head = TensorDictModule(
            nn.Linear(feature_dim, n_actions).to(self.device),
            in_keys=["features"],
            out_keys=["logits"],
        )

        # IMPORTANT: match training nesting
        self.shared_core = TensorDictSequential(feature_module, self.lstm).to(self.device)
        self.actor = TensorDictSequential(self.shared_core, actor_head).to(self.device)

        self.policy = ProbabilisticActor(
            module=self.actor,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=False,
        ).to(self.device)

        # ---- load weights ----
        self.policy.load_state_dict(state["policy"], strict=True)
        self.policy.eval()

        self.env_state = state.get("env_state", None) or state.get("obsnorm", None)
        self.obs_loc = None
        self.obs_scale = None
        if isinstance(self.env_state, dict):
            loc_key = next((k for k in self.env_state.keys() if str(k).endswith("loc")), None)
            scale_key = next((k for k in self.env_state.keys() if str(k).endswith("scale")), None)
            if loc_key and scale_key:
                self.obs_loc = self.env_state[loc_key].detach().to(self.device).reshape(-1)
                self.obs_scale = self.env_state[scale_key].detach().to(self.device).reshape(-1)

        self._h = None
        self._c = None

    def reset(self) -> None:
        self._h = None
        self._c = None

    def _normalize(self, obs_vec: np.ndarray) -> np.ndarray:
        if self.obs_loc is None or self.obs_scale is None:
            return obs_vec
        loc = self.obs_loc.detach().cpu().numpy()
        scale = self.obs_scale.detach().cpu().numpy()
        if loc.shape[0] != obs_vec.size:
            return obs_vec
        return (obs_vec - loc) / (scale + 1e-8)

    @torch.no_grad()
    def act(self, obs_vec: np.ndarray) -> int:
        obs_vec = self._normalize(obs_vec.astype(np.float32))
        td = TensorDict(
            {"observation_flat": torch.tensor(obs_vec, device=self.device).unsqueeze(0)},
            batch_size=[1],
            device=self.device,
        )

        if self._h is None:
            hs = self.h_size
            self._h = torch.zeros(self.n_layers, 1, hs, device=self.device)
            self._c = torch.zeros(self.n_layers, 1, hs, device=self.device)

        td.set("recurrent_state_h", self._h)
        td.set("recurrent_state_c", self._c)
        td.set("is_init", torch.zeros(1, 1, device=self.device, dtype=torch.bool))

        td = self.actor(td)
        logits = td.get("logits").squeeze(0)

        self._h = td.get("recurrent_state_h")
        self._c = td.get("recurrent_state_c")

        if self.greedy:
            a = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            a = int(torch.multinomial(probs, 1).item())

        return int(a) - 1


class IndependentMultiAgentPolicy:
    """
    Independent multi-agent: one identical single-edge policy per edge,
    each agent observes only its own edge features.
    """

    def __init__(self, ckpt_path: str, n_edges: int, obs_dim: int, device: str = "cpu", greedy: bool = True):
        self.n_edges = int(n_edges)
        self.obs_dim = int(obs_dim)
        self.agents = [SingleEdgeLSTMPolicy(ckpt_path, obs_dim, device=device, greedy=greedy) for _ in range(self.n_edges)]

    def reset_episode(self) -> None:
        for a in self.agents:
            a.reset()

    def act(self, obs_mat: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n_edges,), dtype=np.int64)
        for i in range(self.n_edges):
            out[i] = self.agents[i].act(obs_mat[i])  # ONLY local obs
        return out



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
    seed: int,
    rl_policy: Optional[MultiEdgePolicy],
) -> Dict[str, np.ndarray]:
    env = build_env_base(cfg_path)
    env.reset(seed)

    # stabilize initial ratio
    for i, edge in enumerate(env.edge_areas):
        edge.reset(seed=seed + 100 * i)

    rng = np.random.default_rng(seed)

    t_max = int(cfg["run"]["t_max"])
    n_edges = len(env.edge_areas)
    ids_cpu_max = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)

    ids_cpu = np.array([e.ids_cpu / e.budget.cpu for e in env.edge_areas], dtype=np.float32)
    ids_cpu = np.clip(ids_cpu, ids_cpu_min, ids_cpu_max)

    decisions = math.ceil(t_max / decision_interval)

    qoe_ts: List[np.ndarray] = []
    ids_util_ts: List[np.ndarray] = []
    local_num_req_ts: List[np.ndarray] = []
    attack_in_rate_ts: List[np.ndarray] = []
    cpu_to_ids_ratio_ts: List[np.ndarray] = []

    if method == "rl":
        if rl_policy is None:
            raise ValueError("rl_policy is None but method == 'rl'")
        rl_policy.reset_episode()
    obs_ts: List[np.ndarray] = []
    for _k in range(decisions):
        if env.t >= env.t_max:
            break

        obs_mat = build_observation_from_history(env, decision_interval, obs_keys)  # [E, obs_dim]
        obs_ts.append(obs_mat.copy())
        ids_util = decision_ids_util_per_edge(env, decision_interval)              # [E]

        if method.startswith("constant_"):
            constant_cpu = float(method.split("_", 1)[1])
            ids_cpu = np.clip(np.full(n_edges, constant_cpu, dtype=np.float32), ids_cpu_min, ids_cpu_max)
            delta = np.zeros(n_edges, dtype=np.int64)

        elif method == "random":
            delta = rng.integers(-1, 2, size=n_edges, dtype=np.int64)

        elif method == "reactive":
            # per-edge reactive based on per-edge IDS utilization
            delta = np.zeros(n_edges, dtype=np.int64)
            delta[ids_util >= 0.80] = 1
            delta[ids_util <= 0.20] = -1

        elif method == "rl":
            delta = rl_policy.act(obs_mat)

        else:
            raise ValueError(method)

        ids_cpu = apply_delta(ids_cpu, delta, scale_step, ids_cpu_min, ids_cpu_max)

        for _ in range(decision_interval):
            env.step(ids_cpu)
            if env.t >= env.t_max:
                break

        qoe = decision_qoe_per_edge(env, decision_interval)  # [E]
        met = decision_metrics_per_edge(env, decision_interval)

        qoe_ts.append(qoe)
        ids_util_ts.append(met["ids_cpu_utilization"])
        local_num_req_ts.append(met["local_num_req"])
        attack_in_rate_ts.append(met["attack_in_rate"])
        cpu_to_ids_ratio_ts.append(met["cpu_to_ids_ratio"])

    return {
        "qoe": np.asarray(qoe_ts, dtype=np.float32),                     # [K,E]
        "ids_util": np.asarray(ids_util_ts, dtype=np.float32),           # [K,E]
        "local_num_req": np.asarray(local_num_req_ts, dtype=np.float32), # [K,E]
        "attack_in_rate": np.asarray(attack_in_rate_ts, dtype=np.float32),# [K,E]
        "cpu_to_ids_ratio": np.asarray(cpu_to_ids_ratio_ts, dtype=np.float32),# [K,E]
        "obs": np.asarray(obs_ts, dtype=np.float32),  # [K, E, obs_dim]
    }


# =========================================================
# Plotting (per edge, per folder)
# =========================================================
def plot_ts_per_edge(
    results: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
    edge_names: List[str],
    slo_qoe_min: float = 0.2,
):
    panels = [
        ("qoe", "QoE"),
        ("ids_util", "IDS CPU Util"),
        ("local_num_req", "Local #Req"),
        ("attack_in_rate", "Attack in rate"),
        ("cpu_to_ids_ratio", "CPU→IDS Ratio"),
        ("ema_mom", "EMA Momentum"),
    ]

    for e_idx, _e_name in enumerate(edge_names):
        edge_dir = outdir / f"edge_{e_idx}"
        edge_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 10), sharex=True)

        for ax, (k, ylabel) in zip(axes, panels):
            for method, series in results.items():
                y = series.get(k, None)
                if y is None or y.size == 0:
                    continue

                # expected [T,E]
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
    """
    Creates a separate file alongside ts.png:
      outdir/edge_<idx>/obs.png

    Expects:
      results[method]["obs"] shape [T, E, obs_dim]
    """
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

                y = obs[:, e_idx, j]  # [T]
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

def plot_qoe_vio_bars_per_edge(results: Dict[str, Dict[str, np.ndarray]], outdir: Path, edge_names: List[str], qoe_slo_min: float = 0.2, beta: float = 3.0):
    for e_idx, e_name in enumerate(edge_names):
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
    ap.add_argument("--rl_device", type=str, default="cuda")
    ap.add_argument("--rl_greedy", action="store_true")

    # optional: evaluate a subset via CLI
    ap.add_argument(
        "--rl_specs",
        type=str,
        default="",
        help=(
            "Comma-separated specs: <name>:<mode>:<ckpt>. "
            "Example: mappo:collab:checkpoints/multi_edge/ckpt_best.pt,"
            "indep:indep:checkpoints/ppo_simulation_0/ckpt_epoch20.pt"
        ),
    )

    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg["run"]["seed"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    obs_keys = [
        "local_num_req",
        "attack_in_rate",
        "ema_mom",
        "cpu_to_ids_ratio",
        "ids_cpu_utilization",
        # "overhead",
    ]
    obs_dim = len(obs_keys)

    tmp_env = build_env_base(args.cfg)
    n_edges = len(tmp_env.edge_areas)
    edge_names = [str(e.area_id) for e in tmp_env.edge_areas]

    # -----------------------------------------------------
    # Define RL candidates (or override with --rl_specs)
    # -----------------------------------------------------
    rl_candidates = [
        ("mappo_best", "collab", "checkpoints/ma_qoe_weighted/ckpt_iter_000100.pt"),
        # ("indep_epoch20", "indep", "checkpoints/atk1_noSO_2048_ev4_e10_t030_055/ckpt_iter_001000.pt"),
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
            if mode not in ("collab", "indep"):
                raise ValueError(f"Bad mode '{mode}' for {name}, expected collab|indep")
            rl_candidates.append((name, mode, ckpt))

    # -----------------------------------------------------
    # Methods: baselines + all RL candidates
    # -----------------------------------------------------
    baseline_methods = ["random", "constant_0.5", "reactive"]
    # baseline_methods = ["reactive"]
    # baseline_methods = []

    rl_method_keys = [f"rl_{name}" for (name, _mode, _ckpt) in rl_candidates]
    all_methods = baseline_methods + rl_method_keys

    def init_results(methods: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            m: {
                "qoe": np.array([], dtype=np.float32),
                "ids_util": np.array([], dtype=np.float32),
                "local_num_req": np.array([], dtype=np.float32),
                "attack_in_rate": np.array([], dtype=np.float32),
                "cpu_to_ids_ratio": np.array([], dtype=np.float32),
                "obs": np.array([], dtype=np.float32),
            }
            for m in methods
        }

    results_all = init_results(all_methods)

    # -----------------------------------------------------
    # 1) Evaluate baselines once
    # -----------------------------------------------------
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
                seed=ep_seed,
                rl_policy=None,
            )
            for k, v in q.items():
                results_all[m][k] = np.concatenate([results_all[m][k], v], axis=0) if results_all[m][k].size else v

    # -----------------------------------------------------
    # 2) Evaluate each RL candidate, append into its own method key
    # -----------------------------------------------------
    for name, mode, ckpt in rl_candidates:
        if mode == "collab":
            rl_policy = CollaborativeMAPPOPolicy(
                ckpt_path=ckpt,
                n_edges=n_edges,
                obs_dim=obs_dim,
                device=args.rl_device,
                greedy=args.rl_greedy,
            )
        elif mode == "indep":
            rl_policy = IndependentMultiAgentPolicy(
                ckpt_path=ckpt,
                n_edges=n_edges,
                obs_dim=obs_dim,
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
                seed=ep_seed,
                rl_policy=rl_policy,
            )
            for k, v in q.items():
                results_all[mkey][k] = np.concatenate([results_all[mkey][k], v], axis=0) if results_all[mkey][k].size else v

    # -----------------------------------------------------
    # 3) Plot ONCE: all methods in same plot, separated per edge folder
    # -----------------------------------------------------
    out_all = outdir / "all"
    out_all.mkdir(parents=True, exist_ok=True)

    plot_ts_per_edge(results_all, out_all, edge_names=edge_names, slo_qoe_min=0.2)
    plot_obs_per_edge(results_all, out_all, edge_names=edge_names, obs_keys=obs_keys)
    plot_qoe_vio_bars_per_edge(results_all, out_all, edge_names=edge_names, qoe_slo_min=0.2)


if __name__ == "__main__":
    main()