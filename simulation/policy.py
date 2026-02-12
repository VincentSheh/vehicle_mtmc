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
import matplotlib.pyplot as plt

from environment import build_env_base  # your project
from train import ActorNet, FeatureNet
from tqdm import tqdm
from torchrl.modules import LSTMModule
from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
from tensordict import TensorDict


class RLPolicy:
    def __init__(self, ckpt_path: str, obs_size: int, device: str = "cpu", greedy: bool = True):
        self.device = torch.device(device)
        self.greedy = greedy

        state = torch.load(ckpt_path, map_location=self.device)
        self.obsnorm = state.get("obsnorm", None)
        # ---- rebuild training architecture ----
        feature_dim = state["train_cfg"]["model"]["hidden_dim"]        
        n_actions = state["train_cfg"]["model"]["n_actions"]
        self.h_size = feature_dim
        self.n_layers = 1 
        feature_module = TensorDictModule(
            FeatureNet(obs_size, feature_dim).to(self.device),
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

        self.shared_core = TensorDictSequential(feature_module, self.lstm).to(self.device)
        self.actor = TensorDictSequential(self.shared_core, actor_head).to(self.device)

        # match training policy wrapper only if you need log_prob etc
        self.policy = ProbabilisticActor(
            module=self.actor,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=False,
        ).to(self.device)

        # ---- load weights ----
        self.policy.load_state_dict(state["policy"])


        # keep recurrent state for inference
        self._h = None
        self._c = None
        self.policy.eval()
        
        # you saved env.state_dict() under "obsnorm" so it is NOT directly loc/scale
        self.env_state = state.get("obsnorm", None)
        
        self.obs_loc = None
        self.obs_scale = None
        if self.env_state is not None:
            # find loc/scale tensors inside env.state_dict()
            loc_key = next((k for k in self.env_state.keys() if k.endswith("loc")), None)
            scale_key = next((k for k in self.env_state.keys() if k.endswith("scale")), None)
            if loc_key and scale_key:
                self.obs_loc = self.env_state[loc_key].detach().to(self.device).reshape(-1)
                self.obs_scale = self.env_state[scale_key].detach().to(self.device).reshape(-1)        

    def normalize_obs_flat(self, obs_flat: np.ndarray) -> np.ndarray:
        if self.obs_loc is None or self.obs_scale is None:
            return obs_flat
        loc = self.obs_loc.detach().cpu().numpy()
        scale = self.obs_scale.detach().cpu().numpy()
        return (obs_flat - loc) / (scale + 1e-8)

    @torch.no_grad()
    def act_from_obs_flat(self, obs_flat: np.ndarray) -> int:
        td = TensorDict(
            {"observation_flat": torch.tensor(obs_flat, device=self.device).unsqueeze(0)},
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

        td = self.actor(td)                    # writes "logits"
        logits = td.get("logits").squeeze(0)   # [n_actions]

        # update recurrent state from td (LSTMModule writes back)
        self._h = td.get("recurrent_state_h")
        self._c = td.get("recurrent_state_c")

        if self.greedy:
            a = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            a = int(torch.multinomial(probs, 1).item())
        delta = int(a) - 1
        return np.array([delta], dtype=np.int64)


    
    def reset(self):
        self._h = None
        self._c = None    

def build_observation_from_history(env, decision_interval: int, obs_keys: List[str]) -> np.ndarray:
    """
    Returns obs shaped (n_edges, obs_dim)
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


def decision_qoe_mean(env, decision_interval: int) -> float:
    n_edges = len(env.edge_areas)
    if len(env.history) < decision_interval * n_edges:
        return 0.0
    block = env.history[-decision_interval * n_edges :]
    return float(np.mean([m.qoe_mean for m in block]))


def decision_cpu_util(env, decision_interval: int) -> float:
    n_edges = len(env.edge_areas)
    if len(env.history) < decision_interval * n_edges:
        return 0.0

    block = env.history[-decision_interval * n_edges :]
    df = pd.DataFrame([m.__dict__ for m in block])

    utils = []
    for area_id in [e.area_id for e in env.edge_areas]:
        g = df[df["area_id"] == area_id]
        if g.empty:
            continue
        if "ids_cpu_utilization" not in g.columns:
            continue
        ids_util = float(np.mean(g["ids_cpu_utilization"].values))
        utils.append(float(np.clip(ids_util, 0.0, 1.0)))

    return float(max(utils)) if utils else 0.0


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
    seed: int,
    rl_policy: Optional[RLPolicy],
) -> Dict[str, np.ndarray]:
    env = build_env_base(cfg_path)

    # your Environment.reset(seed) signature
    env.reset(seed)

    # stabilize initial ratio
    for i, edge in enumerate(env.edge_areas):
        edge.cpu_to_ids_ratio = 0.5
        edge.reset(seed=seed + 100 * i)

    rng = np.random.default_rng(seed)

    t_max = int(cfg["run"]["t_max"])
    n_edges = len(env.edge_areas)
    ids_cpu_max = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)

    ids_cpu = np.array([e.cpu_to_ids_ratio * e.budget.cpu for e in env.edge_areas], dtype=np.float32)
    ids_cpu = np.clip(ids_cpu, ids_cpu_min, ids_cpu_max)

    decisions = math.ceil(t_max / decision_interval)

    qoe_ts = []
    cpu_util_ts = []
    local_num_req_ts = []
    attack_in_rate_ts = []
    cpu_to_ids_ratio_ts = []

    for _k in range(decisions):
        if env.t >= env.t_max:
            break

        obs = build_observation_from_history(env, decision_interval, obs_keys)
        cpu_util = decision_cpu_util(env, decision_interval)

        # ---------- policy ----------
        if method.startswith("constant_"):
            constant_cpu = float(method.split("_", 1)[1])
            ids_cpu = np.clip(np.full(n_edges, constant_cpu, dtype=np.float32), ids_cpu_min, ids_cpu_max)
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
            rl_policy.reset()   # call once before decisions loop
            if rl_policy is None:
                raise ValueError("rl_policy is None but method == 'rl'")

            obs_flat = obs.reshape(-1).astype(np.float32)
            if rl_policy.obsnorm is not None:
                obs_flat = rl_policy.normalize_obs_flat(obs_flat)

            delta = rl_policy.act_from_obs_flat(obs_flat)

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

        local_num_req_ts.append(float(df["local_num_req"].mean()) if "local_num_req" in df.columns else 0.0)
        attack_in_rate_ts.append(float(df["attack_in_rate"].mean()) if "attack_in_rate" in df.columns else 0.0)
        cpu_to_ids_ratio_ts.append(float(df["cpu_to_ids_ratio"].mean()) if "cpu_to_ids_ratio" in df.columns else 0.0)




    # If you want "final QoE with SLO" for this edge:
    return {
        "qoe": np.asarray(qoe_ts, dtype=np.float32),
        "cpu_util": np.asarray(cpu_util_ts, dtype=np.float32),
        "local_num_req": np.asarray(local_num_req_ts, dtype=np.float32),
        "attack_in_rate": np.asarray(attack_in_rate_ts, dtype=np.float32),
        "cpu_to_ids_ratio": np.asarray(cpu_to_ids_ratio_ts, dtype=np.float32),
    }


def plot_ts_continuous(results: Dict[str, Dict[str, np.ndarray]], outpath: Path, slo_qoe_min: float = 0.2):
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)

    panels = [
        ("qoe", "QoE"),
        ("local_num_req", "Local #Req"),
        ("attack_in_rate", "Attack in rate"),
        ("cpu_to_ids_ratio", "CPU→IDS Ratio"),
    ]

    for ax, (k, ylabel) in zip(axes, panels):
        for method, series in results.items():
            y = series.get(k, None)
            if y is None or y.size == 0:
                continue

            x = np.arange(len(y))

            if k == "qoe":
                y_valid = y[np.isfinite(y)]
                avg_qoe = float(np.nanmean(y_valid)) if y_valid.size else 0.0
                vio_rate = float(np.nanmean((y_valid < float(slo_qoe_min)).astype(np.float32))) if y_valid.size else 0.0
                
                viol = (y_valid < 0.2).astype(np.float32)
                viol_rate = float(viol.mean()) if len(viol) > 0 else 0.0
                V_edge = np.exp(-3 * viol_rate)                
                label = f"{method} (avg={avg_qoe*V_edge:.3f}, vio={vio_rate:.2%})"
                # violation indicator: 1 if QoE below threshold else 0
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


def plot_qoe_vio_bars(results: Dict[str, Dict[str, np.ndarray]],
                      outpath: Path,
                      qoe_slo_min: float = 0.2,
                      beta: float = 3.0):
    methods, avg_qoe, vio_rate = [], [], []

    for method, series in results.items():
        qoe = series.get("qoe", None)
        if qoe is None:
            continue

        q = np.asarray(qoe, dtype=np.float32)
        q = q[np.isfinite(q)]
        if q.size == 0:
            continue

        vr = float(np.mean((q < qoe_slo_min).astype(np.float32)))  # scalar
        v = float(np.exp(-beta * vr))                               # scalar

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
    axes[0].set_title("Average QoE")
    axes[0].grid(axis="y", alpha=0.3)
    for bar in bars_qoe:
        h = float(bar.get_height())
        axes[0].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=9)

    bars_vio = axes[1].bar(x, vio_rate, width)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_ylabel("Violation Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(f"SLO Violations (QoE < {qoe_slo_min})")
    axes[1].grid(axis="y", alpha=0.3)
    for bar in bars_vio:
        h = float(bar.get_height())
        axes[1].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1%}",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="./configs/simulation_0.yaml")
    ap.add_argument("--outdir", type=str, default="eval_out")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--decision_interval", type=int, default=500)
    ap.add_argument("--scale_step", type=float, default=0.5)
    ap.add_argument("--ids_cpu_min", type=float, default=0.5)

    # ap.add_argument("--rl_ckpt", type=str, default="checkpoints/penv4*4_anneal/ckpt_iter_000600.pt")
    # ap.add_argument("--rl_ckpt", type=str, default="checkpoints/atari_cfg/ckpt_iter_000400.pt")
    ap.add_argument("--rl_ckpt", type=str, default="checkpoints/lstm_epoch_20_linear/ckpt_iter_001100.pt")
    # ap.add_argument("--rl_ckpt", type=str, default="checkpoints/ppo_simulation_0/ckpt_epoch20.pt")
    # ap.add_argument("--rl_ckpt", type=str, default="checkpoints/ppo_simulation_0/ckpt_ema_000900.pt")
    ap.add_argument("--rl_device", type=str, default="cuda")
    ap.add_argument("--rl_greedy", action="store_true")

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
    ]
    obs_dim = len(obs_keys)

    # compute obs_size exactly like training: n_edges * obs_dim
    tmp_env = build_env_base(args.cfg)
    n_edges = len(tmp_env.edge_areas)
    obs_size = n_edges * obs_dim

    rl_policy = None
    if args.rl_ckpt:
        rl_policy = RLPolicy(
            ckpt_path=args.rl_ckpt,
            obs_size=obs_size,
            device=args.rl_device,
            greedy=args.rl_greedy,
        )

    methods = ["random", "constant_0.5", "reactive"]
    if rl_policy is not None:
        methods = methods + ["rl"]

    results: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
    for m in methods:
        results[m] = {
            "qoe": np.array([], dtype=np.float32),
            "cpu_util": np.array([], dtype=np.float32),
            "local_num_req": np.array([], dtype=np.float32),
            "attack_in_rate": np.array([], dtype=np.float32),
            "cpu_to_ids_ratio": np.array([], dtype=np.float32),
        }

    for ep in tqdm(range(args.episodes)):
        ep_seed = base_seed + ep * 1000
        for m in methods:
            q = run_episode(
                cfg=cfg,
                cfg_path=args.cfg,
                method=m,
                decision_interval=args.decision_interval,
                obs_keys=obs_keys,
                scale_step=args.scale_step,
                ids_cpu_min=args.ids_cpu_min,
                seed=ep_seed,
                rl_policy=rl_policy if m == "rl" else None,
            )
            for k, v in q.items():
                results[m][k] = np.concatenate([results[m][k], v])

    plot_ts_continuous(results, outdir / "qoe_ts.png", slo_qoe_min=0.2)
    plot_qoe_vio_bars(results, outdir / "summary.png", qoe_slo_min=0.2)


if __name__ == "__main__":
    main()