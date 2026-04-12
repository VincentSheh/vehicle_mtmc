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
import torch.nn as nn
from tqdm import tqdm

from environment import build_env_base

# MLP training code
from train_mlp import ActorNet as MLPActorNet

# LSTM training code
from train_lstm import FeatureNet as LSTMFeatureNet

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import LSTMModule
from tbsa_offline import TBSAPolicy

# =========================================================
# Policy wrappers
# =========================================================
class BaseRLPolicy:
    def normalize_obs_flat(self, obs_flat: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def act_from_obs_flat(self, obs_flat: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        pass


class MLPPolicy(BaseRLPolicy):
    def __init__(self, ckpt_path: str, obs_size: int, device: str = "cpu", greedy: bool = True):
        self.device = torch.device(device)
        self.greedy = greedy

        self.net = MLPActorNet(obs_dim=obs_size, n_actions=3).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device)
        self.obsnorm = state.get("obsnorm", None)

        if isinstance(state, dict) and "actor_net" in state:
            self.net.load_state_dict(state["actor_net"])
        elif isinstance(state, dict) and "state_dict" in state:
            self.net.load_state_dict(state["state_dict"])
        elif isinstance(state, dict):
            self.net.load_state_dict(state)
        else:
            raise ValueError("Unsupported checkpoint format for MLP actor.")

        self.net.eval()

    def normalize_obs_flat(self, obs_flat: np.ndarray) -> np.ndarray:
        if self.obsnorm is None:
            return obs_flat

        loc = self.obsnorm["loc"].detach().cpu().numpy().reshape(-1)
        scale = self.obsnorm["scale"].detach().cpu().numpy().reshape(-1)
        return (obs_flat - loc) / (scale + 1e-8)

    @torch.no_grad()
    def act_from_obs_flat(self, obs_flat: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(obs_flat.astype(np.float32)).to(self.device)
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)

        if self.greedy:
            a = torch.argmax(probs, dim=-1)
        else:
            a = torch.multinomial(probs, num_samples=1).squeeze(-1)

        delta = int(a.item()) - 1
        return np.array([delta], dtype=np.int64)


class LSTMPolicy(BaseRLPolicy):
    def __init__(self, ckpt_path: str, obs_size: int, device: str = "cpu", greedy: bool = True):
        self.device = torch.device(device)
        self.greedy = greedy

        state = torch.load(ckpt_path, map_location=self.device)
        self.obsnorm = state.get("obsnorm", None)

        feature_dim = state["train_cfg"]["model"]["hidden_dim"]
        n_actions = state["train_cfg"]["model"]["n_actions"]

        self.h_size = feature_dim
        self.n_layers = 1

        feature_module = TensorDictModule(
            LSTMFeatureNet(obs_size, feature_dim).to(self.device),
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

        self.actor = TensorDictSequential(feature_module, self.lstm, actor_head).to(self.device)

        policy_sd = state["policy"]

        mapped_sd = {}
        for k, v in policy_sd.items():
            new_k = k
            new_k = new_k.replace("module.0.module.0.module.0.", "module.0.")
            new_k = new_k.replace("module.0.module.0.module.1.", "module.1.")
            new_k = new_k.replace("module.0.module.1.", "module.2.")
            mapped_sd[new_k] = v

        self.actor.load_state_dict(mapped_sd, strict=False)
        self.actor.eval()

        self._h = None
        self._c = None

        self.env_state = state.get("obsnorm", None)
        self.obs_loc = None
        self.obs_scale = None
        if self.env_state is not None:
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
    def act_from_obs_flat(self, obs_flat: np.ndarray) -> np.ndarray:
        td = TensorDict(
            {"observation_flat": torch.tensor(obs_flat, device=self.device).unsqueeze(0)},
            batch_size=[1],
            device=self.device,
        )

        if self._h is None:
            self._h = torch.zeros(self.n_layers, 1, self.h_size, device=self.device)
            self._c = torch.zeros(self.n_layers, 1, self.h_size, device=self.device)

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

        delta = a - 1
        return np.array([delta], dtype=np.int64)

    def reset(self):
        self._h = None
        self._c = None


class RLPolicy(BaseRLPolicy):
    def __init__(
        self,
        policy_type: str,
        ckpt_path: str,
        obs_size: int,
        device: str = "cpu",
        greedy: bool = True,
    ):
        policy_type = policy_type.lower()
        if policy_type == "mlp":
            self.impl = MLPPolicy(ckpt_path, obs_size, device=device, greedy=greedy)
        elif policy_type == "lstm":
            self.impl = LSTMPolicy(ckpt_path, obs_size, device=device, greedy=greedy)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

    @property
    def obsnorm(self):
        return getattr(self.impl, "obsnorm", None)

    def normalize_obs_flat(self, obs_flat: np.ndarray) -> np.ndarray:
        return self.impl.normalize_obs_flat(obs_flat)

    def act_from_obs_flat(self, obs_flat: np.ndarray) -> np.ndarray:
        return self.impl.act_from_obs_flat(obs_flat)

    def reset(self):
        self.impl.reset()


def _lookup_overhead_rate(pending: float, scaling_time_steps: List[int],
                           scaling_quanta: List[float]) -> float:
    """CPU units consumed per tick given current signed pending."""
    abs_p = abs(pending)
    if abs_p < 1e-9:
        return 0.0
    for i, q in enumerate(scaling_quanta):
        if abs_p <= q + 1e-9:
            duration = scaling_time_steps[i]
            return abs_p / duration
    return abs_p / scaling_time_steps[-1]


def build_observation_from_history(env, decision_interval: int, obs_keys: List[str],
                                    scaling_pending: float = 0.0,
                                    scaling_K: float = 2.0) -> np.ndarray:
    n_edges = len(env.edge_areas)
    obs_dim = len(obs_keys) + 1  # +1 for scaling_pending feature
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
            if k == "attack_in_rate_std":
                if "attack_in_rate" in g.columns:
                    obs[i, j] = float(np.std(g["attack_in_rate"].values))
                continue
            if k not in g.columns:
                continue
            vals = g[k].values
            if k == "I_net":
                obs[i, j] = float(np.sum(vals))
            elif k == "cpu_to_ids_ratio":
                obs[i, j] = float(vals[-1])
            else:
                obs[i, j] = float(np.mean(vals))

    # Last feature: normalized signed scaling_pending in [-1, 1]
    obs[:, -1] = float(np.clip(scaling_pending / scaling_K, -1.0, 1.0))
    return obs


def decision_qoe_mean(env, decision_interval: int) -> float:
    n_edges = len(env.edge_areas)
    if len(env.history) < decision_interval * n_edges:
        return 0.0, 0.0
    block = env.history[-decision_interval * n_edges :]
    return (
        float(np.mean([m.qoe_mean for m in block])),
        float(np.mean([m.benign_col_dmg for m in block])),
    )


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
    env,
    cfg: dict,
    method: str,
    decision_interval: int,
    obs_keys: List[str],
    scale_step: float,
    ids_cpu_min: float,
    seed: int,
    rl_policy: Optional[RLPolicy],
    tbsa_policy: Optional[TBSAPolicy] = None,
) -> Dict[str, np.ndarray]:
    env.reset(seed)

    for edge in env.edge_areas:
        edge.ids_cpu = 4.0

    rng = np.random.default_rng(seed)

    t_max = int(cfg["run"]["t_max"])
    n_edges = len(env.edge_areas)
    ids_cpu_max = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)

    ids_cpu = np.array([e.ids_cpu / e.budget.cpu for e in env.edge_areas], dtype=np.float32)
    ids_cpu = np.clip(ids_cpu, ids_cpu_min, ids_cpu_max)

    decisions = math.ceil(t_max / decision_interval)

    if method == "rl" and rl_policy is not None:
        rl_policy.reset()

    # Scaling overhead state (mirrors TorchRLEnvWrapper)
    scaling_time_steps: List[int] = list(cfg["globals"].get("scaling_time_step", [300, 450, 498, 544]))
    scaling_quanta: List[float] = [0.5, 1.0, 1.5, 2.0]
    scaling_K: float = 2.0
    scaling_pending: float = 0.0
    overhead_rate: float = 0.0

    qoe_ts = []
    benign_col_dmg_ts = []
    cpu_util_ts = []
    local_num_req_ts = []
    attack_in_rate_ts = []
    attack_in_rate_std_ts = []
    attack_drop_rate_ts = []
    ema_mom_ts = []
    cpu_to_ids_ratio_ts = []
    reward_lambda_res_ts = []
    reward_benign_col_dmg_ts = []
    reward_qoe_penalty_ts = []

    for _k in range(decisions):
        if env.t >= env.t_max:
            break

        obs = build_observation_from_history(env, decision_interval, obs_keys,
                                             scaling_pending=scaling_pending, scaling_K=scaling_K)
        cpu_util = decision_cpu_util(env, decision_interval)

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
            if rl_policy is None:
                raise ValueError("rl_policy is None but method == 'rl'")

            obs_flat = obs.reshape(-1).astype(np.float32)
            if rl_policy.obsnorm is not None:
                obs_flat = rl_policy.normalize_obs_flat(obs_flat)

            delta = rl_policy.act_from_obs_flat(obs_flat)

        elif method == "tbsa":
            if tbsa_policy is None:
                raise ValueError("tbsa_policy is None but method == 'tbsa'")

            # Read attack rate and request rate from the last tick only
            if env.history:
                last_records = env.history[-n_edges:]
                last_attack = float(np.mean([r.attack_drop_rate for r in last_records]))
                last_req = float(np.mean([r.local_num_req for r in last_records]))
            else:
                last_attack = 0.0
                last_req = 0.0

            target_cpu = tbsa_policy.select_ids_cpu(last_attack, last_req)
            ids_cpu = np.clip(
                np.full(n_edges, target_cpu, dtype=np.float32),
                ids_cpu_min,
                ids_cpu_max,
            )
            delta = np.zeros(n_edges, dtype=np.int64)  # ids_cpu already set

        else:
            raise ValueError(method)

        prev_ids_cpu = ids_cpu.copy()
        ids_cpu = apply_delta(ids_cpu, delta, scale_step, ids_cpu_min, ids_cpu_max)

        # Update scaling pending; same direction stacks, opposite subtracts
        delta_eff = float(ids_cpu[0] - prev_ids_cpu[0])
        scaling_pending = float(np.clip(scaling_pending + delta_eff, -scaling_K, scaling_K))
        overhead_rate = _lookup_overhead_rate(scaling_pending, scaling_time_steps, scaling_quanta)

        for _ in range(decision_interval):
            if abs(scaling_pending) > 1e-9:
                consumed = float(np.sign(scaling_pending)) * min(abs(scaling_pending), overhead_rate)
                scaling_pending -= consumed
                step_overhead = consumed
            else:
                step_overhead = 0.0
            env.step(ids_cpu, step_overhead)
            if env.t >= env.t_max:
                break

        qoe, benign_col_dmg = decision_qoe_mean(env, decision_interval)
        block = env.history[-decision_interval * n_edges :]
        df = pd.DataFrame([m.__dict__ for m in block])

        qoe_ts.append(qoe)
        benign_col_dmg_ts.append(benign_col_dmg)
        cpu_util_ts.append(cpu_util)
        local_num_req_ts.append(float(df["local_num_req"].mean()) if "local_num_req" in df.columns else 0.0)
        attack_in_rate_ts.append(float(df["attack_in_rate"].mean()) if "attack_in_rate" in df.columns else 0.0)
        attack_in_rate_std_ts.append(float(df["attack_in_rate"].std()) if "attack_in_rate" in df.columns else 0.0)
        attack_drop_rate_ts.append(float(df["attack_drop_rate"].mean()) if "attack_drop_rate" in df.columns else 0.0)
        ema_mom_ts.append(float(df["ema_mom"].mean()) if "ema_mom" in df.columns else 0.0)
        ratios = ids_cpu / np.array([e.budget.cpu for e in env.edge_areas], dtype=np.float32)
        cpu_to_ids_ratio_ts.append(float(ratios.mean()))

        # Reward component breakdown (same coefficients as _build_reward)
        _alpha, _beta, _gamma, _q_th = 1.0 / 0.10, 1.0 / 0.20, 1.0 / 0.12, 0.2
        if "attack_in_rate" in df.columns and "attack_drop_rate" in df.columns and "qoe_mean" in df.columns and "benign_col_dmg" in df.columns:
            _atk_in = df["attack_in_rate"].values.astype(np.float32)
            _atk_drop = df["attack_drop_rate"].values.astype(np.float32)
            _atk_pass = np.maximum(0.0, _atk_in - _atk_drop)
            _lres = np.divide(_atk_pass, _atk_in, out=np.zeros_like(_atk_pass), where=_atk_in > 1e-6)
            _qoes = df["qoe_mean"].values.astype(np.float32)
            _shortfall = np.maximum(0.0, _q_th - _qoes) / max(_q_th, 1e-6)
            _bcd = df["benign_col_dmg"].values.astype(np.float32)
            reward_lambda_res_ts.append(float(_beta * np.mean(_lres)))
            reward_benign_col_dmg_ts.append(float(_gamma * np.mean(_bcd)))
            reward_qoe_penalty_ts.append(float(_alpha * np.mean(_shortfall)))
        else:
            reward_lambda_res_ts.append(0.0)
            reward_benign_col_dmg_ts.append(0.0)
            reward_qoe_penalty_ts.append(0.0)

    return {
        "qoe": np.asarray(qoe_ts, dtype=np.float32),
        "benign_col_dmg": np.asarray(benign_col_dmg_ts, dtype=np.float32),
        "cpu_util": np.asarray(cpu_util_ts, dtype=np.float32),
        "local_num_req": np.asarray(local_num_req_ts, dtype=np.float32),
        "attack_in_rate": np.asarray(attack_in_rate_ts, dtype=np.float32),
        "attack_in_rate_std": np.asarray(attack_in_rate_std_ts, dtype=np.float32),
        "attack_drop_rate": np.asarray(attack_drop_rate_ts, dtype=np.float32),
        "ema_mom": np.asarray(ema_mom_ts, dtype=np.float32),
        "cpu_to_ids_ratio": np.asarray(cpu_to_ids_ratio_ts, dtype=np.float32),
        "reward_lambda_res": np.asarray(reward_lambda_res_ts, dtype=np.float32),
        "reward_benign_col_dmg": np.asarray(reward_benign_col_dmg_ts, dtype=np.float32),
        "reward_qoe_penalty": np.asarray(reward_qoe_penalty_ts, dtype=np.float32),
    }


def plot_ts_continuous(results: Dict[str, Dict[str, np.ndarray]], outpath: Path, slo_qoe_min: float = 0.2, beta=3):
    fig, axes = plt.subplots(12, 1, figsize=(15, 22), sharex=True)

    panels = [
        ("qoe", "QoE"),
        ("benign_col_dmg", "Benign Collateral Damage"),
        ("local_num_req", "Local #Req"),
        ("attack_in_rate", "Attack in rate"),
        ("attack_in_rate_std", "Attack in rate std"),
        ("attack_drop_rate", "Attack drop rate"),
        ("cpu_util", "CPU Utilization"),
        ("cpu_to_ids_ratio", "CPU→IDS Ratio"),
        ("ema_mom", "EMA Momentum"),
        ("reward_lambda_res", "Reward: λ_res (attack residual)"),
        ("reward_benign_col_dmg", "Reward: benign collateral dmg"),
        ("reward_qoe_penalty", "Reward: QoE shortfall penalty"),
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
                V_edge = np.exp(-beta * viol_rate)                
                label = f"{method} (avg={avg_qoe:.3f}, vio={vio_rate:.2%})"
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
    avg_benign_col_dmg, avg_attack_drop_pct = [], []

    for method, series in results.items():
        qoe = series.get("qoe", None)
        benign_col_dmg = series.get("benign_col_dmg", None)
        attack_drop_rate_ts = series.get("attack_drop_rate", None)
        attack_in_rate_ts = series.get("attack_in_rate", None)

        if qoe is None:
            continue

        q = np.asarray(qoe, dtype=np.float32)
        q = q[np.isfinite(q)]
        if q.size == 0:
            continue

        vr = float(np.mean((q < qoe_slo_min).astype(np.float32)))
        v = float(np.exp(-beta * vr))

        methods.append(method)
        vio_rate.append(vr)
        avg_qoe.append(float(np.mean(q)) * v)

        if benign_col_dmg is None:
            avg_benign_col_dmg.append(np.nan)
        else:
            b = np.asarray(benign_col_dmg, dtype=np.float32).reshape(-1)
            b = b[np.isfinite(b)]
            avg_benign_col_dmg.append(float(np.mean(b)) if b.size > 0 else np.nan)

        if attack_drop_rate_ts is None or attack_in_rate_ts is None:
            avg_attack_drop_pct.append(np.nan)
        else:
            drop = np.asarray(attack_drop_rate_ts, dtype=np.float32).reshape(-1)
            atk_in = np.asarray(attack_in_rate_ts, dtype=np.float32).reshape(-1)

            m = min(len(drop), len(atk_in))
            drop = drop[:m]
            atk_in = atk_in[:m]

            valid = np.isfinite(drop) & np.isfinite(atk_in) & (atk_in > 0)
            if np.any(valid):
                drop_pct = drop[valid] / atk_in[valid]
                avg_attack_drop_pct.append(float(np.mean(drop_pct)))
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="./configs/simulation_0.yaml")
    ap.add_argument("--outdir", type=str, default="eval_out")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--decision_interval", type=int, default=500)
    ap.add_argument("--scale_step", type=float, default=0.5)
    ap.add_argument("--ids_cpu_min", type=float, default=0.5)
    ap.add_argument("--tbsa_table", type=str, default="tbsa_table.npz",
                    help="Path to TBSA lookup table (from tbsa_offline.py)")

    args = ap.parse_args()

    RL_POLICIES = [
        # {
        #     "name": "rl_mlp",
        #     "policy_type": "mlp",
        #     "ckpt_path": "checkpoints/atari_cfg/ckpt_iter_000250.pt",
        #     "device": "cuda",
        #     "greedy": True,
        # },
        {
            "name": "rl_lstm",
            "policy_type": "lstm",
            "ckpt_path": "checkpoints/lstm_ep_20_so_obs_e8/ckpt_iter_000750.pt",
            "device": "cuda",
            "greedy": True,
        },
    ]

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg["run"]["seed"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    obs_keys = [
        "local_num_req",
        "attack_in_rate",
        # "attack_in_rate_std",
        "ema_mom",
        "cpu_to_ids_ratio",
        "ids_cpu_utilization",
    ]
    obs_dim = len(obs_keys) +1  # +1 for scaling_pending feature

    env = build_env_base(args.cfg)
    n_edges = len(env.edge_areas)
    obs_size = n_edges * obs_dim

    rl_policies: Dict[str, RLPolicy] = {}
    for spec in RL_POLICIES:
        rl_policies[spec["name"]] = RLPolicy(
            policy_type=spec["policy_type"],
            ckpt_path=spec["ckpt_path"],
            obs_size=obs_size,
            device=spec.get("device", "cpu"),
            greedy=spec.get("greedy", True),
        )

    # Load TBSA table if it exists
    tbsa_policy: Optional[TBSAPolicy] = None
    tbsa_table_path = Path(args.tbsa_table)
    if tbsa_table_path.exists():
        tbsa_policy = TBSAPolicy(str(tbsa_table_path))
        print(f"Loaded TBSA table from {tbsa_table_path}")
    else:
        print(
            f"TBSA table not found at {tbsa_table_path}. "
            "Run tbsa_offline.py first to include the 'tbsa' method."
        )

    # methods = ["random", "constant_0.5", "constant_1.5", "reactive", "tbsa"] + list(rl_policies.keys())
    # methods = ["constant_0.0", "constant_0.5", "constant_1.5"] + list(rl_policies.keys())
    methods = ["constant_1.5", "reactive", "tbsa"] + list(rl_policies.keys())

    # methods = ["reactive"] + list(rl_policies.keys())

    results: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
    for m in methods:
        results[m] = {
            "qoe": np.array([], dtype=np.float32),
            "benign_col_dmg": np.array([], dtype=np.float32),
            "cpu_util": np.array([], dtype=np.float32),
            "local_num_req": np.array([], dtype=np.float32),
            "attack_in_rate": np.array([], dtype=np.float32),
            "attack_in_rate_std": np.array([], dtype=np.float32),
            "attack_drop_rate": np.array([], dtype=np.float32),
            "ema_mom": np.array([], dtype=np.float32),
            "cpu_to_ids_ratio": np.array([], dtype=np.float32),
            "reward_lambda_res": np.array([], dtype=np.float32),
            "reward_benign_col_dmg": np.array([], dtype=np.float32),
            "reward_qoe_penalty": np.array([], dtype=np.float32),
        }

    for ep in tqdm(range(args.episodes)):
        ep_seed = base_seed + ep * 1000
        for m in methods:
            this_policy = rl_policies.get(m, None)

            q = run_episode(
                env=env,
                cfg=cfg,
                method="rl" if this_policy is not None else m,
                decision_interval=args.decision_interval,
                obs_keys=obs_keys,
                scale_step=args.scale_step,
                ids_cpu_min=args.ids_cpu_min,
                seed=ep_seed,
                rl_policy=this_policy,
                tbsa_policy=tbsa_policy,
            )
            for k, v in q.items():
                results[m][k] = np.concatenate([results[m][k], v])

    plot_ts_continuous(results, outdir / "qoe_ts.png", slo_qoe_min=0.2, beta=3)
    plot_qoe_vio_bars(results, outdir / "summary.png", qoe_slo_min=0.2, beta=3)
    print("Plots saved in ", outdir)


if __name__ == "__main__":
    main()