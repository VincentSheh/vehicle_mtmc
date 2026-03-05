"""
Single-file MAPPO-ish (CTDE) Recurrent PPO training script (LSTM), cleaned and consistent.

Key invariants for recurrent PPO (this fixes the "runs but doesn't learn" failure mode)
- ("agents","rnn_h") / ("agents","rnn_c") stored at time t are the INPUT hidden states h_t used to sample a_t
- policy writes updated hidden to ("next","agents","rnn_h") / ("next","agents","rnn_c") which become h_{t+1}
- an env Transform copies next hidden back to current hidden each step, so rollout uses true recurrence
- PPO update recomputes new_logp using the SAME (obs_t, h_t) stored in traj, so ratios are valid

Critic recurrence (optional but supported)
- root "rnn_h_v"/"rnn_c_v" are input hidden for value at time t
- critic writes next hidden to ("next","rnn_h_v")/("next","rnn_c_v") and a Transform carries it forward
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv as PZooParallelEnv

from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
    InteractionType,
    set_composite_lp_aggregate,
)

from torch.distributions import Categorical

from torchrl.collectors import SyncDataCollector
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import Compose, InitTracker, Transform, TransformedEnv, ObservationNorm
from torchrl.modules import ProbabilisticActor

from environment import build_env_base
from logger import wandb_init
import wandb


# =========================
# Recurrent multi-agent actor core
# =========================

class _AgentLSTMCore(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
        )
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        z = self.feat(x)
        h2, c2 = self.lstm(z, (h, c))
        logits = self.head(h2)
        return logits, h2, c2


class MultiAgentLSTM(nn.Module):
    """
    inputs:
      obs: [B,E,D] or [B,T,E,D]
      h,c: [B,E,H]
    outputs:
      logits: [B,E,A] or [B,T,E,A]
      h2,c2: [B,E,H] (final state after consuming the last step)
    """
    def __init__(
        self,
        n_agent_inputs: int,
        n_agent_outputs: int,
        n_agents: int,
        *,
        centralized: bool = False,
        share_params: bool = True,
        hidden_dim: int = 256,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.centralized = bool(centralized)
        self.share_params = bool(share_params)
        self.hidden_dim = int(hidden_dim)

        in_dim = n_agent_inputs * n_agents if self.centralized else n_agent_inputs

        if self.share_params:
            self.agent_networks = nn.ModuleList([_AgentLSTMCore(in_dim, hidden_dim, n_agent_outputs)])
        else:
            self.agent_networks = nn.ModuleList(
                [_AgentLSTMCore(in_dim, hidden_dim, n_agent_outputs) for _ in range(n_agents)]
            )

        if device is not None:
            self.to(device)

    def _pre(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-2] != self.n_agents:
            raise ValueError(f"Expected obs.shape[-2]=={self.n_agents}, got {obs.shape}")
        if self.centralized:
            obs = obs.flatten(-2, -1)
        return obs

    def forward(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        obs = self._pre(obs)

        if obs.ndim == 3:
            # [B,E,D] or centralized [B,E*D]
            E = self.n_agents
            logits_list, h_list, c_list = [], [], []
            for e in range(E):
                net = self.agent_networks[0] if self.share_params else self.agent_networks[e]
                xe = obs if self.centralized else obs[:, e]
                le, he, ce = net(xe, h[:, e], c[:, e])
                logits_list.append(le)
                h_list.append(he)
                c_list.append(ce)
            logits = torch.stack(logits_list, dim=1)  # [B,E,A]
            h2 = torch.stack(h_list, dim=1)           # [B,E,H]
            c2 = torch.stack(c_list, dim=1)           # [B,E,H]
            return logits, h2, c2

        if obs.ndim == 4:
            # [B,T,E,D]
            B, T, E, _ = obs.shape
            logits_t = []
            ht, ct = h, c
            for t in range(T):
                lt, ht, ct = self.forward(obs[:, t], ht, ct)
                logits_t.append(lt)
            return torch.stack(logits_t, dim=1), ht, ct  # [B,T,E,A], [B,E,H], [B,E,H]

        raise RuntimeError(f"Unexpected obs.ndim={obs.ndim}")


# =========================
# Centralized recurrent critic core
# =========================

class CTDELSTMCritic(nn.Module):
    """
    inputs:
      obs_flat: [B,F] or [B,T,F]
      h,c:      [B,H]
    outputs:
      v:  [B,E] or [B,T,E]
      h2,c2: [B,H]
    """
    def __init__(self, n_agents: int, flat_dim: int, hidden_dim: int, device: str | torch.device):
        super().__init__()
        self.n_agents = int(n_agents)
        self.hidden_dim = int(hidden_dim)
        self.feat = nn.Sequential(nn.Linear(int(flat_dim), hidden_dim), nn.Tanh()).to(device)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim).to(device)
        self.head = nn.Linear(hidden_dim, self.n_agents).to(device)

    def forward(self, obs_flat: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        if obs_flat.ndim == 2:
            z = self.feat(obs_flat)
            h2, c2 = self.lstm(z, (h, c))
            v = self.head(h2)
            return v, h2, c2

        if obs_flat.ndim == 3:
            B, T, _ = obs_flat.shape
            vs = []
            ht, ct = h, c
            for t in range(T):
                z = self.feat(obs_flat[:, t])
                ht, ct = self.lstm(z, (ht, ct))
                vs.append(self.head(ht))
            return torch.stack(vs, dim=1), ht, ct

        raise RuntimeError(f"Unexpected obs_flat.ndim={obs_flat.ndim}")


# =========================
# PettingZoo environment
# =========================

class EdgeIDSParallelEnv(PZooParallelEnv):
    metadata = {"name": "edge_ids_parallel_v0"}

    def __init__(
        self,
        cfg_path: str,
        decision_interval: int = 3000,
        seed: int = 0,
        scale_step: float = 0.5,
        ids_cpu_min: float = 0.5,
        threshold: float = 0.35,
        alpha: float = 0.6,
    ):
        self.env = build_env_base(cfg_path)
        self.n_edges = len(self.env.edge_areas)
        self.area_ids = [e.area_id for e in self.env.edge_areas]

        self.possible_agents = list(self.area_ids)
        self.agents = list(self.possible_agents)

        self.base_seed = int(seed)
        self.episode_id = 0
        self.decision_interval = int(decision_interval)

        self.scale_step = float(scale_step)
        self.ids_cpu_min = float(ids_cpu_min)

        self.threshold = float(threshold)
        self.alpha = float(alpha)

        self.obs_keys = [
            "local_num_req",
            "attack_in_rate",
            "ema_mom",
            "cpu_to_ids_ratio",
            "ids_cpu_utilization",
        ]
        self.obs_dim = len(self.obs_keys)

        self._obs_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
                "qoe_mean": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self._act_space = spaces.Discrete(3)

        self.ids_cpu = np.asarray([e.ids_cpu for e in self.env.edge_areas], dtype=np.float32)

    def observation_space(self, agent: str):
        return self._obs_space

    def action_space(self, agent: str):
        return self._act_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        self.episode_id += 1
        episode_seed = self._compute_episode_seed(seed)
        np.random.seed(episode_seed)

        self.env.reset(episode_seed)
        self.agents = list(self.possible_agents)
        self.ids_cpu = np.asarray([e.ids_cpu for e in self.env.edge_areas], dtype=np.float32)

        obs_mat = self._build_observation()
        qoe = self._qoe_vec()

        observations = {
            aid: {"obs": obs_mat[i].copy(), "qoe_mean": np.array([qoe[i]], dtype=np.float32)}
            for i, aid in enumerate(self.area_ids)
        }
        infos = {aid: {} for aid in self.area_ids}
        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        act_vec = np.zeros(self.n_edges, dtype=np.int64)
        for i, aid in enumerate(self.area_ids):
            a = int(actions[aid])
            if a < 0 or a > 2:
                raise ValueError(f"Invalid action {a} for agent {aid}, expected 0..2")
            act_vec[i] = a

        delta_cmd = (act_vec.astype(np.float32) - 1.0) * self.scale_step
        prev_ids = self.ids_cpu.copy()

        new_ids = self.ids_cpu + delta_cmd
        for i, edge in enumerate(self.env.edge_areas):
            max_ids = float(edge.budget.cpu) - 0.5
            new_ids[i] = float(np.clip(new_ids[i], self.ids_cpu_min, max_ids))
        self.ids_cpu = new_ids

        overheads = (self.ids_cpu - prev_ids).astype(np.float32, copy=False).tolist()

        total_rew = np.zeros(self.n_edges, dtype=np.float32)
        terminated_flag = False
        steps = 0

        for _ in range(self.decision_interval):
            self.env.step(self.ids_cpu, overheads)
            total_rew += self._build_reward_per_agent()
            steps += 1
            if self.env.t >= self.env.t_max:
                terminated_flag = True
                break

        rew_agents = (total_rew / max(1, steps)).astype(np.float32, copy=False)
        rewards = {aid: float(rew_agents[i]) for i, aid in enumerate(self.area_ids)}

        terminations = {aid: bool(terminated_flag) for aid in self.area_ids}
        truncations = {aid: False for aid in self.area_ids}
        infos = {aid: {} for aid in self.area_ids}

        if terminated_flag:
            self.agents = []

        obs_mat = self._build_observation()
        qoe = self._qoe_vec()

        observations = {
            aid: {"obs": obs_mat[i].copy(), "qoe_mean": np.array([qoe[i]], dtype=np.float32)}
            for i, aid in enumerate(self.area_ids)
        }
        return observations, rewards, terminations, truncations, infos

    def _compute_episode_seed(self, seed: Optional[int]) -> int:
        if seed is None:
            episode_seed = self.base_seed + self.episode_id * 1000
        else:
            episode_seed = int(seed)
            self.base_seed = episode_seed
        return int(episode_seed)

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros((self.n_edges, self.obs_dim), dtype=np.float32)
        if not self.env.history:
            return obs

        records = self.env.history[-self.decision_interval * self.n_edges :]
        df = pd.DataFrame([m.__dict__ for m in records])

        for i, area_id in enumerate(self.area_ids):
            g = df[df["area_id"] == area_id]
            if g.empty:
                continue
            for j, k in enumerate(self.obs_keys):
                vals = g[k].values
                if k == "ema_mom":
                    vals_nz = vals[vals != 0.0]
                    obs[i, j] = float(np.mean(vals_nz)) if len(vals_nz) else 0.0
                else:
                    obs[i, j] = float(np.mean(vals))
        return obs

    def _build_reward_per_agent(self) -> np.ndarray:
        if len(self.env.history) < self.n_edges:
            return np.zeros(self.n_edges, dtype=np.float32)

        last_block = self.env.history[-self.n_edges :]
        q = np.asarray([float(m.qoe_weighted) for m in last_block], dtype=np.float32)
        penalty = self.alpha * (np.maximum(0.0, self.threshold - q) / self.threshold) ** 2
        return (q - penalty).astype(np.float32, copy=False)

    def _qoe_vec(self) -> np.ndarray:
        qoe = np.asarray(getattr(self.env, "final_qoe", 0.0), dtype=np.float32)
        if qoe.ndim == 0:
            qoe = np.full((self.n_edges,), float(qoe), dtype=np.float32)
        elif qoe.shape[0] != self.n_edges:
            q2 = np.zeros((self.n_edges,), dtype=np.float32)
            m = min(self.n_edges, qoe.shape[0])
            q2[:m] = qoe[:m]
            qoe = q2
        return qoe * 30.0


# =========================
# TorchRL transforms
# =========================

class BuildCentralObs(Transform):
    def __init__(self, n_edges: int, obs_dim: int, out_key: str = "observation_flat"):
        super().__init__(in_keys=[("agents", "observation", "obs")], out_keys=[out_key])
        self.out_key = out_key
        self.flat_dim = int(n_edges * obs_dim)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("agents", "observation", "obs"), default=None)
        if obs is not None:
            td.set(self.out_key, obs.reshape(*obs.shape[:-2], self.flat_dim))
        return td

    def _reset(self, td: TensorDictBase, td_reset: TensorDictBase, **kwargs) -> TensorDictBase:
        obs = td_reset.get(("agents", "observation", "obs"), default=None)
        if obs is not None:
            td_reset.set(self.out_key, obs.reshape(*obs.shape[:-2], self.flat_dim))
        return td_reset

    def transform_observation_spec(self, observation_spec):
        agents_obs = observation_spec[("agents", "observation", "obs")]
        batch_shape = observation_spec.shape
        observation_spec[self.out_key] = UnboundedContinuousTensorSpec(
            shape=(*batch_shape, self.flat_dim),
            dtype=agents_obs.dtype,
            device=agents_obs.device,
        )
        return observation_spec


class BuildSharedDone(Transform):
    def __init__(self):
        super().__init__(in_keys=[("agents", "done"), ("agents", "terminated")], out_keys=["done", "terminated", "truncated"])

    @staticmethod
    def _to_root_flag(x: torch.Tensor) -> torch.Tensor:
        # x is [B,E] or [B,E,1] or [B,T,E] etc, we just want "any agent done"
        if x.ndim >= 1 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        flag = x.to(torch.bool).any(dim=-1, keepdim=True)
        return flag

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        done_a = td.get(("agents", "done"), default=None)
        term_a = td.get(("agents", "terminated"), default=None)
        if done_a is None or term_a is None:
            return td
        td.set("done", self._to_root_flag(done_a))
        td.set("terminated", self._to_root_flag(term_a))
        td.set("truncated", torch.zeros_like(td.get("done")))
        return td

    def _reset(self, td: TensorDictBase, td_reset: TensorDictBase, **kwargs) -> TensorDictBase:
        done_a = td_reset.get(("agents", "done"), default=None)
        term_a = td_reset.get(("agents", "terminated"), default=None)
        if done_a is not None and term_a is not None:
            td_reset.set("done", self._to_root_flag(done_a))
            td_reset.set("terminated", self._to_root_flag(term_a))
            td_reset.set("truncated", torch.zeros_like(td_reset.get("done")))
            return td_reset
        bs = tuple(td_reset.batch_size)
        dev = td_reset.device
        td_reset.set("done", torch.zeros((*bs, 1), dtype=torch.bool, device=dev))
        td_reset.set("terminated", torch.zeros((*bs, 1), dtype=torch.bool, device=dev))
        td_reset.set("truncated", torch.zeros((*bs, 1), dtype=torch.bool, device=dev))
        return td_reset


class InitRNNState(Transform):
    """
    Initialize actor and critic hidden states at reset.
    These are the INPUT hidden states for the current timestep.
    """
    def __init__(self, n_edges: int, hidden_dim: int):
        super().__init__(in_keys=[], out_keys=[])
        self.n_edges = int(n_edges)
        self.hidden_dim = int(hidden_dim)

    def _reset(self, td: TensorDictBase, td_reset: TensorDictBase, **kwargs) -> TensorDictBase:
        bs = tuple(td_reset.batch_size)
        dev = td_reset.device
        td_reset.set(("agents", "rnn_h"), torch.zeros((*bs, self.n_edges, self.hidden_dim), device=dev))
        td_reset.set(("agents", "rnn_c"), torch.zeros((*bs, self.n_edges, self.hidden_dim), device=dev))
        td_reset.set("rnn_h_v", torch.zeros((*bs, self.hidden_dim), device=dev))
        td_reset.set("rnn_c_v", torch.zeros((*bs, self.hidden_dim), device=dev))
        return td_reset
    def transform_observation_spec(self, observation_spec):
        # agent hidden: [*, E, H]
        bs = observation_spec.shape
        dev = observation_spec.device
        dtype = torch.float32

        # infer E from agent obs spec
        E = int(observation_spec[("agents","observation","obs")].shape[-2])
        H = self.hidden_dim

        observation_spec[("agents","rnn_h")] = UnboundedContinuousTensorSpec(shape=(*bs, E, H), dtype=dtype, device=dev)
        observation_spec[("agents","rnn_c")] = UnboundedContinuousTensorSpec(shape=(*bs, E, H), dtype=dtype, device=dev)

        observation_spec["rnn_h_v"] = UnboundedContinuousTensorSpec(shape=(*bs, H), dtype=dtype, device=dev)
        observation_spec["rnn_c_v"] = UnboundedContinuousTensorSpec(shape=(*bs, H), dtype=dtype, device=dev)

        # next keys are typically inferred by TorchRL from root keys, so you don’t need to add them explicitly
        return observation_spec

class CarryActorState(Transform):
    """
    rnn_{t+1} written in ("next","agents",...) becomes rnn_t for next step.
    """
    def __init__(self):
        super().__init__(in_keys=[("next", "agents", "rnn_h"), ("next", "agents", "rnn_c")],
                         out_keys=[("agents", "rnn_h"), ("agents", "rnn_c")])

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        td.set(("agents", "rnn_h"), td.get(("next", "agents", "rnn_h")))
        td.set(("agents", "rnn_c"), td.get(("next", "agents", "rnn_c")))
        return td




# =========================
# TensorDict adapters
# =========================

class ActorAdapter(nn.Module):
    def __init__(self, ma_lstm: MultiAgentLSTM):
        super().__init__()
        self.ma_lstm = ma_lstm

    def forward(self, obs, rnn_h, rnn_c):
        # rollout step: obs [B,E,D], h/c [B,E,H]
        if obs.ndim == 3:
            logits, h2, c2 = self.ma_lstm(obs, rnn_h, rnn_c)
            return logits, h2, c2

        # batch call on [B,T,...] only used if you explicitly call module on traj
        # keep it safe: slice initial state
        h0 = rnn_h[:, 0]
        c0 = rnn_c[:, 0]
        logits, _, _ = self.ma_lstm(obs, h0, c0)
        return logits, rnn_h, rnn_c


class CriticAdapter(nn.Module):
    def __init__(self, core: CTDELSTMCritic):
        super().__init__()
        self.core = core

    def forward(self, observation_flat, rnn_h_v, rnn_c_v):
        # rollout step: [B,F], [B,H]
        if observation_flat.ndim == 2:
            v, h2, c2 = self.core(observation_flat, rnn_h_v, rnn_c_v)
            return v, h2, c2

        # batch call: [B,T,F], rnn_* [B,T,H]
        h0 = rnn_h_v[:, 0]
        c0 = rnn_c_v[:, 0]
        v, _, _ = self.core(observation_flat, h0, c0)
        return v, rnn_h_v, rnn_c_v


# =========================
# PPO / GAE helpers
# =========================

def squeeze_last1(td: TensorDictBase, key):
    if key in td.keys(True, True):
        x = td.get(key)
        if isinstance(x, torch.Tensor) and x.ndim >= 1 and x.shape[-1] == 1:
            td.set(key, x.squeeze(-1))


@torch.no_grad()
def compute_gae_inplace(
    traj: TensorDictBase,
    gamma: float,
    lmbda: float,
    n_edges: int,
    adv_key=("agents", "advantage"),
    vt_key=("agents", "value_target"),
):
    reward = traj.get(("agents", "reward"))
    done = traj.get(("agents", "done")).to(torch.bool)
    terminated = traj.get(("agents", "terminated")).to(torch.bool)
    values = traj.get(("agents", "state_value"))
    next_values = traj.get("next").get(("agents", "state_value"))

    not_end = ~(done | terminated)
    not_end = not_end.to(values.dtype)

    B, T, E = reward.shape
    if E != n_edges:
        raise RuntimeError(f"Expected E={n_edges}, got reward last dim={E}")

    adv = torch.zeros_like(reward)
    last_gae = torch.zeros((B, E), device=reward.device, dtype=reward.dtype)

    for t in reversed(range(T)):
        delta = reward[:, t] + gamma * next_values[:, t] * not_end[:, t] - values[:, t]
        last_gae = delta + gamma * lmbda * not_end[:, t] * last_gae
        adv[:, t] = last_gae

    traj.set(adv_key, adv)
    traj.set(vt_key, adv + values)


def sample_sequence_minibatches(traj: TensorDictBase, seq_len: int, minibatch_size: int):
    # traj batch: [B,T]
    B, T = traj.batch_size
    max_start = T - seq_len
    if max_start < 0:
        raise RuntimeError(f"T={T} < seq_len={seq_len}")

    total = B * (max_start + 1)
    ids_all = torch.randperm(total, device=traj.device)

    for start in range(0, total, minibatch_size):
        ids = ids_all[start : start + minibatch_size]
        b_idx = ids // (max_start + 1)
        t0 = ids % (max_start + 1)

        slices = [traj[b_idx, t0 + k] for k in range(seq_len)]
        yield torch.stack(slices, dim=1).to_tensordict()  # [mb, seq, ...]


def apply_anneal(optim: torch.optim.Optimizer, base_lr: float, base_clip_eps: float, alpha: float, train_cfg: dict):
    if bool(train_cfg["optim"].get("anneal_lr", True)):
        lr_now = base_lr * alpha
        for g in optim.param_groups:
            g["lr"] = lr_now

    clip_now = base_clip_eps
    if bool(train_cfg["loss"].get("anneal_clip_epsilon", True)):
        clip_now = base_clip_eps * alpha

    return clip_now


def save_ckpt(path, policy, value, optim, env_cfg, train_cfg, it, device, env):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "optim": optim.state_dict(),
            "iter": int(it),
            "env_cfg": env_cfg,
            "train_cfg": train_cfg,
            "device": str(device),
            "obsnorm": env.state_dict(),
        },
        str(path),
    )


# =========================
# Builders
# =========================

def make_wrapped_env(cfg_path: str, seed: int, decision_interval: int, scale_step: float, ids_cpu_min: float, threshold: float, alpha: float):
    pz = EdgeIDSParallelEnv(
        cfg_path=cfg_path,
        seed=seed,
        decision_interval=decision_interval,
        scale_step=scale_step,
        ids_cpu_min=ids_cpu_min,
        threshold=threshold,
        alpha=alpha,
    )
    group_map = {"agents": list(pz.possible_agents)}
    return PettingZooWrapper(pz, categorical_actions=True, group_map=group_map)


def build_env_stack(env_cfg: dict, train_cfg: dict, cfg_path: str, num_envs: int):
    decision_interval = int(env_cfg["globals"]["decision_interval"])
    hidden_dim = int(train_cfg["model"]["hidden_dim"])

    base = make_wrapped_env(
        cfg_path=cfg_path,
        seed=int(env_cfg["run"]["seed"]),
        decision_interval=decision_interval,
        scale_step=0.5,
        ids_cpu_min=0.5,
        threshold=0.35,
        alpha=0.6,
    )
    obs_spec = base.observation_spec[("agents", "observation", "obs")]
    n_edges = int(obs_spec.shape[-2])
    obs_dim = int(obs_spec.shape[-1])

    def make_one(i: int):
        def _make():
            return make_wrapped_env(
                cfg_path=cfg_path,
                seed=1000 + i,
                decision_interval=decision_interval,
                scale_step=0.5,
                ids_cpu_min=0.5,
                threshold=0.35,
                alpha=0.6,
            )
        return _make

    penv = ParallelEnv(num_envs, [make_one(i) for i in range(num_envs)], device="cpu")

    transforms: List[Transform] = [
        InitTracker(),
        BuildCentralObs(n_edges=n_edges, obs_dim=obs_dim, out_key="observation_flat"),
        BuildSharedDone(),
        InitRNNState(n_edges=n_edges, hidden_dim=hidden_dim),
        CarryActorState(),
    ]

    # --- OPTIONAL: ObservationNorm on observation_flat (one-time init_stats, no running mean) ---
    use_obsnorm = bool(train_cfg.get("use_observation_norm", False))
    if use_obsnorm:
        on_cfg = train_cfg.get("observation_norm", {})
        transforms.append(
            ObservationNorm(
                in_keys=["observation_flat"],
                standard_normal=bool(on_cfg.get("standard_normal", True)),
            )
        )


    env = TransformedEnv(penv, Compose(*transforms))

    # --- populate mean/std ONCE for ObservationNorm ---
    if use_obsnorm:
        on_cfg = train_cfg.get("observation_norm", {})
        num_iter = int(on_cfg.get("num_iter", 100))
        reduce_dim = tuple(on_cfg.get("reduce_dim", (0, 1)))
        cat_dim = int(on_cfg.get("cat_dim", 0))

        env.transform.train()
        # Find its index so we init the right transform.
        on_idx = None
        for i, tr in enumerate(env.transform.transforms):
            if isinstance(tr, ObservationNorm):
                on_idx = i
                break
        if on_idx is None:
            raise RuntimeError("use_observation_norm=True but ObservationNorm not found in env transforms.")

        env.transform.transforms[on_idx].init_stats(
            num_iter=num_iter,
            reduce_dim=reduce_dim,
            cat_dim=cat_dim,
        )
        env.transform.eval()

    td0 = env.reset()
    print("agents done shape:", td0.get(("agents", "done")).shape)
    print("root done shape:", td0.get("done").shape)
    return env, n_edges, obs_dim

# =========================
# Train
# =========================

def train(
    env_cfg_path: str = "./configs/simulation_ma_0.yaml",
    train_cfg_path: str = "./configs/train.yaml",
    resume_ckpt: Optional[str] = None,
    device: str = "cuda",
):
    with open(env_cfg_path, "r") as f:
        env_cfg = yaml.safe_load(f)
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    run = wandb_init(env_cfg, train_cfg)

    seed = int(env_cfg["run"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    t_max = int(env_cfg["run"]["t_max"])
    decision_interval = int(env_cfg["globals"]["decision_interval"])
    num_envs = int(train_cfg["collector"]["num_envs"])
    decisions_per_episode = int(math.ceil(t_max / decision_interval))

    env, n_edges, obs_dim = build_env_stack(env_cfg, train_cfg, env_cfg_path, num_envs)

    set_composite_lp_aggregate(False).set()

    n_actions = int(train_cfg["model"]["n_actions"])
    hidden_dim = int(train_cfg["model"]["hidden_dim"])

    actor_net = MultiAgentLSTM(
        n_agent_inputs=obs_dim,
        n_agent_outputs=n_actions,
        n_agents=n_edges,
        centralized=False,
        share_params=True,
        hidden_dim=hidden_dim,
        device=device,
    )

    actor_td = TensorDictModule(
        ActorAdapter(actor_net),
        in_keys=[("agents", "observation", "obs"), ("agents", "rnn_h"), ("agents", "rnn_c")],
        out_keys=[("agents", "logits"), ("next", "agents", "rnn_h"), ("next", "agents", "rnn_c")],
    )

    policy = ProbabilisticActor(
        module=actor_td,
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    critic_core = CTDELSTMCritic(n_agents=n_edges, flat_dim=n_edges * obs_dim, hidden_dim=hidden_dim, device=device)

    value = TensorDictSequential(
        TensorDictModule(
            CriticAdapter(critic_core),
            in_keys=["observation_flat", "rnn_h_v", "rnn_c_v"],
            out_keys=[("agents", "state_value"), ("next", "rnn_h_v"), ("next", "rnn_c_v")],
        )
    )

    optim = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=float(train_cfg["optim"]["lr"]),
        weight_decay=float(train_cfg["optim"]["weight_decay"]),
        eps=float(train_cfg["optim"]["eps"]),
    )

    if resume_ckpt:
        state = torch.load(resume_ckpt, map_location=device)
        policy.load_state_dict(state["policy"])
        value.load_state_dict(state["value"])
        optim.load_state_dict(state["optim"])
        try:
            env.load_state_dict(state["env_state"])
        except Exception:
            pass

    frames_per_batch = int(train_cfg["collector"].get("frames_per_batch", decisions_per_episode * num_envs))
    total_frames = int(train_cfg["collector"]["total_frames"])

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        trust_policy=bool(train_cfg["collector"]["trust_policy"]),
        split_trajs=False,
    )

    minibatch_size = int(train_cfg["loss"]["mini_batch_size"])
    ppo_epochs = int(train_cfg["loss"]["ppo_epochs"])
    max_grad_norm = float(train_cfg["optim"]["max_grad_norm"])

    entropy_coeff = float(train_cfg["loss"]["entropy_coeff"])
    critic_coeff = float(train_cfg["loss"]["critic_coeff"])
    gamma = float(train_cfg["loss"]["gamma"])
    gae_lambda = float(train_cfg["loss"]["gae_lambda"])

    base_lr = float(train_cfg["optim"]["lr"])
    base_clip_eps = float(train_cfg["loss"]["clip_epsilon"])

    total_network_updates = (
        int(train_cfg["collector"]["total_frames"]) // int(train_cfg["collector"]["frames_per_batch"])
    ) * int(train_cfg["loss"]["ppo_epochs"]) * math.ceil(frames_per_batch / minibatch_size)
    num_network_updates = 0

    ckpt_dir = Path("checkpoints") / run.name
    ckpt_every = 50
    best_qoe = -1e9

    for it, batch in enumerate(collector):
        traj = batch.clone(False)
        with torch.no_grad():
            # pick one batch element and one time step
            b = 0
            t = 0

            obs = traj.get(("agents","observation","obs"))[b, t]           # [E,D]
            act = traj.get(("agents","action"))[b, t].long()               # [E] or [E,1]
            if act.ndim == 2 and act.shape[-1] == 1:
                act = act.squeeze(-1)

            h = traj.get(("agents","rnn_h"))[b, t]                          # [E,H]
            c = traj.get(("agents","rnn_c"))[b, t]                          # [E,H]
            logits, _, _ = actor_net(obs.unsqueeze(0), h.unsqueeze(0), c.unsqueeze(0))  # -> [1,E,A]
            logp_re = Categorical(logits=logits.squeeze(0)).log_prob(act)   # [E]

            logp_stored = traj.get(("agents","sample_log_prob"))[b, t]      # [E] or [E,1]
            if logp_stored.ndim == 2 and logp_stored.shape[-1] == 1:
                logp_stored = logp_stored.squeeze(-1)

            print("max|re-stored|:", (logp_re - logp_stored).abs().max().item())        

        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated"), ("agents", "state_value")]:
            squeeze_last1(traj, k)
            squeeze_last1(traj.get("next"), k)

        traj.set(("agents", "done"), traj.get(("agents", "done")).to(torch.bool))
        traj.set(("agents", "terminated"), traj.get(("agents", "terminated")).to(torch.bool))
        traj.get("next").set(("agents", "done"), traj.get("next").get(("agents", "done")).to(torch.bool))
        traj.get("next").set(("agents", "terminated"), traj.get("next").get(("agents", "terminated")).to(torch.bool))

        # TorchRL often stores these under "next", but GAE expects them at root
        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated")]:
            nk = ("next",) + k
            if nk in traj.keys(True, True):
                traj.set(k, traj.get(nk))
            if nk in traj.get("next").keys(True, True):
                traj.get("next").set(k, traj.get("next").get(nk))

        # Squeeze trailing singleton dims (e.g. [...,1]) for GAE math
        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated")]:
            squeeze_last1(traj, k)
            squeeze_last1(traj.get("next"), k)                
        # value + GAE
        with torch.no_grad():
            value(traj)
            value(traj.get("next"))
            compute_gae_inplace(traj, gamma=gamma, lmbda=gae_lambda, n_edges=n_edges)

        last_total_loss = last_policy_loss = last_critic_loss = last_entropy = None
        seq_len = int(train_cfg["loss"].get("seq_len", 16))

        for _ in range(ppo_epochs):
            for sub in sample_sequence_minibatches(traj, seq_len=seq_len, minibatch_size=minibatch_size):
                alpha = 1.0 - (num_network_updates / max(1, total_network_updates))
                if alpha < 0.0:
                    alpha = 0.0
                clip_eps_now = apply_anneal(optim, base_lr, base_clip_eps, alpha, train_cfg)
                num_network_updates += 1

                obs = sub.get(("agents", "observation", "obs"))        # [mb,seq,E,D]
                act = sub.get(("agents", "action")).long()             # [mb,seq,E] or [mb,seq,E,1]
                if act.ndim == 4 and act.shape[-1] == 1:
                    act = act.squeeze(-1)

                old_logp = sub.get(("agents", "sample_log_prob"))      # [mb,seq,E] or [...,1]
                if old_logp.ndim == 4 and old_logp.shape[-1] == 1:
                    old_logp = old_logp.squeeze(-1)

                adv = sub.get(("agents", "advantage"))
                v_targ = sub.get(("agents", "value_target"))

                # normalize advantage (helps stability)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # recompute policy logp with the SAME h_t stored in traj
                h0 = sub.get(("agents", "rnn_h"))[:, 0]                # [mb,E,H]
                c0 = sub.get(("agents", "rnn_c"))[:, 0]                # [mb,E,H]
                logits, _, _ = actor_net(obs, h0, c0)                  # [mb,seq,E,A]
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act)                          # [mb,seq,E]
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps_now, 1.0 + clip_eps_now) * adv
                policy_loss = -(torch.min(surr1, surr2)).mean()

                # critic recompute
                obs_flat = sub.get("observation_flat")
                hv0 = sub.get("rnn_h_v")[:, 0]
                cv0 = sub.get("rnn_c_v")[:, 0]
                v_pred, _, _ = critic_core(obs_flat, hv0, cv0)         # [mb,seq,E]
                critic_loss = 0.5 * (v_targ - v_pred).pow(2).mean()

                entropy_loss = -entropy.mean()
                total_loss = policy_loss + critic_coeff * critic_loss + entropy_coeff * entropy_loss

                optim.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), max_grad_norm)
                optim.step()

                last_total_loss = total_loss.detach()
                last_policy_loss = policy_loss.detach()
                last_critic_loss = critic_loss.detach()
                last_entropy = entropy.mean().detach()

        collector.update_policy_weights_()

        reward_mean = float(traj.get(("agents", "reward")).mean().item())
        qoe_mean = float(traj.get(("next", "agents", "observation", "qoe_mean")).mean().item())

        if (it + 1) % ckpt_every == 0:
            save_ckpt(ckpt_dir / f"ckpt_iter_{it+1:06d}.pt", policy, value, optim, env_cfg, train_cfg, it + 1, device, env)

        if qoe_mean > best_qoe:
            best_qoe = qoe_mean
            save_ckpt(ckpt_dir / "ckpt_best.pt", policy, value, optim, env_cfg, train_cfg, it + 1, device, env)

        print(f"it={it} reward_mean={reward_mean:.4f}, qoe_mean={qoe_mean:.4f}")

        wandb.log(
            {
                "iter": it,
                "qoe/mean": qoe_mean,
                "reward/mean": reward_mean,
                "loss/total": float(last_total_loss.item()) if last_total_loss is not None else 0.0,
                "loss/policy": float(last_policy_loss.item()) if last_policy_loss is not None else 0.0,
                "loss/critic": float(last_critic_loss.item()) if last_critic_loss is not None else 0.0,
                "entropy": float(last_entropy.item()) if last_entropy is not None else 0.0,
                "train/alpha": alpha,
                "train/lr": optim.param_groups[0]["lr"],
                "train/clip_eps": clip_eps_now,
            }
        )


if __name__ == "__main__":
    train()