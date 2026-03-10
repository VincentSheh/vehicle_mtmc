

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

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType, set_composite_lp_aggregate
from torch.distributions import Categorical

from torchrl.collectors import SyncDataCollector
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import Compose, InitTracker, Transform, TransformedEnv, ObservationNorm
from torchrl.modules import ProbabilisticActor, LSTMModule

from environment import build_env_base
from logger import wandb_init
import wandb


# =========================================================
# Utils
# =========================================================

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


def squeeze_last1(td: TensorDictBase, key):
    if key in td.keys(True, True):
        x = td.get(key)
        if torch.is_tensor(x) and x.ndim >= 1 and x.shape[-1] == 1:
            td.set(key, x.squeeze(-1))


# =========================================================
# Env (PettingZoo ParallelEnv)
# =========================================================

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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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

    def step(self, actions: Dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        act_vec = np.zeros(self.n_edges, dtype=np.int64)
        for i, aid in enumerate(self.area_ids):
            a = int(actions[aid])
            if a < 0 or a > 2:
                raise ValueError(f"Invalid action {a} for agent {aid}, expected 0..2")
            act_vec[i] = a
            
        # RL action drives allocation
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
            return int(self.base_seed + self.episode_id * 1000)
        self.base_seed = int(seed)
        return int(seed)

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

        target_ids_cpu = 2.0

        r = np.zeros(self.n_edges, dtype=np.float32)
        for i, aid in enumerate(self.area_ids):
            r[i] = -abs(float(self.ids_cpu[i]) - target_ids_cpu)

        return r
        
    def _qoe_vec(self) -> np.ndarray:
        qoe = np.asarray(getattr(self.env, "final_qoe", 0.0), dtype=np.float32)
        if qoe.ndim == 0:
            qoe = np.full((self.n_edges,), float(qoe), dtype=np.float32)
        return qoe * 30.0


# =========================================================
# Transforms
# =========================================================

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
        bs = observation_spec.shape
        observation_spec[self.out_key] = UnboundedContinuousTensorSpec(
            shape=(*bs, self.flat_dim),
            dtype=agents_obs.dtype,
            device=agents_obs.device,
        )
        return observation_spec

class AddAgentID(Transform):
    def __init__(self, n_edges: int, in_key=("agents","observation","obs"), out_key=("agents","observation","obs")):
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.n_edges = int(n_edges)
        self.in_key = in_key
        self.out_key = out_key

    def _append_id(self, obs: torch.Tensor) -> torch.Tensor:
        E = self.n_edges
        eye = torch.eye(E, device=obs.device, dtype=obs.dtype)
        if obs.ndim == 3:
            B = obs.shape[0]
            ids = eye.unsqueeze(0).expand(B, E, E)
            return torch.cat([obs, ids], dim=-1)
        if obs.ndim == 4:
            B, T = obs.shape[:2]
            ids = eye.view(1,1,E,E).expand(B, T, E, E)
            return torch.cat([obs, ids], dim=-1)
        raise RuntimeError(f"obs must be 3D or 4D, got {obs.shape}")

    def _call(self, td):
        obs = td.get(self.in_key, None)
        if obs is not None:
            td.set(self.out_key, self._append_id(obs))

        nxt = td.get("next", None)
        if nxt is not None:
            obs2 = nxt.get(self.in_key, None)
            if obs2 is not None:
                nxt.set(self.out_key, self._append_id(obs2))
        return td

    def _reset(self, td, td_reset, **kwargs):
        obs = td_reset.get(self.in_key, None)
        if obs is not None:
            td_reset.set(self.out_key, self._append_id(obs))
        return td_reset

    def transform_observation_spec(self, observation_spec):
        spec = observation_spec[self.in_key]
        new_shape = (*spec.shape[:-1], spec.shape[-1] + self.n_edges)
        observation_spec[self.out_key] = UnboundedContinuousTensorSpec(
            shape=new_shape, dtype=spec.dtype, device=spec.device
        )
        return observation_spec

class WriteRecurrentOutToNext(Transform):
    def __init__(self):
        super().__init__(
            in_keys=[
                ("agents","recurrent_state_h_out"),
                ("agents","recurrent_state_c_out"),
                ("agents","recurrent_state_h_v_out"),
                ("agents","recurrent_state_c_v_out"),
            ],
            out_keys=[],
        )

    def _call(self, td):
        nxt = td.get("next")
        if nxt is None:
            return td

        nxt.set(("agents","recurrent_state_h"), td.get(("agents","recurrent_state_h_out")))
        nxt.set(("agents","recurrent_state_c"), td.get(("agents","recurrent_state_c_out")))

        nxt.set(("agents","recurrent_state_h_v"), td.get(("agents","recurrent_state_h_v_out")))
        nxt.set(("agents","recurrent_state_c_v"), td.get(("agents","recurrent_state_c_v_out")))
        return td

class InitRecurrentState(Transform):
    def __init__(self, n_edges: int, actor_hidden_dim: int, critic_hidden_dim: int):
        super().__init__(
            in_keys=[], 
            out_keys=[
                ("agents", "recurrent_state_h"), ("agents", "recurrent_state_c"),
                ("agents", "recurrent_state_h_out"), ("agents", "recurrent_state_c_out"),
                ("agents", "recurrent_state_h_v"), ("agents", "recurrent_state_c_v"),
                ("agents", "recurrent_state_h_v_out"), ("agents", "recurrent_state_c_v_out"),
            ]
        )
        self.n_edges = int(n_edges)
        self.actor_hidden_dim = int(actor_hidden_dim)
        self.critic_hidden_dim = int(critic_hidden_dim)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        # Bypass the base class zip logic; we only act on reset, not during step
        return td

    def _reset(self, td, td_reset, **kwargs):
        bs = tuple(td_reset.batch_size)
        dev = td_reset.device
        E = self.n_edges
        Ha = self.actor_hidden_dim
        Hc = self.critic_hidden_dim

        # actor
        td_reset.set(("agents", "recurrent_state_h"), torch.zeros((*bs, E, 1, Ha), device=dev))
        td_reset.set(("agents", "recurrent_state_c"), torch.zeros((*bs, E, 1, Ha), device=dev))
        td_reset.set(("agents", "recurrent_state_h_out"), torch.zeros((*bs, E, 1, Ha), device=dev))
        td_reset.set(("agents", "recurrent_state_c_out"), torch.zeros((*bs, E, 1, Ha), device=dev))

        # critic
        td_reset.set(("agents", "recurrent_state_h_v"), torch.zeros((*bs, E, 1, Hc), device=dev))
        td_reset.set(("agents", "recurrent_state_c_v"), torch.zeros((*bs, E, 1, Hc), device=dev))
        td_reset.set(("agents", "recurrent_state_h_v_out"), torch.zeros((*bs, E, 1, Hc), device=dev))
        td_reset.set(("agents", "recurrent_state_c_v_out"), torch.zeros((*bs, E, 1, Hc), device=dev))
        return td_reset

    def transform_observation_spec(self, observation_spec):
        bs = observation_spec.shape
        E = self.n_edges
        Ha = self.actor_hidden_dim
        Hc = self.critic_hidden_dim

        # Register the actor states
        observation_spec[("agents", "recurrent_state_h")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Ha), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_c")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Ha), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_h_out")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Ha), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_c_out")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Ha), device=observation_spec.device)

        # Register the critic states
        observation_spec[("agents", "recurrent_state_h_v")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Hc), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_c_v")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Hc), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_h_v_out")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Hc), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_c_v_out")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Hc), device=observation_spec.device)
        
        return observation_spec

class CarryActorRecurrentState(Transform):
    def __init__(self):
        super().__init__(
            in_keys=[("next","agents","recurrent_state_h"), ("next","agents","recurrent_state_c")],
            out_keys=[("agents","recurrent_state_h"), ("agents","recurrent_state_c")],
        )

    def _call(self, td):
        td.set(("agents","recurrent_state_h"), td.get(("next","agents","recurrent_state_h")))
        td.set(("agents","recurrent_state_c"), td.get(("next","agents","recurrent_state_c")))
        return td

class CarryCriticState(Transform):
    def __init__(self):
        super().__init__(
            in_keys=[
                ("next","agents","recurrent_state_h_v"),
                ("next","agents","recurrent_state_c_v"),
            ],
            out_keys=[
                ("agents","recurrent_state_h_v"),
                ("agents","recurrent_state_c_v"),
            ],
        )

    def _call(self, td):
        td.set(("agents","recurrent_state_h_v"), td.get(("next","agents","recurrent_state_h_v")))
        td.set(("agents","recurrent_state_c_v"), td.get(("next","agents","recurrent_state_c_v")))
        return td
    
# =========================================================
# Networks (Feature + LSTMModule + heads)
# =========================================================

class FeatureNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AgentRecurrentCore(nn.Module):
    in_keys = [
        ("agents", "observation", "obs"),
        "is_init",
        ("agents", "recurrent_state_h"),
        ("agents", "recurrent_state_c"),
    ]
    out_keys = [
        ("agents", "features"),
        ("agents", "recurrent_state_h_out"),
        ("agents", "recurrent_state_c_out"),
    ]    
    def __init__(self, n_edges: int, obs_dim: int, hidden_dim: int, device: str):
        super().__init__()
        self.n_edges = int(n_edges)
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)

        self.feature = FeatureNet(self.obs_dim, self.hidden_dim).to(device)
        self.lstm = LSTMModule(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            in_key="x_t",
            out_key="x_t",
            device=device,
        )

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("agents","observation","obs"))
        is_init = td.get("is_init")

        step_mode = (obs.ndim == 3)
        if step_mode:
            obs = obs.unsqueeze(1)
        if is_init.ndim == 2:
            is_init = is_init.unsqueeze(1)

        B, T, E, D = obs.shape
        H = self.hidden_dim

        obs_be = obs.reshape(B * E, T, D)
        feats_be = self.feature(obs_be)
        is_init_be = is_init.repeat_interleave(E, dim=0)

        h = td.get(("agents","recurrent_state_h"))
        c = td.get(("agents","recurrent_state_c"))

        if h.ndim == 5:
            h0 = h[:, 0]
            c0 = c[:, 0]
        else:
            h0 = h
            c0 = c

        h_be = h0.reshape(B * E, 1, H).contiguous()
        c_be = c0.reshape(B * E, 1, H).contiguous()

        td_be = TensorDict(
            {"recurrent_state_h": h_be, "recurrent_state_c": c_be},
            batch_size=[B * E],
            device=feats_be.device,
        )

        out = []
        for t in range(T):
            td_be.set("is_init", is_init_be[:, t])
            td_be.set("x_t", feats_be[:, t])
            self.lstm(td_be)
            out.append(td_be.get("x_t"))

        feats_be2 = torch.stack(out, dim=1)
        feats = feats_be2.view(B, E, T, H).transpose(1, 2).contiguous()

        if step_mode:
            td.set(("agents","features"), feats[:, 0])
            td.set(("agents","recurrent_state_h_out"), td_be.get("recurrent_state_h").view(B, E, 1, H))
            td.set(("agents","recurrent_state_c_out"), td_be.get("recurrent_state_c").view(B, E, 1, H))
        else:
            td.set(("agents","features"), feats)

        return td

class CriticRecurrentCore(nn.Module):
    in_keys = [
        ("agents", "observation", "obs"),
        "is_init",
        ("agents", "recurrent_state_h_v"),
        ("agents", "recurrent_state_c_v"),
    ]
    out_keys = [
        ("agents", "vf_features"),
        ("agents", "recurrent_state_h_v_out"),
        ("agents", "recurrent_state_c_v_out"),
    ]
    def __init__(self, n_edges: int, obs_dim: int, hidden_dim: int, device: str):
        super().__init__()
        self.n_edges = int(n_edges)
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)

        self.feature = FeatureNet(self.obs_dim, self.hidden_dim).to(device)
        self.lstm = LSTMModule(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            in_key="v_t",
            out_key="v_t",
            device=device,
        )

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("agents","observation","obs"))
        is_init = td.get("is_init")

        step_mode = (obs.ndim == 3)
        if step_mode:
            obs = obs.unsqueeze(1)
        if is_init.ndim == 2:
            is_init = is_init.unsqueeze(1)

        B, T, E, D = obs.shape
        H = self.hidden_dim

        obs_be = obs.reshape(B * E, T, D)
        feats_be = self.feature(obs_be)
        is_init_be = is_init.repeat_interleave(E, dim=0)

        h = td.get(("agents","recurrent_state_h_v"))
        c = td.get(("agents","recurrent_state_c_v"))

        if h.ndim == 5:
            h0 = h[:, 0]
            c0 = c[:, 0]
        else:
            h0 = h
            c0 = c

        h_be = h0.reshape(B * E, 1, H).contiguous()
        c_be = c0.reshape(B * E, 1, H).contiguous()

        td_be = TensorDict(
            {"recurrent_state_h": h_be, "recurrent_state_c": c_be},
            batch_size=[B * E],
            device=feats_be.device,
        )

        out = []
        for t in range(T):
            td_be.set("is_init", is_init_be[:, t])
            td_be.set("v_t", feats_be[:, t])
            self.lstm(td_be)
            out.append(td_be.get("v_t"))

        v_be = torch.stack(out, dim=1)
        v = v_be.view(B, E, T, H).transpose(1, 2).contiguous()

        if step_mode:
            td.set(("agents","vf_features"), v[:, 0])
            td.set(("agents","recurrent_state_h_v_out"), td_be.get("recurrent_state_h").view(B, E, 1, H))
            td.set(("agents","recurrent_state_c_v_out"), td_be.get("recurrent_state_c").view(B, E, 1, H))
        else:
            td.set(("agents","vf_features"), v)

        return td

# =========================================================
# PPO / GAE
# =========================================================

@torch.no_grad()
def compute_gae_inplace(traj: TensorDictBase, gamma: float, lmbda: float, n_edges: int):
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

    traj.set(("agents", "advantage"), adv)
    traj.set(("agents", "value_target"), adv + values)


def valid_sequence_minibatches(traj: TensorDictBase, seq_len: int, minibatch_size: int):
    B, T = traj.batch_size
    if T < seq_len:
        raise RuntimeError(f"T={T} < seq_len={seq_len}")

    done = traj.get(("agents", "done")).to(torch.bool)
    while done.ndim > 3:
        done = done.squeeze(-1)
    done_any = done.any(dim=-1)

    max_t0 = T - seq_len
    csum = torch.cumsum(done_any.to(torch.int32), dim=1)

    left = csum[:, : max_t0 + 1]
    right = csum[:, seq_len - 1 : seq_len - 1 + (max_t0 + 1)]
    prev_left = torch.cat([torch.zeros(B, 1, device=traj.device, dtype=csum.dtype), left[:, :-1]], dim=1)
    window_sum = right - prev_left
    valid = window_sum == 0

    valid_idx = valid.nonzero(as_tuple=False)
    if valid_idx.numel() == 0:
        all_b = torch.arange(B, device=traj.device).repeat_interleave(max_t0 + 1)
        all_t0 = torch.arange(max_t0 + 1, device=traj.device).repeat(B)
        valid_idx = torch.stack([all_b, all_t0], dim=1)

    perm = torch.randperm(valid_idx.shape[0], device=traj.device)
    valid_idx = valid_idx[perm]

    for start in range(0, valid_idx.shape[0], minibatch_size):
        idx = valid_idx[start : start + minibatch_size]
        b_idx = idx[:, 0]
        t0 = idx[:, 1]
        slices = [traj[b_idx, t0 + k] for k in range(seq_len)]
        yield torch.stack(slices, dim=1).to_tensordict()


# =========================================================
# Builders
# =========================================================

def make_wrapped_env(cfg_path: str, seed: int, decision_interval: int):
    pz = EdgeIDSParallelEnv(cfg_path=cfg_path, seed=seed, decision_interval=decision_interval)
    group_map = {"agents": list(pz.possible_agents)}
    return PettingZooWrapper(pz, categorical_actions=True, group_map=group_map)

def build_env_stack(env_cfg: dict, train_cfg: dict, cfg_path: str, num_envs: int):
    decision_interval = int(env_cfg["globals"]["decision_interval"])

    base = make_wrapped_env(cfg_path=cfg_path, seed=int(env_cfg["run"]["seed"]), decision_interval=decision_interval)
    obs_spec = base.observation_spec[("agents", "observation", "obs")]
    n_edges = int(obs_spec.shape[-2])
    obs_dim = int(obs_spec.shape[-1])
    obs_dim_with_id = obs_dim + n_edges

    def make_one(i: int):
        def _make():
            return make_wrapped_env(cfg_path=cfg_path, seed=1000 + i, decision_interval=decision_interval)
        return _make

    penv = ParallelEnv(num_envs, [make_one(i) for i in range(num_envs)], device="cpu")

    hidden_dim = int(train_cfg["model"]["hidden_dim"])
    actor_h = hidden_dim
    critic_h = hidden_dim * 2

    transforms: List[Transform] = [
        InitTracker(),
        AddAgentID(n_edges=n_edges),
        InitRecurrentState(n_edges=n_edges, actor_hidden_dim=actor_h, critic_hidden_dim=critic_h),
        WriteRecurrentOutToNext(),
    ]

    env = TransformedEnv(penv, Compose(*transforms))
    td0 = env.reset()
    print("obs step shape:", td0.get(("agents","observation","obs")).shape)

    return env, n_edges, obs_dim_with_id


# =========================================================
# Train
# =========================================================

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

    # --- actor ---
    actor_core = AgentRecurrentCore(
        n_edges=n_edges,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        device=device,
    )   
    actor_head = TensorDictModule(
        nn.Linear(hidden_dim, n_actions).to(device),
        in_keys=[("agents", "features")],
        out_keys=[("agents", "logits")],
    )
    policy = ProbabilisticActor(
        module=TensorDictSequential(actor_core, actor_head),
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    critic_core = CriticRecurrentCore(n_edges=n_edges, obs_dim=obs_dim, hidden_dim=hidden_dim*2, device=device)

    critic_head = TensorDictModule(
        nn.Linear(hidden_dim * 2, 1).to(device),
        in_keys=[("agents","vf_features")],
        out_keys=[("agents","state_value")],
    )

    value_net = TensorDictSequential(critic_core, critic_head)

    # Combining actor and critic into a single Sequential ensures the Critic is also run 
    # during collection to correctly track and propagate its recurrent states!
    collector_policy = TensorDictSequential(policy, value_net)

    optim = torch.optim.Adam(
        list(actor_core.parameters())
        + list(actor_head.parameters())
        + list(critic_core.parameters())
        + list(critic_head.parameters()),
        lr=float(train_cfg["optim"]["lr"]),
        weight_decay=float(train_cfg["optim"]["weight_decay"]),
        eps=float(train_cfg["optim"]["eps"]),
    )

    if resume_ckpt:
        state = torch.load(resume_ckpt, map_location=device)
        policy.load_state_dict(state["policy"])
        value_net.load_state_dict(state["value"])
        optim.load_state_dict(state["optim"])
        try:
            env.load_state_dict(state["obsnorm"])
        except Exception:
            pass

    frames_per_batch = int(train_cfg["collector"].get("frames_per_batch", decisions_per_episode * num_envs))
    total_frames = int(train_cfg["collector"]["total_frames"])

    collector = SyncDataCollector(
        env,
        policy=collector_policy, # Using the combined policy here
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

    ckpt_dir = Path("checkpoints") / run.name
    ckpt_every = 50
    best_qoe = -1e9

    total_updates_est = max(1, (total_frames // frames_per_batch) * ppo_epochs * math.ceil(frames_per_batch / minibatch_size))
    updates_done = 0

    for it, batch in enumerate(collector):
        traj = batch.clone(False)

        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated")]:
            nk = ("next",) + k
            if k not in traj.keys(True, True) and nk in traj.keys(True, True):
                traj.set(k, traj.get(nk))

            squeeze_last1(traj, k)
            squeeze_last1(traj.get("next"), k)

        if ("agents","done") in traj.keys(True, True):
            traj.set(("agents", "done"), traj.get(("agents", "done")).to(torch.bool))
        if ("agents","terminated") in traj.keys(True, True):
            traj.set(("agents", "terminated"), traj.get(("agents", "terminated")).to(torch.bool))
        if ("agents","done") in traj.get("next").keys(True, True):
            traj.get("next").set(("agents", "done"), traj.get("next").get(("agents","done")).to(torch.bool))
        if ("agents","terminated") in traj.get("next").keys(True, True):
            traj.get("next").set(("agents", "terminated"), traj.get("next").get(("agents","terminated")).to(torch.bool))

        # --- values + GAE ---
        with torch.no_grad():
            # Since value_net ran during collection, traj already holds values and proper recurrent context!
            # Recomputing values on `traj["next"]` is safe and will yield the proper bootstrap values.
            value_net(traj)
            value_net(traj.get("next"))
            squeeze_last1(traj, ("agents","state_value"))
            squeeze_last1(traj.get("next"), ("agents","state_value"))          
            compute_gae_inplace(traj, gamma=gamma, lmbda=gae_lambda, n_edges=n_edges)

        last_total_loss = last_policy_loss = last_critic_loss = last_entropy = None
        seq_len = int(train_cfg["loss"].get("seq_len", 32))

        for _ in range(ppo_epochs):
            for sub in valid_sequence_minibatches(traj, seq_len=seq_len, minibatch_size=minibatch_size):
                alpha = 1.0 - (updates_done / total_updates_est)
                alpha = max(alpha, 0.0)
                if bool(train_cfg["optim"].get("anneal_lr", True)):
                    lr_now = base_lr * alpha
                    for g in optim.param_groups:
                        g["lr"] = lr_now
                clip_eps_now = base_clip_eps * alpha if bool(train_cfg["loss"].get("anneal_clip_epsilon", True)) else base_clip_eps
                updates_done += 1

                actor_core(sub)
                actor_head(sub)
                value_net(sub)
                squeeze_last1(sub, ("agents", "state_value"))
                act = sub.get(("agents", "action")).long()
                if act.ndim == 4 and act.shape[-1] == 1:
                    act = act.squeeze(-1)

                old_logp = sub.get(("agents", "sample_log_prob"))
                if old_logp.ndim == 4 and old_logp.shape[-1] == 1:
                    old_logp = old_logp.squeeze(-1)

                adv = sub.get(("agents","advantage"))
                m = adv.mean(dim=(0,1), keepdim=True)
                s = adv.std(dim=(0,1), keepdim=True).clamp_min(1e-8)
                adv = (adv - m) / s
                sub.set(("agents","advantage"), adv)

                logits = sub.get(("agents", "logits"))
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act)
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps_now, 1.0 + clip_eps_now) * adv
                policy_loss = -(torch.min(surr1, surr2)).mean()

                v_pred = sub.get(("agents", "state_value"))
                v_targ = sub.get(("agents", "value_target"))
                critic_loss = 0.5 * (v_targ - v_pred).pow(2).mean()

                entropy_loss = -entropy.mean()
                total_loss = policy_loss + critic_coeff * critic_loss + entropy_coeff * entropy_loss

                optim.zero_grad(set_to_none=True)
                total_loss.backward()
                params = (
                    list(actor_core.parameters())
                    + list(actor_head.parameters())
                    + list(critic_core.parameters())
                    + list(critic_head.parameters())
                )
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optim.step()

                last_total_loss = total_loss.detach()
                last_policy_loss = policy_loss.detach()
                last_critic_loss = critic_loss.detach()
                last_entropy = entropy.mean().detach()

        collector.update_policy_weights_()

        reward_mean = float(traj.get(("agents", "reward")).mean().item())
        qoe_mean = float(traj.get(("next", "agents", "observation", "qoe_mean")).mean().item())
        obs = traj.get(("next","agents","observation","obs"))
        print("obs shape", obs.shape)
        print("obs nan?", torch.isnan(obs).any().item())
        print("obs mean per feature", obs.mean(dim=(0,1,2)).cpu().numpy())
        print("obs std per feature", obs.std(dim=(0,1,2)).cpu().numpy())
        print("obs min/max", obs.min().item(), obs.max().item())
        
        if (it + 1) % ckpt_every == 0:
            save_ckpt(ckpt_dir / f"ckpt_iter_{it+1:06d}.pt", policy, value_net, optim, env_cfg, train_cfg, it + 1, device, env)

        if qoe_mean > best_qoe:
            best_qoe = qoe_mean
            save_ckpt(ckpt_dir / "ckpt_best.pt", policy, value_net, optim, env_cfg, train_cfg, it + 1, device, env)

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
                "train/lr": float(optim.param_groups[0]["lr"]),
                "train/clip_eps": float(clip_eps_now),
            }
        )


if __name__ == "__main__":
    train()
    
    


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

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType, set_composite_lp_aggregate
from torch.distributions import Categorical

from torchrl.collectors import SyncDataCollector
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import Compose, InitTracker, Transform, TransformedEnv, ObservationNorm
from torchrl.modules import ProbabilisticActor, LSTMModule

from environment import build_env_base
from logger import wandb_init
import wandb
from torchrl.objectives import ClipPPOLoss

# =========================================================
# Utils
# =========================================================

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


def squeeze_last1(td: TensorDictBase, key):
    if key in td.keys(True, True):
        x = td.get(key)
        if torch.is_tensor(x) and x.ndim >= 1 and x.shape[-1] == 1:
            td.set(key, x.squeeze(-1))


# =========================================================
# Env (PettingZoo ParallelEnv)
# =========================================================

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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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

    def step(self, actions: Dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        act_vec = np.zeros(self.n_edges, dtype=np.int64)
        for i, aid in enumerate(self.area_ids):
            a = int(actions[aid])
            if a < 0 or a > 2:
                raise ValueError(f"Invalid action {a} for agent {aid}, expected 0..2")
            act_vec[i] = a
            
        # RL action drives allocation
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
            return int(self.base_seed + self.episode_id * 1000)
        self.base_seed = int(seed)
        return int(seed)

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

        target_ids_cpu = 2.0

        r = np.zeros(self.n_edges, dtype=np.float32)
        for i, aid in enumerate(self.area_ids):
            r[i] = -abs(float(self.ids_cpu[i]) - target_ids_cpu)

        return r
        
    def _qoe_vec(self) -> np.ndarray:
        qoe = np.asarray(getattr(self.env, "final_qoe", 0.0), dtype=np.float32)
        if qoe.ndim == 0:
            qoe = np.full((self.n_edges,), float(qoe), dtype=np.float32)
        return qoe * 30.0


# =========================================================
# Transforms
# =========================================================

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
        bs = observation_spec.shape
        observation_spec[self.out_key] = UnboundedContinuousTensorSpec(
            shape=(*bs, self.flat_dim),
            dtype=agents_obs.dtype,
            device=agents_obs.device,
        )
        return observation_spec

class AddAgentID(Transform):
    def __init__(self, n_edges: int, in_key=("agents","observation","obs"), out_key=("agents","observation","obs")):
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.n_edges = int(n_edges)
        self.in_key = in_key
        self.out_key = out_key

    def _append_id(self, obs: torch.Tensor) -> torch.Tensor:
        E = self.n_edges
        eye = torch.eye(E, device=obs.device, dtype=obs.dtype)
        if obs.ndim == 3:
            B = obs.shape[0]
            ids = eye.unsqueeze(0).expand(B, E, E)
            return torch.cat([obs, ids], dim=-1)
        if obs.ndim == 4:
            B, T = obs.shape[:2]
            ids = eye.view(1,1,E,E).expand(B, T, E, E)
            return torch.cat([obs, ids], dim=-1)
        raise RuntimeError(f"obs must be 3D or 4D, got {obs.shape}")

    def _call(self, td):
        obs = td.get(self.in_key, None)
        if obs is not None:
            td.set(self.out_key, self._append_id(obs))

        nxt = td.get("next", None)
        if nxt is not None:
            obs2 = nxt.get(self.in_key, None)
            if obs2 is not None:
                nxt.set(self.out_key, self._append_id(obs2))
        return td

    def _reset(self, td, td_reset, **kwargs):
        obs = td_reset.get(self.in_key, None)
        if obs is not None:
            td_reset.set(self.out_key, self._append_id(obs))
        return td_reset

    def transform_observation_spec(self, observation_spec):
        spec = observation_spec[self.in_key]
        new_shape = (*spec.shape[:-1], spec.shape[-1] + self.n_edges)
        observation_spec[self.out_key] = UnboundedContinuousTensorSpec(
            shape=new_shape, dtype=spec.dtype, device=spec.device
        )
        return observation_spec

class WriteRecurrentOutToNext(Transform):
    def __init__(self):
        super().__init__(
            in_keys=[
                ("agents","recurrent_state_h_out"),
                ("agents","recurrent_state_c_out"),
                ("agents","recurrent_state_h_v_out"),
                ("agents","recurrent_state_c_v_out"),
            ],
            out_keys=[],
        )

    def _call(self, td):
        nxt = td.get("next")
        if nxt is None:
            return td

        nxt.set(("agents","recurrent_state_h"), td.get(("agents","recurrent_state_h_out")))
        nxt.set(("agents","recurrent_state_c"), td.get(("agents","recurrent_state_c_out")))

        nxt.set(("agents","recurrent_state_h_v"), td.get(("agents","recurrent_state_h_v_out")))
        nxt.set(("agents","recurrent_state_c_v"), td.get(("agents","recurrent_state_c_v_out")))
        return td

class InitRecurrentState(Transform):
    def __init__(self, n_edges: int, actor_hidden_dim: int, critic_hidden_dim: int):
        super().__init__(
            in_keys=[], 
            out_keys=[
                ("agents", "recurrent_state_h"), ("agents", "recurrent_state_c"),
                ("agents", "recurrent_state_h_out"), ("agents", "recurrent_state_c_out"),
                ("agents", "recurrent_state_h_v"), ("agents", "recurrent_state_c_v"),
                ("agents", "recurrent_state_h_v_out"), ("agents", "recurrent_state_c_v_out"),
            ]
        )
        self.n_edges = int(n_edges)
        self.actor_hidden_dim = int(actor_hidden_dim)
        self.critic_hidden_dim = int(critic_hidden_dim)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        # Bypass the base class zip logic; we only act on reset, not during step
        return td

    def _reset(self, td, td_reset, **kwargs):
        bs = tuple(td_reset.batch_size)
        dev = td_reset.device
        E = self.n_edges
        Ha = self.actor_hidden_dim
        Hc = self.critic_hidden_dim

        # actor
        td_reset.set(("agents", "recurrent_state_h"), torch.zeros((*bs, E, 1, Ha), device=dev))
        td_reset.set(("agents", "recurrent_state_c"), torch.zeros((*bs, E, 1, Ha), device=dev))
        td_reset.set(("agents", "recurrent_state_h_out"), torch.zeros((*bs, E, 1, Ha), device=dev))
        td_reset.set(("agents", "recurrent_state_c_out"), torch.zeros((*bs, E, 1, Ha), device=dev))

        # critic
        td_reset.set(("agents", "recurrent_state_h_v"), torch.zeros((*bs, E, 1, Hc), device=dev))
        td_reset.set(("agents", "recurrent_state_c_v"), torch.zeros((*bs, E, 1, Hc), device=dev))
        td_reset.set(("agents", "recurrent_state_h_v_out"), torch.zeros((*bs, E, 1, Hc), device=dev))
        td_reset.set(("agents", "recurrent_state_c_v_out"), torch.zeros((*bs, E, 1, Hc), device=dev))
        return td_reset

    def transform_observation_spec(self, observation_spec):
        bs = observation_spec.shape
        E = self.n_edges
        Ha = self.actor_hidden_dim
        Hc = self.critic_hidden_dim

        # Register the actor states
        observation_spec[("agents", "recurrent_state_h")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Ha), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_c")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Ha), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_h_out")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Ha), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_c_out")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Ha), device=observation_spec.device)

        # Register the critic states
        observation_spec[("agents", "recurrent_state_h_v")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Hc), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_c_v")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Hc), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_h_v_out")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Hc), device=observation_spec.device)
        observation_spec[("agents", "recurrent_state_c_v_out")] = UnboundedContinuousTensorSpec(shape=(*bs, E, 1, Hc), device=observation_spec.device)
        
        return observation_spec

    
# =========================================================
# Networks (Feature + LSTMModule + heads)
# =========================================================

class FeatureNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AgentRecurrentCore(nn.Module):
    in_keys = [
        ("agents", "observation", "obs"),
        "is_init",
        ("agents", "recurrent_state_h"),
        ("agents", "recurrent_state_c"),
    ]
    out_keys = [
        ("agents", "features"),
        ("agents", "recurrent_state_h_out"),
        ("agents", "recurrent_state_c_out"),
    ]    
    def __init__(self, n_edges: int, obs_dim: int, hidden_dim: int, device: str):
        super().__init__()
        self.n_edges = int(n_edges)
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)

        self.feature = FeatureNet(self.obs_dim, self.hidden_dim).to(device)
        self.lstm = LSTMModule(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            in_key="x_t",
            out_key="x_t",
            device=device,
        )

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("agents","observation","obs"))
        is_init = td.get("is_init")

        step_mode = (obs.ndim == 3)
        if step_mode:
            obs = obs.unsqueeze(1)
            
        B, T, E, D = obs.shape
        H = self.hidden_dim

        # Secure is_init formatting
        if is_init is None:
            is_init_be = torch.zeros((B * E, T, 1), device=obs.device, dtype=torch.bool)
        else:
            if is_init.ndim == 2:
                is_init = is_init.unsqueeze(-1)
            if step_mode:
                is_init_be = is_init.view(B, 1, 1).expand(B, E, 1).reshape(B * E, 1, 1)
            else:
                is_init_be = is_init.view(B, 1, T).expand(B, E, T).reshape(B * E, T, 1)

        # --- FIX: Transpose Time and Agents BEFORE reshaping to preserve history ---
        obs_be = obs.transpose(1, 2).contiguous().reshape(B * E, T, D)
        feats_be = self.feature(obs_be)

        h = td.get(("agents","recurrent_state_h")) # Use _v for Critic
        c = td.get(("agents","recurrent_state_c")) # Use _v for Critic

        if h.ndim == 5:
            h0 = h[:, 0]
            c0 = c[:, 0]
        else:
            h0 = h
            c0 = c

        h_be = h0.reshape(B * E, 1, H).contiguous()
        c_be = c0.reshape(B * E, 1, H).contiguous()

        td_be = TensorDict(
            {"recurrent_state_h": h_be, "recurrent_state_c": c_be},
            batch_size=[B * E],
            device=feats_be.device,
        )

        out = []
        for t in range(T):
            td_be.set("is_init", is_init_be[:, t])
            td_be.set("x_t", feats_be[:, t]) # Use v_t for Critic
            self.lstm(td_be)
            out.append(td_be.get("x_t"))     # Use v_t for Critic
            
            # --- FIX: Carry state forward through BPTT ---
            nxt = td_be.get("next")
            td_be.set("recurrent_state_h", nxt.get("recurrent_state_h"))
            td_be.set("recurrent_state_c", nxt.get("recurrent_state_c"))
            td_be.del_("next") 

        # --- FIX: Stack, un-flatten, and transpose back to [B, T, E, H] ---
        feats_be2 = torch.stack(out, dim=1)
        feats = feats_be2.view(B, E, T, H).transpose(1, 2).contiguous()

        if step_mode:
            td.set(("agents","features"), feats[:, 0]) # Use vf_features for Critic
            td.set(("agents","recurrent_state_h_out"), td_be.get("recurrent_state_h").view(B, E, 1, H))
            td.set(("agents","recurrent_state_c_out"), td_be.get("recurrent_state_c").view(B, E, 1, H))
        else:
            td.set(("agents","features"), feats) # Use vf_features for Critic

        return td

class CriticRecurrentCore(nn.Module):
    in_keys = [
        ("agents", "observation", "obs"),
        "is_init",
        ("agents", "recurrent_state_h_v"),
        ("agents", "recurrent_state_c_v"),
    ]
    out_keys = [
        ("agents", "vf_features"),
        ("agents", "recurrent_state_h_v_out"),
        ("agents", "recurrent_state_c_v_out"),
    ]
    def __init__(self, n_edges: int, obs_dim: int, hidden_dim: int, device: str):
        super().__init__()
        self.n_edges = int(n_edges)
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)

        self.feature = FeatureNet(self.obs_dim, self.hidden_dim).to(device)
        self.lstm = LSTMModule(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            in_key="v_t",
            out_key="v_t",
            device=device,
        )

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("agents","observation","obs"))
        is_init = td.get("is_init")

        step_mode = (obs.ndim == 3)
        if step_mode:
            obs = obs.unsqueeze(1)
            
        B, T, E, D = obs.shape
        H = self.hidden_dim

        # Secure is_init formatting
        if is_init is None:
            is_init_be = torch.zeros((B * E, T, 1), device=obs.device, dtype=torch.bool)
        else:
            if is_init.ndim == 2:
                is_init = is_init.unsqueeze(-1)
            if step_mode:
                is_init_be = is_init.view(B, 1, 1).expand(B, E, 1).reshape(B * E, 1, 1)
            else:
                is_init_be = is_init.view(B, 1, T).expand(B, E, T).reshape(B * E, T, 1)

        # Transpose Time and Agents BEFORE reshaping to preserve history
        obs_be = obs.transpose(1, 2).contiguous().reshape(B * E, T, D)
        feats_be = self.feature(obs_be)

        # --- FIX: Fetch the Critic's specific _v states! ---
        h = td.get(("agents","recurrent_state_h_v")) 
        c = td.get(("agents","recurrent_state_c_v")) 

        if h.ndim == 5:
            h0 = h[:, 0]
            c0 = c[:, 0]
        else:
            h0 = h
            c0 = c

        h_be = h0.reshape(B * E, 1, H).contiguous()
        c_be = c0.reshape(B * E, 1, H).contiguous()

        # LSTMModule natively looks for "recurrent_state_h", so we map our Critic states 
        # into standard names for the sub-dictionary
        td_be = TensorDict(
            {"recurrent_state_h": h_be, "recurrent_state_c": c_be},
            batch_size=[B * E],
            device=feats_be.device,
        )

        out = []
        for t in range(T):
            td_be.set("is_init", is_init_be[:, t])
            td_be.set("v_t", feats_be[:, t]) 
            self.lstm(td_be)
            out.append(td_be.get("v_t"))     
            
            # Carry state forward through BPTT
            nxt = td_be.get("next")
            td_be.set("recurrent_state_h", nxt.get("recurrent_state_h"))
            td_be.set("recurrent_state_c", nxt.get("recurrent_state_c"))
            td_be.del_("next") 

        # Stack, un-flatten, and transpose back to [B, T, E, H]
        feats_be2 = torch.stack(out, dim=1)
        feats = feats_be2.view(B, E, T, H).transpose(1, 2).contiguous()

        if step_mode:
            td.set(("agents","vf_features"), feats[:, 0]) 
            # --- FIX: Write back to the Critic's specific _v_out keys! ---
            td.set(("agents","recurrent_state_h_v_out"), td_be.get("recurrent_state_h").view(B, E, 1, H))
            td.set(("agents","recurrent_state_c_v_out"), td_be.get("recurrent_state_c").view(B, E, 1, H))
        else:
            td.set(("agents","vf_features"), feats) 

        return td

class SqueezeLast(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1)    
    



# =========================================================
# PPO / GAE
# =========================================================

@torch.no_grad()
def compute_gae_inplace(traj: TensorDictBase, gamma: float, lmbda: float, n_edges: int):
    reward = traj.get(("agents", "reward"))
    done = traj.get(("agents", "done")).to(torch.bool)
    terminated = traj.get(("agents", "terminated")).to(torch.bool)
    values = traj.get(("agents", "state_value"))
    next_values = traj.get("next").get(("agents", "state_value"))

    # Force trailing dimension so ClipPPOLoss processes uniform shapes
    if reward.ndim == 3: reward = reward.unsqueeze(-1)
    if done.ndim == 3: done = done.unsqueeze(-1)
    if terminated.ndim == 3: terminated = terminated.unsqueeze(-1)
    if values.ndim == 3: values = values.unsqueeze(-1)
    if next_values.ndim == 3: next_values = next_values.unsqueeze(-1)

    not_end = ~(done | terminated)
    not_end = not_end.to(values.dtype)

    B, T, E, _ = reward.shape

    adv = torch.zeros_like(reward)
    last_gae = torch.zeros((B, E, 1), device=reward.device, dtype=reward.dtype)

    for t in reversed(range(T)):
        delta = reward[:, t] + gamma * next_values[:, t] * not_end[:, t] - values[:, t]
        last_gae = delta + gamma * lmbda * not_end[:, t] * last_gae
        adv[:, t] = last_gae

    traj.set(("agents", "advantage"), adv)
    traj.set(("agents", "value_target"), adv + values)


def valid_sequence_minibatches(traj: TensorDictBase, seq_len: int, minibatch_size: int):
    B, T = traj.batch_size
    if T < seq_len:
        raise RuntimeError(f"T={T} < seq_len={seq_len}")

    done = traj.get(("agents", "done")).to(torch.bool)
    while done.ndim > 3:
        done = done.squeeze(-1)
    done_any = done.any(dim=-1)

    max_t0 = T - seq_len
    csum = torch.cumsum(done_any.to(torch.int32), dim=1)

    left = csum[:, : max_t0 + 1]
    right = csum[:, seq_len - 1 : seq_len - 1 + (max_t0 + 1)]
    prev_left = torch.cat([torch.zeros(B, 1, device=traj.device, dtype=csum.dtype), left[:, :-1]], dim=1)
    window_sum = right - prev_left
    valid = window_sum == 0

    valid_idx = valid.nonzero(as_tuple=False)
    if valid_idx.numel() == 0:
        all_b = torch.arange(B, device=traj.device).repeat_interleave(max_t0 + 1)
        all_t0 = torch.arange(max_t0 + 1, device=traj.device).repeat(B)
        valid_idx = torch.stack([all_b, all_t0], dim=1)

    perm = torch.randperm(valid_idx.shape[0], device=traj.device)
    valid_idx = valid_idx[perm]

    for start in range(0, valid_idx.shape[0], minibatch_size):
        idx = valid_idx[start : start + minibatch_size]
        b_idx = idx[:, 0]
        t0 = idx[:, 1]
        slices = [traj[b_idx, t0 + k] for k in range(seq_len)]
        yield torch.stack(slices, dim=1).to_tensordict()


# =========================================================
# Builders
# =========================================================

def make_wrapped_env(cfg_path: str, seed: int, decision_interval: int):
    pz = EdgeIDSParallelEnv(cfg_path=cfg_path, seed=seed, decision_interval=decision_interval)
    group_map = {"agents": list(pz.possible_agents)}
    return PettingZooWrapper(pz, categorical_actions=True, group_map=group_map)

def build_env_stack(env_cfg: dict, train_cfg: dict, cfg_path: str, num_envs: int):
    decision_interval = int(env_cfg["globals"]["decision_interval"])

    base = make_wrapped_env(cfg_path=cfg_path, seed=int(env_cfg["run"]["seed"]), decision_interval=decision_interval)
    obs_spec = base.observation_spec[("agents", "observation", "obs")]
    n_edges = int(obs_spec.shape[-2])
    obs_dim = int(obs_spec.shape[-1])
    obs_dim_with_id = obs_dim + n_edges

    def make_one(i: int):
        def _make():
            return make_wrapped_env(cfg_path=cfg_path, seed=1000 + i, decision_interval=decision_interval)
        return _make

    penv = ParallelEnv(num_envs, [make_one(i) for i in range(num_envs)], device="cpu")

    hidden_dim = int(train_cfg["model"]["hidden_dim"])
    actor_h = hidden_dim
    critic_h = hidden_dim * 2

    transforms: List[Transform] = [
        InitTracker(),
        AddAgentID(n_edges=n_edges),
        InitRecurrentState(n_edges=n_edges, actor_hidden_dim=actor_h, critic_hidden_dim=critic_h),
        WriteRecurrentOutToNext(),
    ]

    env = TransformedEnv(penv, Compose(*transforms))
    td0 = env.reset()
    print("obs step shape:", td0.get(("agents","observation","obs")).shape)

    return env, n_edges, obs_dim_with_id


# =========================================================
# Train
# =========================================================

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

    # --- actor ---
    actor_core = AgentRecurrentCore(
        n_edges=n_edges,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        device=device,
    )   
    actor_head = TensorDictModule(
        nn.Linear(hidden_dim, n_actions).to(device),
        in_keys=[("agents", "features")],
        out_keys=[("agents", "logits")],
    )
    policy = ProbabilisticActor(
        module=TensorDictSequential(actor_core, actor_head),
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    critic_core = CriticRecurrentCore(n_edges=n_edges, obs_dim=obs_dim, hidden_dim=hidden_dim*2, device=device)

    critic_head = TensorDictModule(
        nn.Linear(hidden_dim * 2, 1).to(device),
        in_keys=[("agents","vf_features")],
        out_keys=[("agents","state_value")],
    )

    value_net = TensorDictSequential(critic_core, critic_head)

    # Combining actor and critic into a single Sequential ensures the Critic is also run 
    # during collection to correctly track and propagate its recurrent states!
    collector_policy = TensorDictSequential(policy, value_net)

    optim = torch.optim.Adam(
        list(actor_core.parameters())
        + list(actor_head.parameters())
        + list(critic_core.parameters())
        + list(critic_head.parameters()),
        lr=float(train_cfg["optim"]["lr"]),
        weight_decay=float(train_cfg["optim"]["weight_decay"]),
        eps=float(train_cfg["optim"]["eps"]),
    )
    # --- Initialize TorchRL's built-in PPO Loss ---
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_net,
        clip_epsilon=train_cfg["loss"]["clip_epsilon"],
        entropy_bonus=True,
        entropy_coef=train_cfg["loss"]["entropy_coeff"],
        critic_coef=train_cfg["loss"]["critic_coeff"],
        loss_critic_type=train_cfg["loss"].get("loss_critic_type", "l2"),
        normalize_advantage=True,
    )
    
    # Map the nested keys so ClipPPOLoss knows where to find our multi-agent data
    loss_module.set_keys(
        advantage=("agents", "advantage"),
        value_target=("agents", "value_target"),
        value=("agents", "state_value"),
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
    )    

    if resume_ckpt:
        state = torch.load(resume_ckpt, map_location=device)
        policy.load_state_dict(state["policy"])
        value_net.load_state_dict(state["value"])
        optim.load_state_dict(state["optim"])
        try:
            env.load_state_dict(state["obsnorm"])
        except Exception:
            pass

    frames_per_batch = int(train_cfg["collector"].get("frames_per_batch", decisions_per_episode * num_envs))
    total_frames = int(train_cfg["collector"]["total_frames"])

    collector = SyncDataCollector(
        env,
        policy=collector_policy, # Using the combined policy here
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

    ckpt_dir = Path("checkpoints") / run.name
    ckpt_every = 50
    best_qoe = -1e9

    total_updates_est = max(1, (total_frames // frames_per_batch) * ppo_epochs * math.ceil(frames_per_batch / minibatch_size))
    updates_done = 0

    for it, batch in enumerate(collector):
        traj = batch.clone(False)

        # Move root keys
        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated")]:
            nk = ("next",) + k
            if k not in traj.keys(True, True) and nk in traj.keys(True, True):
                traj.set(k, traj.get(nk))

        if ("agents","done") in traj.keys(True, True):
            traj.set(("agents", "done"), traj.get(("agents", "done")).to(torch.bool))
        if ("agents","terminated") in traj.keys(True, True):
            traj.set(("agents", "terminated"), traj.get(("agents", "terminated")).to(torch.bool))
        if ("agents","done") in traj.get("next").keys(True, True):
            traj.get("next").set(("agents", "done"), traj.get("next").get(("agents","done")).to(torch.bool))
        if ("agents","terminated") in traj.get("next").keys(True, True):
            traj.get("next").set(("agents", "terminated"), traj.get("next").get(("agents","terminated")).to(torch.bool))

        with torch.no_grad():
            value_net(traj)
            value_net(traj.get("next"))
            compute_gae_inplace(traj, gamma=gamma, lmbda=gae_lambda, n_edges=n_edges)

        last_total_loss = last_policy_loss = last_critic_loss = last_entropy = None
        seq_len = int(train_cfg["loss"].get("seq_len", 32))

        for _ in range(ppo_epochs):
            for sub in valid_sequence_minibatches(traj, seq_len=seq_len, minibatch_size=minibatch_size):
                
                alpha = 1.0 - (updates_done / total_updates_est)
                alpha = max(alpha, 0.0)
                if bool(train_cfg["optim"].get("anneal_lr", True)):
                    lr_now = base_lr * alpha
                    for g in optim.param_groups:
                        g["lr"] = lr_now
                if bool(train_cfg["loss"].get("anneal_clip_epsilon", True)):
                    loss_module.clip_epsilon = base_clip_eps * alpha
                    
                updates_done += 1

                # --- FIX: Removed manual forward passes. 
                # ClipPPOLoss handles the LSTM BPTT unrolling natively! ---
                loss_dict = loss_module(sub)

                policy_loss = loss_dict["loss_objective"]
                critic_loss = loss_dict["loss_critic"]
                entropy_loss = loss_dict["loss_entropy"]
                
                total_loss = policy_loss + critic_loss + entropy_loss

                optim.zero_grad(set_to_none=True)
                total_loss.backward()
                
                params = (
                    list(actor_core.parameters())
                    + list(actor_head.parameters())
                    + list(critic_core.parameters())
                    + list(critic_head.parameters())
                )
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optim.step()

        reward_mean = float(traj.get(("agents", "reward")).mean().item())
        qoe_mean = float(traj.get(("next", "agents", "observation", "qoe_mean")).mean().item())
        obs = traj.get(("next","agents","observation","obs"))
        print("obs shape", obs.shape)
        print("obs nan?", torch.isnan(obs).any().item())
        print("obs mean per feature", obs.mean(dim=(0,1,2)).cpu().numpy())
        print("obs std per feature", obs.std(dim=(0,1,2)).cpu().numpy())
        print("obs min/max", obs.min().item(), obs.max().item())
        
        if (it + 1) % ckpt_every == 0:
            save_ckpt(ckpt_dir / f"ckpt_iter_{it+1:06d}.pt", policy, value_net, optim, env_cfg, train_cfg, it + 1, device, env)

        if qoe_mean > best_qoe:
            best_qoe = qoe_mean
            save_ckpt(ckpt_dir / "ckpt_best.pt", policy, value_net, optim, env_cfg, train_cfg, it + 1, device, env)

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
                "train/lr": float(optim.param_groups[0]["lr"]),
            }
        )


if __name__ == "__main__":
    train()    