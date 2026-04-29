from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Type

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv as PZooParallelEnv

from tensordict import TensorDictBase, TensorDict
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

from environment import build_env_base, VA_CPU_RESERVE
from logger import wandb_init, wandb_log_obs_steps
import wandb
import argparse

N_TEMPORAL = 2  # local_num_req, attack_in_rate fed through LSTM; rest bypass as static

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
        ids_cpu_min: float = VA_CPU_RESERVE,
        threshold: float = 0.20,
        alpha: float = 0.10,
        beta: float = 0.20,
        gamma_r: float = 0.12,
        n_actions: int = 9,
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
        self.n_actions = int(n_actions)

        _cfg_raw = Path(cfg_path).read_text(encoding="utf-8")
        _cfg = yaml.safe_load(_cfg_raw)
        _rew = _cfg.get("globals", {}).get("reward", {})
        self.threshold = float(_rew.get("q_th", threshold))
        self.alpha     = float(_rew["alpha_inv"]) if "alpha_inv" in _rew else float(alpha)
        self.beta      = float(_rew["beta_inv"])  if "beta_inv"  in _rew else float(beta)
        self.gamma_r   = float(_rew["gamma_inv"]) if "gamma_inv" in _rew else float(gamma_r)

        self.obs_keys = [
            "local_num_req",
            "attack_in_rate",
            "ema_mom",
            "cpu_to_ids_ratio",
            "ids_cpu_utilization",
        ]
        if self.n_edges > 1:
            self.obs_keys += [
                "neighbor_ids_util",
                "neighbor_delta",
                "neighbor_atk_rate",
            ]

        self.obs_keys += ["prev_slo_vio"]

        self.obs_keys += [
            "transition_ticks_norm",
            "delta_in_flight_norm",
            "queue_ahead_norm",
        ]
        self.obs_dim = len(self.obs_keys)

        self.scaling_time_steps = list(_cfg["globals"].get("scaling_time_step", [300, 450, 498, 544]))
        self.scaling_quanta = [0.5, 1.0, 1.5, 2.0]

        self._obs_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
                "qoe_mean": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "reward_lambda_res": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "reward_benign_col_dmg": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "reward_qoe_penalty": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "qoe_vio_rate": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self._act_space = spaces.Discrete(self.n_actions)

        self.ids_cpu = np.asarray([e.ids_cpu for e in self.env.edge_areas], dtype=np.float32)
        self.ids_cpu_settled = self.ids_cpu.copy()
        self.ids_cpu_target = self.ids_cpu.copy()
        self.transition_ticks_remaining = np.zeros(self.n_edges, dtype=np.int32)
        self.transition_ticks_total = np.ones(self.n_edges, dtype=np.int32)

    def _compute_step_reward(
        self,
        lres: np.ndarray,
        bcd: np.ndarray,
        qsf: np.ndarray,
        vio: np.ndarray,
        attack_in: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _lookup_scaling_duration(self, magnitude: float) -> int:
        for i, q in enumerate(self.scaling_quanta):
            if magnitude <= q + 1e-9:
                return self.scaling_time_steps[i]
        return self.scaling_time_steps[-1]

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

        self.ids_cpu_settled = self.ids_cpu.copy()
        self.ids_cpu_target = self.ids_cpu.copy()
        self.transition_ticks_remaining = np.zeros(self.n_edges, dtype=np.int32)
        self.transition_ticks_total = np.ones(self.n_edges, dtype=np.int32)

        obs_mat = self._build_observation()
        qoe = self._qoe_vec()

        observations = {
            aid: {
                "obs": obs_mat[i].copy(),
                "qoe_mean": np.array([qoe[i]], dtype=np.float32),
                "reward_lambda_res": np.zeros(1, dtype=np.float32),
                "reward_benign_col_dmg": np.zeros(1, dtype=np.float32),
                "reward_qoe_penalty": np.zeros(1, dtype=np.float32),
                "qoe_vio_rate": np.zeros(1, dtype=np.float32),
            }
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
            if a < 0 or a >= self.n_actions:
                raise ValueError(f"Invalid action {a} for agent {aid}, expected 0..{self.n_actions-1}")
            act_vec[i] = a

        delta_cmd = (act_vec.astype(np.float32) - ((self.n_actions - 1) / 2.0)) * self.scale_step

        for i in range(self.n_edges):
            ids_cpu_max = float(self.env.edge_areas[i].budget.cpu - VA_CPU_RESERVE)
            prev = self.ids_cpu[i].copy()
            
            _settled = float(self.ids_cpu_settled[i])
            _max_q   = float(self.scaling_quanta[-1])
            self.ids_cpu[i] = np.clip(
                self.ids_cpu[i] + delta_cmd[i], 
                max(self.ids_cpu_min, _settled - _max_q), 
                min(ids_cpu_max, _settled + _max_q)
            )
            delta_eff = float(self.ids_cpu[i] - prev)

            if self.transition_ticks_remaining[i] <= 0 and abs(delta_eff) > 1e-9:
                self.ids_cpu_target[i] = self.ids_cpu[i]
                gap = abs(float(self.ids_cpu_target[i]) - float(self.ids_cpu_settled[i]))
                dur = self._lookup_scaling_duration(gap)
                self.transition_ticks_total[i]     = dur
                self.transition_ticks_remaining[i] = dur

        ids_cpu_eff = self.ids_cpu_settled.copy()
        step_overhead = np.zeros(self.n_edges, dtype=np.float32)
        for i in range(self.n_edges):
            if self.transition_ticks_remaining[i] > 0:
                delta_to_settled = float(self.ids_cpu_target[i]) - float(self.ids_cpu_settled[i])
                if abs(delta_to_settled) > 1e-9:
                    step_overhead[i] = -delta_to_settled

        total_rew  = np.zeros(self.n_edges, dtype=np.float32)
        total_lres = np.zeros(self.n_edges, dtype=np.float32)
        total_bcd  = np.zeros(self.n_edges, dtype=np.float32)
        total_qsf  = np.zeros(self.n_edges, dtype=np.float32)
        total_vio  = np.zeros(self.n_edges, dtype=np.float32)
        active_ticks = np.zeros(self.n_edges, dtype=np.int32)

        terminated_flag = False
        steps = 0

        for _ in range(self.decision_interval):
            for i in range(self.n_edges):
                if self.transition_ticks_remaining[i] > 0:
                    self.transition_ticks_remaining[i] -= 1
                    if self.transition_ticks_remaining[i] == 0:
                        self.ids_cpu_settled[i] = self.ids_cpu_target[i]
                        queued_delta = float(self.ids_cpu[i]) - float(self.ids_cpu_settled[i])
                        if abs(queued_delta) > 1e-9:
                            self.ids_cpu_target[i] = self.ids_cpu[i]
                            gap = abs(queued_delta)
                            dur = self._lookup_scaling_duration(gap)
                            self.transition_ticks_total[i]     = dur
                            self.transition_ticks_remaining[i] = dur
                            new_d = float(self.ids_cpu_target[i]) - float(self.ids_cpu_settled[i])
                            step_overhead[i] = -new_d if abs(new_d) > 1e-9 else 0.0
                            ids_cpu_eff[i] = float(self.ids_cpu_settled[i])
                        else:
                            ids_cpu_eff[i] = float(self.ids_cpu_settled[i])
                            step_overhead[i] = 0.0

            self.env.step(ids_cpu_eff.tolist(), step_overhead.tolist())

            if len(self.env.history) >= self.n_edges:
                last_block = self.env.history[-self.n_edges:]
                qoe        = np.asarray([float(m.qoe_mean)        for m in last_block], dtype=np.float32)
                bcd        = np.asarray([float(m.benign_col_dmg)   for m in last_block], dtype=np.float32)
                attack_in  = np.asarray([float(m.attack_in_rate)   for m in last_block], dtype=np.float32)
                attack_drop = np.asarray([float(m.attack_drop_rate) for m in last_block], dtype=np.float32)

                attack_pass = np.maximum(0.0, attack_in - attack_drop)
                lres = np.where(attack_in > 1e-6, attack_pass / attack_in, 0.0).astype(np.float32)
                qsf  = np.maximum(0.0, self.threshold - qoe) / max(self.threshold, 1e-6)
                vio  = (qoe < self.threshold).astype(np.float32)

                rew_d, lres_d, bcd_d, qsf_d, vio_d = self._compute_step_reward(
                    lres, bcd, qsf, vio, attack_in
                )
                total_rew  += rew_d
                total_lres += lres_d
                total_bcd  += bcd_d
                total_qsf  += qsf_d
                total_vio  += vio_d
                active_ticks += (attack_in > 1e-6).astype(np.int32)

            steps += 1
            if self.env.t >= self.env.t_max:
                terminated_flag = True
                break

        rew_agents = (total_rew  / max(1, steps)).astype(np.float32, copy=False)
        info_lres  = np.where(active_ticks > 0, total_lres / np.maximum(active_ticks, 1), 0.0).astype(np.float32)
        info_bcd   = (total_bcd  / max(1, steps)).astype(np.float32, copy=False)
        info_qsf   = (total_qsf  / max(1, steps)).astype(np.float32, copy=False)
        info_vio   = (total_vio  / max(1, steps)).astype(np.float32, copy=False)

        rewards      = {aid: float(rew_agents[i]) for i, aid in enumerate(self.area_ids)}
        terminations = {aid: bool(terminated_flag) for aid in self.area_ids}
        truncations  = {aid: False for aid in self.area_ids}
        infos = {
            aid: {
                "reward_lambda_res":     float(info_lres[i]),
                "reward_benign_col_dmg": float(info_bcd[i]),
                "reward_qoe_penalty":    float(info_qsf[i]),
                "qoe_vio_rate":          float(info_vio[i]),
            }
            for i, aid in enumerate(self.area_ids)
        }

        if terminated_flag:
            self.agents = []

        obs_mat = self._build_observation()
        qoe = self._qoe_vec()
        observations = {
            aid: {
                "obs": obs_mat[i].copy(),
                "qoe_mean": np.array([qoe[i]], dtype=np.float32),
                "reward_lambda_res":     np.array([info_lres[i]], dtype=np.float32),
                "reward_benign_col_dmg": np.array([info_bcd[i]], dtype=np.float32),
                "reward_qoe_penalty":    np.array([info_qsf[i]], dtype=np.float32),
                "qoe_vio_rate":          np.array([info_vio[i]], dtype=np.float32),
            }
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

        # 1. Local base metrics
        base_keys = ["local_num_req", "attack_in_rate", "ema_mom", "cpu_to_ids_ratio", "ids_cpu_utilization"]
        for i, area_id in enumerate(self.area_ids):
            g = df[df["area_id"] == area_id]
            if g.empty:
                continue
            for k in base_keys:
                if k not in self.obs_keys: continue
                idx = self.obs_keys.index(k)
                vals = g[k].values
                if k == "cpu_to_ids_ratio":
                    obs[i, idx] = float(vals[-1])
                elif k == "ema_mom":
                    vals_nz = vals[vals != 0.0]
                    if len(vals_nz) == 0:
                        obs[i, idx] = 0.0
                    else:
                        obs[i, idx] = float(np.mean(vals_nz))
                else:
                    obs[i, idx] = float(np.mean(vals))

        # 2. Neighbor metrics (only if present)
        if self.n_edges > 1:
            edge_ids_util: Dict[str, float] = {}
            edge_atk_rate: Dict[str, float] = {}
            for area_id in self.area_ids:
                g = df[df["area_id"] == area_id]
                if g.empty:
                    edge_ids_util[area_id] = 0.0
                    edge_atk_rate[area_id] = 0.0
                else:
                    edge_ids_util[area_id] = float(np.clip(np.mean(g["ids_cpu_utilization"].values), 0.0, 1.0))
                    edge_atk_rate[area_id] = float(np.mean(g["attack_in_rate"].values))

            max_delta = self.scale_step * (self.n_actions - 1) / 2.0
            for i, area_id in enumerate(self.area_ids):
                nbr_utils: List[float] = []
                nbr_deltas: List[float] = []
                nbr_atk: List[float] = []
                for j in range(self.n_edges):
                    if j == i:
                        continue
                    other_id = self.area_ids[j]
                    nbr_utils.append(edge_ids_util[other_id])
                    nbr_atk.append(edge_atk_rate[other_id])
                    delta = float(self.ids_cpu_target[j]) - float(self.ids_cpu_settled[j])
                    nbr_deltas.append(float(np.clip(delta / max(max_delta, 1e-6), -1.0, 1.0)))

                obs[i, self.obs_keys.index("neighbor_ids_util")] = float(np.mean(nbr_utils))  if nbr_utils  else 0.0
                obs[i, self.obs_keys.index("neighbor_delta")]    = float(np.mean(nbr_deltas)) if nbr_deltas else 0.0
                obs[i, self.obs_keys.index("neighbor_atk_rate")] = float(np.mean(nbr_atk))    if nbr_atk    else 0.0

        # 2b. SLO violation flag (local signal, valid for single- and multi-edge)
        for i, area_id in enumerate(self.area_ids):
            g = df[df["area_id"] == area_id]
            if not g.empty:
                last_qoe = float(g["qoe_mean"].values[-1])
                threshold = float(self.env.edge_areas[i].slo_threshold)
                obs[i, self.obs_keys.index("prev_slo_vio")] = 1.0 if last_qoe < threshold else 0.0

        # 3. Scaling metrics (always present)
        max_dur = float(self.scaling_time_steps[-1])
        max_delta = self.scale_step * (self.n_actions - 1) / 2.0
        for i in range(self.n_edges):
            obs[i, self.obs_keys.index("transition_ticks_norm")] = float(self.transition_ticks_remaining[i]) / max(max_dur, 1.0)
            delta_in_flight = float(self.ids_cpu_target[i]) - float(self.ids_cpu_settled[i])
            obs[i, self.obs_keys.index("delta_in_flight_norm")] = float(np.clip(delta_in_flight / max(max_delta, 1e-6), -1.0, 1.0))
            queue_ahead = float(self.ids_cpu[i]) - float(self.ids_cpu_target[i])
            obs[i, self.obs_keys.index("queue_ahead_norm")] = float(np.clip(queue_ahead / max(max_delta, 1e-6), -1.0, 1.0))

        return obs

    def _qoe_vec(self) -> np.ndarray:
        if self.env.history:
            last_block = self.env.history[-self.n_edges:]
            df = pd.DataFrame([m.__dict__ for m in last_block])
            qoe = np.zeros(self.n_edges, dtype=np.float32)
            for i, aid in enumerate(self.area_ids):
                g = df[df["area_id"] == aid]
                if not g.empty:
                    qoe[i] = float(g["qoe_mean"].iloc[-1])
            return qoe
        return np.zeros(self.n_edges, dtype=np.float32)


class EdgeIDSCMAParallelEnv(EdgeIDSParallelEnv):
    """Centralised Multi-Agent (CMA) reward variant."""
    def _compute_step_reward(self, lres, bcd, qsf, vio, attack_in):
        active = attack_in > 1e-6
        r_lres = float(np.mean(lres[active])) if np.any(active) else 0.0
        r_bcd  = float(np.mean(bcd))
        r_qsf  = float(np.mean(qsf))
        r_vio  = float(np.mean(vio))
        rew_scalar = -(self.alpha * r_qsf + self.beta * r_lres + self.gamma_r * r_bcd)
        E = len(lres)
        return (
            np.full(E, rew_scalar, dtype=np.float32),
            np.full(E, r_lres,     dtype=np.float32),
            np.full(E, r_bcd,      dtype=np.float32),
            np.full(E, r_qsf,      dtype=np.float32),
            np.full(E, r_vio,      dtype=np.float32),
        )


class EdgeIDSIMAParallelEnv(EdgeIDSParallelEnv):
    """Independent Multi-Agent (IMA) reward variant."""
    def _compute_step_reward(self, lres, bcd, qsf, vio, attack_in):
        rew = -(self.alpha * qsf + self.beta * lres + self.gamma_r * bcd)
        return rew, lres, bcd, qsf, vio



# =========================================================
# Transforms
# =========================================================
class AddAgentID(Transform):
    def __init__(
        self, n_edges: int, in_key=("agents", "observation", "obs"), out_key=("agents", "observation", "obs")
    ):
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.n_edges = int(n_edges)
        self.in_key = in_key
        self.out_key = out_key

    def _append_id(self, obs: torch.Tensor) -> torch.Tensor:
        e = self.n_edges
        eye = torch.eye(e, device=obs.device, dtype=obs.dtype)
        if obs.ndim == 3:
            b = obs.shape[0]
            ids = eye.unsqueeze(0).expand(b, e, e)
            return torch.cat([obs, ids], dim=-1)
        if obs.ndim == 4:
            b, t = obs.shape[:2]
            ids = eye.view(1, 1, e, e).expand(b, t, e, e)
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
                ("agents", "recurrent_state_h_out"),
                ("agents", "recurrent_state_c_out"),
                ("agents", "recurrent_state_h_v_out"),
                ("agents", "recurrent_state_c_v_out"),
            ],
            out_keys=[],
        )

    def _call(self, td):
        nxt = td.get("next", None)
        if nxt is None: return td
        if ("agents", "recurrent_state_h_out") in td.keys(True, True):
            nxt.set(("agents", "recurrent_state_h"), td.get(("agents", "recurrent_state_h_out")))
            nxt.set(("agents", "recurrent_state_c"), td.get(("agents", "recurrent_state_c_out")))
        if ("agents", "recurrent_state_h_v_out") in td.keys(True, True):
            nxt.set(("agents", "recurrent_state_h_v"), td.get(("agents", "recurrent_state_h_v_out")))
            nxt.set(("agents", "recurrent_state_c_v"), td.get(("agents", "recurrent_state_c_v_out")))
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
            ],
        )
        self.n_edges = int(n_edges)
        self.actor_hidden_dim = int(actor_hidden_dim)
        self.critic_hidden_dim = int(critic_hidden_dim)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        return td

    def _reset(self, td, td_reset, **kwargs):
        bs = tuple(td_reset.batch_size)
        dev = td_reset.device
        e, ha, hc = self.n_edges, self.actor_hidden_dim, self.critic_hidden_dim

        td_reset.set(("agents", "recurrent_state_h"), torch.zeros((*bs, e, 1, ha), device=dev))
        td_reset.set(("agents", "recurrent_state_c"), torch.zeros((*bs, e, 1, ha), device=dev))
        td_reset.set(("agents", "recurrent_state_h_out"), torch.zeros((*bs, e, 1, ha), device=dev))
        td_reset.set(("agents", "recurrent_state_c_out"), torch.zeros((*bs, e, 1, ha), device=dev))

        td_reset.set(("agents", "recurrent_state_h_v"), torch.zeros((*bs, e, 1, hc), device=dev))
        td_reset.set(("agents", "recurrent_state_c_v"), torch.zeros((*bs, e, 1, hc), device=dev))
        td_reset.set(("agents", "recurrent_state_h_v_out"), torch.zeros((*bs, e, 1, hc), device=dev))
        td_reset.set(("agents", "recurrent_state_c_v_out"), torch.zeros((*bs, e, 1, hc), device=dev))
        return td_reset

    def transform_observation_spec(self, observation_spec):
        bs = observation_spec.shape
        e, ha, hc = self.n_edges, self.actor_hidden_dim, self.critic_hidden_dim
        for key, dim in [
            ("recurrent_state_h", ha), ("recurrent_state_c", ha), 
            ("recurrent_state_h_out", ha), ("recurrent_state_c_out", ha),
            ("recurrent_state_h_v", hc), ("recurrent_state_c_v", hc),
            ("recurrent_state_h_v_out", hc), ("recurrent_state_c_v_out", hc)
        ]:
            observation_spec[("agents", key)] = UnboundedContinuousTensorSpec(
                shape=(*bs, e, 1, dim), device=observation_spec.device
            )
        return observation_spec

# =========================================================
# Networks (Rewritten cleanly with nn.LSTMCell)
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
    in_keys = [("agents", "observation", "obs"), "is_init", ("agents", "recurrent_state_h"), ("agents", "recurrent_state_c")]
    out_keys = [("agents", "features"), ("agents", "recurrent_state_h_out"), ("agents", "recurrent_state_c_out")]

    def __init__(self, n_edges: int, n_temporal: int, n_static: int, hidden_dim: int, device: str):
        super().__init__()
        self.n_edges, self.n_temporal, self.n_static, self.hidden_dim = n_edges, n_temporal, n_static, hidden_dim
        self.feature = FeatureNet(n_temporal, hidden_dim).to(device)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim).to(device)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("agents", "observation", "obs"))
        is_init = td.get("is_init", None)
        step_mode = obs.ndim == 3
        if step_mode: obs = obs.unsqueeze(1)

        b, t, e, d = obs.shape
        hdim = self.hidden_dim

        temporal = obs[..., :self.n_temporal]   # [b, t, e, n_temporal]
        static   = obs[..., self.n_temporal:]   # [b, t, e, n_static]

        if is_init is None:
            mask = torch.ones((b * e, t, 1), device=obs.device)
        else:
            if is_init.ndim == 1: is_init = is_init.unsqueeze(1).expand(b, t)
            if is_init.shape[1] == 1 and t > 1: is_init = is_init.expand(b, t)
            mask = (~is_init).float().view(b, 1, t, 1).expand(b, e, t, 1).reshape(b * e, t, 1)

        temporal_be = temporal.transpose(1, 2).contiguous().reshape(b * e, t, self.n_temporal)
        feats_be = self.feature(temporal_be)

        h = td.get(("agents", "recurrent_state_h"))
        c = td.get(("agents", "recurrent_state_c"))
        h_curr = h[:, 0].reshape(b * e, hdim).contiguous() if h.ndim == 5 else h.reshape(b * e, hdim).contiguous()
        c_curr = c[:, 0].reshape(b * e, hdim).contiguous() if c.ndim == 5 else c.reshape(b * e, hdim).contiguous()

        out_h, out_c = [], []
        for ti in range(t):
            m = mask[:, ti]
            h_curr, c_curr = h_curr * m, c_curr * m
            h_curr, c_curr = self.lstm(feats_be[:, ti], (h_curr, c_curr))
            out_h.append(h_curr)
            out_c.append(c_curr)

        lstm_out = torch.stack(out_h, dim=1).view(b, e, t, hdim).transpose(1, 2).contiguous()  # [b, t, e, hdim]
        all_h    = torch.stack(out_h, dim=1).view(b, e, t, hdim).transpose(1, 2).unsqueeze(-2).contiguous()
        all_c    = torch.stack(out_c, dim=1).view(b, e, t, hdim).transpose(1, 2).unsqueeze(-2).contiguous()

        feats = torch.cat([lstm_out, static], dim=-1)  # [b, t, e, hdim + n_static]

        if step_mode:
            td.set(("agents", "features"), feats[:, 0])
            td.set(("agents", "recurrent_state_h_out"), all_h[:, 0])
            td.set(("agents", "recurrent_state_c_out"), all_c[:, 0])
        else:
            td.set(("agents", "features"), feats)
            td.set(("agents", "recurrent_state_h_out"), all_h)
            td.set(("agents", "recurrent_state_c_out"), all_c)

        return td

class CriticRecurrentCore(nn.Module):
    in_keys = [("agents", "observation", "obs"), "is_init", ("agents", "recurrent_state_h_v"), ("agents", "recurrent_state_c_v")]
    out_keys = [("agents", "vf_features"), ("agents", "recurrent_state_h_v_out"), ("agents", "recurrent_state_c_v_out")]

    def __init__(self, n_edges: int, n_temporal: int, n_static: int, hidden_dim: int, device: str):
        super().__init__()
        self.n_edges, self.n_temporal, self.n_static, self.hidden_dim = n_edges, n_temporal, n_static, hidden_dim
        self.feature = FeatureNet(n_temporal, hidden_dim).to(device)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim).to(device)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("agents", "observation", "obs"))
        is_init = td.get("is_init", None)
        step_mode = obs.ndim == 3
        if step_mode: obs = obs.unsqueeze(1)

        b, t, e, d = obs.shape
        hdim = self.hidden_dim

        temporal = obs[..., :self.n_temporal]   # [b, t, e, n_temporal]
        static   = obs[..., self.n_temporal:]   # [b, t, e, n_static]

        if is_init is None:
            mask = torch.ones((b * e, t, 1), device=obs.device)
        else:
            if is_init.ndim == 1: is_init = is_init.unsqueeze(1).expand(b, t)
            if is_init.shape[1] == 1 and t > 1: is_init = is_init.expand(b, t)
            mask = (~is_init).float().view(b, 1, t, 1).expand(b, e, t, 1).reshape(b * e, t, 1)

        temporal_be = temporal.transpose(1, 2).contiguous().reshape(b * e, t, self.n_temporal)
        feats_be = self.feature(temporal_be)

        h = td.get(("agents", "recurrent_state_h_v"))
        c = td.get(("agents", "recurrent_state_c_v"))
        h_curr = h[:, 0].reshape(b * e, hdim).contiguous() if h.ndim == 5 else h.reshape(b * e, hdim).contiguous()
        c_curr = c[:, 0].reshape(b * e, hdim).contiguous() if c.ndim == 5 else c.reshape(b * e, hdim).contiguous()

        out_v_h, out_v_c = [], []
        for ti in range(t):
            m = mask[:, ti]
            h_curr, c_curr = h_curr * m, c_curr * m
            h_curr, c_curr = self.lstm(feats_be[:, ti], (h_curr, c_curr))
            out_v_h.append(h_curr)
            out_v_c.append(c_curr)

        lstm_out = torch.stack(out_v_h, dim=1).view(b, e, t, hdim).transpose(1, 2).contiguous()  # [b, t, e, hdim]
        all_h    = torch.stack(out_v_h, dim=1).view(b, e, t, hdim).transpose(1, 2).unsqueeze(-2).contiguous()
        all_c    = torch.stack(out_v_c, dim=1).view(b, e, t, hdim).transpose(1, 2).unsqueeze(-2).contiguous()

        feats = torch.cat([lstm_out, static], dim=-1)  # [b, t, e, hdim + n_static]

        if step_mode:
            td.set(("agents", "vf_features"), feats[:, 0])
            td.set(("agents", "recurrent_state_h_v_out"), all_h[:, 0])
            td.set(("agents", "recurrent_state_c_v_out"), all_c[:, 0])
        else:
            td.set(("agents", "vf_features"), feats)
            td.set(("agents", "recurrent_state_h_v_out"), all_h)
            td.set(("agents", "recurrent_state_c_v_out"), all_c)

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

    # Strip any spurious trailing dims beyond [B, T, E] without touching the tensordict
    while reward.ndim > 3:     reward = reward.squeeze(-1)
    while done.ndim > 3:       done = done.squeeze(-1)
    while terminated.ndim > 3: terminated = terminated.squeeze(-1)
    while values.ndim > 3:     values = values.squeeze(-1)
    while next_values.ndim > 3: next_values = next_values.squeeze(-1)

    not_end = (~(done | terminated)).to(values.dtype)
    b, t, e = reward.shape

    adv = torch.zeros_like(reward)
    last_gae = torch.zeros((b, e), device=reward.device, dtype=reward.dtype)

    for ti in reversed(range(t)):
        delta = reward[:, ti] + gamma * next_values[:, ti] * not_end[:, ti] - values[:, ti]
        last_gae = delta + gamma * lmbda * not_end[:, ti] * last_gae
        adv[:, ti] = last_gae

    traj.set(("agents", "advantage"), adv)
    traj.set(("agents", "value_target"), adv + values)

def valid_sequence_minibatches(traj: TensorDictBase, seq_len: int, minibatch_size: int):
    b, t = traj.batch_size
    if t < seq_len: raise RuntimeError(f"T={t} < seq_len={seq_len}")

    done = traj.get(("agents", "done")).to(torch.bool)
    while done.ndim > 3: done = done.squeeze(-1)
    done_any = done.any(dim=-1)

    max_t0 = t - seq_len
    csum = torch.cumsum(done_any.to(torch.int32), dim=1)

    left = csum[:, : max_t0 + 1]
    right = csum[:, seq_len - 1 : seq_len - 1 + (max_t0 + 1)]
    prev_left = torch.cat([torch.zeros(b, 1, device=traj.device, dtype=csum.dtype), left[:, :-1]], dim=1)
    window_sum = right - prev_left
    valid = window_sum == 0

    valid_idx = valid.nonzero(as_tuple=False)
    if valid_idx.numel() == 0:
        all_b = torch.arange(b, device=traj.device).repeat_interleave(max_t0 + 1)
        all_t0 = torch.arange(max_t0 + 1, device=traj.device).repeat(b)
        valid_idx = torch.stack([all_b, all_t0], dim=1)

    perm = torch.randperm(valid_idx.shape[0], device=traj.device)
    valid_idx = valid_idx[perm]

    for start in range(0, valid_idx.shape[0], minibatch_size):
        idx = valid_idx[start : start + minibatch_size]
        b_idx, t0 = idx[:, 0], idx[:, 1]
        slices = [traj[b_idx, t0 + k] for k in range(seq_len)]
        yield torch.stack(slices, dim=1).to_tensordict()


# =========================================================
# Builders
# =========================================================
def make_wrapped_env(cfg_path: str, seed: int, decision_interval: int, n_actions: int,
                     env_class: Type[EdgeIDSParallelEnv] = None):
    if env_class is None:
        env_class = EdgeIDSParallelEnv
    pz = env_class(cfg_path=cfg_path, seed=seed, decision_interval=decision_interval, n_actions=n_actions)
    group_map = {"agents": list(pz.possible_agents)}
    return PettingZooWrapper(pz, categorical_actions=True, group_map=group_map)


def build_env_stack(env_cfg: dict, train_cfg: dict, cfg_path: str, num_envs: int,
                    env_class: Type[EdgeIDSParallelEnv] = None):
    if env_class is None:
        env_class = EdgeIDSParallelEnv

    decision_interval = int(env_cfg["globals"]["decision_interval"])
    n_actions = int(train_cfg["model"]["n_actions"])

    base = make_wrapped_env(cfg_path=cfg_path, seed=int(env_cfg["run"]["seed"]),
                            decision_interval=decision_interval, n_actions=n_actions,
                            env_class=env_class)
    obs_key = ("agents", "observation", "obs")
    obs_spec = base.observation_spec[obs_key]
    n_edges, obs_dim = int(obs_spec.shape[-2]), int(obs_spec.shape[-1])
    obs_dim_with_id = obs_dim + n_edges

    def make_one(i: int):
        def _make():
            return make_wrapped_env(cfg_path=cfg_path, seed=1000 + i,
                                    decision_interval=decision_interval, n_actions=n_actions,
                                    env_class=env_class)
        return _make

    penv = ParallelEnv(num_envs, [make_one(i) for i in range(num_envs)], device="cpu")
    hidden_dim = int(train_cfg["model"]["hidden_dim"])
    norm_transform = ObservationNorm(in_keys=[obs_key], standard_normal=True)
    transforms = [
        InitTracker(),
        norm_transform,
        AddAgentID(n_edges=n_edges),
        InitRecurrentState(n_edges=n_edges, actor_hidden_dim=hidden_dim, critic_hidden_dim=hidden_dim * 2),
        WriteRecurrentOutToNext(),
    ]

    env = TransformedEnv(penv, Compose(*transforms))

    on_cfg = train_cfg.get("observation_norm", {})
    env.transform.train()
    norm_transform.init_stats(
        num_iter=int(on_cfg.get("num_iter", 100)),
        reduce_dim=tuple(on_cfg.get("reduce_dim", (0, 1, 2))),
        cat_dim=0,
    )
    env.transform.eval()

    td0 = env.reset()
    print("obs step shape:", td0.get(("agents", "observation", "obs")).shape)
    return env, n_edges, obs_dim_with_id


# =========================================================
# Train
# =========================================================
def train(
    env_cfg_path: str = "./configs/simulation_0.yaml",
    train_cfg_path: str = "./configs/train.yaml",
    resume_ckpt: Optional[str] = None,
    device: str = "cuda",
    env_class: Type[EdgeIDSParallelEnv] = None,
):
    if env_class is None:
        env_class = EdgeIDSParallelEnv

    with open(env_cfg_path, "r") as f: env_cfg = yaml.safe_load(f)
    with open(train_cfg_path, "r") as f: train_cfg = yaml.safe_load(f)

    run = wandb_init(env_cfg, train_cfg)
    seed = int(env_cfg["run"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    decision_interval = int(env_cfg["globals"]["decision_interval"])
    num_envs = int(train_cfg["collector"]["num_envs"])
    decisions_per_episode = int(math.ceil(int(env_cfg["run"]["t_max"]) / decision_interval))

    env, n_edges, obs_dim = build_env_stack(env_cfg, train_cfg, env_cfg_path, num_envs, env_class=env_class)
    set_composite_lp_aggregate(False).set()

    hidden_dim = int(train_cfg["model"]["hidden_dim"])
    _tmp_env = env_class(cfg_path=env_cfg_path, seed=0, decision_interval=decision_interval)
    obs_keys = _tmp_env.obs_keys
    raw_obs_dim = len(obs_keys)  # without agent-ID appended by AddAgentID

    n_temporal = N_TEMPORAL
    n_static = obs_dim - n_temporal  # remaining dims including agent-ID one-hot

    actor_core = AgentRecurrentCore(n_edges=n_edges, n_temporal=n_temporal, n_static=n_static, hidden_dim=hidden_dim, device=device)
    actor_head = TensorDictModule(nn.Linear(hidden_dim + n_static, int(train_cfg["model"]["n_actions"])).to(device), in_keys=[("agents", "features")], out_keys=[("agents", "logits")])
    policy = ProbabilisticActor(
        module=TensorDictSequential(actor_core, actor_head),
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    critic_core = CriticRecurrentCore(n_edges=n_edges, n_temporal=n_temporal, n_static=n_static, hidden_dim=hidden_dim * 2, device=device)
    critic_head = TensorDictModule(nn.Linear(hidden_dim * 2 + n_static, 1).to(device), in_keys=[("agents", "vf_features")], out_keys=[("agents", "state_value")])
    value_net = TensorDictSequential(critic_core, critic_head)
    collector_policy = TensorDictSequential(policy, value_net)

    optim = torch.optim.Adam(
        list(actor_core.parameters()) + list(actor_head.parameters()) + list(critic_core.parameters()) + list(critic_head.parameters()),
        lr=float(train_cfg["optim"]["lr"]), weight_decay=float(train_cfg["optim"]["weight_decay"]), eps=float(train_cfg["optim"]["eps"]),
    )

    if resume_ckpt:
        state = torch.load(resume_ckpt, map_location=device)
        policy.load_state_dict(state["policy"])
        value_net.load_state_dict(state["value"])
        optim.load_state_dict(state["optim"])

    frames_per_batch = int(train_cfg["collector"].get("frames_per_batch", decisions_per_episode * num_envs))
    total_frames = int(train_cfg["collector"]["total_frames"])

    collector = SyncDataCollector(
        env, policy=collector_policy, frames_per_batch=frames_per_batch,
        total_frames=total_frames, device=device, trust_policy=bool(train_cfg["collector"]["trust_policy"]), split_trajs=False,
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
    loss_critic_type = train_cfg["loss"].get("loss_critic_type", "smooth_l1")

    ckpt_dir = Path("checkpoints") / run.name
    total_updates_est = max(1, (total_frames // frames_per_batch) * ppo_epochs * math.ceil(frames_per_batch / minibatch_size))
    updates_done, best_qoe = 0, -1e9
    global_decision_step = 0

    for it, batch in enumerate(collector):
        traj = batch.clone(False)

        # Move reward / done / terminated to root
        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated")]:
            nk = ("next",) + k
            if k not in traj.keys(True, True) and nk in traj.keys(True, True):
                traj.set(k, traj.get(nk))

        if ("agents", "done") in traj.keys(True, True): traj.set(("agents", "done"), traj.get(("agents", "done")).to(torch.bool))
        if ("agents", "terminated") in traj.keys(True, True): traj.set(("agents", "terminated"), traj.get(("agents", "terminated")).to(torch.bool))
        if ("agents", "done") in traj.get("next").keys(True, True): traj.get("next").set(("agents", "done"), traj.get("next").get(("agents", "done")).to(torch.bool))
        if ("agents", "terminated") in traj.get("next").keys(True, True): traj.get("next").set(("agents", "terminated"), traj.get("next").get(("agents", "terminated")).to(torch.bool))

        with torch.no_grad():
            # --- FIX 2: Episode Bleeding ---
            # Recompute Values only for 'next' to ensure accurate targets, passing correct Episode Boundary Masks!
            nxt = traj.get("next")
            # nxt.set("is_init", traj.get(("agents", "done")).any(dim=-1)) #! Suspicious
            
            value_net(traj)
            value_net(nxt)
            squeeze_last1(traj, ("agents", "state_value"))
            squeeze_last1(nxt, ("agents", "state_value"))
            
            compute_gae_inplace(traj, gamma=gamma, lmbda=gae_lambda, n_edges=n_edges)

        # --- 1. Cleaned: Global, Per-Agent Advantage Normalization ---
        adv_full = traj.get(("agents", "advantage"))
        m = adv_full.mean(dim=(0, 1), keepdim=True)
        s = adv_full.std(dim=(0, 1), keepdim=True).clamp_min(1e-5) # Slightly higher clamp for stability
        traj.set(("agents", "advantage"), (adv_full - m) / s)

        # --- 2. Cleaned: Save old values for clipping ---
        traj.set(("agents", "old_state_value"), traj.get(("agents", "state_value")).clone())

        last_total_loss = last_policy_loss = last_critic_loss = last_entropy = None
        seq_len = int(train_cfg["loss"].get("seq_len", 8)) # Use the optimal 8 you found!

        for _ in range(ppo_epochs):
            for sub in valid_sequence_minibatches(traj, seq_len=seq_len, minibatch_size=minibatch_size):
                alpha = max(1.0 - (updates_done / total_updates_est), 0.0)
                if bool(train_cfg["optim"].get("anneal_lr", True)):
                    for g in optim.param_groups: g["lr"] = base_lr * alpha
                clip_eps_now = base_clip_eps * alpha if bool(train_cfg["loss"].get("anneal_clip_epsilon", True)) else base_clip_eps
                updates_done += 1

                # Recompute logits and values
                actor_core(sub)
                actor_head(sub)
                value_net(sub)

                squeeze_last1(sub, ("agents", "state_value"))
                act = sub.get(("agents", "action")).long()
                while act.ndim > 3: act = act.squeeze(-1)

                old_logp = sub.get(("agents", "sample_log_prob")).detach()
                while old_logp.ndim > 3: old_logp = old_logp.squeeze(-1)

                adv = sub.get(("agents", "advantage")).detach()
                while adv.ndim > 3: adv = adv.squeeze(-1)

                logits = sub.get(("agents", "logits"))
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act)
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps_now, 1.0 + clip_eps_now) * adv
                policy_loss = -(torch.min(surr1, surr2)).mean()

                v_pred = sub.get(("agents", "state_value"))
                v_targ = sub.get(("agents", "value_target")).detach()
                v_old = sub.get(("agents", "old_state_value")).detach()

                # Value Clipping
                v_pred_clipped = v_old + torch.clamp(v_pred - v_old, -clip_eps_now, clip_eps_now)

                if loss_critic_type == "smooth_l1":
                    vf_loss1 = nn.functional.smooth_l1_loss(v_pred, v_targ, reduction="none")
                    vf_loss2 = nn.functional.smooth_l1_loss(v_pred_clipped, v_targ, reduction="none")
                    critic_loss = torch.max(vf_loss1, vf_loss2).mean()
                else:
                    critic_loss = 0.5 * torch.max(
                        (v_targ - v_pred).pow(2),
                        (v_targ - v_pred_clipped).pow(2),
                    ).mean()

                entropy_loss = -entropy.mean()
                total_loss = policy_loss + critic_coeff * critic_loss + entropy_coeff * entropy_loss

                optim.zero_grad(set_to_none=True)
                total_loss.backward()

                params = list(actor_core.parameters()) + list(actor_head.parameters()) + list(critic_core.parameters()) + list(critic_head.parameters())
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optim.step()

                last_total_loss, last_policy_loss = total_loss.detach(), policy_loss.detach()
                last_critic_loss, last_entropy = critic_loss.detach(), entropy.mean().detach()

        collector.update_policy_weights_()

        reward_mean = float(traj.get(("agents", "reward")).mean().item())
        qoe_mean    = float(traj.get(("next", "agents", "observation", "qoe_mean")).mean().item())
        _r_lres = float(traj.get(("next", "agents", "observation", "reward_lambda_res")).mean().item())
        _r_bcd  = float(traj.get(("next", "agents", "observation", "reward_benign_col_dmg")).mean().item())
        _r_qoe  = float(traj.get(("next", "agents", "observation", "reward_qoe_penalty")).mean().item())
        _vio    = float(traj.get(("next", "agents", "observation", "qoe_vio_rate")).mean().item())

        print(
            f"Iter={it:4d} | rew={reward_mean:+.4f} qoe={qoe_mean:.4f} "
            f"| atk_pass={_r_lres:.3f} bcd={_r_bcd:.3f} qoe_sf={_r_qoe:.3f} "
            f"| qoe_vio={_vio:.1%}"
        )
        reward_mean_per_agent = traj.get(("agents", "reward")).mean(dim=(0, 1))
        qoe_mean_per_agent    = traj.get(("next", "agents", "observation", "qoe_mean")).mean(dim=(0, 1))
        if qoe_mean_per_agent.ndim > 1:
            qoe_mean_per_agent = qoe_mean_per_agent.squeeze(-1)
        for i in range(n_edges):
            print(
                f"  agent={i} reward={reward_mean_per_agent[i].item():.4f} "
                f"qoe={qoe_mean_per_agent[i].item():.4f}"
            )

        # Log per-step obs (raw obs only, not the appended agent-ID dims)
        obs = traj.get(("agents", "observation", "obs"))[..., :raw_obs_dim]
        obs_log = obs.clone()
        norm_t = env.transform[1]
        for k in ["cpu_to_ids_ratio", "ema_mom"]:
            if k not in obs_keys:
                continue
            idx = obs_keys.index(k)
            # If loc is [n_edges, raw_obs_dim], we take [:, idx] to get [n_edges]
            # If loc is [raw_obs_dim], we take [idx] to get a scalar (broadcasts across n_edges)
            if norm_t.loc.numel() == n_edges * raw_obs_dim:
                loc   = norm_t.loc.view(n_edges, raw_obs_dim)[:, idx].to(obs_log.device)
                scale = norm_t.scale.view(n_edges, raw_obs_dim)[:, idx].to(obs_log.device)
            else:
                loc   = norm_t.loc[idx].to(obs_log.device)
                scale = norm_t.scale[idx].to(obs_log.device)
            obs_log[..., idx] = obs[..., idx] * scale + loc
        global_decision_step = wandb_log_obs_steps(
            obs_log, obs_keys, keep_keys={"cpu_to_ids_ratio", "ema_mom"},
            global_step_start=global_decision_step,
        )

        if (it + 1) % 50 == 0:
            save_ckpt(ckpt_dir / f"ckpt_iter_{it+1:06d}.pt", policy, value_net, optim,
                      env_cfg, train_cfg, it + 1, device, env)
        if qoe_mean > best_qoe:
            best_qoe = qoe_mean
            save_ckpt(ckpt_dir / "ckpt_best.pt", policy, value_net, optim,
                      env_cfg, train_cfg, it + 1, device, env)

        wandb.log({
            "iter":                  it,
            "qoe/mean":              qoe_mean,
            "qoe/vio_rate":          _vio,
            "reward/mean":           reward_mean,
            "reward/lambda_res":     _r_lres,
            "reward/benign_col_dmg": _r_bcd,
            "reward/qoe_penalty":    _r_qoe,
            "loss/total":   float(last_total_loss.item())  if last_total_loss  is not None else 0.0,
            "loss/policy":  float(last_policy_loss.item()) if last_policy_loss is not None else 0.0,
            "loss/critic":  float(last_critic_loss.item()) if last_critic_loss is not None else 0.0,
            "entropy":      float(last_entropy.item())     if last_entropy     is not None else 0.0,
            "train/lr":       float(optim.param_groups[0]["lr"]),
            "train/clip_eps": float(clip_eps_now),
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cma", choices=["cma", "ima"])
    parser.add_argument("--cfg", type=str, default="./configs/simulation_0.yaml")
    args = parser.parse_args()

    env_cls = EdgeIDSCMAParallelEnv if args.mode == "cma" else EdgeIDSIMAParallelEnv
    train(env_class=env_cls, env_cfg_path=args.cfg)

# python train_ma_lstm_old.py --mode cma --cfg configs/simulation_ma_0.yaml
# python train_ma_lstm_old.py --mode ima --cfg configs/simulation_ma_0.yaml