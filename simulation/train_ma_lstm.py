"""
Shared infrastructure for MAPPO-ish (CTDE) Recurrent PPO.

Subclass EdgeIDSParallelEnv and implement _compute_step_reward to choose
between independent (per-agent) and centralised (shared scalar) reward modes.
"""


import math
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Type

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
from torchrl.modules import ProbabilisticActor

from environment import build_env_base, VA_CPU_RESERVE
from logger import wandb_init, wandb_log_obs_steps
import wandb
import argparse

TEMPORAL_OBS_KEYS = {"local_num_req", "attack_in_rate", "neighbor_atk_rate"}


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
                "prev_slo_vio",
                
            ]
        
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
        """Return (rew[E], lres_acc[E], bcd_acc[E], qsf_acc[E], vio_acc[E]).

        Subclasses implement IMA (per-agent) or CMA (shared scalar) variants.
        """
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
            
            # Netting queue logic: clamp requested target relative to settled
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
                
                # SLA History
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


# =========================================================
# Transforms
# =========================================================

class BuildCentralObs(Transform):
    """Adds root key 'observation_flat' by flattening agent obs over (E, D)."""

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


class WriteRecurrentOutToNext(Transform):
    def __init__(self):
        super().__init__(
            in_keys=[
                ("agents", "recurrent_state_h_out"),
                ("agents", "recurrent_state_c_out"),
                "recurrent_state_h_v_out",
                "recurrent_state_c_v_out",
            ],
            out_keys=[],
        )

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        nxt = td.get("next")
        if nxt is None:
            return td

        nxt.set(("agents", "recurrent_state_h"), td.get(("agents", "recurrent_state_h_out")))
        nxt.set(("agents", "recurrent_state_c"), td.get(("agents", "recurrent_state_c_out")))
        h_v_out = td.get("recurrent_state_h_v_out", default=None)
        c_v_out = td.get("recurrent_state_c_v_out", default=None)
        if h_v_out is not None:
            nxt.set("recurrent_state_h_v", h_v_out)
            nxt.set("recurrent_state_c_v", c_v_out)
        else:
            h_v = td.get("recurrent_state_h_v", default=None)
            c_v = td.get("recurrent_state_c_v", default=None)
            if h_v is not None:
                nxt.set("recurrent_state_h_v", h_v)
                nxt.set("recurrent_state_c_v", c_v)
        return td


class InitRecurrentState(Transform):
    def __init__(self, n_edges: int, actor_hidden_dim: int, critic_hidden_dim: int):
        super().__init__(in_keys=[], out_keys=[])
        self.n_edges = int(n_edges)
        self.actor_hidden_dim = int(actor_hidden_dim)
        self.critic_hidden_dim = int(critic_hidden_dim)

    def _reset(self, td, td_reset, **kwargs):
        bs = tuple(td_reset.batch_size)
        dev = td_reset.device
        B = bs[0] if len(bs) else 1
        E = self.n_edges
        Ha = self.actor_hidden_dim
        Hc = self.critic_hidden_dim

        td_reset.set(("agents", "recurrent_state_h"),     torch.zeros((B, E, 1, Ha), device=dev))
        td_reset.set(("agents", "recurrent_state_c"),     torch.zeros((B, E, 1, Ha), device=dev))
        td_reset.set(("agents", "recurrent_state_h_out"), torch.zeros((B, E, 1, Ha), device=dev))
        td_reset.set(("agents", "recurrent_state_c_out"), torch.zeros((B, E, 1, Ha), device=dev))

        td_reset.set("recurrent_state_h_v",     torch.zeros((B, 1, Hc), device=dev))
        td_reset.set("recurrent_state_c_v",     torch.zeros((B, 1, Hc), device=dev))
        td_reset.set("recurrent_state_h_v_out", torch.zeros((B, 1, Hc), device=dev))
        td_reset.set("recurrent_state_c_v_out", torch.zeros((B, 1, Hc), device=dev))
        return td_reset


class CarryCriticState(Transform):
    def __init__(self):
        super().__init__(
            in_keys=[("next", "recurrent_state_h_v"), ("next", "recurrent_state_c_v")],
            out_keys=["recurrent_state_h_v", "recurrent_state_c_v"],
        )

    def _call(self, td):
        h = td.get(("next", "recurrent_state_h_v"), default=None)
        c = td.get(("next", "recurrent_state_c_v"), default=None)
        if h is not None:
            td.set("recurrent_state_h_v", h)
            td.set("recurrent_state_c_v", c)
        return td


class FlatToAgentsObs(Transform):
    """Reconstruct ("agents","observation","obs") from "observation_flat"."""

    def __init__(self, n_edges: int, obs_dim: int,
                 in_key: str = "observation_flat",
                 out_key=("agents", "observation", "obs")):
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.n_edges = int(n_edges)
        self.obs_dim = int(obs_dim)
        self.in_key = in_key
        self.out_key = out_key

    def _flat_to_agents(self, x: torch.Tensor) -> torch.Tensor:
        E, D = self.n_edges, self.obs_dim
        if x.ndim == 2:
            return x.view(x.shape[0], E, D)
        if x.ndim == 3:
            B, T = x.shape[:2]
            return x.view(B, T, E, D)
        raise RuntimeError(f"{self.in_key} expected 2D or 3D, got {x.shape}")

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        x = td.get(self.in_key, default=None)
        if x is not None:
            td.set(self.out_key, self._flat_to_agents(x))
        nxt = td.get("next", default=None)
        if nxt is not None:
            x2 = nxt.get(self.in_key, default=None)
            if x2 is not None:
                nxt.set(self.out_key, self._flat_to_agents(x2))
        return td

    def _reset(self, td: TensorDictBase, td_reset: TensorDictBase, **kwargs) -> TensorDictBase:
        x = td_reset.get(self.in_key, default=None)
        if x is not None:
            td_reset.set(self.out_key, self._flat_to_agents(x))
        return td_reset

    def transform_observation_spec(self, observation_spec):
        bs = observation_spec.shape
        flat_spec = observation_spec[self.in_key]
        observation_spec[self.out_key] = UnboundedContinuousTensorSpec(
            shape=(*bs, self.n_edges, self.obs_dim),
            dtype=flat_spec.dtype,
            device=flat_spec.device,
        )
        return observation_spec


# =========================================================
# Networks
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


class SharedRecurrentCore(nn.Module):
    def __init__(self, n_edges: int, obs_dim: int, hidden_dim: int, device: str,
                 temporal_local_idx: List[int], static_local_idx: List[int]):
        super().__init__()
        self.n_edges = int(n_edges)
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_static = len(static_local_idx)

        self.register_buffer("temporal_idx", torch.tensor(temporal_local_idx, dtype=torch.long))
        self.register_buffer("static_idx", torch.tensor(static_local_idx, dtype=torch.long))

        # Baseline parity: No FeatureNet before LSTM for temporal features
        self._lstm = nn.LSTM(len(temporal_local_idx), self.hidden_dim, batch_first=True).to(device)

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        x = td.get("observation_flat")
        is_init = td.get("is_init")

        E, D, H = self.n_edges, self.obs_dim, self.hidden_dim

        step_mode = (x.ndim == 2)
        if step_mode:
            x = x.unsqueeze(1)
        while is_init.ndim < 3:
            is_init = is_init.unsqueeze(-1)

        B, T, FD = x.shape
        # obs_be: [B*E, T, D]
        obs_be = x.view(B, T, E, D).transpose(1, 2).reshape(B * E, T, D)
        temporal_be = obs_be[..., self.temporal_idx]
        static_be   = obs_be[..., self.static_idx]

        # Hidden state handling
        h = td.get(("agents", "recurrent_state_h"), default=None)
        c = td.get(("agents", "recurrent_state_c"), default=None)
        
        if h is None:
            # Initialize with zeros if missing
            h0 = torch.zeros(B, E, 1, H, device=x.device)
            c0 = torch.zeros(B, E, 1, H, device=x.device)
        elif not step_mode and h.ndim == 5:
            # During training (T > 1), h has shape [B, T, E, 1, H]. 
            # We only need the state from the first step of the sequence.
            h0 = h[:, 0] # [B, E, 1, H]
            c0 = c[:, 0]
        else:
            h0 = h # [B, E, 1, H]
            c0 = c

        # Reset states at episode boundaries
        # reset_mask: [B, 1, 1, 1] applied to h0: [B, E, 1, H]
        mask_val = is_init[:, 0].float()
        while mask_val.ndim < 4:
            mask_val = mask_val.unsqueeze(-1)
        h0 = h0 * (1.0 - mask_val)
        c0 = c0 * (1.0 - mask_val)

        # Reshape for LSTM: (num_layers, batch, hidden_size)
        # Here num_layers=1, batch=B*E
        h_lstm = h0.transpose(1, 2).reshape(1, B * E, H).contiguous()
        c_lstm = c0.transpose(1, 2).reshape(1, B * E, H).contiguous()

        lstm_out, (h_n, c_n) = self._lstm(temporal_be, (h_lstm, c_lstm))

        merged_be = torch.cat([lstm_out, static_be], dim=-1)
        F_out = H + self.n_static
        
        # feats_be_out: [B, E, T, H+S]
        feats_be_out = merged_be.reshape(B, E, T, F_out)
        
        # For Actor: Local features per agent
        # We need [B, T, E, H+S] for TorchRL compatibility
        feats_btEH = feats_be_out.transpose(1, 2).contiguous()
        
        # For Critic: Flattened centralized features
        # [B, T, E*(H+S)]
        feats_bt_flat = feats_be_out.transpose(1, 2).reshape(B, T, E * F_out).contiguous()

        if step_mode:
            td.set(("agents", "features"), feats_btEH[:, 0])
            td.set("vf_features", feats_bt_flat[:, 0])
            
            h_out = h_n.transpose(0, 1).reshape(B, E, 1, H)
            c_out = c_n.transpose(0, 1).reshape(B, E, 1, H)
            td.set(("agents", "recurrent_state_h_out"), h_out)
            td.set(("agents", "recurrent_state_c_out"), c_out)
        else:
            td.set(("agents", "features"), feats_btEH)
            td.set("vf_features", feats_bt_flat)

        return td


# =========================================================
# PPO / GAE
# =========================================================

@torch.no_grad()
def compute_gae_inplace(traj: TensorDictBase, gamma: float, lmbda: float, n_edges: int):
    reward = traj.get(("agents", "reward"))
    done = traj.get(("agents", "done"))
    terminated = traj.get(("agents", "terminated"))
    values = traj.get(("agents", "state_value"))
    next_values = traj.get("next").get(("agents", "state_value"))

    # Ensure all are [B, T, E]
    if reward.ndim == 4 and reward.shape[-1] == 1: reward = reward.squeeze(-1)
    if done.ndim == 4 and done.shape[-1] == 1: done = done.squeeze(-1)
    if terminated.ndim == 4 and terminated.shape[-1] == 1: terminated = terminated.squeeze(-1)
    if values.ndim == 4 and values.shape[-1] == 1: values = values.squeeze(-1)
    if next_values.ndim == 4 and next_values.shape[-1] == 1: next_values = next_values.squeeze(-1)

    done = done.to(torch.bool)
    terminated = terminated.to(torch.bool)
    not_end = (~(done | terminated)).to(values.dtype)

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
    """Sample windows that do not cross episode boundaries. traj batch: [B, T]."""
    B, T = traj.batch_size
    if T < seq_len:
        raise RuntimeError(f"T={T} < seq_len={seq_len}")

    done = traj.get(("agents", "done")).to(torch.bool)
    while done.ndim > 3:
        done = done.squeeze(-1)
    done_any = done.any(dim=-1)

    max_t0 = T - seq_len
    csum = torch.cumsum(done_any.to(torch.int32), dim=1)

    left  = csum[:, : max_t0 + 1]
    right = csum[:, seq_len - 1 : seq_len - 1 + (max_t0 + 1)]
    prev_left = torch.cat([torch.zeros(B, 1, device=traj.device, dtype=csum.dtype), left[:, :-1]], dim=1)
    window_sum = right - prev_left
    valid = window_sum == 0

    valid_idx = valid.nonzero(as_tuple=False)
    if valid_idx.numel() == 0:
        all_b  = torch.arange(B, device=traj.device).repeat_interleave(max_t0 + 1)
        all_t0 = torch.arange(max_t0 + 1, device=traj.device).repeat(B)
        valid_idx = torch.stack([all_b, all_t0], dim=1)

    perm = torch.randperm(valid_idx.shape[0], device=traj.device)
    valid_idx = valid_idx[perm]

    for start in range(0, valid_idx.shape[0], minibatch_size):
        idx = valid_idx[start : start + minibatch_size]
        b_idx = idx[:, 0]
        t0    = idx[:, 1]
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
    obs_spec = base.observation_spec[("agents", "observation", "obs")]
    n_edges = int(obs_spec.shape[-2])
    obs_dim = int(obs_spec.shape[-1])

    def make_one(i: int):
        def _make():
            return make_wrapped_env(cfg_path=cfg_path, seed=1000 + i,
                                    decision_interval=decision_interval, n_actions=n_actions,
                                    env_class=env_class)
        return _make

    penv = ParallelEnv(num_envs, [make_one(i) for i in range(num_envs)], device="cpu")

    hidden_dim = int(train_cfg["model"]["hidden_dim"])
    actor_h  = hidden_dim
    critic_h = hidden_dim * 2
    transforms: List[Transform] = [
        InitTracker(),
        BuildCentralObs(n_edges=n_edges, obs_dim=obs_dim, out_key="observation_flat"),
        ObservationNorm(in_keys=["observation_flat"], standard_normal=True),
        FlatToAgentsObs(n_edges=n_edges, obs_dim=obs_dim),
        InitRecurrentState(n_edges=n_edges, actor_hidden_dim=actor_h, critic_hidden_dim=critic_h),
        WriteRecurrentOutToNext(),
        CarryCriticState(),
    ]

    env = TransformedEnv(penv, Compose(*transforms))

    on_cfg = train_cfg.get("observation_norm", {})
    env.transform.train()
    env.transform[2].init_stats(
        num_iter=int(on_cfg.get("num_iter", 100)),
        reduce_dim=tuple(on_cfg.get("reduce_dim", (0, 1))),
        cat_dim=int(on_cfg.get("cat_dim", 0)),
    )
    env.transform.eval()

    td0 = env.reset()
    print("env.batch_size:", env.batch_size)
    print("obs step shape:", td0.get(("agents", "observation", "obs")).shape)
    print("actor h shape:", td0.get(("agents", "recurrent_state_h")).shape)
    print("critic h shape:", td0.get("recurrent_state_h_v").shape)

    return env, n_edges, obs_dim


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

    env, n_edges, obs_dim = build_env_stack(env_cfg, train_cfg, env_cfg_path, num_envs, env_class=env_class)

    set_composite_lp_aggregate(False).set()

    n_actions  = int(train_cfg["model"]["n_actions"])
    hidden_dim = int(train_cfg["model"]["hidden_dim"])

    _tmp_env = env_class(cfg_path=env_cfg_path, seed=0, decision_interval=decision_interval)
    temporal_local_idx = [i for i, k in enumerate(_tmp_env.obs_keys) if k in TEMPORAL_OBS_KEYS]
    static_local_idx   = [i for i, k in enumerate(_tmp_env.obs_keys) if k not in TEMPORAL_OBS_KEYS]
    n_static_per_edge  = len(static_local_idx)
    actor_feat_dim     = hidden_dim + n_static_per_edge
    critic_hidden_dim  = hidden_dim # shared hidden size
    critic_feat_dim    = n_edges * (hidden_dim + n_static_per_edge)

    shared_core = SharedRecurrentCore(
        n_edges=n_edges, obs_dim=obs_dim, hidden_dim=hidden_dim, device=device,
        temporal_local_idx=temporal_local_idx, static_local_idx=static_local_idx,
    )
    
    actor_head = TensorDictModule(
        nn.Linear(actor_feat_dim, n_actions).to(device),
        in_keys=[("agents", "features")],
        out_keys=[("agents", "logits")],
    )
    policy = ProbabilisticActor(
        module=TensorDictSequential(shared_core, actor_head),
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    critic_head = TensorDictModule(
        nn.Linear(critic_feat_dim, n_edges).to(device),
        in_keys=["vf_features"],
        out_keys=[("agents", "state_value")],
    )
    # Both use shared_core
    value_net = TensorDictSequential(shared_core, critic_head)

    optim = torch.optim.Adam(
        list(shared_core.parameters())
        + list(actor_head.parameters())
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
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        trust_policy=bool(train_cfg["collector"]["trust_policy"]),
        split_trajs=False,
    )

    minibatch_size  = int(train_cfg["loss"]["mini_batch_size"])
    ppo_epochs      = int(train_cfg["loss"]["ppo_epochs"])
    max_grad_norm   = float(train_cfg["optim"]["max_grad_norm"])
    entropy_coeff   = float(train_cfg["loss"]["entropy_coeff"])
    critic_coeff    = float(train_cfg["loss"]["critic_coeff"])
    gamma           = float(train_cfg["loss"]["gamma"])
    gae_lambda      = float(train_cfg["loss"]["gae_lambda"])
    base_lr         = float(train_cfg["optim"]["lr"])
    base_clip_eps   = float(train_cfg["loss"]["clip_epsilon"])
    loss_critic_type = train_cfg["loss"].get("loss_critic_type", "mse")

    ckpt_dir   = Path("checkpoints") / run.name
    ckpt_every = 50
    ema_alpha  = float(train_cfg.get("logger", {}).get("ema_alpha", 0.1))
    ema_reward = ema_qoe = ema_vio = None
    best_reward = best_qoe = -1e9
    best_vio   = 1e9
    obs_keys   = _tmp_env.obs_keys

    total_updates_est = max(
        1, (total_frames // frames_per_batch) * ppo_epochs * math.ceil(frames_per_batch / minibatch_size)
    )
    updates_done       = 0
    global_decision_step = 0

    for it, batch in enumerate(collector):
        traj = batch.clone(False)

        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated")]:
            nk = ("next",) + k
            if k not in traj.keys(True, True) and nk in traj.keys(True, True):
                traj.set(k, traj.get(nk))

        if ("agents", "done") in traj.keys(True, True):
            traj.set(("agents", "done"), traj.get(("agents", "done")).to(torch.bool))
        if ("agents", "terminated") in traj.keys(True, True):
            traj.set(("agents", "terminated"), traj.get(("agents", "terminated")).to(torch.bool))
        if ("agents", "done") in traj.get("next").keys(True, True):
            traj.get("next").set(("agents", "done"), traj.get("next").get(("agents", "done")).to(torch.bool))
        if ("agents", "terminated") in traj.get("next").keys(True, True):
            traj.get("next").set(("agents", "terminated"), traj.get("next").get(("agents", "terminated")).to(torch.bool))

        with torch.no_grad():
            value_net(traj)
            vals = traj.get(("agents", "state_value"))

            h_v_seq = traj.get("recurrent_state_h_v_seq", default=None)
            c_v_seq = traj.get("recurrent_state_c_v_seq", default=None)
            if h_v_seq is not None:
                traj.set("recurrent_state_h_v", h_v_seq)
            if c_v_seq is not None:
                traj.set("recurrent_state_c_v", c_v_seq)

            B_sz = vals.shape[0]
            Hc   = hidden_dim * 2
            dev  = vals.device
            h_v_final = traj.get("recurrent_state_h_v_final", default=None)
            c_v_final = traj.get("recurrent_state_c_v_final", default=None)
            if h_v_final is not None and h_v_final.ndim == 4:
                h_v_final = h_v_final[:, -1]
            if c_v_final is not None and c_v_final.ndim == 4:
                c_v_final = c_v_final[:, -1]

            _next = traj.get("next")
            # Prefer the hidden state written into "next" by WriteRecurrentOutToNext;
            # fall back to the start-of-last-step state from traj (one step behind,
            # but far better than zeros).
            _h_next = _next.get(("agents", "recurrent_state_h"), default=None)
            _c_next = _next.get(("agents", "recurrent_state_c"), default=None)
            boot_h = _h_next[:, -1] if _h_next is not None else traj.get(("agents", "recurrent_state_h"))[:, -1]
            boot_c = _c_next[:, -1] if _c_next is not None else traj.get(("agents", "recurrent_state_c"))[:, -1]
            boot_td = TensorDict(
                {
                    "observation_flat": _next.get("observation_flat")[:, -1],
                    "is_init": torch.zeros(B_sz, 1, dtype=torch.bool, device=dev),
                    "recurrent_state_h_v": h_v_final if h_v_final is not None else torch.zeros(B_sz, 1, Hc, device=dev),
                    "recurrent_state_c_v": c_v_final if c_v_final is not None else torch.zeros(B_sz, 1, Hc, device=dev),
                    ("agents", "recurrent_state_h"): boot_h,
                    ("agents", "recurrent_state_c"): boot_c,
                },
                batch_size=[B_sz],
                device=dev,
            )
            value_net(boot_td)
            v_boot = boot_td.get(("agents", "state_value"))

            next_vals = torch.cat([vals[:, 1:], v_boot.unsqueeze(1)], dim=1)
            traj.set(("next", "agents", "state_value"), next_vals)

            compute_gae_inplace(traj, gamma=gamma, lmbda=gae_lambda, n_edges=n_edges)
            traj.set(("agents", "state_value_old"), vals.clone())

        last_total_loss = last_policy_loss = last_critic_loss = last_entropy = None
        seq_len = int(train_cfg["loss"].get("seq_len", 32))

        for _ in range(ppo_epochs):
            for sub in valid_sequence_minibatches(traj, seq_len=seq_len, minibatch_size=minibatch_size):
                alpha = max(0.0, 1.0 - (updates_done / total_updates_est))
                if bool(train_cfg["optim"].get("anneal_lr", True)):
                    lr_now = base_lr * alpha
                    for g in optim.param_groups:
                        g["lr"] = lr_now
                clip_eps_now = base_clip_eps * alpha if bool(train_cfg["loss"].get("anneal_clip_epsilon", True)) else base_clip_eps
                updates_done += 1

                shared_core(sub)
                actor_head(sub)
                critic_head(sub)

                act = sub.get(("agents", "action")).long()
                if act.ndim == 4 and act.shape[-1] == 1:
                    act = act.squeeze(-1)

                old_logp = sub.get(("agents", "sample_log_prob"))
                if old_logp.ndim == 4 and old_logp.shape[-1] == 1:
                    old_logp = old_logp.squeeze(-1)

                adv = sub.get(("agents", "advantage"))
                adv = (adv - adv.mean(dim=(0, 1), keepdim=True)) / (adv.std(dim=(0, 1), keepdim=True) + 1e-8)

                logits = sub.get(("agents", "logits"))
                dist   = Categorical(logits=logits)
                new_logp = dist.log_prob(act)
                entropy  = dist.entropy()

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps_now, 1.0 + clip_eps_now) * adv
                policy_loss = -(torch.min(surr1, surr2)).mean()

                v_pred = sub.get(("agents", "state_value"))
                v_targ = sub.get(("agents", "value_target"))
                v_old  = sub.get(("agents", "state_value_old"))
                v_pred_clipped = v_old + (v_pred - v_old).clamp(-clip_eps_now, clip_eps_now)
                if loss_critic_type == "smooth_l1":
                    critic_loss = torch.max(
                        nn.functional.smooth_l1_loss(v_pred, v_targ, reduction='none'),
                        nn.functional.smooth_l1_loss(v_pred_clipped, v_targ, reduction='none'),
                    ).mean()
                else:
                    critic_loss = 0.5 * torch.max(
                        (v_targ - v_pred).pow(2),
                        (v_targ - v_pred_clipped).pow(2),
                    ).mean()

                entropy_loss = -entropy.mean()
                total_loss = policy_loss + critic_coeff * critic_loss + entropy_coeff * entropy_loss

                optim.zero_grad(set_to_none=True)
                total_loss.backward()
                params = (
                    list(shared_core.parameters())
                    + list(actor_head.parameters())
                    + list(critic_head.parameters())
                )
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optim.step()

                last_total_loss  = total_loss.detach()
                last_policy_loss = policy_loss.detach()
                last_critic_loss = critic_loss.detach()
                last_entropy     = entropy.mean().detach()

        collector.update_policy_weights_()

        reward_mean = float(traj.get(("agents", "reward")).mean().item())
        qoe_mean    = float(traj.get(("next", "agents", "observation", "qoe_mean")).mean().item())

        _r_lres = float(traj.get(("next", "agents", "observation", "reward_lambda_res")).mean().item())
        _r_bcd  = float(traj.get(("next", "agents", "observation", "reward_benign_col_dmg")).mean().item())
        _r_qoe  = float(traj.get(("next", "agents", "observation", "reward_qoe_penalty")).mean().item())
        _vio    = float(traj.get(("next", "agents", "observation", "qoe_vio_rate")).mean().item())

        print(
            f"Iter={it:4d} | rew={reward_mean:+.4f} "
            f"| atk_pass={_r_lres:.3f} bcd={_r_bcd:.3f} qoe_sf={_r_qoe:.3f} "
            f"| qoe_vio={_vio:.1%}"
        )

        obs = traj.get(("agents", "observation", "obs"))
        obs_log = obs.clone()
        norm_t = env.transform[2]
        for k in ["cpu_to_ids_ratio", "ema_mom"]:
            idx = obs_keys.index(k)
            # If loc is [n_edges, obs_dim], we take [:, idx] to get [n_edges]
            # If loc is [obs_dim], we take [idx] to get a scalar (broadcasts across n_edges)
            if norm_t.loc.numel() == n_edges * obs_dim:
                loc   = norm_t.loc.view(n_edges, obs_dim)[:, idx].to(obs_log.device)
                scale = norm_t.scale.view(n_edges, obs_dim)[:, idx].to(obs_log.device)
            else:
                loc   = norm_t.loc[idx].to(obs_log.device)
                scale = norm_t.scale[idx].to(obs_log.device)
            obs_log[..., idx] = obs[..., idx] * scale + loc

        global_decision_step = wandb_log_obs_steps(
            obs_log, obs_keys, keep_keys={"cpu_to_ids_ratio", "ema_mom"}, global_step_start=global_decision_step
        )

        if (it + 1) % ckpt_every == 0:
            save_ckpt(ckpt_dir / f"ckpt_iter_{it+1:06d}.pt", policy, value_net, optim,
                      env_cfg, train_cfg, it + 1, device, env)

        ema_reward = reward_mean if ema_reward is None else ema_alpha * reward_mean + (1 - ema_alpha) * ema_reward
        ema_qoe    = qoe_mean    if ema_qoe    is None else ema_alpha * qoe_mean    + (1 - ema_alpha) * ema_qoe
        ema_vio    = _vio        if ema_vio    is None else ema_alpha * _vio        + (1 - ema_alpha) * ema_vio

        if ema_reward > best_reward:
            best_reward = ema_reward
            save_ckpt(ckpt_dir / "ckpt_best_reward.pt", policy, value_net, optim,
                      env_cfg, train_cfg, it + 1, device, env)
        if ema_qoe > best_qoe:
            best_qoe = ema_qoe
            save_ckpt(ckpt_dir / "ckpt_best_qoe.pt", policy, value_net, optim,
                      env_cfg, train_cfg, it + 1, device, env)
        if ema_vio < best_vio:
            best_vio = ema_vio
            save_ckpt(ckpt_dir / "ckpt_best_vio.pt", policy, value_net, optim,
                      env_cfg, train_cfg, it + 1, device, env)

        wandb.log(
            {
                "iter": it,
                "qoe/mean":              float(traj.get(("next", "agents", "observation", "qoe_mean")).mean().item()),
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
            }
        )
"""
Multi-Agent Reward Variants for Recurrent (LSTM) Training.
Includes both Centralised (CMA) and Independent (IMA) configurations.
"""

_Base = EdgeIDSParallelEnv


class EdgeIDSCMAParallelEnv(_Base):
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


class EdgeIDSIMAParallelEnv(_Base):
    """Independent Multi-Agent (IMA) reward variant."""
    def _compute_step_reward(self, lres, bcd, qsf, vio, attack_in):
        rew = -(self.alpha * qsf + self.beta * lres + self.gamma_r * bcd)
        return rew, lres, bcd, qsf, vio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cma", choices=["cma", "ima"])
    parser.add_argument("--cfg", type=str, default="./configs/simulation_0.yaml")
    args = parser.parse_args()

    env_cls = EdgeIDSCMAParallelEnv if args.mode == "cma" else EdgeIDSIMAParallelEnv
    train(env_class=env_cls, env_cfg_path=args.cfg)

# python train_ma_lstm.py --mode cma --cfg configs/simulation_ma_0.yaml
# python train_ma_lstm.py --mode ima --cfg configs/simulation_ma_0.yaml 