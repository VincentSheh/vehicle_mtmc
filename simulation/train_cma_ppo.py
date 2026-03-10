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
from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType, set_composite_lp_aggregate
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
            "local_num_req",
            "attack_in_rate",
            "cpu_to_ids_ratio",
            "ids_cpu_utilization",
            "total_cpu_to_ids_ratio",
            "ema_mom"
        ]
        # Current ids_cpu allocation included
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
        def _reactive_delta_vec() -> np.ndarray:
            # default: hold
            delta = np.zeros(self.n_edges, dtype=np.int64)

            if not self.env.history:
                return delta

            # use just the most recent per-edge block (same style as before)
            last_block = self.env.history[-self.n_edges:]
            df = pd.DataFrame([m.__dict__ for m in last_block])

            # thresholds you can tune
            util_hi = 0.8   # IDS is overloaded -> increase IDS CPU
            util_lo = 0.2   # IDS underutilized -> decrease IDS CPU

            for i, aid in enumerate(self.area_ids):
                g = df[df["area_id"] == aid]
                if g.empty:
                    continue

                util = float(g["ids_cpu_utilization"].iloc[-1])

                if util >= util_hi:
                    delta[i] = +1
                elif util <= util_lo:
                    delta[i] = -1
                else:
                    delta[i] = 0

            return delta             

        act_vec = np.zeros(self.n_edges, dtype=np.int64)
        for i, aid in enumerate(self.area_ids):
            a = int(actions[aid])
            if a < 0 or a > 2:
                raise ValueError(f"Invalid action {a} for agent {aid}, expected 0..2")
            act_vec[i] = a
            
        delta_cmd = (act_vec.astype(np.float32) - 1.0) * self.scale_step
        # delta_cmd = _reactive_delta_vec().astype(np.float32) * self.scale_step
        prev_ids = self.ids_cpu.copy()

        new_ids = self.ids_cpu + delta_cmd
        for i, edge in enumerate(self.env.edge_areas):
            max_ids = float(edge.budget.cpu) - 0.5
            new_ids[i] = float(np.clip(new_ids[i], self.ids_cpu_min, max_ids))
            # Constant
            # new_ids[i] = float(0.5)
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

    # def _build_reward_per_agent(self) -> np.ndarray:
    #     if len(self.env.history) < self.n_edges:
    #         return np.zeros(self.n_edges, dtype=np.float32)

    #     target_ids_cpu = 2.0
    #     r = np.zeros(self.n_edges, dtype=np.float32)
    #     for i, aid in enumerate(self.area_ids):
    #         r[i] = -abs(float(self.ids_cpu[i]) - target_ids_cpu)
    #     return r
    
    def _build_reward_per_agent(self) -> np.ndarray:
        if len(self.env.history) < self.n_edges:
            return np.zeros(self.n_edges, dtype=np.float32)

        # 1. Get local QoE for each edge (maintains area_id order)
        last_block = self.env.history[-self.n_edges:]
        q_local = np.asarray([float(m.qoe_mean) for m in last_block], dtype=np.float32)
        
        # 2. Calculate local penalty: how far is THIS specific agent below the threshold?
        # Use np.maximum for element-wise comparison
        viol = np.maximum(0.0, self.threshold - q_local)
        penalty = (self.alpha * (viol ** 2)).astype(np.float32)

        local_rewards = q_local - penalty
        # Return the average reward to all agents (Global Reward)
        global_rew = np.mean(local_rewards)
        return np.full(self.n_edges, global_rew, dtype=np.float32)
        
    def _qoe_vec(self) -> np.ndarray:
        # qoe = np.asarray(getattr(self.env, "final_qoe", 0.0), dtype=np.float32)
        # if qoe.ndim == 0:
        #     qoe = np.full((self.n_edges,), float(qoe), dtype=np.float32)
        # return qoe * 30.0
        
        if not getattr(self.env, "history", None) or len(self.env.history) == 0:
            return np.zeros(self.n_edges, dtype=np.float32)

        q_vec = np.zeros(self.n_edges, dtype=np.float32)

        for i, edge in enumerate(self.env.edge_areas):
            # all history entries for this edge over the whole episode
            h = [m for m in self.env.history if m.area_id == edge.area_id]
            if not h:
                q_vec[i] = 0.0
                continue

            q = np.asarray([float(m.qoe_mean) for m in h], dtype=np.float32)
            if q.size == 0:
                q_vec[i] = 0.0
                continue

            slo_thr = float(getattr(edge, "slo_threshold", self.threshold))
            slo_beta = float(getattr(edge, "slo_beta", self.alpha))

            viol_rate = float((q < slo_thr).mean())
            V_edge = float(np.exp(-slo_beta * viol_rate))

            q_vec[i] = float(q.mean()) * V_edge

        return q_vec


# =========================================================
# Transforms
# =========================================================

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


# =========================================================
# Networks (Pure MLPs)
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

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("agents", "observation", "obs"))
        # PyTorch natively handles arbitrary leading dimensions (B, T, E, D)
        td.set(("agents", "features"), self.net(obs))
        return td

class CentralizedCriticMLPCore(nn.Module):
    # We now look at the global entry or a flattened version of all agent obs
    in_keys = [("agents", "observation", "obs")] 
    out_keys = [("agents", "vf_features")]

    def __init__(self, n_agents: int, local_obs_dim: int, hidden_dim: int, device: str):
        super().__init__()
        # Input size is local_obs_dim * n_agents
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim * n_agents, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        ).to(device)
        self.n_agents = n_agents

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        # obs shape: [Batch, Time, Agents, Dim]
        obs = td.get(("agents", "observation", "obs"))
        
        # Create global state by flattening the Agent dimension into the Feature dimension
        # Result shape: [Batch, Time, Agents, Dim * Agents]
        # We broadcast the global state to all agents so each gets a centralized value estimate
        shape = obs.shape
        # Flatten last two dims: [B, T, E, D] -> [B, T, E*D]
        global_state = obs.view(*shape[:-2], -1) 
        
        # Expand back so each agent i has the full state as input
        # [B, T, E*D] -> [B, T, E, E*D]
        global_input = global_state.unsqueeze(-2).expand(*shape[:-1], global_state.shape[-1])
        
        td.set(("agents", "vf_features"), self.net(global_input))
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

    not_end = ~(done | terminated)
    not_end = not_end.to(values.dtype)

    B, T, E = reward.shape

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

    # 1. Get specs from a dummy env
    base = make_wrapped_env(cfg_path=cfg_path, seed=int(env_cfg["run"]["seed"]), decision_interval=decision_interval)
    obs_key = ("agents", "observation", "obs") # The specific key where data lives
    obs_spec = base.observation_spec[obs_key]
    n_edges = int(obs_spec.shape[-2])
    obs_dim = int(obs_spec.shape[-1])
    obs_dim_with_id = obs_dim + n_edges

    def make_one(i: int):
        def _make():
            return make_wrapped_env(cfg_path=cfg_path, seed=1000 + i, decision_interval=decision_interval)
        return _make

    penv = ParallelEnv(num_envs, [make_one(i) for i in range(num_envs)], device="cpu")

    # 2. Define transforms with correct key mapping
    norm_transform = ObservationNorm(
        in_keys=[obs_key],  # Changed from "obs_keys" to the actual nested key
        standard_normal=True,
    )

    transforms: List[Transform] = [
        InitTracker(),
        norm_transform,
        AddAgentID(n_edges=n_edges),
    ]

    env = TransformedEnv(penv, Compose(*transforms))

    # 3. Initialize stats
    on_cfg = train_cfg.get("observation_norm", {})
    env.transform.train()
    
    # reduce_dim should include (0, 1, 2) to reduce over [Batch, Time, Agents]
    # cat_dim is 0 to concatenate samples along the batch dimension during init
    norm_transform.init_stats(
        num_iter=int(on_cfg.get("num_iter", 100)),
        reduce_dim=tuple(on_cfg.get("reduce_dim", (0, 1, 2))), 
        cat_dim=0,
    )
    
    env.transform.eval()    
    td0 = env.reset()
    print("obs reset shape:", td0.get(obs_key).shape)

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

    # --- Actor ---
    actor_core = AgentMLPCore(
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

    # --- Critic ---
    critic_core = CentralizedCriticMLPCore(
        n_agents=n_edges, 
        local_obs_dim=obs_dim, 
        hidden_dim=hidden_dim * 2, # Global state usually requires more capacity
        device=device
    )
    critic_head = TensorDictModule(
        nn.Sequential(
            nn.Linear(hidden_dim * 2, 1).to(device),
            SqueezeLast() # Output strictly [..., E]
        ),
        in_keys=[("agents","vf_features")],
        out_keys=[("agents","state_value")],
    )

    value_net = TensorDictSequential(critic_core, critic_head)
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
        policy=collector_policy,
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

        # Force strict [B, T, E] shaping across all core tensors
        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated"), 
                  ("agents", "action"), ("agents", "sample_log_prob"), ("agents", "state_value")]:
            squeeze_last1(traj, k)
            if "next" in traj.keys(True, True):
                squeeze_last1(traj.get("next"), k)

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

        # --- Clean, Verified, Manual PPO Updates ---
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

                # 1. Forward pass
                actor_core(sub)
                actor_head(sub)
                value_net(sub)
                
                act = sub.get(("agents", "action")).long()
                old_logp = sub.get(("agents", "sample_log_prob"))

                # 2. Centralized Advantage Normalization
                adv = sub.get(("agents","advantage"))
                # Normalize across Batch, Time, AND Agents
                m = adv.mean()
                s = adv.std().clamp_min(1e-8)
                adv = (adv - m) / s

                # 3. Policy Loss
                logits = sub.get(("agents", "logits"))
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act)
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps_now, 1.0 + clip_eps_now) * adv
                policy_loss = -(torch.min(surr1, surr2)).mean()

                # 4. Critic Loss
                v_pred = sub.get(("agents", "state_value"))
                v_targ = sub.get(("agents", "value_target"))
                critic_loss = 0.5 * (v_targ - v_pred).pow(2).mean()

                # 5. Backprop
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
        
        if (it + 1) % ckpt_every == 0:
            save_ckpt(ckpt_dir / f"ckpt_iter_{it+1:06d}.pt", policy, value_net, optim, env_cfg, train_cfg, it + 1, device, env)

        if qoe_mean > best_qoe:
            best_qoe = qoe_mean
            save_ckpt(ckpt_dir / "ckpt_best.pt", policy, value_net, optim, env_cfg, train_cfg, it + 1, device, env)

        print(f"it={it} reward_mean={reward_mean:.4f}, qoe_mean={qoe_mean:.4f}")
        reward_mean_per_agent = traj.get(("agents", "reward")).mean(dim=(0, 1))  # [E]
        qoe_mean_per_agent = traj.get(("next", "agents", "observation", "qoe_mean")).mean(dim=(0, 1))  # [E,1] or [E]

        if qoe_mean_per_agent.ndim > 1:
            qoe_mean_per_agent = qoe_mean_per_agent.squeeze(-1)

        for i in range(n_edges):
            print(
                f"it={it} agent={i} reward_mean={reward_mean_per_agent[i].item():.4f} "
                f"qoe_mean={qoe_mean_per_agent[i].item():.4f}"
            )

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