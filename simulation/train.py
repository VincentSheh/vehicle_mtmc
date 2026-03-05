"""
Single-file MAPPO-ish (CTDE) PPO training script, refactored

Key points
- PettingZoo ParallelEnv wrapping your build_env_base()
- TorchRL PettingZooWrapper with group_map -> ("agents", ...) keys
- CTDE: decentralized actor on per-agent obs, centralized critic on flattened global obs
- Manual PPO update (no ClipPPOLoss()) to avoid TorchRL multi-agent shape pitfalls
- GAE computed manually (no ClipPPOLoss.value_estimator) and uses reward/done from ROOT keys
- Infos are {} to avoid TorchRL packing crashes
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
from torchrl.modules import MultiAgentMLP, ProbabilisticActor

from environment import build_env_base
from logger import wandb_init
import wandb

class FeatureNet(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

    
# =========================
# PettingZoo environment
# =========================

class EdgeIDSParallelEnv(PZooParallelEnv):
    """
    PettingZoo ParallelEnv around build_env_base().

    Agent id: area_id (string)
    obs: Box(obs_dim,)
    action: Discrete(3) interpreted as {-1,0,+1} * scale_step
    reward: per-agent QoE - penalty
    infos: always {} to avoid TorchRL packing issues
    """

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
            # "overhead",
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

        obs_mat = self._build_observation()          # [E, obs_dim]
        qoe = self._qoe_vec()                        # [E]

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
        # if qoe[i] > 0:
        #     print(qoe)
        
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
                if k in ("cpu_to_ids_ratio", "overhead"):
                    obs[i, j] = float(vals[-1])
                elif k == "ema_mom":
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
        return qoe * 30

# =========================
# TorchRL transforms
# =========================

class BuildCentralObs(Transform):
    """
    Adds root key "observation_flat" by flattening ("agents","observation") over (E, obs_dim).
    """

    def __init__(self, n_edges: int, obs_dim: int, out_key: str = "observation_flat"):
        super().__init__(in_keys=[("agents", "observation", "obs")], out_keys=[out_key])
        self.out_key = out_key
        self.flat_dim = int(n_edges * obs_dim)

    def _call(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("agents","observation","obs"), default=None)
        if obs is not None:
            td.set(self.out_key, obs.reshape(*obs.shape[:-2], self.flat_dim))
        return td

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase, **kwargs) -> TensorDictBase:
        obs = tensordict_reset.get(("agents","observation","obs"), default=None)
        if obs is not None:
            tensordict_reset.set(self.out_key, obs.reshape(*obs.shape[:-2], self.flat_dim))
        return tensordict_reset

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
    """
    Adds scalar root done/terminated/truncated. Useful for some collectors, still keeps agent-level done intact.
    """

    def __init__(self):
        super().__init__(in_keys=[("agents", "done"), ("agents", "terminated")], out_keys=["done", "terminated", "truncated"])

    @staticmethod
    def _agent_dim(x: torch.Tensor) -> int:
        return -2 if (x.ndim >= 2 and x.shape[-1] == 1) else -1

    @staticmethod
    def _to_root_flag(x: torch.Tensor) -> torch.Tensor:
        adim = BuildSharedDone._agent_dim(x)
        flag = x.to(torch.bool).any(dim=adim, keepdim=False)
        if flag.ndim == 0:
            flag = flag.view(1)
        if flag.shape[-1] != 1:
            flag = flag.unsqueeze(-1)
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

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase, **kwargs) -> TensorDictBase:
        done_a = tensordict_reset.get(("agents", "done"), default=None)
        term_a = tensordict_reset.get(("agents", "terminated"), default=None)
        if done_a is not None and term_a is not None:
            tensordict_reset.set("done", self._to_root_flag(done_a))
            tensordict_reset.set("terminated", self._to_root_flag(term_a))
            tensordict_reset.set("truncated", torch.zeros_like(tensordict_reset.get("done")))
            return tensordict_reset

        bs = tuple(tensordict_reset.batch_size)
        shape = (*bs, 1) if len(bs) else (1,)
        dev = tensordict_reset.device
        tensordict_reset.set("done", torch.zeros(shape, dtype=torch.bool, device=dev))
        tensordict_reset.set("terminated", torch.zeros(shape, dtype=torch.bool, device=dev))
        tensordict_reset.set("truncated", torch.zeros(shape, dtype=torch.bool, device=dev))
        return tensordict_reset


# =========================
# Models
# =========================

class SqueezeLast(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1)


def build_policy_value_ctde(n_edges: int, obs_dim: int, n_actions: int, hidden_dim: int, device: str):
    # Actor: decentralized, uses per-agent obs
    actor_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=n_actions,
        n_agents=n_edges,
        centralised=False,
        share_params=True,
        device=device,
        depth=3,
        num_cells=hidden_dim,
        activation_class=torch.nn.Tanh,
    )
    actor = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation", "obs")],
        out_keys=[("agents", "logits")],
    )
    policy = ProbabilisticActor(
        module=actor,
        in_keys=[("agents", "logits")],
        out_keys=[("agents", "action")],
        distribution_class=Categorical,
        distribution_kwargs={},
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    # Critic: centralized, uses flattened global obs then outputs per-agent values
    critic_net = nn.Sequential(
        nn.Linear(n_edges * obs_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, n_edges),
    ).to(device)

    value = TensorDictSequential(
        TensorDictModule(
            critic_net,
            in_keys=["observation_flat"],            # CTDE explicit
            out_keys=[("agents", "state_value")],    # [..., E]
        ),
    )
    return policy, value


# =========================
# Env builders
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


def build_env_stack(env_cfg: dict, cfg_path: str, num_envs: int, train_cfg: dict):
    decision_interval = int(env_cfg["globals"]["decision_interval"])

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

    # --- build transforms ---
    transforms: List[Transform] = [
        InitTracker(),
        BuildCentralObs(n_edges=n_edges, obs_dim=obs_dim, out_key="observation_flat"),
        BuildSharedDone(),
    ]

    # --- OPTIONAL: ObservationNorm on observation_flat (no running mean) ---
    use_obsnorm = bool(train_cfg.get("use_observation_norm", True))
    if use_obsnorm:
        on_cfg = train_cfg.get("observation_norm", {})
        transforms.append(
            ObservationNorm(
                in_keys=["observation_flat"],
                standard_normal=bool(on_cfg.get("standard_normal", True)),
                # eps can be set if you want: eps=float(on_cfg.get("eps", 1e-5))
            )
        )

    env = TransformedEnv(penv, Compose(*transforms))

    # --- populate mean/std ONCE from rollouts ---
    if use_obsnorm:
        on_cfg = train_cfg.get("observation_norm", {})
        num_iter = int(on_cfg.get("num_iter", 100))
        reduce_dim = tuple(on_cfg.get("reduce_dim", (0, 1)))
        cat_dim = int(on_cfg.get("cat_dim", 0))

        env.transform.train()
        # ObservationNorm is last in Compose
        env.transform[-1].init_stats(
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
# PPO / GAE helpers
# =========================

def squeeze_last1(td: TensorDictBase, key):
    if key in td.keys(True, True):
        x = td.get(key)
        if isinstance(x, torch.Tensor) and x.ndim >= 1 and x.shape[-1] == 1:
            td.set(key, x.squeeze(-1))


def ensure_mbE_2d(x: torch.Tensor, n_edges: int) -> torch.Tensor:
    if x.ndim == 2 and x.shape[0] == n_edges and x.shape[1] != n_edges:
        return x.t().contiguous()
    return x.contiguous()


@torch.no_grad()
def compute_gae_inplace(
    traj: TensorDictBase,
    gamma: float,
    lmbda: float,
    n_edges: int,
    adv_key=("agents", "advantage"),
    vt_key=("agents", "value_target"),
):
    """
    Expects in traj:
      - ("agents","reward"), ("agents","done"), ("agents","terminated") at ROOT
      - ("agents","state_value") at ROOT from value(traj)
      - next ("agents","state_value") in traj["next"] from value(traj["next"])
    Shapes:
      reward, done, terminated: [B,T,E]
      values: [B,T,E]
      next_values: [B,T,E] (aligned with next step in traj["next"])
    """

    reward = traj.get(("agents", "reward"))
    done = traj.get(("agents", "done")).to(torch.bool)
    terminated = traj.get(("agents", "terminated")).to(torch.bool)
    values = traj.get(("agents", "state_value"))
    next_values = traj.get("next").get(("agents", "state_value"))

    # not_done is typical PPO style: bootstrap unless episode ended
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


def flatten_BT_to_N(traj: TensorDictBase) -> TensorDictBase:
    # [B,T,...] -> [N,...]
    return traj.reshape(-1)


def apply_anneal(
    optim: torch.optim.Optimizer,
    base_lr: float,
    base_clip_eps: float,
    alpha: float,
    train_cfg: dict,
):
    # alpha in [0,1], usually 1 -> 0 over training
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

    env, n_edges, obs_dim = build_env_stack(env_cfg, env_cfg_path, num_envs, train_cfg)

    # keep per-agent log_prob, do not aggregate across agents
    set_composite_lp_aggregate(False).set()

    n_actions = int(train_cfg["model"]["n_actions"])
    hidden_dim = int(train_cfg["model"]["hidden_dim"])
    policy, value = build_policy_value_ctde(n_edges, obs_dim, n_actions, hidden_dim, device)

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

    clip_eps = float(train_cfg["loss"]["clip_epsilon"])
    entropy_coeff = float(train_cfg["loss"]["entropy_coeff"])
    critic_coeff = float(train_cfg["loss"]["critic_coeff"])
    gamma = float(train_cfg["loss"]["gamma"])
    gae_lambda = float(train_cfg["loss"]["gae_lambda"])
    
    base_lr = float(train_cfg["optim"]["lr"])
    base_clip_eps = float(train_cfg["loss"]["clip_epsilon"])

    total_network_updates = (
        int(train_cfg["collector"]["total_frames"]) // int(train_cfg["collector"]["frames_per_batch"])
    ) * int(train_cfg["loss"]["ppo_epochs"]) * math.ceil(
        (frames_per_batch) / int(train_cfg["loss"]["mini_batch_size"])
    )
    num_network_updates = 0 
    
    ckpt_dir = Path("checkpoints") / run.name
    ckpt_every = 50
    best_qoe = -1e9       

    for it, batch in enumerate(collector):
        traj = batch.clone(False)

        # squeeze last singleton if exists
        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated"), ("agents", "state_value")]:
            squeeze_last1(traj, k)
            squeeze_last1(traj.get("next"), k)

        # enforce bool
        traj.set(("agents", "done"), traj.get(("agents", "done")).to(torch.bool))
        traj.set(("agents", "terminated"), traj.get(("agents", "terminated")).to(torch.bool))
        traj.get("next").set(("agents", "done"), traj.get("next").get(("agents", "done")).to(torch.bool))
        traj.get("next").set(("agents", "terminated"), traj.get("next").get(("agents", "terminated")).to(torch.bool))
        
        
        # IMPORTANT: copy reward/done/terminated from "next" to ROOT before computing GAE
        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated")]:
            if k not in traj.keys(True, True) and ("next",) + k in traj.keys(True, True):
                traj.set(k, traj.get(("next",) + k))

        with torch.no_grad():
            value(traj)
            value(traj.get("next"))
            compute_gae_inplace(traj, gamma=gamma, lmbda=gae_lambda, n_edges=n_edges)

        if it == 0:
            print(traj.keys)
            print("reward", traj.get(("agents", "reward")).shape)
            print("value ", traj.get(("agents", "state_value")).shape)
            print("done  ", traj.get(("agents", "done")).shape)
            print("term  ", traj.get(("agents", "terminated")).shape)

        flat = flatten_BT_to_N(traj)
        N = flat.batch_size[0]

        last_total_loss = last_policy_loss = last_critic_loss = last_entropy = None

        for _ in range(ppo_epochs):
            perm = torch.randperm(N, device=flat.device)
            for start in range(0, N, minibatch_size):
                alpha = 1.0 - (num_network_updates / max(1, total_network_updates))
                if alpha < 0.0:
                    alpha = 0.0

                clip_eps_now = apply_anneal(
                    optim=optim,
                    base_lr=base_lr,
                    base_clip_eps=base_clip_eps,
                    alpha=alpha,
                    train_cfg=train_cfg,
                )

                num_network_updates += 1                
                idx = perm[start : start + minibatch_size]
                sub = flat[idx].clone()

                actions = ensure_mbE_2d(sub.get(("agents", "action")).long(), n_edges)           # [mb,E]
                old_logp = ensure_mbE_2d(sub.get(("agents", "sample_log_prob")), n_edges)       # [mb,E]
                adv = ensure_mbE_2d(sub.get(("agents", "advantage")), n_edges)                  # [mb,E]
                v_targ = ensure_mbE_2d(sub.get(("agents", "value_target")), n_edges)            # [mb,E]

                # distribution from current policy
                dist = policy.get_dist(sub)
                base = getattr(dist, "base_dist", dist)
                new_logp = ensure_mbE_2d(base.log_prob(actions), n_edges)                        # [mb,E]
                entropy = ensure_mbE_2d(base.entropy(), n_edges)                                 # [mb,E]

                # PPO ratio and clipped surrogate
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps_now, 1.0 + clip_eps_now) * adv
                policy_loss = -(torch.min(surr1, surr2)).mean()

                # critic loss (CTDE uses observation_flat)
                sub = value(sub)
                v_pred = ensure_mbE_2d(sub.get(("agents", "state_value")), n_edges)
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
        qoe_mean = float(traj.get(("next","agents","observation","qoe_mean")).mean().item())
        if (it + 1) % ckpt_every == 0:
            save_ckpt(
                ckpt_dir / f"ckpt_iter_{it+1:06d}.pt",
                policy, value, optim, env_cfg, train_cfg, it + 1, device, env,
            )

        if qoe_mean > best_qoe:
            best_qoe = qoe_mean
            save_ckpt(
                ckpt_dir / "ckpt_best.pt",
                policy, value, optim, env_cfg, train_cfg, it + 1, device, env,
            )        
        print(traj.get(("agents","observation","qoe_mean")).shape)
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
                "train/alpha": alpha, "train/lr": optim.param_groups[0]["lr"], "train/clip_eps": clip_eps_now
            }
        )


if __name__ == "__main__":
    train()