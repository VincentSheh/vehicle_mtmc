import yaml
import math
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import pandas as pd
from typing import List, Optional
from tensordict import TensorDict

from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.distributions import Categorical
from environment import build_env_base, VA_CPU_RESERVE
from torchrl.envs.transforms import ObservationNorm, TransformedEnv
from torchrl.envs import ParallelEnv, EnvBase
from torchrl.modules import LSTMModule
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import InitTracker
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from logger import *

from pathlib import Path

# First N_TEMPORAL_PER_EDGE features per edge (local_num_req, attack_in_rate)
# are fed through the LSTM. The remaining features bypass directly to the
# actor/critic heads.
N_TEMPORAL_PER_EDGE = 2


class IndepTorchRLEnvWrapper(EnvBase):
    """
    Independent Multi-Agent (IMA) TorchRL EnvBase wrapper.
    Treats each edge as an independent agent sharing a single batch dimension.
    """

    def __init__(
        self,
        cfg_path: str,
        decision_interval: int = 300,
        n_actions: int = 9,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.env = build_env_base(cfg_path)
        self.n_edges = len(self.env.edge_areas)
        
        # Batch size is [n_edges] so agents are treated as independent parallel environments
        super().__init__(device=torch.device(device), batch_size=torch.Size([self.n_edges]))

        self.area_ids = [e.area_id for e in self.env.edge_areas]
        self.episode_id = 0
        self.base_seed = seed
        self.decision_interval = int(decision_interval)
        self.n_actions = int(n_actions)

        _ext = n_actions != 3
        self.obs_keys = [
            "local_num_req",
            "attack_in_rate",
            "ema_mom",
            "cpu_to_ids_ratio",
            "ids_cpu_utilization",
            "neighbor_ids_util",
            *( ["neighbor_delta"] if _ext else [] ),
            "neighbor_atk_rate",
            "prev_slo_vio",
            *( ["transition_ticks_norm", "delta_in_flight_norm", "queue_ahead_norm"] if _ext else [] ),
        ]

        self.obs_dim = len(self.obs_keys)

        _cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.scaling_time_steps: List[int] = list(
            _cfg["globals"].get("scaling_time_step", [300, 450, 498, 544])
        )
        self.scaling_quanta = [0.5, 1.0, 1.5, 2.0]
        self.scale_step = 0.5

        _reward_cfg = _cfg["globals"].get("reward", {})
        self.reward_alpha = float(_reward_cfg.get("alpha_inv", 0.10))
        self.reward_beta  = float(_reward_cfg.get("beta_inv",  0.20))
        self.reward_gamma = float(_reward_cfg.get("gamma_inv", 0.12))
        self.reward_q_th  = float(_reward_cfg.get("q_th", 0.20))

        self.ids_cpu = torch.zeros(self.n_edges, device=self.device)
        for i, e in enumerate(self.env.edge_areas):
            self.ids_cpu[i] = e.ids_cpu
            
        self.ids_cpu_settled = self.ids_cpu.clone()
        self.ids_cpu_target = self.ids_cpu.clone()
        self.transition_ticks_remaining = torch.zeros(self.n_edges, dtype=torch.int32, device=self.device)
        self.transition_ticks_total = torch.ones(self.n_edges, dtype=torch.int32, device=self.device)

        self._make_specs()

    def _make_specs(self):
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self.n_edges, self.obs_dim),
                dtype=torch.float32,
                device=self.device,
            ),
            # extra keys that survive rollout collection
            qoe_mean=UnboundedContinuousTensorSpec(
                shape=(self.n_edges, 1), dtype=torch.float32, device=self.device,
            ),
            reward_lambda_res=UnboundedContinuousTensorSpec(
                shape=(self.n_edges, 1), dtype=torch.float32, device=self.device,
            ),
            reward_benign_col_dmg=UnboundedContinuousTensorSpec(
                shape=(self.n_edges, 1), dtype=torch.float32, device=self.device,
            ),
            reward_qoe_penalty=UnboundedContinuousTensorSpec(
                shape=(self.n_edges, 1), dtype=torch.float32, device=self.device,
            ),
            qoe_vio_rate=UnboundedContinuousTensorSpec(
                shape=(self.n_edges, 1), dtype=torch.float32, device=self.device,
            ),
            shape=self.batch_size,
        )

        self.action_spec = CompositeSpec(
            action=DiscreteTensorSpec(
                n=self.n_actions,
                shape=(self.n_edges,),
                device=self.device,
            ),
            shape=self.batch_size,
        )

        self.reward_spec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec(
                shape=(self.n_edges, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=self.batch_size,
        )

        self.done_spec = CompositeSpec(
            done=BoundedTensorSpec(
                low=0, high=1, shape=(self.n_edges, 1), dtype=torch.bool, device=self.device,
            ),
            terminated=BoundedTensorSpec(
                low=0, high=1, shape=(self.n_edges, 1), dtype=torch.bool, device=self.device,
            ),
            truncated=BoundedTensorSpec(
                low=0, high=1, shape=(self.n_edges, 1), dtype=torch.bool, device=self.device,
            ),
            shape=self.batch_size,
        )

    def _reset(self, tensordict=None):
        self.episode_id += 1
        episode_seed = self.base_seed + self.episode_id * 1000
        torch.manual_seed(episode_seed)
        self.env.reset(episode_seed)

        self.ids_cpu = torch.zeros(self.n_edges, device=self.device)
        for i, e in enumerate(self.env.edge_areas):
            self.ids_cpu[i] = e.ids_cpu

        self.ids_cpu_settled = self.ids_cpu.clone()
        self.ids_cpu_target = self.ids_cpu.clone()
        self.transition_ticks_remaining.fill_(0)
        self.transition_ticks_total.fill_(1)

        obs = self._build_observation()
        _zeroE1 = torch.zeros((self.n_edges, 1), dtype=torch.float32, device=self.device)
        
        return TensorDict(
            {
                "observation": obs,
                "qoe_mean":              _zeroE1.clone(),
                "reward_lambda_res":     _zeroE1.clone(),
                "reward_benign_col_dmg": _zeroE1.clone(),
                "reward_qoe_penalty":    _zeroE1.clone(),
                "qoe_vio_rate":          _zeroE1.clone(),
                "done":       torch.zeros((self.n_edges, 1), dtype=torch.bool, device=self.device),
                "terminated": torch.zeros((self.n_edges, 1), dtype=torch.bool, device=self.device),
                "truncated":  torch.zeros((self.n_edges, 1), dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _lookup_scaling_duration(self, magnitude: float) -> int:
        for i, q in enumerate(self.scaling_quanta):
            if magnitude <= q + 1e-9:
                return self.scaling_time_steps[i]
        return self.scaling_time_steps[-1]

    def _set_seed(self, seed: Optional[int]):
        if seed is None:
            return None
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        return seed

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]  # [n_edges]
        delta_cmd = (action.float() - ((self.n_actions - 1) / 2.0)) * self.scale_step

        for i in range(self.n_edges):
            ids_cpu_max = float(self.env.edge_areas[i].budget.cpu - VA_CPU_RESERVE)
            prev = self.ids_cpu[i].item()
            
            _settled = float(self.ids_cpu_settled[i].item())
            _max_q   = float(self.scaling_quanta[-1])
            self.ids_cpu[i] = torch.clamp(
                self.ids_cpu[i] + delta_cmd[i], 
                min=max(0.5, _settled - _max_q), 
                max=min(ids_cpu_max, _settled + _max_q)
            )
            delta_eff = float(self.ids_cpu[i].item() - prev)

            if self.transition_ticks_remaining[i] <= 0 and abs(delta_eff) > 1e-9:
                self.ids_cpu_target[i] = self.ids_cpu[i]
                gap = abs(float(self.ids_cpu_target[i].item()) - float(self.ids_cpu_settled[i].item()))
                dur = self._lookup_scaling_duration(gap)
                self.transition_ticks_total[i]     = dur
                self.transition_ticks_remaining[i] = dur

        total_reward = torch.zeros((self.n_edges, 1), device=self.device)
        comp_lres = torch.zeros((self.n_edges, 1), device=self.device)
        comp_bcd = torch.zeros((self.n_edges, 1), device=self.device)
        comp_qsf = torch.zeros((self.n_edges, 1), device=self.device)
        comp_vio = torch.zeros((self.n_edges, 1), device=self.device)
        
        terminated_flag = False
        steps = 0

        for _ in range(self.decision_interval):
            ids_cpu_eff = self.ids_cpu_settled.detach().cpu().numpy()
            step_overhead = np.zeros(self.n_edges, dtype=np.float32)

            for i in range(self.n_edges):
                if self.transition_ticks_remaining[i] > 0:
                    delta_to_settled = float(self.ids_cpu_target[i].item()) - float(self.ids_cpu_settled[i].item())
                    if abs(delta_to_settled) > 1e-9:
                        step_overhead[i] = -delta_to_settled

                    self.transition_ticks_remaining[i] -= 1
                    if self.transition_ticks_remaining[i] == 0:
                        self.ids_cpu_settled[i] = self.ids_cpu_target[i]
                        # Bug 1 fix: start a new transition for any queued command
                        queued = float(self.ids_cpu[i].item()) - float(self.ids_cpu_settled[i].item())
                        if abs(queued) > 1e-9:
                            self.ids_cpu_target[i] = self.ids_cpu[i]
                            dur = self._lookup_scaling_duration(abs(queued))
                            self.transition_ticks_total[i] = dur
                            self.transition_ticks_remaining[i] = dur
                            new_d = float(self.ids_cpu_target[i].item()) - float(self.ids_cpu_settled[i].item())
                            step_overhead[i] = -new_d if abs(new_d) > 1e-9 else 0.0
                        else:
                            step_overhead[i] = 0.0
                        # Bug 2 fix: apply newly settled value within the same tick
                        ids_cpu_eff[i] = float(self.ids_cpu_settled[i].item())

            self.env.step(ids_cpu_eff.tolist(), step_overhead.tolist())
            
            # IMA reward computation per edge
            r_dict = self._build_ima_reward()
            total_reward += r_dict["reward"]
            comp_lres += r_dict["lres"]
            comp_bcd += r_dict["bcd"]
            comp_qsf += r_dict["qsf"]
            comp_vio += r_dict["vio"]
            
            steps += 1
            if self.env.t >= self.env.t_max:
                terminated_flag = True
                break

        n = max(1, steps)
        obs = self._build_observation()
        
        terminated = torch.full((self.n_edges, 1), terminated_flag, dtype=torch.bool, device=self.device)
        truncated = torch.zeros((self.n_edges, 1), dtype=torch.bool, device=self.device)
        
        # Get latest QoE from history
        qoe_vec = torch.zeros((self.n_edges, 1), device=self.device)
        if self.env.history:
            last_block = self.env.history[-self.n_edges:]
            for i, aid in enumerate(self.area_ids):
                for m in reversed(last_block):
                    if m.area_id == aid:
                        qoe_vec[i, 0] = float(m.qoe_mean)
                        break

        return TensorDict(
            {
                "observation": obs,
                "reward": total_reward / n,
                "qoe_mean": qoe_vec,
                "reward_lambda_res": comp_lres / n,
                "reward_benign_col_dmg": comp_bcd / n,
                "reward_qoe_penalty": comp_qsf / n,
                "qoe_vio_rate": comp_vio / n,
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _build_observation(self) -> torch.Tensor:
        obs = torch.zeros((self.n_edges, self.obs_dim), dtype=torch.float32, device=self.device)
        if not self.env.history:
            return obs

        records = self.env.history[-self.decision_interval * self.n_edges:]
        df = pd.DataFrame([m.__dict__ for m in records])

        # 1. Local metrics
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
                    obs[i, idx] = float(np.mean(vals_nz)) if len(vals_nz) > 0 else 0.0
                else:
                    obs[i, idx] = float(np.mean(vals))

        # 2. Neighbor metrics
        if self.n_edges > 1:
            edge_ids_util = {}
            edge_atk_rate = {}
            for aid in self.area_ids:
                g = df[df["area_id"] == aid]
                if g.empty:
                    edge_ids_util[aid] = 0.0
                    edge_atk_rate[aid] = 0.0
                else:
                    edge_ids_util[aid] = float(np.clip(np.mean(g["ids_cpu_utilization"].values), 0.0, 1.0))
                    edge_atk_rate[aid] = float(np.mean(g["attack_in_rate"].values))

            max_delta = self.scale_step * (self.n_actions - 1) / 2.0
            for i, area_id in enumerate(self.area_ids):
                nbr_utils = []
                nbr_deltas = []
                nbr_atk = []
                for j in range(self.n_edges):
                    if j == i: continue
                    other_id = self.area_ids[j]
                    nbr_utils.append(edge_ids_util[other_id])
                    nbr_atk.append(edge_atk_rate[other_id])
                    delta = float(self.ids_cpu_target[j].item()) - float(self.ids_cpu_settled[j].item())
                    nbr_deltas.append(float(np.clip(delta / max(max_delta, 1e-6), -1.0, 1.0)))

                obs[i, self.obs_keys.index("neighbor_ids_util")] = float(np.mean(nbr_utils)) if nbr_utils else 0.0
                if "neighbor_delta" in self.obs_keys:
                    obs[i, self.obs_keys.index("neighbor_delta")] = float(np.mean(nbr_deltas)) if nbr_deltas else 0.0
                obs[i, self.obs_keys.index("neighbor_atk_rate")] = float(np.mean(nbr_atk)) if nbr_atk else 0.0

        # 2b. SLO violation flag (local signal, valid for single- and multi-edge)
        for i, area_id in enumerate(self.area_ids):
            g = df[df["area_id"] == area_id]
            if not g.empty:
                last_qoe = float(g["qoe_mean"].values[-1])
                threshold = float(self.env.edge_areas[i].slo_threshold)
                obs[i, self.obs_keys.index("prev_slo_vio")] = 1.0 if last_qoe < threshold else 0.0

        # 3. Scaling features (omitted when n_actions == 3)
        if "transition_ticks_norm" in self.obs_keys:
            max_dur = float(self.scaling_time_steps[-1])
            max_delta = self.scale_step * (self.n_actions - 1) / 2.0
            for i in range(self.n_edges):
                obs[i, self.obs_keys.index("transition_ticks_norm")] = float(self.transition_ticks_remaining[i].item()) / max(max_dur, 1.0)
                dif = float(self.ids_cpu_target[i].item()) - float(self.ids_cpu_settled[i].item())
                obs[i, self.obs_keys.index("delta_in_flight_norm")] = float(np.clip(dif / max(max_delta, 1e-6), -1.0, 1.0))
                qa = float(self.ids_cpu[i].item()) - float(self.ids_cpu_target[i].item())
                obs[i, self.obs_keys.index("queue_ahead_norm")] = float(np.clip(qa / max(max_delta, 1e-6), -1.0, 1.0))

        return obs

    def _build_ima_reward(self) -> dict:
        last_block = self.env.history[-self.n_edges:]
        rew = torch.zeros((self.n_edges, 1), device=self.device)
        lres_vec = torch.zeros((self.n_edges, 1), device=self.device)
        bcd_vec = torch.zeros((self.n_edges, 1), device=self.device)
        qsf_vec = torch.zeros((self.n_edges, 1), device=self.device)
        vio_vec = torch.zeros((self.n_edges, 1), device=self.device)

        for i, aid in enumerate(self.area_ids):
            m = None
            for record in reversed(last_block):
                if record.area_id == aid:
                    m = record
                    break
            if m is None: continue
            
            attack_pass = max(0.0, m.attack_in_rate - m.attack_drop_rate)
            lres = attack_pass / m.attack_in_rate if m.attack_in_rate > 1e-6 else 0.0
            qsf = max(0.0, self.reward_q_th - m.qoe_mean) / max(self.reward_q_th, 1e-6)
            vio = 1.0 if m.qoe_mean < self.reward_q_th else 0.0
            
            rew[i, 0] = -(self.reward_alpha * qsf + self.reward_beta * lres + self.reward_gamma * m.benign_col_dmg)
            lres_vec[i, 0] = lres
            bcd_vec[i, 0] = m.benign_col_dmg
            qsf_vec[i, 0] = qsf
            vio_vec[i, 0] = vio
            
        return {"reward": rew, "lres": lres_vec, "bcd": bcd_vec, "qsf": qsf_vec, "vio": vio_vec}


class SplitObsModule(nn.Module):
    def __init__(self, temporal_idx: list, static_idx: list):
        super().__init__()
        self.register_buffer("t_idx", torch.tensor(temporal_idx, dtype=torch.long))
        self.register_buffer("s_idx", torch.tensor(static_idx, dtype=torch.long))

    def forward(self, obs: torch.Tensor):
        return obs[..., self.t_idx], obs[..., self.s_idx]


class MergeModule(nn.Module):
    def forward(self, lstm_out: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return torch.cat([lstm_out, static], dim=-1)


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


def assert_finite(td, prefix=""):
    for k in td.keys(True, True):
        v = td.get(k)
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            bad = v[~torch.isfinite(v)]
            print(prefix, "NON-FINITE at key:", k, "example:", bad.flatten()[:5])
            raise RuntimeError(f"NaN/Inf in {k}")


def orthogonal_init(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

import argparse

def train(env_cfg_path="./configs/simulation_0.yaml", train_cfg_path="./configs/train.yaml", device="cuda", total_frames_override=None, resume_ckpt: str = None, ckpt_dir: str = None):
    with open(env_cfg_path, "r") as f:
        env_cfg = yaml.safe_load(f)
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    if total_frames_override:
        train_cfg["collector"]["total_frames"] = total_frames_override

    # Level override from train_cfg
    atk_lvl  = train_cfg.get("env", {}).get("atk_level", "default")
    user_lvl = train_cfg.get("env", {}).get("user_level", "default")

    if "attack_sampler" in env_cfg.get("globals", {}):
        env_cfg["globals"]["attack_sampler"]["level"] = atk_lvl
    if "user_sampler" in env_cfg.get("globals", {}) and "synthetic" in env_cfg["globals"]["user_sampler"]:
        env_cfg["globals"]["user_sampler"]["synthetic"]["level"] = user_lvl

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(env_cfg, tmp)
        env_cfg_path_tmp = tmp.name

    logger_cfg = train_cfg.setdefault("logger", {})
    if "exp_name" not in logger_cfg:
        logger_cfg["exp_name"] = "ppo_indep"

    try:
        t_max = env_cfg["run"]["t_max"]
        run = wandb_init(env_cfg, train_cfg)
        wandb.config.update({"env/atk_level": atk_lvl, "env/user_level": user_lvl}, allow_val_change=True)

        torch.manual_seed(env_cfg["run"]["seed"])
        np.random.seed(env_cfg["run"]["seed"])

        decision_interval = env_cfg["globals"]["decision_interval"]
        num_envs = train_cfg["collector"]["num_envs"]
        decisions_per_episode = math.ceil(t_max / decision_interval)

        _phase = "_phase2" if resume_ckpt else "_phase1"
        ckpt_dir = Path(ckpt_dir) if ckpt_dir else Path("checkpoints") / _phase / run.name
        ckpt_every = 50
        best_qoe = -1e9

        feature_dim = train_cfg["model"]["hidden_dim"]
        seq_len = int(train_cfg["loss"].get("seq_len", 8))

        # Build base env for info
        base_env = IndepTorchRLEnvWrapper(
            cfg_path=env_cfg_path_tmp,
            n_actions=train_cfg["model"]["n_actions"],
            seed=env_cfg["run"]["seed"],
            device=device,
            decision_interval=decision_interval,
        )
        n_edges = base_env.n_edges
        obs_dim = base_env.obs_dim

        temporal_idx = [i for i, k in enumerate(base_env.obs_keys) if i < N_TEMPORAL_PER_EDGE]
        static_idx = [i for i, k in enumerate(base_env.obs_keys) if i >= N_TEMPORAL_PER_EDGE]

        lstm = LSTMModule(
            input_size=len(temporal_idx),
            hidden_size=feature_dim,
            in_key="temporal_obs",
            out_key="lstm_out",
            device=device,
        )

        def make_env(seed_offset):
            return lambda: IndepTorchRLEnvWrapper(
                cfg_path=env_cfg_path_tmp,
                n_actions=train_cfg["model"]["n_actions"],
                seed=seed_offset,
                device="cpu",
                decision_interval=decision_interval,
            )

        penv = ParallelEnv(num_envs, [make_env(1000 + i) for i in range(num_envs)], device="cpu")

        env = TransformedEnv(
            penv,
            Compose(
                InitTracker(),
                lstm.make_tensordict_primer(),
                ObservationNorm(in_keys=["observation"], standard_normal=True),
            ),
        )
        
        env.transform.train()
        env.transform[-1].init_stats(num_iter=100, reduce_dim=(0, 1, 2), cat_dim=0)
        env.transform.eval()

        split_module = TensorDictModule(
            SplitObsModule(temporal_idx, static_idx).to(device),
            in_keys=["observation"],
            out_keys=["temporal_obs", "static_obs"],
        )
        merge_module = TensorDictModule(MergeModule(), in_keys=["lstm_out", "static_obs"], out_keys=["features_merged"])
        shared_core = TensorDictSequential(split_module, lstm, merge_module)

        merged_dim = feature_dim + len(static_idx)
        actor_head = TensorDictModule(nn.Linear(merged_dim, train_cfg["model"]["n_actions"]).to(device), in_keys=["features_merged"], out_keys=["logits"])
        critic_head = TensorDictModule(nn.Linear(merged_dim, 1).to(device), in_keys=["features_merged"], out_keys=["state_value"])

        # Orthogonal Initialization
        actor_head.apply(orthogonal_init)
        critic_head.apply(orthogonal_init)
        # Apply orthogonal init to shared_core (LSTM and bypass modules)
        shared_core.apply(orthogonal_init)

        collector_policy = ProbabilisticActor(
            module=TensorDictSequential(shared_core, actor_head),
            in_keys=["logits"], out_keys=["action"],
            distribution_class=Categorical, return_log_prob=True,
        )
        loss_actor = ProbabilisticActor(module=actor_head, in_keys=["logits"], out_keys=["action"], distribution_class=Categorical, return_log_prob=True)
        value = critic_head

        adv = GAE(gamma=train_cfg["loss"]["gamma"], lmbda=train_cfg["loss"]["gae_lambda"], value_network=value)
        adv.set_keys(value="state_value", advantage="advantage", value_target="value_target", reward="reward", done="done", terminated="terminated")

        loss = ClipPPOLoss(
            actor_network=loss_actor, critic_network=value,
            clip_epsilon=train_cfg["loss"]["clip_epsilon"],
            entropy_bonus=True, entropy_coef=train_cfg["loss"]["entropy_coeff"],
            critic_coef=train_cfg["loss"]["critic_coeff"],
            loss_critic_type=train_cfg["loss"]["loss_critic_type"],
            normalize_advantage=True,
        )
        loss.set_keys(value="state_value", advantage="advantage", value_target="value_target")

        frames_per_batch = train_cfg["collector"].get("frames_per_batch", decisions_per_episode * num_envs * n_edges)
        total_frames = train_cfg["collector"]["total_frames"]

        collector = SyncDataCollector(
            env, policy=collector_policy, frames_per_batch=frames_per_batch,
            total_frames=total_frames, device=device, trust_policy=True, split_trajs=False,
        )

        optim = torch.optim.Adam(
            list(shared_core.parameters()) + list(actor_head.parameters()) + list(critic_head.parameters()),
            lr=train_cfg["optim"]["lr"], weight_decay=train_cfg["optim"]["weight_decay"], eps=train_cfg["optim"]["eps"],
        )

        ppo_epochs = train_cfg["loss"]["ppo_epochs"]
        minibatch_size = train_cfg["loss"]["mini_batch_size"]
        iters_total = max(1, total_frames // frames_per_batch)
        num_network_updates = 0
        global_decision_step = 0

        def _apply_anneal(alpha: float):
            if bool(train_cfg["optim"]["anneal_lr"]):
                for g in optim.param_groups: g["lr"] = train_cfg["optim"]["lr"] * alpha
            if bool(train_cfg["loss"].get("anneal_clip_epsilon", True)):
                if torch.is_tensor(loss.clip_epsilon):
                    loss.clip_epsilon.copy_(torch.as_tensor(train_cfg["loss"]["clip_epsilon"] * alpha, device=device))
                else: loss.clip_epsilon = train_cfg["loss"]["clip_epsilon"] * alpha

        env.reset()

        if resume_ckpt:
            ckpt = torch.load(resume_ckpt, map_location=device)
            collector_policy.load_state_dict(ckpt["policy"])
            value.load_state_dict(ckpt["value"])
            # Re-init obsnorm: Phase 1 left neighbor dims at zero;
            # Phase 2 multi-edge env will have non-zero values there.
            _norm = env.transform[-1]
            _norm.loc = torch.nn.UninitializedBuffer()
            _norm.scale = torch.nn.UninitializedBuffer()
            env.transform.train()
            env.transform[-1].init_stats(num_iter=100, reduce_dim=(0, 1, 2), cat_dim=0)
            env.transform[-1].to(device)
            env.transform.eval()
            print(f"[resume] Loaded weights from {resume_ckpt}, re-initialized obsnorm")

        for it, batch in enumerate(collector):
            assert_finite(batch, "BATCH")
            assert_finite(batch["next"], "NEXT")

            # batch.batch_size: [num_envs, n_edges, T]
            B_env, E_edges, T_batch = batch.batch_size

            # Rearrange to [B_total, T, ...] for LSTM (batch_first=True)
            data = batch.view(B_env * E_edges, T_batch).contiguous() # [B_total, T]

            # ---- per-decision-step obs logging ----
            _obs_log = data["observation"].float().cpu() # [B, T, obs_dim], normalized
            _norm = env.transform[-1]
            _obs_log = _obs_log * _norm.scale.cpu() + _norm.loc.cpu()  # de-normalize
            _B_total, _T, _D = _obs_log.shape
            for _t in range(_T):
                _step_log = {"decision_step": global_decision_step + _t}
                # Log average across all agents/envs
                _obs_avg = _obs_log[:, _t].mean(dim=0)
                for _j, _name in enumerate(base_env.obs_keys):
                    _step_log[f"obs/{_name}"] = float(_obs_avg[_j].item())
                wandb.log(_step_log)
            global_decision_step += _T


            # ---- build PPO traj ----
            traj = data.clone(False)
            traj.set("reward", traj.get(("next", "reward")))
            traj.set("done", traj.get(("next", "done")).to(torch.bool))
            traj.set("terminated", traj.get(("next", "terminated")).to(torch.bool))
            traj.set("truncated", traj.get(("next", "truncated")).to(torch.bool))

            with torch.no_grad():
                shared_core(traj)
                critic_head(traj)
                shared_core(traj["next"])
                critic_head(traj["next"])
                adv(traj)

            B, T = traj.batch_size
            seq_len_eff = min(seq_len, T)
            if seq_len_eff < 2:
                collector.update_policy_weights_()
                continue

            done_bt = traj.get("done").squeeze(-1).contiguous() # [B, T]
            max_t0 = T - seq_len_eff
            csum = torch.cumsum(done_bt.to(torch.int32), dim=1)
            left = csum[:, : max_t0 + 1]
            right = csum[:, seq_len_eff - 1 : seq_len_eff - 1 + (max_t0 + 1)]
            prev_left = torch.cat([torch.zeros(B, 1, device=device, dtype=csum.dtype), left[:, :-1]], dim=1)
            valid = (right - prev_left) == 0
            valid_idx = valid.nonzero(as_tuple=False)
            if valid_idx.numel() == 0:
                all_b = torch.arange(B, device=device).repeat_interleave(max_t0 + 1)
                all_t0 = torch.arange(max_t0 + 1, device=device).repeat(B)
                valid_idx = torch.stack([all_b, all_t0], dim=1)

            num_sequences = valid_idx.shape[0]
            minibatches_per_epoch = max(1, math.ceil(num_sequences / minibatch_size))
            total_network_updates = max(1, iters_total * ppo_epochs * minibatches_per_epoch)

            for _ in range(ppo_epochs):
                perm = torch.randperm(num_sequences, device=device)
                valid_idx_epoch = valid_idx[perm]
                for mb_i in range(minibatches_per_epoch):
                    start, end = mb_i * minibatch_size, min((mb_i + 1) * minibatch_size, num_sequences)
                    idx = valid_idx_epoch[start:end]
                    b_idx, t0_idx = idx[:, 0], idx[:, 1]
                    
                    mb_td = torch.stack([traj[b_idx, t0_idx + k] for k in range(seq_len_eff)], dim=1).detach()
                    alpha = max(0.0, 1.0 - (num_network_updates / total_network_updates))
                    _apply_anneal(alpha)
                    num_network_updates += 1

                    shared_core(mb_td)
                    out = loss(mb_td)
                    total_loss = out["loss_objective"] + out["loss_critic"] + out.get("loss_entropy", 0.0)
                    optim.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(shared_core.parameters()) + list(actor_head.parameters()) + list(critic_head.parameters()), float(train_cfg["optim"]["max_grad_norm"]))
                    optim.step()

            qoe_score = float(batch["next", "qoe_mean"].mean().item())
            if (it + 1) % ckpt_every == 0:
                save_ckpt(ckpt_dir / f"ckpt_iter_{it+1:06d}.pt", collector_policy, value, optim, env_cfg, train_cfg, it + 1, device, env)
            if qoe_score > best_qoe:
                best_qoe = qoe_score
                save_ckpt(ckpt_dir / "ckpt_best.pt", collector_policy, value, optim, env_cfg, train_cfg, it + 1, device, env)

            _r_lres = float(batch["next", "reward_lambda_res"].mean().item())
            _r_bcd  = float(batch["next", "reward_benign_col_dmg"].mean().item())
            _r_qoe  = float(batch["next", "reward_qoe_penalty"].mean().item())
            _vio    = float(batch["next", "qoe_vio_rate"].mean().item())
            
            print(f"Iter={it:4d} | rew={batch['next','reward'].mean().item():+.4f} | atk_pass={_r_lres:.3f} bcd={_r_bcd:.3f} qoe_sf={_r_qoe:.3f} | qoe_vio={_vio:.1%}")
            wandb.log({
                "iter": it, "qoe/mean": qoe_score, "qoe/vio_rate": _vio, "reward/mean": float(batch["next", "reward"].mean().item()),
                "reward/lambda_res": _r_lres, "reward/benign_col_dmg": _r_bcd, "reward/qoe_penalty": _r_qoe,
                "loss/total": float(total_loss.item()), "loss/policy": float(out["loss_objective"].item()), "loss/critic": float(out["loss_critic"].item()),
            })
            collector.update_policy_weights_()
    finally:
        if os.path.exists(env_cfg_path_tmp): os.remove(env_cfg_path_tmp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/simulation_0.yaml")
    parser.add_argument("--train_cfg", type=str, default="./configs/train.yaml")
    parser.add_argument("--total_frames", type=int, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    args = parser.parse_args()
    train(env_cfg_path=args.cfg, train_cfg_path=args.train_cfg, total_frames_override=args.total_frames, resume_ckpt=args.resume_ckpt, ckpt_dir=args.ckpt_dir)

"""
  Usage:                                                                                                                                                            
  # Phase 1 — single-edge pretraining                                                                                                                               
  python train_indep_lstm.py --cfg configs/simulation_0.yaml --train_cfg configs/train.yaml                                                                         
                                                                                                                                                                    
  # Phase 2 — multi-edge fine-tuning                                                                                                                                
  python train_indep_lstm.py --cfg configs/simulation_ma_0.yaml --train_cfg configs/train.yaml \                                                                    
    --resume_ckpt checkpoints/<phase1_run>/ckpt_best.pt  
"""
