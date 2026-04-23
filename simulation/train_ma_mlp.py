"""
Shared infrastructure for MLP-based MAPPO-ish (CTDE) Multi-Agent RL.

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
from train_ma_lstm import EdgeIDSParallelEnv as _BaseEnv, BuildCentralObs, FlatToAgentsObs, squeeze_last1, save_ckpt
import argparse


# Use the base env from the LSTM version as the logic is identical
EdgeIDSParallelEnv = _BaseEnv
_Base = EdgeIDSParallelEnv

# =========================================================
# Networks
# =========================================================

class AgentMLPCore(nn.Module):
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
        feats = self.net(obs)
        td.set(("agents", "features"), feats)
        return td


class CriticMLPCore(nn.Module):
    def __init__(self, n_edges: int, obs_dim: int, hidden_dim: int, device: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_edges * obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        ).to(device)

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        x = td.get("observation_flat")
        vf_feats = self.net(x)
        td.set("vf_features", vf_feats)
        return td


# =========================================================
# PPO / GAE
# =========================================================

@torch.no_grad()
def compute_gae_inplace(traj: TensorDictBase, gamma: float, lmbda: float, n_edges: int):
    # Standard GAE calculation
    reward = traj.get(("agents", "reward"))
    done = traj.get(("agents", "done")).to(torch.bool)
    terminated = traj.get(("agents", "terminated")).to(torch.bool)
    values = traj.get(("agents", "state_value"))
    next_values = traj.get("next").get(("agents", "state_value"))

    not_end = (~(done | terminated)).to(values.dtype)

    B, T, E = reward.shape
    adv = torch.zeros_like(reward)
    last_gae = torch.zeros((B, E), device=reward.device, dtype=reward.dtype)

    for t in reversed(range(T)):
        delta = reward[:, t] + gamma * next_values[:, t] * not_end[:, t] - values[:, t]
        last_gae = delta + gamma * lmbda * not_end[:, t] * last_gae
        adv[:, t] = last_gae

    traj.set(("agents", "advantage"), adv)
    traj.set(("agents", "value_target"), adv + values)


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

    transforms: List[Transform] = [
        InitTracker(),
        BuildCentralObs(n_edges=n_edges, obs_dim=obs_dim, out_key="observation_flat"),
        ObservationNorm(in_keys=["observation_flat"], standard_normal=True),
        FlatToAgentsObs(n_edges=n_edges, obs_dim=obs_dim),
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
    return env, n_edges, obs_dim


# =========================================================
# Train
# =========================================================

def train(
    env_cfg_path: str = "./configs/simulation_ma_0.yaml",
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

    actor_core = AgentMLPCore(n_edges=n_edges, obs_dim=obs_dim, hidden_dim=hidden_dim, device=device)
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

    critic_core = CriticMLPCore(n_edges=n_edges, obs_dim=obs_dim, hidden_dim=hidden_dim * 2, device=device)
    critic_head = TensorDictModule(
        nn.Linear(hidden_dim * 2, n_edges).to(device),
        in_keys=["vf_features"],
        out_keys=[("agents", "state_value")],
    )
    value_net = TensorDictSequential(critic_core, critic_head)

    optim = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
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
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=int(train_cfg["collector"]["total_frames"]),
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

    ckpt_dir = Path("checkpoints") / run.name
    best_qoe = -1e9
    obs_keys = env_class(cfg_path=env_cfg_path, seed=0).obs_keys

    updates_done = 0
    global_decision_step = 0
    total_updates_est = (int(train_cfg["collector"]["total_frames"]) // frames_per_batch) * ppo_epochs

    for it, batch in enumerate(collector):
        traj = batch.clone(False)

        for k in [("agents", "reward"), ("agents", "done"), ("agents", "terminated")]:
            nk = ("next",) + k
            if k not in traj.keys(True, True) and nk in traj.keys(True, True):
                traj.set(k, traj.get(nk))
            squeeze_last1(traj, k)
            squeeze_last1(traj.get("next"), k)

        with torch.no_grad():
            value_net(traj)
            vals = traj.get(("agents", "state_value"))
            
            # Bootstrapping
            boot_td = traj.get("next")[:, -1].clone()
            value_net(boot_td)
            v_boot = boot_td.get(("agents", "state_value"))
            next_vals = torch.cat([vals[:, 1:], v_boot.unsqueeze(1)], dim=1)
            traj.set(("next", "agents", "state_value"), next_vals)

            compute_gae_inplace(traj, gamma=gamma, lmbda=gae_lambda, n_edges=n_edges)
            traj.set(("agents", "state_value_old"), vals.clone())

        # Flatten for MLP training
        flat_traj = traj.flatten(0, 1)

        for _ in range(ppo_epochs):
            perm = torch.randperm(flat_traj.batch_size[0], device=device)
            for start in range(0, flat_traj.batch_size[0], minibatch_size):
                idx = perm[start : start + minibatch_size]
                sub = flat_traj[idx]

                alpha = max(0.0, 1.0 - (updates_done / total_updates_est))
                lr_now = base_lr * alpha
                for g in optim.param_groups: g["lr"] = lr_now
                clip_eps_now = base_clip_eps * alpha
                updates_done += 1

                policy(sub)
                value_net(sub)

                act = sub.get(("agents", "action")).long()
                old_logp = sub.get(("agents", "sample_log_prob"))
                adv = sub.get(("agents", "advantage"))
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

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

                loss = policy_loss + critic_coeff * critic_loss - entropy_coeff * entropy.mean()

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                optim.step()

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
            loc   = norm_t.loc.view(n_edges, obs_dim)[:, idx].to(obs_log.device)
            scale = norm_t.scale.view(n_edges, obs_dim)[:, idx].to(obs_log.device)
            obs_log[..., idx] = obs[..., idx] * scale + loc

        global_decision_step = wandb_log_obs_steps(
            obs_log, obs_keys, keep_keys={"cpu_to_ids_ratio", "ema_mom"}, global_step_start=global_decision_step
        )

        wandb.log(
            {
                "iter": it,
                "qoe/mean":              qoe_mean,
                "qoe/vio_rate":          _vio,
                "reward/mean":           reward_mean,
                "reward/lambda_res":     _r_lres,
                "reward/benign_col_dmg": _r_bcd,
                "reward/qoe_penalty":    _r_qoe,
            }
        )

        if qoe_mean > best_qoe:
            best_qoe = qoe_mean
            save_ckpt(ckpt_dir / "ckpt_best.pt", policy, value_net, optim, env_cfg, train_cfg, it, device, env)

    collector.shutdown()

"""
Multi-Agent Reward Variants for MLP-based Training.
Includes both Centralised (CMA) and Independent (IMA) configurations.
"""




class EdgeIDSCMAParallelEnv(_Base):
    """Centralised Multi-Agent (CMA) reward variant for MLP."""
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
    """Independent Multi-Agent (IMA) reward variant for MLP."""
    def _compute_step_reward(self, lres, bcd, qsf, vio, attack_in):
        rew = -(self.alpha * qsf + self.beta * lres + self.gamma_r * bcd)
        return rew, lres, bcd, qsf, vio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cma", choices=["cma", "ima"])
    parser.add_argument("--cfg", type=str, default="./configs/simulation_ma_0.yaml")
    args = parser.parse_args()

    env_cls = EdgeIDSCMAParallelEnv if args.mode == "cma" else EdgeIDSIMAParallelEnv
    train(env_class=env_cls, env_cfg_path=args.cfg)
