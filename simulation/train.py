import yaml
import math
import torch
import torch.nn as nn
import numpy as np
from tensordict import TensorDict

from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.distributions import Categorical, Independent
from environment import TorchRLEnvWrapper
from torchrl.envs.transforms import ObservationNorm, VecNorm, TransformedEnv
from torchrl.envs import ParallelEnv  # or ParallelEnv if you want multiprocessing

from logger import *


from pathlib import Path

def save_ckpt(path: str, actor_net, critic_net, optim, env_cfg, train_cfg, it: int, device: str, env):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_net": actor_net.state_dict(),
            "critic_net": critic_net.state_dict(),
            "optim": optim.state_dict(),
            "iter": int(it),
            "env_cfg": env_cfg,                 # optional but convenient
            "train_cfg": train_cfg,                 # optional but convenient
            "device": str(device),
            "obsnorm": env.state_dict(),
        },
        str(path),
    )

def orthogonal_init(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)

class ActorNet(nn.Module):
    def __init__(self, obs_dim, n_actions=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.apply(lambda m: orthogonal_init(m, gain=nn.init.calculate_gain("tanh")))
        orthogonal_init(self.net[-1], gain=0.01)

    def forward(self, obs):
        return self.net(obs)

class CriticNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.apply(lambda m: orthogonal_init(m, gain=nn.init.calculate_gain("tanh")))
        orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

def train(env_cfg_path="./configs/simulation_0.yaml", train_cfg_path="./configs/train.yaml", device="cuda"):
    with open(env_cfg_path, "r") as f:
        env_cfg = yaml.safe_load(f)
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)
        
    t_max = env_cfg["run"]["t_max"]
    run = wandb_init(env_cfg, train_cfg)

    torch.manual_seed(env_cfg["run"]["seed"])
    np.random.seed(env_cfg["run"]["seed"])

    decision_interval = env_cfg["globals"]["decision_interval"]
    num_envs = train_cfg["collector"]["num_envs"]
    
    decisions_per_episode = math.ceil(t_max / decision_interval)

    ckpt_dir = Path("checkpoints") / run.name
    ckpt_every = 50
    best_qoe = -1e9

    base_env = TorchRLEnvWrapper(
        cfg_path=env_cfg_path,
        seed=env_cfg["run"]["seed"],
        device=device,
        decision_interval=decision_interval
    )

    def make_env(seed_offset):
        def _make():
            return TorchRLEnvWrapper(
                cfg_path=env_cfg_path,
                seed=seed_offset,
                device=device,
                decision_interval=decision_interval,
            )
        return _make

    penv = ParallelEnv(
        num_envs,
        [make_env(1000 + i) for i in range(num_envs)],
    )

    env = TransformedEnv(
        penv,
        ObservationNorm(
            in_keys=["observation_flat"],
            standard_normal=train_cfg["observation_norm"]["standard_normal"],
        ),
    )

    # populate mean/std from rollouts
    env.transform.train()
    env.transform.init_stats(
        num_iter=5,
        reduce_dim=(0, 1),
        cat_dim=0,
    )
    env.transform.eval()

    obs_size = int(env.observation_spec["observation_flat"].shape[-1])

    actor_net = ActorNet(
        obs_size,
        n_actions=train_cfg["model"]["n_actions"],
        hidden=train_cfg["model"]["hidden_dim"],
    ).to(device)

    critic_net = CriticNet(
        obs_size,
        hidden=train_cfg["model"]["hidden_dim"],
    ).to(device)

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation_flat"],
        out_keys=["logits"],
    )

    policy = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    value_module = TensorDictModule(
        critic_net,
        in_keys=["observation_flat"],
        out_keys=["state_value"],
    )
    value = ValueOperator(value_module)

    adv = GAE(
        gamma=train_cfg["loss"]["gamma"],
        lmbda=train_cfg["loss"]["gae_lambda"],
        value_network=value,
    )

    adv.set_keys(
        value="state_value",
        advantage="advantage",
        value_target="value_target",
        reward="reward",
        done="done",
        terminated="terminated",
    )

    loss = ClipPPOLoss(
        actor_network=policy,
        critic_network=value,
        clip_epsilon=train_cfg["loss"]["clip_epsilon"],
        entropy_bonus=True,
        entropy_coef=train_cfg["loss"]["entropy_coeff"],
        critic_coef=train_cfg["loss"]["critic_coeff"],
        loss_critic_type=train_cfg["loss"]["loss_critic_type"],
        # normalize_advantage=True, 
    )

    loss.set_keys(
        value="state_value",
        advantage="advantage",
        value_target="value_target",
    )

    frames_per_batch = train_cfg.get("collector",{}).get("frames_per_batch", None) if not None else decisions_per_episode * num_envs * num_envs
    total_frames = train_cfg["collector"]["total_frames"]

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        trust_policy=train_cfg["collector"]["trust_policy"],
    )

    optim = torch.optim.Adam(
        list(actor_net.parameters()) + list(critic_net.parameters()),
        lr=train_cfg["optim"]["lr"],
        weight_decay=train_cfg["optim"]["weight_decay"],
        eps=train_cfg["optim"]["eps"],
    )

    ppo_epochs = train_cfg["loss"]["ppo_epochs"]
    minibatch_size = train_cfg["loss"]["mini_batch_size"]

    def assert_finite(td, prefix=""):
        for k in td.keys(True, True):
            v = td.get(k)
            if torch.is_tensor(v) and not torch.isfinite(v).all():
                bad = v[~torch.isfinite(v)]
                print(prefix, "NON-FINITE at key:", k, "example:", bad.flatten()[:5])
                raise RuntimeError(f"NaN/Inf in {k}")

    # -----------------------
    # Annealing bookkeeping
    # -----------------------
    iters_total = max(1, total_frames // frames_per_batch)
    num_network_updates = 0

    def _num_minibatches(B: int) -> int:
        return (B + minibatch_size - 1) // minibatch_size

    def _apply_anneal(alpha: float):
        if bool(train_cfg["optim"]["anneal_lr"]):
            lr_now = train_cfg["optim"]["lr"] * alpha
            for g in optim.param_groups:
                g["lr"] = lr_now
        if bool(train_cfg["loss"].get("anneal_clip_epsilon", True)):
            # ClipPPOLoss stores clip_epsilon as a tensor internally
            if torch.is_tensor(loss.clip_epsilon):
                loss.clip_epsilon.copy_(torch.as_tensor(train_cfg["loss"]["clip_epsilon"] * alpha, device=device))
            else:
                loss.clip_epsilon = train_cfg["loss"]["clip_epsilon"] * alpha  # fallback

    env.reset()

    for it, batch in enumerate(collector):
        # env.transform.eval() 
        x = batch["observation_flat"]
        print("obs_flat mean/std/min/max:", float(x.mean()), float(x.std()), float(x.min()), float(x.max()))
        assert_finite(batch, "BATCH")
        assert_finite(batch["next"], "NEXT")

        # remove reset + terminal
        traj = batch[:]

        traj.set("reward", traj.get(("next", "reward")))
        traj.set("done", traj.get(("next", "done")))
        traj.set("terminated", traj.get(("next", "terminated")))
        traj.set("truncated", traj.get(("next", "truncated")))
        
        with torch.no_grad():
            value(traj)          # writes "state_value" into traj
            value(traj["next"])  # writes "state_value" into traj["next"]
            adv(traj)
        flat = traj.flatten(0, -1)
        B = int(flat.batch_size[0])
        n_mb = _num_minibatches(B)
        total_network_updates = max(1, iters_total * ppo_epochs * n_mb)

        adv_t = traj["advantage"]
        traj.set(
            "advantage",
            (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        )


        # train
        for _ in range(ppo_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, minibatch_size):
                mb = flat[perm[start:start + minibatch_size]]

                # linear schedule over total network updates (Atari)
                alpha = 1.0 - (num_network_updates / total_network_updates)
                if alpha < 0.0:
                    alpha = 0.0
                _apply_anneal(alpha)
                num_network_updates += 1

                out = loss(mb)
                total_loss = (
                    out["loss_objective"]
                    + out["loss_critic"]
                    + out.get("loss_entropy", torch.tensor(0.0, device=device))
                )

                optim.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor_net.parameters()) + list(critic_net.parameters()),
                    float(train_cfg["optim"]["max_grad_norm"]),
                )
                optim.step()

        qoe_score = float(batch["next", "reward"].mean().item())

        # periodic
        if (it + 1) % ckpt_every == 0:
            save_ckpt(
                ckpt_dir / f"ckpt_iter_{it+1:06d}.pt",
                actor_net, critic_net, optim, env_cfg, train_cfg, it + 1, device, env,
            )

        # best
        if qoe_score > best_qoe:
            best_qoe = qoe_score
            save_ckpt(
                ckpt_dir / "ckpt_best.pt",
                actor_net, critic_net, optim, env_cfg, train_cfg, it + 1, device, env,
            )                
        print(
            f"Iteration={it} "
            f"reward_mean={batch['next', 'reward'].mean().item():.4f}"
        )                
        obs  = batch["next", "observation"].detach()       # [T,B,E,D] or [T,E,D]
        act  = batch["action"].detach()                    # action taken at time t
        qoe  = batch["next", "qoe_mean"].detach()
        tint = batch["next", "t_internal"].detach()
        done = batch["next", "done"].detach()

        # normalize shapes
        if obs.ndim == 3:       # [T, E, D] -> add B=1
            obs = obs[:, None, :, :]
        if act.ndim == 1:
            act = act[:, None]
        if qoe.ndim == 2:
            qoe = qoe[:, None, :]
        if tint.ndim == 2:
            tint = tint[:, None, :]
        if done.ndim == 2:
            done = done[:, None, :]

        T, B, E, D = obs.shape
        base = it*T
        for t in range(T):
            ts_step = base+t
            # for b in range(B):
            b=0
            step_id = int(tint[t, b, 0].item())   # your internal timestep marker
            a = int(act[t, b].item())
            q = float(qoe[t, b, 0].item())

            log_dict = {
                f"ts_step": ts_step,
                f"ts/action": a,
                f"ts/qoe": q,
            }

            for e in range(E):
                for j, k in enumerate(base_env.obs_keys):
                    log_dict[f"ts/edge_{e}/{k}"] = float(obs[t, b, e, j].item())

            wandb.log(log_dict, commit=True)
        wandb.log(
            {
                "iter": it,
                "reward/mean": float(batch["next", "reward"].mean().item()),
                "loss/total": float(total_loss.detach().item()),
                "loss/policy": float(out["loss_objective"].detach().item()),
                "loss/critic": float(out["loss_critic"].detach().item()),
                "loss/entropy": float(out.get("loss_entropy", torch.tensor(0.0, device=device)).detach().item()),
            },
        )
        # collector.update_policy_weights_()
        # env.transform.train() 
        # stop once episode is finished
        if batch["done"].any():
            continue
    wandb_save_plots_from_history(base_env.env)
    wandb.finish()        

if __name__ == "__main__":
    train()