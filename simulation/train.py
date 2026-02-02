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
from logger import *


from pathlib import Path

def save_ckpt(path: str, actor_net, critic_net, optim, cfg, it: int, device: str, env):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_net": actor_net.state_dict(),
            "critic_net": critic_net.state_dict(),
            "optim": optim.state_dict(),
            "iter": int(it),
            "cfg": cfg,                 # optional but convenient
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

def train(cfg_path="./configs/simulation_0.yaml", device="cpu"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)    
    run = wandb_init(cfg, cfg_path)    
    torch.manual_seed(0)
    np.random.seed(0)
    decision_interval=500
    decisions_per_episode = math.ceil(cfg["run"]["t_max"] / decision_interval)
    ckpt_dir = Path("checkpoints") / run.name  # run.name from wandb_init
    ckpt_every = 50  # iterations
    best_qoe = -1e9    

    base_env = TorchRLEnvWrapper(
        cfg_path=cfg_path,
        seed=0,
        device=device,
        decision_interval=decision_interval
    )

    # env = TransformedEnv(
    #     base_env,
    #     VecNorm(
    #         in_keys=["observation_flat"],
    #         decay=0.99,
    #         eps=1e-6,
    #     ),        
    # )
    
    env = TransformedEnv(
        base_env,
        ObservationNorm(in_keys=["observation_flat"], standard_normal=True),
    )
    # populate mean/std from rollouts
    env.transform.train() 
    env.transform.init_stats(
        num_iter=1000,     # bump if needed
        reduce_dim=0,
        cat_dim=0,
    )

    env.transform.eval()   # freeze normalization constants for training
    obs_size = env.n_edges * env.obs_dim
    action_dim = env.action_dim

    actor_net = ActorNet(obs_size).to(device)
    critic_net = CriticNet(obs_size).to(device)

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation_flat"],
        out_keys=["logits"],
    )


    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
        # default_interaction_type=InteractionType.RANDOM,
    )
    def policy(td: TensorDict) -> TensorDict:
        td_in = TensorDict(
            {"observation_flat": td["observation_flat"]},
            batch_size=td.batch_size,
            device=td.device,
        )

        td_out = actor(td_in)

        td.set("action", td_out["action"])
        td.set("action_log_prob", td_out["action_log_prob"])
        td.set("logits", td_out["logits"])

        return td
    
    value_module = TensorDictModule(
        critic_net,
        in_keys=["observation_flat"],
        out_keys=["state_value"],
    )
    value = ValueOperator(value_module)
    
    adv = GAE(gamma=0.95, lmbda=0.95, value_network=value)
    adv.set_keys(
        value="state_value",
        advantage="advantage",
        value_target="value_target",
        reward="reward",
        done="done",
        terminated="terminated",
    )

    loss = ClipPPOLoss(
        actor_network=actor,
        critic_network=value,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=1e-4,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
    loss.set_keys(
        value="state_value",
        advantage="advantage",
        value_target="value_target",
    )    
    
    
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=decisions_per_episode,
        total_frames=decisions_per_episode * 1000,
        device=device,
        trust_policy=True,
    )

    optim = torch.optim.Adam(
        list(actor_net.parameters()) + list(critic_net.parameters()),
        lr=1e-4,
    )

    ppo_epochs = 4
    minibatch_size = 1024

    def assert_finite(td, prefix=""):
        for k in td.keys(True, True):
            v = td.get(k)
            if torch.is_tensor(v) and not torch.isfinite(v).all():
                bad = v[~torch.isfinite(v)]
                print(prefix, "NON-FINITE at key:", k, "example:", bad.flatten()[:5])
                raise RuntimeError(f"NaN/Inf in {k}")

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

        adv_t = traj["advantage"]
        traj.set(
            "advantage",
            (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        )

        flat = traj.flatten(0, -1)
        B = flat.numel()
        for _ in range(ppo_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, minibatch_size):
                mb = flat[perm[start:start + minibatch_size]]

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
                    1.0,
                )
                optim.step()
                # after training update
            qoe_score = float(batch["next", "reward"].mean().item())  # or your qoe_mean if you prefer

        # periodic
        if (it + 1) % ckpt_every == 0:
            save_ckpt(
                ckpt_dir / f"ckpt_iter_{it+1:06d}.pt",
                actor_net, critic_net, optim, cfg, it + 1, device, env,
            )

        # best
        if qoe_score > best_qoe:
            best_qoe = qoe_score
            save_ckpt(
                ckpt_dir / "ckpt_best.pt",
                actor_net, critic_net, optim, cfg, it + 1, device, env,
            )                
        print(
            f"Iteration={it} "
            f"reward_mean={batch['next', 'reward'].mean().item():.4f}"
        )                

        obs = batch["observation"].detach()          # [T, n_edges, obs_dim] or [T, B, n_edges, obs_dim]
        act = batch["action"].detach()              # [T] or [T, B]
        rew = batch["next", "reward"].detach()      # [T, 1] or [T, B, 1]
        done = batch["next", "done"].detach()       # [T, 1] or [T, B, 1]

        # squeeze B if B=1
        if obs.ndim == 4: obs = obs[:, 0]
        if act.ndim == 2: act = act[:, 0]
        if rew.ndim == 3: rew = rew[:, 0]
        if done.ndim == 3: done = done[:, 0]

        T, E, D = obs.shape
        DI = base_env.decision_interval
        qoe_hist = base_env.env.last_history

        if not qoe_hist:
            return  # or continue outer loop

        base = it * T

        for t in range(T):
            if bool(done[t].squeeze(-1).item()):
                continue

            ts_step = base + t
            a = int(act[t].item())

            start = t * DI * E
            end = (t + 1) * DI * E

            if end > len(qoe_hist):
                break

            qoe_block = qoe_hist[start:end]

            qoe_mean = float(np.mean([m.qoe_mean for m in qoe_block]))

            log_dict = {
                "ts_step": ts_step,
                "ts/action": a,
                "ts/qoe": qoe_mean,
            }

            for e in range(E):
                for j, k in enumerate(base_env.obs_keys):
                    log_dict[f"ts/edge_{e}/{k}"] = float(obs[t, e, j].item())

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
        # env.transform.train() 
        # stop once episode is finished
        if batch["done"].any():
            continue
    wandb_save_plots_from_history(base_env.env)
    wandb.finish()        

if __name__ == "__main__":
    train("./configs/simulation_0.yaml", device="cuda")