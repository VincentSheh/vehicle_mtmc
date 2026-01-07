import math
import torch
import torch.nn as nn
import numpy as np
from tensordict import TensorDict

from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.distributions import Categorical, Independent
from environment import TorchRLEnvWrapper


class ActorNet(nn.Module):
    def __init__(self, n_edges: int, obs_dim: int):
        super().__init__()
        self.n_edges = n_edges
        obs_size = n_edges * obs_dim

        self.backbone = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.proj = nn.Linear(256, n_edges * 4)

    def forward(self, obs):
        x = obs.reshape(*obs.shape[:-2], -1)
        h = self.backbone(x)

        logits = self.proj(h)
        logits = logits.view(*h.shape[:-1], self.n_edges, 4)
        return logits


class CriticNet(nn.Module):
    def __init__(self, n_edges: int, obs_dim: int):
        super().__init__()
        obs_size = n_edges * obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs):
        # obs: (..., n_edges, obs_dim)
        x = obs.reshape(*obs.shape[:-2], -1)  # -> (..., n_edges*obs_dim)
        return self.net(x)  # -> (..., 1)


def train(cfg_path="./configs/simulation_0.yaml", device="cpu"):
    torch.manual_seed(0)
    np.random.seed(0)
    decision_interval=100
    decisions_per_episode = math.ceil(20000 / decision_interval)

    env = TorchRLEnvWrapper(
        cfg_path=cfg_path,
        seed=0,
        device=device,
        decision_interval=decision_interval
    )

    obs_size = env.n_edges * env.obs_dim
    action_dim = env.action_dim

    # Flatten obs inside tensordict: (n_edges, obs_dim) -> (obs_size,)
    def add_obs_flat_td(td):
        obs = td["observation"]
        td.set("observation_flat", obs.flatten(-2, -1))

        # if next exists, also create next/observation_flat
        if "next" in td.keys(True, True):
            nxt_obs = td["next", "observation"]
            td.set(("next", "observation_flat"), nxt_obs.flatten(-2, -1))
        return td

    actor_net = ActorNet(env.n_edges, env.obs_dim).to(device)
    critic_net = CriticNet(env.n_edges, env.obs_dim).to(device)

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["logits"],
    )


    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=lambda logits: Independent(
            Categorical(logits=logits),
            reinterpreted_batch_ndims=1,
        ),
        return_log_prob=True,
    )

    value_module = TensorDictModule(
        critic_net,
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    value = ValueOperator(value_module)
    
    adv = GAE(gamma=0.995, lmbda=0.95, value_network=value)
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
        entropy_coef=1e-3,
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
        policy=actor,
        frames_per_batch=decisions_per_episode,
        total_frames=decisions_per_episode * 100,
        device=device,
        trust_policy=True,
    )

    optim = torch.optim.Adam(
        list(actor_net.parameters()) + list(critic_net.parameters()),
        lr=3e-4,
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
        batch = add_obs_flat_td(batch)

        # safety checks
        assert_finite(batch, "BATCH")
        assert_finite(batch["next"], "NEXT")

        # remove reset + terminal
        traj = batch[1:-1]

        traj.set("reward", traj.get(("next", "reward")))
        traj.set("done", traj.get(("next", "done")))
        traj.set("terminated", traj.get(("next", "terminated")))
        traj.set("truncated", traj.get(("next", "truncated")))

        with torch.no_grad():
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

        print(
            f"Iteration={it} "
            f"reward_mean={batch['next', 'reward'].mean().item():.4f}"
        )

        # stop once episode is finished
        if batch["done"].any():
            break

if __name__ == "__main__":
    train("./configs/simulation_0.yaml", device="cuda")