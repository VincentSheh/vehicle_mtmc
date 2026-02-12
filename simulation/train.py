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
from torchrl.modules import LSTMModule
from torchrl.envs.transforms import Compose
from torchrl.envs.transforms import InitTracker
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from logger import *


from pathlib import Path

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

def orthogonal_init(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)

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

def train(env_cfg_path="./configs/simulation_0.yaml", train_cfg_path="./configs/train.yaml", resume_ckpt=None, device="cuda"):
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


    feature_dim = train_cfg["model"]["hidden_dim"]
    seq_len = int(train_cfg["loss"].get("seq_len", 32))  # 16/32/64 common

    lstm = LSTMModule(
        input_size=feature_dim,
        hidden_size=feature_dim,
        in_key="features",
        out_key="features",
        device=device,
    )

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
        Compose(
            InitTracker(),  # provides "is_init" so LSTM can reset hidden state
            lstm.make_tensordict_primer(),
            ObservationNorm(
                in_keys=["observation_flat"],
                standard_normal=train_cfg["observation_norm"]["standard_normal"],
            ),
        ),
    )
    
    print("env.batch_size:", env.batch_size)

    # populate mean/std from rollouts
    env.transform.train()
    env.transform[-1].init_stats(  # ObservationNorm is last in your Compose
        num_iter=100,
        reduce_dim=(0, 1),
        cat_dim=0,
    )
    env.transform.eval()

    obs_size = int(env.observation_spec["observation_flat"].shape[-1])

    
    # ---- actor ----
    feature_module = TensorDictModule(
        FeatureNet(obs_size, feature_dim).to(device),
        in_keys=["observation_flat"],
        out_keys=["features"],
    )

    shared_core = TensorDictSequential(feature_module, lstm)

    actor_head = TensorDictModule(
        nn.Linear(feature_dim, train_cfg["model"]["n_actions"]).to(device),
        in_keys=["features"],
        out_keys=["logits"],
    )

    critic_head = TensorDictModule(
        nn.Linear(feature_dim, 1).to(device),
        in_keys=["features"],
        out_keys=["state_value"],
    )



    # actor for loss, head-only
    loss_actor = ProbabilisticActor(
        module=actor_head,          # reads "features" -> writes "logits"
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )


    # policy used by collector: DOES include shared_core
    collector_policy = ProbabilisticActor(
        module=TensorDictSequential(shared_core, actor_head),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )
    # critic for loss, head-only
    value = critic_head   # reads "features" -> writes "state_value"
    optim = torch.optim.Adam(
        list(shared_core.parameters()) +
        list(actor_head.parameters()) +
        list(critic_head.parameters()),
        lr=train_cfg["optim"]["lr"],
        weight_decay=train_cfg["optim"]["weight_decay"],
        eps=train_cfg["optim"]["eps"],
    )
    if resume_ckpt is not None:
        print(f"Loading checkpoint: {resume_ckpt}")
        state = torch.load(resume_ckpt, map_location=device)
        collector_policy.load_state_dict(state["policy"])
        value.load_state_dict(state["value"])
        optim.load_state_dict(state["optim"])    
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
        actor_network=loss_actor,
        critic_network=value,
        clip_epsilon=train_cfg["loss"]["clip_epsilon"],
        entropy_bonus=True,
        entropy_coef=train_cfg["loss"]["entropy_coeff"],
        critic_coef=train_cfg["loss"]["critic_coeff"],
        loss_critic_type=train_cfg["loss"]["loss_critic_type"],
        normalize_advantage=True,
    )

    loss.set_keys(value="state_value", advantage="advantage", value_target="value_target")

    frames_per_batch = train_cfg["collector"].get(
       "frames_per_batch",
        decisions_per_episode * num_envs * num_envs #Double Intentional
    )
    total_frames = train_cfg["collector"]["total_frames"]

    collector = SyncDataCollector(
        env,
        policy=collector_policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        trust_policy=train_cfg["collector"]["trust_policy"],
        split_trajs=False,
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
    
    sampler = SliceSampler(
        slice_len=seq_len,
        end_key="done",
        cache_values=True,
        strict_length=False,
    )

    # PPO is on-policy, so we reuse a rollout buffer and overwrite it every iteration
    # Store on GPU to avoid cpu<->gpu ping-pong
    rollout_storage = LazyTensorStorage(
        max_size=num_envs,   # store B trajectories, each item is [T,...]
        device=device,
    )

    rb = TensorDictReplayBuffer(
        storage=rollout_storage,
        sampler=sampler,
        batch_size=minibatch_size,  # sequences per minibatch
    )    

    for it, batch in enumerate(collector):
        assert_finite(batch, "BATCH")
        assert_finite(batch["next"], "NEXT")
        if it == 0:
            print("batch.batch_size:", batch.batch_size)
            print("batch['action'].shape:", batch['action'].shape)
            print("batch['observation_flat'].shape:", batch['observation_flat'].shape)
            print("batch['next','reward'].shape:", batch['next','reward'].shape)
            print("batch keys:", batch.keys(True, True))        

        # ---- build PPO traj ----
        traj = batch.clone(False)
        traj.set("reward", traj.get(("next", "reward")))
        traj.set("qoe_mean", traj.get(("next", "qoe_mean")))
        traj.set("done", traj.get(("next", "done")).to(torch.bool))
        traj.set("terminated", traj.get(("next", "terminated")).to(torch.bool))
        traj.set("truncated", traj.get(("next", "truncated")).to(torch.bool))

        # Optional: if your env returns only done and not terminated/truncated, keep terminated/truncated but ensure keys exist.
        # If your GAE uses done/terminated keys, this is fine as-is.

        # ---- compute features + values + advantages on full rollout ----
        with torch.no_grad():
            # main sequence
            shared_core(traj)          # writes "features" and recurrent states along time
            critic_head(traj)          # writes "state_value"
            # next sequence for bootstrap
            shared_core(traj["next"])
            critic_head(traj["next"])
            adv(traj)                  # writes "advantage" and "value_target"

        data = traj  # currently [B, T, ...] in your run

        # ---- normalize to time-major [T, B, ...] ----
        # env.batch_size is [1], and your collector gives [1, 1024] so it's [B, T]
        if data.batch_size[0] == env.batch_size[0]:
            data = data.transpose(0, 1).contiguous()  # -> [T, B, ...]

        T, B = data.batch_size[:2]
        seq_len_eff = min(seq_len, T)
        if seq_len_eff < 2:
            collector.update_policy_weights_()
            continue

        # ---- get done as [B, T] bool ----
        done = data.get("done")  # could be [T,B] or [T,B,1] etc
        while done.ndim > 2:
            done = done.squeeze(-1)
        done = done.to(torch.bool)              # [T, B]
        done_bt = done.transpose(0, 1).contiguous()  # [B, T]

        max_t0 = T - seq_len_eff
        if max_t0 < 0:
            raise RuntimeError(f"seq_len={seq_len} > rollout length T={T}. Reduce seq_len or increase frames_per_batch.")

        # valid start if there is NO done in [t0, t0+seq_len-1]
        # use cumulative sum trick
        done_int = done_bt.to(torch.int32)          # [B, T]
        csum = torch.cumsum(done_int, dim=1)        # [B, T]

        left = csum[:, : max_t0 + 1]               # [B, max_t0+1]
        right = csum[:, seq_len_eff - 1 : seq_len_eff - 1 + (max_t0 + 1)]  # [B, max_t0+1]

        prev_left = torch.cat(
            [torch.zeros(B, 1, device=device, dtype=csum.dtype), left[:, :-1]],
            dim=1,
        )                                          # [B, max_t0+1]

        window_sum = right - prev_left             # [B, max_t0+1]
        valid = window_sum == 0

        valid_idx = valid.nonzero(as_tuple=False)  # [N,2] = (b, t0)
        if valid_idx.numel() == 0:
            all_b = torch.arange(B, device=device).repeat_interleave(max_t0 + 1)
            all_t0 = torch.arange(max_t0 + 1, device=device).repeat(B)
            valid_idx = torch.stack([all_b, all_t0], dim=1)
        if valid_idx.numel() == 0:
            # fallback: allow all starts
            all_b = torch.arange(B, device=device).repeat_interleave(max_t0 + 1)
            all_t0 = torch.arange(max_t0 + 1, device=device).repeat(B)
            valid_idx = torch.stack([all_b, all_t0], dim=1)

        num_sequences = valid_idx.shape[0]
        # how many minibatches per epoch
        minibatches_per_epoch = max(1, math.ceil(num_sequences / minibatch_size))
        total_network_updates = max(1, iters_total * ppo_epochs * minibatches_per_epoch)

        last_out = None
        last_total_loss = None

        # ---- PPO epochs ----
        for _ in range(ppo_epochs):
            # shuffle valid starts each epoch
            perm = torch.randperm(num_sequences, device=device)
            valid_idx_epoch = valid_idx[perm]

            for mb_i in range(minibatches_per_epoch):
                start = mb_i * minibatch_size
                end = min((mb_i + 1) * minibatch_size, num_sequences)
                idx = valid_idx_epoch[start:end]   # [mb,2]
                b_idx = idx[:, 0]                  # [mb]
                t0_idx = idx[:, 1]                 # [mb]

                # build sequence tensor by gathering per-sample slices
                # data is [T,B,...] so we slice time then select envs
                # shape becomes [seq_len, mb, ...]
                seq_list = []
                for k in range(seq_len_eff):
                    seq_list.append(data[t0_idx + k, b_idx])
                mb_td = torch.stack(seq_list, dim=0).detach()
                mb_td = torch.stack(seq_list, dim=0)  # TensorDict stacked on time dim

                mb_td = mb_td.detach()

                alpha = 1.0 - (num_network_updates / total_network_updates)
                if alpha < 0.0:
                    alpha = 0.0
                _apply_anneal(alpha)
                num_network_updates += 1

                # Recompute features for current shared_core weights on this sequence
                shared_core(mb_td)

                out = loss(mb_td)
                total_loss = (
                    out["loss_objective"]
                    + out["loss_critic"]
                    + out.get("loss_entropy", torch.tensor(0.0, device=device))
                )

                optim.zero_grad(set_to_none=True)
                total_loss.backward()

                params = (
                    list(shared_core.parameters())
                    + list(actor_head.parameters())
                    + list(critic_head.parameters())
                )
                torch.nn.utils.clip_grad_norm_(params, float(train_cfg["optim"]["max_grad_norm"]))
                optim.step()

                last_out = out
                last_total_loss = total_loss

        # ---- metrics / ckpt ----
        qoe_score = float(batch["next", "reward"].mean().item())

        if (it + 1) % ckpt_every == 0:
            save_ckpt(
                ckpt_dir / f"ckpt_iter_{it+1:06d}.pt",
                collector_policy, value, optim, env_cfg, train_cfg, it + 1, device, env,
            )

        if qoe_score > best_qoe:
            best_qoe = qoe_score
            save_ckpt(
                ckpt_dir / "ckpt_best.pt",
                collector_policy, value, optim, env_cfg, train_cfg, it + 1, device, env,
            )

        print(f"Iteration={it} reward_mean={batch['next','reward'].mean().item():.4f} qoe_mean={batch['next','qoe_mean'].mean().item():.4f}")
        print("after norm T,B:", T, B, "done shape:", done.shape, "done_bt shape:", done_bt.shape)

        if last_out is not None and last_total_loss is not None:
            wandb.log(
                {
                    "iter": it,
                    "qoe/mean": float(batch["next", "qoe_mean"].mean().item()),
                    "reward/mean": float(batch["next", "reward"].mean().item()),
                    "loss/total": float(last_total_loss.detach().item()),
                    "loss/policy": float(last_out["loss_objective"].detach().item()),
                    "loss/critic": float(last_out["loss_critic"].detach().item()),
                    "loss/entropy": float(last_out.get("loss_entropy", torch.tensor(0.0, device=device)).detach().item()),
                },
            )

        collector.update_policy_weights_()
if __name__ == "__main__":
    # train(resume_ckpt="checkpoints/lstm_epoch_20_linear/ckpt_iter_000950.pt")
    train()