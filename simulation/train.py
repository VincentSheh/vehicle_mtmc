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
from torch.distributions import Categorical, Independent, Distribution
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

class MultiEdgeCategoricalSampler(nn.Module):
    # logits: [..., E, A] -> action: [..., E], sample_log_prob: [..., 1]
    def forward(self, logits: torch.Tensor):
        dist = Categorical(logits=logits)                 # batch [..., E]
        action = dist.sample()                            # [..., E]
        logp = dist.log_prob(action).sum(-1, keepdim=True)  # [..., 1]
        return action, logp


class MultiEdgeCategoricalLogProb(nn.Module):
    # logits: [..., E, A], action: [..., E] -> sample_log_prob: [..., 1]
    def forward(self, logits: torch.Tensor, action: torch.Tensor):
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action).sum(-1, keepdim=True)  # [..., 1]
        return logp

class MeanPoolEdges(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_edges, feature_dim]
        return x.mean(dim=-2)


class EdgewiseLSTM(nn.Module):
    """
    Applies an LSTM independently to each edge by flattening (B, E) -> (B*E).
    Input td["features"]: [B,E,F] or [T,B,E,F]
    Output td["features"]: same shape but with hidden_size
    Stores recurrent state in td under ("rnn","h") and ("rnn","c") and next under ("next","rnn",...).
    Resets hidden where td["is_init"] is True (shape [B] or [T,B]).
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, device="cpu"):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device(device)
    def forward(self, td: TensorDict) -> TensorDict:
        x = td["features"]  # [B,E,F] or [T,B,E,F] or [B,T,E,F]
        L, H = self.num_layers, self.hidden_size

        if x.ndim == 3:
            # step mode
            B, E, F = x.shape
            x_tm = x.reshape(1, B * E, F)          # [1, B*E, F]
            seq_mode = False
            batch_major = None
            T = 1

        elif x.ndim == 4:
            # sequence mode, infer layout from x itself
            if x.shape[0] == td.batch_size[0] and x.shape[1] == td.batch_size[1]:
                # x matches td layout, but could be [B,T,...] or [T,B,...]
                # decide by checking which axis is "time" using a heuristic:
                # if td has keys like "done" with same first 2 dims, follow td layout
                # simplest robust way: treat x as [B,T,...] if td.batch_size == [B,T]
                # and as [T,B,...] if td.batch_size == [T,B]
                B0, T0 = int(td.batch_size[0]), int(td.batch_size[1])
                # assume td.batch_size is [B,T] in your pipeline, so x is [B,T,...]
                batch_major = True
                B, T = B0, T0
                _, _, E, F = x.shape
                x_tm = x.transpose(0, 1).contiguous().reshape(T, B * E, F)  # [T, B*E, F]
            else:
                # fallback, treat as time-major [T,B,...]
                T, B, E, F = x.shape
                batch_major = False
                x_tm = x.contiguous().reshape(T, B * E, F)
            seq_mode = True
        else:
            raise ValueError(f"EdgewiseLSTM expected 3D/4D features, got {x.ndim}D")

        # init h,c
        if not seq_mode:
            h = torch.zeros(L, B * E, H, device=x.device, dtype=x.dtype)
            c = torch.zeros(L, B * E, H, device=x.device, dtype=x.dtype)
            # optional reset using td["is_init"] as you had before
        else:
            h = torch.zeros(L, B * E, H, device=x.device, dtype=x.dtype)
            c = torch.zeros(L, B * E, H, device=x.device, dtype=x.dtype)

        y, (h2, c2) = self.lstm(x_tm, (h, c))  # y: [T, B*E, H]

        if not seq_mode:
            # IMPORTANT: step mode must be [B,E,H], no time dim
            y_out = y[0].view(B, E, H)  # or y.squeeze(0).view(B,E,H)
            td.set("features", y_out)
            # store rnn state as before
            h2_be = h2.view(L, B, E, H).permute(1, 2, 0, 3).contiguous()
            c2_be = c2.view(L, B, E, H).permute(1, 2, 0, 3).contiguous()
            td.set(("rnn", "h"), h2_be)
            td.set(("rnn", "c"), c2_be)
            td.set(("next", "rnn", "h"), h2_be)
            td.set(("next", "rnn", "c"), c2_be)
        else:
            y_tb = y.view(T, B, E, H)  # [T,B,E,H]
            if batch_major:
                y_out = y_tb.transpose(0, 1).contiguous()  # [B,T,E,H]
            else:
                y_out = y_tb                                # [T,B,E,H]
            td.set("features", y_out)

        return td

    
class UnflattenObs(nn.Module):
    def __init__(self, n_edges: int, obs_dim: int):
        super().__init__()
        self.n_edges = n_edges
        self.obs_dim = obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_edges*obs_dim]
        return x.view(*x.shape[:-1], self.n_edges, self.obs_dim)
    
    
class MultiEdgeCategorical(Distribution):
    has_rsample = False

    def __init__(self, logits: torch.Tensor):
        # logits: [..., E, A]
        self.logits = logits
        self._cat = Categorical(logits=logits)  # batch_shape [..., E]
        super().__init__(batch_shape=self._cat.batch_shape, event_shape=torch.Size([logits.shape[-2]]))

    def sample(self, sample_shape=torch.Size()):
        a = self._cat.sample(sample_shape)      # sample_shape + [..., E]
        return a

    def log_prob(self, value):
        # value: sample_shape + [..., E]
        lp = self._cat.log_prob(value)          # sample_shape + [..., E]
        return lp.sum(-1)                       # sample_shape + [...]

    def entropy(self):
        ent = self._cat.entropy()               # [..., E]
        return ent.sum(-1)     
    
class IndependentCategoricalEdges:
    # TorchRL will call this as IndependentCategoricalEdges(logits=...)
    def __init__(self, logits: torch.Tensor):
        # logits: [..., E, A]
        self._dist = Independent(Categorical(logits=logits), 1)  # event_dim = E

    def sample(self, sample_shape=torch.Size()):
        return self._dist.sample(sample_shape)  # [..., E]

    def log_prob(self, value):
        # return [..., 1] so PPO broadcasts nicely
        return self._dist.log_prob(value).unsqueeze(-1)

    def entropy(self):
        return self._dist.entropy().unsqueeze(-1)
    
class MultiEdgeIndependentCategorical(Distribution):
    arg_constraints = {}  # keep simple

    def __init__(self, logits: torch.Tensor, validate_args=None):
        # logits: [..., E, A]
        base = Categorical(logits=logits)
        self._dist = Independent(base, 1)  # event_dim = E, sums log_prob/entropy across edges
        super().__init__(
            batch_shape=self._dist.batch_shape,
            event_shape=self._dist.event_shape,
            validate_args=validate_args,
        )

    def sample(self, sample_shape=torch.Size()):
        return self._dist.sample(sample_shape)     # [..., E]

    def log_prob(self, value):
        return self._dist.log_prob(value)          # [...]

    def entropy(self):
        return self._dist.entropy()                # [...]    
            
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

def train(env_cfg_path="./configs/simulation_ma_0.yaml", train_cfg_path="./configs/train.yaml", resume_ckpt=None, device="cuda"):
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

    def make_env(seed_offset):
        def _make():
            return TorchRLEnvWrapper(
                cfg_path=env_cfg_path,
                seed=seed_offset,
                device=device,
                decision_interval=decision_interval,
            )
        return _make
    
    base_env = TorchRLEnvWrapper(
                cfg_path=env_cfg_path,
                seed=env_cfg["run"]["seed"],
                device="cpu",
                decision_interval=decision_interval,
            )

    n_edges = base_env.n_edges
    obs_dim = base_env.obs_dim          # len(obs_keys)
    obs_flat_dim = n_edges * obs_dim    # or: base_env.obs_size
    penv = ParallelEnv(
        num_envs,
        [make_env(1000 + i) for i in range(num_envs)],
    )

    env = TransformedEnv(
        penv,
        Compose(
            InitTracker(),  # provides "is_init" so LSTM can reset hidden state
            ObservationNorm(
                in_keys=["observation_flat"],
                standard_normal=train_cfg["observation_norm"]["standard_normal"],
            ),
        ),
    )
    

    # populate mean/std from rollouts
    norm = env.transform[-1]
    env.transform.train()
    norm.init_stats(num_iter=100, reduce_dim=0, cat_dim=None)
    env.transform.eval()
    
    # ---- FIX: your build stores sample bank in loc/scale with shape (N,D); compress to (D,) ----
    def _get_buf(mod, names):
        for n in names:
            if hasattr(mod, n):
                return getattr(mod, n), n
            if n in mod._buffers:
                return mod._buffers[n], n
        raise KeyError(f"none of {names} found in buffers/attrs")

    def _set_buf(mod, name, tensor):
        if name in mod._buffers:
            mod._buffers[name] = tensor
        else:
            mod.register_buffer(name, tensor)

    loc_buf, loc_name = _get_buf(norm, ["loc", "_loc"])
    scale_buf, scale_name = _get_buf(norm, ["scale", "_scale"])

    # If loc is (N,D), treat it as samples and compute mean/std
    if loc_buf.ndim == 2:
        samples = loc_buf  # (N,D)
        mu = samples.mean(dim=0)
        sigma = samples.std(dim=0, unbiased=False).clamp_min(1e-6)

        _set_buf(norm, loc_name, mu)
        _set_buf(norm, scale_name, sigma)

    # sanity
    loc_buf2, _ = _get_buf(norm, ["loc", "_loc"])
    scale_buf2, _ = _get_buf(norm, ["scale", "_scale"])

    # ---- actor ----
    unflatten = TensorDictModule(
        UnflattenObs(n_edges=n_edges, obs_dim=obs_dim),
        in_keys=["observation_flat"],
        out_keys=["observation"],
    )

    pool = TensorDictModule(
        MeanPoolEdges(),
        in_keys=["features"],
        out_keys=["features_pooled"],
    )

    feature_module = TensorDictModule(
        FeatureNet(obs_dim, feature_dim).to(device),  # obs_dim, not obs_flat_dim
        in_keys=["observation"],                      # [*, n_edges, obs_dim]
        out_keys=["features"],                        # [*, n_edges, feature_dim]
    )

    shared_core = TensorDictSequential(
        unflatten,
        feature_module,
        EdgewiseLSTM(feature_dim, feature_dim, num_layers=1, device=device).to(device),
    )


    actor_head = TensorDictModule(
        nn.Linear(feature_dim, train_cfg["model"]["n_actions"]).to(device),
        in_keys=["features"],
        out_keys=["logits"],
    )

    critic_head = TensorDictModule(
        nn.Linear(feature_dim, 1).to(device),
        in_keys=["features_pooled"],
        out_keys=["state_value"],
    )

    critic_core = TensorDictSequential(shared_core, pool, critic_head)
    
    sampler_td = TensorDictModule(
        MultiEdgeCategoricalSampler().to(device),
        in_keys=["logits"],
        out_keys=["action", "sample_log_prob"],
    )

    logprob_td = TensorDictModule(
        MultiEdgeCategoricalLogProb().to(device),
        in_keys=["logits", "action"],
        out_keys=["sample_log_prob"],
    )    


    policy_backbone = TensorDictSequential(shared_core, actor_head)  # writes "logits"

    prob_actor = ProbabilisticActor(
        module=policy_backbone,
        in_keys=["logits"],
        distribution_class=MultiEdgeIndependentCategorical,
        distribution_kwargs={"validate_args": False},
        out_keys=["action"],
        return_log_prob=True,
        log_prob_key="sample_log_prob",
        default_interaction_type=InteractionType.RANDOM,
    )

    collector_policy = prob_actor
    loss_actor = prob_actor
    
    class SumIfEdgewise(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # if log_prob came out as [..., E], sum it
            if x.ndim >= 1 and x.shape[-1] == n_edges:
                return x.sum(-1, keepdim=True)  # [..., 1]
            # if already scalar, keep as is
            return x if x.ndim == 0 else x.unsqueeze(-1) if x.ndim == 1 else x

    # critic for loss, head-only
    value = critic_core   # reads "features" -> writes "state_value"
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
        deactivate_vmap=True,
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
        functional=False,
    )

    loss.set_keys(
        value="state_value",
        advantage="advantage",
        value_target="value_target",
        sample_log_prob="sample_log_prob",   # this is the key change
        action="action",
    )

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
            # print("batch['observation_flat'].shape:", batch['observation_flat'].shape)
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
            critic_core(traj)          # writes features, features_pooled, state_value
            critic_core(traj["next"])  # same for next
            adv(traj)

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
                seq_list = [data[t0_idx + k, b_idx] for k in range(seq_len_eff)]

                mb_td = torch.stack(seq_list, dim=0).to_tensordict()   # [T, mb, ...]
                mb_td = mb_td.transpose(0, 1).contiguous()            # [mb, T, ...]    

                
                for k in ["advantage", "value_target", "state_value", "sample_log_prob"]:
                    if k in mb_td.keys():
                        v = mb_td.get(k)
                        if torch.is_tensor(v) and v.ndim == 3 and v.shape[-1] == 1:
                            mb_td.set(k, v.squeeze(-1))                

                alpha = 1.0 - (num_network_updates / total_network_updates)
                if alpha < 0.0:
                    alpha = 0.0
                _apply_anneal(alpha)
                num_network_updates += 1
                if it == 0:
                    td_tmp = mb_td.clone(False)
                    loss_actor(td_tmp)
                    print("cur logp", td_tmp["sample_log_prob"].shape)
                    print("old logp", mb_td["sample_log_prob"].shape)
                    print("adv", mb_td["advantage"].shape)                 
                                    
                policy_backbone(mb_td)                 # writes "logits" into mb_td
                logits = mb_td["logits"]               # [mb, T, E, A] or [mb, T, E, A] depending on your net
                dist = Independent(Categorical(logits=logits), 1)  # event_dim=E, joint action

                cur_logp = dist.log_prob(mb_td["action"])          # [mb, T]
                old_logp = mb_td["sample_log_prob"].detach()       # [mb, T]

                advantage = mb_td["advantage"].detach()            # [mb, T]
                ratio = (cur_logp - old_logp).exp()                # [mb, T]

                eps = float(train_cfg["loss"]["clip_epsilon"])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantage
                policy_loss = -(torch.minimum(surr1, surr2)).mean()

                entropy = dist.entropy().mean()
                entropy_coeff = float(train_cfg["loss"]["entropy_coeff"])
                entropy_loss = -entropy_coeff * entropy            # minus because we want to maximize entropy

                # ---- critic loss (with grad) ----
                critic_core(mb_td)                                  # writes "state_value" with grad
                value_pred = mb_td["state_value"]                   # [mb, T] or [mb, T, 1]
                value_tgt = mb_td["value_target"].detach()

                # make both [mb, T]
                if value_pred.ndim == 3 and value_pred.shape[-1] == 1:
                    value_pred = value_pred.squeeze(-1)
                if value_tgt.ndim == 3 and value_tgt.shape[-1] == 1:
                    value_tgt = value_tgt.squeeze(-1)

                critic_loss = 0.5 * (value_pred - value_tgt).pow(2).mean()
                critic_coeff = float(train_cfg["loss"]["critic_coeff"])

                total_loss = policy_loss + critic_coeff * critic_loss + entropy_loss

                optim.zero_grad(set_to_none=True)
                total_loss.backward()

                params = (
                    list(shared_core.parameters())
                    + list(actor_head.parameters())
                    + list(critic_head.parameters())
                )
                torch.nn.utils.clip_grad_norm_(params, float(train_cfg["optim"]["max_grad_norm"]))
                optim.step()

                last_total_loss = total_loss
                last_policy_loss = policy_loss
                last_critic_loss = critic_loss
                last_entropy = entropy

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

        wandb.log(
            {
                "iter": it,
                "qoe/mean": float(batch["next", "qoe_mean"].mean().item()),
                "reward/mean": float(batch["next", "reward"].mean().item()),
                "loss/total": float(last_total_loss.detach().item()),
                "loss/policy": float(last_policy_loss.detach().item()),
                "loss/critic": float(last_critic_loss.detach().item()),
                "loss/entropy": float(last_entropy.detach().item()),
            },
        )

        collector.update_policy_weights_()
if __name__ == "__main__":
    # train(resume_ckpt="checkpoints/lstm_epoch_20_linear/ckpt_iter_000950.pt")
    train()