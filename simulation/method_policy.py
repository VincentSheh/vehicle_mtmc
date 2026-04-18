"""
Baseline policy implementations for evaluation.

Each policy implements act(ctx: ActContext) -> (ids_cpu, delta) where:
  - ids_cpu: np.ndarray of shape (n_edges,) with absolute CPU values (or None to use delta)
  - delta:   np.ndarray of shape (n_edges,) with integer deltas (-1/0/+1)

If ids_cpu is not None, delta is ignored (ids_cpu is used directly after clipping).
If ids_cpu is None, delta is applied via apply_delta.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tbsa_offline import TBSAPolicy


# ---------------------------------------------------------------------------
# Context object passed to each policy's act() call
# ---------------------------------------------------------------------------
@dataclass
class ActContext:
    """All information a policy may need to make a decision."""
    env: object                    # Environment instance (for history, edge_areas)
    ids_cpu: np.ndarray            # last-committed (settled) IDS CPU, shape (n_edges,)
    ids_cpu_min: float
    ids_cpu_max: np.ndarray        # shape (n_edges,)
    cpu_util: float                # scalar decision-window CPU utilisation
    decision_interval: int
    rng: np.random.Generator       # seeded RNG for stochastic policies
    transition_ticks_norm: float = 0.0   # remaining ticks / max_duration ∈ [0, 1]
    delta_in_flight_norm: float = 0.0    # (ids_cpu_target - ids_cpu_settled) / ids_cpu_max ∈ [-1, 1]
    obs_flat: Optional[np.ndarray] = None  # pre-built normalised obs for RL policies


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaselinePolicy(abc.ABC):
    @abc.abstractmethod
    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        """
        Returns (ids_cpu_abs, delta).
        If ids_cpu_abs is not None, caller should use it directly (after clip).
        Otherwise caller should apply delta via apply_delta.
        """

    def reset(self) -> None:
        """Called at the start of each episode."""


# ---------------------------------------------------------------------------
# Constant CPU policy
# ---------------------------------------------------------------------------
class ConstantPolicy(BaselinePolicy):
    def __init__(self, cpu_value: float):
        self.cpu_value = cpu_value

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        n_edges = len(ctx.ids_cpu)
        ids_cpu_abs = np.clip(
            np.full(n_edges, self.cpu_value, dtype=np.float32),
            ctx.ids_cpu_min,
            ctx.ids_cpu_max,
        )
        return ids_cpu_abs, np.zeros(n_edges, dtype=np.int64)


# ---------------------------------------------------------------------------
# Random delta policy
# ---------------------------------------------------------------------------
class RandomPolicy(BaselinePolicy):
    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        n_edges = len(ctx.ids_cpu)
        delta = ctx.rng.integers(-1, 2, size=n_edges, dtype=np.int64)
        return None, delta


# ---------------------------------------------------------------------------
# Reactive threshold policy
# ---------------------------------------------------------------------------
class ReactivePolicy(BaselinePolicy):
    def __init__(self, high_threshold: float = 0.80, low_threshold: float = 0.20):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        n_edges = len(ctx.ids_cpu)
        if ctx.cpu_util >= self.high_threshold:
            delta = np.ones(n_edges, dtype=np.int64)
        elif ctx.cpu_util <= self.low_threshold:
            delta = -np.ones(n_edges, dtype=np.int64)
        else:
            delta = np.zeros(n_edges, dtype=np.int64)
        return None, delta


# ---------------------------------------------------------------------------
# TBSA wrapper policy
# ---------------------------------------------------------------------------
class TBSAWrapperPolicy(BaselinePolicy):
    def __init__(self, tbsa: TBSAPolicy):
        self._tbsa = tbsa

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        n_edges = len(ctx.ids_cpu)
        env = ctx.env

        if env.history:
            last_records = env.history[-n_edges:]
            last_attack = float(np.mean([r.attack_drop_rate for r in last_records]))
            last_req = float(np.mean([r.local_num_req for r in last_records]))
        else:
            last_attack = 0.0
            last_req = 0.0

        target_cpu = self._tbsa.select_ids_cpu(last_attack, last_req)
        ids_cpu_abs = np.clip(
            np.full(n_edges, target_cpu, dtype=np.float32),
            ctx.ids_cpu_min,
            ctx.ids_cpu_max,
        )
        return ids_cpu_abs, np.zeros(n_edges, dtype=np.int64)


# ---------------------------------------------------------------------------
# LSTM RL policy (loads a train_lstm.py checkpoint for greedy inference)
# ---------------------------------------------------------------------------
class LSTMRLPolicy(BaselinePolicy):
    """
    Wraps a checkpoint produced by train_lstm.py (split → LSTM → merge → actor).

    Expects ActContext.obs_flat to be pre-built and pre-normalised by the
    eval loop (run_baseline.py sets this before calling act()).
    Returns (ids_cpu_abs, zeros) so the eval loop uses it directly.
    """

    def __init__(
        self,
        ckpt_path: str,
        obs_keys: List[str],
        device: str = "cpu",
        greedy: bool = True,
    ):
        import torch
        import torch.nn as nn
        from tensordict.nn import TensorDictModule, TensorDictSequential
        from torchrl.modules import LSTMModule as TorchRLLSTM
        from train_lstm import SplitObsModule, MergeModule

        self.device = torch.device(device)
        self.greedy = greedy
        self.obs_keys = list(obs_keys)

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        train_cfg = state["train_cfg"]

        feature_dim   = train_cfg["model"]["hidden_dim"]
        n_actions     = train_cfg["model"]["n_actions"]
        n_temporal    = train_cfg["model"]["n_temporal"]
        n_static      = train_cfg["model"]["n_static"]
        temporal_idx  = train_cfg["model"]["temporal_idx"]
        static_idx    = train_cfg["model"]["static_idx"]

        self.n_actions  = n_actions
        self.scale_step = 0.5
        self.h_size     = feature_dim

        # Reconstruct the same topology used during training
        split_module = TensorDictModule(
            SplitObsModule(temporal_idx, static_idx).to(self.device),
            in_keys=["observation_flat"],
            out_keys=["temporal_obs", "static_obs"],
        )

        self._lstm_mod = TorchRLLSTM(
            input_size=n_temporal,
            hidden_size=feature_dim,
            in_key="temporal_obs",
            out_key="lstm_out",
            device=self.device,
        )

        merge_module = TensorDictModule(
            MergeModule(),
            in_keys=["lstm_out", "static_obs"],
            out_keys=["features_merged"],
        )

        actor_head = TensorDictModule(
            nn.Linear(feature_dim + n_static, n_actions).to(self.device),
            in_keys=["features_merged"],
            out_keys=["logits"],
        )

        # inference network: split → lstm → merge → actor_head
        self.net = TensorDictSequential(
            split_module, self._lstm_mod, merge_module, actor_head
        ).to(self.device)

        # ---- key remapping -----------------------------------------------
        # Confirmed checkpoint key structure (from observed warnings):
        #
        #   ProbabilisticActor stores TDS as self.module  →  prefix "module."
        #   Outer TDS(shared_core, actor_head) stores children under .module[i]
        #     .module[0] = shared_core = TDS(split, lstm, merge)  → "module.0."
        #     .module[1] = actor_head  = TDM(nn.Linear)           → "module.0.module.1."
        #                  ↑ note: this is .module[0].module[1] relative to PA.module
        #   shared_core TDS stores its children under .module[i]
        #     .module[0] = split_module  = TDM(SplitObsModule)    → "module.0.module.0."
        #     .module[1] = lstm          = LSTMModule              → "module.0.module.0.module.1."
        #     .module[2] = merge_module  = TDM(MergeModule)        → (no params)
        #   TDM wraps its inner module as .module:
        #     split_module.module = SplitObsModule → buffers at "…module.0.module.0.module."
        #
        # Observed ckpt prefixes → target net prefix (our TDS(split,lstm,merge,actor_head)):
        #   "module.0.module.0.module.0."  → "module.0."    (split_module TDM layer)
        #   "module.0.module.0.module.1."  → "module.1."    (LSTMModule)
        #   "module.0.module.0.module.2."  → "module.2."    (merge_module, no params)
        #   "module.0.module.1."           → "module.3."    (actor_head TDM layer)
        # ------------------------------------------------------------------
        policy_sd = state["policy"]
        mapped_sd: dict = {}
        for k, v in policy_sd.items():
            nk = k
            if nk.startswith("module.0.module.0.module.0."):
                nk = "module.0." + nk[len("module.0.module.0.module.0."):]
            elif nk.startswith("module.0.module.0.module.1."):
                nk = "module.1." + nk[len("module.0.module.0.module.1."):]
            elif nk.startswith("module.0.module.0.module.2."):
                nk = "module.2." + nk[len("module.0.module.0.module.2."):]
            elif nk.startswith("module.0.module.1."):
                nk = "module.3." + nk[len("module.0.module.1."):]
            mapped_sd[nk] = v

        missing, unexpected = self.net.load_state_dict(mapped_sd, strict=False)
        if missing:
            print(f"[LSTMRLPolicy] WARNING: missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[LSTMRLPolicy] WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        self.net.eval()

        # ---- observation normalisation -----------------------------------
        # TransformedEnv(Compose(InitTracker, lstm_primer, ObservationNorm))
        # saves state as: transforms.2.loc / transforms.2.scale / transforms.2.standard_normal
        _OBSNORM_LOC_KEY   = "transforms.2.loc"
        _OBSNORM_SCALE_KEY = "transforms.2.scale"
        _OBSNORM_STD_KEY   = "transforms.2.standard_normal"

        obsnorm = state.get("obsnorm", None)
        self.obs_loc   = None
        self.obs_scale = None
        if obsnorm is None:
            print("[LSTMRLPolicy] WARNING: checkpoint has no 'obsnorm' — running WITHOUT normalisation")
        elif _OBSNORM_LOC_KEY in obsnorm and _OBSNORM_SCALE_KEY in obsnorm:
            _sn = obsnorm.get(_OBSNORM_STD_KEY, True)
            std_normal = bool(_sn.item() if hasattr(_sn, "item") else _sn)
            if not std_normal:
                raise ValueError(
                    "LSTMRLPolicy only supports standard_normal=True ObservationNorm; "
                    "checkpoint was saved with standard_normal=False"
                )
            self.obs_loc   = obsnorm[_OBSNORM_LOC_KEY].detach().to(self.device).reshape(-1)
            self.obs_scale = obsnorm[_OBSNORM_SCALE_KEY].detach().to(self.device).reshape(-1)
            print(f"[LSTMRLPolicy] ObsNorm loaded: loc={self.obs_loc.tolist()}, scale={self.obs_scale.tolist()}")
        else:
            # Fallback: search by suffix (warns so silent failure is impossible)
            loc_key   = next((k for k in obsnorm if k.endswith("loc")),   None)
            scale_key = next((k for k in obsnorm if k.endswith("scale")), None)
            if loc_key and scale_key:
                print(f"[LSTMRLPolicy] WARNING: expected keys {_OBSNORM_LOC_KEY!r}/{_OBSNORM_SCALE_KEY!r} "
                      f"not found; falling back to suffix match: {loc_key!r}/{scale_key!r}")
                self.obs_loc   = obsnorm[loc_key].detach().to(self.device).reshape(-1)
                self.obs_scale = obsnorm[scale_key].detach().to(self.device).reshape(-1)
            else:
                print(f"[LSTMRLPolicy] WARNING: no loc/scale keys found in obsnorm "
                      f"(keys: {list(obsnorm.keys())}) — running WITHOUT normalisation")

        self._h: Optional[object] = None
        self._c: Optional[object] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._h = None
        self._c = None

    def _normalise(self, obs_flat: np.ndarray) -> np.ndarray:
        if self.obs_loc is None:
            return obs_flat
        loc   = self.obs_loc.detach().cpu().numpy()
        scale = self.obs_scale.detach().cpu().numpy()
        return (obs_flat - loc) / (scale + 1e-8)

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        import torch
        from tensordict import TensorDict

        n_edges = len(ctx.ids_cpu)

        if ctx.obs_flat is None:
            return None, np.zeros(n_edges, dtype=np.int64)

        obs_np = self._normalise(ctx.obs_flat.astype(np.float32))
        obs_t  = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)   # [1, obs_size]

        if self._h is None:
            self._h = torch.zeros(1, 1, self.h_size, device=self.device)
            self._c = torch.zeros(1, 1, self.h_size, device=self.device)

        td = TensorDict(
            {
                "observation_flat":  obs_t,
                "recurrent_state_h": self._h,
                "recurrent_state_c": self._c,
                "is_init": torch.zeros(1, 1, device=self.device, dtype=torch.bool),
            },
            batch_size=[1],
            device=self.device,
        )

        with torch.no_grad():
            td = self.net(td)

        self._h = td.get(("next", "recurrent_state_h"))
        self._c = td.get(("next", "recurrent_state_c"))

        logits = td.get("logits").squeeze(0)
        if self.greedy:
            action = int(torch.argmax(logits).item())
        else:
            probs  = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())

        # Map discrete action → absolute CPU (netting: relative to current queue)
        delta_cmd = (action - (self.n_actions - 1) / 2.0) * self.scale_step
        ids_cpu_abs = np.clip(
            ctx.ids_cpu + float(delta_cmd),   # ctx.ids_cpu == current queue position
            ctx.ids_cpu_min,
            ctx.ids_cpu_max,
        ).astype(np.float32)

        return ids_cpu_abs, np.zeros(n_edges, dtype=np.int64)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_baseline_policy(
    name: str,
    tbsa_table_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    obs_keys: Optional[List[str]] = None,
    device: str = "cpu",
) -> BaselinePolicy:
    """
    name examples: "random", "constant_0.5", "constant_1.5", "reactive", "tbsa", "lstm_rl"
    """
    if name.startswith("constant_"):
        cpu_val = float(name.split("_", 1)[1])
        return ConstantPolicy(cpu_val)
    elif name == "random":
        return RandomPolicy()
    elif name == "reactive":
        return ReactivePolicy()
    elif name == "tbsa":
        if tbsa_table_path is None:
            raise ValueError("tbsa_table_path must be provided for 'tbsa' policy")
        tbsa = TBSAPolicy(tbsa_table_path)
        return TBSAWrapperPolicy(tbsa)
    elif name == "lstm_rl":
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided for 'lstm_rl' policy")
        if obs_keys is None:
            raise ValueError("obs_keys must be provided for 'lstm_rl' policy")
        return LSTMRLPolicy(ckpt_path=ckpt_path, obs_keys=obs_keys, device=device, greedy=False)
    else:
        raise ValueError(f"Unknown baseline policy name: {name!r}")
