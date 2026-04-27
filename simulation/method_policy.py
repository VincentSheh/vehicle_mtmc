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
from typing import Dict, List, Optional

import numpy as np

from tbsa_offline import TBSAPolicy


# Human-readable labels for offload modes (shared across eval scripts)
OFFLOAD_DISPLAY_NAMES: Dict[str, str] = {
    "none":           "No Offload",
    "balance":        "Balance",
    "delay_workload": "Delay-Weighted",
    "full":           "Full Score",
    "cto":            "CTO",
    "cto_acc":        "CTO+Acc",
}

# Human-readable labels for model types (gm = global model, lm = local model)
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "gm": "Global Model",
    "lm": "Local Model",
}


# ---------------------------------------------------------------------------
# Context object passed to each policy's act() call
# ---------------------------------------------------------------------------
@dataclass
class ActContext:
    """All information a policy may need to make a decision."""
    env: object                    # Environment instance (for history, edge_areas)
    ids_cpu: np.ndarray            # current queue position (ids_cpu), shape (n_edges,)
    ids_cpu_min: float
    ids_cpu_max: np.ndarray        # shape (n_edges,)
    cpu_util: float                # scalar decision-window CPU utilisation (max across edges)
    decision_interval: int
    rng: np.random.Generator       # seeded RNG for stochastic policies
    transition_ticks_norm: np.ndarray    # shape (n_edges,)
    delta_in_flight_norm: np.ndarray     # shape (n_edges,)
    obs_flat: Optional[np.ndarray] = None  # pre-built normalised obs for RL policies


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaselinePolicy(abc.ABC):
    min_cpu_override: Optional[float] = None

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
# No IDS policy (IDS fully disabled, CPU = 0)
# ---------------------------------------------------------------------------
class NoIDSPolicy(BaselinePolicy):
    min_cpu_override = 0.0

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        n = len(ctx.ids_cpu)
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.int64)


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
# App workload autoscaling policy (wrong signal: VA utilization, not attack load)
# ---------------------------------------------------------------------------
class AppAutoscalePolicy(BaselinePolicy):
    def __init__(self, high_threshold: float = 0.80, low_threshold: float = 0.20):
        self.high_threshold = high_threshold
        self.low_threshold  = low_threshold

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        import pandas as pd
        n   = len(ctx.ids_cpu)
        env = ctx.env
        va_util = 0.0
        if env.history:
            records = env.history[-ctx.decision_interval * n:]
            df = pd.DataFrame([m.__dict__ for m in records])
            if "va_cpu_utilization" in df.columns:
                va_util = float(np.mean(df["va_cpu_utilization"].values))
        if va_util >= self.high_threshold:
            delta = np.ones(n, dtype=np.int64)
        elif va_util <= self.low_threshold:
            delta = -np.ones(n, dtype=np.int64)
        else:
            delta = np.zeros(n, dtype=np.int64)
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
        from train_sa_lstm import SplitObsModule, MergeModule

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

        self.n_actions       = n_actions
        self.scale_step      = 0.5
        self.h_size          = feature_dim
        self.expected_obs_dim = n_temporal + n_static

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

        # Per-edge LSTM hidden states (list of (h, c) tensors, one per edge)
        self._hs: Optional[list] = None
        self._cs: Optional[list] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._hs = None
        self._cs = None

    def _normalise(self, obs_flat: np.ndarray) -> np.ndarray:
        if self.obs_loc is None:
            return obs_flat
        loc   = self.obs_loc.detach().cpu().numpy()
        scale = self.obs_scale.detach().cpu().numpy()
        return (obs_flat - loc) / (scale + 1e-8)

    def _extract_per_edge_obs(self, obs_flat: np.ndarray, n_edges: int) -> np.ndarray:
        """
        Returns array of shape (n_edges, expected_obs_dim).

        When obs_flat matches expected_obs_dim exactly (single-edge), wraps it.
        When obs_flat is larger (multi-edge with neighbor features), reshapes to
        (n_edges, per_edge_full) then strips the neighbor columns in the middle,
        producing (n_edges, expected_obs_dim).

        Obs layout assumed (from _build_obs_flat in eval_baselines.py):
          [base(n_base) | neighbor(n_nbr, multi-edge only) | slo(1) | transition(2)]
        where n_tail = 3 (slo + transition) and n_base = expected_obs_dim - n_tail.
        """
        if obs_flat.shape[0] == self.expected_obs_dim:
            return obs_flat.reshape(1, self.expected_obs_dim)

        per_edge_full = obs_flat.shape[0] // n_edges
        obs_2d = obs_flat.reshape(n_edges, per_edge_full)
        n_tail  = 3   # slo_vio + transition_ticks_norm + delta_in_flight_norm
        n_base  = self.expected_obs_dim - n_tail
        n_nbr   = per_edge_full - self.expected_obs_dim
        return np.concatenate([obs_2d[:, :n_base], obs_2d[:, n_base + n_nbr:]], axis=1)

    def _run_one_edge(self, obs_np: np.ndarray, h: object, c: object):
        """Run one forward pass for a single edge. Returns (logits, h_next, c_next)."""
        import torch
        from tensordict import TensorDict

        obs_t = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)  # [1, obs_dim]
        td = TensorDict(
            {
                "observation_flat":  obs_t,
                "recurrent_state_h": h,
                "recurrent_state_c": c,
                "is_init": torch.zeros(1, 1, device=self.device, dtype=torch.bool),
            },
            batch_size=[1],
            device=self.device,
        )
        with torch.no_grad():
            td = self.net(td)
        return (
            td.get("logits").squeeze(0),
            td.get(("next", "recurrent_state_h")),
            td.get(("next", "recurrent_state_c")),
        )

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        import torch

        n_edges = len(ctx.ids_cpu)

        if ctx.obs_flat is None:
            return None, np.zeros(n_edges, dtype=np.int64)

        # Lazily init one (h, c) pair per edge
        if self._hs is None:
            self._hs = [torch.zeros(1, 1, self.h_size, device=self.device) for _ in range(n_edges)]
            self._cs = [torch.zeros(1, 1, self.h_size, device=self.device) for _ in range(n_edges)]

        # Extract per-edge obs slices (strips neighbor features when in multi-edge env)
        per_edge_obs = self._extract_per_edge_obs(ctx.obs_flat.astype(np.float32), n_edges)

        ids_cpu_abs = np.empty(n_edges, dtype=np.float32)
        for i in range(n_edges):
            obs_np = self._normalise(per_edge_obs[i])
            logits, h_next, c_next = self._run_one_edge(obs_np, self._hs[i], self._cs[i])
            self._hs[i] = h_next
            self._cs[i] = c_next

            if self.greedy:
                action = int(torch.argmax(logits).item())
            else:
                probs  = torch.softmax(logits, dim=-1)
                action = int(torch.multinomial(probs, 1).item())

            delta_cmd = (action - (self.n_actions - 1) / 2.0) * self.scale_step
            ids_cpu_abs[i] = float(np.clip(
                ctx.ids_cpu[i] + delta_cmd,
                ctx.ids_cpu_min,
                ctx.ids_cpu_max[i],
            ))

        return ids_cpu_abs, np.zeros(n_edges, dtype=np.int64)


# ---------------------------------------------------------------------------
# Multi-Agent LSTM RL policy (loads a train_ma_lstm.py checkpoint)
# ---------------------------------------------------------------------------
class MALSTMRLPolicy(BaselinePolicy):
    """
    Wraps a multi-agent checkpoint produced by train_ma_lstm.py (AgentRecurrentCore + actor head).

    Builds per-edge observations directly from the environment history using the
    same 11-feature layout used during MA training, runs a single joint forward
    pass for all edges, and returns per-edge CPU targets.
    """
    # Obs layout from EdgeIDSParallelEnv.obs_keys — fixed for all MA checkpoints
    _OBS_KEYS: List[str] = [
        "ids_user_in_rate",
        "attack_in_rate",
        "ema_mom",
        "cpu_to_ids_ratio",
        "ids_cpu_utilization",
        "neighbor_ids_util",
        "neighbor_delta",
        "neighbor_atk_rate",
        "prev_slo_vio",
        "transition_ticks_norm",
        "delta_in_flight_norm",
    ]
    _OBS_DIM = 11
    _TEMPORAL_OBS_KEYS = {"ids_user_in_rate", "attack_in_rate", "neighbor_atk_rate"}
    # First 5 are read from history; last 6 are computed from transition/neighbor state
    _BASE_N = 5

    def __init__(self, ckpt_path: str, device: str = "cpu", greedy: bool = False):
        import torch
        import torch.nn as nn
        from train_ma_lstm import AgentRecurrentCore

        self.device = torch.device(device)
        self.greedy = greedy
        self.scale_step = 0.5

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        train_cfg = state["train_cfg"]

        hidden_dim     = int(train_cfg["model"]["hidden_dim"])
        self.n_actions = int(train_cfg["model"]["n_actions"])
        self.hidden_dim = hidden_dim

        # Derive n_edges from obsnorm shape  (loc shape = n_edges * obs_dim)
        obsnorm = state.get("obsnorm", {})
        _LOC_KEY   = "transforms.2.loc"
        _SCALE_KEY = "transforms.2.scale"
        if _LOC_KEY in obsnorm:
            self.n_edges = int(obsnorm[_LOC_KEY].shape[0]) // self._OBS_DIM
        else:
            env_cfg = state.get("env_cfg", {})
            self.n_edges = max(1, len(env_cfg.get("edge_areas", [])))

        temporal_local_idx = [i for i, k in enumerate(self._OBS_KEYS) if k in self._TEMPORAL_OBS_KEYS]
        static_local_idx   = [i for i, k in enumerate(self._OBS_KEYS) if k not in self._TEMPORAL_OBS_KEYS]
        n_static       = len(static_local_idx)
        actor_feat_dim = hidden_dim + n_static

        self.actor_core = AgentRecurrentCore(
            n_edges=self.n_edges,
            obs_dim=self._OBS_DIM,
            hidden_dim=hidden_dim,
            device=str(self.device),
            temporal_local_idx=temporal_local_idx,
            static_local_idx=static_local_idx,
        )
        self.actor_head = nn.Linear(actor_feat_dim, self.n_actions).to(self.device)

        # Map checkpoint keys:
        #   ProbabilisticActor(TDS(core, head)).state_dict() →
        #     "module.module.0.*"        (AgentRecurrentCore)
        #     "module.module.1.module.*" (actor head TDM's inner nn.Linear)
        policy_sd = state["policy"]
        CORE_PFX  = "module.module.0."
        HEAD_PFX  = "module.module.1.module."
        core_sd = {k[len(CORE_PFX):]: v for k, v in policy_sd.items() if k.startswith(CORE_PFX)}
        head_sd = {k[len(HEAD_PFX):]: v for k, v in policy_sd.items() if k.startswith(HEAD_PFX)}

        if not core_sd:
            prefixes = sorted({".".join(k.split(".")[:3]) for k in policy_sd.keys()})
            print(f"[MALSTMRLPolicy] WARNING: no keys matched core prefix '{CORE_PFX}'; "
                  f"found prefixes: {prefixes}")

        missing, unexpected = self.actor_core.load_state_dict(core_sd, strict=False)
        if missing:
            print(f"[MALSTMRLPolicy] WARNING: missing core keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[MALSTMRLPolicy] WARNING: unexpected core keys ({len(unexpected)}): {unexpected[:5]}")
        missing, unexpected = self.actor_head.load_state_dict(head_sd, strict=False)
        if missing:
            print(f"[MALSTMRLPolicy] WARNING: missing head keys ({len(missing)}): {missing[:5]}")

        self.actor_core.eval()
        self.actor_head.eval()

        # Observation normalisation (ObservationNorm at transform index 2)
        self.obs_loc = self.obs_scale = None
        if _LOC_KEY in obsnorm and _SCALE_KEY in obsnorm:
            self.obs_loc   = obsnorm[_LOC_KEY].detach().to(self.device)
            self.obs_scale = obsnorm[_SCALE_KEY].detach().to(self.device)
            print(f"[MALSTMRLPolicy] ObsNorm loaded: shape={self.obs_loc.shape}, n_edges={self.n_edges}")
        else:
            print("[MALSTMRLPolicy] WARNING: no obsnorm found — running WITHOUT normalisation")

        self._h: Optional[object] = None
        self._c: Optional[object] = None

    def reset(self) -> None:
        self._h = None
        self._c = None

    def _build_ma_obs(self, ctx: ActContext) -> np.ndarray:
        """Replicate EdgeIDSParallelEnv._build_observation() from the eval-loop context."""
        import pandas as pd

        env      = ctx.env
        n_edges  = len(ctx.ids_cpu)
        area_ids = [e.area_id for e in env.edge_areas]
        obs      = np.zeros((n_edges, self._OBS_DIM), dtype=np.float32)

        if not env.history:
            return obs.reshape(-1)

        records = env.history[-ctx.decision_interval * n_edges:]
        df      = pd.DataFrame([m.__dict__ for m in records])

        # Base 5 features read from history (indices 0-4)
        for i, area_id in enumerate(area_ids):
            g = df[df["area_id"] == area_id]
            if g.empty:
                continue
            for j, k in enumerate(self._OBS_KEYS[:self._BASE_N]):
                if k not in g.columns:
                    continue
                obs[i, j] = float(g[k].values[-1]) if k == "cpu_to_ids_ratio" else float(np.mean(g[k].values))

        # Neighbor features: ids_util (5), delta (6), atk_rate (7)
        edge_ids_util: Dict[str, float] = {}
        edge_atk_rate: Dict[str, float] = {}
        for area_id in area_ids:
            g = df[df["area_id"] == area_id]
            if g.empty:
                edge_ids_util[area_id] = 0.0
                edge_atk_rate[area_id] = 0.0
            else:
                edge_ids_util[area_id] = float(np.clip(
                    np.mean(g["ids_cpu_utilization"].values if "ids_cpu_utilization" in g.columns else [0.0]),
                    0.0, 1.0))
                edge_atk_rate[area_id] = float(np.mean(
                    g["attack_in_rate"].values if "attack_in_rate" in g.columns else [0.0]))

        for i, area_id in enumerate(area_ids):
            nbr_utils, nbr_deltas, nbr_atk = [], [], []
            for j in range(n_edges):
                if j == i:
                    continue
                nbr_utils.append(edge_ids_util[area_ids[j]])
                nbr_atk.append(edge_atk_rate[area_ids[j]])
                nbr_deltas.append(float(ctx.delta_in_flight_norm[j]))  # already normalised
            obs[i, 5] = float(np.mean(nbr_utils))  if nbr_utils  else 0.0
            obs[i, 6] = float(np.mean(nbr_deltas)) if nbr_deltas else 0.0
            obs[i, 7] = float(np.mean(nbr_atk))    if nbr_atk    else 0.0

        # prev_slo_vio (index 8)
        for i, area_id in enumerate(area_ids):
            g = df[df["area_id"] == area_id]
            if g.empty:
                obs[i, 8] = 0.0
                continue
            last_qoe  = float(g["qoe_mean"].values[-1]) if "qoe_mean" in g.columns else 0.0
            threshold = float(env.edge_areas[i].slo_threshold)
            obs[i, 8] = 1.0 if last_qoe < threshold else 0.0

        # Transition state (indices 9, 10) — taken directly from ctx
        for i in range(n_edges):
            obs[i, 9]  = float(ctx.transition_ticks_norm[i])
            obs[i, 10] = float(ctx.delta_in_flight_norm[i])

        return obs.reshape(-1).astype(np.float32)

    def _normalise(self, obs_flat: np.ndarray) -> np.ndarray:
        if self.obs_loc is None:
            return obs_flat
        import torch
        x = torch.from_numpy(obs_flat).to(self.device)
        return ((x - self.obs_loc) / (self.obs_scale + 1e-8)).cpu().numpy()

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        import torch
        from tensordict import TensorDict

        n_edges = len(ctx.ids_cpu)

        obs_flat = self._build_ma_obs(ctx)
        obs_norm = self._normalise(obs_flat)

        if self._h is None:
            self._h = torch.zeros(1, n_edges, 1, self.hidden_dim, device=self.device)
            self._c = torch.zeros(1, n_edges, 1, self.hidden_dim, device=self.device)

        obs_t = torch.from_numpy(obs_norm).float().to(self.device).unsqueeze(0)  # (1, n_edges*obs_dim)

        td = TensorDict(
            {
                "observation_flat":              obs_t,
                "is_init":                       torch.zeros(1, 1, device=self.device, dtype=torch.bool),
                ("agents", "recurrent_state_h"): self._h,
                ("agents", "recurrent_state_c"): self._c,
            },
            batch_size=[1],
            device=self.device,
        )

        with torch.no_grad():
            td       = self.actor_core(td)
            features = td.get(("agents", "features"))        # (1, n_edges, actor_feat_dim)
            logits   = self.actor_head(features).squeeze(0)  # (n_edges, n_actions)
            self._h  = td.get(("agents", "recurrent_state_h_out"))
            self._c  = td.get(("agents", "recurrent_state_c_out"))

        ids_cpu_abs = np.empty(n_edges, dtype=np.float32)
        for i in range(n_edges):
            edge_logits = logits[i]
            if self.greedy:
                action = int(torch.argmax(edge_logits).item())
            else:
                probs  = torch.softmax(edge_logits, dim=-1)
                action = int(torch.multinomial(probs, 1).item())
            delta_cmd   = (action - (self.n_actions - 1) / 2.0) * self.scale_step
            ids_cpu_abs[i] = float(np.clip(
                ctx.ids_cpu[i] + delta_cmd,
                ctx.ids_cpu_min,
                ctx.ids_cpu_max[i],
            ))

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
    name examples: "random", "constant_0.5", "constant_1.5", "reactive", "tbsa", "lstm_rl", "ma_lstm_rl"
    """
    if name.startswith("constant_"):
        cpu_val = float(name.split("_", 1)[1])
        return ConstantPolicy(cpu_val)
    elif name == "no_ids":
        return NoIDSPolicy()
    elif name == "static_low":
        return ConstantPolicy(0.5)
    elif name == "static_balanced":
        return ConstantPolicy(2.0)
    elif name == "static_high":
        return ConstantPolicy(4.0)
    elif name == "random":
        return RandomPolicy()
    elif name in ("reactive", "autoscale_def"):
        return ReactivePolicy()
    elif name == "autoscale_app":
        return AppAutoscalePolicy()
    elif name in ("tbsa", "offline_optimal"):
        if tbsa_table_path is None:
            raise ValueError(f"tbsa_table_path must be provided for '{name}' policy")
        tbsa = TBSAPolicy(tbsa_table_path)
        return TBSAWrapperPolicy(tbsa)
    elif name == "lstm_rl":
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided for 'lstm_rl' policy")
        if obs_keys is None:
            raise ValueError("obs_keys must be provided for 'lstm_rl' policy")
        return LSTMRLPolicy(ckpt_path=ckpt_path, obs_keys=obs_keys, device=device, greedy=True)
    elif name == "ma_lstm_rl":
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided for 'ma_lstm_rl' policy")
        return MALSTMRLPolicy(ckpt_path=ckpt_path, device=device, greedy=False)
    else:
        raise ValueError(f"Unknown baseline policy name: {name!r}")
