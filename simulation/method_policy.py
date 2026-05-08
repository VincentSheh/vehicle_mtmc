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
    "delay_workload": "qos_offloading",
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
    queue_ahead_norm: Optional[np.ndarray] = None  # (ids_cpu - ids_cpu_target) / max_delta, shape (n_edges,)
    cpu_utils: Optional[np.ndarray] = None # per-edge CPU utilisation, shape (n_edges,)


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
        if cpu_value == 0.0:
            self.min_cpu_override = 0.0

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
        utils = ctx.cpu_utils if ctx.cpu_utils is not None else np.full(n_edges, ctx.cpu_util)
        
        delta = np.zeros(n_edges, dtype=np.int64)
        for i in range(n_edges):
            if utils[i] >= self.high_threshold:
                delta[i] = 1
            elif utils[i] <= self.low_threshold:
                delta[i] = -1
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
        n_edges = len(ctx.ids_cpu)
        env     = ctx.env
        delta   = np.zeros(n_edges, dtype=np.int64)
        
        if env.history:
            records = env.history[-ctx.decision_interval * n_edges:]
            df = pd.DataFrame([m.__dict__ for m in records])
            if "va_cpu_utilization" in df.columns:
                for i, edge in enumerate(env.edge_areas):
                    g = df[df["area_id"] == edge.area_id]
                    if g.empty: continue
                    va_util = float(np.mean(g["va_cpu_utilization"].values))
                    if va_util >= self.high_threshold:
                        delta[i] = 1
                    elif va_util <= self.low_threshold:
                        delta[i] = -1
        return None, delta


# ---------------------------------------------------------------------------
# TBSA wrapper policy
# ---------------------------------------------------------------------------
class TBSAWrapperPolicy(BaselinePolicy):
    def __init__(self, tbsa: TBSAPolicy):
        self._tbsa = tbsa

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        import pandas as pd
        n_edges = len(ctx.ids_cpu)
        env = ctx.env
        ids_cpu_abs = np.empty(n_edges, dtype=np.float32)

        if env.history:
            records = env.history[-ctx.decision_interval * n_edges:]
            df = pd.DataFrame([m.__dict__ for m in records])
            for i, edge in enumerate(env.edge_areas):
                g = df[df["area_id"] == edge.area_id]
                if g.empty:
                    last_attack, last_req = 0.0, 0.0
                else:
                    last_attack = float(np.mean(g["attack_drop_rate"].values if "attack_drop_rate" in g.columns else [0.0]))
                    last_req    = float(np.mean(g["local_num_req"].values if "local_num_req" in g.columns else [0.0]))

                target_cpu = self._tbsa.select_ids_cpu(last_attack, last_req)
                ids_cpu_abs[i] = np.clip(target_cpu, ctx.ids_cpu_min, ctx.ids_cpu_max[i])
        else:
            # Initial step or empty history
            ids_cpu_abs = np.clip(np.full(n_edges, 0.5, dtype=np.float32), ctx.ids_cpu_min, ctx.ids_cpu_max)

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
# Multi-Agent LSTM RL policy (loads a train_ma_cur_lstm.py checkpoint)
# ---------------------------------------------------------------------------
class MALSTMRLPolicy(BaselinePolicy):
    """
    Wraps a checkpoint produced by train_ma_cur_lstm.py
    (SplitObsModule + LSTMModule + MergeModule + actor head).

    Builds per-edge observations directly from the environment history using the
    same feature layout used during training (9 features for n_actions=3, 12 for
    n_actions!=3), runs a single forward pass for all edges, and returns per-edge
    CPU targets.
    """
    _N_TEMPORAL = 2   # first N_TEMPORAL_PER_EDGE features fed through LSTM

    @staticmethod
    def _make_obs_keys(n_actions: int) -> List[str]:
        _ext = n_actions != 3
        return [
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

    def __init__(self, ckpt_path: str, device: str = "cpu", greedy: bool = False):
        import torch
        import torch.nn as nn
        from torchrl.modules import LSTMModule
        from tensordict.nn import TensorDictModule, TensorDictSequential
        from train_ma_cur_lstm import SplitObsModule, MergeModule

        self.device     = torch.device(device)
        self.greedy     = greedy
        self.scale_step = 0.5
        self.is_phase1  = "phase1" in str(ckpt_path).lower()

        state     = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        train_cfg = state["train_cfg"]

        hidden_dim      = int(train_cfg["model"]["hidden_dim"])
        self.n_actions  = int(train_cfg["model"]["n_actions"])
        self.hidden_dim = hidden_dim

        # Robustly remap policy weights to handle nested sequential/PA prefixes.
        # Checkpoint (from train_ma_cur_lstm.py) uses: PA(TDS(shared_core, actor_head))
        # where shared_core is another TDS(split, lstm, merge).
        # Sample checkpoint key: module.0.module.0.module.0.module.t_idx
        # Target actor_core key: module.0.module.t_idx
        policy_sd = state["policy"]
        core_sd = {}
        head_sd = {}
        for k, v in policy_sd.items():
            if k.startswith("module.0.module.0."):
                # Core keys: strip the PA and outer TDS prefix
                nk = k[len("module.0.module.0."):]
                core_sd[nk] = v
            elif k.startswith("module.0.module.1.module."):
                # Head keys: strip the PA, outer TDS, and TDM prefixes
                nk = k[len("module.0.module.1.module."):]
                head_sd[nk] = v
            elif k.startswith("module.1.module."): # fallback for single-nested PA
                nk = k[len("module.1.module."):]
                head_sd[nk] = v
            elif k.startswith("module.0."): # fallback for single-nested PA
                nk = k[len("module.0."):]
                core_sd[nk] = v

        _head_w = head_sd.get("weight")
        if _head_w is not None:
            # actor head shape: (n_actions, hidden_dim + n_static)
            obs_dim_from_weights = self._N_TEMPORAL + (int(_head_w.shape[1]) - hidden_dim)
            # find the obs_keys layout whose length matches
            _matched = False
            for _na in (3, 5, 7, 9):
                _cand = self._make_obs_keys(_na)
                if len(_cand) == obs_dim_from_weights:
                    self._obs_keys = _cand
                    self._obs_dim  = obs_dim_from_weights
                    _matched = True
                    break
            if not _matched:
                # unknown layout — fall back to n_actions from train_cfg
                self._obs_keys = self._make_obs_keys(self.n_actions)
                self._obs_dim  = len(self._obs_keys)
                print(f"[MALSTMRLPolicy] WARNING: actor head implies obs_dim={obs_dim_from_weights} "
                      f"but no known layout matches; falling back to n_actions={self.n_actions}")
            elif obs_dim_from_weights != len(self._make_obs_keys(self.n_actions)):
                print(f"[MALSTMRLPolicy] train_cfg n_actions={self.n_actions} implies obs_dim="
                      f"{len(self._make_obs_keys(self.n_actions))}, but actor head implies "
                      f"obs_dim={obs_dim_from_weights}; using weights")
        else:
            self._obs_keys = self._make_obs_keys(self.n_actions)
            self._obs_dim  = len(self._obs_keys)

        obsnorm    = state.get("obsnorm", {})
        _LOC_KEY   = "transforms.2.loc"
        _SCALE_KEY = "transforms.2.scale"
        env_cfg      = state.get("env_cfg", {})
        self.n_edges = max(1, len(env_cfg.get("edge_areas", [])))

        temporal_idx = list(range(self._N_TEMPORAL))
        static_idx   = list(range(self._N_TEMPORAL, self._obs_dim))
        n_static     = len(static_idx)

        split_module = TensorDictModule(
            SplitObsModule(temporal_idx, static_idx).to(self.device),
            in_keys=["observation"], out_keys=["temporal_obs", "static_obs"],
        )
        lstm = LSTMModule(
            input_size=self._N_TEMPORAL, hidden_size=hidden_dim,
            in_key="temporal_obs", out_key="lstm_out", device=str(self.device),
        )
        print(f"[MALSTMRLPolicy] n_actions={self.n_actions}, obs_dim={self._obs_dim}, obs_keys={self._obs_keys}")
        merge_module = TensorDictModule(
            MergeModule(),
            in_keys=["lstm_out", "static_obs"], out_keys=["features_merged"],
        )
        self.actor_core = TensorDictSequential(split_module, lstm, merge_module).to(self.device)
        self.actor_head = nn.Linear(hidden_dim + n_static, self.n_actions).to(self.device)

        if not core_sd:
            prefixes = sorted({".".join(k.split(".")[:2]) for k in policy_sd.keys()})
            print(f"[MALSTMRLPolicy] WARNING: no core keys matched (shared_core prefix 'module.0.'); "
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

        self.obs_loc = self.obs_scale = None
        if _LOC_KEY in obsnorm and _SCALE_KEY in obsnorm:
            self.obs_loc   = obsnorm[_LOC_KEY].detach().to(self.device)
            self.obs_scale = obsnorm[_SCALE_KEY].detach().to(self.device)
            _numel = self.obs_loc.numel()
            if _numel == 1:
                print(f"[MALSTMRLPolicy] ObsNorm loaded: scalar stats (reduce_dim=(0,1,2) training), broadcasting over obs_dim={self._obs_dim}")
            elif _numel == self._obs_dim:
                print(f"[MALSTMRLPolicy] ObsNorm loaded: per-feature stats, obs_dim={self._obs_dim}")
            else:
                print(f"[MALSTMRLPolicy] WARNING: obs_loc numel={_numel} != obs_dim={self._obs_dim} "
                      f"— stale checkpoint from different obs layout; normalisation will be skipped")
        else:
            print("[MALSTMRLPolicy] WARNING: no obsnorm found — running WITHOUT normalisation")

        self._h: Optional[object] = None
        self._c: Optional[object] = None

    def reset(self) -> None:
        self._h = None
        self._c = None

    def _build_ma_obs(self, ctx: ActContext) -> np.ndarray:
        """Replicates IndepTorchRLEnvWrapper._build_observation() from the eval-loop context."""
        import pandas as pd

        env      = ctx.env
        n_edges  = len(ctx.ids_cpu)
        area_ids = [e.area_id for e in env.edge_areas]
        obs      = np.zeros((n_edges, self._obs_dim), dtype=np.float32)

        if not env.history:
            return obs.reshape(-1)

        records = env.history[-ctx.decision_interval * n_edges:]
        df      = pd.DataFrame([m.__dict__ for m in records])

        # Base 5 features from history
        BASE_KEYS = ["local_num_req", "attack_in_rate", "ema_mom", "cpu_to_ids_ratio", "ids_cpu_utilization"]
        for i, area_id in enumerate(area_ids):
            g = df[df["area_id"] == area_id]
            if g.empty:
                continue
            for k in BASE_KEYS:
                if k not in g.columns or k not in self._obs_keys:
                    continue
                j    = self._obs_keys.index(k)
                vals = g[k].values
                if k == "cpu_to_ids_ratio":
                    obs[i, j] = float(vals[-1])
                elif k == "ema_mom":
                    vals_nz = vals[vals != 0.0]
                    obs[i, j] = float(np.mean(vals_nz)) if len(vals_nz) > 0 else 0.0
                else:
                    obs[i, j] = float(np.mean(vals))

        # Neighbor features
        if self.is_phase1:
            # Phase 1 models were trained in single-edge envs where neighbor features were always 0.
            # When evaluating in multi-edge, we zero them out to avoid "observation noise" 
            # that the policy hasn't seen before.
            obs[:, self._obs_keys.index("neighbor_ids_util")] = 0.0
            obs[:, self._obs_keys.index("neighbor_atk_rate")] = 0.0
            if "neighbor_delta" in self._obs_keys:
                obs[:, self._obs_keys.index("neighbor_delta")] = 0.0
        else:
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

            _has_delta = "neighbor_delta" in self._obs_keys
            for i, area_id in enumerate(area_ids):
                nbr_utils, nbr_deltas, nbr_atk = [], [], []
                for j in range(n_edges):
                    if j == i:
                        continue
                    nbr_utils.append(edge_ids_util[area_ids[j]])
                    nbr_atk.append(edge_atk_rate[area_ids[j]])
                    nbr_deltas.append(float(ctx.delta_in_flight_norm[j]))
                obs[i, self._obs_keys.index("neighbor_ids_util")] = float(np.mean(nbr_utils)) if nbr_utils else 0.0
                obs[i, self._obs_keys.index("neighbor_atk_rate")] = float(np.mean(nbr_atk))   if nbr_atk   else 0.0
                if _has_delta:
                    obs[i, self._obs_keys.index("neighbor_delta")] = float(np.mean(nbr_deltas)) if nbr_deltas else 0.0

        # prev_slo_vio
        _slo_idx = self._obs_keys.index("prev_slo_vio")
        for i, area_id in enumerate(area_ids):
            g = df[df["area_id"] == area_id]
            if g.empty:
                obs[i, _slo_idx] = 0.0
                continue
            last_qoe  = float(g["qoe_mean"].values[-1]) if "qoe_mean" in g.columns else 0.0
            threshold = float(env.edge_areas[i].slo_threshold)
            obs[i, _slo_idx] = 1.0 if last_qoe < threshold else 0.0

        # Transition state (only present for n_actions != 3)
        if "transition_ticks_norm" in self._obs_keys:
            for i in range(n_edges):
                obs[i, self._obs_keys.index("transition_ticks_norm")] = float(ctx.transition_ticks_norm[i])
                obs[i, self._obs_keys.index("delta_in_flight_norm")]  = float(ctx.delta_in_flight_norm[i])
                obs[i, self._obs_keys.index("queue_ahead_norm")]      = float(ctx.queue_ahead_norm[i]) if ctx.queue_ahead_norm is not None else 0.0

        return obs.reshape(-1).astype(np.float32)

    def _normalise(self, obs_flat: np.ndarray) -> np.ndarray:
        if self.obs_loc is None:
            return obs_flat
        import torch
        n_edges = len(obs_flat) // self._obs_dim
        x     = torch.from_numpy(obs_flat).to(self.device).view(n_edges, self._obs_dim)
        loc   = self.obs_loc.view(-1)    # (obs_dim,) or scalar — both broadcast over edges
        scale = self.obs_scale.view(-1)
        # numel==1: scalar stats from reduce_dim=(0,1,2) training — broadcast is correct.
        # numel!=1 and !=obs_dim: truly stale checkpoint from a different obs layout; skip.
        if loc.numel() != 1 and loc.numel() != self._obs_dim:
            return obs_flat
        return ((x - loc) / (scale + 1e-8)).view(-1).cpu().numpy()

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        import torch

        n_edges  = len(ctx.ids_cpu)
        obs_flat = self._build_ma_obs(ctx)
        obs_norm = self._normalise(obs_flat)

        obs_t = torch.from_numpy(obs_norm).float().to(self.device).view(n_edges, self._obs_dim)

        if self._h is None:
            self._h = torch.zeros(1, n_edges, self.hidden_dim, device=self.device)
            self._c = torch.zeros(1, n_edges, self.hidden_dim, device=self.device)

        # actor_core is TDS(split_module, lstm_module, merge_module)
        split_obs = self.actor_core.module[0].module   # SplitObsModule
        nn_lstm   = self.actor_core.module[1].lstm     # nn.LSTM (batch_first=True)

        temporal = obs_t[..., split_obs.t_idx]   # (n_edges, N_TEMPORAL)
        static   = obs_t[..., split_obs.s_idx]   # (n_edges, n_static)

        # batch_first=True → input (n_edges, seq=1, N_TEMPORAL)
        with torch.no_grad():
            lstm_out, (h_new, c_new) = nn_lstm(
                temporal.unsqueeze(1), (self._h, self._c)
            )
            self._h    = h_new
            self._c    = c_new
            lstm_out   = lstm_out.squeeze(1)                         # (n_edges, hidden_dim)
            features   = torch.cat([lstm_out, static], dim=-1)      # (n_edges, hidden_dim+n_static)
            logits     = self.actor_head(features)                   # (n_edges, n_actions)

        ids_cpu_abs = np.empty(n_edges, dtype=np.float32)
        for i in range(n_edges):
            edge_logits = logits[i]
            if self.greedy:
                action = int(torch.argmax(edge_logits).item())
            else:
                probs  = torch.softmax(edge_logits, dim=-1)
                action = int(torch.multinomial(probs, 1).item())
            delta_cmd      = (action - (self.n_actions - 1) / 2.0) * self.scale_step
            ids_cpu_abs[i] = float(np.clip(
                ctx.ids_cpu[i] + delta_cmd, ctx.ids_cpu_min, ctx.ids_cpu_max[i],
            ))

        return ids_cpu_abs, np.zeros(n_edges, dtype=np.int64)


# ---------------------------------------------------------------------------
# Multi-Agent MLP RL policy (loads a train_ma_cur_mlp.py checkpoint)
# ---------------------------------------------------------------------------
class MAMLPRLPolicy(BaselinePolicy):
    """
    Wraps a checkpoint produced by train_ma_cur_mlp.py (pure MLP actor).
    Obs layout mirrors IndepTorchRLEnvWrapper._build_observation(), including
    stateful last-attack-interval and attack-rate ring-buffer tracking.
    """
    _ATK_WINDOW = 10

    @staticmethod
    def _make_obs_keys(n_actions: int) -> List[str]:
        _ext = n_actions != 3
        return [
            "local_num_req",
            "attack_in_rate",
            "last_atk_intervals",
            "last_atk_intensity",
            "cpu_to_ids_ratio",
            "ids_cpu_utilization",
            "neighbor_ids_util",
            *( ["neighbor_delta"] if _ext else [] ),
            "neighbor_atk_rate",
            "prev_slo_vio",
            *( ["transition_ticks_norm", "delta_in_flight_norm", "queue_ahead_norm"] if _ext else [] ),
        ]

    def __init__(self, ckpt_path: str, device: str = "cpu", greedy: bool = False):
        import torch
        import torch.nn as nn

        self.device     = torch.device(device)
        self.greedy     = greedy
        self.scale_step = 0.5
        self.is_phase1  = "phase1" in str(ckpt_path).lower()

        state     = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        train_cfg = state["train_cfg"]

        hidden_dim     = int(train_cfg["model"]["hidden_dim"])
        self.n_actions = int(train_cfg["model"]["n_actions"])
        num_layers     = int(train_cfg["model"].get("num_layers", 2))

        # Checkpoint: ProbabilisticActor(TensorDictModule(nn.Sequential))
        # PA wraps TDM → "module." prefix; TDM may wrap another TDM at index 0
        # giving "module.0.module.<idx>.<param>" or "module.module.<idx>.<param>"
        policy_sd = state["policy"]
        mapped_sd = {}
        for k, v in policy_sd.items():
            if k.startswith("module.module."):
                mapped_sd[k[len("module.module."):]] = v
            elif k.startswith("module.0.module."):
                mapped_sd[k[len("module.0.module."):]] = v
            elif k.startswith("module."):
                mapped_sd[k[len("module."):]] = v

        # Determine obs_keys / obs_dim: prefer stored obs_keys, fall back to weight shape
        # Keys added after initial training (newest first); dropped in order until dims match
        _ADDED_KEYS = ["last_atk_intensity"]
        if state.get("obs_keys"):
            self._obs_keys = list(state["obs_keys"])
        else:
            canonical = self._make_obs_keys(self.n_actions)
            ckpt_obs_dim = next(
                (v.shape[1] for k, v in mapped_sd.items() if k == "0.weight"), len(canonical)
            )
            keys = list(canonical)
            for drop in _ADDED_KEYS:
                if len(keys) <= ckpt_obs_dim:
                    break
                if drop in keys:
                    keys.remove(drop)
            self._obs_keys = keys
        self._obs_dim = len(self._obs_keys)

        # Rebuild MLP matching build_mlp() from train_ma_cur_mlp.py
        layers, in_dim = [], self._obs_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.n_actions))
        self.actor = nn.Sequential(*layers).to(self.device)

        missing, unexpected = self.actor.load_state_dict(mapped_sd, strict=False)
        if missing:
            print(f"[MAMLPRLPolicy] WARNING: missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[MAMLPRLPolicy] WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        self.actor.eval()

        # ObsNorm: TransformedEnv(Compose(ObservationNorm)) → transforms.0.loc/scale
        obsnorm    = state.get("obsnorm", {})
        _LOC_KEY   = "transforms.0.loc"
        _SCALE_KEY = "transforms.0.scale"
        self.obs_loc = self.obs_scale = None
        if _LOC_KEY in obsnorm and _SCALE_KEY in obsnorm:
            self.obs_loc   = obsnorm[_LOC_KEY].detach().to(self.device)
            self.obs_scale = obsnorm[_SCALE_KEY].detach().to(self.device)
        else:
            loc_key   = next((k for k in obsnorm if k.endswith("loc")),   None)
            scale_key = next((k for k in obsnorm if k.endswith("scale")), None)
            if loc_key and scale_key:
                print(f"[MAMLPRLPolicy] WARNING: expected {_LOC_KEY!r} not found; "
                      f"falling back to {loc_key!r}/{scale_key!r}")
                self.obs_loc   = obsnorm[loc_key].detach().to(self.device)
                self.obs_scale = obsnorm[scale_key].detach().to(self.device)
            else:
                print(f"[MAMLPRLPolicy] WARNING: no obsnorm found — running WITHOUT normalisation "
                      f"(keys: {list(obsnorm.keys())})")

        print(f"[MAMLPRLPolicy] n_actions={self.n_actions}, obs_dim={self._obs_dim}, obs_keys={self._obs_keys}")

        # Stateful per-episode tracking (reset each episode)
        self._last_atk_intervals: Optional[np.ndarray] = None
        self._last_atk_intensity: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_atk_intervals = None
        self._last_atk_intensity = None

    def _build_ma_obs(self, ctx: ActContext) -> np.ndarray:
        import pandas as pd

        env      = ctx.env
        n_edges  = len(ctx.ids_cpu)
        area_ids = [e.area_id for e in env.edge_areas]
        obs      = np.zeros((n_edges, self._obs_dim), dtype=np.float32)

        if self._last_atk_intervals is None:
            self._last_atk_intervals = np.full(n_edges, 10, dtype=np.int32)
            self._last_atk_intensity = np.zeros(n_edges, dtype=np.float32)

        if not env.history:
            return obs.reshape(-1)

        records = env.history[-ctx.decision_interval * n_edges:]
        df      = pd.DataFrame([m.__dict__ for m in records])

        BASE_KEYS = ["local_num_req", "attack_in_rate", "cpu_to_ids_ratio", "ids_cpu_utilization"]
        for i, area_id in enumerate(area_ids):
            g = df[df["area_id"] == area_id]
            if g.empty:
                continue
            for k in BASE_KEYS:
                if k not in self._obs_keys or k not in g.columns:
                    continue
                vals = g[k].values
                j    = self._obs_keys.index(k)
                obs[i, j] = float(vals[-1]) if k == "cpu_to_ids_ratio" else float(np.mean(vals))

            atk_vals = g["attack_in_rate"].values if "attack_in_rate" in g.columns else np.zeros(1)
            mean_atk = float(np.mean(atk_vals))
            if mean_atk > 1e-6:
                self._last_atk_intervals[i] = 0
                self._last_atk_intensity[i] = max(float(self._last_atk_intensity[i]), mean_atk)
            else:
                self._last_atk_intervals[i] = min(int(self._last_atk_intervals[i]) + 1, 10)
                if self._last_atk_intervals[i] >= 10:
                    self._last_atk_intensity[i] = 0.0

            obs[i, self._obs_keys.index("last_atk_intervals")] = float(self._last_atk_intervals[i])
            if "last_atk_intensity" in self._obs_keys:
                obs[i, self._obs_keys.index("last_atk_intensity")] = float(self._last_atk_intensity[i])

        if n_edges > 1:
            if self.is_phase1:
                obs[:, self._obs_keys.index("neighbor_ids_util")] = 0.0
                obs[:, self._obs_keys.index("neighbor_atk_rate")] = 0.0
                if "neighbor_delta" in self._obs_keys:
                    obs[:, self._obs_keys.index("neighbor_delta")] = 0.0
            else:
                edge_ids_util: Dict[str, float] = {}
                edge_atk_rate: Dict[str, float] = {}
                for area_id in area_ids:
                    g = df[df["area_id"] == area_id]
                    if g.empty:
                        edge_ids_util[area_id] = 0.0
                        edge_atk_rate[area_id] = 0.0
                    else:
                        edge_ids_util[area_id] = float(np.clip(np.mean(g["ids_cpu_utilization"].values if "ids_cpu_utilization" in g.columns else [0.0]), 0.0, 1.0))
                        edge_atk_rate[area_id] = float(np.mean(g["attack_in_rate"].values if "attack_in_rate" in g.columns else [0.0]))

                _has_delta = "neighbor_delta" in self._obs_keys
                for i, area_id in enumerate(area_ids):
                    nbr_utils, nbr_deltas, nbr_atk = [], [], []
                    for j in range(n_edges):
                        if j == i:
                            continue
                        nbr_utils.append(edge_ids_util[area_ids[j]])
                        nbr_atk.append(edge_atk_rate[area_ids[j]])
                        nbr_deltas.append(float(ctx.delta_in_flight_norm[j]))
                    obs[i, self._obs_keys.index("neighbor_ids_util")] = float(np.mean(nbr_utils)) if nbr_utils else 0.0
                    obs[i, self._obs_keys.index("neighbor_atk_rate")] = float(np.mean(nbr_atk))   if nbr_atk   else 0.0
                    if _has_delta:
                        obs[i, self._obs_keys.index("neighbor_delta")] = float(np.mean(nbr_deltas)) if nbr_deltas else 0.0

        _slo_idx = self._obs_keys.index("prev_slo_vio")
        for i, area_id in enumerate(area_ids):
            g = df[df["area_id"] == area_id]
            if g.empty:
                continue
            last_qoe  = float(g["qoe_mean"].values[-1]) if "qoe_mean" in g.columns else 0.0
            threshold = float(env.edge_areas[i].slo_threshold)
            obs[i, _slo_idx] = 1.0 if last_qoe < threshold else 0.0

        if "transition_ticks_norm" in self._obs_keys:
            for i in range(n_edges):
                obs[i, self._obs_keys.index("transition_ticks_norm")] = float(ctx.transition_ticks_norm[i])
                obs[i, self._obs_keys.index("delta_in_flight_norm")]  = float(ctx.delta_in_flight_norm[i])
                obs[i, self._obs_keys.index("queue_ahead_norm")]      = float(ctx.queue_ahead_norm[i]) if ctx.queue_ahead_norm is not None else 0.0

        return obs.reshape(-1).astype(np.float32)

    def _normalise(self, obs_flat: np.ndarray) -> np.ndarray:
        if self.obs_loc is None:
            return obs_flat
        import torch
        n_edges = len(obs_flat) // self._obs_dim
        x     = torch.from_numpy(obs_flat).to(self.device).view(n_edges, self._obs_dim)
        loc   = self.obs_loc.view(-1)
        scale = self.obs_scale.view(-1)
        if loc.numel() != 1 and loc.numel() != self._obs_dim:
            return obs_flat
        return ((x - loc) / (scale + 1e-8)).view(-1).cpu().numpy()

    def act(self, ctx: ActContext) -> tuple[Optional[np.ndarray], np.ndarray]:
        import torch

        n_edges  = len(ctx.ids_cpu)
        obs_flat = self._build_ma_obs(ctx)
        obs_norm = self._normalise(obs_flat)

        obs_t = torch.from_numpy(obs_norm).float().to(self.device).view(n_edges, self._obs_dim)
        with torch.no_grad():
            logits = self.actor(obs_t)  # (n_edges, n_actions)

        ids_cpu_abs = np.empty(n_edges, dtype=np.float32)
        for i in range(n_edges):
            edge_logits = logits[i]
            if self.greedy:
                action = int(torch.argmax(edge_logits).item())
            else:
                probs  = torch.softmax(edge_logits, dim=-1)
                action = int(torch.multinomial(probs, 1).item())
            delta_cmd      = (action - (self.n_actions - 1) / 2.0) * self.scale_step
            ids_cpu_abs[i] = float(np.clip(
                ctx.ids_cpu[i] + delta_cmd, ctx.ids_cpu_min, ctx.ids_cpu_max[i],
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
    name examples: "random", "constant_0.5", "constant_2.0", "reactive", "tbsa", "lstm_rl", "ma_lstm_rl"
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
        return MALSTMRLPolicy(ckpt_path=ckpt_path, device=device, greedy=True)
    elif name == "ma_mlp_rl":
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided for 'ma_mlp_rl' policy")
        return MAMLPRLPolicy(ckpt_path=ckpt_path, device=device, greedy=True)
    else:
        raise ValueError(f"Unknown baseline policy name: {name!r}")
