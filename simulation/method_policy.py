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
    ids_cpu: np.ndarray            # current IDS CPU allocation, shape (n_edges,)
    ids_cpu_min: float
    ids_cpu_max: np.ndarray        # shape (n_edges,)
    cpu_util: float                # scalar decision-window CPU utilisation
    decision_interval: int
    rng: np.random.Generator       # seeded RNG for stochastic policies


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
# Factory
# ---------------------------------------------------------------------------
def make_baseline_policy(
    name: str,
    tbsa_table_path: Optional[str] = None,
) -> BaselinePolicy:
    """
    name examples: "random", "constant_0.5", "constant_1.5", "reactive", "tbsa"
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
    else:
        raise ValueError(f"Unknown baseline policy name: {name!r}")
