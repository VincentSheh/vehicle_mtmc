from __future__ import annotations
from edgearea import EdgeArea

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd



@dataclass
class StepMetrics:
    """Per-step bookkeeping output."""
    t: int
    area_id: str
    qoe_mean: float
    qoe_min: float
    cpu_util: float
    uplink_util: float
    ids_coverage: float
    dropped_attack_rate: float
    
class Environment:
    def __init__(self, edge_areas: List[EdgeArea], t_max: int, seed: int):
        self.edge_areas = edge_areas
        self.t_max = int(t_max)
        self.seed = int(seed)
        self.t = 0
        self.history: List[StepMetrics] = []

        np.random.seed(self.seed)

    def reset(self):
        self.t = 0
        self.history.clear()

    def step(self) -> List[StepMetrics]:
        batch = []
        for ea in self.edge_areas:
            m = ea.step(self.t)
            batch.append(m)
            self.history.append(m)
        self.t += 1
        return batch

    def run(self) -> pd.DataFrame:
        while self.t < self.t_max:
            self.step()
        return pd.DataFrame([m.__dict__ for m in self.history])
