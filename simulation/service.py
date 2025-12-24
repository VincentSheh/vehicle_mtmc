import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class IDS:
    """
    Global processing speed but per-area accuracy.

    YAML uses: [FPR, FNR]
    Internally we convert to:
      - TPR = 1 - FNR
      - FPR = FPR
    """
    def __init__(self, processing_speed: float, accuracy_by_type_fpr_fnr: Dict[str, Tuple[float, float]]):
        self.processing_speed = float(processing_speed)

        self.acc_tpr_fpr: Dict[str, Tuple[float, float]] = {}
        for atk_type, (fpr, fnr) in accuracy_by_type_fpr_fnr.items():
            fpr = float(fpr)
            fnr = float(fnr)
            tpr = 1.0 - fnr
            self.acc_tpr_fpr[str(atk_type)] = (tpr, fpr)

    def effective_speed(self, cpu_ratio_to_ids: float) -> float:
        # simplest model: IDS throughput scales linearly with CPU share
        return self.processing_speed * float(max(0.0, min(1.0, cpu_ratio_to_ids)))

    def classify_rates(
        self,
        attack_df: pd.DataFrame,
        user_rate: float,
        cpu_ratio_to_ids: float,
    ) -> Dict[str, float]:
        """
        Return aggregate rates after filtering.
        """
        total_attack = float(attack_df["lambda_req"].sum()) if not attack_df.empty else 0.0
        total_in = float(user_rate + total_attack)

        speed = self.effective_speed(cpu_ratio_to_ids)
        coverage = float(min(1.0, speed / total_in)) if total_in > 0 else 0.0

        # attacks: drop = coverage * TPR per type
        attack_by_type = (
            attack_df.groupby("attack_type")["lambda_req"].sum().to_dict()
            if not attack_df.empty else {}
        )

        attack_drop = 0.0
        for atk_type, lam in attack_by_type.items():
            tpr, _fpr = self.acc_tpr_fpr.get(str(atk_type), (0.0, 0.0))
            attack_drop += coverage * tpr * float(lam)

        attack_pass = max(0.0, total_attack - attack_drop)

        # user false drops: coverage * avg_fpr * user_rate
        if self.acc_tpr_fpr:
            avg_fpr = float(np.mean([v[1] for v in self.acc_tpr_fpr.values()]))
        else:
            avg_fpr = 0.0

        user_drop = coverage * avg_fpr * float(user_rate)
        user_pass = max(0.0, float(user_rate) - user_drop)

        return {
            "coverage": coverage,
            "attack_in_rate": total_attack,
            "attack_drop_rate": attack_drop,
            "attack_pass_rate": attack_pass,
            "user_drop_rate": user_drop,
            "user_pass_rate": user_pass,
        }


class VideoPipeline:
    """
    Global pipeline configs:
      key = (detector, base_resolution_h)
      value = detection_latency_ms

    Also holds global reid_latency (ms per object).
    """
    def __init__(self, reid_latency_ms_per_object: float, configs: List[dict]):
        self.reid_latency = float(reid_latency_ms_per_object)
        self.det_latency: Dict[Tuple[str, int], float] = {}

        for c in configs:
            det = str(c["detector"])
            h = int(c["base_resolution_h"])
            lat = float(c["latency_ms"]["detection"])
            self.det_latency[(det, h)] = lat

    def detection_ms(self, detector: str, base_resolution_h: int) -> float:
        return float(self.det_latency.get((detector, int(base_resolution_h)), 1e9))

    def total_latency_ms(self, detector: str, base_resolution_h: int, num_objects: float) -> float:
        return self.detection_ms(detector, base_resolution_h) + self.reid_latency * float(num_objects)

    def all_actions(self) -> List[Tuple[str, int]]:
        return list(self.det_latency.keys())