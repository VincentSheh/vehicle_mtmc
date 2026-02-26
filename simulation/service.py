import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class IDS:
    """
    Cycle-based IDS model.

    Init:
      - processing_speed_pkt_per_ms (at full CPU) -> cycles_per_packet

    Runtime:
      - effective_speed_pkt_per_ms = ids_cycles_per_ms / cycles_per_packet
    """

    def __init__(
        self,
        cycles_per_packet: float,
        accuracy_by_type_fpr_fnr: Dict[str, Tuple[float, float]],
        cpu_cycle_per_ms: float,
        cpu_cores: int,
        slot_ms: int,
    ):
        if cycles_per_packet <= 0:
            raise ValueError("IDS processing speed must be > 0")

        self.total_cycles_per_ms_per_core: float = cpu_cycle_per_ms

        # cycles/packet at full CPU
        self.cycles_per_packet: float = cycles_per_packet
        self.slot_ms: int = slot_ms

        # store (TPR, FPR) per attack type
        self.acc_tpr_fpr: Dict[str, Tuple[float, float]] = {}
        for atk_type, (fpr, fnr) in accuracy_by_type_fpr_fnr.items():
            fpr = float(fpr)
            fnr = float(fnr)
            tpr = 1.0 - fnr
            self.acc_tpr_fpr[str(atk_type)] = (tpr, fpr)

    def effective_cycles_per_step(self, ids_cpu: float) -> float:
        return ids_cpu * self.total_cycles_per_ms_per_core * self.slot_ms

    def effective_speed_pkt_per_step(self, ids_cpu: float) -> float:
        ids_cycles = self.effective_cycles_per_step(ids_cpu)
        if self.cycles_per_packet <= 0:
            return 0.0
        return ids_cycles / self.cycles_per_packet

    def classify_workload(
        self,
        attack_dict: Dict[str, Any],
        user_in: float,
        atk_in: float,
        ids_cpu: float,
    ) -> Dict[str, float]:
        """
        All inputs/outputs are COUNTS per epoch (NOT fractions).
        - user_in: number of user inspection items this epoch
        - atk_in: number of attack inspection items this epoch

        Returns:
          - *_in_cnt, *_drop_cnt, *_pass_cnt are counts per epoch
          - *_in_rate, *_drop_rate, *_pass_rate are provided only for backward compatibility
            and are numerically equal to counts unless the caller converts them to per-second.
        """
        user_in = float(max(user_in, 0.0))
        atk_in = float(max(atk_in, 0.0))
        total_in = user_in + atk_in

        speed = float(self.effective_speed_pkt_per_step(ids_cpu))  # pkt/epoch
        coverage = float(min(1.0, speed / total_in)) if total_in > 0 else 1.0

        # --- attack drops (count) ---
        attack_drop = 0.0
        by_type = attack_dict.get("by_type", {})

        if by_type:
            sum_types = float(sum(by_type.values()))
            # scale by_type distribution to match atk_in
            scale = (atk_in / sum_types) if sum_types > 0 else 0.0
            for atk_type, lam in by_type.items():
                tpr, _fpr = self.acc_tpr_fpr[str(atk_type)]
                attack_drop += coverage * float(tpr) * float(lam) * scale
        else:
            if self.acc_tpr_fpr:
                avg_tpr = float(np.mean([v[0] for v in self.acc_tpr_fpr.values()]))
            else:
                avg_tpr = 0.0
            attack_drop = coverage * avg_tpr * atk_in

        attack_drop = round(float(np.clip(attack_drop, 0.0, atk_in)))
        attack_pass = int(atk_in - attack_drop)

        # --- user false drops (count) ---
        avg_fpr = float(np.mean([v[1] for v in self.acc_tpr_fpr.values()])) if self.acc_tpr_fpr else 0.0

        # effective drop probability for a benign user request
        p_drop = float(np.clip(coverage * avg_fpr, 0.0, 1.0))

        # user_drop as a probabilistic (binomial) realization, not a deterministic expectation
        n_user = int(round(float(user_in)))
        user_drop = int(np.random.binomial(n=n_user, p=p_drop))

        user_pass = n_user - user_drop

        # --- utilization ---
        ids_cycles_available = float(self.effective_cycles_per_step(ids_cpu))
        ids_used_cycles = min(total_in * float(self.cycles_per_packet), ids_cycles_available)
        ids_cpu_util = float(min(1.0, ids_used_cycles / (ids_cycles_available + 1e-6)))

        # avg_tpr for logging even if by_type is used
        if by_type:
            avg_tpr = float(np.mean([v[0] for v in self.acc_tpr_fpr.values()])) if self.acc_tpr_fpr else 0.0

        return {
            "coverage": coverage,

            # counts per epoch (authoritative)
            "user_in_cnt": user_in,
            "user_drop_cnt": user_drop,
            "user_pass_cnt": user_pass,
            "atk_in_cnt": atk_in,
            "atk_drop_cnt": attack_drop,
            "atk_pass_cnt": attack_pass,
            "inspect_in_cnt": total_in,

            # backward-compat names (these are NOT fractions)
            # "attack_in_rate": atk_in,
            # "attack_drop_rate": attack_drop,
            # "attack_pass_rate": attack_pass,
            # "user_drop_rate": user_drop,
            # "user_pass_rate": user_pass,

            "ids_cpu_util": ids_cpu_util,
            "tpr": float(avg_tpr),
            "fpr": float(avg_fpr),
        }
class VideoPipeline:
    """
    Global video analytics pipeline (cycle-based).

    New config supports per-detector accuracy by upload resolution:
      res_to_acc: [{base_resolution_h: int, map: float}, ...]

    Stored:
      - det_cycles[det] = cycles per request (independent of upload size, since resized)
      - det_quality[det] = "best" map (fallback, usually highest resolution)
      - res_to_acc[(det, h)] = map for that (detector, base_resolution_h)
      - supported_resolutions[det] = sorted list of base_resolution_h
    """

    def __init__(
        self,
        reid_latency_ms_per_object: float,
        configs: List[dict],
        cpu_cycle_per_ms: float,
        cpu_cores: int,
    ):
        self.reid_cycles_per_object: float = (
            float(reid_latency_ms_per_object) * float(cpu_cycle_per_ms) * int(cpu_cores)
        )

        self.det_cycles: Dict[str, float] = {}
        self.det_quality: Dict[str, float] = {}
        self.res_to_acc: Dict[Tuple[str, int], float] = {}
        self.supported_resolutions: Dict[str, List[int]] = {}

        for c in configs:
            det = str(c["detector"])

            # cycles per request (latency is same across res because input is resized)
            cycles = 0.0
            if c.get("cycles", {}).get("detection", 0) and float(c["cycles"]["detection"]) > 0:
                cycles = float(c["cycles"]["detection"])
            else:
                lat_ms = float(c["latency_ms"]["detection"])
                cycles = lat_ms * float(cpu_cycle_per_ms) * int(cpu_cores)
            self.det_cycles[det] = float(cycles)

            # per-resolution accuracy table
            res_rows = c.get("res_to_acc", None)
            if res_rows:
                hs: List[int] = []
                best_map = -1e9
                for r in res_rows:
                    h = int(r["base_resolution_h"])
                    m = float(r["map"])
                    self.res_to_acc[(det, h)] = m
                    hs.append(h)
                    if m > best_map:
                        best_map = m
                self.supported_resolutions[det] = sorted(set(hs))
                self.det_quality[det] = float(best_map)
            else:
                # backward-compatible fallback: single "map"
                m = float(c.get("map", 0.0))
                self.det_quality[det] = m
                self.supported_resolutions[det] = []

        if not self.det_cycles:
            raise ValueError("VideoPipeline initialized with no detection configs")


    def detection_cycles(self, detector: str) -> float:
        """
        Detection cost in cycles for one frame, one camera.
        """
        return float(
            self.det_cycles.get((detector), float("inf"))
        )

    def tracking_cycles_per_object(self) -> float:
        """
        ReID cost per object in cycles.
        """
        return self.reid_cycles_per_object

    def total_cycles(
        self,
        detector: str,
        base_resolution_h: int,
        num_objects: float,
    ) -> float:
        """
        Total VA cost in cycles:
          detection + tracking
        """
        return (
            self.detection_cycles(detector, base_resolution_h)
            + float(num_objects) * self.reid_cycles_per_object
        )

    def all_actions(self) -> List[Tuple[str, int]]:
        """
        All available (detector, resolution) configurations.
        """
        return list(self.det_cycles.keys())