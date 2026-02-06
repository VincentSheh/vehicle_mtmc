import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class IDS:
    """
    Cycle-based IDS model.

    Init:
      - processing_speed_pkt_per_ms (at full CPU) -> cycles_per_packet

    Runtime:
      - ids_cycles_per_ms = cpu_ratio_to_ids * total_cycles_per_ms_full
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

        self.total_cycles_per_ms_full: float = float(cpu_cycle_per_ms) * int(cpu_cores)

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

    def effective_cycles_per_step(self, cpu_ratio_to_ids: float) -> float:
        cpu_ratio = float(np.clip(cpu_ratio_to_ids, 0.0, 1.0))
        return cpu_ratio * self.total_cycles_per_ms_full * self.slot_ms

    def effective_speed_pkt_per_step(self, cpu_ratio_to_ids: float) -> float:
        ids_cycles = self.effective_cycles_per_step(cpu_ratio_to_ids)
        if self.cycles_per_packet <= 0:
            return 0.0
        return ids_cycles / self.cycles_per_packet

    def classify_rates(
        self,
        attack_dict: Dict[str,Any],
        user_rate: float,
        cpu_ratio_to_ids: float,
    ) -> Dict[str, float]:
        total_attack = float(attack_dict["flows"])
        total_in = float(user_rate + total_attack)

        speed = self.effective_speed_pkt_per_step(cpu_ratio_to_ids)
        coverage = float(min(1.0, speed / total_in)) if total_in > 0 else 1.0

        # attacks: expected dropped = coverage * TPR * rate
        attack_drop = 0.0
        by_type = attack_dict.get("by_type", {})
        if by_type:
            for atk_type, lam in by_type.items():
                tpr, _fpr = self.acc_tpr_fpr[str(atk_type)]
                attack_drop += coverage * tpr * float(lam)
        else:
            # if you choose not to track by_type, you need a fallback
            # simplest: assume average TPR across all types
            if self.acc_tpr_fpr:
                avg_tpr = float(np.mean([v[0] for v in self.acc_tpr_fpr.values()]))
            else:
                avg_tpr = 0.0
            attack_drop = coverage * avg_tpr * total_attack

        attack_pass = max(0.0, total_attack - attack_drop)

        # users: expected false drops
        avg_fpr = float(np.mean([v[1] for v in self.acc_tpr_fpr.values()])) if self.acc_tpr_fpr else 0.0
        user_drop = coverage * avg_fpr * float(user_rate)
        user_pass = max(0.0, float(user_rate) - user_drop)

        ids_cycles_available = self.effective_cycles_per_step(cpu_ratio_to_ids)
        ids_used_cycles = min(total_in * self.cycles_per_packet, ids_cycles_available)
        ids_cpu_util = min(1.0, ids_used_cycles / (ids_cycles_available + 1e-6))

        return {
            "coverage": coverage,
            "attack_in_rate": total_attack,
            "attack_drop_rate": attack_drop,
            "attack_pass_rate": attack_pass,
            "user_drop_rate": user_drop,
            "user_pass_rate": user_pass,
            "ids_cpu_util": ids_cpu_util,
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