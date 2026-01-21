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
        attack_df: pd.DataFrame,
        user_rate: float,
        cpu_ratio_to_ids: float,
    ) -> Dict[str, float]:
        total_attack = float(attack_df["flows_per_sec"].sum()) if not attack_df.empty else 0.0 #? Or use forward_packets_per_sec || flows_per_sec
        total_in = float(user_rate + total_attack) #! TODO: Change User rate to Packet

        speed = self.effective_speed_pkt_per_step(cpu_ratio_to_ids)
        coverage = float(min(1.0, speed / total_in)) if total_in > 0 else 0.0 #! TODO: Offloadinfg and add delay to the request

        attack_by_type = (
            attack_df.groupby("attack_type")["flows_per_sec"].sum().to_dict()
            if not attack_df.empty else {}
        )

        # attacks: expected dropped = coverage * TPR * rate
        attack_drop = 0.0
        for atk_type, lam in attack_by_type.items():
            tpr, _fpr = self.acc_tpr_fpr[str(atk_type)]
            attack_drop += coverage * tpr * float(lam)

        attack_pass = max(0.0, total_attack - attack_drop)

        # users: expected false drops = coverage * avg_fpr * user_rate
        avg_fpr = float(np.mean([v[1] for v in self.acc_tpr_fpr.values()])) if self.acc_tpr_fpr else 0.0
        user_drop = coverage * avg_fpr * float(user_rate)
        user_pass = max(0.0, float(user_rate) - user_drop)
        ids_cycles_available = self.effective_cycles_per_step(cpu_ratio_to_ids)
        ids_used_cycles = min(total_in * self.cycles_per_packet, ids_cycles_available)
        ids_cpu_util = min(1.0, ids_used_cycles / (ids_cycles_available+1e-6))
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

    All latencies are converted to CPU cycles during initialization.
    Runtime operates purely in cycles.

    Keys:
      (detector, base_resolution_h) -> detection_cycles

    Also holds:
      reid_cycles_per_object
    """

    def __init__(
        self,
        reid_latency_ms_per_object: float,
        configs: List[dict],
        cpu_cycle_per_ms: float,
        cpu_cores: int,
    ):
        # Convert ReID latency to cycles once
        self.reid_cycles_per_object: float = (
            float(reid_latency_ms_per_object)
            * float(cpu_cycle_per_ms)
            * int(cpu_cores)
        )

        # Detection cycles lookup
        self.det_cycles: Dict[Tuple[str, int], float] = {}

        for c in configs:
            det = str(c["detector"])
            h = int(c["base_resolution_h"])

            # Prefer explicit cycle annotation if provided
            if c.get("cycles", {}).get("detection", 0) > 0:
                cycles = float(c["cycles"]["detection"])
            else:
                # Convert latency → cycles once
                lat_ms = float(c["latency_ms"]["detection"])
                cycles = lat_ms * cpu_cycle_per_ms * cpu_cores

            self.det_cycles[(det, h)] = float(cycles)

        # Safety check
        if not self.det_cycles:
            raise ValueError("VideoPipeline initialized with no detection configs")

    # -------------------------------------------------
    # Cycle-based API (used everywhere in simulation)
    # -------------------------------------------------

    def detection_cycles(self, detector: str, base_resolution_h: int) -> float:
        """
        Detection cost in cycles for one frame, one camera.
        """
        return float(
            self.det_cycles.get((detector, int(base_resolution_h)), float("inf"))
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