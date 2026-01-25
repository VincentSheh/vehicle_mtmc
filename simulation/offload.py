from dataclasses import dataclass

@dataclass
class OffloadState:
    area_id: str

    # local objects that REMAIN after offloading
    n_req: int

    # objects received from other edges
    recv_q_obj: int

    va_avail_cycles_per_ms: float
    avail_cycles_aft_atk_per_ms: float
    uplink_available: float
    uplink_util: float
    track_cycles_per_obj: float
    latest_track_finish_ms: float
    idx: int

    def total_q(self) -> int:
        return self.local_q_obj + self.recv_q_obj


@dataclass
class OffloadDecision:
    src_idx: int
    dst_idx: int
    num_obj: int