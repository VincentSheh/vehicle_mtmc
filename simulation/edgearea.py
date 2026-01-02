import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


from service import IDS, VideoPipeline

from request import UserCamera, Attacker

from offload import (
    OffloadState,
    OffloadDecision,
)

@dataclass
class ResourceBudget:
    cpu: float
    mem: float
    uplink: float


@dataclass
class ResourceSplit:
    """Split an EdgeArea budget into application (VA) and defense (IDS)."""
    va: ResourceBudget
    ids: ResourceBudget

def cooperative_offload_ot(
    states: List[OffloadState],
    delay_ms: np.ndarray,
) -> Tuple[List[OffloadState], List[OffloadDecision]]:
    """
    Synchronized greedy OT offloading.

    states: list of OffloadState, one per edge area.
    delay_ms[i, j]: propagation delay from i to j in ms.

    Returns:
    - updated states (q_obj redistributed)
    - list of offload decisions
    """
    n = len(states)
    for i, st in enumerate(states):
        st.idx = i

    decisions: List[OffloadDecision] = []

    def finish_time_ms(st: OffloadState) -> float:
        # time to finish OT queue if processed locally at this edge
        if st.q_obj <= 0:
            return 0.0
        cycles = st.q_obj * st.track_cycles_per_obj
        return cycles / max(1e-9, st.avail_cycles_per_ms)

    # Build TOP by expected finish time (fastest first)
    TOP = list(range(n))
    TOP.sort(key=lambda i: finish_time_ms(states[i]))

    changed = True
    while changed:
        changed = False

        # process slower edges first (descending finish time)
        slow_order = sorted(TOP, key=lambda i: finish_time_ms(states[i]), reverse=True)

        for src in slow_order:
            src_st = states[src]
            if src_st.q_obj <= 0:
                continue

            # try offloading objects one by one from the tail of the slowest queue
            obj_idx = 0
            while obj_idx < src_st.q_obj:
                # local completion time for this object if it stays
                # approximate by completion time of the whole queue (coarse but consistent)
                local_C = finish_time_ms(src_st)

                moved = False
                # TO candidates are fastest first
                fast_order = sorted(TOP, key=lambda i: finish_time_ms(states[i]))
                for dst in fast_order:
                    if dst == src:
                        continue
                    dst_st = states[dst]
                    if dst_st.avail_cycles_per_ms <= 0:
                        continue
                    # remote completion time if one object is added to dst
                    remote_cycles = (dst_st.q_obj + 1) * dst_st.track_cycles_per_obj
                    remote_C = remote_cycles / max(1e-9, dst_st.avail_cycles_per_ms)
                    remote_C += float(delay_ms[src, dst])

                    # detection-safe constraint at receiver
                    if remote_C > dst_st.latest_track_finish_ms + 1e-9:
                        continue

                    # strict improvement
                    if remote_C + 1e-9 < local_C:
                        # apply offload
                        src_st.q_obj -= 1
                        dst_st.q_obj += 1

                        decisions.append(OffloadDecision(src_idx=src, dst_idx=dst, num_obj=1))

                        # update TOP ordering because finish times changed
                        TOP.sort(key=lambda i: finish_time_ms(states[i]))
                        changed = True
                        moved = True
                        break

                if not moved:
                    # cannot move this object beneficially
                    obj_idx += 1

    return states, decisions        

class EdgeArea:
    """
    EdgeArea emulation unit.

    Core flow per timestep:
      1) Split resources between IDS and VA.
      2) Apply IDS filtering to compute effective loads.
      3) Compute VA remaining compute budget (in CPU cycles/ms).
      4) Choose (detector, resolution_h) with max performance among feasible configs.
      5) Emit StepMetrics.

    Notes:
      - cpu_cycle_per_ms is per-core cycles per ms, e.g. 3.8e6 for 3.8 GHz.
      - budget.cpu is number of "CPU units" treated as parallel cores.
      - VA feasibility is checked in cycles/ms against VA required cycles/ms.
    """

    def __init__(
        self,
        area_id: str,
        cpu_cycle_per_ms: float,
        slot_ms: float,
        budget: ResourceBudget,
        constraints: Dict[str, float],
        ids: IDS,
        users: List[UserCamera],
        attackers: List[Attacker],
        pipeline: VideoPipeline,
    ):
        self.area_id = str(area_id)
        self.cpu_cycle_per_ms = float(cpu_cycle_per_ms)
        self.budget = budget
        self.slot_ms = slot_ms

        self.constraints = {
            "D_Max": float(constraints.get("D_Max", 1e9)),
            "MOTA_min": float(constraints.get("MOTA_min", 0.0)),
            "Gamma": float(constraints.get("Gamma", 0.0)),
        }

        self.ids = ids
        self.users = list(users)
        self.attackers = list(attackers)
        self.pipeline = pipeline

        self.cpu_to_ids_ratio = 0.0 #! TODO

        self._last_action: Optional[Tuple[str, int]] = None

    # --------------------------
    # Resource model (cycles)
    # --------------------------

    def split_resources(self) -> ResourceSplit:
        ids_cpu = self.budget.cpu * self.cpu_to_ids_ratio
        va_cpu = self.budget.cpu - ids_cpu

        return ResourceSplit(
            va=ResourceBudget(cpu=va_cpu, mem=self.budget.mem, uplink=self.budget.uplink),
            ids=ResourceBudget(cpu=ids_cpu, mem=self.budget.mem, uplink=self.budget.uplink),
        )

    def total_cycles_per_ms(self) -> float:
        # total parallel cycles per ms
        return float(self.cpu_cycle_per_ms * self.budget.cpu)

    def avail_cycles_per_ms(self) -> float:
        # remaining cycles for VA after reserving IDS CPU share
        return float(self.total_cycles_per_ms() * (1.0 - self.cpu_to_ids_ratio))

    def ids_cycles_per_ms(self) -> float:
        return float(self.total_cycles_per_ms() * self.cpu_to_ids_ratio)

    # --------------------------
    # Load aggregation
    # --------------------------

    def _attack_df_at(self, t: int) -> pd.DataFrame:
        atk_rows = [atk.load_at(t) for atk in self.attackers]
        assert atk_rows
        return pd.concat(atk_rows, ignore_index=True)

    def aggregate_load_after_ids(self, t: int) -> Dict[str, float]:
        """
        Returns IDS output stats using the current cpu_to_ids_ratio.
        """
        attack_df = self._attack_df_at(t)
        user_rate = float(len(self.users))

        return self.ids.classify_rates(
            attack_df=attack_df,
            user_rate=user_rate,
            cpu_ratio_to_ids=self.cpu_to_ids_ratio,
        )

    def estimate_detection_cycles_this_frame(
        self,
        detector: str,
        proc_h: int,
        num_cameras: int,
    ) -> float:
        return (
            self.pipeline.detection_cycles(detector, proc_h)
            * int(num_cameras)
        )

    def tracking_cycles_per_object(self) -> float:
        return self.pipeline.tracking_cycles_per_object()

    def detection_safe_latest_track_finish_ms(self) -> float:
        """
        Constraint: do not accept OT that would interfere with next OD.
        For synchronized OD, the simplest framing is:
          OT must finish before next frame OD begins.

        You can set slot_ms in policy, default 1000ms.
        You can also reserve a margin.
        """
        return max(0.0, self.slot_ms)

    def build_offload_state(
        self,
        t: int,
        proc_action: Tuple[str, int],
        q_obj: int,
        avail_cycles_per_ms: float,
    ) -> OffloadState:
        """
        Build offloading state for the CURRENT frame after:
        - IDS filtering
        - uplink decision
        - OD configuration selection
        - object detection has produced q_obj tracking tasks

        Assumes q_obj already reflects post-IDS surviving objects.
        """
        det, proc_h = proc_action

        # --- Reserve compute for object detection (OD) ---
        od_cycles = self.estimate_detection_cycles_this_frame(
            det,
            proc_h,
            num_cameras=len(self.users),
        )

        # OD execution time in ms
        od_time_ms = od_cycles / max(1e-9, avail_cycles_per_ms)

        # Detection-safe deadline for OT
        latest_track_finish_ms = self.detection_safe_latest_track_finish_ms()

        # Remaining window for OT in this frame
        remaining_time_ms = max(0.0, latest_track_finish_ms - od_time_ms)

        return OffloadState(
            area_id=self.area_id,
            local_q_obj=int(q_obj),   # all local initially
            recv_q_obj=0,             # nothing received yet
            avail_cycles_per_ms=float(avail_cycles_per_ms),
            track_cycles_per_obj=self.tracking_cycles_per_object(),
            latest_track_finish_ms=remaining_time_ms,
            idx=-1,
        )
                
    def step_local(
        self,
        t: int,
    ) -> Tuple[OffloadState, Dict[str, float]]:
        """
        Local decision step.
        No knowledge of other edges.
        Returns:
        - OffloadState (pre-offloading)
        - cache dict for QoE finalization
        """

        # 1) IDS filtering
        ids_out = self.aggregate_load_after_ids(t)

        total_users = float(len(self.users))
        user_pass_rate = float(ids_out.get("user_pass_rate", total_users))
        user_pass_frac = (
            user_pass_rate / max(1e-6, total_users) if total_users > 0 else 0.0
        )

        # 2) uplink after attacks
        attack_df = self._attack_df_at(t)
        atk_in = float(ids_out.get("attack_in_rate", 0.0))
        atk_pass = float(ids_out.get("attack_pass_rate", 0.0))
        atk_pass_frac = atk_pass / atk_in if atk_in > 0 else 0.0

        if "uplink_mbps" in attack_df.columns and not attack_df.empty:
            attack_uplink_in = float(attack_df["uplink_mbps"].sum())
        else:
            attack_uplink_in = (
                float(attack_df["forward_bytes_per_sec"].sum())
                if "forward_bytes_per_sec" in attack_df.columns
                else 0.0
            )

        uplink_available = max(
            0.0, self.budget.uplink / (1000.0/self.slot_ms) - attack_uplink_in * atk_pass_frac
        )

        # 3) choose upload resolution
        upload_candidates = sorted({h for (_, h) in self.pipeline.all_actions()})

        def uplink_mbps_for_h(h: int) -> float:
            ASPECT_W = 16
            ASPECT_H = 9
            w = h * 16 / 9
            size_kb = 0.5*(w * h * 3) / 1024.0
            return size_kb / 1024.0

        feasible_uploads = [
            h
            for h in upload_candidates
            if user_pass_frac * total_users * uplink_mbps_for_h(h)
            <= uplink_available + 1e-9
        ]
        upload_h = max(feasible_uploads) if feasible_uploads else min(upload_candidates)
        
        # 4) VA compute supply (after attacks)
        total_cycles_per_ms = self.cpu_cycle_per_ms * self.budget.cpu
        avail_cycles_per_ms = total_cycles_per_ms * (1.0 - self.cpu_to_ids_ratio)

        # IDS pass fraction for attacks
        atk_in = float(ids_out.get("attack_in_rate", 0.0))
        atk_pass = float(ids_out.get("attack_pass_rate", 0.0))
        atk_pass_frac = atk_pass / atk_in if atk_in > 0 else 0.0
        
        attack_cycles_per_ms = 0.0

        # IDS pass fraction
        atk_in = float(ids_out.get("attack_in_rate", 0.0))
        atk_pass = float(ids_out.get("attack_pass_rate", 0.0))
        atk_pass_frac = atk_pass / atk_in if atk_in > 0 else 0.0

        for _, row in attack_df.iterrows():
            attacker_id = row.get("attacker_id")

            # find the attacker object
            attacker = next(
                a for a in self.attackers if a.attacker_id == attacker_id
            )

            flows_i = float(row["flows_per_sec"])          # flows/sec for this attacker
            cpu_per_flow = attacker.cpu_usage_const         # cycles/flow

            attack_cycles_per_ms += (
                flows_i * cpu_per_flow * atk_pass_frac / 1000.0
            )

        avail_cycles_per_ms = max(0.0, avail_cycles_per_ms - attack_cycles_per_ms)

        # 5) choose VA configuration locally
        D_Max = self.constraints["D_Max"]
        MOTA_min = self.constraints["MOTA_min"]
        gamma = self.constraints["Gamma"]

        best_q = -np.inf
        best_action = None
        best_mean_mota = 0.0
        best_od_cycles = 0.0

        for det, proc_h in self.pipeline.all_actions():
            if proc_h > upload_h:
                continue

            demand_cycles = 0.0
            motas = []

            for u in self.users:
                nobj = u.get_num_objects(t - 1, det, proc_h)

                # total VA cost for this user in cycles
                user_cycles = self.pipeline.total_cycles(
                    detector=det,
                    base_resolution_h=proc_h,
                    num_objects=nobj,
                )

                demand_cycles += user_cycles
                motas.append(u.get_mota(det, proc_h))

            demand_cycles *= user_pass_frac
            mean_latency = (
                demand_cycles / max(1e-9, avail_cycles_per_ms)
                if avail_cycles_per_ms > 0
                else float("inf")
            )
            mean_mota = float(np.mean(motas)) if motas else 0.0

            # q = (mean_mota - MOTA_min) * np.exp(-gamma * (mean_latency - D_Max))
            q = mean_mota if mean_latency <= D_Max else 0
            if q > best_q:
                best_q = q
                best_action = (det, proc_h)
                od_cycles = self.pipeline.detection_cycles(det, proc_h) * len(self.users)
                best_od_cycles = od_cycles
                best_mean_mota = mean_mota

        if best_action is None:
            best_action = min(
                self.pipeline.all_actions(),
                key=lambda a: self.pipeline.detection_cycles(a[0], a[1]),
            )

        self._last_action = best_action

        # 6) object queue for OT
        q_obj = sum(
            int(u.get_num_objects(t - 1, best_action[0], best_action[1]))
            for u in self.users
        )
        q_obj = int(np.floor(q_obj * user_pass_frac))

        state = self.build_offload_state(
            t=t,
            proc_action=best_action,
            q_obj=q_obj,
            avail_cycles_per_ms=avail_cycles_per_ms,
        )

        cache = {
            "best_od_cycles": best_od_cycles,
            "best_mean_mota": best_mean_mota,
            "ids_out": ids_out,
        }

        return state, cache