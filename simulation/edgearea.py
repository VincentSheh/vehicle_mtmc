import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math


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

        self.cpu_to_ids_ratio = 0.9375
        # self.cpu_to_ids_ratio = 0.5

        self._last_action: Optional[Tuple[str, int]] = None

    def reset(self, seed: int | None = None):
        """
        Reset EdgeArea stochastic state.

        - Re-seeds internal RNG
        - Re-seeds all users and attackers independently
        - Resets per-episode dynamic state
        """

        # 1) Reset own RNG
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif not hasattr(self, "rng"):
            self.rng = np.random.default_rng()


        # 2) Reset users (independent seeds)
        for i, user in enumerate(self.users):
            user_seed = int(self.rng.integers(0, 2**32))
            user.reset(seed=user_seed)

        # 3) Reset attackers (independent seeds)
        for i, atk in enumerate(self.attackers):
            atk_seed = int(self.rng.integers(0, 2**32))
            atk.reset(seed=atk_seed)        
    
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
        atk_rows = [
            atk.load_at(t)
            for atk in self.attackers
            if not atk.load_at(t).empty
        ]

        if not atk_rows:
            return pd.DataFrame()

        return pd.concat(atk_rows, ignore_index=True)

    def aggregate_load_after_ids(self, t: int) -> Dict[str, float]:
        """
        Returns IDS output stats using the current cpu_to_ids_ratio.
        """
        attack_df = self._attack_df_at(t)
        user_rate = float(sum(u.num_requests_at(t) for u in self.users))

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
        od_plan: Tuple[str, int],
        n_req: int,
        va_avail_cycles_per_ms: float,
        avail_cycles_aft_atk_per_ms: float,
        uplink_available:float,
        uplink_util:float
    ) -> OffloadState:
        """
        Build offloading state for the CURRENT frame after:
        - IDS filtering
        - uplink decision
        - OD configuration selection
        - object detection has produced q_obj tracking tasks

        Assumes q_obj already reflects post-IDS surviving objects.
        """
        if proc_action is None:
            return OffloadState(
                area_id=self.area_id,
                n_req=0,
                recv_q_obj=0,
                va_avail_cycles_per_ms=float(va_avail_cycles_per_ms),
                avail_cycles_aft_atk_per_ms=avail_cycles_aft_atk_per_ms,
                track_cycles_per_obj=0.0,
                latest_track_finish_ms=0.0,
                uplink_available=uplink_available,
                uplink_util=uplink_util,
                idx=-1,
            )        

        det, proc_h = proc_action
        # --- Reserve compute for object detection (OD) ---
        od_cycles = self.estimate_detection_cycles_this_frame(
            det,
            proc_h,
            num_cameras=len(self.users),
        )

        # OD execution time in ms
        od_time_ms = od_cycles / max(1e-9, avail_cycles_aft_atk_per_ms)

        # Detection-safe deadline for OT
        latest_track_finish_ms = self.detection_safe_latest_track_finish_ms()

        # Remaining window for OT in this frame
        remaining_time_ms = max(0.0, latest_track_finish_ms - od_time_ms)

        return OffloadState(
            area_id=self.area_id,
            n_req=int(n_req),   # all local initially
            recv_q_obj=0,             # nothing received yet
            va_avail_cycles_per_ms=float(va_avail_cycles_per_ms),
            avail_cycles_aft_atk_per_ms=float(avail_cycles_aft_atk_per_ms),
            track_cycles_per_obj=self.tracking_cycles_per_object(),
            latest_track_finish_ms=remaining_time_ms,
            uplink_available=uplink_available,
            uplink_util=uplink_util,
            idx=-1,
        )
    def allocate_detectors(
        self,
        det_costs: dict,            # {det: cycles_per_req}
        det_quality: dict | None,   # {det: score}, optional
        N: int,
        C: float,                   # cycle budget in this slot (cycles)
        mu_cycles_per_ms: float,    # avail_cycles_aft_atk_per_ms
        D_Max: float,
        gamma: float = 0.0,         # latency penalty strength (0 = no penalty)
    ):
        """
        Feasible detector mixing under a hard per-slot cycle budget, plus QoE.

        Returns:
        plan: {det: n_req}
        feasible_all: bool
        dropped: int
        used_cycles: float
        mean_latency_ms: float
        qoe: float  (average QoE over the original N requests, dropped contribute 0)
        """
        if N <= 0:
            return {}, True, 0, 0.0, 0.0, 0.0

        # sort detectors by quality (desc), if missing then by cost (desc)
        dets = list(det_costs.keys())
        if det_quality is not None:
            dets.sort(key=lambda d: float(det_quality[d]), reverse=True)
            qmax = max(1e-12, max(float(det_quality[d]) for d in dets))
            qnorm = {d: float(det_quality[d]) / qmax for d in dets}  # 0..1
        else:
            dets.sort(key=lambda d: float(det_costs[d]), reverse=True)
            qnorm = {d: 1.0 for d in dets}

        # lightest (cheapest) detector as feasibility backstop
        det_light = min(dets, key=lambda d: float(det_costs[d]))
        cL = float(det_costs[det_light])

        # helper to compute QoE after plan is built
        def _qoe_from_plan(plan: dict, dropped: int, used_cycles: float) -> tuple[float, float]:
            served = int(sum(plan.values()))
            mean_latency = used_cycles / max(1e-9, mu_cycles_per_ms) if served > 0 else float("inf")

            if mean_latency <= D_Max:
                lat_pen = 1.0
            else:
                lat_pen = math.exp(-gamma * (mean_latency - D_Max)) if gamma > 0 else 0.0

            quality_sum = 0.0
            for det, n in plan.items():
                quality_sum += float(n) * float(qnorm.get(det, 0.0))

            # average over original N, dropped contributes 0
            qoe = (quality_sum / float(N)) * float(lat_pen) if N > 0 else 0.0
            return float(qoe), float(mean_latency)

        # if even all-light doesn't fit, serve as many as possible with lightest and drop rest
        if N * cL > C + 1e-9:
            served = int(C // cL) if cL > 0 else 0
            served = max(0, min(N, served))
            plan = {det_light: served} if served > 0 else {}
            used = served * cL
            dropped = N - served
            qoe, mean_lat = _qoe_from_plan(plan, dropped, used)
            return plan, False, dropped, float(used), float(mean_lat), float(qoe)

        plan = {d: 0 for d in dets}
        B = float(C)
        R = int(N)

        # allocate from best to worse, but never break feasibility for the remainder via lightest
        for det in dets:
            if det == det_light:
                continue
            if R <= 0:
                break

            ck = float(det_costs[det])
            if ck <= cL + 1e-12:
                continue

            numer = B - R * cL
            denom = ck - cL
            n_max = math.floor(numer / denom + 1e-12) if denom > 0 else 0
            n = max(0, min(R, int(n_max)))

            if n > 0:
                plan[det] += n
                B -= n * ck
                R -= n

        # assign remaining to lightest
        if R > 0:
            plan[det_light] += R
            B -= R * cL
            R = 0

        used_cycles = C - B
        assert sum(plan.values()) == N
        assert used_cycles <= C + 1e-6

        qoe, mean_lat = _qoe_from_plan(plan, 0, used_cycles)
        return plan, True, 0, float(used_cycles), float(mean_lat), float(qoe)


    def step_local(self, t: int):
        """
        Request-based, OD-only, one fixed upload resolution.
        If uplink bandwidth is not enough, drop user requests to fit uplink,
        then allocate detector mix under compute, QoE computed inside allocate_detectors().
        """

        # 0) total user requests arriving this step
        total_req_in = float(sum(u.num_requests_at(t) for u in self.users))
        local_num_request = int(np.floor(total_req_in))

        # 1) IDS filtering
        ids_out = self.aggregate_load_after_ids(t)
        user_pass_rate = float(ids_out.get("user_pass_rate", total_req_in))
        passed_req_pre_uplink = int(np.floor(max(0.0, user_pass_rate)))

        # 2) uplink after attacks (fixed upload resolution)
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

        uplink_total = self.budget.uplink / (1000.0 / self.slot_ms)

        UPLOAD_H = 416

        def uplink_mbps_for_h(h: int) -> float:
            w = h
            size_kb = 0.5 * (w * h * 3) / 1024.0
            return size_kb / 1024.0  # keep your existing convention

        per_req_uplink = uplink_mbps_for_h(UPLOAD_H)
        uplink_attack_used = attack_uplink_in * atk_pass_frac

        # remaining uplink after passed attacks
        uplink_available = max(0.0, uplink_total - uplink_attack_used)

        # drop user requests if uplink is not enough
        max_req_uplink = int(uplink_available // max(1e-12, per_req_uplink)) if per_req_uplink > 0 else 0
        served_req_uplink = min(passed_req_pre_uplink, max_req_uplink)
        dropped_uplink = max(0, passed_req_pre_uplink - served_req_uplink)

        uplink_user_used = served_req_uplink * per_req_uplink
        uplink_used = uplink_user_used + uplink_attack_used
        uplink_util = min(1.0, uplink_used / max(1e-9, uplink_total))

        # 3) VA compute supply (after attacks)
        total_cycles_per_ms = self.cpu_cycle_per_ms * self.budget.cpu
        avail_cycles_per_ms = total_cycles_per_ms * (1.0 - self.cpu_to_ids_ratio)

        attack_cycles_per_ms = 0.0
        for _, row in attack_df.iterrows():
            attacker_id = row.get("attacker_id")
            attacker = next(a for a in self.attackers if a.attacker_id == attacker_id)

            flows_i = float(row["flows_per_sec"])
            cpu_per_flow = attacker.cpu_usage_const
            attack_cycles_per_ms += flows_i * cpu_per_flow * atk_pass_frac / 1000.0

        avail_cycles_aft_atk_per_ms = max(0.0, avail_cycles_per_ms - attack_cycles_per_ms)

        # If nothing survives uplink or no compute, return outage-ish state
        if served_req_uplink <= 0 or avail_cycles_aft_atk_per_ms <= 1e-12:
            # state = self.build_offload_state(
            #     t=t,
            #     proc_action=None,
            #     q_obj=0,
            #     va_avail_cycles_per_ms=avail_cycles_per_ms,
            #     avail_cycles_aft_atk_per_ms=avail_cycles_aft_atk_per_ms,
            #     uplink_available=uplink_available,
            #     uplink_util=uplink_util,
            # )
            cache = {
                "ids_out": ids_out,
                "upload_h": UPLOAD_H,
                "local_num_request": local_num_request,
                "served_req": 0,
                "dropped_uplink": int(dropped_uplink),
                "dropped_compute": int(served_req_uplink),
                "dropped_total": int(dropped_uplink + served_req_uplink),
                "od_plan": {},
                "used_cycles": 0.0,
                "mean_latency_ms": float("inf"),
                "qoe": 0.0,
                "force_zero_qoe": True,
            }
            return cache

        # 4) allocate detector mix under compute budget and compute QoE inside allocate_detectors
        D_Max = float(self.constraints["D_Max"])
        gamma = float(self.constraints.get("Gamma", 0.0))

        # cycle budget implied by latency constraint
        C_budget = D_Max * avail_cycles_aft_atk_per_ms

        # derive detector list
        dets = []
        for a in self.pipeline.all_actions():
            det = a[0] if isinstance(a, tuple) else a
            dets.append(det)
        dets = sorted(set(dets))

        # cost per request for each detector
        det_costs = {det: float(self.pipeline.detection_cycles(det)) for det in dets}

        # optional quality
        det_quality = None
        if hasattr(self.pipeline, "det_quality"):
            det_quality = {det: float(self.pipeline.det_quality(det)) for det in dets}

        od_plan, feasible_all, dropped_compute, used_cycles, mean_latency_ms, qoe = self.allocate_detectors(
            det_costs=det_costs,
            det_quality=det_quality,
            N=int(served_req_uplink),
            C=float(C_budget),
            mu_cycles_per_ms=float(avail_cycles_aft_atk_per_ms),
            D_Max=float(D_Max),
            gamma=float(gamma),
        )

        served_compute = int(sum(od_plan.values()))
        assert served_compute + int(dropped_compute) == int(served_req_uplink)


        # 5) build offload state (reuse q_obj as served requests locally)
        # state = self.build_offload_state(
        #     t=t,
        #     od_plan=od_plan,
        #     q_obj=served_compute,
        #     va_avail_cycles_per_ms=avail_cycles_per_ms,
        #     avail_cycles_aft_atk_per_ms=avail_cycles_aft_atk_per_ms,
        #     uplink_available=uplink_available,
        #     uplink_util=uplink_util,
        # )

        cache = {
            "ids_out": ids_out,
            "local_num_request": local_num_request,
            "dropped_uplink": int(dropped_uplink),
            "od_plan": od_plan,  # {det: n_req}
            "served_req": int(served_compute),
            "dropped_compute": int(dropped_compute),
            "va_cpu_utilization": float(used_cycles),
            "uplink_util": uplink_util,
            "mean_latency_ms": float(mean_latency_ms),
            "qoe": float(qoe),
        }
        return cache