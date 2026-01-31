import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math


from service import IDS, VideoPipeline

from request import User, Attacker

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
        users: List[User],
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
        idx = int(self.rng.integers(0, len(self.attackers)))
        self.cur_attacker = [self.attackers[idx]]
    
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
            for atk in self.cur_attacker
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
        num_cameras: int,
    ) -> float:
        return (
            self.pipeline.detection_cycles(detector)
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

    def select_resolution(
        self,
        passed_req_pre_uplink: int,
        uplink_available: float,        # megabits available for users in this slot
        uplink_attack_used: float,      # already-consumed uplink by attacks (Mb)
        uplink_total_mb: float,         # total uplink budget for utilization calc (Mb)
        upload_hs=(223, 320, 416),
    ) -> Tuple[Dict[int, int], int, int, float, Dict[int, float]]:
        """
        Throughput-first greedy with quality refinement.

        Stage 1: maximize served requests using lowest resolution
        Stage 2: upgrade some requests to higher resolutions using leftover uplink

        Returns:
        upload_plan: {h: n_req}
        served_req_uplink: int
        dropped_uplink: int
        uplink_util: float
        per_req_uplink_by_h: {h: Mb_per_req}
        """

        # --- uplink cost per request in MEGABITS ---
        def uplink_mbps_for_h(h: int) -> float:
            w = h
            size_bits = 0.1 * (w * h * 3) * 8          # RGB bits
            return size_bits / (1024.0 * 1024.0) # Mb

        hs = sorted(set(int(h) for h in upload_hs))

        # trivial cases
        if passed_req_pre_uplink <= 0 or uplink_available <= 0 or not hs:
            upload_plan = {h: 0 for h in hs}
            uplink_util = min(1.0, uplink_attack_used / max(1e-9, uplink_total_mb))
            return (
                upload_plan,
                0,
                int(max(0, passed_req_pre_uplink)),
                float(uplink_util),
                {h: uplink_mbps_for_h(h) for h in hs},
            )

        # per-request uplink cost
        per_req = {h: float(uplink_mbps_for_h(h)) for h in hs}

        # --------------------------------------------------
        # Stage 1: Throughput-first (lowest resolution)
        # --------------------------------------------------
        h_min = min(hs, key=lambda h: per_req[h])
        c_min = per_req[h_min]

        max_served = int(uplink_available // max(1e-12, c_min))
        served_req = min(passed_req_pre_uplink, max_served)

        upload_plan: Dict[int, int] = {h: 0 for h in hs}
        upload_plan[h_min] = served_req

        remaining_uplink = uplink_available - served_req * c_min

        # --------------------------------------------------
        # Stage 2: Quality refinement (upgrade requests)
        # --------------------------------------------------
        for h in sorted(hs, reverse=True):
            if h == h_min:
                continue

            delta = per_req[h] - c_min  # extra cost to upgrade
            if delta <= 0 or remaining_uplink <= 0:
                continue

            can_upgrade = int(remaining_uplink // delta)
            take = min(upload_plan[h_min], can_upgrade)

            if take > 0:
                upload_plan[h_min] -= take
                upload_plan[h] += take
                remaining_uplink -= take * delta

        # --------------------------------------------------
        # Final accounting
        # --------------------------------------------------
        served_req_uplink = int(sum(upload_plan.values()))
        dropped_uplink = int(passed_req_pre_uplink - served_req_uplink)

        uplink_user_used = sum(upload_plan[h] * per_req[h] for h in hs)
        uplink_used = float(uplink_user_used + uplink_attack_used)
        uplink_util = min(1.0, uplink_used / max(1e-9, uplink_total_mb))

        return (
            upload_plan,
            served_req_uplink,
            dropped_uplink,
            float(uplink_util),
            per_req,
        )

    def allocate_detectors(
        self,
        det_costs: dict,            # {det: cycles_per_req}
        det_quality: dict | None,   # {det: score}, optional
        N: int,
        mu_cycles_per_ms: float,    # avail_cycles_aft_atk_per_ms (cycles per ms)
        gamma: float = 0.0,         # latency penalty strength (0 = hard cutoff)
    ):
        """
        Feasible detector mixing where feasibility is defined by a latency constraint
        computed from the current available CPU cycles.

        Latency model (per slot):
        total_cycles = sum_det n_det * c_det
        mean_latency_ms = total_cycles / mu_cycles_per_ms

        Constraint:
        mean_latency_ms <= D_Max
        Equivalent cycle budget:
        total_cycles <= D_Max * mu_cycles_per_ms

        Returns:
        plan: {det: n_req}
        feasible_all: bool
        dropped: int
        used_cycles: float
        mean_latency_ms: float
        qoe: float  (average QoE over original N, dropped contribute 0)
        """
        import math


        # no requests means no violation and full satisfaction
        if N == 0:
            return {}, True, 0, 0.0, 0.0, 1.0

        # invalid compute or latency bound means cannot serve any positive demand
        if N < 0:
            N = 0
        if mu_cycles_per_ms <= 1e-12 or self.slot_ms <= 0:
            return {}, False, int(N), 0.0, float("inf"), 0.0

        # cycle budget implied by current compute and latency bound
        C_budget = float(self.slot_ms) * float(mu_cycles_per_ms)

        # sort detectors by quality (desc), if missing then by cost (desc)
        dets = list(det_costs.keys())
        if not dets:
            return {}, False, int(N), 0.0, float("inf"), 0.0

        dets.sort(key=lambda d: float(det_quality[d]), reverse=True)
        qmax = max(1e-12, max(float(det_quality[d]) for d in dets))
        qnorm = {d: float(det_quality[d]) / qmax for d in dets}  # 0..1

        # cheapest detector as backstop
        det_light = min(dets, key=lambda d: float(det_costs[d]))
        cL = float(det_costs[det_light])

        # helper QoE from plan
        def _qoe_from_plan(plan: dict, dropped: int, used_cycles: float):
            served = int(sum(plan.values()))
            mean_latency = used_cycles / max(1e-9, mu_cycles_per_ms) if served > 0 else float("inf")

            if mean_latency >= self.slot_ms:
                lat_pen = 1.0
            else:
                lat_pen = math.exp(-gamma * (mean_latency - self.slot_ms)) if gamma > 0 else 0.0
            lat_pen = 1.0
            quality_sum = 0.0
            for det, n in plan.items():
                quality_sum += float(n) * float(qnorm.get(det, 0.0))

            qoe = (quality_sum / float(N)) * float(lat_pen) if N > 0 else 0.0
            return float(qoe), float(mean_latency)

        # if even all-light violates latency bound, serve what we can with lightest
        if N * cL > C_budget + 1e-9:
            served = int(C_budget // cL) if cL > 0 else 0
            served = max(0, min(N, served))
            plan = {det_light: served} if served > 0 else {}
            used = served * cL
            dropped = N - served
            qoe, mean_lat = _qoe_from_plan(plan, dropped, used)
            return plan, False, int(dropped), float(used), float(mean_lat), float(qoe)

        # otherwise, all N can meet latency bound, now maximize quality with feasibility backstop
        plan = {d: 0 for d in dets}
        B = float(C_budget)
        R = int(N)

        for det in dets:
            if det == det_light:
                continue
            if R <= 0:
                break

            ck = float(det_costs[det])
            if ck <= cL + 1e-12:
                continue

            # keep enough budget to run remaining requests using the lightest detector
            # n <= (B - R*cL) / (ck - cL)
            numer = B - R * cL
            denom = ck - cL
            n_max = math.floor(numer / denom + 1e-12) if denom > 0 else 0
            n = max(0, min(R, int(n_max)))

            if n > 0:
                plan[det] += n
                B -= n * ck
                R -= n

        if R > 0:
            plan[det_light] += R
            B -= R * cL
            R = 0

        used_cycles = C_budget - B
        assert sum(plan.values()) == N
        assert used_cycles <= C_budget + 1e-6

        qoe, mean_lat = _qoe_from_plan(plan, 0, used_cycles)
        return plan, True, 0, float(used_cycles), float(mean_lat), float(qoe)

    def match_detectors_to_resolutions(
        self,
        upload_plan: Dict[int, int],
        det_plan: Dict[str, int],
    ) -> Tuple[float, Dict[Tuple[str, int], int]]:
        det_res_map = self.pipeline.res_to_acc  # {(det,h): map}

        up = {int(h): int(n) for h, n in upload_plan.items() if int(n) > 0}
        dp = {str(d): int(n) for d, n in det_plan.items() if int(n) > 0}
        if not up or not dp:
            return 0.0, {}

        sum_up = int(sum(up.values()))
        sum_dp = int(sum(dp.values()))

        # If mismatch, shrink upload_plan by reducing the lowest resolution first.
        # This is safe because it only drops some uploaded requests, it does not fabricate uploads.
        if sum_up != sum_dp:
            if sum_up < sum_dp:
                raise ValueError(
                    f"Mismatch where uploads < detections: sum(upload_plan)={sum_up} < sum(det_plan)={sum_dp}. "
                    "This means detector allocation exceeds uploaded requests. Fix upstream (served_req_uplink vs served_compute)."
                )

            # sum_up > sum_dp: drop (sum_up - sum_dp) uploads from lowest resolution bins
            excess = sum_up - sum_dp
            for h in sorted(up.keys()):  # lowest resolution first
                if excess <= 0:
                    break
                drop = min(up[h], excess)
                up[h] -= drop
                excess -= drop
                if up[h] == 0:
                    del up[h]

            sum_up = int(sum(up.values()))
            if sum_up != sum_dp:
                raise RuntimeError(f"Failed to reconcile counts after trimming uploads: sum_up={sum_up}, sum_dp={sum_dp}")

        # monotone matching low-res -> low-quality detector (quality proxy = detector max mAP)
        dets: List[str] = list(dp.keys())
        det_proxy: Dict[str, float] = {}
        for d in dets:
            vals = [float(v) for (dd, _h), v in det_res_map.items() if dd == d]
            det_proxy[d] = max(vals) if vals else 0.0

        res_list = sorted(up.keys())  # low->high
        det_list = sorted(dets, key=lambda d: (det_proxy.get(d, 0.0), d))  # low->high

        assign: Dict[Tuple[str, int], int] = {}
        rem_res = {h: up[h] for h in res_list}
        rem_det = {d: dp[d] for d in det_list}

        i = 0
        j = 0
        while i < len(res_list) and j < len(det_list):
            h = res_list[i]
            d = det_list[j]

            take = min(rem_res[h], rem_det[d])
            if take > 0:
                assign[(d, h)] = assign.get((d, h), 0) + int(take)
                rem_res[h] -= int(take)
                rem_det[d] -= int(take)

            if rem_res[h] == 0:
                i += 1
            if rem_det[d] == 0:
                j += 1

        if sum(rem_res.values()) != 0 or sum(rem_det.values()) != 0:
            raise RuntimeError(
                f"Post-match leftovers: uploads={sum(rem_res.values())}, dets={sum(rem_det.values())}. "
                "Counts should match after trimming."
            )

        total = int(sum(assign.values()))
        if total <= 0:
            return 0.0, {}


        missing = []
        for (d, h), n in assign.items():
            if (d, int(h)) not in det_res_map:
                missing.append((d, int(h)))

        if missing:
            raise KeyError(f"Missing (det,h) in pipeline.res_to_acc: {missing[:10]}")

        # global normalization constant over ALL available (det,h) pairs
        qmax = max(1e-12, max(float(v) for v in det_res_map.values()))

        total = sum(assign.values())
        qoe_sum = 0.0
        for (d, h), n in assign.items():
            q = float(det_res_map[(d, int(h))]) / qmax
            qoe_sum += float(n) * q

        qoe = qoe_sum / float(total) if total > 0 else 0.0
        return float(qoe), assign

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
        passed_req_pre_uplink = int(np.ceil(max(0.0, user_pass_rate)))

        # 2) uplink and cycles after attacks (fixed upload resolution)
        attack_df = self._attack_df_at(t)
        atk_in = float(ids_out.get("attack_in_rate", 0.0))
        atk_pass = float(ids_out.get("attack_pass_rate", 0.0))
        atk_pass_frac = atk_pass / atk_in if atk_in > 0 else 0.0
        
        attack_uplink_in = 0.0
        attack_cycles_per_ms = 0.0
        for _, row in attack_df.iterrows():
            attacker_id = row.get("attacker_id")
            attacker = next(a for a in self.cur_attacker if a.attacker_id == attacker_id)

            flows_i = float(row["flows_per_sec"])
            bw_per_flow = attacker.bw_per_flow
            attack_uplink_in += flows_i * bw_per_flow * atk_pass_frac
            cpu_per_flow = attacker.cycle_per_flow
            attack_cycles_per_ms += flows_i * cpu_per_flow * atk_pass_frac / 1000.0            

        uplink_total_mb = self.budget.uplink / (1000.0 / self.slot_ms)

        uplink_attack_used = attack_uplink_in

        uplink_available = max(0.0, uplink_total_mb - uplink_attack_used)
        
        # 3) VA compute supply (after attacks)
        total_cycles_per_ms = self.cpu_cycle_per_ms * self.budget.cpu
        avail_cycles_per_ms = total_cycles_per_ms * (1.0 - self.cpu_to_ids_ratio)

        avail_cycles_aft_atk_per_ms = max(0.0, avail_cycles_per_ms - attack_cycles_per_ms)
        # If nothing survives uplink or no compute, return outage-ish state
        if avail_cycles_aft_atk_per_ms <= 1e-12 or uplink_available <=1e-12:
            cache = {
                "ids_out": ids_out,
                "local_num_request": local_num_request,
                "local_gt_num_request": local_num_request,
                "dropped_uplink": int(total_req_in),
                "od_plan": {},
                "served_req": 0,
                "dropped_compute": int(total_req_in),
                "va_cpu_utilization": (total_cycles_per_ms - avail_cycles_aft_atk_per_ms) / total_cycles_per_ms,
                "uplink_util": 1,
                "mean_latency_ms": float("inf"),
                "qoe": 0.0,

            }
            return cache

        upload_plan, served_req_uplink, dropped_uplink, uplink_util, per_req_by_h = self.select_resolution(
            passed_req_pre_uplink=int(passed_req_pre_uplink),
            uplink_available=float(uplink_available),
            uplink_attack_used=float(uplink_attack_used),
            uplink_total_mb=float(uplink_total_mb),
            upload_hs=(224, 320, 412),
        )

        # 4) allocate detector mix under compute budget and compute QoE inside allocate_detectors
        D_Max = float(self.constraints["D_Max"])
        gamma = float(self.constraints.get("Gamma", 0.0))

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
            det_quality = {det: float(self.pipeline.det_quality[det]) for det in dets}

        od_plan, feasible_all, dropped_compute, used_cycles, mean_latency_ms, qoe = self.allocate_detectors(
            det_costs=det_costs,
            det_quality=det_quality,
            N=int(served_req_uplink),
            mu_cycles_per_ms=float(avail_cycles_aft_atk_per_ms),
            gamma=float(gamma),
        )
        qoe, od_and_res_plan = self.match_detectors_to_resolutions(upload_plan, od_plan)
        
        va_cpu_utilization = used_cycles / (avail_cycles_per_ms * self.slot_ms)

        va_cpu_utilization = used_cycles / (avail_cycles_per_ms * self.slot_ms)

        served_compute = int(sum(od_plan.values()))
        assert served_compute + int(dropped_compute) == int(served_req_uplink)

        if total_req_in <= 0:
            qoe = 1.0
        else:
            qoe =  qoe * (served_compute / total_req_in)
            
        cache = {
            "ids_out": ids_out,
            "local_num_request": local_num_request,
            "local_gt_num_request": local_num_request,
            "dropped_uplink": int(dropped_uplink),
            "od_plan": od_plan,  # {det: n_req}
            "served_req": int(served_compute),
            "dropped_compute": int(dropped_compute),
            "va_cpu_utilization": float(va_cpu_utilization),
            "uplink_util": uplink_util,
            "mean_latency_ms": float(mean_latency_ms),
            "qoe": float(qoe),
        }
        return cache
    
 