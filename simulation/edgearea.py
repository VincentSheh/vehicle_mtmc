import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math


from service import IDS, VideoPipeline

from request import User, Attacker, AttackTypeLibrary

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
        slo_beta: float,
        slo_threshold: float,
        budget: ResourceBudget,
        constraints: Dict[str, float],
        ids: IDS,
        users: List[User],
        attackers: List[Attacker],
        pipeline: VideoPipeline,
        attack_type_library: Optional[AttackTypeLibrary] = None,
        t_max: int = 30000,
        dirichlet_alpha: float = 1.0,
    ):
        self.area_id = str(area_id)
        self.cpu_cycle_per_ms = float(cpu_cycle_per_ms)
        self.budget = budget
        self.slot_ms = slot_ms
        self.slo_beta = slo_beta
        self.slo_threshold = slo_threshold

        self.constraints = {
            "D_Max": float(constraints.get("D_Max", 1e9)),
            "MOTA_min": float(constraints.get("MOTA_min", 0.0)),
            "Gamma": float(constraints.get("Gamma", 0.0)),
        }

        self.ids = ids
        self.users = list(users)
        self._all_attackers = list(attackers)  # original static list (legacy)
        self.attackers = list(attackers)

        self.attack_type_library = attack_type_library
        self._t_max = t_max
        self.dirichlet_alpha = dirichlet_alpha

        self.pipeline = pipeline

        if attack_type_library is not None:
            print(f"\n[{area_id}] Attack type library (fixed for this run):")
            print(f"  {'ID':<4} {'pattern':<8} {'λ_base':>8} {'noise_σ':>8} "
                  f"{'t_min(s)':>9} {'t_max(s)':>9} {'lat_ms':>7} {'bw_Mbps':>8}")
            for tid in range(attack_type_library.n_types):
                s = attack_type_library.get(tid)
                print(f"  {tid:<4} {s.pattern_type:<8} {s.lambda_base:>8.1f} {s.noise_std:>8.3f} "
                      f"{s.t_min_pattern:>9.1f} {s.t_max_pattern:>9.1f} "
                      f"{s.latency_per_flow:>7.3f} {s.bw_per_flow:>8.4f}")

        self.ids_cpu = 0.5
        self.va_cpu = self.budget.cpu - self.ids_cpu

        self._last_action: Optional[Tuple[str, int]] = None
        
        # --- running attack EMA/momentum computed from *actual received* workload ---
        self._atk_ema_inited = False
        self._atk_ema = 0.0
        self._atk_mom_ema = 0.0  # smoothed momentum

        # half-life in seconds for smoothing (same spirit as your Attacker hl=50.0)
        hl_sec = 50.0
        hl_steps = max(1.0, hl_sec / (self.slot_ms / 1000.0))
        self._atk_alpha = 1.0 - math.exp(math.log(0.5) / hl_steps)

        # momentum smoothing can be same or a bit faster, here same
        self._atk_mom_alpha = self._atk_alpha  
        
    def reset_running_attack_stats(self):
        self._atk_ema_inited = False
        self._atk_ema = 0.0
        self._atk_mom_ema = 0.0                    

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

        # 3) Sample a new attack type from the library for this episode
        if self.attack_type_library is not None:
            n = self.attack_type_library.n_types
            concentration = np.ones(n) * self.dirichlet_alpha
            p_attack_type = self.rng.dirichlet(concentration)

            chosen_type_id = int(self.rng.choice(n, p=p_attack_type))
            spec = self.attack_type_library.get(chosen_type_id)

            print(f"[{self.area_id}] Sampled type_{chosen_type_id} "
                  f"(pattern={spec.pattern_type}, λ_base={spec.lambda_base:.1f})")

            atk_seed = int(self.rng.integers(0, 2**32))
            self.attackers = [Attacker(
                attacker_id=f"atk_type_{chosen_type_id}",
                spec=spec,
                slot_ms=self.slot_ms,
                t_max=self._t_max,
                seed=atk_seed,
                cpu_cycle_per_ms=self.cpu_cycle_per_ms,
                cpu_cores=int(self.budget.cpu),
            )]
        else:
            # Legacy path: select one from the static attacker list
            for atk in self._all_attackers:
                atk_seed = int(self.rng.integers(0, 2**32))
                atk.reset(seed=atk_seed)
            idx = int(self.rng.integers(0, len(self._all_attackers)))
            self.attackers = [self._all_attackers[idx]]

    def get_state(self) -> dict:
        return {
            "ids_cpu": self.ids_cpu,
            "va_cpu": self.va_cpu,
            "_atk_ema_inited": self._atk_ema_inited,
            "_atk_ema": self._atk_ema,
            "_atk_mom_ema": self._atk_mom_ema,
            "attacker_states": [atk.get_state() for atk in self.attackers],
        }

    def set_state(self, state: dict):
        self.ids_cpu = state["ids_cpu"]
        self.va_cpu = state["va_cpu"]
        self._atk_ema_inited = state["_atk_ema_inited"]
        self._atk_ema = state["_atk_ema"]
        self._atk_mom_ema = state["_atk_mom_ema"]
        for i, atk_state in enumerate(state["attacker_states"]):
            self.attackers[i].set_state(atk_state)

    def get_state(self) -> dict:
        return {
            "ids_cpu": self.ids_cpu,
            "va_cpu": self.va_cpu,
            "_atk_ema_inited": self._atk_ema_inited,
            "_atk_ema": self._atk_ema,
            "_atk_mom_ema": self._atk_mom_ema,
            "attacker_states": [atk.get_state() for atk in self.attackers],
        }

    def set_state(self, state: dict):
        self.ids_cpu = state["ids_cpu"]
        self.va_cpu = state["va_cpu"]
        self._atk_ema_inited = state["_atk_ema_inited"]
        self._atk_ema = state["_atk_ema"]
        self._atk_mom_ema = state["_atk_mom_ema"]
        for i, atk_state in enumerate(state["attacker_states"]):
            self.attackers[i].set_state(atk_state)

    # --------------------------
    # Load aggregation
    # --------------------------

    def _attack_agg_at(self, t: int) -> dict:
        # aggregate only what you use later
        total_flows = 0.0
        total_bw_in = 0.0          # flows * bw_per_flow
        total_cycles_per_s = 0.0   # flows * cycle_per_flow
        ema = 0.0
        mom = 0.0

        for atk in self.attackers:
         
            if not getattr(atk, "episode_active", True):
                continue
            r = atk.load_at(t)            
            
            if r is None:
                continue

            flows = float(r["flows_per_sec"])
            total_flows += flows
            total_bw_in += flows * float(atk.bw_per_flow)
            total_cycles_per_s += flows * float(atk.cycle_per_flow)

            # if multiple attackers: pick one policy
            # option A: sum (most consistent if you treat as total intensity)
            # Use 0.0 if not provided by the attacker (e.g. in patterned mode)
            ema += float(r.get("flows_per_sec_ema", 0.0))
            mom += float(r.get("flows_per_sec_ema_mom", 0.0))

        return {
            "flows": total_flows,
            "bw_in": total_bw_in,
            "cycles_per_s": total_cycles_per_s,
            "ema": ema,
            "mom": mom,
        }

    def aggregate_load_after_ids(self, t: int, attack_dict = Dict[str, Any]) -> Dict[str, float]:
        user_rate = float(sum(u.num_requests_at(t) for u in self.users))
        return self.ids.classify_rates(
            attack_dict=attack_dict,
            user_rate=user_rate,
            ids_cpu=self.ids_cpu,
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

    from typing import Dict, Tuple, List


    def select_resolution(
        self,
        passed_req_pre_uplink: int,
        uplink_available: float,        # Mb available for benign users in this slot
        uplink_attack_used: float,      # Mb already consumed by attacks
        uplink_total_mb: float,         # total uplink budget for utilization
        upload_hs=(223, 320, 416),
    ) -> Tuple[Dict[int, int], int, int, float, Dict[int, float]]:
        """
        Algorithm 1: Resolution selection under uplink budget.

        Stage 1: assign as many requests as possible to the lowest-cost resolution
        Stage 2: upgrade some served requests to higher resolutions using leftover uplink

        Returns
        -------
        upload_plan : {h: n_req}
        served_req_uplink : int
        dropped_uplink : int
        uplink_util : float
        per_req_uplink_by_h : {h: Mb_per_req}
        """

        def uplink_mbps_for_h(h: int) -> float:
            # Consistent with previous implementation
            w = h
            size_bits = 0.1 * (w * h * 3) * 8
            return size_bits / (1024.0 * 1024.0)

        hs = sorted(set(int(h) for h in upload_hs))
        per_req = {h: float(uplink_mbps_for_h(h)) for h in hs}

        if passed_req_pre_uplink <= 0 or uplink_available <= 0 or not hs:
            upload_plan = {h: 0 for h in hs}
            uplink_util = min(1.0, uplink_attack_used / max(1e-9, uplink_total_mb))
            return (
                upload_plan,
                0,
                int(max(0, passed_req_pre_uplink)),
                float(uplink_util),
                per_req,
            )

        # Stage 1: throughput-first at minimum-cost resolution
        h_min = min(hs, key=lambda h: per_req[h])
        c_min = per_req[h_min]

        max_served = int(uplink_available // max(1e-12, c_min))
        served_req = min(int(passed_req_pre_uplink), max_served)

        upload_plan: Dict[int, int] = {h: 0 for h in hs}
        upload_plan[h_min] = served_req

        remaining_uplink = float(uplink_available) - served_req * c_min

        # Stage 2: upgrade served requests using leftover uplink
        for h in sorted(hs, reverse=True):
            if h == h_min:
                continue

            delta = float(per_req[h] - c_min)
            if delta <= 1e-12 or remaining_uplink <= 1e-12:
                continue

            can_upgrade = int(remaining_uplink // delta)
            take = min(upload_plan[h_min], can_upgrade)

            if take > 0:
                upload_plan[h_min] -= take
                upload_plan[h] += take
                remaining_uplink -= take * delta

        served_req_uplink = int(sum(upload_plan.values()))
        dropped_uplink = int(max(0, passed_req_pre_uplink - served_req_uplink))

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
        det_quality: dict | None,   # {det: quality_score}
        N: int,
        mu_cycles_per_ms: float,    # available cycles per ms
        gamma: float = 0.0,         # kept for interface compatibility
    ):
        """
        Algorithm 2: Model/detector selection under CPU budget.

        Stage 1: assign as many requests as possible to the cheapest detector
        Stage 2: upgrade some served requests to higher-cost detectors using leftover CPU budget

        Feasibility is defined by the latency-implied cycle budget:
            total_cycles <= self.slot_ms * mu_cycles_per_ms

        Returns
        -------
        plan : {det: n_req}
        feasible_all : bool
        dropped : int
        used_cycles : float
        mean_latency_ms : float
        qoe : float
        """
        import math

        if N <= 0:
            return {}, True, 0, 0.0, 0.0, 1.0

        if mu_cycles_per_ms <= 1e-12 or self.slot_ms <= 0:
            return {}, False, int(max(0, N)), 0.0, float("inf"), 0.0

        dets = list(det_costs.keys())
        if not dets:
            return {}, False, int(N), 0.0, float("inf"), 0.0

        # latency-implied cycle budget
        C_budget = float(self.slot_ms) * float(mu_cycles_per_ms)

        # quality normalization
        if det_quality is None:
            det_quality = {d: 1.0 for d in dets}

        qmax = max(1e-12, max(float(det_quality[d]) for d in dets))
        qnorm = {d: float(det_quality[d]) / qmax for d in dets}

        # cheapest detector baseline
        det_min = min(dets, key=lambda d: float(det_costs[d]))
        c_min = float(det_costs[det_min])

        def _qoe_from_plan(plan: dict, total_N: int, used_cycles: float):
            served = int(sum(plan.values()))
            mean_latency = used_cycles / max(1e-9, mu_cycles_per_ms) if served > 0 else float("inf")
            quality_sum = sum(float(n) * float(qnorm.get(d, 0.0)) for d, n in plan.items())
            qoe = (quality_sum / float(total_N)) if total_N > 0 else 0.0
            return float(qoe), float(mean_latency)

        # Stage 1: throughput-first at cheapest detector
        max_served = int(C_budget // max(1e-12, c_min))
        served_req = min(int(N), max_served)

        plan = {d: 0 for d in dets}
        if served_req > 0:
            plan[det_min] = served_req

        remaining_budget = float(C_budget) - served_req * c_min

        # If not all requests can even be served with cheapest detector
        if served_req < N:
            used_cycles = served_req * c_min
            dropped = int(N - served_req)
            qoe, mean_lat = _qoe_from_plan(
                {d: n for d, n in plan.items() if n > 0},
                int(N),
                used_cycles,
            )
            return (
                {d: n for d, n in plan.items() if n > 0},
                False,
                dropped,
                float(used_cycles),
                float(mean_lat),
                float(qoe),
            )

        # Stage 2: upgrade served requests using leftover CPU budget
        # Use descending quality order to mirror the resolution algorithm
        dets_upgrade = sorted(
            dets,
            key=lambda d: (float(det_quality[d]), float(det_costs[d])),
            reverse=True,
        )

        for det in dets_upgrade:
            if det == det_min:
                continue

            delta = float(det_costs[det]) - c_min
            if delta <= 1e-12 or remaining_budget <= 1e-12:
                continue

            can_upgrade = int(remaining_budget // delta)
            take = min(plan[det_min], can_upgrade)

            if take > 0:
                plan[det_min] -= take
                plan[det] += take
                remaining_budget -= take * delta

        plan = {d: int(n) for d, n in plan.items() if n > 0}
        used_cycles = sum(n * float(det_costs[d]) for d, n in plan.items())

        assert sum(plan.values()) == int(N)
        assert used_cycles <= C_budget + 1e-6

        qoe, mean_lat = _qoe_from_plan(plan, int(N), used_cycles)
        return plan, True, 0, float(used_cycles), float(mean_lat), float(qoe)


    def match_detectors_to_resolutions(
        self,
        upload_plan: Dict[int, int],
        det_plan: Dict[str, int],
    ) -> Tuple[float, Dict[Tuple[str, int], int]]:
        """
        Monotone matching:
        low resolution -> low-quality detector
        high resolution -> high-quality detector

        Assumes both plans are already produced by the separate pseudocode-aligned
        resolution and detector selection stages.
        """
        det_res_map = self.pipeline.res_to_acc  # {(det,h): acc}

        up = {int(h): int(n) for h, n in upload_plan.items() if int(n) > 0}
        dp = {str(d): int(n) for d, n in det_plan.items() if int(n) > 0}
        if not up or not dp:
            return 0.0, {}

        sum_up = int(sum(up.values()))
        sum_dp = int(sum(dp.values()))

        # Trim excess uploads if compute serves fewer requests than uplink admitted
        if sum_up != sum_dp:
            if sum_up < sum_dp:
                raise ValueError(
                    f"Mismatch where uploads < detections: sum(upload_plan)={sum_up} < sum(det_plan)={sum_dp}."
                )

            excess = sum_up - sum_dp
            for h in sorted(up.keys()):  # drop lowest resolutions first
                if excess <= 0:
                    break
                drop = min(up[h], excess)
                up[h] -= drop
                excess -= drop
                if up[h] == 0:
                    del up[h]

            if int(sum(up.values())) != sum_dp:
                raise RuntimeError(
                    f"Failed to reconcile counts after trimming uploads: sum_up={sum(up.values())}, sum_dp={sum_dp}"
                )

        # Detector quality proxy = best achievable accuracy across resolutions
        dets: List[str] = list(dp.keys())
        det_proxy: Dict[str, float] = {}
        for d in dets:
            vals = [float(v) for (dd, _h), v in det_res_map.items() if dd == d]
            det_proxy[d] = max(vals) if vals else 0.0

        res_list = sorted(up.keys())  # low -> high
        det_list = sorted(dets, key=lambda d: (det_proxy.get(d, 0.0), d))  # low -> high

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
                f"Post-match leftovers: uploads={sum(rem_res.values())}, dets={sum(rem_det.values())}."
            )

        missing = [(d, int(h)) for (d, h) in assign if (d, int(h)) not in det_res_map]
        if missing:
            raise KeyError(f"Missing (det,h) in pipeline.res_to_acc: {missing[:10]}")

        total = int(sum(assign.values()))
        if total <= 0:
            return 0.0, {}

        qmax = max(1e-12, max(float(v) for v in det_res_map.values()))
        qoe_sum = 0.0
        for (d, h), n in assign.items():
            q = float(det_res_map[(d, int(h))]) / qmax
            qoe_sum += float(n) * q

        qoe = qoe_sum / float(total)
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
        attack_dict = self._attack_agg_at(t)

        ids_out = self.aggregate_load_after_ids(t, attack_dict)  # update signature
        
        # Calculate Yo-Yo provisioning state and update attackers
        util = float(ids_out.get("ids_cpu_util", 0.0))
        if util > 0.8 :
            z_t = 1
        elif util < 0.2 and self.ids_cpu < 1.0:
            z_t = -1
        else:
            z_t = 0
            
        for atk in self.attackers:
            if hasattr(atk, "z_t"):
                atk.z_t = z_t

        user_pass_rate = float(ids_out.get("user_pass_rate", total_req_in))
        passed_req_pre_uplink = int(np.ceil(max(0.0, user_pass_rate)))

        atk_in = float(ids_out.get("attack_in_rate", 0.0))
        atk_pass = float(ids_out.get("attack_pass_rate", 0.0))
        atk_pass_frac = atk_pass / atk_in if atk_in > 0 else 0.0
        attack_uplink_in = attack_dict["bw_in"] * atk_pass_frac
        attack_cycles_per_ms = (attack_dict["cycles_per_s"] * atk_pass_frac) / 1000.0

        # -------------------------------------------------
        # Attack EMA / momentum based on admitted attack load
        # -------------------------------------------------
        atk_signal = float(atk_in)

        if not getattr(self, "_atk_ema_inited", False):
            self._atk_ema_inited = True
            self._atk_ema = atk_signal
            self._atk_mom_ema = 0.0
        else:
            prev_ema = float(self._atk_ema)
            a = float(self._atk_alpha)

            self._atk_ema = a * atk_signal + (1.0 - a) * prev_ema

            mom_raw = float(self._atk_ema) - prev_ema
            ma = float(self._atk_mom_alpha)
            self._atk_mom_ema = ma * mom_raw + (1.0 - ma) * float(self._atk_mom_ema)

        attack_ema = float(self._atk_ema)
        attack_mom = float(self._atk_mom_ema)

        uplink_total_mb = self.budget.uplink / (1000.0 / self.slot_ms)

        uplink_attack_used = attack_uplink_in

        uplink_available = max(0.0, uplink_total_mb - uplink_attack_used)
        
        # 3) VA compute supply (after attacks)
        total_cycles_per_ms = self.cpu_cycle_per_ms * self.budget.cpu
        avail_cycles_per_ms = self.cpu_cycle_per_ms * self.va_cpu

        avail_cycles_aft_atk_per_ms = max(0.0, avail_cycles_per_ms - attack_cycles_per_ms)
        
        # If nothing survives uplink or no compute, return outage-ish state
        if avail_cycles_aft_atk_per_ms <= 1e-12 or uplink_available <=1e-12:
            cache = {
                "ids_out": ids_out,
                "local_num_request": local_num_request,
                "ema": attack_ema,
                "ema_mom": attack_mom,
                "dropped_uplink": int(total_req_in),
                "od_plan": {},
                "served_req": 0,
                "dropped_compute": int(total_req_in),
                "va_cpu_utilization": min(1,(attack_cycles_per_ms + self.ids_cpu * self.cpu_cycle_per_ms) / avail_cycles_per_ms),
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
        
        va_cpu_utilization = min(1,(used_cycles + attack_cycles_per_ms) / (avail_cycles_per_ms))
        va_cpu_utilization = min(1,(used_cycles + attack_cycles_per_ms + self.ids_cpu * self.cpu_cycle_per_ms) / (total_cycles_per_ms))
        served_compute = int(sum(od_plan.values()))
        assert served_compute + int(dropped_compute) == int(served_req_uplink)

        if total_req_in <= 0:
            qoe = 1.0
        else:
            qoe =  qoe * (served_compute / total_req_in)
        cache = {
            "ids_out": ids_out,
            "local_num_request": local_num_request,
            "ema": attack_ema,
            "ema_mom": attack_mom,
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
    
 