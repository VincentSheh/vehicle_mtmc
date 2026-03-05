import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math


from service import IDS, VideoPipeline

from request import User, Attacker

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
        self.attackers = list(attackers)
        
        self.pipeline = pipeline

        self.ids_cpu = self.budget.cpu - 0.5
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

    def reset_running_attack_stats(self):
        self._atk_ema_inited = False
        self._atk_ema = 0.0
        self._atk_mom_ema = 0.0    
        
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

        for atk in self.cur_attacker:
            r = atk.load_at(t)
            if r is None:
                continue

            flows = float(r["flows_per_sec"])
            total_flows += flows
            total_bw_in += flows * float(atk.bw_per_flow)
            total_cycles_per_s += flows * float(atk.cycle_per_flow)

            # if multiple attackers: pick one policy
            # option A: sum (most consistent if you treat as total intensity)
            ema += float(r.get("flows_per_sec_ema", 0.0))
            mom += float(r.get("flows_per_sec_ema_mom", 0.0))

            # option B: max magnitude for mom (if you want “worst burst”)
            # m = float(r.get("flows_per_sec_ema_mom", 0.0))
            # if abs(m) > abs(mom): mom = m

        return {
            "flows": total_flows,
            "bw_in": total_bw_in,
            "cycles_per_s": total_cycles_per_s,
            "ema": ema,
            "mom": mom,
        }

    def aggregate_load_after_ids(
        self,
        t: int,
        user_in: int,
        atk_in: int,
        inspect_in: int,
        attack_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Executor-side IDS outcome based on OFFLOADED COUNTS.
        All *_cnt fields are counts per epoch.
        Also attaches *_per_s fields for convenience (counts / slot_s).
        """
        user_in = int(max(user_in, 0))
        atk_in = int(max(atk_in, 0))
        inspect_in = int(max(inspect_in, 0))
        if inspect_in != user_in + atk_in:
            inspect_in = user_in + atk_in

        attack_dict = attack_dict or {}

        out = self.ids.classify_workload(
            attack_dict=attack_dict,
            user_in=float(user_in),
            atk_in=float(atk_in),
            ids_cpu=float(self.ids_cpu),
        )

        # optional per-second views (true "rates")
        slot_s = max(float(self.slot_ms) / 1000.0, 1e-9)

        return out
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

    def observe_arrivals(self, t: int) -> Dict:
        # user workload
        total_req_in = float(sum(u.num_requests_at(t) for u in self.users))
        user_req_in = int(np.floor(total_req_in))

        # attack workload (count proxy)
        attack_dict = self._attack_agg_at(t)
        atk_req_in = float(attack_dict["flows"])

        # 2) otherwise convert rate to count
        if atk_req_in <= 0.0:
            if "lambda_req" in attack_dict:
                # interpret as req/s
                atk_req_in = float(attack_dict["lambda_req"]) * (float(self.slot_ms) / 1000.0)
            elif "attack_in_rate" in attack_dict:
                # interpret as req/s
                atk_req_in = float(attack_dict["attack_in_rate"]) * (float(self.slot_ms) / 1000.0)
            elif "flows_per_s" in attack_dict:
                # interpret as req/s
                atk_req_in = float(attack_dict["flows_per_s"]) * (float(self.slot_ms) / 1000.0)

        atk_req_in_int = int(np.floor(max(0.0, atk_req_in)))

        # total workload used by IDS stage proxy
        total_workload_in = int(user_req_in + atk_req_in_int)

        return {
            "total_req_in": total_req_in,            # keep for backwards compatibility
            "user_req_in": int(user_req_in),
            "atk_req_in": int(atk_req_in_int),
            "total_workload_in": int(total_workload_in),
            "attack_dict": attack_dict,
            "local_num_request": int(user_req_in),   # keep name used in your history logger
        }

    def process_ids(self, t: int, user_in: int, atk_in: int, inspect_in: int, attack_dict: Dict,) -> Dict:
        """
        IDS executor-side processing.
        Inputs are COUNTS for this epoch after IDS offloading has been applied.
        Returns COUNTS (pass/drop) + utilization for routing math and logging.

        user_in:  user inspection items received by this executor this epoch
        atk_in:   attack inspection items received by this executor this epoch
        inspect_in: user_in + atk_in (kept explicit to avoid mismatch)
        """
        ids_out = self.aggregate_load_after_ids(
            t=t,
            user_in=int(user_in),
            atk_in=int(atk_in),
            inspect_in=int(inspect_in),
            attack_dict=attack_dict
        )
        return ids_out

    def process_va(
        self,
        t: int,
        admitted_user_req_in: int,
        attack_dict: Dict,
        ids_out: Dict,
    ) -> Dict:
        """
        Executes VA pipeline for a given admitted workload count.

        Updated to work with new IDS outputs:
        - Prefer count-style keys: atk_in_cnt / atk_pass_cnt
        - Fallback to legacy rate-style keys: attack_in_rate / attack_pass_rate
        """
        total_req_in = float(admitted_user_req_in)
        local_num_request = int(admitted_user_req_in)

        # --- UPDATED: compute attack pass fraction robustly ---
        atk_in = float(ids_out.get("atk_in_cnt", ids_out.get("attack_in_rate", 0.0)))
        atk_pass = float(ids_out.get("atk_pass_cnt", ids_out.get("attack_pass_rate", 0.0)))
        atk_pass_frac = (atk_pass / atk_in) if atk_in > 0 else 0.0
        atk_pass_frac = float(np.clip(atk_pass_frac, 0.0, 1.0))

        # attack pressure that survives IDS and reaches VA/uplink
        attack_uplink_in = float(attack_dict.get("bw_in", 0.0)) * atk_pass_frac
        attack_cycles_per_ms = (float(attack_dict.get("cycles_per_s", 0.0)) * atk_pass_frac) / 1000.0

        # --- Compute Attack EMA Momentum ---
        atk_signal = float(atk_in)  # choose pressure that matters to VA/uplink

        if not getattr(self, "_atk_ema_inited", False):
            self._atk_ema_inited = True
            self._atk_ema = atk_signal
            self._atk_mom_ema = 0.0
        else:
            prev_ema = float(self._atk_ema)
            a = float(self._atk_alpha)

            # EMA update
            self._atk_ema = a * atk_signal + (1.0 - a) * prev_ema

            # momentum = delta EMA, then smooth it (optional but recommended)
            mom_raw = float(self._atk_ema) - prev_ema
            ma = float(self._atk_mom_alpha)
            self._atk_mom_ema = ma * mom_raw + (1.0 - ma) * float(self._atk_mom_ema)

        attack_ema = float(self._atk_ema)
        attack_mom = float(self._atk_mom_ema)

        uplink_total_mb = self.budget.uplink / (1000.0 / self.slot_ms)
        uplink_attack_used = attack_uplink_in
        uplink_available = max(0.0, uplink_total_mb - uplink_attack_used)

        total_cycles_per_ms = self.cpu_cycle_per_ms * self.budget.cpu
        avail_cycles_per_ms = self.cpu_cycle_per_ms * self.va_cpu
        avail_cycles_aft_atk_per_ms = max(0.0, avail_cycles_per_ms - attack_cycles_per_ms)

        if avail_cycles_aft_atk_per_ms <= 1e-12 or uplink_available <= 1e-12:
            return {
                "ids_out": ids_out,
                "local_num_request": local_num_request,
                "ema": attack_ema,
                "ema_mom": attack_mom,
                "dropped_uplink": int(total_req_in),
                "od_plan": {},
                "served_req": 0,
                "dropped_compute": int(total_req_in),
                "va_cpu_utilization": (total_cycles_per_ms - avail_cycles_aft_atk_per_ms)
                / max(total_cycles_per_ms, 1e-9),
                "uplink_util": 1.0,
                "mean_latency_ms": float("inf"),
                "qoe": 0.0,
            }

        # treat admitted_user_req_in as "passed pre uplink"
        passed_req_pre_uplink = int(admitted_user_req_in)

        upload_plan, served_req_uplink, dropped_uplink, uplink_util, per_req_by_h = self.select_resolution(
            passed_req_pre_uplink=int(passed_req_pre_uplink),
            uplink_available=float(uplink_available),
            uplink_attack_used=float(uplink_attack_used),
            uplink_total_mb=float(uplink_total_mb),
            upload_hs=(224, 320, 412),
        )

        gamma = float(self.constraints.get("Gamma", 0.0))

        dets = []
        for a in self.pipeline.all_actions():
            det = a[0] if isinstance(a, tuple) else a
            dets.append(det)
        dets = sorted(set(dets))

        det_costs = {det: float(self.pipeline.detection_cycles(det)) for det in dets}

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

        va_cpu_utilization = used_cycles / max(avail_cycles_per_ms * self.slot_ms, 1e-9)

        served_compute = int(sum(od_plan.values()))
        assert served_compute + int(dropped_compute) == int(served_req_uplink)

        if total_req_in <= 0:
            qoe = 1.0
        else:
            qoe = float(qoe) * (served_compute / total_req_in)

        return {
            "ids_out": ids_out,
            "local_num_request": local_num_request,
            "ema": attack_ema,
            "ema_mom": attack_mom,
            "dropped_uplink": int(dropped_uplink),
            "od_plan": od_plan,
            "served_req": int(served_compute),
            "dropped_compute": int(dropped_compute),
            "va_cpu_utilization": float(va_cpu_utilization),
            "uplink_util": float(uplink_util),
            "mean_latency_ms": float(mean_latency_ms),
            "qoe": float(qoe),
        }