import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from simulation.service import IDS, VideoPipeline
from simulation.request import UserCamera, Attacker
from simulation.environment import StepMetrics

@dataclass
class OffloadState:
    """Per-edge snapshot used by the offloader for one timestep."""
    area_id: str

    # Tracking queue in number of objects
    q_obj: int

    # Available tracking compute in cycles/ms after IDS and after OD reservation
    track_cycles_per_ms: float

    # Cost per tracking object in cycles (computed under receiver CPU model)
    track_cycles_per_obj: float

    # Detection safety: latest time OT is allowed to finish (ms from 'now')
    latest_track_finish_ms: float

    # Inter-edge delay lookup key
    idx: int


@dataclass
class OffloadDecision:
    src_idx: int
    dst_idx: int
    num_obj: int
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
        return cycles / max(1e-9, st.track_cycles_per_ms)

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

                    # remote completion time if one object is added to dst
                    remote_cycles = (dst_st.q_obj + 1) * dst_st.track_cycles_per_obj
                    remote_C = remote_cycles / max(1e-9, dst_st.track_cycles_per_ms)
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
        budget: ResourceBudget,
        constraints: Dict[str, float],
        ids: IDS,
        users: List[UserCamera],
        attackers: List[Attacker],
        pipeline: VideoPipeline,
        policy: Optional[Dict[str, Any]] = None,
    ):
        self.area_id = str(area_id)
        self.cpu_cycle_per_ms = float(cpu_cycle_per_ms)
        self.budget = budget

        self.constraints = {
            "D_min": float(constraints.get("D_min", 1e9)),
            "MOTA_min": float(constraints.get("MOTA_min", 0.0)),
        }

        self.ids = ids
        self.users = list(users)
        self.attackers = list(attackers)
        self.pipeline = pipeline

        self.policy = policy or {}
        self.cpu_to_ids_ratio = float(self.policy.get("cpu_to_ids_ratio", 0.2))

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

    def va_cycles_per_ms(self) -> float:
        # remaining cycles for VA after reserving IDS CPU share
        return float(self.total_cycles_per_ms() * (1.0 - self.cpu_to_ids_ratio))

    def ids_cycles_per_ms(self) -> float:
        return float(self.total_cycles_per_ms() * self.cpu_to_ids_ratio)

    # --------------------------
    # Load aggregation
    # --------------------------

    def _attack_df_at(self, t: int) -> pd.DataFrame:
        atk_rows = [atk.load_at(t) for atk in self.attackers]
        if not atk_rows:
            return pd.DataFrame(columns=["t", "attack_type", "lambda_req"])
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

    # --------------------------
    # VA compute cost and feasibility
    # --------------------------

    def compute_required_cycles_per_ms(
        self,
        t: int,
        detector: str,
        base_resolution_h: int,
    ) -> float:
        """
        Required VA cycles per ms for this area under a given config.

        cost_cycles = (sum_user_latency_ms) * cpu_cycle_per_ms * budget.cpu

        Where sum_user_latency_ms includes:
          - detection latency per frame (per user)
          - reid latency per object (per user)
        """
        total_latency_ms = 0.0

        for u in self.users:
            nobj = u.get_num_objects(t, detector, base_resolution_h)
            total_latency_ms += self.pipeline.total_latency_ms(detector, base_resolution_h, nobj)

        return float(total_latency_ms * self.cpu_cycle_per_ms * self.budget.cpu)

    def is_action_feasible(self, t: int, detector: str, base_resolution_h: int) -> bool:
        required = self.compute_required_cycles_per_ms(t, detector, base_resolution_h)
        available = self.va_cycles_per_ms()
        return required <= available

    # --------------------------
    # Performance / objective
    # --------------------------

    def compute_performance(
        self,
        t: int,
        detector: str,
        base_resolution_h: int,
        gamma: float = 0.01,
    ) -> float:
        """
        Mean performance across users based on:

        q_u^{y,r,i} =
          (MOTA^{y,r,i} - MOTA_min_e) * exp( -gamma * (D_u^i - D_min_e) )
        """
        D_min = self.constraints["D_min"]
        MOTA_min = self.constraints["MOTA_min"]

        if not self.users:
            return 0.0

        q_vals: List[float] = []
        for u in self.users:
            nobj = u.get_num_objects(t-1, detector, base_resolution_h)
            D_u = self.pipeline.total_latency_ms(detector, base_resolution_h, nobj)
            mota = u.get_mota(detector, base_resolution_h)

            q_u = (mota - MOTA_min) * np.exp(-gamma * (D_u - D_min))
            q_vals.append(float(q_u))

        return float(np.mean(q_vals)) if q_vals else 0.0

    # --------------------------
    # Action selection
    # --------------------------

    def choose_action(self, t: int) -> Tuple[str, int]:
        """
        Choose feasible (detector, resolution_h) maximizing performance.
        Fallback: fastest detection config (ignoring feasibility) if none feasible.
        """
        best_perf = -np.inf
        best_action: Optional[Tuple[str, int]] = None

        for det, h in self.pipeline.all_actions():
            if not self.is_action_feasible(t, det, h):
                continue

            perf = self.compute_performance(t, det, h)
            if perf > best_perf:
                best_perf = perf
                best_action = (det, h)

        if best_action is None:
            # fallback: fastest detector latency, this guarantees a choice
            best_action = min(
                self.pipeline.all_actions(),
                key=lambda a: self.pipeline.detection_ms(a[0], a[1]),
            )

        self._last_action = best_action
        return best_action


    def estimate_detection_cycles_this_frame(
        self,
        detector: str,
        proc_h: int,
        num_cameras: int,
    ) -> float:
        """
        OD runs once per camera each frame.
        Convert OD latency to cycles using your existing convention.
        """
        det_ms = self.pipeline.detection_ms(detector, proc_h)
        return float(num_cameras * det_ms * self.total_cycles_per_ms())

    def tracking_cycles_per_object(self) -> float:
        """
        OT cost per object in cycles.
        Uses reid_latency(ms) and this edge's total cycles/ms.
        """
        return float(self.pipeline.reid_latency * self.total_cycles_per_ms())

    def detection_safe_latest_track_finish_ms(self) -> float:
        """
        Constraint: do not accept OT that would interfere with next OD.
        For synchronized OD, the simplest framing is:
          OT must finish before next frame OD begins.

        You can set slot_ms in policy, default 1000ms.
        You can also reserve a margin.
        """
        slot_ms = float(self.policy.get("slot_ms", 1000.0))
        guard_ms = float(self.policy.get("od_guard_ms", 0.0))
        return max(0.0, slot_ms - guard_ms)

    def build_offload_state(
        self,
        t: int,
        proc_action: Tuple[str, int],
        q_obj: int,
        va_cycles_per_ms: float,
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
        od_time_ms = od_cycles / max(1e-9, va_cycles_per_ms)

        # Detection-safe deadline for OT
        latest_track_finish_ms = self.detection_safe_latest_track_finish_ms()

        # Remaining window for OT in this frame
        remaining_time_ms = max(0.0, latest_track_finish_ms - od_time_ms)

        return OffloadState(
            area_id=self.area_id,
            q_obj=int(q_obj),
            track_cycles_per_ms=float(va_cycles_per_ms),
            track_cycles_per_obj=self.tracking_cycles_per_object(),
            latest_track_finish_ms=remaining_time_ms,
            idx=-1,  # filled by offloader
        )
    
        
    def step(
        self,
        t: int,
        all_edge_areas: List["EdgeArea"],
        delay_ms: np.ndarray,
    ) -> StepMetrics:
        """
        One synchronized timestep with cooperative OT offloading.

        all_edge_areas: list of EdgeArea (including self)
        delay_ms[i, j]: inter-edge propagation delay in ms
        """

        # ============================================================
        # 1) IDS filtering (network-level, before uplink & CPU use)
        # ============================================================
        ids_out = self.aggregate_load_after_ids(t)

        total_users = float(len(self.users))
        user_pass_rate = float(ids_out.get("user_pass_rate", total_users))
        user_pass_frac = (
            user_pass_rate / max(1e-6, total_users) if total_users > 0 else 0.0
        )

        # ============================================================
        # 2) Effective uplink after surviving attacks
        # ============================================================
        attack_df = self._attack_df_at(t)

        atk_in = float(ids_out.get("attack_in_rate", 0.0))
        atk_pass = float(ids_out.get("attack_pass_rate", 0.0))
        atk_pass_frac = atk_pass / atk_in if atk_in > 0 else 0.0

        if "uplink_mbps" in attack_df.columns and not attack_df.empty:
            attack_uplink_in = float(attack_df["uplink_mbps"].sum())
        else:
            attack_mbps_per_req = float(self.policy.get("attack_mbps_per_req", 0.0))
            attack_uplink_in = (
                float(attack_df["lambda_req"].sum()) * attack_mbps_per_req
                if "lambda_req" in attack_df.columns
                else 0.0
            )

        attack_uplink_pass = attack_uplink_in * atk_pass_frac
        uplink_available = max(0.0, self.budget.uplink - attack_uplink_pass)

        # ============================================================
        # 3) Decide upload resolution under uplink constraint
        # ============================================================
        upload_candidates = sorted({h for (_, h) in self.pipeline.all_actions()})

        uplink_table = self.policy.get("uplink_mbps_by_h", None)

        def uplink_mbps_for_h(h: int) -> float:
            if isinstance(uplink_table, dict) and h in uplink_table:
                return float(uplink_table[h])
            base = float(self.policy.get("uplink_mbps_at_736", 6.0))
            return base * (h / 736.0) ** 2

        feasible_uploads = []
        for h in upload_candidates:
            per_user = uplink_mbps_for_h(h)
            if user_pass_frac * total_users * per_user <= uplink_available + 1e-9:
                feasible_uploads.append(h)

        upload_h = max(feasible_uploads) if feasible_uploads else min(upload_candidates)

        # ============================================================
        # 4) VA compute supply after IDS and surviving attack CPU load
        # ============================================================
        total_cycles_per_ms = self.cpu_cycle_per_ms * self.budget.cpu
        va_cycles_per_ms = total_cycles_per_ms * (1.0 - self.cpu_to_ids_ratio)

        attack_pass_rate = float(ids_out.get("attack_pass_rate", 0.0))
        attack_cycles_per_ms = (
            attack_pass_rate * self.ids.cycles_per_request / 1000.0
        )
        va_cycles_per_ms = max(0.0, va_cycles_per_ms - attack_cycles_per_ms)

        # ============================================================
        # 5) Choose local VA configuration (detector, proc_h)
        # ============================================================
        D_min = self.constraints["D_min"]
        MOTA_min = self.constraints["MOTA_min"]
        gamma = float(self.policy.get("gamma", 0.01))

        best_q = -np.inf
        best_action = None
        best_mean_latency = 0.0
        best_mean_mota = 0.0

        for det, proc_h in self.pipeline.all_actions():
            if proc_h > upload_h:
                continue

            demand_cycles = 0.0
            motas = []

            det_ms = self.pipeline.detection_ms(det, proc_h)

            for u in self.users:
                nobj = u.get_num_objects(t - 1, det, proc_h)
                per_user_ms = det_ms + nobj * self.pipeline.reid_latency
                demand_cycles += per_user_ms * total_cycles_per_ms
                motas.append(u.get_mota(det, proc_h))

            demand_cycles *= user_pass_frac
            mean_latency = (
                demand_cycles / max(1e-9, va_cycles_per_ms)
                if va_cycles_per_ms > 0
                else float("inf")
            )
            mean_mota = float(np.mean(motas)) if motas else 0.0

            q = (mean_mota - MOTA_min) * np.exp(-gamma * (mean_latency - D_min))

            if q > best_q:
                best_q = q
                best_action = (det, proc_h)
                best_mean_latency = mean_latency
                best_mean_mota = mean_mota

        if best_action is None:
            candidates = [(d, h) for (d, h) in self.pipeline.all_actions() if h <= upload_h]
            best_action = (
                min(candidates, key=lambda a: self.pipeline.detection_ms(a[0], a[1]))
                if candidates
                else min(
                    self.pipeline.all_actions(),
                    key=lambda a: self.pipeline.detection_ms(a[0], a[1]),
                )
            )

        self._last_action = best_action

        # ============================================================
        # 6) Build offloading states for all edges
        # ============================================================
        states = []
        for e in all_edge_areas:
            e_ids_out = e.aggregate_load_after_ids(t)
            states.append(
                e.build_offload_state(
                    t=t,
                    proc_action=e._last_action,
                    ids_out=e_ids_out,
                    upload_h=upload_h,
                    va_cycles_per_ms=va_cycles_per_ms,
                )
            )

        # ============================================================
        # 7) Cooperative OT offloading
        # ============================================================

        states, _ = cooperative_offload_ot(states, delay_ms)

        # ============================================================
        # 8) Final QoE computation using post-offloading queues
        # ============================================================
        my_state = next(s for s in states if s.area_id == self.area_id)

        ot_cycles = my_state.q_obj * my_state.track_cycles_per_obj
        ot_latency = ot_cycles / max(1e-9, my_state.track_cycles_per_ms)

        final_latency = best_mean_latency + ot_latency
        final_q = (best_mean_mota - MOTA_min) * np.exp(-gamma * (final_latency - D_min))

        return StepMetrics(
            t=t,
            area_id=self.area_id,
            qoe_mean=float(final_q),
            qoe_min=float(final_q),
            mean_latency_ms=float(final_latency),
            mean_mota=float(best_mean_mota),
            ids_coverage=float(ids_out.get("coverage", 0.0)),
            attack_in_rate=float(ids_out.get("attack_in_rate", 0.0)),
            attack_drop_rate=float(ids_out.get("attack_drop_rate", 0.0)),
            user_drop_rate=float(ids_out.get("user_drop_rate", 0.0)),
        )