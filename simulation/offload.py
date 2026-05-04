from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import random
import numpy as np


EPS = 1e-9

@dataclass
class OffloadPlan:
    # flow counts, NOT fractions
    flow: Dict[Any, Dict[Any, int]]        # flow[src_area_id][dst_area_id] = n_tasks
    assigned_dst: Dict[Any, int]           # assigned_dst[dst_area_id] = total tasks executed at dst


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_and_conserve(
    flow_float: Dict[Any, Dict[Any, float]],
    W: Dict[Any, float],
) -> Dict[Any, Dict[Any, int]]:
    """Round float flows to int, then fix each sender's conservation on the diagonal."""
    flow: Dict[Any, Dict[Any, int]] = {}
    for s, mp in flow_float.items():
        flow[s] = {d: int(round(v)) for d, v in mp.items()}
    for s in flow:
        sent = sum(flow[s].values())
        want = int(round(W[s]))
        if sent != want:
            flow[s][s] = flow[s].get(s, 0) + (want - sent)
    return flow


def _build_plan(
    flow: Dict[Any, Dict[Any, int]],
    area_ids: List[Any],
) -> OffloadPlan:
    assigned_dst: Dict[Any, int] = {e: 0 for e in area_ids}
    for s in area_ids:
        for d, n in flow.get(s, {}).items():
            assigned_dst[d] += int(n)
    return OffloadPlan(flow=flow, assigned_dst=assigned_dst)


# ---------------------------------------------------------------------------
# Mode a — no offload (all-local identity plan)
# ---------------------------------------------------------------------------

def no_offload(
    area_ids: List[Any],
    W_src: Dict[Any, float],
) -> OffloadPlan:
    flow = {e: {e: int(round(float(W_src[e])))} for e in area_ids}
    return _build_plan(flow, area_ids)


# ---------------------------------------------------------------------------
# Mode b — workload balance (greedy, no delay filter)
# ---------------------------------------------------------------------------

def balance_workload(
    area_ids: List[Any],
    W_src: Dict[Any, float],
    c_dst: Dict[Any, float],
    cap_dst: Optional[Dict[Any, float]] = None,
) -> OffloadPlan:
    """Balance load proportionally to capacity; no propagation delay filtering."""
    W_tot = float(sum(W_src[e] for e in area_ids))
    C_tot = float(sum(c_dst[e] for e in area_ids))
    W     = {e: float(W_src[e]) for e in area_ids}

    if W_tot <= EPS or C_tot <= EPS:
        flow = {e: {e: int(round(W[e]))} for e in area_ids}
        return _build_plan(flow, area_ids)

    N_star = {e: float(c_dst[e]) * (W_tot / C_tot) for e in area_ids}
    if cap_dst is not None:
        N_star = {e: min(N_star[e], float(cap_dst[e])) for e in area_ids}

    supply    = {e: max(0.0, W[e] - N_star[e]) for e in area_ids}
    demand    = {e: max(0.0, N_star[e] - W[e]) for e in area_ids}
    senders   = [e for e in area_ids if supply[e] > EPS]
    receivers = [e for e in area_ids if demand[e] > EPS]

    random.shuffle(senders)
    random.shuffle(receivers)

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    rem_supply = {e: supply[e] for e in senders}
    rem_demand = {e: demand[e] for e in receivers}

    for s in senders:
        s_rem = rem_supply[s]
        for r in receivers:
            if s_rem <= EPS:
                break
            r_rem = rem_demand.get(r, 0.0)
            if r_rem <= EPS:
                continue
            u = min(s_rem, r_rem)
            flow_float[s][s] -= u
            flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
            s_rem            -= u
            rem_demand[r]    -= u
        rem_supply[s] = s_rem

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)


# ---------------------------------------------------------------------------
# Mode b2 — CTO (Collaborative Task Offloading, propagation-aware greedy)
# ---------------------------------------------------------------------------

def balance_workload_cto(
    area_ids: List[Any],
    W_src: Dict[Any, float],
    c_dst: Dict[Any, float],
    propagation_delays: Dict[Any, Dict[Any, float]],
    cap_dst: Optional[Dict[Any, float]] = None,
    max_prop_delay: float = 1e9,
    slot_ms: float = 200.0,
) -> OffloadPlan:
    """
    CTO: proportional redistribution target, then greedy forwarding over
    sender→receiver links sorted by ascending propagation delay.

    max_prop_delay: links exceeding this delay are excluded; tasks that
    cannot reach any eligible receiver stay local.
    """
    W_tot = float(sum(W_src[e] for e in area_ids))
    W     = {e: float(W_src[e]) for e in area_ids}

    if W_tot <= EPS:
        flow = {e: {e: int(round(W[e]))} for e in area_ids}
        return _build_plan(flow, area_ids)

    # c_dst is task throughput capacity (tasks/slot); use directly as the target.
    N_star = {e: float(c_dst[e]) for e in area_ids}
    if cap_dst is not None:
        N_star = {e: min(N_star[e], float(cap_dst[e])) for e in area_ids}
    supply    = {e: max(0.0, W[e] - N_star[e]) for e in area_ids}
    # demand here is a bit tricky since it depends on the sender's d_prop.
    # We'll calculate it per-link in the greedy loop.
    senders   = [e for e in area_ids if supply[e] > EPS]
    receivers = [e for e in area_ids if W[e] < N_star[e] + EPS]

    all_links: List[Tuple[Any, Any, float]] = [
        (s, r, propagation_delays[s][r])
        for s in senders
        for r in receivers
        if s != r and propagation_delays[s][r] <= max_prop_delay
    ]
    random.shuffle(all_links)
    all_links.sort(key=lambda x: x[2])

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    rem_supply = {e: supply[e] for e in senders}
    # current_W tracks total workload assigned to each node during planning
    current_W = {e: W[e] for e in area_ids}

    for s, r, d_prop in all_links:
        if rem_supply[s] <= EPS:
            continue
        
        # Effective capacity at receiver r for tasks from sender s
        eff_cap = N_star[r] * max(0.0, 1.0 - d_prop / slot_ms)
        r_demand = max(0.0, eff_cap - current_W[r])
        
        if r_demand <= EPS:
            continue
            
        u = min(rem_supply[s], r_demand)
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        rem_supply[s]    -= u
        current_W[r]     += u

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)


# ---------------------------------------------------------------------------
# Mode b3 — CTO with accuracy-aware link scoring (cto_acc)
# ---------------------------------------------------------------------------
# Link cost (s → r):
#   cost = d_prop(s,r)          * w3       ← propagation delay penalty
#        - (FNR[s,s] - FNR[s,r]) * w1     ← FNR improvement reduces cost
#        - (FPR[s,s] - FPR[s,r]) * w2     ← FPR improvement reduces cost
#
# Links are sorted ascending by cost (lowest cost = closest + most accurate first).
# With w1=w2=0 this reduces to ascending delay, identical to balance_workload_cto.
# Redistribution target and greedy forwarding are identical to balance_workload_cto.
# ---------------------------------------------------------------------------

def balance_workload_cto_acc(
    area_ids: List[Any],
    W_src: Dict[Any, float],
    c_dst: Dict[Any, float],
    propagation_delays: Dict[Any, Dict[Any, float]],
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.01),
    fnr_matrix: Optional[np.ndarray] = None,
    fpr_matrix: Optional[np.ndarray] = None,
    id_to_idx: Optional[Dict[Any, int]] = None,
    cap_dst: Optional[Dict[Any, float]] = None,
    slot_ms: float = 200.0,
) -> OffloadPlan:
    """
    CTO with accuracy-aware link scoring: same redistribution target as
    balance_workload_cto, but links are sorted by a composite score that
    rewards FNR/FPR improvement and penalises propagation delay.
    """
    w1, w2, w3 = weights

    W_tot = float(sum(W_src[e] for e in area_ids))
    W     = {e: float(W_src[e]) for e in area_ids}

    if W_tot <= EPS:
        flow = {e: {e: int(round(W[e]))} for e in area_ids}
        return _build_plan(flow, area_ids)

    N_star = {e: float(c_dst[e]) for e in area_ids}
    if cap_dst is not None:
        N_star = {e: min(N_star[e], float(cap_dst[e])) for e in area_ids}

    supply    = {e: max(0.0, W[e] - N_star[e]) for e in area_ids}
    senders   = [e for e in area_ids if supply[e] > EPS]
    receivers = [e for e in area_ids if W[e] < N_star[e] + EPS]

    scored_links: List[Tuple[float, Any, Any, float]] = []
    for s in senders:
        si = id_to_idx[s] if id_to_idx is not None else None
        for r in receivers:
            if s == r:
                continue
            d_prop = float(propagation_delays[s][r])
            delay_penalty = d_prop * w3

            # Improvement (acc) calculation with TPR/TNR (higher is better):
            # acc = (Local_Rate - Remote_Rate)
            # If Remote is BETTER, acc is POSITIVE.
            acc = 0.0
            if (fnr_matrix is not None and fpr_matrix is not None
                    and si is not None and id_to_idx is not None):
                ri = id_to_idx[r]
                acc = (
                    (float(fnr_matrix[si, si]) - float(fnr_matrix[si, ri])) * w1
                    + (float(fpr_matrix[si, si]) - float(fpr_matrix[si, ri])) * w2
                )

            # cost = delay_penalty - acc:
            # Since acc is positive for better remotes, subtracting it REDUCES cost.
            # lower cost = better link (closer + more accurate)
            cost = delay_penalty - acc
            scored_links.append((cost, s, r, d_prop))

    random.shuffle(scored_links)
    scored_links.sort(key=lambda x: x[0])

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    rem_supply = {e: supply[e] for e in senders}
    current_W = {e: W[e] for e in area_ids}

    for _, s, r, d_prop in scored_links:
        if rem_supply[s] <= EPS:
            continue
            
        eff_cap = N_star[r] * max(0.0, 1.0 - d_prop / slot_ms)
        r_demand = max(0.0, eff_cap - current_W[r])
        if r_demand <= EPS:
            continue
            
        u = min(rem_supply[s], r_demand)
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        rem_supply[s]    -= u
        current_W[r]     += u

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)


def balance_workload_cto_acc_inv(
    area_ids: List[Any],
    W_src: Dict[Any, float],
    c_dst: Dict[Any, float],
    propagation_delays: Dict[Any, Dict[Any, float]],
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.01),
    fnr_matrix: Optional[np.ndarray] = None,
    fpr_matrix: Optional[np.ndarray] = None,
    id_to_idx: Optional[Dict[Any, int]] = None,
    cap_dst: Optional[Dict[Any, float]] = None,
    slot_ms: float = 200.0,
) -> OffloadPlan:
    """
    Inverse CTO with accuracy-aware link scoring: prioritizes links that lead to
    WORSE accuracy (higher FNR/FPR) compared to the local edge.
    """
    w1, w2, w3 = weights

    W_tot = float(sum(W_src[e] for e in area_ids))
    W     = {e: float(W_src[e]) for e in area_ids}

    if W_tot <= EPS:
        flow = {e: {e: int(round(W[e]))} for e in area_ids}
        return _build_plan(flow, area_ids)

    N_star = {e: float(c_dst[e]) for e in area_ids}
    if cap_dst is not None:
        N_star = {e: min(N_star[e], float(cap_dst[e])) for e in area_ids}

    supply    = {e: max(0.0, W[e] - N_star[e]) for e in area_ids}
    senders   = [e for e in area_ids if supply[e] > EPS]
    receivers = [e for e in area_ids if W[e] < N_star[e] + EPS]

    scored_links: List[Tuple[float, Any, Any, float]] = []
    for s in senders:
        si = id_to_idx[s] if id_to_idx is not None else None
        for r in receivers:
            if s == r:
                continue
            d_prop = float(propagation_delays[s][r])
            delay_penalty = d_prop * w3

            # Improvement (acc) calculation: (Local_Rate - Remote_Rate)
            # If Remote is BETTER, acc is POSITIVE.
            acc = 0.0
            if (fnr_matrix is not None and fpr_matrix is not None
                    and si is not None and id_to_idx is not None):
                ri = id_to_idx[r]
                # If Remote is BETTER, acc is POSITIVE.
                acc = (
                    (float(fnr_matrix[si, si]) - float(fnr_matrix[si, ri])) * w1
                    + (float(fpr_matrix[si, si]) - float(fpr_matrix[si, ri])) * w2
                )

            # cost = delay_penalty + acc:
            # Since acc is positive for better remotes, adding it INCREASES cost.
            # higher cost = better remote is penalized (inverse behavior)
            cost = delay_penalty + acc
            scored_links.append((cost, s, r, d_prop))

    random.shuffle(scored_links)
    scored_links.sort(key=lambda x: x[0])

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    rem_supply = {e: supply[e] for e in senders}
    current_W = {e: W[e] for e in area_ids}

    for _, s, r, d_prop in scored_links:
        if rem_supply[s] <= EPS:
            continue
            
        eff_cap = N_star[r] * max(0.0, 1.0 - d_prop / slot_ms)
        r_demand = max(0.0, eff_cap - current_W[r])
        if r_demand <= EPS:
            continue
            
        u = min(rem_supply[s], r_demand)
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        rem_supply[s]    -= u
        current_W[r]     += u

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)


# ---------------------------------------------------------------------------
# Modes c & d — score-based routing
# ---------------------------------------------------------------------------


def cto_balanced(
    area_ids: List[Any],
    W_src: Dict[Any, float],
    c_dst: Dict[Any, float],
    propagation_delays: Dict[Any, Dict[Any, float]],
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.01),
    fnr_matrix: Optional[np.ndarray] = None,
    fpr_matrix: Optional[np.ndarray] = None,
    id_to_idx: Optional[Dict[Any, int]] = None,
    cap_dst: Optional[Dict[Any, float]] = None,
    max_prop_delay: float = 1e9,
    slot_ms: float = 200.0,
) -> OffloadPlan:
    """
    Original score_based_offload logic: proportional offloading triggered 
    only by capacity overload.
    """
    w1, w2, w3 = weights

    W_tot = float(sum(W_src[e] for e in area_ids))
    W     = {e: float(W_src[e]) for e in area_ids}

    if W_tot <= EPS:
        flow = {e: {e: int(round(W[e]))} for e in area_ids}
        return _build_plan(flow, area_ids)

    # Triggered only by raw capacity overload
    supply = {e: max(0.0, W[e] - float(c_dst[e])) for e in area_ids}
    
    senders   = [e for e in area_ids if supply[e] > EPS]
    # Receivers are nodes that can accept more work within their window
    receivers = [e for e in area_ids if W[e] < float(c_dst[e]) + EPS]

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    rem_supply = {e: supply[e] for e in senders}
    current_W = {e: W[e] for e in area_ids}

    for src in senders:
        si      = id_to_idx[src] if id_to_idx is not None else None
        to_move = rem_supply[src]
        if to_move <= EPS:
            continue

        is_overloaded = W[src] > float(c_dst[src]) + EPS

        raw: Dict[Any, float] = {}
        eff_cap_raw: Dict[Any, float] = {}
        for r in receivers:
            if r == src:
                continue
            d_prop = float(propagation_delays[src][r])
            if d_prop > max_prop_delay:
                continue
            
            # Effective capacity at receiver r for tasks from sender src
            eff_cap = float(c_dst[r]) * max(0.0, 1.0 - d_prop / slot_ms)
            r_spare = max(0.0, eff_cap - current_W[r])
            if r_spare <= EPS:
                continue
                
            eff_cap_raw[r] = r_spare
            delay_penalty = d_prop * w3
            acc = 0.0
            if (fnr_matrix is not None and fpr_matrix is not None
                    and si is not None and id_to_idx is not None):
                ri  = id_to_idx[r]
                acc = (
                    (float(fnr_matrix[si, si]) - float(fnr_matrix[si, ri])) * w1
                    + (float(fpr_matrix[si, si]) - float(fpr_matrix[si, ri])) * w2
                )
            # score = acc - delay_penalty
            # score > 0 means accuracy improvement exceeds delay cost.
            raw[r] = acc - delay_penalty

        if not raw:
            continue

        if is_overloaded:
            min_sc = min(raw.values())
            pos = {r: (sc - min_sc + EPS) for r, sc in raw.items()}
        else:
            pos = {r: sc for r, sc in raw.items() if sc > 0.0}

        total = sum(pos.values())
        if total < EPS:
            continue

        items = list(pos.items())
        random.shuffle(items)
        sorted_items = sorted(items, key=lambda kv: -kv[1])
        for r, p in sorted_items:
            if p < EPS or to_move <= EPS:
                break
            ceiling = eff_cap_raw.get(r, 0.0)
            share = min((p / total) * to_move, ceiling)
            if share <= EPS:
                continue
            flow_float[src][src] -= share
            flow_float[src][r]    = flow_float[src].get(r, 0.0) + share
            to_move              -= share
            current_W[r]         += share
            eff_cap_raw[r]       -= share

        if to_move > EPS:
            for r, _ in sorted_items:
                if to_move <= EPS:
                    break
                ceiling = eff_cap_raw.get(r, 0.0)
                extra = min(ceiling, to_move)
                if extra <= EPS:
                    continue
                flow_float[src][src] -= extra
                flow_float[src][r]    = flow_float[src].get(r, 0.0) + extra
                to_move              -= extra
                current_W[r]         += extra
                eff_cap_raw[r]       -= extra

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)


def score_based_offload(
    area_ids: List[Any],
    W_src: Dict[Any, float],
    c_dst: Dict[Any, float],
    propagation_delays: Dict[Any, Dict[Any, float]],
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.01),
    fnr_matrix: Optional[np.ndarray] = None,
    fpr_matrix: Optional[np.ndarray] = None,
    id_to_idx: Optional[Dict[Any, int]] = None,
    cap_dst: Optional[Dict[Any, float]] = None,
    max_prop_delay: float = 1e9,
    slot_ms: float = 200.0,
) -> OffloadPlan:
    """
    Score-based offloading with Fair Share balancing.
    Goal: Minimize delay by balancing utilization and maximize accuracy via scores.
    """
    w1, w2, w3 = weights

    W_tot = float(sum(W_src[e] for e in area_ids))
    C_tot = float(sum(c_dst[e] for e in area_ids))
    W     = {e: float(W_src[e]) for e in area_ids}

    if W_tot <= EPS:
        flow = {e: {e: int(round(W[e]))} for e in area_ids}
        return _build_plan(flow, area_ids)

    # 1. Calculate Fair Share (Target utilization)
    if C_tot > EPS:
        # Every node targets the same utilization ratio (W_tot / C_tot)
        fair_share = {e: float(c_dst[e]) * (W_tot / C_tot) for e in area_ids}
    else:
        fair_share = {e: W_tot / len(area_ids) for e in area_ids}

    # Senders: All nodes can be senders (proactive)
    supply = {e: W[e] for e in area_ids}
    
    senders = [e for e in area_ids if supply[e] > EPS]

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    current_W = {e: W[e] for e in area_ids}
    
    # Track fair share demand: target - current
    def get_fair_demand():
        return {r: max(0.0, fair_share[r] - current_W[r]) for r in area_ids}

    for src in senders:
        si = id_to_idx[src] if id_to_idx is not None else None
        
        raw: Dict[Any, float] = {}
        eff_cap_raw: Dict[Any, float] = {}
        for r in area_ids:
            if r == src: continue
            d_prop = float(propagation_delays[src][r])
            if d_prop > max_prop_delay: continue
            
            # Effective capacity at receiver r for tasks from sender src
            eff_cap = float(c_dst[r]) * max(0.0, 1.0 - d_prop / slot_ms)
            r_spare = max(0.0, eff_cap - current_W[r])
            if r_spare <= EPS: continue
            
            eff_cap_raw[r] = r_spare
            delay_penalty = d_prop * w3
            acc = 0.0
            if (fnr_matrix is not None and fpr_matrix is not None
                    and si is not None and id_to_idx is not None):
                ri  = id_to_idx[r]
                acc = (
                    (float(fnr_matrix[si, si]) - float(fnr_matrix[si, ri])) * w1
                    + (float(fpr_matrix[si, si]) - float(fpr_matrix[si, ri])) * w2
                )
            raw[r] = acc - delay_penalty

        if not raw: continue

        # --- Phase 1: Proactive Accuracy Offloading (Score > 0) ---
        # Move workload to nodes that are strictly BETTER than local, 
        # up to their physical capacity (time-window adjusted).
        pos = {r: sc for r, sc in raw.items() if sc > 0.0}
        total_pos = sum(pos.values())
        if total_pos > EPS:
            items = list(pos.items())
            random.shuffle(items)
            sorted_items = sorted(items, key=lambda kv: -kv[1])
            initial_w = flow_float[src][src]
            for r, p in sorted_items:
                to_move = flow_float[src][src]
                if to_move <= EPS: break
                
                # Accuracy moves can fill nodes beyond fair_share if they are better
                ceiling = eff_cap_raw.get(r, 0.0)
                share = min((p / total_pos) * initial_w, ceiling, to_move)
                if share <= EPS: continue
                
                flow_float[src][src] -= share
                flow_float[src][r]    = flow_float[src].get(r, 0.0) + share
                current_W[r]         += share
                eff_cap_raw[r]       -= share

        # --- Phase 2: Load Balancing (W > Fair Share) ---
        # Move surplus to reach fair_share, even if scores are negative.
        target = fair_share[src]
        if flow_float[src][src] > target + EPS:
            surplus = flow_float[src][src] - target
            
            # Only consider receivers (nodes below their fair share)
            rem_demand = get_fair_demand()
            bal_targets = {r: sc for r, sc in raw.items() if r in rem_demand and rem_demand[r] > EPS}
            if bal_targets:
                min_sc = min(bal_targets.values())
                # Shift scores to be positive for proportional distribution
                pos_bal = {r: (sc - min_sc + EPS) for r, sc in bal_targets.items()}
                total_bal = sum(pos_bal.values())
                
                items = list(pos_bal.items())
                random.shuffle(items)
                sorted_items = sorted(items, key=lambda kv: -kv[1])
                for r, p in sorted_items:
                    to_move_bal = flow_float[src][src] - target
                    if to_move_bal <= EPS: break
                    
                    # Balancing moves are restricted by the receiver's fair_share demand
                    ceiling = min(eff_cap_raw.get(r, 0.0), rem_demand.get(r, 0.0))
                    share = min((p / total_bal) * surplus, ceiling, to_move_bal)
                    if share <= EPS: continue
                    
                    flow_float[src][src] -= share
                    flow_float[src][r]    = flow_float[src].get(r, 0.0) + share
                    current_W[r]         += share
                    rem_demand[r]        -= share
                    eff_cap_raw[r]       -= share

            # --- Phase 3: Survival Fallback (W > Capacity) ---
            # If still above physical capacity, dump to anywhere with spare cycles.
            local_cap = float(c_dst[src])
            if flow_float[src][src] > local_cap + EPS:
                # Same as Phase 2 but uses physical spare capacity as ceiling
                all_items = sorted(raw.items(), key=lambda x: -x[1])
                for r, _ in all_items:
                    to_move_emergency = flow_float[src][src] - local_cap
                    if to_move_emergency <= EPS: break
                    
                    ceiling = eff_cap_raw.get(r, 0.0)
                    share = min(ceiling, to_move_emergency)
                    if share <= EPS: continue
                    
                    flow_float[src][src] -= share
                    flow_float[src][r]    = flow_float[src].get(r, 0.0) + share
                    current_W[r]         += share
                    eff_cap_raw[r]       -= share

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)

# ---------------------------------------------------------------------------
# Backward-compat alias (called by old environment code paths)
# ---------------------------------------------------------------------------

def balance_with_caps_and_prop_filter(
    area_ids: List[Any],
    edges: Optional[Dict[Any, Any]] = None,
    W_src: Optional[Dict[Any, float]] = None,
    c_dst: Optional[Dict[Any, float]] = None,
    kappa_min: Optional[float] = None,
    prop_delay: Optional[Dict[Tuple[Any, Any], float]] = None,
    tau_loc: Optional[Dict[Any, float]] = None,
) -> OffloadPlan:
    """Alias for balance_workload; edges / kappa_min / prop_delay / tau_loc ignored."""
    return balance_workload(area_ids, W_src, c_dst)
