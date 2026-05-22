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
        
        # Correct physical capacity check:
        # W_local + W_offloaded/(1-d/tau) <= C_raw
        # => W_offloaded <= (Raw_Cap - W_local) * (1 - d/tau)
        r_spare = (float(N_star[r]) - current_W[r]) * max(0.0, 1.0 - d_prop / slot_ms)
        r_demand = max(0.0, r_spare)
        
        if r_demand <= EPS:
            continue
            
        u = min(rem_supply[s], r_demand)
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        rem_supply[s]    -= u
        
        # Cycle-aware update: remote tasks are "heavier"
        window = max(EPS, 1.0 - d_prop / slot_ms)
        current_W[r]     += u / window
        current_W[s]     -= u # Update sender workload (local cycles saved)

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
    max_prop_delay: float = 1e9,
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
            if d_prop > max_prop_delay:
                continue

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
            
        # Correct physical capacity check:
        # W_local + W_offloaded/(1-d/tau) <= C_raw
        # => W_offloaded <= (Raw_Cap - W_local) * (1 - d/tau)
        r_spare = (float(N_star[r]) - current_W[r]) * max(0.0, 1.0 - d_prop / slot_ms)
        r_demand = max(0.0, r_spare)

        if r_demand <= EPS:
            continue
            
        u = min(rem_supply[s], r_demand)
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        rem_supply[s]    -= u
        
        # Cycle-aware update: remote tasks are "heavier"
        window = max(EPS, 1.0 - d_prop / slot_ms)
        current_W[r]     += u / window
        current_W[s]     -= u # Update sender workload (local cycles saved)

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)


def balance_workload_cto_acc_rnd(
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
    CTO with randomized accuracy-aware link scoring: similar to cto_acc,
    but instead of using the exact remote accuracy, it uses the AVERAGE 
    FPR and FNR of all local models sampled for this run.
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

    # Calculate per-source average of cross-edge (non-diagonal) FPR/FNR
    avg_fpr_per_src = {}
    avg_fnr_per_src = {}
    if fpr_matrix is not None and fnr_matrix is not None:
        n = fpr_matrix.shape[0]
        for i in range(n):
            # Mask the diagonal to get only cross-edge performance
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            avg_fpr_per_src[i] = float(np.mean(fpr_matrix[i, mask]))
            avg_fnr_per_src[i] = float(np.mean(fnr_matrix[i, mask]))

    scored_links: List[Tuple[float, Any, Any, float]] = []
    for s in senders:
        si = id_to_idx[s] if id_to_idx is not None else None
        for r in receivers:
            if s == r:
                continue
            d_prop = float(propagation_delays[s][r])
            if d_prop > max_prop_delay:
                continue

            delay_penalty = d_prop * w3

            # Improvement (acc) calculation: (Local_Rate - Remote_Rate)
            # Remote_Rate is the average of how other models perform on this data.
            acc = 0.0
            if (fnr_matrix is not None and fpr_matrix is not None
                    and si is not None):
                acc = (
                    (float(fnr_matrix[si, si]) - avg_fnr_per_src[si]) * w1
                    + (float(fpr_matrix[si, si]) - avg_fpr_per_src[si]) * w2
                )

            # cost = delay_penalty - acc
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
            
        r_spare = (float(N_star[r]) - current_W[r]) * max(0.0, 1.0 - d_prop / slot_ms)
        r_demand = max(0.0, r_spare)

        if r_demand <= EPS:
            continue
            
        u = min(rem_supply[s], r_demand)
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        rem_supply[s]    -= u
        
        # Cycle-aware update: remote tasks are "heavier"
        window = max(EPS, 1.0 - d_prop / slot_ms)
        current_W[r]     += u / window
        current_W[s]     -= u 

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
    max_prop_delay: float = 1e9,
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
            if d_prop > max_prop_delay:
                continue

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
            
        # Correct physical capacity check:
        # W_local + W_offloaded/(1-d/tau) <= C_raw
        # => W_offloaded <= (Raw_Cap - W_local) * (1 - d/tau)
        r_spare = (float(N_star[r]) - current_W[r]) * max(0.0, 1.0 - d_prop / slot_ms)
        r_demand = max(0.0, r_spare)

        if r_demand <= EPS:
            continue
            
        u = min(rem_supply[s], r_demand)
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        rem_supply[s]    -= u
        
        # Cycle-aware update: remote tasks are "heavier"
        window = max(EPS, 1.0 - d_prop / slot_ms)
        current_W[r]     += u / window
        current_W[s]     -= u # Update sender workload (local cycles saved)

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)


# ---------------------------------------------------------------------------
# Mode b4 — PD-BTO (Propagation-Delay-Based Balanced Task Offloading)
# ---------------------------------------------------------------------------
# Algorithm 3: balances load ratios (W/C) across nodes instead of just
# offloading overflow. Uses propagation delay as the sorting cost.
# ---------------------------------------------------------------------------

def pd_bto(
    area_ids: List[Any],
    W_src: Dict[Any, float],
    c_dst: Dict[Any, float],
    propagation_delays: Dict[Any, Dict[Any, float]],
    max_prop_delay: float = 1e9,
    slot_ms: float = 200.0,
) -> OffloadPlan:
    W_tot = float(sum(W_src[e] for e in area_ids))
    C_tot = float(sum(c_dst[e] for e in area_ids))
    W     = {e: float(W_src[e]) for e in area_ids}

    if W_tot <= EPS or C_tot <= EPS:
        flow = {e: {e: int(round(W[e]))} for e in area_ids}
        return _build_plan(flow, area_ids)

    # Load ratios
    ell = {e: W[e] / max(EPS, float(c_dst[e])) for e in area_ids}

    # Build candidate links: sender → receiver where sender is more loaded
    all_links: List[Tuple[float, Any, Any, float]] = []
    for s in area_ids:
        for r in area_ids:
            if s == r:
                continue
            d_prop = float(propagation_delays[s][r])
            if d_prop > max_prop_delay:
                continue
            delta = max(0.0, 1.0 - d_prop / slot_ms)
            if delta <= EPS:
                continue
            # Only consider if sender is more loaded than receiver
            # after accounting for the delay tax
            if ell[s] > ell[r]:
                all_links.append((d_prop, s, r, delta))

    all_links.sort(key=lambda x: x[0])

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    current_W = {e: float(W[e]) for e in area_ids}

    for d_prop, s, r, delta in all_links:
        C_s = max(EPS, float(c_dst[s]))
        C_r = max(EPS, float(c_dst[r]))
        
        u_s = current_W[s] / C_s
        u_r = current_W[r] / C_r
        
        # Delay tax: each offloaded task costs 1/(delta * C_r) utilization at receiver
        delay_tax = 1.0 / (delta * C_r)
        
        # Only offload if sender util > receiver util + delay tax per task
        if u_s <= u_r + delay_tax:
            continue
        
        # How many tasks can we move before utilizations equalize (accounting for delay)?
        # Moving u tasks: sender util becomes (W_s - u) / C_s
        #                 receiver util becomes (W_r + u/delta) / C_r
        # Equalize: (W_s - u) / C_s = (W_r + u/delta) / C_r
        # => u = (W_s/C_s - W_r/C_r) / (1/C_s + 1/(delta * C_r))
        u_max = (u_s - u_r) / (1.0/C_s + 1.0/(delta * C_r))
        u_max = max(0.0, u_max)
        
        # Also cap at physical spare at receiver
        phys_spare = max(0.0, float(c_dst[r]) - current_W[r]) * delta
        
        u = min(u_max, phys_spare)
        if u <= EPS:
            continue
        
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        current_W[s]     -= u
        current_W[r]     += u / delta

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)

# ---------------------------------------------------------------------------
# Mode c & d — score-based routing
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
            
            # Correct physical capacity check:
            # W_local + W_offloaded/(1-d/tau) <= C_raw
            # => W_offloaded <= (Raw_Cap - W_local) * (1 - d/tau)
            r_spare = (float(c_dst[r]) - current_W[r]) * max(0.0, 1.0 - d_prop / slot_ms)
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
            current_W[src]       -= share # Update sender workload
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
                current_W[src]       -= extra # Update sender workload
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
    balance_floor: float = 0.5,
) -> OffloadPlan:
    """
    Score-based offloading (Delay Workload) with Fair Share balancing.
    Goal: Minimize delay by balancing utilization. Accuracy is NOT considered.

    balance_floor: per-link proactive threshold scales with the offload tax τ = d_prop/slot_ms:

        threshold(src→r) = balance_floor + (1 − balance_floor) × τ

    A sender at utilisation u_src may only proactively offload to receiver r when
    u_src > threshold(src→r). Near receivers (small τ, small tax) are engaged early;
    distant receivers (large τ, large tax) are only used when the sender is genuinely
    stressed. Phase 2 emergency overflow is always unconditional.
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
        # but we MUST NOT target more than 100% utilization in the balancing phase.
        util_target = min(1.0, W_tot / C_tot)
        fair_share = {e: float(c_dst[e]) * util_target for e in area_ids}
    else:
        fair_share = {e: W_tot / len(area_ids) for e in area_ids}

    # Senders: Nodes that have work
    supply = {e: W[e] for e in area_ids}
    
    senders = [e for e in area_ids if supply[e] > EPS]

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    current_W = {e: W[e] for e in area_ids}
    
    # Track fair share demand: target - current
    def get_fair_demand():
        return {r: max(0.0, fair_share[r] - current_W[r]) for r in area_ids}

    for src in senders:
        # Link scoring based PURELY on negative delay penalty
        raw: Dict[Any, float] = {}
        eff_cap_raw: Dict[Any, float] = {}
        for r in area_ids:
            if r == src: continue
            d_prop = float(propagation_delays[src][r])
            if d_prop > max_prop_delay: continue
            
            # Correct physical capacity check
            r_spare = (float(c_dst[r]) - current_W[r]) * max(0.0, 1.0 - d_prop / slot_ms)
            if r_spare <= EPS: continue
            
            eff_cap_raw[r] = r_spare
            delay_penalty = d_prop * w3
            raw[r] = -delay_penalty

        if not raw: continue

        # --- Phase 1: Load Balancing (W > Fair Share) ---
        # Per-link threshold: threshold(src→r) = balance_floor + (1−balance_floor) × τ
        # where τ = d_prop/slot_ms is the offload tax for that link.
        # Near receivers (low tax) are engaged early; distant receivers only when stressed.
        util_src = W[src] / max(EPS, float(c_dst[src]))
        target = fair_share[src]
        if flow_float[src][src] > target + EPS:
            rem_demand = get_fair_demand()
            # Include a receiver only when sender utilisation exceeds its per-link threshold.
            bal_targets = {}
            for r, sc in raw.items():
                if rem_demand.get(r, 0.0) <= EPS:
                    continue
                tau = float(propagation_delays[src][r]) / max(EPS, slot_ms)
                link_threshold = balance_floor + (1.0 - balance_floor) * tau
                if util_src > link_threshold:
                    bal_targets[r] = sc
            
            while bal_targets and flow_float[src][src] > target + EPS:
                surplus = flow_float[src][src] - target
                min_sc = min(bal_targets.values())
                # Shift scores to be positive for proportional distribution
                pos_bal = {r: (sc - min_sc + EPS) for r, sc in bal_targets.items()}
                total_bal = sum(pos_bal.values())
                
                # In each iteration, we try to move work to all candidates
                next_bal_targets = {}
                any_moved = False
                
                # To avoid ordering bias, shuffle but maintain the proportional logic
                target_ids = list(bal_targets.keys())
                random.shuffle(target_ids)
                
                for r in target_ids:
                    p = pos_bal[r]
                    d_p = float(propagation_delays[src][r])
                    window = max(EPS, 1.0 - d_p / slot_ms)
                    
                    # Target share based on this iteration's surplus
                    share = (p / total_bal) * surplus
                    
                    # Ceiling: physical room within fair_share
                    ceiling = min(eff_cap_raw.get(r, 0.0), rem_demand.get(r, 0.0) * window)
                    
                    if ceiling <= EPS:
                        continue
                    
                    actual_move = min(share, ceiling)
                    if actual_move <= EPS:
                        continue
                        
                    flow_float[src][src] -= actual_move
                    flow_float[src][r]    = flow_float[src].get(r, 0.0) + actual_move
                    
                    # Cycle-aware update
                    current_W[r]         += actual_move / window
                    current_W[src]       -= actual_move
                    rem_demand[r]        -= actual_move / window
                    eff_cap_raw[r]       -= actual_move
                    any_moved = True
                    
                    # If we didn't hit the ceiling, it might still take more in next round
                    if ceiling - actual_move > EPS:
                        next_bal_targets[r] = bal_targets[r]
                
                if not any_moved: break
                bal_targets = next_bal_targets

            # --- Phase 2: Survival Fallback (W > Capacity) ---
            # Emergency dumping uses Greedy Fill to prioritize efficiency during saturation.
            local_cap = float(c_dst[src])
            if flow_float[src][src] > local_cap + EPS:
                all_items = sorted(raw.items(), key=lambda x: -x[1])
                for r, _ in all_items:
                    to_move_emergency = flow_float[src][src] - local_cap
                    if to_move_emergency <= EPS: break
                    
                    d_p = float(propagation_delays[src][r])
                    window = max(EPS, 1.0 - d_p / slot_ms)
                    
                    ceiling = eff_cap_raw.get(r, 0.0)
                    share = min(ceiling, to_move_emergency)
                    if share <= EPS: continue
                    
                    flow_float[src][src] -= share
                    flow_float[src][r]    = flow_float[src].get(r, 0.0) + share
                    
                    current_W[r]         += share / window
                    current_W[src]       -= share
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
