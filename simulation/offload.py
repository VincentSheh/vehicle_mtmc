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
) -> OffloadPlan:
    """
    CTO: proportional redistribution target, then greedy forwarding over
    sender→receiver links sorted by ascending propagation delay.
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
    demand    = {e: max(0.0, N_star[e] - W[e]) for e in area_ids}
    senders   = [e for e in area_ids if supply[e] > EPS]
    receivers = [e for e in area_ids if demand[e] > EPS]

    all_links: List[Tuple[Any, Any, float]] = [
        (s, r, propagation_delays[s][r])
        for s in senders
        for r in receivers
        if s != r
    ]
    random.shuffle(all_links)
    all_links.sort(key=lambda x: x[2])

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    rem_supply = {e: supply[e] for e in senders}
    rem_demand = {e: demand[e] for e in receivers}

    for s, r, _ in all_links:
        if rem_supply[s] <= EPS or rem_demand[r] <= EPS:
            continue
        u = min(rem_supply[s], rem_demand[r])
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        rem_supply[s]    -= u
        rem_demand[r]    -= u

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
    demand    = {e: max(0.0, N_star[e] - W[e]) for e in area_ids}
    senders   = [e for e in area_ids if supply[e] > EPS]
    receivers = [e for e in area_ids if demand[e] > EPS]

    scored_links: List[Tuple[float, Any, Any]] = []
    for s in senders:
        si = id_to_idx[s] if id_to_idx is not None else None
        for r in receivers:
            if s == r:
                continue
            d_prop = float(propagation_delays[s][r])
            delay_penalty = d_prop * w3

            acc = 0.0
            if (fnr_matrix is not None and fpr_matrix is not None
                    and si is not None and id_to_idx is not None):
                ri = id_to_idx[r]
                acc = (
                    (float(fnr_matrix[si, si]) - float(fnr_matrix[si, ri])) * w1
                    + (float(fpr_matrix[si, si]) - float(fpr_matrix[si, ri])) * w2
                )

            # cost = delay_penalty - acc:
            # lower cost = better link (closer + more accurate)
            # with w1=w2=0 this reduces to ascending delay, identical to plain cto
            cost = delay_penalty - acc
            scored_links.append((cost, s, r))

    random.shuffle(scored_links)
    scored_links.sort(key=lambda x: x[0])

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    rem_supply = {e: supply[e] for e in senders}
    rem_demand = {e: demand[e] for e in receivers}

    for _, s, r in scored_links:
        if rem_supply[s] <= EPS or rem_demand[r] <= EPS:
            continue
        u = min(rem_supply[s], rem_demand[r])
        flow_float[s][s] -= u
        flow_float[s][r]  = flow_float[s].get(r, 0.0) + u
        rem_supply[s]    -= u
        rem_demand[r]    -= u

    return _build_plan(_round_and_conserve(flow_float, W), area_ids)


# ---------------------------------------------------------------------------
# Modes c & d — score-based routing
# ---------------------------------------------------------------------------
# Identical to balance_workload_cto_acc except traffic is distributed
# proportionally to link scores instead of greedy winner-takes-all sorting.
#
# Link score (src → dst):
#   score = acc_improvement(src, dst)   [mode d, via w1/w2]
#         - d_prop(src, dst) * w3       [modes c & d]
#
#   acc_improvement = (FNR[si,si] - FNR[si,di]) * w1
#                   + (FPR[si,si] - FPR[si,di]) * w2
#
# For each sender, supply is split across eligible receivers proportional to
# max(0, score). If all scores ≤ 0 the sender keeps all traffic locally.
# ---------------------------------------------------------------------------

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
) -> OffloadPlan:
    """
    Score-based offload (modes c / d): same redistribution target and link
    scoring as balance_workload_cto_acc, but each sender splits its supply
    proportionally to receiver scores rather than greedily in sorted order.
    """
    w1, w2, w3 = weights

    W_tot = float(sum(W_src[e] for e in area_ids))
    W     = {e: float(W_src[e]) for e in area_ids}

    if W_tot <= EPS:
        flow = {e: {e: int(round(W[e]))} for e in area_ids}
        return _build_plan(flow, area_ids)

    C_tot = float(sum(c_dst[e] for e in area_ids))
    if C_tot > EPS:
        N_star = {e: float(c_dst[e]) * (W_tot / C_tot) for e in area_ids}
    else:
        N_star = {e: 0.0 for e in area_ids}
    # cap each edge at its own capacity to prevent overcapacity when W_tot > C_tot
    N_star = {e: min(N_star[e], float(c_dst[e])) for e in area_ids}

    supply    = {e: max(0.0, W[e] - N_star[e]) for e in area_ids}
    demand    = {e: max(0.0, N_star[e] - W[e]) for e in area_ids}
    senders   = [e for e in area_ids if supply[e] > EPS]
    receivers = [e for e in area_ids if demand[e] > EPS]

    flow_float: Dict[Any, Dict[Any, float]] = {e: {e: W[e]} for e in area_ids}
    rem_supply = {e: supply[e] for e in senders}
    rem_demand = {e: demand[e] for e in receivers}

    for src in senders:
        si      = id_to_idx[src] if id_to_idx is not None else None
        to_move = rem_supply[src]
        if to_move <= EPS:
            continue

        raw: Dict[Any, float] = {}
        for r in receivers:
            if r == src or rem_demand.get(r, 0.0) <= EPS:
                continue
            d_prop        = float(propagation_delays[src][r])
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

        if not raw:
            continue
        min_sc = min(raw.values())
        pos    = {r: (sc - min_sc + EPS) for r, sc in raw.items()}
        total  = sum(pos.values())
        if total <= EPS:
            continue

        items = list(pos.items())
        random.shuffle(items)
        sorted_items = sorted(items, key=lambda kv: -kv[1])
        for r, p in sorted_items:
            if p <= EPS or to_move <= EPS:
                break
            share = min((p / total) * rem_supply[src], rem_demand.get(r, 0.0), to_move)
            if share <= EPS:
                continue
            flow_float[src][src] -= share
            flow_float[src][r]    = flow_float[src].get(r, 0.0) + share
            to_move              -= share
            rem_demand[r]        -= share

        # Drain any supply stranded by demand caps in the proportional pass
        if to_move > EPS:
            for r, _ in sorted_items:
                if to_move <= EPS:
                    break
                extra = min(rem_demand.get(r, 0.0), to_move)
                if extra <= EPS:
                    continue
                flow_float[src][src] -= extra
                flow_float[src][r]    = flow_float[src].get(r, 0.0) + extra
                to_move              -= extra
                rem_demand[r]        -= extra

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
