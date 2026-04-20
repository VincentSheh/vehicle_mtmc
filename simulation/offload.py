from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np


EPS = 1e-9
D_MAX_MS = 200.0

@dataclass
class OffloadPlan:
    # flow counts, NOT fractions
    flow: Dict[int, Dict[int, int]]          # flow[src_area_id][dst_area_id] = n_tasks
    assigned_dst: Dict[int, int]             # assigned_dst[dst_area_id] = total tasks executed at dst


def round_flow_to_int(flow_float: Dict[int, Dict[int, float]]) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {}
    for s, mp in flow_float.items():
        out[s] = {}
        for d, v in mp.items():
            out[s][d] = int(round(v))
    return out

def compute_N_max_delay_cap(area_ids, edges, c_va_dst, kappa_min):
    N_max = {}
    for eid in area_ids:
        edge = edges[eid]
        mu_cycles_per_ms = edge.cpu_cycle_per_ms * float(c_va_dst[eid])
        cap = (mu_cycles_per_ms * D_MAX_MS) / max(kappa_min, 1e-9)
        N_max[eid] = max(0, int(math.floor(cap)))
        N_max[eid] = 1e9
    return N_max

def balance_with_caps_and_prop_filter(
    area_ids: List[int],
    edges: List[int],
    W_src: Dict[int, float],
    c_dst: Dict[int, float],
    kappa_min: float,
    prop_delay: Optional[Dict[Tuple[int, int], float]] = None,
    tau_loc: Optional[Dict[int, float]] = None,
) -> OffloadPlan:
    """
    Compute offloading flows that try to balance service share c_dst / N_dst,
    subject to receiver cap N_dst <= floor(c_dst / s_min) and propagation filter:
    do not offload src->dst if d(src,dst) > tau_loc[src].

    Complexity: O(|E| + |F|) assuming the feasible receiver list per sender is short.
    If you allow all pairs, it becomes O(|E|^2).
    """
    prop_delay = prop_delay or {}
    tau_loc = tau_loc or {}

    # 1) totals
    W_tot = float(sum(W_src[e] for e in area_ids))
    C_tot = float(sum(c_dst[e] for e in area_ids))

    # no load or no capacity
    if W_tot <= EPS or C_tot <= EPS:
        flow = {e: {e: int(round(W_src[e]))} for e in area_ids}
        assigned = {e: int(round(W_src[e])) for e in area_ids}
        return OffloadPlan(flow=flow, assigned_dst=assigned)

    # 2) balanced targets N*_e
    # N*_e = c_e * W_tot / C_tot
    N_star = {e: float(c_dst[e]) * (W_tot / C_tot) for e in area_ids}

    # 3) receiver caps from delay bound
    N_max = compute_N_max_delay_cap(area_ids, edges, c_dst, kappa_min)

    # current present workload (what src "owns" before offload)
    W = {e: float(W_src[e]) for e in area_ids}

    # headroom if this edge were to receive tasks
    # headroom = {e: max(0.0, float(N_max[e]) - W[e]) for e in area_ids}
    headroom = {e: 1e6 for e in area_ids} #! No cap for receiving task

    # 4) supply and demand relative to balanced target, demand capped by headroom
    supply = {e: max(0.0, W[e] - N_star[e]) for e in area_ids}
    demand = {e: min(max(0.0, N_star[e] - W[e]), headroom[e]) for e in area_ids}

    senders = [e for e in area_ids if supply[e] > 0.0]
    receivers = [e for e in area_ids if demand[e] > 0.0]

    # initialize all local
    flow_float: Dict[int, Dict[int, float]] = {e: {e: W[e]} for e in area_ids}

    # 5) build feasible receiver list per sender with propagation filter
    # Feasible if receiver has remaining demand and propagation delay does not exceed local proxy time.
    # If tau_loc missing for a sender, we treat it as +inf (allow).
    feasible_receivers: Dict[int, List[int]] = {}
    for s in senders:
        tau_s = float(tau_loc.get(s, float("inf")))
        lst = []
        for r in receivers:
            d_sr = float(prop_delay.get((s, r), 0.0))
            if d_sr <= tau_s:
                lst.append(r)
        # simple heuristic, keep receiver order as given
        feasible_receivers[s] = lst

    # 6) greedy matching using two pointers but restricted by feasible sets
    # This stays near O(|E|) if each sender has a small feasible list.
    rem_supply = {e: supply[e] for e in senders}
    rem_demand = {e: demand[e] for e in receivers}

    for s in senders:
        s_rem = rem_supply[s]
        if s_rem <= 0:
            continue

        for r in feasible_receivers.get(s, []):
            r_rem = rem_demand.get(r, 0.0)
            if r_rem <= 0.0:
                continue

            u = min(s_rem, r_rem)
            if u <= 0.0:
                continue

            # move u from s to r
            flow_float[s][s] -= u
            flow_float[s][r] = flow_float[s].get(r, 0.0) + u

            s_rem -= u
            rem_demand[r] -= u

            if s_rem <= 0.0:
                break

        rem_supply[s] = s_rem

    # 7) convert to integer flow counts, then fix conservation per sender by adjusting diagonal
    flow = round_flow_to_int(flow_float)
    for s in area_ids:
        sent = sum(flow[s].values())
        want = int(round(W[s]))
        if sent != want:
            flow[s][s] = flow[s].get(s, 0) + (want - sent)

    assigned_dst: Dict[int, int] = {e: 0 for e in area_ids}
    for s in area_ids:
        for d, n in flow[s].items():
            assigned_dst[d] += int(n)

    return OffloadPlan(flow=flow, assigned_dst=assigned_dst)


def sp2_inspection_offload(
    area_ids: List[str],
    edges: Dict[str, Any],
    W_src: Dict[str, float],
    fnr_matrix: np.ndarray,
    fpr_matrix: np.ndarray,
    prop_delay: Dict[Tuple[str, str], float],
    ids_util_prev: Dict[str, float],
    weights: Tuple[float, float, float, float],
    id_to_idx: Dict[str, int],
) -> OffloadPlan:
    """
    SP2 Phase 1: Accuracy-Aware Inspection Offloading.

    Routes each source edge's inspection workload across feasible receivers
    proportionally to their suitability score:
      Score(e, e') = -w1*FNR[e,e'] - w2*FPR[e,e'] - w3*d_prop(e,e') - w4*rho(e')

    Feasibility: e' is eligible only if it has residual IDS capacity.
    Allocation: softmax over max(0, Score); if all scores <= 0, keep local.
    """
    omega1, omega2, omega3, omega4 = weights

    # IDS capacity (packets/step) and current assigned load at each edge
    mu_def = {
        eid: edges[eid].ids.effective_speed_pkt_per_step(edges[eid].ids_cpu)
        for eid in area_ids
    }
    Lambda_in = {eid: float(W_src[eid]) for eid in area_ids}

    # Initialize all traffic kept local
    flow_float: Dict[str, Dict[str, float]] = {e: {e: float(W_src[e])} for e in area_ids}

    for src in area_ids:
        si = id_to_idx[src]
        w = float(W_src[src])
        if w <= EPS:
            continue

        # Step 1: feasible receivers (residual IDS capacity > 0)
        feasible = [ep for ep in area_ids if mu_def[ep] - Lambda_in[ep] > 0.0]
        if not feasible:
            continue

        # Step 2: score each feasible receiver
        raw_scores: Dict[str, float] = {}
        for dst in feasible:
            di = id_to_idx[dst]
            fnr = float(fnr_matrix[si, di])
            fpr = float(fpr_matrix[si, di])
            d_prop = float(prop_delay.get((src, dst), 0.0))
            rho = float(ids_util_prev.get(dst, 0.0))
            raw_scores[dst] = -omega1 * fnr - omega2 * fpr - omega3 * d_prop - omega4 * rho

        # Step 3: proportional allocation over positive scores only
        pos_scores = {ep: max(0.0, s) for ep, s in raw_scores.items()}
        total_pos = sum(pos_scores.values())

        if total_pos <= EPS:
            continue

        # Overwrite local-only initialisation with scored fractions
        flow_float[src] = {dst: (pos_scores[dst] / total_pos) * w for dst in feasible}

    # Round to integers and conserve per-source total
    flow = round_flow_to_int(flow_float)
    for s in area_ids:
        sent = sum(flow.get(s, {}).values())
        want = int(round(W_src[s]))
        diff = want - sent
        if diff != 0:
            flow.setdefault(s, {})[s] = flow[s].get(s, 0) + diff

    assigned_dst: Dict[str, int] = {e: 0 for e in area_ids}
    for s in area_ids:
        for d, n_tasks in flow.get(s, {}).items():
            assigned_dst[d] = assigned_dst.get(d, 0) + int(n_tasks)

    return OffloadPlan(flow=flow, assigned_dst=assigned_dst)

