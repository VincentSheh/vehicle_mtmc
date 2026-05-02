from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import json
import re
import torch

from pathlib import Path
import yaml
import os
import copy
import numpy as np
import pandas as pd

from service import IDS, VideoPipeline

from request import User, Attacker, AttackTypeLibrary

from edgearea import ResourceBudget, EdgeArea

from offload import (
    OffloadPlan,
    no_offload,
    balance_workload,
    score_based_offload,
    balance_workload_cto,
    balance_workload_cto_acc,
    balance_workload_cto_acc_inv,
    balance_with_caps_and_prop_filter,
)


# CPU cores reserved for VA — must be consistent across IDS min, IDS max, and VA floor.
VA_CPU_RESERVE: float = 0.5

@dataclass(frozen=True)
class GlobalConfig:
    cpu_cycle_per_ms: float
    cpu_cores: int
    slot_ms: float
    ids_cycles_per_packet: float

@dataclass
class StepMetrics:
    t: int
    area_id: str

    # QoE
    qoe_mean: float
    qoe_mean_ideal: float
    benign_col_dmg: float

    # Requests (OD-only pipeline)
    local_num_req: int                  # served (after IDS + uplink + compute)
    ema: float
    ema_mom: float

    # IDS / attack
    ids_coverage: float
    attack_in_rate: float
    attack_drop_rate: float
    user_drop_rate: float
    cpu_to_ids_ratio: float
    ids_cpu_utilization: float

    # VA / BW
    va_cpu_utilization: float
    attack_cpu_frac: float           # fraction of total CPU budget consumed by attacker
    bw_utilization: float

    overhead: float

    # Multi-edge additions
    qoe_weighted: float = 0.0
    total_cpu_to_ids_ratio: float = 0.0

    # Per-step plan (detector mixing)
    od_plan: Dict[str, int] = field(default_factory=dict)

    # Optional network impact metric
    I_net: float = 0.0

    # Users routed to this edge's IDS after offloading (exec_user_in)
    ids_user_in_rate: float = 0.0

    
def load_globals(cfg: dict) -> GlobalConfig:
    g = cfg["globals"]

    cpu_cycle_per_ms = float(g["cpu_clock_cycle"])
    cpu_cores = int(g["cpu_cores"])

    fps = float(g.get("fps", 5.0))
    slot_ms = 1000.0 / fps

    ids_latency_ms = float(g["ids"]["latency_ms"])
    ids_cycles_per_packet = ids_latency_ms * cpu_cycle_per_ms * cpu_cores

    return GlobalConfig(
        cpu_cycle_per_ms=cpu_cycle_per_ms,
        cpu_cores=cpu_cores,
        slot_ms=slot_ms,
        ids_cycles_per_packet=ids_cycles_per_packet,
    )

class Environment:
    def __init__(
        self,
        edge_areas: List[EdgeArea],
        delay_ms: np.ndarray,
        t_max: int,
        seed: int = 0,
        offload_mode: str = "balance",
        va_attack_offload: bool = False,
    ):
        self.edge_areas = edge_areas
        self.delay_ms = np.asarray(delay_ms, dtype=np.float32)
        self.offload_mode = str(offload_mode)  # "none" | "balance" | "delay_workload" | "full"
        self.va_attack_offload = bool(va_attack_offload)
        self.t_max = int(t_max)
        self.t = 0
        self.history: List[StepMetrics] = []
        self.last_history: List[StepMetrics] = []
        self.final_qoe = 0

        # Multi-edge routing state
        self.area_ids: List[str] = [str(e.area_id) for e in self.edge_areas]
        self.id_to_idx: Dict[str, int] = {eid: i for i, eid in enumerate(self.area_ids)}

        n = len(self.edge_areas)
        assert self.delay_ms.shape == (n, n), f"delay_ms must be ({n},{n}), got {self.delay_ms.shape}"

        self.prop_delay: Dict[Tuple[str, str], float] = {}
        for s in self.area_ids:
            si = self.id_to_idx[s]
            for r in self.area_ids:
                ri = self.id_to_idx[r]
                self.prop_delay[(s, r)] = float(self.delay_ms[si, ri])

        self.rng = np.random.default_rng(seed)
        self.active_attack = None
        self.acc_by_region: Optional[np.ndarray] = None  # shape (n_edges, n_edges, 2): [src, exec, fpr/fnr]
        self._max_prop_delay_ms: float = 1e9
        self._offload_weights: Tuple[float, float, float] = (1.0, 1.0, 0.01)

        np.random.seed(seed)

        # Sample attack type probability matrix once per run (fixed across episodes)
        edges_with_lib = [e for e in self.edge_areas if e.attack_type_library is not None]
        self._p_attack_type_matrix = None
        self._dirichlet_alpha = None
        if edges_with_lib:
            n_types = edges_with_lib[0].attack_type_library.n_types
            n_edges = len(self.edge_areas)
            alpha = edges_with_lib[0].dirichlet_alpha
            self._dirichlet_alpha = alpha
            p_joint = self.rng.dirichlet(np.ones(n_types * n_edges) * alpha)
            p_matrix = p_joint.reshape(n_edges, n_types)
            self._p_attack_type_matrix = p_matrix / p_matrix.sum(axis=1, keepdims=True)
            header = "  ".join(f"type_{j:02d}" for j in range(n_types))
            # print(f"\n[Attack type probabilities per edge (alpha={alpha:.3f}, fixed for this run)]")
            # print(f"{'Edge':<12}  {header}")
            # for i, edge in enumerate(self.edge_areas):
            #     row = "  ".join(f"{self._p_attack_type_matrix[i, j]:.4f}" for j in range(n_types))
            #     print(f"{str(edge.area_id):<12}  {row}")

    def reset(self, seed):
        self.t = 0
        self.last_history = list(self.history)
        self.history.clear()
        self.final_qoe = 0

        # Resample accuracy matrix (clients + run) for this episode
        acc_mat_data = getattr(self, "_acc_mat_data", None)
        if acc_mat_data is not None:
            acc, _clients, _ak, _rk = _load_accuracy_matrix(
                acc_mat_data,
                self._acc_mat_alpha,
                self._acc_mat_model,
                len(self.edge_areas),
                np.random.default_rng(seed),
            )
            self.acc_by_region = acc

        p_matrix = self._p_attack_type_matrix
        for i, edge in enumerate(self.edge_areas):
            p = p_matrix[i] if p_matrix is not None else None
            edge.reset(seed=seed + i*100, p_attack_type=p)

    def _snapshot_edges(self):
        return [edge.get_state() for edge in self.edge_areas], np.random.get_state()

    def _restore_edges(self, snapshot):
        edge_states, rng_state = snapshot
        np.random.set_state(rng_state)
        for i, edge in enumerate(self.edge_areas):
            edge.set_state(edge_states[i])
    
    def step(self, ids_cpus, overhead=0):
        snapshot = self._snapshot_edges()

        # 1. ideal pass: attacks disabled, IDS CPU = 0 (max VA)
        ideal_cache = self._run_step_multi_edge(
            ids_cpus=[0.0] * len(self.edge_areas),
            overhead=overhead,
            disable_attack=True,
            forced_mode="cto",
        )

        # 2. restore pre-step state
        self._restore_edges(snapshot)

        # Observe arrivals for weighting and logging
        obs = {e.area_id: e.observe_arrivals(self.t) for e in self.edge_areas}

        # 3. real pass from same pre-step state
        real_cache = self._run_step_multi_edge(
            ids_cpus=ids_cpus,
            overhead=overhead,
            disable_attack=False,
        )

        edges_by_id = {e.area_id: e for e in self.edge_areas}
        # tot_req should be the total users arriving across all edges in this slot (source-centric)
        tot_req = sum(int(obs[aid]["user_req_in"]) for aid in self.area_ids)
        total_cpu_to_ids_ratio = sum(
            edges_by_id[aid].ids_cpu / edges_by_id[aid].budget.cpu for aid in self.area_ids
        )

        overheads = (
            [float(x) for x in overhead]
            if isinstance(overhead, (list, tuple, np.ndarray))
            else [float(overhead)] * len(self.edge_areas)
        )

        for i, edge in enumerate(self.edge_areas):
            aid = edge.area_id
            cache = real_cache[aid]
            cache_ideal = ideal_cache[aid]
            ids_out = cache["ids_out"]
            
            # n is the number of users who originated at this edge
            n = int(obs[aid]["user_req_in"])
            # Since we can't easily map executor QoE back to sources without per-flow tracking,
            # we use the executor's QoE as a proxy for global performance. 
            # FIX: If we want to evaluate offloading, we must use the executor's QoE 
            # but weight it by the work it actually performed.
            n_executed = int(cache.get("original_user_in", 0))
            qoe_weighted = float(cache["qoe"]) * (n_executed / tot_req) if tot_req > 0 else float(cache["qoe"])

            self.history.append(
                StepMetrics(
                    t=self.t,
                    area_id=aid,
                    qoe_mean=float(cache["qoe"]),
                    qoe_mean_ideal=float(cache_ideal["qoe"]),
                    benign_col_dmg=float(cache_ideal["qoe"] - cache["qoe"]),
                    qoe_weighted=qoe_weighted * len(self.edge_areas),
                    total_cpu_to_ids_ratio=total_cpu_to_ids_ratio,

                    ids_coverage=float(ids_out.get("coverage", 0.0)),
                    attack_in_rate=float(ids_out.get("attack_in_rate", 0.0)),
                    user_drop_rate=float(ids_out.get("user_drop_rate", 0.0)),
                    attack_drop_rate=float(ids_out.get("attack_drop_rate", 0.0)),
                    od_plan=cache["od_plan"],

                    local_num_req=n,
                    ema=float(cache["ema"]),
                    ema_mom=float(cache["ema_mom"]),
                    cpu_to_ids_ratio=edge.ids_cpu / edge.budget.cpu,
                    va_cpu_utilization=float(cache["va_cpu_utilization"]),
                    attack_cpu_frac=float(cache["attack_cpu_frac"]),
                    ids_cpu_utilization=float(ids_out["ids_cpu_util"]),
                    bw_utilization=float(cache["uplink_util"]),

                    overhead=float(overheads[i]),
                    ids_user_in_rate=float(cache.get("exec_user_in", 0.0)),
                )
            )

        self.t += 1

        if self.t >= self.t_max:
            num = 0.0
            den = 0.0
            for edge in self.edge_areas:
                h = [m for m in self.history if m.area_id == edge.area_id]
                if not h:
                    continue
                last_block = h[-self.t_max:]
                qoes = np.asarray([float(m.qoe_mean) for m in last_block], dtype=np.float32)
                viol = (qoes < edge.slo_threshold).astype(np.float32)
                viol_rate = float(viol.mean()) if viol.size else 0.0
                V_edge = float(np.exp(-float(edge.slo_beta) * viol_rate))
                score = float(qoes.mean()) * V_edge
                w = float(sum(int(m.local_num_req) for m in last_block))
                num += w * score
                den += w
            self.final_qoe = (num / den) if den > 0.0 else 0.0

    def _make_offload_plan(
        self,
        W_src: Dict[str, float],
        c_dst: Dict[str, float],
        stage: str = "ids",
        forced_mode: Optional[str] = None,
    ) -> OffloadPlan:
        """Dispatch to the correct offload function based on self.offload_mode."""
        mode = forced_mode if forced_mode is not None else self.offload_mode
        if mode == "none":
            return no_offload(self.area_ids, W_src)

        if mode == "balance":
            return balance_workload(self.area_ids, W_src, c_dst, cap_dst=c_dst)

        if mode == "cto":
            nested: Dict[str, Dict[str, float]] = {
                s: {r: float(self.prop_delay.get((s, r), 0.0)) for r in self.area_ids}
                for s in self.area_ids
            }
            return balance_workload_cto(self.area_ids, W_src, c_dst, nested)

        if mode == "cto_acc":
            nested = {
                s: {r: float(self.prop_delay.get((s, r), 0.0)) for r in self.area_ids}
                for s in self.area_ids
            }
            fnr_mat = fpr_mat = None
            if self.acc_by_region is not None and stage == "ids":
                fpr_mat = self.acc_by_region[:, :, 0]
                fnr_mat = self.acc_by_region[:, :, 1]
            w1, w2, w3 = self._offload_weights
            return balance_workload_cto_acc(
                area_ids=self.area_ids,
                W_src=W_src,
                c_dst=c_dst,
                propagation_delays=nested,
                weights=(w1, w2, w3),
                fnr_matrix=fnr_mat,
                fpr_matrix=fpr_mat,
                id_to_idx=self.id_to_idx,
            )

        if mode == "cto_acc_inv":
            nested = {
                s: {r: float(self.prop_delay.get((s, r), 0.0)) for r in self.area_ids}
                for s in self.area_ids
            }
            fnr_mat = fpr_mat = None
            if self.acc_by_region is not None and stage == "ids":
                fpr_mat = self.acc_by_region[:, :, 0]
                fnr_mat = self.acc_by_region[:, :, 1]
            w1, w2, w3 = self._offload_weights
            return balance_workload_cto_acc_inv(
                area_ids=self.area_ids,
                W_src=W_src,
                c_dst=c_dst,
                propagation_delays=nested,
                weights=(w1, w2, w3),
                fnr_matrix=fnr_mat,
                fpr_matrix=fpr_mat,
                id_to_idx=self.id_to_idx,
            )

        # modes "delay_workload" and "full" use score_based_offload
        nested = {
            s: {r: float(self.prop_delay.get((s, r), 0.0)) for r in self.area_ids}
            for s in self.area_ids
        }
        fnr_matrix = fpr_matrix = None
        if mode == "full" and self.acc_by_region is not None and stage == "ids":
            fpr_matrix = self.acc_by_region[:, :, 0]
            fnr_matrix = self.acc_by_region[:, :, 1]

        return score_based_offload(
            area_ids=self.area_ids,
            W_src=W_src,
            c_dst=c_dst,
            propagation_delays=nested,
            fnr_matrix=fnr_matrix,
            fpr_matrix=fpr_matrix,
            weights=self._offload_weights,
            id_to_idx=self.id_to_idx,
            max_prop_delay=self._max_prop_delay_ms,
        )

    def _run_step_multi_edge(self, ids_cpus, overhead=0.0, disable_attack=False, forced_mode: Optional[str] = None):
        """Two-stage IDS→VA offload step. Preserves single-edge overhead logic."""
        edges = {e.area_id: e for e in self.edge_areas}
        self._edges_by_id = edges  # cache for _make_offload_plan

        if isinstance(ids_cpus, torch.Tensor):
            ids_cpus = ids_cpus.detach().cpu().tolist()

        overheads = (
            [float(x) for x in overhead]
            if isinstance(overhead, (list, tuple, np.ndarray))
            else [float(overhead)] * len(self.edge_areas)
        )

        # Set CPU splits (single-edge overhead logic)
        for i, edge in enumerate(self.edge_areas):
            oh = overheads[i]
            overhead_ids = oh if oh > 0.0 else 0.0
            overhead_va  = abs(oh) if oh < 0.0 else 0.0
            ids_cpu_eff = float(np.clip(float(ids_cpus[i]) - overhead_ids, 0.0, edge.budget.cpu))
            va_cpu_eff  = float(np.clip(edge.budget.cpu - float(ids_cpus[i]) - overhead_va, VA_CPU_RESERVE, edge.budget.cpu))
            total = ids_cpu_eff + va_cpu_eff
            if total > edge.budget.cpu:
                va_cpu_eff = max(VA_CPU_RESERVE, va_cpu_eff - (total - edge.budget.cpu))
            edge.ids_cpu = ids_cpu_eff
            edge.va_cpu  = va_cpu_eff

        # Temporarily disable attacks for ideal pass.
        # _attack_agg_at iterates cur_attacker, so clearing it suppresses all attack load.
        if disable_attack:
            for edge in self.edge_areas:
                edge._tmp_cur_attacker = edge.cur_attacker
                edge.cur_attacker = []
                edge._tmp_attackers = edge.attackers
                edge.attackers = []

        # Stage A: observe arrivals at each edge
        obs = {aid: edges[aid].observe_arrivals(self.t) for aid in self.area_ids}

        # Stage B: IDS offload plan
        W_ids = {aid: float(obs[aid]["total_workload_in"]) for aid in self.area_ids}
        c_ids = {
            aid: float(edges[aid].ids.effective_speed_pkt_per_step(edges[aid].ids_cpu))
            for aid in self.area_ids
        }
        plan_ids = self._make_offload_plan(W_ids, c_ids, stage="ids", forced_mode=forced_mode)

        exec_user_in: Dict[str, int] = {aid: 0 for aid in self.area_ids}
        exec_atk_in:  Dict[str, int] = {aid: 0 for aid in self.area_ids}
        exec_local_user_in: Dict[str, int] = {aid: 0 for aid in self.area_ids}
        ids_remote_d_num: Dict[str, float] = {aid: 0.0 for aid in self.area_ids}
        ids_remote_n:     Dict[str, float] = {aid: 0.0 for aid in self.area_ids}
        # Per-flow attack resource usage aggregated per IDS-executor (pre-IDS, cross-edge safe)
        exec_atk_bw_mb_pre:  Dict[str, float] = {aid: 0.0 for aid in self.area_ids}
        exec_atk_cycles_pre: Dict[str, float] = {aid: 0.0 for aid in self.area_ids}
        for src in self.area_ids:
            u = float(obs[src]["user_req_in"])
            a = float(obs[src]["atk_req_in"])
            tot = max(u + a, 1.0)
            a_share = a / tot
            # Per-flow rates at this source (safe even when a == 0)
            src_flows = max(float(obs[src]["attack_dict"]["flows"]), 1e-9)
            bw_per_atk_flow  = float(obs[src]["attack_dict"].get("bw_in", 0.0)) / src_flows
            cyc_per_atk_flow = float(obs[src]["attack_dict"].get("cycles_per_step", 0.0)) / src_flows
            for dst, n_sent in plan_ids.flow.get(src, {}).items():
                n_sent_f = float(n_sent)
                nu = int(round(n_sent_f * u / tot))
                na = int(n_sent_f) - nu
                exec_user_in[dst] += nu
                exec_atk_in[dst]  += na
                if src == dst:
                    exec_local_user_in[dst] += nu
                else:
                    d = float(self.prop_delay.get((src, dst), 0.0))
                    ids_remote_d_num[dst] += d * n_sent_f
                    ids_remote_n[dst]     += n_sent_f
                # Accumulate attack bw/cycles by per-flow rate × routed attack count
                exec_atk_bw_mb_pre[dst]  += n_sent_f * a_share * bw_per_atk_flow
                exec_atk_cycles_pre[dst] += n_sent_f * a_share * cyc_per_atk_flow
        ids_d_remote_avg: Dict[str, float] = {
            aid: ids_remote_d_num[aid] / ids_remote_n[aid] if ids_remote_n[aid] > 0 else 0.0
            for aid in self.area_ids
        }

        # Compute per-executor FPR/FNR overrides from accuracy_by_region (weighted by flow volume)
        exec_fpr_ov: Dict[str, Optional[float]] = {aid: None for aid in self.area_ids}
        exec_tpr_ov: Dict[str, Optional[float]] = {aid: None for aid in self.area_ids}
        if self.acc_by_region is not None:
            for e_exec in self.area_ids:
                exec_idx = self.id_to_idx[e_exec]
                total_dst = float(plan_ids.assigned_dst.get(e_exec, 0))
                if total_dst <= 1e-9:
                    continue
                fpr_w = fnr_w = 0.0
                for e_src in self.area_ids:
                    src_idx = self.id_to_idx[e_src]
                    n = float(plan_ids.flow.get(e_src, {}).get(e_exec, 0))
                    fpr_w += n * float(self.acc_by_region[src_idx, exec_idx, 0])
                    fnr_w += n * float(self.acc_by_region[src_idx, exec_idx, 1])
                exec_fpr_ov[e_exec] = fpr_w / total_dst
                exec_tpr_ov[e_exec] = 1.0 - (fnr_w / total_dst)

        # Execute IDS at each executor
        ids_out_exec: Dict[str, dict] = {
            aid: edges[aid].process_ids(
                t=self.t,
                user_in=exec_user_in[aid],
                atk_in=exec_atk_in[aid],
                inspect_in=exec_user_in[aid] + exec_atk_in[aid],
                attack_dict=obs[aid]["attack_dict"],
                fpr_override=exec_fpr_ov[aid],
                tpr_override=exec_tpr_ov[aid],
            )
            for aid in self.area_ids
        }

        # Executor keeps admitted workload (no return to owner)
        admitted_user: Dict[str, int] = {
            aid: int(ids_out_exec[aid].get("user_pass_cnt", exec_user_in[aid]))
            for aid in self.area_ids
        }
        admitted_atk: Dict[str, int] = {
            aid: int(ids_out_exec[aid].get("atk_pass_cnt", 0))
            for aid in self.area_ids
        }
        # Locally-originated admitted users: scale local fraction by IDS pass rate
        admitted_local_user: Dict[str, int] = {
            aid: int(round(
                float(exec_local_user_in[aid])
                * (float(admitted_user[aid]) / max(float(exec_user_in[aid]), 1.0))
            ))
            for aid in self.area_ids
        }

        # Post-IDS attack bw/cycles at each executor (scale pre-IDS values by IDS pass rate)
        exec_atk_bw_mb:  Dict[str, float] = {}
        exec_atk_cycles: Dict[str, float] = {}
        for aid in self.area_ids:
            ids_atk_pass = float(admitted_atk[aid]) / max(float(exec_atk_in[aid]), 1.0)
            exec_atk_bw_mb[aid]  = exec_atk_bw_mb_pre[aid] * ids_atk_pass
            exec_atk_cycles[aid] = exec_atk_cycles_pre[aid] * ids_atk_pass

        # Stage C: VA offload plan.
        # Mode 1 (va_attack_offload=False): only benign users are offloaded; admitted attacks
        #   consume local VA capacity before the offload plan is built.
        # Mode 2 (va_attack_offload=True): attacks are offloaded together with benign users;
        #   VA capacity is offered at full value and attack resource consumption is distributed
        #   across VA executors after routing.
        va_user_in: Dict[str, int] = {aid: 0 for aid in self.area_ids}
        va_atk_in:  Dict[str, int] = {aid: 0 for aid in self.area_ids}
        va_local_user_in: Dict[str, int] = {aid: 0 for aid in self.area_ids}
        va_original_user_in: Dict[str, int] = {aid: 0 for aid in self.area_ids}
        va_remote_d_num: Dict[str, float] = {aid: 0.0 for aid in self.area_ids}
        va_remote_n:     Dict[str, float] = {aid: 0.0 for aid in self.area_ids}

        if not self.va_attack_offload:
            # Mode 1: attacks stay at their IDS executor; capacity pre-reduced by attack load.
            W_va = {aid: float(admitted_user[aid]) for aid in self.area_ids}
            c_va: Dict[str, float] = {}
            for aid in self.area_ids:
                edge = edges[aid]
                avail_cycles = max(
                    0.0,
                    float(edge.va_cpu) * edge.cpu_cycle_per_ms * edge.slot_ms
                    - exec_atk_cycles[aid],
                )
                min_det_cycles = min(edge.pipeline.det_cycles.values())
                c_va[aid] = avail_cycles / min_det_cycles if min_det_cycles > 0 else 0.0
            plan_va = self._make_offload_plan(W_va, c_va, stage="va", forced_mode=forced_mode)

            # Admitted attacks stay local — seed bw/cycles at their IDS executor
            va_atk_bw_mb:  Dict[str, float] = {aid: float(exec_atk_bw_mb[aid])  for aid in self.area_ids}
            va_atk_cycles: Dict[str, float] = {aid: float(exec_atk_cycles[aid]) for aid in self.area_ids}
            for src in self.area_ids:
                u = float(admitted_user[src])
                local_frac    = float(admitted_local_user[src]) / max(u, 1.0)
                ids_pass_rate = u / max(float(exec_user_in[src]), 1.0)
                va_atk_in[src] += int(admitted_atk[src])
                for dst, n_sent in plan_va.flow.get(src, {}).items():
                    n_sent_f = float(n_sent)
                    va_user_in[dst] += n_sent
                    va_original_user_in[dst] += int(round(n_sent_f / max(ids_pass_rate, 1e-9)))
                    if src == dst:
                        va_local_user_in[dst] += int(round(n_sent_f * local_frac))
                    else:
                        d = float(self.prop_delay.get((src, dst), 0.0))
                        va_remote_d_num[dst] += d * n_sent_f
                        va_remote_n[dst]     += n_sent_f
        else:
            # Mode 2: attacks offloaded with benign users; full VA capacity offered.
            W_va = {aid: float(admitted_user[aid] + admitted_atk[aid]) for aid in self.area_ids}
            c_va = {}
            for aid in self.area_ids:
                edge = edges[aid]
                avail_cycles = float(edge.va_cpu) * edge.cpu_cycle_per_ms * edge.slot_ms
                min_det_cycles = min(edge.pipeline.det_cycles.values())
                c_va[aid] = avail_cycles / min_det_cycles if min_det_cycles > 0 else 0.0
            plan_va = self._make_offload_plan(W_va, c_va, stage="va", forced_mode=forced_mode)

            # Attack bw/cycles distributed to VA executors proportional to routed attack count
            va_atk_bw_mb  = {aid: 0.0 for aid in self.area_ids}
            va_atk_cycles = {aid: 0.0 for aid in self.area_ids}
            for src in self.area_ids:
                u = float(admitted_user[src])
                a = float(admitted_atk[src])
                tot = max(u + a, 1.0)
                local_frac    = float(admitted_local_user[src]) / max(u, 1.0)
                ids_pass_rate = u / max(float(exec_user_in[src]), 1.0)
                for dst, n_sent in plan_va.flow.get(src, {}).items():
                    n_sent_f = float(n_sent)
                    nu = int(round(n_sent_f * u / tot))
                    na = int(n_sent_f) - nu
                    va_user_in[dst] += nu
                    va_atk_in[dst]  += na
                    va_original_user_in[dst] += int(round(float(nu) / max(ids_pass_rate, 1e-9)))
                    if src == dst:
                        va_local_user_in[dst] += int(round(float(nu) * local_frac))
                    else:
                        d = float(self.prop_delay.get((src, dst), 0.0))
                        va_remote_d_num[dst] += d * n_sent_f
                        va_remote_n[dst]     += n_sent_f
                    # Distribute attack bw/cycles proportional to attack fraction routed
                    atk_frac = float(na) / max(a, 1.0)
                    va_atk_bw_mb[dst]  += exec_atk_bw_mb[src]  * atk_frac
                    va_atk_cycles[dst] += exec_atk_cycles[src] * atk_frac
        va_d_remote_avg: Dict[str, float] = {
            aid: va_remote_d_num[aid] / va_remote_n[aid] if va_remote_n[aid] > 0 else 0.0
            for aid in self.area_ids
        }

        # Execute VA at each executor — combine IDS-stage and VA-stage propagation delays
        local_cache: Dict[str, dict] = {
            aid: edges[aid].process_va(
                t=self.t,
                admitted_user_req_in=va_user_in[aid],
                admitted_atk_req_in=va_atk_in[aid],
                attack_bw_mb=va_atk_bw_mb[aid],
                attack_cycles_per_step=va_atk_cycles[aid],
                n_local_user=va_local_user_in[aid],
                original_user_in=va_original_user_in[aid],
                d_remote_avg_ms=ids_d_remote_avg[aid] + va_d_remote_avg[aid],
                ids_out=ids_out_exec[aid],
            )
            for aid in self.area_ids
        }
        # Annotate each cache entry with the original user count for weighting in step()
        for aid in self.area_ids:
            local_cache[aid]["original_user_in"] = va_original_user_in[aid]
            local_cache[aid]["exec_user_in"] = exec_user_in[aid]

        # Restore attack state
        if disable_attack:
            for edge in self.edge_areas:
                if hasattr(edge, "_tmp_cur_attacker"):
                    edge.cur_attacker = edge._tmp_cur_attacker
                    del edge._tmp_cur_attacker
                if hasattr(edge, "_tmp_attackers"):
                    edge.attackers = edge._tmp_attackers
                    del edge._tmp_attackers

        for edge in self.edge_areas:
            assert edge.ids_cpu >= 0.0
            assert edge.va_cpu >= 0.0
            assert edge.ids_cpu + edge.va_cpu <= edge.budget.cpu + 1e-6

        return local_cache
                
        
        
def _resolve_offload_mode(g: dict) -> str:
    """Read offload_mode from config; fall back to legacy offload: true/false."""
    if "offload_mode" in g:
        return str(g["offload_mode"])
    legacy = g.get("offload", True)
    return "balance" if legacy else "none"


def _load_accuracy_matrix(
    data: dict,
    dirichlet_alpha: float,
    model_type: str,
    n_edges: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[int], str, str]:
    """
    Sample an acc_by_region matrix from a pre-loaded accuracy_matrix dict.

    Key format in JSON:
      "lmN->EM"  local model N evaluated on client M's test data  (model_type="lm")
      "gm->EM"   global model evaluated on client M's test data   (model_type="gm")

    acc_by_region[src_idx, exec_idx] = [FPR, FNR]
      src_idx  = traffic origin edge   → selects client EN (whose data distribution)
      exec_idx = IDS executor edge     → selects local model lmN (or gm)

    With model_type="gm" every executor has the same FPR/FNR on a given source's data.
    With model_type="lm" each executor uses its own personalized model.

    Returns:
      acc_by_region    : (n_edges, n_edges, 2) float32 [FPR, FNR]
      selected_clients : 1-based client indices assigned to each edge
      alpha_key        : matched JSON top-level key
      run_key          : selected run index string
    """
    # Parse alpha values from top-level keys; match closest in log-space
    alpha_map: Dict[float, str] = {}
    for key in data:
        m = re.search(r'alpha([\d.]+)', key)
        if m:
            alpha_map[float(m.group(1))] = key

    log_target = np.log(float(dirichlet_alpha))
    closest_alpha = min(alpha_map, key=lambda a: abs(np.log(a) - log_target))
    alpha_key = alpha_map[closest_alpha]

    # Random run
    runs = list(data[alpha_key].keys())
    run_key = runs[int(rng.integers(len(runs)))]

    metrics  = data[alpha_key][run_key]["hybrid_mse_avg"]["testing"]
    tpr_data: Dict[str, float] = metrics["tpr"]
    tnr_data: Dict[str, float] = metrics["tnr"]

    # Discover available client indices from target part of keys (->EN)
    all_clients: List[int] = sorted({
        int(m.group(1))
        for k in tpr_data
        for m in [re.search(r'->E(\d+)', k)]
        if m
    })

    if n_edges > len(all_clients):
        raise ValueError(
            f"Requested {n_edges} edges but accuracy matrix only has "
            f"{len(all_clients)} clients."
        )

    selected: List[int] = [int(c) for c in rng.choice(all_clients, size=n_edges, replace=False)]

    # Build (n_edges, n_edges, 2) matrix
    acc = np.zeros((n_edges, n_edges, 2), dtype=np.float32)
    for src_i, src_c in enumerate(selected):
        for exec_i, exec_c in enumerate(selected):
            if model_type == "gm":
                lookup = f"gm->E{src_c}"
            else:
                lookup = f"lm{exec_c}->E{src_c}"
                if lookup not in tpr_data:      # fall back to gm when lm absent
                    lookup = f"gm->E{src_c}"
            tpr = float(tpr_data.get(lookup, 1.0))
            tnr = float(tnr_data.get(lookup, 1.0))
            acc[src_i, exec_i, 0] = 1.0 - tnr  # FPR
            acc[src_i, exec_i, 1] = 1.0 - tpr  # FNR

    # client_labels = [f"E{c}" for c in selected]
    # col_w = 10
    # header = " " * col_w + "".join(f"→{lbl:<{col_w}}" for lbl in client_labels)
    # print(f"\n[Accuracy Matrix] dirichlet_alpha={dirichlet_alpha} → {alpha_key!r}, run={run_key}")
    # print(f"  model_type={model_type!r}, selected clients: {client_labels}")
    # print(f"  FPR [src→exec]:")
    # print(f"  {header}")
    # for src_i, lbl in enumerate(client_labels):
    #     row = "".join(f"{acc[src_i, exec_i, 0]:<{col_w}.4f}" for exec_i in range(n_edges))
    #     print(f"  {lbl:<{col_w}}{row}")
    # print(f"  FNR [src→exec]:")
    # print(f"  {header}")
    # for src_i, lbl in enumerate(client_labels):
    #     row = "".join(f"{acc[src_i, exec_i, 1]:<{col_w}.4f}" for exec_i in range(n_edges))
    #     print(f"  {lbl:<{col_w}}{row}")

    return acc, selected, alpha_key, run_key


def build_env_from_cfg(cfg: dict):
    globals_cfg = load_globals(cfg)

    # --------------------------------------------------
    # Build AttackTypeLibrary (if attack_sampler block present)
    # --------------------------------------------------
    sampler_cfg = cfg["globals"].get("attack_sampler")
    if sampler_cfg:
        lib_rng = np.random.default_rng(cfg["run"]["seed"])
        attack_type_library = AttackTypeLibrary(
            n_types=int(sampler_cfg.get("n_types", 10)),
            sampler_cfg=sampler_cfg,
            rng=lib_rng,
        )
    else:
        attack_type_library = None

    # --------------------------------------------------
    # Build shared VideoPipeline
    # --------------------------------------------------
    video_pipeline = VideoPipeline(
        reid_latency_ms_per_object=cfg["globals"]["video_pipeline"]["reid_latency"],
        configs=cfg["globals"]["video_pipeline"]["configs"],
        cpu_cycle_per_ms=globals_cfg.cpu_cycle_per_ms,
        cpu_cores=globals_cfg.cpu_cores,
    )
    # --------------------------------------------------
    # Build EdgeAreas
    # --------------------------------------------------
    edge_areas = []

    # Global fallback IDS accuracy (used when area has no ids_config)
    global_ids_accuracy = {"default": (0.0, 0.0)}
    global_user_cfg = cfg["globals"].get("user_sampler")

    for area_cfg in cfg["edge_areas"]:
        ids_accuracy = (
            {k: tuple(v) for k, v in area_cfg["ids_config"]["accuracy_by_type"].items()}
            if "ids_config" in area_cfg
            else global_ids_accuracy
        )
        ids = IDS(
            cycles_per_packet=globals_cfg.ids_cycles_per_packet,
            accuracy_by_type_fpr_fnr=ids_accuracy,
            cpu_cycle_per_ms=globals_cfg.cpu_cycle_per_ms,
            cpu_cores=globals_cfg.cpu_cores,
            slot_ms=globals_cfg.slot_ms,
        )

        users = []
        area_users_cfg = area_cfg.get("users", [])
        
        if not area_users_cfg and global_user_cfg:
            # Create a default user if none specified but global sampler exists
            users.append(
                User(
                    user_id=0,
                    slot_ms=globals_cfg.slot_ms,
                    t_max=cfg["run"]["t_max"],
                    seed=cfg["run"]["seed"],
                    synth_cfg=global_user_cfg["synthetic"],
                )
            )
        else:
            for u in area_users_cfg:
                # Prefer global user_sampler if present
                synth_cfg = global_user_cfg["synthetic"] if global_user_cfg else u["synthetic"]
                users.append(
                    User(
                        user_id=u["user_id"],
                        slot_ms=globals_cfg.slot_ms,
                        t_max=cfg["run"]["t_max"],
                        seed=cfg["run"]["seed"],
                        synth_cfg=synth_cfg,
                    )
                )

        # With attack_type_library, attackers are built dynamically at episode reset
        attackers = []

        edge = EdgeArea(
            area_id=area_cfg["area_id"],
            cpu_cycle_per_ms=area_cfg.get("cpu_cycle_per_ms"),
            slot_ms=globals_cfg.slot_ms,
            slo_beta=area_cfg["slo_beta"],
            slo_threshold=area_cfg["slo_threshold"],
            budget=ResourceBudget(**area_cfg["budget"]),
            constraints=area_cfg["constraints"],
            ids=ids,
            users=users,
            attackers=attackers,
            pipeline=video_pipeline,
            attack_type_library=attack_type_library,
            t_max=cfg["run"]["t_max"],
            dirichlet_alpha=float(area_cfg.get("dirichlet_concentration", 1.0)),
        )

        edge_areas.append(edge)


    # Build delay matrix from config if present, else default to 2 ms inter-edge
    n = len(edge_areas)
    if "delay_ms" in cfg["globals"]:
        delay_ms = np.array(cfg["globals"]["delay_ms"], dtype=np.float32)
    else:
        delay_ms = np.where(np.eye(n, dtype=bool), 0.0, 2.0).astype(np.float32)

    env = Environment(
        edge_areas=edge_areas,
        delay_ms=delay_ms,
        t_max=cfg["run"]["t_max"],
        seed=cfg["run"]["seed"],
        offload_mode=_resolve_offload_mode(cfg["globals"]),
        va_attack_offload=bool(cfg["globals"].get("va_attack_offload", False)),
    )
    env._max_prop_delay_ms = float(cfg["globals"].get("max_prop_delay_ms", 1e9))
    w = cfg["globals"].get("offload_weights", {})
    env._offload_weights = (
        float(w.get("w1", 1.0)),
        float(w.get("w2", 1.0)),
        float(w.get("w3", 0.01)),
    )

    # Accuracy matrix: load JSON once; resample clients/run at every reset().
    acc_mat_cfg = cfg["globals"].get("accuracy_matrix")
    if acc_mat_cfg is not None:
        env._acc_mat_data  = json.loads(Path(acc_mat_cfg["path"]).read_text())
        env._acc_mat_alpha = float(
            cfg["globals"].get("attack_sampler", {}).get("dirichlet_alpha", 1.0)
        )
        env._acc_mat_model = acc_mat_cfg.get("model", "lm")
    else:
        acc_by_region_raw = cfg["globals"].get("accuracy_by_region")
        if acc_by_region_raw is not None:
            env.acc_by_region = np.array(acc_by_region_raw, dtype=np.float32)

    env.reset(cfg["run"]["seed"])
    return env




def build_env_base(cfg_path: str):
    cfg_text = Path(cfg_path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text)
    return build_env_from_cfg(cfg)


           
def _reactive_ids_cpu(env, ids_cpu: np.ndarray, decision_interval: int,
                       scale_step: float = 0.5, ids_cpu_min: float = 0.5) -> np.ndarray:
    """Compute new ids_cpu allocation using reactive thresholding on IDS utilization."""
    n_edges = len(env.edge_areas)
    ids_cpu_max = np.array([e.budget.cpu - VA_CPU_RESERVE for e in env.edge_areas], dtype=np.float32)

    if len(env.history) < decision_interval * n_edges:
        return ids_cpu.copy()

    block = env.history[-decision_interval * n_edges:]
    df = pd.DataFrame([m.__dict__ for m in block])

    utils = []
    for edge in env.edge_areas:
        g = df[df["area_id"] == edge.area_id]
        if g.empty or "ids_cpu_utilization" not in g.columns:
            continue
        utils.append(float(np.clip(np.mean(g["ids_cpu_utilization"].values), 0.0, 1.0)))

    util = float(max(utils)) if utils else 0.0

    if util >= 0.80:
        delta = np.ones(n_edges, dtype=np.float32)
    elif util <= 0.20:
        delta = -np.ones(n_edges, dtype=np.float32)
    else:
        delta = np.zeros(n_edges, dtype=np.float32)

    new_ids_cpu = ids_cpu + delta * scale_step
    new_ids_cpu = np.clip(new_ids_cpu, ids_cpu_min, ids_cpu_max)
    return new_ids_cpu


def test_environment_run(cfg_path: str, plot=False, decision_interval: int = 500,
                         method: str = "reactive", constant_cpu: float = 0.5):
    """
    method: "reactive"    - threshold-based IDS CPU adjustment every decision_interval steps
            "constant"    - fixed ids_cpu = constant_cpu for all steps
    """
    with open(cfg_path) as _f:
        _cfg = yaml.safe_load(_f)
    q_th = float(_cfg["globals"].get("reward", {}).get("q_th", 0.20))

    env = build_env_base(cfg_path)

    dfs = []
    for i in range(3):
        env.reset(seed=1000 + i)

        n_edges = len(env.edge_areas)
        ids_cpu_max = np.array([e.budget.cpu - VA_CPU_RESERVE for e in env.edge_areas], dtype=np.float32)

        if method == "constant":
            ids_cpu = np.clip(np.full(n_edges, constant_cpu, dtype=np.float32), 0.0, ids_cpu_max)
        else:
            ids_cpu = np.array([e.ids_cpu for e in env.edge_areas], dtype=np.float32)

        t = 0
        while t < env.t_max:
            if method == "reactive":
                ids_cpu = _reactive_ids_cpu(env, ids_cpu, decision_interval)

            for _ in range(decision_interval):
                env.step(ids_cpu.tolist())
                t += 1
                if t >= env.t_max:
                    break

        df = pd.DataFrame([m.__dict__ for m in env.history])
        df["episode"] = i
        df["t"] = i * env.t_max + df["t"]
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    assert not all_df.empty, "No metrics produced"
    assert np.isfinite(all_df["qoe_mean"]).all(), "Invalid QoE values"
    assert all_df["ids_coverage"].between(0, 1).all(), "IDS coverage out of range"


    out_dir = "logs/test"
    os.makedirs(out_dir, exist_ok=True)

    # QoE over time
    qoe_pivot = all_df.pivot(index="t", columns="area_id", values="qoe_mean")
    ax_qoe = qoe_pivot.plot(figsize=(10, 4), title="QoE over time", alpha=0.25)
    qoe_pivot.rolling(500, min_periods=1).mean().plot(ax=ax_qoe, linewidth=2)
    ax_qoe.get_figure().savefig(f"{out_dir}/qoe_over_time.png", bbox_inches="tight")

    # Latency over time
    (
        all_df.pivot(index="t", columns="area_id", values="ids_cpu_utilization")
        .plot(figsize=(10, 4), title="IDS CPU Utilization")
        .get_figure()
        .savefig(f"{out_dir}/ids_cpu_utilization.png", bbox_inches="tight")
    )

    ax = all_df.pivot(index="t", columns="area_id", values="local_num_req").plot(
        figsize=(10, 4),
        title="Num Request",
        alpha=0.25,
    )

    all_df.pivot(index="t", columns="area_id", values="local_num_req") \
        .rolling(500, min_periods=1) \
        .mean() \
        .plot(ax=ax, linewidth=2)

    ax.get_figure().savefig(f"{out_dir}/local_num_req_combined.png", bbox_inches="tight")
    # Ema mom
    (
        all_df.pivot(index="t", columns="area_id", values="ema_mom")
        .plot(figsize=(10, 4), title="EMA Momentum")
        .get_figure()
        .savefig(f"{out_dir}/ema_mom.png", bbox_inches="tight")
    )
    
    # Attack in 
    (
        all_df.pivot(index="t", columns="area_id", values=["attack_in_rate", "attack_drop_rate"])
        .plot(figsize=(10, 4), title="Attack In Rate")
        .get_figure()
        .savefig(f"{out_dir}/attack_in_rate.png", bbox_inches="tight")
    )

    all_df["machine_count"] = all_df["cpu_to_ids_ratio"] * 16

    (
        all_df.pivot(index="t", columns="area_id", values="machine_count")
        .plot(figsize=(10, 4), title="Machine Count")
        .get_figure()
        .savefig(f"{out_dir}/machine_count.png", bbox_inches="tight")
    )    

    violation_rate = float((all_df["qoe_mean"] < q_th).mean())

    atk_in   = all_df["attack_in_rate"].sum()
    atk_drop = all_df["attack_drop_rate"].sum()
    user_in   = all_df["local_num_req"].sum()
    user_drop = all_df["user_drop_rate"].sum()
    
    malicious_drop_pct = 100.0 * atk_drop / atk_in if atk_in > 0 else 0.0
    benign_drop_pct = 100.0 * user_drop / user_in if user_in > 0 else 0.0

    avg_qoe = all_df["qoe_mean"].mean()
    print(f"Average QoE (qoe_mean):       {avg_qoe:.4f}")
    print(f"Average QoE (benign_col_dmg): {all_df['benign_col_dmg'].mean():.4f}")
    print(f"SLO violation rate (q_th={q_th:.2f}): {violation_rate:.4f} ({100*violation_rate:.1f}%)")
    print(f"Benign requests dropped:      {user_drop:.0f} ({benign_drop_pct:.1f}%)")
    print(f"Malicious traffic dropped:    {atk_drop:.0f} ({malicious_drop_pct:.1f}%)")
    print(f"Plots saved to {out_dir}/")    
        
if __name__ == "__main__":
    test_environment_run("./configs/simulation_ma_0.yaml", plot=True, method="reactive", constant_cpu=3.0)
