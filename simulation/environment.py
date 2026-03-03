from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BoundedTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
)

from pathlib import Path
import yaml
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from service import IDS, VideoPipeline

from request import User, Attacker

from edgearea import ResourceBudget, EdgeArea

from offload import OffloadPlan, balance_with_caps_and_prop_filter



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
    qoe_weighted: float
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
    bw_utilization: float

    overhead: float
    
    # Per-step plan (detector mixing)
    od_plan: Dict[str, int] = field(default_factory=dict)

    # Optional network impact metric
    I_net: float = 0.0

    
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
    ):
        self.edge_areas = edge_areas
        self.delay_ms = np.asarray(delay_ms, dtype=np.float32)
        self.t_max = int(t_max)
        self.t = 0
        self.history: List[StepMetrics] = []
        self.last_history: List[StepMetrics] = []
        self.final_qoe = 0

        # area_id mapping for multi-edge routing
        self.area_ids: List[str] = [str(e.area_id) for e in self.edge_areas]
        self.id_to_idx: Dict[str, int] = {eid: i for i, eid in enumerate(self.area_ids)}
        self.idx_to_id: Dict[int, str] = {i: eid for eid, i in self.id_to_idx.items()}

        n = len(self.edge_areas)
        assert self.delay_ms.shape == (n, n), f"delay_ms must be ({n},{n}), got {self.delay_ms.shape}"

        # propagation delay accessor used by offload planner
        self.prop_delay: Dict[Tuple[int, int], float] = {}
        for s in self.area_ids:
            si = self.id_to_idx[s]
            for r in self.area_ids:
                ri = self.id_to_idx[r]
                self.prop_delay[(s, r)] = float(self.delay_ms[si, ri])

        np.random.seed(seed)

    def _area_ids(self) -> List[int]:
        return self.area_ids

    def _edge_by_id(self) -> Dict[str, EdgeArea]:
        return {str(e.area_id): e for e in self.edge_areas}

    def reset(self, seed):
        self.t = 0
        self.last_history = list(self.history)
        self.history.clear()
        self.final_qoe = 0
        for i, edge in enumerate(self.edge_areas):
            edge.reset(seed=seed + i*99)        

    def _compute_tau_loc(self, W_va: Dict[int, float], c_va: Dict[int, float]) -> Dict[int, float]:
        # local processing proxy, higher means slower, used as threshold against propagation delay
        # tau_loc[e] = W_va[e] / c_va[e]
        out: Dict[int, float] = {}
        for e, w in W_va.items():
            out[e] = float(w) / max(float(c_va[e]), 1e-9)
        return out    
    
    def step(self, ids_cpus, overhead: float = 0.0):
        edges = self._edge_by_id()
        area_ids = self._area_ids()

        if isinstance(ids_cpus, torch.Tensor):
            ids_cpus = ids_cpus.detach().cpu().tolist()

        # overhead can be scalar or per-edge vector
        if isinstance(overhead, (list, tuple, np.ndarray)):
            overheads = [float(x) for x in overhead]
            assert len(overheads) == len(self.edge_areas)
        else:
            overheads = [float(overhead)] * len(self.edge_areas)

        # 0) set cpu split on each edge
        for i, edge in enumerate(self.edge_areas):
            oh = overheads[i]
            overhead_ids = oh if oh > 0.0 else 0.0
            overhead_va  = abs(oh) if oh < 0.0 else 0.0

            edge.ids_cpu = float(ids_cpus[i]) - float(overhead_ids)
            edge.va_cpu  = float(edge.budget.cpu) - float(ids_cpus[i]) - float(overhead_va)

            assert edge.ids_cpu + edge.va_cpu <= edge.budget.cpu + 1e-6
            assert edge.ids_cpu >= 0.5
            assert edge.va_cpu >= 0.5

        # 1) observe arrivals at owners (ingress)
        obs = {eid: edges[eid].observe_arrivals(self.t) for eid in area_ids}

        # -----------------------
        # A) IDS OFFLOAD + EXECUTE
        # -----------------------
        W_def_src = {eid: float(obs[eid]["total_workload_in"]) for eid in area_ids}
        c_def_dst = {eid: float(edges[eid].ids_cpu) for eid in area_ids}

        # multi-edge safe: pick min cycles per packet across edges (or define from globals)
        kappa_ids_min = min(float(edges[eid].ids.cycles_per_packet) for eid in area_ids)

        plan_def = balance_with_caps_and_prop_filter(
            area_ids=area_ids,
            edges=edges,
            kappa_min=kappa_ids_min,
            W_src=W_def_src,
            c_dst=c_def_dst,
            prop_delay=self.prop_delay,
            tau_loc=None,
        )
        ids_in_dst = plan_def.assigned_dst

        # 2) derive what each EXECUTOR actually receives (split by owner's mix)
        exec_user_in = {e: 0 for e in area_ids}
        exec_atk_in  = {e: 0 for e in area_ids}

        for e_owner in area_ids:
            u = float(obs[e_owner]["user_req_in"])
            a = float(obs[e_owner]["atk_req_in"])
            tot = max(u + a, 1.0)
            u_share = u / tot
            a_share = a / tot

            for e_exec, n_sent in plan_def.flow.get(e_owner, {}).items():
                n_sent = float(n_sent)
                exec_user_in[e_exec] += int(round(n_sent * u_share))
                exec_atk_in[e_exec]  += int(round(n_sent * a_share))

        # 3) execute IDS at executors
        ids_out_exec = {}
        for e_exec in area_ids:
            edge = edges[e_exec]
            ids_out_exec[e_exec] = edge.process_ids(
                t=self.t,
                user_in=int(exec_user_in[e_exec]),
                atk_in=int(exec_atk_in[e_exec]),
                inspect_in=int(exec_user_in[e_exec] + exec_atk_in[e_exec]),
                attack_dict=obs[e_exec]["attack_dict"],
            )

        # 4) executor verdict fractions (counts)
        user_pass_frac_exec = {}
        atk_pass_frac_exec  = {}

        for e_exec in area_ids:
            u_in = max(int(exec_user_in[e_exec]), 0)
            a_in = max(int(exec_atk_in[e_exec]), 0)

            u_pass = float(ids_out_exec[e_exec].get("user_pass_cnt", u_in))
            a_pass = float(ids_out_exec[e_exec].get("atk_pass_cnt", 0))

            user_pass_frac_exec[e_exec] = 1.0 if u_in <= 0 else float(np.clip(u_pass / u_in, 0.0, 1.0))
            atk_pass_frac_exec[e_exec]  = 0.0 if a_in <= 0 else float(np.clip(a_pass / a_in, 0.0, 1.0))

        # 5) owner aggregates returned verdicts
        admitted_user_owner = {e: 0 for e in area_ids}
        admitted_atk_owner  = {e: 0 for e in area_ids}

        for e_owner in area_ids:
            u = float(obs[e_owner]["user_req_in"])
            a = float(obs[e_owner]["atk_req_in"])
            tot = max(u + a, 1.0)
            u_share = u / tot
            a_share = a / tot

            for e_exec, n_sent in plan_def.flow.get(e_owner, {}).items():
                n_sent = float(n_sent)
                sent_u = n_sent * u_share
                sent_a = n_sent * a_share

                admitted_user_owner[e_owner] += int(round(sent_u * user_pass_frac_exec.get(e_exec, 1.0)))
                admitted_atk_owner[e_owner]  += int(round(sent_a * atk_pass_frac_exec.get(e_exec, 0.0)))

        # -----------------------
        # B) VA OFFLOAD + EXECUTE
        # -----------------------
        W_va_src = {eid: float(admitted_user_owner[eid]) for eid in area_ids}
        c_va_dst = {eid: float(edges[eid].va_cpu) for eid in area_ids}
        tau_loc  = self._compute_tau_loc(W_va_src, c_va_dst)

        # multi-edge safe: min per-request VA cycles across edges/models
        kappa_va_min = min(
            float(edges[eid].pipeline.detection_cycles("nanoDet-m"))
            for eid in area_ids
        )

        plan_va = balance_with_caps_and_prop_filter(
            area_ids=area_ids,
            edges=edges,
            W_src=W_va_src,
            c_dst=c_va_dst,
            kappa_min=kappa_va_min,
            prop_delay=self.prop_delay,
            tau_loc=tau_loc,
        )
        va_in_dst = plan_va.assigned_dst

        # execute VA at each destination edge
        local_cache = {}
        tot_req = 0
        for e_exec in area_ids:
            edge = edges[e_exec]
            cache = edge.process_va(
                t=self.t,
                admitted_user_req_in=int(va_in_dst.get(e_exec, 0)),
                attack_dict=obs[e_exec]["attack_dict"],
                ids_out=ids_out_exec[e_exec],
            )
            local_cache[e_exec] = cache
            tot_req += int(cache.get("local_num_request", 0))

        # log metrics
        for i,eid in enumerate(area_ids):
            edge = edges[eid]
            cache = local_cache[eid]
            ids_out = cache["ids_out"]

            n = int(cache.get("local_num_request", 0))
            qoe_weighted = float(cache["qoe"]) * (n / tot_req) if tot_req > 0 else float(cache["qoe"])

            self.history.append(
                StepMetrics(
                    t=self.t,
                    area_id=eid,
                    qoe_mean=float(cache["qoe"]),
                    qoe_weighted=qoe_weighted * 3,
                    ids_coverage=float(ids_out.get("coverage")),
                    attack_in_rate=float(ids_out.get("atk_in_cnt")),
                    user_drop_rate=float(ids_out.get("user_drop_cnt")),
                    od_plan=cache["od_plan"],
                    local_num_req=n,
                    ema=cache["ema"],
                    ema_mom=cache["ema_mom"],
                    attack_drop_rate=float(ids_out.get("atk_drop_cnt")),
                    cpu_to_ids_ratio=edge.ids_cpu / edge.budget.cpu,
                    va_cpu_utilization=cache["va_cpu_utilization"],
                    ids_cpu_utilization=float(ids_out.get("ids_cpu_util")),
                    bw_utilization=cache["uplink_util"],
                    overhead=float(overheads[i]),
                )
            )

        self.t += 1
        qoe_slo = []
        # inside Environment.step(), replace the final_qoe computation block
        if self.t >= self.t_max:
            num = 0.0
            den = 0.0

            for edge in self.edge_areas:
                h = [m for m in self.history if m.area_id == edge.area_id]
                if len(h) == 0:
                    continue

                last_block = h[-self.t_max:]  # window
                q = np.asarray([float(m.qoe_weighted) for m in last_block], dtype=np.float32)

                # SLO penalty term
                viol = (q < edge.slo_threshold).astype(np.float32)
                viol_rate = float(viol.mean()) if viol.size else 0.0
                V_edge = float(np.exp(-edge.slo_beta * viol_rate))

                # edge score (SLO-adjusted)
                score = float(q.mean()) * V_edge

                # weight by total served requests in the window
                w = float(np.sum([int(m.local_num_req) for m in last_block]))

                num += w * score
                den += w

            self.final_qoe = (num / den) if den > 0 else 0.0    
            
        # print("obs", obs, "\n",
        #     "ids_in_dst", ids_in_dst, "\n",
        #     "ids_out_exec", ids_out_exec,"\n",
        #     "plan_def", plan_def, "\n",
        #     "plan_va", plan_va, "\n",
        #     "admitted_user_owner", admitted_user_owner,"\n",
        #     "admitted_atk_owner", admitted_atk_owner, "\n",
        #     "======"
        # )
        return {
            "cache": local_cache,
            "plan_def": plan_def,
            "plan_va": plan_va,
            "admitted_user_owner": admitted_user_owner,
            "admitted_atk_owner": admitted_atk_owner,
        }
        
def build_env_base(cfg_path: str):
    cfg_text = Path(cfg_path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text)

    globals_cfg = load_globals(cfg)

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

    for area_cfg in cfg["edge_areas"]:
        ids = IDS(
            cycles_per_packet=globals_cfg.ids_cycles_per_packet,
            accuracy_by_type_fpr_fnr={
                k: tuple(v)
                for k, v in area_cfg["ids_config"]["accuracy_by_type"].items()
            },
            cpu_cycle_per_ms=globals_cfg.cpu_cycle_per_ms,
            cpu_cores=globals_cfg.cpu_cores,       
            slot_ms=globals_cfg.slot_ms,  
        )

        users = []
        for u in area_cfg.get("users", []):
            users.append(
                User(
                    user_id=u["user_id"],
                    slot_ms=globals_cfg.slot_ms,
                    t_max=cfg["run"]["t_max"],
                    seed=cfg["run"]["seed"],
                    synth_cfg=u["synthetic"],
                )
            )

        attackers = []

        for atk_ref in area_cfg.get("attackers", []):
            atk_type = atk_ref["attacker_type"]

            if atk_type not in cfg["globals"]["attack"]:
                raise KeyError(f"Unknown attacker_type: {atk_type}")

            atk_cfg = cfg["globals"]["attack"][atk_type]

            attackers.append(
                Attacker(
                    attacker_id=atk_type,
                    attack_type=atk_cfg["type"],
                    ts_df=pd.read_csv(atk_cfg["ts_path"]),
                    latency_per_flow=atk_cfg["latency_per_flow"],
                    bw_per_flow=atk_cfg["bw_per_flow"],
                    base_scaling=atk_cfg["scaling"],
                    mean_rep=atk_cfg["mean_rep"],
                    non_defendable_bw_const=atk_cfg["non_defendable_bw_const"],
                    slot_ms=globals_cfg.slot_ms,
                    t_max=cfg["run"]["t_max"],
                    seed=cfg["run"]["seed"],
                    cpu_cycle_per_ms=globals_cfg.cpu_cycle_per_ms,
                    cpu_cores=globals_cfg.cpu_cores,                      
                )
            )

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
        )

        edge_areas.append(edge)
        

    # Build delay matrix (simple symmetric test case)
    n = len(edge_areas)
    delay_ms = np.zeros((n, n), dtype=np.float32) #TODO!
    for i in range(n):
        for j in range(n):
            delay_ms[i, j] = 2.0 if i != j else 0.0

    # Build environment
    env = Environment(
        edge_areas=edge_areas,
        delay_ms=delay_ms,
        t_max=cfg["run"]["t_max"],
        seed=cfg["run"]["seed"],
    )
    env.reset(cfg["run"]["seed"])
    return env
    
    
def test_environment_run(cfg_path: str, plot=False):
    env = build_env_base(cfg_path)

    dfs = []
    for i in range(1):
        env.reset(seed=1000 + i)

        for _ in range(env.t_max):
            env.step([2.5]*len(env.edge_areas))
            # env.step([7.5, 0.5, 0.5])
            

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
    (
        all_df.pivot(index="t", columns="area_id", values="qoe_mean")
        .plot(figsize=(10, 4), title="QoE over time")
        .get_figure()
        .savefig(f"{out_dir}/qoe_over_time.png", bbox_inches="tight")
    )

    # Latency over time
    (
        all_df.pivot(index="t", columns="area_id", values="bw_utilization")
        .plot(figsize=(10, 4), title="Utilized uplink over time")
        .get_figure()
        .savefig(f"{out_dir}/uplink_utilization.png", bbox_inches="tight")
    )

    (
        all_df.pivot(index="t", columns="area_id", values="local_num_req")
        .plot(figsize=(10, 4), title="Num Request")
        .get_figure()
        .savefig(f"{out_dir}/local_num_req.png", bbox_inches="tight")
    )

    # Ema mom
    (
        all_df.pivot(index="t", columns="area_id", values="ema_mom")
        .plot(figsize=(10, 4), title="EMA Momentum")
        .get_figure()
        .savefig(f"{out_dir}/ema_mom.png", bbox_inches="tight")
    )
    
    # Attack in 
    (
        all_df.pivot(index="t", columns="area_id", values=["attack_in_rate", "ema"])
        .plot(figsize=(10, 4), title="Attack In Rate")
        .get_figure()
        .savefig(f"{out_dir}/attack_in_rate.png", bbox_inches="tight")
    )
    avg_qoe = all_df["qoe_weighted"].mean()
    print(f"Average QoE (qoe_mean): {avg_qoe:.4f}")
    print(f"Plots saved to {out_dir}/")    
        
if __name__ == "__main__":
    test_environment_run("./configs/simulation_ma_0.yaml", plot=True)