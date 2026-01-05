from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import yaml
import os
import numpy as np
import pandas as pd

from service import IDS, VideoPipeline

from request import UserCamera, Attacker

from edgearea import ResourceBudget, EdgeArea

from offload import OffloadDecision, OffloadState

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
    qoe_mean: float
    qoe_min: float
    uplink_available: float
    mean_mota: float
    local_num_objects: int
    num_objects: int
    ids_coverage: float
    attack_in_rate: float
    attack_drop_rate: float
    user_drop_rate: float
    
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
        self.delay_ms = delay_ms
        self.t_max = int(t_max)
        self.t = 0
        self.history: List[StepMetrics] = []

        np.random.seed(seed)

    def reset(self):
        self.t = 0
        self.history.clear()

    def run(self) -> pd.DataFrame:
        while self.t < self.t_max:
            self.step()
        return pd.DataFrame([m.__dict__ for m in self.history])
    
    def cooperative_offload_ot(
        self,
        states: List[OffloadState],
    ) -> Tuple[List[OffloadState], List[OffloadDecision]]:
        """
        Synchronized greedy OT offloading across all edges.
        """
        n = len(states)
        for i, st in enumerate(states):
            st.idx = i

        decisions: List[OffloadDecision] = []

        def finish_time_ms(st: OffloadState) -> float:
            if st.total_q() <= 0:
                return 0.0
            cycles = st.total_q() * st.track_cycles_per_obj
            if st.avail_cycles_per_ms <= 0:
                return float("inf")            
            return cycles / max(1e-9, st.avail_cycles_per_ms)

        TOP = list(range(n))
        TOP.sort(key=lambda i: finish_time_ms(states[i]))

        changed = True
        while changed:
            changed = False

            slow_order = sorted(
                TOP,
                key=lambda i: finish_time_ms(states[i]),
                reverse=True,
            )

            for src in slow_order:
                src_st = states[src]
                if src_st.total_q() <= 0:
                    continue

                local_C = finish_time_ms(src_st)

                for dst in TOP:
                    if dst == src:
                        continue

                    dst_st = states[dst]

                    remote_cycles = (dst_st.total_q() + 1) * dst_st.track_cycles_per_obj
                    remote_C = (
                        remote_cycles / max(1e-9, dst_st.avail_cycles_per_ms)
                        + self.delay_ms[src, dst]
                    )

                    if remote_C > dst_st.latest_track_finish_ms:
                        continue

                    if remote_C < local_C:
                        src_st.local_q_obj -= 1
                        dst_st.recv_q_obj += 1

                        decisions.append(
                            OffloadDecision(src_idx=src, dst_idx=dst, num_obj=1)
                        )

                        TOP.sort(key=lambda i: finish_time_ms(states[i]))
                        changed = True
                        break

        return states, decisions    
    
    def step(self):
        """
        One global timestep:
        1) Each edge makes local decisions independently
        2) Environment aggregates offload states
        3) Cooperative OT offloading
        4) Final QoE computation
        """

        # 1) Local decisions (no cross-edge information)
        offload_states: List[OffloadState] = []
        local_cache: Dict[str, Dict[str, Any]] = {}

        for edge in self.edge_areas:
            state, cache = edge.step_local(self.t)
            offload_states.append(state)
            local_cache[edge.area_id] = cache

        # 2) Cooperative OT offloading (global)
        offload_states, _ = self.cooperative_offload_ot(offload_states)

        # 3) Final QoE computation per edge
        for edge in self.edge_areas:
            cache = local_cache[edge.area_id]
            state = next(s for s in offload_states if s.area_id == edge.area_id)
            
            od_latency = cache["best_od_cycles"] / max(1e-9, state.avail_cycles_per_ms)

            ot_cycles = state.local_q_obj * state.track_cycles_per_obj
            ot_latency = ot_cycles / max(1e-9, state.avail_cycles_per_ms)

            final_latency = od_latency + ot_latency

            D_Max = edge.constraints["D_Max"]
            MOTA_min = edge.constraints["MOTA_min"]
            gamma = edge.constraints["Gamma"]

            # final_q = (cache["best_mean_mota"] - MOTA_min) * np.exp(
            #     -gamma * (final_latency - D_Max)
            # )
            final_q = cache["best_mean_mota"] if final_latency <= D_Max else 0

            ids_out = cache["ids_out"]

            self.history.append(
                StepMetrics(
                    t=self.t,
                    area_id=edge.area_id,
                    qoe_mean=float(final_q),
                    qoe_min=float(final_q),
                    uplink_available=float(state.uplink_available),
                    mean_mota=float(cache["best_mean_mota"]),
                    local_num_objects = state.local_q_obj,
                    num_objects = state.total_q(),
                    ids_coverage=float(ids_out.get("coverage", 0.0)),
                    attack_in_rate=float(ids_out.get("attack_in_rate", 0.0)),
                    attack_drop_rate=float(ids_out.get("attack_drop_rate", 0.0)),
                    user_drop_rate=float(ids_out.get("user_drop_rate", 0.0)),
                )
            )

        self.t += 1
        
        
def test_environment_run(cfg_path: str, plot=False):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

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
                UserCamera(
                    user_id=u["user_id"],
                    input_dir=u["input_dir"],
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
                    cpu_usage_const=atk_cfg["cpu_usage_const"],
                    non_defendable_bw_const=atk_cfg["non_defendable_bw_const"],
                    slot_ms=globals_cfg.slot_ms,
                    t_max=cfg["run"]["t_max"],
                    scaling=atk_cfg["scaling"],
                )
            )

        edge = EdgeArea(
            area_id=area_cfg["area_id"],
            cpu_cycle_per_ms=area_cfg.get("cpu_cycle_per_ms"),
            slot_ms=globals_cfg.slot_ms,
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
    delay_ms = np.zeros((n, n)) #! TODO
    for i in range(n):
        for j in range(n):
            delay_ms[i, j] = 2.0 if i != j else 0.0  # 2 ms inter-edge delay

    # Build environment
    env = Environment(
        edge_areas=edge_areas,
        delay_ms=delay_ms,
        t_max=cfg["run"]["t_max"],
        seed=cfg["run"]["seed"],
    )

    # Run a short simulation (sanity check)
    env.reset()
    for _ in range(env.t_max):  # do NOT run full 2000 in test
        env.step()

    df = pd.DataFrame([m.__dict__ for m in env.history])

    # --------------------------------------------------
    # Assertions / sanity checks
    # --------------------------------------------------
    assert not df.empty, "No metrics produced"
    assert np.isfinite(df["qoe_mean"]).all(), "Invalid QoE values"
    assert df["ids_coverage"].between(0, 1).all(), "IDS coverage out of range"


    out_dir = "logs/test"
    os.makedirs(out_dir, exist_ok=True)

    # QoE over time
    (
        df.pivot(index="t", columns="area_id", values="qoe_mean")
        .plot(figsize=(10, 4), title="QoE over time")
        .get_figure()
        .savefig(f"{out_dir}/qoe_over_time.png", bbox_inches="tight")
    )

    # Latency over time
    (
        df.pivot(index="t", columns="area_id", values="uplink_available")
        .plot(figsize=(10, 4), title="Available uplink over time")
        .get_figure()
        .savefig(f"{out_dir}/uplink_available.png", bbox_inches="tight")
    )

    # Executed objects (post-offload)
    (
        df.pivot(index="t", columns="area_id", values="num_objects")
        .plot(figsize=(10, 4), title="Post-offload tracking load")
        .get_figure()
        .savefig(f"{out_dir}/num_objects.png", bbox_inches="tight")
    )

    # IDS coverage
    (
        df.pivot(index="t", columns="area_id", values="ids_coverage")
        .plot(figsize=(10, 4), title="IDS coverage")
        .get_figure()
        .savefig(f"{out_dir}/ids_coverage.png", bbox_inches="tight")
    )
    
    # Attack in 
    (
        df.pivot(index="t", columns="area_id", values="attack_in_rate")
        .plot(figsize=(10, 4), title="Attack In Rate")
        .get_figure()
        .savefig(f"{out_dir}/attack_in_rate.png", bbox_inches="tight")
    )

    print(f"Plots saved to {out_dir}/")    
        
if __name__ == "__main__":
    test_environment_run("./configs/simulation_0.yaml", plot=True)