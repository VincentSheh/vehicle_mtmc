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
import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from service import IDS, VideoPipeline

from request import User, Attacker, AttackTypeLibrary

from edgearea import ResourceBudget, EdgeArea

from offload import OffloadPlan, balance_with_caps_and_prop_filter

N_ACTION = 9

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
        offload: bool = True,
    ):
        self.edge_areas = edge_areas
        self.delay_ms = np.asarray(delay_ms, dtype=np.float32)
        self.offload = bool(offload)
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
            print(f"\n[Attack type probabilities per edge (alpha={alpha:.3f}, fixed for this run)]")
            print(f"{'Edge':<12}  {header}")
            for i, edge in enumerate(self.edge_areas):
                row = "  ".join(f"{self._p_attack_type_matrix[i, j]:.4f}" for j in range(n_types))
                print(f"{str(edge.area_id):<12}  {row}")

    def reset(self, seed):
        self.t = 0
        self.last_history = list(self.history)
        self.history.clear()
        self.final_qoe = 0
        p_matrix = self._p_attack_type_matrix
        for i, edge in enumerate(self.edge_areas):
            p = p_matrix[i] if p_matrix is not None else None
            edge.reset(seed=seed + i, p_attack_type=p)

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
        )

        # 2. restore pre-step state
        self._restore_edges(snapshot)

        # 3. real pass from same pre-step state
        real_cache = self._run_step_multi_edge(
            ids_cpus=ids_cpus,
            overhead=overhead,
            disable_attack=False,
        )

        edges_by_id = {e.area_id: e for e in self.edge_areas}
        tot_req = sum(int(real_cache[aid].get("local_num_request", 0)) for aid in self.area_ids)
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
            n = int(cache.get("local_num_request", 0))
            qoe_weighted = float(cache["qoe"]) * (n / tot_req) if tot_req > 0 else float(cache["qoe"])

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

    def _run_step_multi_edge(self, ids_cpus, overhead=0.0, disable_attack=False):
        """Two-stage IDS→VA offload step. Preserves single-edge overhead logic."""
        edges = {e.area_id: e for e in self.edge_areas}

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
            va_cpu_eff  = float(np.clip(edge.budget.cpu - float(ids_cpus[i]) - overhead_va, 0.5, edge.budget.cpu))
            total = ids_cpu_eff + va_cpu_eff
            if total > edge.budget.cpu:
                va_cpu_eff = max(0.5, va_cpu_eff - (total - edge.budget.cpu))
            edge.ids_cpu = ids_cpu_eff
            edge.va_cpu  = va_cpu_eff

        # Temporarily disable attacks for ideal pass
        if disable_attack:
            for edge in self.edge_areas:
                edge._tmp_cur_attacker = edge.cur_attacker
                edge.cur_attacker = []

        # Stage A: observe arrivals at each edge
        obs = {aid: edges[aid].observe_arrivals(self.t) for aid in self.area_ids}

        # Stage B: IDS — offload or process locally
        if self.offload:
            W_ids = {aid: float(obs[aid]["total_workload_in"]) for aid in self.area_ids}
            c_ids = {aid: float(edges[aid].ids_cpu) for aid in self.area_ids}
            kappa_ids_min = min(float(edges[aid].ids.cycles_per_packet) for aid in self.area_ids)
            plan_ids = balance_with_caps_and_prop_filter(
                area_ids=self.area_ids,
                edges=edges,
                W_src=W_ids,
                c_dst=c_ids,
                kappa_min=kappa_ids_min,
                prop_delay=self.prop_delay,
                tau_loc=None,
            )
            exec_user_in: Dict[str, int] = {aid: 0 for aid in self.area_ids}
            exec_atk_in:  Dict[str, int] = {aid: 0 for aid in self.area_ids}
            for src in self.area_ids:
                u = float(obs[src]["user_req_in"])
                a = float(obs[src]["atk_req_in"])
                tot = max(u + a, 1.0)
                for dst, n_sent in plan_ids.flow.get(src, {}).items():
                    n_sent = float(n_sent)
                    exec_user_in[dst] += int(round(n_sent * u / tot))
                    exec_atk_in[dst]  += int(round(n_sent * a / tot))
        else:
            plan_ids = OffloadPlan(
                flow={aid: {aid: int(obs[aid]["total_workload_in"])} for aid in self.area_ids},
                assigned_dst={aid: int(obs[aid]["total_workload_in"]) for aid in self.area_ids},
            )
            exec_user_in = {aid: obs[aid]["user_req_in"] for aid in self.area_ids}
            exec_atk_in  = {aid: obs[aid]["atk_req_in"]  for aid in self.area_ids}

        # Execute IDS at each executor
        ids_out_exec: Dict[str, dict] = {
            aid: edges[aid].process_ids(
                t=self.t,
                user_in=exec_user_in[aid],
                atk_in=exec_atk_in[aid],
                inspect_in=exec_user_in[aid] + exec_atk_in[aid],
                attack_dict=obs[aid]["attack_dict"],
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

        # Stage C: VA — offload or process locally
        W_va = {aid: float(admitted_user[aid] + admitted_atk[aid]) for aid in self.area_ids}
        if self.offload:
            c_va = {aid: float(edges[aid].va_cpu) for aid in self.area_ids}
            tau_loc = {aid: W_va[aid] / max(c_va[aid], 1e-9) for aid in self.area_ids}
            kappa_va_min = min(
                float(edges[aid].pipeline.detection_cycles("nanoDet-m")) for aid in self.area_ids
            )
            plan_va = balance_with_caps_and_prop_filter(
                area_ids=self.area_ids,
                edges=edges,
                W_src=W_va,
                c_dst=c_va,
                kappa_min=kappa_va_min,
                prop_delay=self.prop_delay,
                tau_loc=tau_loc,
            )
            va_user_in: Dict[str, int] = {aid: 0 for aid in self.area_ids}
            va_atk_in:  Dict[str, int] = {aid: 0 for aid in self.area_ids}
            for src in self.area_ids:
                u = float(admitted_user[src])
                a = float(admitted_atk[src])
                tot = max(u + a, 1.0)
                for dst, n_sent in plan_va.flow.get(src, {}).items():
                    n_sent = float(n_sent)
                    va_user_in[dst] += int(round(n_sent * u / tot))
                    va_atk_in[dst]  += int(round(n_sent * a / tot))
        else:
            plan_va = OffloadPlan(
                flow={aid: {aid: int(W_va[aid])} for aid in self.area_ids},
                assigned_dst={aid: int(W_va[aid]) for aid in self.area_ids},
            )
            va_user_in = {aid: admitted_user[aid] for aid in self.area_ids}
            va_atk_in  = {aid: admitted_atk[aid]  for aid in self.area_ids}

        # Execute VA at each executor
        local_cache: Dict[str, dict] = {
            aid: edges[aid].process_va(
                t=self.t,
                admitted_user_req_in=va_user_in[aid],
                admitted_atk_req_in=va_atk_in[aid],
                attack_dict=obs[aid]["attack_dict"],
                ids_out=ids_out_exec[aid],
            )
            for aid in self.area_ids
        }

        # Restore attack active flags
        if disable_attack:
            for edge in self.edge_areas:
                if hasattr(edge, "_tmp_cur_attacker"):
                    edge.cur_attacker = edge._tmp_cur_attacker
                    del edge._tmp_cur_attacker

        for edge in self.edge_areas:
            assert edge.ids_cpu >= 0.0
            assert edge.va_cpu >= 0.0
            assert edge.ids_cpu + edge.va_cpu <= edge.budget.cpu + 1e-6

        return local_cache
                
        
        
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
        offload=bool(cfg["globals"].get("offload", True)),
    )
    env.reset(cfg["run"]["seed"])
    return env


def build_env_base(cfg_path: str):
    cfg_text = Path(cfg_path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text)
    return build_env_from_cfg(cfg)


class TorchRLEnvWrapper(EnvBase):
    """
    Correct TorchRL EnvBase wrapper.

    - reset() returns a td with keys: observation, done, terminated (and optionally reward)
    - _step(td) returns NEXT td with keys: observation, reward, done, terminated
      TorchRL will create td["next"] automatically.
    """

    def __init__(
        self,
        cfg_path: str,
        decision_interval: int = 300,
        n_actions: int = N_ACTION,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=torch.device(device), batch_size=[])

        self.env = build_env_base(cfg_path)
        self.n_edges = len(self.env.edge_areas)
        self.area_ids = [e.area_id for e in self.env.edge_areas]
        self.episode_id = 0
        self.base_seed = seed
        self.decision_interval = int(decision_interval)
        self.n_actions = int(n_actions)
        # self._step_count = 0

        self.obs_keys = [
            "local_num_req",
            # "attack_drop_rate",
            "attack_in_rate",
            "ema_mom",
            "cpu_to_ids_ratio",
            # "va_cpu_utilization",
            "ids_cpu_utilization",
            # "bw_utilization",
            # "I_net",
        ]

        # Scaling state: pending signed CPU units of overhead remaining
        _cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.scaling_time_steps: List[int] = list(
            _cfg["globals"].get("scaling_time_step", [300, 450, 498, 544])
        )

        # Reward weights — read from config so they're tracked and reproducible
        _reward_cfg = _cfg["globals"].get("reward", {})
        self.reward_alpha = float(_reward_cfg.get("alpha_inv", 0.10))
        self.reward_beta  = float(_reward_cfg.get("beta_inv",  0.20))
        self.reward_gamma = float(_reward_cfg.get("gamma_inv", 0.12))
        self.reward_q_th  = float(_reward_cfg.get("q_th", 0.20))
        self.scaling_quanta: List[float] = [0.5, 1.0, 1.5, 2.0]

        # Method 2 serialised-scaling state
        self.ids_cpu_target: torch.Tensor      # current transition target (= settled when not transitioning)
        self.transition_ticks_remaining: int = 0   # 0 = settled
        self.transition_ticks_total:     int = 1   # avoid div-by-zero

        self.obs_dim = len(self.obs_keys) + 2   # +2: ticks_remaining_norm, delta_in_flight_norm
        self.obs_size = self.n_edges * self.obs_dim

        self.action_dim = self.n_edges
        self._last_action = torch.zeros(self.action_dim, device=self.device, dtype=torch.float32)
        self.scale_step = 0.5  # CPU units per scale
        self.ids_cpu_min = 0.5

        self.ids_cpu = torch.tensor(
            [e.ids_cpu for e in self.env.edge_areas],
            device=self.device,
            dtype=torch.float32,
        )
        self.ids_cpu_settled = self.ids_cpu.clone()
        self.ids_cpu_target  = self.ids_cpu.clone()

        self._set_seed(seed)
        self._make_specs()

    # ---------------- Scaling helpers ----------------

    def _lookup_scaling_duration(self, magnitude: float) -> int:
        """Return ticks to drain `magnitude` CPU units of scaling work."""
        for i, q in enumerate(self.scaling_quanta):
            if magnitude <= q + 1e-9:
                return self.scaling_time_steps[i]
        return self.scaling_time_steps[-1]

    # ---------------- TorchRL required ----------------

    def _set_seed(self, seed: Optional[int]):
        if seed is None:
            return None
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        return seed

    def _make_specs(self):
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self.n_edges, self.obs_dim),
                dtype=torch.float32,
                device=self.device,
            ),
            observation_flat=UnboundedContinuousTensorSpec(
                shape=(self.obs_size,),
                dtype=torch.float32,
                device=self.device,
            ),

            # extra keys that survive rollout collection
            qoe_mean=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self.device,
            ),
            # reward component breakdown (β·λ_res, γ·bcd, α·qoe_shortfall)
            reward_lambda_res=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.float32, device=self.device,
            ),
            reward_benign_col_dmg=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.float32, device=self.device,
            ),
            reward_qoe_penalty=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.float32, device=self.device,
            ),
            qoe_vio_rate=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.float32, device=self.device,
            ),
            t_internal=BoundedTensorSpec(
                low=0,
                high=max(1, int(self.env.t_max)),
                shape=(1,),
                dtype=torch.int64,
                device=self.device,
            ),
        )

        if self.n_edges == 1:
            action_spec = DiscreteTensorSpec(n=self.n_actions, device=self.device)
        else:
            action_spec = MultiDiscreteTensorSpec(
                nvec=[self.n_actions] * self.n_edges,
                device=self.device,
            )
        self.action_spec = CompositeSpec(action=action_spec)

        self.reward_spec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.done_spec = CompositeSpec(
            done=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool,
                device=self.device,
            ),
            truncated=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool,
                device=self.device,
            ),
        )

    # ---------------- Reset / Step ----------------
    def _reset(self, tensordict=None):
        self.episode_id += 1
        episode_seed = self.base_seed + self.episode_id * 1000

        # Seed ONLY torch (policy randomness)
        torch.manual_seed(episode_seed)

        # Reset env with explicit seeds
        self.env.reset(episode_seed)

        # Clear scaling state
        self.ids_cpu_settled             = self.ids_cpu.clone()
        self.ids_cpu_target              = self.ids_cpu.clone()
        self.transition_ticks_remaining  = 0
        self.transition_ticks_total      = 1


        obs = self._build_observation().to(self.device)
        obs_flat = obs.reshape(-1)

        _zero1 = torch.zeros(1, dtype=torch.float32, device=self.device)
        return TensorDict(
            {
                "observation": obs,
                "observation_flat": obs_flat,
                "qoe_mean":              _zero1.clone(),
                "reward_lambda_res":     _zero1.clone(),
                "reward_benign_col_dmg": _zero1.clone(),
                "reward_qoe_penalty":    _zero1.clone(),
                "qoe_vio_rate":          _zero1.clone(),
                "t_internal": torch.tensor([int(self.env.t)], dtype=torch.int64, device=self.device),
                "done":       torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
                "truncated":  torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )
        
    def _decision_ids_util(self) -> float:
        """Max IDS CPU utilization across edges over the last decision window."""
        if len(self.env.history) < self.decision_interval * self.n_edges:
            return 0.0
        records = self.env.history[-self.decision_interval * self.n_edges:]
        df = pd.DataFrame([m.__dict__ for m in records])

        utils = []
        for area_id in self.area_ids:
            g = df[df["area_id"] == area_id]
            if g.empty or "ids_cpu_utilization" not in g.columns:
                continue
            utils.append(float(np.clip(np.mean(g["ids_cpu_utilization"].values), 0.0, 1.0)))
        return float(max(utils)) if utils else 0.0

    def _reactive_delta(self) -> float:
        """Return delta in {-1,0,+1} based on IDS util thresholding."""
        util = self._decision_ids_util()
        if util >= 0.80:
            return 1.0
        if util <= 0.20:
            return -1.0
        return 0.0        

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"].to(self.device)
        # Normalise to shape (n_edges,) for both single- and multi-edge
        if action.dim() == 0:
            action = action.unsqueeze(0).expand(self.n_edges)
        elif action.dim() == 1 and action.shape[0] == 1 and self.n_edges > 1:
            action = action.expand(self.n_edges)

        # Apply per-edge delta commands
        for i in range(self.n_edges):
            delta_cmd = (action[i].float() - ((self.n_actions - 1) / 2.0)) * self.scale_step
            ids_cpu_max = float(self.env.edge_areas[i].budget.cpu - 0.5)
            prev = self.ids_cpu[i].clone()
            self.ids_cpu[i] = torch.clamp(
                self.ids_cpu[i] + delta_cmd, min=self.ids_cpu_min, max=ids_cpu_max
            )
            delta_eff = float((self.ids_cpu[i] - prev).item())

            if self.transition_ticks_remaining <= 0 and abs(delta_eff) > 1e-9:
                self.ids_cpu_target[i] = self.ids_cpu[i]
                gap = abs(float(self.ids_cpu_target[i].item()) - float(self.ids_cpu_settled[i].item()))
                self.transition_ticks_total     = self._lookup_scaling_duration(gap)
                self.transition_ticks_remaining = self.transition_ticks_total

        # Compute per-edge effective CPU for this transition (asymmetric scale-up/down)
        ids_cpu_eff = self.ids_cpu_settled.clone()
        step_overhead = 0.0
        if self.transition_ticks_remaining > 0:
            for i in range(self.n_edges):
                target  = float(self.ids_cpu_target[i].item())
                settled = float(self.ids_cpu_settled[i].item())
                delta_to_settled = target - settled
                if delta_to_settled > 1e-9:      # scale-up: IDS holds at settled
                    ids_cpu_eff[i] = settled
                else:                             # scale-down: IDS drops immediately
                    ids_cpu_eff[i] = target
                step_overhead = min(step_overhead, -abs(delta_to_settled))

        total_reward = 0.0
        total_lambda_res = 0.0
        total_benign_col_dmg = 0.0
        total_qoe_penalty = 0.0
        terminated_flag = False
        steps = 0

        # Simulate decision_interval internal timesteps
        for _ in range(self.decision_interval):
            if self.transition_ticks_remaining > 0:
                self.transition_ticks_remaining -= 1
                if self.transition_ticks_remaining == 0:
                    for i in range(self.n_edges):
                        self.ids_cpu_settled[i] = self.ids_cpu_target[i]
                        queued_delta = float(self.ids_cpu[i].item()) - float(self.ids_cpu_settled[i].item())
                        if abs(queued_delta) > 1e-9:
                            self.ids_cpu_target[i] = self.ids_cpu[i]
                            gap = abs(queued_delta)
                            self.transition_ticks_total     = self._lookup_scaling_duration(gap)
                            self.transition_ticks_remaining = self.transition_ticks_total
                            new_d = float(self.ids_cpu_target[i].item()) - float(self.ids_cpu_settled[i].item())
                            step_overhead = -abs(new_d)
                            if new_d > 1e-9:
                                ids_cpu_eff[i] = float(self.ids_cpu_settled[i].item())
                            else:
                                ids_cpu_eff[i] = float(self.ids_cpu_target[i].item())
                        else:
                            ids_cpu_eff[i] = float(self.ids_cpu_settled[i].item())
                            step_overhead = 0.0

            self.env.step(ids_cpu_eff, step_overhead)
            r = self._build_reward()
            total_reward           += float(r["reward"].item())
            total_lambda_res       += r["lambda_res"]
            total_benign_col_dmg   += r["benign_col_dmg"]
            total_qoe_penalty      += r["qoe_penalty"]
            steps += 1
            if self.env.t >= self.env.t_max:
                terminated_flag = True
                break

        n = max(1, steps)
        reward = torch.tensor([total_reward / n], dtype=torch.float32, device=self.device)

        # QoE violation rate over the decision window
        window = self.env.history[-self.decision_interval * self.n_edges:]
        if window:
            qoes = np.asarray([m.qoe_mean for m in window], dtype=np.float32)
            qoe_vio_rate = float(np.mean(qoes < self.reward_q_th))
        else:
            qoe_vio_rate = 0.0

        # 3) Build aggregated outputs
        obs = self._build_observation().to(self.device)
        obs_flat = obs.reshape(-1)

        terminated = torch.tensor(
            [terminated_flag], dtype=torch.bool, device=self.device
        )
        truncated = torch.zeros(1, dtype=torch.bool, device=self.device)
        done = terminated | truncated

        t_internal_end = int(self.env.t)
        qoe_mean = float(self.env.final_qoe)

        _f32 = lambda v: torch.tensor([v], dtype=torch.float32, device=self.device)
        return TensorDict(
            {
                "observation":      obs,
                "observation_flat": obs_flat,
                "reward":           reward,
                "qoe_mean":              _f32(qoe_mean * 30),
                "reward_lambda_res":     _f32(total_lambda_res    / n),
                "reward_benign_col_dmg": _f32(total_benign_col_dmg / n),
                "reward_qoe_penalty":    _f32(total_qoe_penalty   / n),
                "qoe_vio_rate":          _f32(qoe_vio_rate),
                "t_internal": torch.tensor([t_internal_end], dtype=torch.int64, device=self.device),
                "done":       done,
                "terminated": terminated,
                "truncated":  truncated,
            },
            batch_size=[],
            device=self.device,
        )

    # ---------------- Helpers ----------------
    def _build_observation(self) -> torch.Tensor:
        obs = torch.zeros((self.n_edges, self.obs_dim), dtype=torch.float32, device=self.device)

        if not self.env.history:
            return obs

        records = self.env.history[-self.decision_interval * self.n_edges:]

        df = pd.DataFrame([m.__dict__ for m in records])

        for i, area_id in enumerate(self.area_ids):
            g = df[df["area_id"] == area_id]
            if g.empty:
                continue
            for j, k in enumerate(self.obs_keys):
                vals = g[k].values
                if k == "I_net":
                    obs[i, j] = float(np.sum(vals))
                elif k == "cpu_to_ids_ratio":
                    obs[i, j] = float(vals[-1])
                elif k == "ema_mom":
                    vals_nz = vals[vals != 0.0]
                    if len(vals_nz) == 0:
                        obs[i, j] = 0.0
                        continue
                    obs[i, j] = float(np.mean(vals_nz))
                else:
                    obs[i, j] = float(np.mean(vals))

        # Feature -2: remaining transition ticks normalised by max possible duration ∈ [0, 1]
        # Using max_duration (not T_total) gives a consistent drain rate across all transition
        # sizes, and encodes duration in the initial value (0.55 = T=300, 1.0 = T=544).
        max_dur = float(self.scaling_time_steps[-1])
        obs[:, -2] = float(self.transition_ticks_remaining) / max(max_dur, 1.0)

        # Feature -1: per-edge delta in flight — (ids_cpu_target - ids_cpu_settled) / max_delta ∈ [-1, 1]
        max_delta = self.scale_step * (self.n_actions - 1) / 2.0
        for i in range(self.n_edges):
            delta_in_flight = float(self.ids_cpu_target[i].item()) - float(self.ids_cpu_settled[i].item())
            obs[i, -1] = float(np.clip(delta_in_flight / max(max_delta, 1e-6), -1.0, 1.0))

        return obs

    def _build_reward(self) -> dict:
        """Return reward scalar + the three scaled component magnitudes."""
        _zero = {"reward": torch.zeros(1, dtype=torch.float32, device=self.device),
                 "lambda_res": 0.0, "benign_col_dmg": 0.0, "qoe_penalty": 0.0}
        if not self.env.history:
            return _zero

        last_block = self.env.history[-self.n_edges:]

        qoe        = np.asarray([float(m.qoe_mean)        for m in last_block], dtype=np.float32)
        bcd        = np.asarray([float(m.benign_col_dmg)  for m in last_block], dtype=np.float32)
        attack_in  = np.asarray([float(m.attack_in_rate)  for m in last_block], dtype=np.float32)
        attack_drop = np.asarray([float(m.attack_drop_rate) for m in last_block], dtype=np.float32)

        alpha = self.reward_alpha
        beta  = self.reward_beta
        gamma = self.reward_gamma
        q_th  = self.reward_q_th

        attack_pass = np.maximum(0.0, attack_in - attack_drop)
        lambda_res  = np.divide(attack_pass, attack_in, out=np.zeros_like(attack_pass), where=attack_in > 1e-6).astype(np.float32)
        
        qoe_shortfall = np.maximum(0.0, q_th - qoe) / max(q_th, 1e-6)

        # raw (unweighted) components — used for tracking and print
        # lambda_res only averaged over ticks where attacks are present so that
        # quiet windows (lres=0) don't dilute the penalty and match eval semantics.
        active = attack_in > 1e-6
        r_lambda_res  = float(np.mean(lambda_res[active])) if np.any(active) else 0.0
        r_bcd         = float(np.mean(bcd))             # QoE units
        r_qoe_penalty = float(np.mean(qoe_shortfall))   # normalised shortfall [0, 1]

        reward = -(alpha * r_qoe_penalty + beta * r_lambda_res + gamma * r_bcd)
        return {
            "reward":          torch.tensor([reward], dtype=torch.float32, device=self.device),
            "lambda_res":      r_lambda_res,
            "benign_col_dmg":  r_bcd,
            "qoe_penalty":     r_qoe_penalty,
        }
            
def _reactive_ids_cpu(env, ids_cpu: np.ndarray, decision_interval: int,
                       scale_step: float = 0.5, ids_cpu_min: float = 0.5) -> np.ndarray:
    """Compute new ids_cpu allocation using reactive thresholding on IDS utilization."""
    n_edges = len(env.edge_areas)
    ids_cpu_max = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)

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
    env = build_env_base(cfg_path)

    dfs = []
    for i in range(3):
        env.reset(seed=1000 + i)

        n_edges = len(env.edge_areas)
        ids_cpu_max = np.array([e.budget.cpu - 0.5 for e in env.edge_areas], dtype=np.float32)

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
    (
        all_df.pivot(index="t", columns="area_id", values=["qoe_mean", "qoe_mean_ideal", "benign_col_dmg"])
        .plot(figsize=(10, 4), title="QoE over time")
        .get_figure()
        .savefig(f"{out_dir}/qoe_over_time.png", bbox_inches="tight")
    )

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

    avg_qoe = df["qoe_mean"].mean()
    print(f"Average QoE (qoe_mean): {avg_qoe:.4f}")
    print(f"Average QoE (benign_col_dmg): {df['benign_col_dmg'].mean():.4f}")
    print(f"Plots saved to {out_dir}/")    
        
if __name__ == "__main__":
    test_environment_run("./configs/simulation_ma_0.yaml", plot=True, method="constant", constant_cpu=4.0)
