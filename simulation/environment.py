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
    attack_cpu_frac: float
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
        self.rng = np.random.default_rng(seed)
        self.active_attack = None        

    def _area_ids(self) -> List[int]:
        return self.area_ids

    def _edge_by_id(self) -> Dict[str, EdgeArea]:
        return {str(e.area_id): e for e in self.edge_areas}

    def reset(self, seed):
        self.t = 0
        self.last_history = list(self.history)
        self.history.clear()
        self.final_qoe = 0

        # reset edges
        for i, edge in enumerate(self.edge_areas):
            edge.reset(seed=seed + i * 99)
            # edge.ids_cpu = 4.0

        # ---- enforce: only ONE attacker active in this episode ----
        # gather all attacker instances across all edges
        candidates: List[Tuple[int, int]] = []
        for ei, edge in enumerate(self.edge_areas):
            for ai, atk in enumerate(getattr(edge, "attackers", [])):
                candidates.append((ei, ai))

        # disable all attackers by default
        for edge in self.edge_areas:
            for atk in getattr(edge, "attackers", []):
                atk.episode_active = False

        if candidates:
            # pick exactly one attacker to activate
            # if you want deterministic per seed, reseed rng here
            rng = np.random.default_rng(seed)
            ei, ai = candidates[int(rng.integers(0, len(candidates)))]

            self.edge_areas[ei].attackers[ai].episode_active = True
            self.active_attack = (ei, ai)
        else:
            self.active_attack = None

    def _compute_tau_loc(self, W_va: Dict[int, float], c_va: Dict[int, float]) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for e, w in W_va.items():
            out[e] = float(w) / max(float(c_va[e]), 1e-9)
        return out

    def _snapshot_edges(self):
        return [edge.get_state() for edge in self.edge_areas], np.random.get_state()

    def _restore_edges(self, snapshot):
        edge_states, rng_state = snapshot
        np.random.set_state(rng_state)
        for i, edge in enumerate(self.edge_areas):
            edge.set_state(edge_states[i])

    def _run_step_once(
        self,
        ids_cpus,
        overhead: float = 0.0,
        disable_attack: bool = False,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Execute one simulation tick of the full multi-edge pipeline.

        Returns (local_cache, obs, va_atk_in_dst) without modifying self.t or self.history.
        """
        edges = self._edge_by_id()
        area_ids = self._area_ids()

        if isinstance(ids_cpus, torch.Tensor):
            ids_cpus = ids_cpus.detach().cpu().tolist()

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

            ids_cpu_eff = float(ids_cpus[i]) - float(overhead_ids)
            va_cpu_eff  = float(edge.budget.cpu) - float(ids_cpus[i]) - float(overhead_va)

            ids_cpu_eff = float(np.clip(ids_cpu_eff, 0.0, edge.budget.cpu))
            va_cpu_eff  = float(np.clip(va_cpu_eff,  0.0, edge.budget.cpu))

            total = ids_cpu_eff + va_cpu_eff
            if total > edge.budget.cpu:
                excess = total - edge.budget.cpu
                va_cpu_eff = max(0.5, va_cpu_eff - excess)

            edge.ids_cpu = ids_cpu_eff
            edge.va_cpu  = va_cpu_eff

        # optionally disable attacks for ideal (no-attack) run
        if disable_attack:
            for edge in self.edge_areas:
                for atk in getattr(edge, "attackers", []):
                    atk._tmp_prev_active = getattr(atk, "episode_active", True)
                    atk.episode_active = False

        try:
            # 1) observe arrivals at owners (ingress)
            obs = {eid: edges[eid].observe_arrivals(self.t) for eid in area_ids}

            # A) IDS OFFLOAD + EXECUTE
            W_def_src = {eid: float(obs[eid]["total_workload_in"]) for eid in area_ids}
            c_def_dst = {eid: float(edges[eid].ids_cpu) for eid in area_ids}

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

            admitted_user_exec = {e: 0 for e in area_ids}
            admitted_atk_exec  = {e: 0 for e in area_ids}

            for e_exec in area_ids:
                admitted_user_exec[e_exec] = int(ids_out_exec[e_exec].get("user_pass_cnt", exec_user_in[e_exec]))
                admitted_atk_exec[e_exec]  = int(ids_out_exec[e_exec].get("atk_pass_cnt", 0))

            W_va_src = {
                eid: float(admitted_user_exec[eid] + admitted_atk_exec[eid])
                for eid in area_ids
            }
            c_va_dst = {eid: float(edges[eid].va_cpu) for eid in area_ids}
            tau_loc  = self._compute_tau_loc(W_va_src, c_va_dst)

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

            va_user_in_dst = {e: 0 for e in area_ids}
            va_atk_in_dst  = {e: 0 for e in area_ids}

            for e_src in area_ids:
                u = float(admitted_user_exec[e_src])
                a = float(admitted_atk_exec[e_src])
                tot = max(u + a, 1.0)
                u_share = u / tot
                a_share = a / tot

                for e_dst, n_sent in plan_va.flow.get(e_src, {}).items():
                    n_sent = float(n_sent)
                    va_user_in_dst[e_dst] += int(round(n_sent * u_share))
                    va_atk_in_dst[e_dst]  += int(round(n_sent * a_share))

            local_cache = {}
            for e_exec in area_ids:
                edge = edges[e_exec]
                cache = edge.process_va(
                    t=self.t,
                    admitted_user_req_in=int(va_user_in_dst.get(e_exec, 0)),
                    admitted_atk_req_in=int(va_atk_in_dst.get(e_exec, 0)),
                    attack_dict=obs[e_exec]["attack_dict"],
                    ids_out=ids_out_exec[e_exec],
                )
                local_cache[e_exec] = cache

        finally:
            if disable_attack:
                for edge in self.edge_areas:
                    for atk in getattr(edge, "attackers", []):
                        if hasattr(atk, "_tmp_prev_active"):
                            atk.episode_active = atk._tmp_prev_active
                            del atk._tmp_prev_active

        return local_cache, obs, va_atk_in_dst

    def step(self, ids_cpus, overhead: float = 0.0):
        if isinstance(overhead, (list, tuple, np.ndarray)):
            overheads = [float(x) for x in overhead]
            assert len(overheads) == len(self.edge_areas)
        else:
            overheads = [float(overhead)] * len(self.edge_areas)

        area_ids = self._area_ids()
        edges = self._edge_by_id()

        # snapshot → ideal run (no attacks) → restore → real run
        snapshot = self._snapshot_edges()
        ideal_cache, _, _ = self._run_step_once(ids_cpus, overhead, disable_attack=True)
        self._restore_edges(snapshot)
        real_cache, _, va_atk_in_dst = self._run_step_once(ids_cpus, overhead, disable_attack=False)

        for i, eid in enumerate(area_ids):
            edge  = edges[eid]
            cache = real_cache[eid]
            cache_ideal = ideal_cache[eid]
            ids_out = cache["ids_out"]

            self.history.append(
                StepMetrics(
                    t=self.t,
                    area_id=eid,
                    qoe_mean=float(cache["qoe"]),
                    qoe_mean_ideal=float(cache_ideal["qoe"]),
                    benign_col_dmg=float(cache_ideal["qoe"] - cache["qoe"]),
                    ids_coverage=float(ids_out.get("coverage", 0.0)),
                    attack_in_rate=float(ids_out.get("atk_in_cnt", 0.0)),
                    attack_drop_rate=float(ids_out.get("atk_drop_cnt", 0.0)),
                    user_drop_rate=float(ids_out.get("user_drop_cnt", 0.0)),
                    od_plan=cache["od_plan"],
                    local_num_req=int(cache.get("local_num_request", 0)),
                    ema=cache["ema"],
                    ema_mom=cache["ema_mom"],
                    cpu_to_ids_ratio=edge.ids_cpu / edge.budget.cpu,
                    va_cpu_utilization=cache["va_cpu_utilization"],
                    attack_cpu_frac=float(cache.get("attack_cpu_frac", 0.0)),
                    ids_cpu_utilization=float(ids_out.get("ids_cpu_util", 0.0)),
                    bw_utilization=cache["uplink_util"],
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

                q = np.asarray([float(m.qoe_mean) for m in last_block], dtype=np.float32)

                viol = (q < float(edge.slo_threshold)).astype(np.float32)
                viol_rate = float(viol.mean()) if viol.size else 0.0
                V_edge = float(np.exp(-float(edge.slo_beta) * viol_rate))

                score = float(q.mean()) * V_edge
                w = float(np.sum([int(m.local_num_req) for m in last_block]))

                num += w * score
                den += w

            self.final_qoe = (num / den) if den > 0.0 else 0.0
        
def build_env_base(cfg_path: str):
    cfg_text = Path(cfg_path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text)

    globals_cfg = load_globals(cfg)

    # --------------------------------------------------
    # Build AttackTypeLibrary (optional)
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

        # When attack_type_library is set, attackers are created dynamically at episode reset
        attackers = []

        if attack_type_library is None:
            for atk_ref in area_cfg.get("attackers", []):
                atk_type = atk_ref["attacker_type"]

                if atk_type not in cfg["globals"].get("attack", {}):
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
            attack_type_library=attack_type_library,
            t_max=cfg["run"]["t_max"],
            dirichlet_alpha=float(area_cfg.get("dirichlet_concentration", 1.0)),
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
    
    
class TorchRLEnvWrapper(EnvBase):
    """
    TorchRL EnvBase wrapper for the multi-edge simulation.

    - _reset() returns a td with keys: observation, done, terminated
    - _step(td) returns NEXT td with keys: observation, reward, done, terminated
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

        self.obs_keys = [
            "local_num_req",
            "attack_in_rate",
            "ema_mom",
            "cpu_to_ids_ratio",
            "ids_cpu_utilization",
        ]

        _cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.scaling_time_steps: List[int] = list(
            _cfg["globals"].get("scaling_time_step", [300, 450, 498, 544])
        )

        _reward_cfg = _cfg["globals"].get("reward", {})
        self.reward_alpha = float(_reward_cfg.get("alpha_inv", 0.10))
        self.reward_beta  = float(_reward_cfg.get("beta_inv",  0.20))
        self.reward_gamma = float(_reward_cfg.get("gamma_inv", 0.12))
        self.reward_q_th  = float(_reward_cfg.get("q_th", 0.20))
        self.scaling_quanta: List[float] = [0.5]

        self.transition_ticks_remaining: int = 0
        self.transition_ticks_total:     int = 1

        self.obs_dim  = len(self.obs_keys)
        self.obs_size = self.n_edges * self.obs_dim
        self.action_dim = self.n_edges
        self._last_action = torch.zeros(self.action_dim, device=self.device, dtype=torch.float32)
        self.scale_step  = 0.5
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

    def _lookup_scaling_duration(self, magnitude: float) -> int:
        for i, q in enumerate(self.scaling_quanta):
            if magnitude <= q + 1e-9:
                return self.scaling_time_steps[i]
        return self.scaling_time_steps[-1]

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
            qoe_mean=UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32, device=self.device),
            reward_lambda_res=UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32, device=self.device),
            reward_benign_col_dmg=UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32, device=self.device),
            reward_qoe_penalty=UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32, device=self.device),
            qoe_vio_rate=UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32, device=self.device),
            t_internal=BoundedTensorSpec(
                low=0,
                high=max(1, int(self.env.t_max)),
                shape=(1,),
                dtype=torch.int64,
                device=self.device,
            ),
        )

        self.action_spec = CompositeSpec(
            action=DiscreteTensorSpec(n=self.n_actions, device=self.device)
        )

        self.reward_spec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32, device=self.device)
        )

        self.done_spec = CompositeSpec(
            done=BoundedTensorSpec(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            terminated=BoundedTensorSpec(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            truncated=BoundedTensorSpec(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
        )

    def _reset(self, tensordict=None):
        self.episode_id += 1
        episode_seed = self.base_seed + self.episode_id * 1000
        torch.manual_seed(episode_seed)
        self.env.reset(episode_seed)

        self.ids_cpu_settled            = self.ids_cpu.clone()
        self.ids_cpu_target             = self.ids_cpu.clone()
        self.transition_ticks_remaining = 0
        self.transition_ticks_total     = 1

        obs = self._build_observation().to(self.device)
        obs_flat = obs.reshape(-1)

        _zero1 = torch.zeros(1, dtype=torch.float32, device=self.device)
        return TensorDict(
            {
                "observation":           obs,
                "observation_flat":      obs_flat,
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

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]
        delta_cmd = (action.to(self.device).float() - ((self.n_actions - 1) / 2.0)) * self.scale_step

        prev_ids = self.ids_cpu[0].clone()
        edge = self.env.edge_areas[0]
        ids_cpu_max_val = float(edge.budget.cpu - 0.5)

        _settled = float(self.ids_cpu_settled[0].item())
        _max_q   = float(self.scaling_quanta[-1])
        self.ids_cpu[0] = torch.clamp(
            self.ids_cpu[0] + delta_cmd,
            min=max(self.ids_cpu_min, _settled - _max_q),
            max=min(ids_cpu_max_val,  _settled + _max_q),
        )
        delta_eff = float((self.ids_cpu[0] - prev_ids).item())

        if self.transition_ticks_remaining <= 0:
            if abs(delta_eff) > 1e-9:
                self.ids_cpu_target[0] = self.ids_cpu[0]
                gap = abs(float(self.ids_cpu_target[0].item()) - float(self.ids_cpu_settled[0].item()))
                self.transition_ticks_total     = self._lookup_scaling_duration(gap)
                self.transition_ticks_remaining = self.transition_ticks_total

        target  = float(self.ids_cpu_target[0].item())
        settled = float(self.ids_cpu_settled[0].item())
        delta_to_settled = target - settled

        ids_cpu_eff = self.ids_cpu_settled.clone()
        if self.transition_ticks_remaining > 0:
            if delta_to_settled > 1e-9:
                ids_cpu_eff[0] = settled
            else:
                ids_cpu_eff[0] = target
        step_overhead = -abs(delta_to_settled) if self.transition_ticks_remaining > 0 else 0.0

        total_reward = 0.0
        total_lambda_res = 0.0
        total_benign_col_dmg = 0.0
        total_qoe_penalty = 0.0
        terminated_flag = False
        steps = 0

        for _ in range(self.decision_interval):
            if self.transition_ticks_remaining > 0:
                self.transition_ticks_remaining -= 1
                if self.transition_ticks_remaining == 0:
                    self.ids_cpu_settled[0] = self.ids_cpu_target[0]
                    queued_delta = float(self.ids_cpu[0].item()) - float(self.ids_cpu_settled[0].item())
                    if abs(queued_delta) > 1e-9:
                        self.ids_cpu_target[0] = self.ids_cpu[0]
                        gap = abs(queued_delta)
                        self.transition_ticks_total     = self._lookup_scaling_duration(gap)
                        self.transition_ticks_remaining = self.transition_ticks_total
                        new_d = float(self.ids_cpu_target[0].item()) - float(self.ids_cpu_settled[0].item())
                        step_overhead = -abs(new_d)
                        if new_d > 1e-9:
                            ids_cpu_eff[0] = float(self.ids_cpu_settled[0].item())
                        else:
                            ids_cpu_eff[0] = float(self.ids_cpu_target[0].item())
                    else:
                        ids_cpu_eff[0] = float(self.ids_cpu_settled[0].item())
                        step_overhead = 0.0

            self.env.step(ids_cpu_eff, step_overhead)
            r = self._build_reward()
            total_reward         += float(r["reward"].item())
            total_lambda_res     += r["lambda_res"]
            total_benign_col_dmg += r["benign_col_dmg"]
            total_qoe_penalty    += r["qoe_penalty"]
            steps += 1
            if self.env.t >= self.env.t_max:
                terminated_flag = True
                break

        n = max(1, steps)
        reward = torch.tensor([total_reward / n], dtype=torch.float32, device=self.device)

        window = self.env.history[-self.decision_interval * self.n_edges:]
        if window:
            qoes = np.asarray([m.qoe_mean for m in window], dtype=np.float32)
            qoe_vio_rate = float(np.mean(qoes < self.reward_q_th))
        else:
            qoe_vio_rate = 0.0

        obs = self._build_observation().to(self.device)
        obs_flat = obs.reshape(-1)

        terminated = torch.tensor([terminated_flag], dtype=torch.bool, device=self.device)
        truncated  = torch.zeros(1, dtype=torch.bool, device=self.device)
        done = terminated | truncated

        _f32 = lambda v: torch.tensor([v], dtype=torch.float32, device=self.device)
        return TensorDict(
            {
                "observation":           obs,
                "observation_flat":      obs_flat,
                "reward":                reward,
                "qoe_mean":              _f32(float(self.env.final_qoe) * 30),
                "reward_lambda_res":     _f32(total_lambda_res    / n),
                "reward_benign_col_dmg": _f32(total_benign_col_dmg / n),
                "reward_qoe_penalty":    _f32(total_qoe_penalty   / n),
                "qoe_vio_rate":          _f32(qoe_vio_rate),
                "t_internal": torch.tensor([int(self.env.t)], dtype=torch.int64, device=self.device),
                "done":       done,
                "terminated": terminated,
                "truncated":  truncated,
            },
            batch_size=[],
            device=self.device,
        )

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
                    obs[i, j] = float(np.mean(vals_nz)) if len(vals_nz) > 0 else 0.0
                else:
                    obs[i, j] = float(np.mean(vals))

        return obs

    def _build_reward(self) -> dict:
        _zero = {"reward": torch.zeros(1, dtype=torch.float32, device=self.device),
                 "lambda_res": 0.0, "benign_col_dmg": 0.0, "qoe_penalty": 0.0}
        if not self.env.history:
            return _zero

        last_block = self.env.history[-self.n_edges:]

        qoe        = np.asarray([float(m.qoe_mean)        for m in last_block], dtype=np.float32)
        bcd        = np.asarray([float(m.benign_col_dmg)  for m in last_block], dtype=np.float32)
        attack_in  = np.asarray([float(m.attack_in_rate)  for m in last_block], dtype=np.float32)
        attack_drop = np.asarray([float(m.attack_drop_rate) for m in last_block], dtype=np.float32)

        attack_pass   = np.maximum(0.0, attack_in - attack_drop)
        lambda_res    = np.divide(attack_pass, attack_in, out=np.zeros_like(attack_pass), where=attack_in > 1e-6).astype(np.float32)
        qoe_shortfall = np.maximum(0.0, self.reward_q_th - qoe) / max(self.reward_q_th, 1e-6)

        active = attack_in > 1e-6
        r_lambda_res  = float(np.mean(lambda_res[active])) if np.any(active) else 0.0
        r_bcd         = float(np.mean(bcd))
        r_qoe_penalty = float(np.mean(qoe_shortfall))

        reward = -(self.reward_alpha * r_qoe_penalty + self.reward_beta * r_lambda_res + self.reward_gamma * r_bcd)
        return {
            "reward":         torch.tensor([reward], dtype=torch.float32, device=self.device),
            "lambda_res":     r_lambda_res,
            "benign_col_dmg": r_bcd,
            "qoe_penalty":    r_qoe_penalty,
        }


def _reactive_ids_cpu(env, ids_cpu: np.ndarray, decision_interval: int,
                      scale_step: float = 0.5, ids_cpu_min: float = 0.5) -> np.ndarray:
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

    return np.clip(ids_cpu + delta * scale_step, ids_cpu_min, ids_cpu_max)


def test_environment_run(cfg_path: str, plot=False, decision_interval: int = 500,
                         method: str = "reactive", constant_cpu: float = 0.5):
    """
    method: "reactive"  - threshold-based IDS CPU adjustment every decision_interval steps
            "constant"  - fixed ids_cpu = constant_cpu for all steps
    """
    env = build_env_base(cfg_path)

    dfs = []
    for i in range(5):
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

    (
        all_df.pivot(index="t", columns="area_id", values=["qoe_mean", "qoe_mean_ideal", "benign_col_dmg"])
        .plot(figsize=(10, 4), title="QoE over time")
        .get_figure()
        .savefig(f"{out_dir}/qoe_over_time.png", bbox_inches="tight")
    )

    (
        all_df.pivot(index="t", columns="area_id", values="ids_cpu_utilization")
        .plot(figsize=(10, 4), title="IDS CPU Utilization")
        .get_figure()
        .savefig(f"{out_dir}/ids_cpu_utilization.png", bbox_inches="tight")
    )

    ax = all_df.pivot(index="t", columns="area_id", values="local_num_req").plot(
        figsize=(10, 4), title="Num Request", alpha=0.25,
    )
    all_df.pivot(index="t", columns="area_id", values="local_num_req") \
        .rolling(500, min_periods=1).mean().plot(ax=ax, linewidth=2)
    ax.get_figure().savefig(f"{out_dir}/local_num_req_combined.png", bbox_inches="tight")

    (
        all_df.pivot(index="t", columns="area_id", values="ema_mom")
        .plot(figsize=(10, 4), title="EMA Momentum")
        .get_figure()
        .savefig(f"{out_dir}/ema_mom.png", bbox_inches="tight")
    )

    (
        all_df.pivot(index="t", columns="area_id", values=["attack_in_rate", "attack_drop_rate"])
        .plot(figsize=(10, 4), title="Attack In Rate")
        .get_figure()
        .savefig(f"{out_dir}/attack_in_rate.png", bbox_inches="tight")
    )


    avg_qoe = all_df["qoe_mean"].mean()
    avg_bcd = all_df["benign_col_dmg"].mean()
    print(f"Average QoE (qoe_mean): {avg_qoe:.4f}")
    print(f"Average Benign Collision Damage: {avg_bcd:.4f}")
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    test_environment_run("./configs/simulation_ma_0.yaml", plot=True, method="constant", constant_cpu=4.0)