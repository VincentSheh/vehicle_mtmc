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
                    ids_coverage=float(ids_out.get("coverage", 0.0)),
                    attack_in_rate=float(ids_out.get("attack_in_rate", 0.0)),
                    user_drop_rate=float(ids_out.get("user_drop_rate", 0.0)),
                    od_plan=cache["od_plan"],
                    local_num_req=n,
                    ema=cache["ema"],
                    ema_mom=cache["ema_mom"],
                    attack_drop_rate=float(ids_out.get("attack_drop_rate", 0.0)),
                    cpu_to_ids_ratio=edge.ids_cpu / edge.budget.cpu,
                    va_cpu_utilization=cache["va_cpu_utilization"],
                    ids_cpu_utilization=float(ids_out.get("ids_cpu_util", 0.0)),
                    bw_utilization=cache["uplink_util"],
                    overhead=float(overheads[i]),
                )
            )

        self.t += 1
        qoe_slo = []
        if self.t >= self.t_max:
            # Calculate the QoE with SLO Violation Rate
            for i,edge in enumerate(self.edge_areas):
                h = [m for m in self.history if m.area_id == edge.area_id]
                if len(h) == 0:
                    continue

                last_block = h[-self.t_max:]  # sliding window at the end
                qoes = np.asarray([float(m.qoe_mean) for m in last_block], dtype=np.float32)

                # violation indicator: 1 if QoE below threshold else 0
                viol = (qoes < edge.slo_threshold).astype(np.float32)

                # violation rate in [0,1]
                viol_rate = float(viol.mean()) if len(viol) > 0 else 0.0

                # SLO term V in (0,1]
                V_edge = np.exp(-edge.slo_beta * viol_rate)

                # If you want "final QoE with SLO" for this edge:
                qoe_slo.append(float(qoes.mean()) * V_edge)
            self.final_qoe = np.mean(qoe_slo)        
            
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
        decision_interval: int = 3000,
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
        # self._step_count = 0

        self.obs_keys = [
            "local_num_req",
            # "attack_drop_rate",
            "attack_in_rate",
            "ema_mom",
            "cpu_to_ids_ratio",
            # "va_cpu_utilization",
            "ids_cpu_utilization",
            "overhead",
            # "bw_utilization",
            # "I_net",
        ]
        
        self.obs_dim = len(self.obs_keys)
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

        self._set_seed(seed)
        self._make_specs()

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
            qoe_mean=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self.device,
            ),
            t_internal=BoundedTensorSpec(
                low=0,
                high=max(1, int(self.env.t_max)),   # or env.t_max if already built
                shape=(1,),
                dtype=torch.int64,
                device=self.device,
            ),
        )

        # Multi-agent action: one discrete action per edge agent
        self.action_spec = CompositeSpec(
            action=MultiDiscreteTensorSpec(
                nvec=torch.full((self.n_edges,), 3, dtype=torch.int64, device=self.device),
                shape=(self.n_edges,),
                device=self.device,
            )
        )

        # Per-agent reward
        self.reward_spec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.done_spec = CompositeSpec(
            done=BoundedTensorSpec(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            terminated=BoundedTensorSpec(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            truncated=BoundedTensorSpec(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
        )

    # ---------------- Reset / Step ----------------
    def _reset(self, tensordict=None):
        self.episode_id += 1
        episode_seed = self.base_seed + self.episode_id * 1000

        # Seed ONLY torch (policy randomness)
        torch.manual_seed(episode_seed)

        # Reset env with explicit seeds
        self.env.reset(episode_seed)


        obs = self._build_observation().to(self.device)
        obs_flat = obs.reshape(-1)
        self.ids_cpu = torch.tensor(
            [e.ids_cpu for e in self.env.edge_areas],
            device=self.device,
            dtype=torch.float32,
        )        

        return TensorDict(
            {
                # "observation": obs,
                "observation_flat": obs_flat,
                "qoe_mean": torch.zeros(1, dtype=torch.float32, device=self.device),
                "t_internal": torch.tensor([int(self.env.t)], dtype=torch.int64, device=self.device),                
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(1, dtype=torch.bool, device=self.device),
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
        action = tensordict["action"].to(self.device)  # shape (n_edges,), values 0/1/2
        if action.ndim == 0:
            action = action.view(1)

        # delta per agent in {-scale_step, 0, +scale_step}
        delta_cmd = (action.float() - 1.0) * self.scale_step  # shape (n_edges,)

        prev_ids = self.ids_cpu.clone()

        # clamp each edge separately by its own cpu budget
        new_ids = self.ids_cpu + delta_cmd
        for i, edge in enumerate(self.env.edge_areas):
            new_ids[i] = torch.clamp(
                new_ids[i],
                min=self.ids_cpu_min,
                max=float(edge.budget.cpu) - 0.5,
            )

        self.ids_cpu = new_ids
        ids_cpu_vec = self.ids_cpu.clone()

        # overhead per edge equals effective change (signed)
        delta_eff = (self.ids_cpu - prev_ids).detach()
        overheads = delta_eff.detach().cpu().tolist()

        total_reward = torch.zeros(self.n_edges, dtype=torch.float32, device=self.device)
        terminated_flag = False
        steps = 0

        # simulate internal timesteps
        for _ in range(self.decision_interval):
            self.env.step(ids_cpu_vec, overheads)  # overheads is per-edge now
            total_reward += self._build_reward_per_agent()  # (n_edges,)
            steps += 1
            if self.env.t >= self.env.t_max:
                terminated_flag = True
                break

        reward = total_reward / max(1, steps)  # (n_edges,)
        reward_agents = reward  # (n_edges,)
        reward = reward_agents.mean().view(1)  # (1,)        

        obs = self._build_observation().to(self.device)
        obs_flat = obs.reshape(-1)

        terminated = torch.tensor([terminated_flag], dtype=torch.bool, device=self.device)
        truncated = torch.zeros(1, dtype=torch.bool, device=self.device)
        done = terminated | truncated

        t_internal_end = int(self.env.t)
        qoe_mean = float(self.env.final_qoe)

        return TensorDict(
            {
                # "observation": obs,
                "observation_flat": obs_flat,
                "reward": reward,  # per-agent

                "qoe_mean": torch.tensor([qoe_mean * 30], dtype=torch.float32, device=self.device),
                "t_internal": torch.tensor([t_internal_end], dtype=torch.int64, device=self.device),

                "done": done,
                "terminated": terminated,
                "truncated": truncated,
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
                elif k == "cpu_to_ids_ratio" or k=="overhead":
                    obs[i, j] = float(vals[-1])
                elif k == "ema_mom":
                    vals_nz = vals[vals != 0.0]
                    if len(vals_nz) == 0:
                        obs[i, j] = 0.0
                        continue
                    obs[i, j] = float(np.mean(vals_nz))  
                else:
                    obs[i, j] = float(np.mean(vals))

        return obs

    def _build_reward_per_agent(self) -> torch.Tensor:
        if not self.env.history:
            return torch.zeros(self.n_edges, dtype=torch.float32, device=self.device)

        last_block = self.env.history[-self.n_edges:]
        q = np.asarray([float(m.qoe_mean) for m in last_block], dtype=np.float32)

        threshold = 0.35
        alpha = 0.6
        penalty = alpha * (np.maximum(0.0, threshold - q) / threshold) ** 2
        q_adj = q - penalty

        return torch.tensor(q_adj, dtype=torch.float32, device=self.device)
        
def test_environment_run(cfg_path: str, plot=False):
    env = build_env_base(cfg_path)

    dfs = []
    for i in range(1):
        env.reset(seed=1000 + i)

        for _ in range(env.t_max):
            env.step([1.5]*len(env.edge_areas))
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
        all_df.pivot(index="t", columns="area_id", values=["attack_drop_rate", "ema"])
        .plot(figsize=(10, 4), title="Attack In Rate")
        .get_figure()
        .savefig(f"{out_dir}/attack_drop_rate.png", bbox_inches="tight")
    )
    avg_qoe = all_df["qoe_weighted"].mean()
    print(f"Average QoE (qoe_mean): {avg_qoe:.4f}")
    print(f"Plots saved to {out_dir}/")    
        
if __name__ == "__main__":
    test_environment_run("./configs/simulation_ma_0.yaml", plot=True)