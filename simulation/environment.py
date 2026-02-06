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

    # QoE
    qoe_mean: float

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
        self.delay_ms = delay_ms
        self.t_max = int(t_max)
        self.t = 0
        self.history: List[StepMetrics] = []
        self.last_history: List[StepMetrics] = []

        np.random.seed(seed)

    def reset(self, seed):
        self.t = 0
        self.last_history = list(self.history)
        self.history.clear()
        for i, edge in enumerate(self.edge_areas):
            edge.reset(seed=seed + i)        

    def cooperative_offload_ot(
        self,
        states: List[OffloadState],
    ) -> Tuple[List[OffloadState], List[OffloadDecision], np.ndarray]:
        """
        Synchronized greedy OT offloading across all edges.

        Returns:
        - updated states
        - list of OffloadDecision
        - I_net array of shape (n,) where I_net[i] = I_in[i] - I_out[i]
        """
        n = len(states)
        for i, st in enumerate(states):
            st.idx = i

        decisions: List[OffloadDecision] = []

        # per-slot offload counters
        I_in = np.zeros(n, dtype=np.int64)
        I_out = np.zeros(n, dtype=np.int64)

        def finish_time_ms(st: OffloadState) -> float:
            q = st.total_q()
            if q <= 0:
                return 0.0
            if st.avail_cycles_aft_atk_per_ms <= 0:
                return float("inf")
            cycles = q * st.track_cycles_per_obj
            return cycles / max(1e-9, st.avail_cycles_aft_atk_per_ms)

        TOP = list(range(n))
        TOP.sort(key=lambda i: finish_time_ms(states[i]))

        changed = True
        while changed:
            changed = False

            slow_order = sorted(TOP, key=lambda i: finish_time_ms(states[i]), reverse=True)

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
                        remote_cycles / max(1e-9, dst_st.avail_cycles_aft_atk_per_ms)
                        + float(self.delay_ms[src, dst])
                    )

                    # must still meet dst slack window
                    if remote_C > dst_st.latest_track_finish_ms:
                        continue

                    # beneficial move
                    if remote_C < local_C:
                        # move exactly one object
                        src_st.local_q_obj -= 1
                        dst_st.recv_q_obj += 1

                        I_out[src] += 1
                        I_in[dst] += 1

                        decisions.append(OffloadDecision(src_idx=src, dst_idx=dst, num_obj=1))

                        # resort since workloads changed
                        TOP.sort(key=lambda i: finish_time_ms(states[i]))
                        changed = True
                        break

        I_net = I_in - I_out  # shape: (n,)
        return states, decisions, I_net
    
    def step(self, action):
        offload_states = []
        local_cache = {}

        # action expected shape: (n_edges,)
        for i, edge in enumerate(self.edge_areas):
            ai = action[i]
            if torch.is_tensor(ai):
                ai = float(ai.detach().item())
            edge.cpu_to_ids_ratio =  ai / edge.budget.cpu

            cache = edge.step_local(self.t)
            local_cache[edge.area_id] = cache

        # 3) Final QoE computation per edge
        for edge in self.edge_areas:
            cache = local_cache[edge.area_id]
            
            D_Max = edge.constraints["D_Max"]
 
            ids_out = cache["ids_out"]
            # I_net_e = I_net[state.idx]
            self.history.append(
                StepMetrics(
                    t=self.t,
                    area_id=edge.area_id,
                    qoe_mean=float(cache["qoe"]),
                    
                    ids_coverage=float(ids_out.get("coverage", 0.0)),
                    attack_in_rate=float(ids_out.get("attack_in_rate", 0.0)),
                    user_drop_rate=float(ids_out.get("user_drop_rate", 0.0)),
                    od_plan = cache["od_plan"],
                    
                    # RL Observation
                    local_num_req = cache["local_num_request"],
                    ema = cache["ema"],
                    ema_mom = cache["ema_mom"],
                    attack_drop_rate=float(ids_out.get("attack_drop_rate", 0.0)),
                    cpu_to_ids_ratio = edge.cpu_to_ids_ratio,
                    va_cpu_utilization = cache["va_cpu_utilization"],
                    ids_cpu_utilization = ids_out["ids_cpu_util"],
                    bw_utilization = cache["uplink_util"],
                    # I_net = I_net_e,
                    # od_plan = od_plan,
                )
            )

        self.t += 1
        
        
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
            [e.cpu_to_ids_ratio * e.budget.cpu for e in self.env.edge_areas],
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

            # add these so they survive rollout collection
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

        self.action_spec = CompositeSpec(
            action=DiscreteTensorSpec(
                n=3,   # {-1, 0, +1}
                # shape=(self.n_edges,),
                device=self.device,
            )
        )

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


        obs = self._build_observation().to(self.device)
        obs_flat = obs.reshape(-1)

        return TensorDict(
            {
                "observation": obs,
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
        action = tensordict["action"]                       # scalar 0/1/2
        delta = action.to(self.device).float() - 1.0
        # delta = self._reactive_delta()  # -1, 0, +1

        edge = self.env.edge_areas[0]
        self.ids_cpu[0] = torch.clamp(
            self.ids_cpu[0] + delta * self.scale_step,
            min=self.ids_cpu_min,
            max=edge.budget.cpu - 0.5,
            # max=edge.budget.cpu / 2.0,
        )
        ids_cpu = self.ids_cpu.clone()
        # ids_cpu = [1.5]
        # print(action, delta, ids_cpu)

        total_reward = 0.0
        terminated_flag = False
        steps = 0

        # 2) Simulate decision_interval internal timesteps
        for _ in range(self.decision_interval):
            self.env.step(ids_cpu)
            total_reward += float(self._build_reward())
            steps += 1
            if self.env.t >= self.env.t_max:
                terminated_flag = True
                break

        reward = torch.tensor([
            total_reward 
            / max(1, steps)
            ], dtype=torch.float32, device=self.device)

        # 3) Build aggregated outputs
        obs = self._build_observation().to(self.device)
        obs_flat = obs.reshape(-1)

        terminated = torch.tensor(
            [terminated_flag], dtype=torch.bool, device=self.device
        )
        truncated = torch.zeros(1, dtype=torch.bool, device=self.device)
        done = terminated | truncated

        t_internal_end = int(self.env.t)  # end-of-window index (exclusive)

        last_block = self.env.history[-self.n_edges:]
        qoes = [float(m.qoe_mean) for m in last_block]
        qoe_mean = float(np.mean(qoes))

        return TensorDict(
            {
                "observation": obs,
                "observation_flat": obs_flat,
                "reward": reward,

                # logging keys that ParallelEnv can batch safely
                "qoe_mean": torch.tensor([qoe_mean], dtype=torch.float32, device=self.device),
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

        return obs

    def _build_reward(self) -> torch.Tensor:
        if not self.env.history:
            return torch.zeros(1, dtype=torch.float32, device=self.device)
        last_block = self.env.history[-self.n_edges:]
        qoes = [float(m.qoe_mean) for m in last_block]

        qoes = np.asarray(qoes, dtype=np.float32)

        threshold = 0.35
        alpha = 0.6  # max penalty strength
        
        penalty = alpha * (np.maximum(0.0, threshold - qoes) / threshold) ** 2        
        # penalty = alpha * np.maximum(0.0, 1.0 - qoes / threshold)

        qoes_adj = qoes - penalty

        reward = float(qoes_adj.mean())

        # qoes = np.array(qoes, dtype=float)
        # qoes[qoes < 0.2] -= 0.6

        # mean_qoe = float(qoes.mean())
        # reward = mean_qoe

        return torch.tensor(
            [reward],
            dtype=torch.float32,
            device=self.device,
        )        
        
def test_environment_run(cfg_path: str, plot=False):
    env = build_env_base(cfg_path)

    dfs = []
    for i in range(5):
        env.reset(seed=1000 + i)

        for _ in range(env.t_max):
            env.step([0.0])

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
    avg_qoe = df["qoe_mean"].mean()
    print(f"Average QoE (qoe_mean): {avg_qoe:.4f}")
    print(f"Plots saved to {out_dir}/")    
        
if __name__ == "__main__":
    test_environment_run("./configs/simulation_0.yaml", plot=True)