from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch

from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
)

from simulation.environment import Environment
from simulation.edgearea import EdgeArea


class EdgeDefenseTorchRLEnv(EnvBase):
    """

    Action per edge area:
      - ids_ratio_idx: index into ids_ratio_grid
      - pipeline_idx: index into pipeline_actions

    Observation per edge area (example):
      - num_users
      - attack_in_rate
      - ids_coverage
      - mean_latency_ms
      - mean_mota
    """

    def __init__(
        self,
        env: Environment,
        device: Optional[torch.device] = None,
        ids_ratio_grid: Optional[List[float]] = None,
    ):
        super().__init__(device=device)

        self.env = env
        self.device = device if device is not None else torch.device("cpu")

        self.edge_areas: List[EdgeArea] = self.env.edge_areas
        self.n_areas = len(self.edge_areas)

        # discrete grid for cpu_to_ids_ratio
        self.ids_ratio_grid = ids_ratio_grid or [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        self.n_ratio = len(self.ids_ratio_grid)

        # global pipeline action list from the shared pipeline catalog
        # assumes all areas share the same pipeline object and it exposes all_actions()
        pipeline = self.edge_areas[0].pipeline
        self.pipeline_actions: List[Tuple[str, int]] = pipeline.all_actions()
        self.n_pipeline = len(self.pipeline_actions)

        # observation vector dimension
        self.obs_per_area = 5
        self.obs_dim = self.n_areas * self.obs_per_area

        self._make_specs()

    def _make_specs(self) -> None:
        # action is MultiDiscrete, shape [n_areas, 2]
        # each row: [ids_ratio_idx, pipeline_idx]
        self.action_spec = CompositeSpec(
            action=MultiDiscreteTensorSpec(
                nvec=torch.tensor([self.n_ratio, self.n_pipeline], dtype=torch.long).repeat(self.n_areas, 1),
                shape=(self.n_areas, 2),
                device=self.device,
            )
        )

        # observation is a flat float vector
        # bounds are loose, you can tighten later
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self.obs_dim,),
                device=self.device,
                dtype=torch.float32,
            )
        )

        self.reward_spec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec(
                shape=(1,),
                device=self.device,
                dtype=torch.float32,
            )
        )

        self.done_spec = CompositeSpec(
            done=DiscreteTensorSpec(
                n=2,
                shape=(1,),
                device=self.device,
                dtype=torch.bool,
            )
        )

    # -------------------------
    # Helpers
    # -------------------------

    def _apply_action(self, action: torch.Tensor) -> None:
        """
        action: tensor shape (n_areas, 2)
        """
        action = action.detach().cpu().long().numpy()

        for i, ea in enumerate(self.edge_areas):
            ratio_idx = int(action[i, 0])
            pipe_idx = int(action[i, 1])

            ratio_idx = max(0, min(self.n_ratio - 1, ratio_idx))
            pipe_idx = max(0, min(self.n_pipeline - 1, pipe_idx))

            ea.cpu_to_ids_ratio = float(self.ids_ratio_grid[ratio_idx])

            det, h = self.pipeline_actions[pipe_idx]
            # store as a forced action for this step
            # you need a small hook in EdgeArea.choose_action() to use it
            ea.forced_pipeline_action = (det, h)

    def _collect_obs(self, metrics_batch: List[Any]) -> torch.Tensor:
        """
        metrics_batch: list of StepMetrics returned by env.step()
        """
        feats = []
        for i, ea in enumerate(self.edge_areas):
            m = metrics_batch[i]

            num_users = float(len(ea.users))
            attack_in = float(m.attack_in_rate)
            cov = float(m.ids_coverage)
            lat = float(m.mean_latency_ms)
            mota = float(m.mean_mota)

            feats.extend([num_users, attack_in, cov, lat, mota])

        obs = torch.tensor(feats, dtype=torch.float32, device=self.device)
        return obs

    def _reward_from_metrics(self, metrics_batch: List[Any]) -> float:
        # global mean QoE across areas
        vals = [float(m.qoe_mean) for m in metrics_batch]
        return float(np.mean(vals)) if vals else 0.0

    # -------------------------
    # TorchRL required API
    # -------------------------

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        self.env.reset()

        # create initial metrics by stepping with current defaults
        metrics_batch = self.env.step()

        obs = self._collect_obs(metrics_batch)
        reward = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        done = torch.tensor([False], dtype=torch.bool, device=self.device)

        return TensorDict(
            {
                "observation": obs,
                "reward": reward,
                "done": done,
            },
            batch_size=[],
            device=self.device,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]
        self._apply_action(action)

        metrics_batch = self.env.step()

        obs = self._collect_obs(metrics_batch)
        r = self._reward_from_metrics(metrics_batch)
        reward = torch.tensor([r], dtype=torch.float32, device=self.device)

        done_flag = bool(self.env.t >= self.env.t_max)
        done = torch.tensor([done_flag], dtype=torch.bool, device=self.device)

        return TensorDict(
            {
                "observation": obs,
                "reward": reward,
                "done": done,
            },
            batch_size=[],
            device=self.device,
        )