"""The environment surface the trainer relies on.

Implementations: envs.driving.DrivingEnv, envs.lander.LanderEnv,
envs.pendulum.PendulumEnv, envs.drone.DroneEnv.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Env(Protocol):
    obs_dim: int
    n_continuous: int
    n_binary: int          # buffer act_dim = n_continuous + n_binary
    max_steps: int
    dt: float              # seconds per agent step (ghost replay rate)
    episode_reward: float
    steps: int

    def reset(self) -> np.ndarray: ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...

    def frame_payload(self) -> dict:
        """Scenario-specific body of the live `frame` WS message."""
        ...

    def episode_summary(self) -> dict:
        """Episode reward/steps/cause/metric and an explicit success boolean."""
        ...

    def ghost_sample(self) -> list[float]:
        """[x, y, rot, drift, speed] — one replay row, same shape for all envs."""
        ...


@dataclass(frozen=True)
class TrainingCurriculumSpec:
    """Immutable trainer/environment contract for a gated segment curriculum."""

    id: str
    frontier_order: tuple[int, ...]
    active_frontier_probability: float
    success_rate_threshold: float
    consecutive_confirmations: int
    evaluation_suite_version: str
    evaluation_episodes: int
    evaluation_seed_base: int
    segment_seed_stride: int
    start_state_description: str
    make_evaluation_env: Callable[[int], Env]
    frontier_success_rate_thresholds: tuple[tuple[int, float], ...] = ()

    def success_rate_threshold_for(self, frontier: int) -> float:
        """Return the advancement threshold for one curriculum frontier."""
        if frontier not in self.frontier_order:
            raise ValueError(f"segment {frontier} is outside the curriculum")
        thresholds = dict(self.frontier_success_rate_thresholds)
        return thresholds.get(frontier, self.success_rate_threshold)

    def evaluation_seed(self, segment: int, episode_index: int) -> int:
        if segment not in self.frontier_order:
            raise ValueError(f"segment {segment} is outside the curriculum")
        if not 0 <= episode_index < self.evaluation_episodes:
            raise ValueError("episode index is outside the fixed segment suite")
        return (self.evaluation_seed_base
                + segment * self.segment_seed_stride
                + episode_index)

    def evaluation_suite_id(self, segment: int) -> str:
        if segment not in self.frontier_order:
            raise ValueError(f"segment {segment} is outside the curriculum")
        return (f"{self.evaluation_suite_version}-k{segment}"
                f"-n{self.evaluation_episodes}")

    def protocol(self) -> dict:
        """JSON-safe exact disclosure stored with every policy checkpoint."""
        return {
            "id": self.id,
            "frontier_order": list(self.frontier_order),
            "start_sampling": {
                "active_frontier_probability": self.active_frontier_probability,
                "mastered_later_segments": "uniform remainder",
                "empty_mastered_fallback": "100% active frontier",
            },
            "gate": {
                "success_rate_threshold": self.success_rate_threshold,
                "success_rate_threshold_by_frontier": [
                    {
                        "frontier": frontier,
                        "threshold": self.success_rate_threshold_for(frontier),
                    }
                    for frontier in self.frontier_order
                ],
                "comparison": ">=",
                "consecutive_confirmations": self.consecutive_confirmations,
                "distinct_checkpoint_episodes": True,
            },
            "segment_evaluation": {
                "suite_version": self.evaluation_suite_version,
                "episodes": self.evaluation_episodes,
                "seed_base": self.evaluation_seed_base,
                "segment_seed_stride": self.segment_seed_stride,
                "seed_formula": (
                    "seed_base + segment * segment_seed_stride + episode_index"
                ),
                "deterministic_policy": True,
                "start_state": self.start_state_description,
            },
            "checkpoint_selection": (
                "segment evaluation is training-only diagnostic; fixed full-course "
                "evaluation remains the checkpoint-selection signal"
            ),
        }
