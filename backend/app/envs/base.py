"""The environment surface the trainer relies on.

Implementations: envs.driving.DrivingEnv, envs.lander.LanderEnv,
envs.pendulum.PendulumEnv, envs.drone.DroneEnv.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

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
