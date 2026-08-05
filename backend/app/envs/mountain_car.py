"""Continuous Mountain Car benchmark with the canonical hill dynamics."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GOAL_POSITION = 0.45
POWER = 0.0015
# The canonical benchmark equations advance one discrete control step at a time.
# This value is used only for replay pacing and human-readable horizon metadata.
DT = 0.04


def _world(position: float) -> tuple[float, float]:
    x = 100.0 + (position - MIN_POSITION) / (MAX_POSITION - MIN_POSITION) * 800.0
    y = 430.0 - math.sin(3.0 * position) * 150.0
    return x, y


def scene() -> dict:
    points = [_world(float(p)) for p in np.linspace(MIN_POSITION, MAX_POSITION, 180)]
    gx, gy = _world(GOAL_POSITION)
    return {
        "kind": "generic",
        "bounds": [1000, 700],
        "primary_shape": "mountain-car",
        "statics": [
            {"shape": "hill", "points": [[round(x, 1), round(y, 1)] for x, y in points]},
            {"shape": "flag", "x": round(gx, 1), "y": round(gy, 1)},
        ],
    }


@dataclass
class MountainCarEnv:
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)

    obs_dim = 3
    n_continuous = 1
    n_binary = 0
    max_steps = 999
    dt = DT

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> np.ndarray:
        self.position = self.rng.uniform(-0.6, -0.4) if self.jitter else -0.5
        self.velocity = 0.0
        self.peak_position = self.position
        self.steps = 0
        self.episode_reward = 0.0
        self.control_effort = 0.0
        self.cause = "running"
        self._action = 0.0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self._action = float(np.clip(action[0], -1.0, 1.0))
        self.velocity += self._action * POWER - 0.0025 * math.cos(3.0 * self.position)
        self.velocity = float(np.clip(self.velocity, -MAX_SPEED, MAX_SPEED))
        self.position += self.velocity
        self.position = float(np.clip(self.position, MIN_POSITION, MAX_POSITION))
        if self.position <= MIN_POSITION and self.velocity < 0.0:
            self.velocity = 0.0
        self.peak_position = max(self.peak_position, self.position)
        self.steps += 1

        reached = self.position >= GOAL_POSITION
        done = reached or self.steps >= self.max_steps
        effort = 0.1 * self._action ** 2
        self.control_effort += effort
        reward = (100.0 if reached else 0.0) - effort
        if reached:
            self.cause = "summit"
        elif done:
            self.cause = "timeout"
        self.episode_reward += reward
        deadline = done and not reached
        return self._obs(), reward, done, {
            "truncated": deadline,
            "task_deadline": deadline,
        }

    def _obs(self) -> np.ndarray:
        return np.array([
            2.0 * (self.position - MIN_POSITION) / (MAX_POSITION - MIN_POSITION) - 1.0,
            self.velocity / MAX_SPEED,
            max(0.0, 1.0 - self.steps / self.max_steps),
        ], dtype=np.float32)

    def frame_payload(self) -> dict:
        x, y = _world(self.position)
        slope = 3.0 * math.cos(3.0 * self.position)
        return {
            "objects": [{"shape": "mountain-car", "x": round(x, 1),
                         "y": round(y, 1), "rot": round(math.atan2(-slope, 1.0), 3),
                         "force": round(self._action, 2)}],
            "peak_position": round(self.peak_position, 3),
        }

    def episode_summary(self) -> dict:
        success = self.cause == "summit"
        return {
            "reward": round(self.episode_reward, 2),
            "steps": self.steps,
            "cause": self.cause,
            # Completion rate is ranked first. Among successful policies, the
            # benchmark objective is then the energy used to reach the flag.
            "metric": round(self.control_effort, 3) if success else None,
            "failure_progress": None if success else round(self.peak_position, 3),
            "success": success,
        }

    def ghost_sample(self) -> list[float]:
        x, y = _world(self.position)
        slope = 3.0 * math.cos(3.0 * self.position)
        return [round(x, 1), round(y, 1), round(math.atan2(-slope, 1.0), 3),
                0.0, round(abs(self.velocity), 3)]
