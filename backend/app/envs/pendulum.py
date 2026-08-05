"""Classic pendulum swing-up: one continuous torque, theta=0 is upright."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

DT = 0.04
G, M, L = 10.0, 1.0, 1.0
MAX_TORQUE = 2.0
MAX_SPEED = 8.0

PIVOT = (500.0, 350.0)
ROD_LEN = 180.0


def scene() -> dict:
    return {
        "kind": "generic",
        "bounds": [1000, 700],
        "primary_shape": "rod",
        "statics": [{"shape": "circle", "x": PIVOT[0], "y": PIVOT[1], "r": 6}],
    }


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class PendulumEnv:
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)

    obs_dim = 3
    n_continuous = 1
    n_binary = 0
    max_steps = 400
    dt = DT

    def __post_init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        if self.jitter:
            self.theta = self.rng.uniform(-math.pi, math.pi)
            self.theta_dot = self.rng.uniform(-1.0, 1.0)
        else:
            self.theta, self.theta_dot = math.pi, 0.0  # hanging down
        self.steps = 0
        self.episode_reward = 0.0
        self._recent: list[float] = []
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        u = float(np.clip(action[0], -1.0, 1.0)) * MAX_TORQUE
        theta_acc = (3.0 * G / (2.0 * L) * math.sin(self.theta)
                     + 3.0 / (M * L * L) * u)
        self.theta_dot = float(np.clip(self.theta_dot + theta_acc * DT,
                                       -MAX_SPEED, MAX_SPEED))
        self.theta = _wrap(self.theta + self.theta_dot * DT)
        self.steps += 1

        reward = -(self.theta ** 2 + 0.1 * self.theta_dot ** 2 + 0.001 * u ** 2)
        self.episode_reward += reward
        self._recent.append(reward)
        if len(self._recent) > 100:
            self._recent.pop(0)

        done = self.steps >= self.max_steps
        return self._obs(), reward, done, {"truncated": done}

    def _obs(self) -> np.ndarray:
        return np.array([math.cos(self.theta), math.sin(self.theta),
                         self.theta_dot / MAX_SPEED], dtype=np.float32)

    def frame_payload(self) -> dict:
        return {
            "objects": [{"shape": "rod", "x": PIVOT[0], "y": PIVOT[1],
                         "rot": round(self.theta, 3), "len": ROD_LEN}],
        }

    def episode_summary(self) -> dict:
        balance = sum(self._recent) / max(len(self._recent), 1)
        return {
            "reward": round(self.episode_reward, 2),
            "steps": self.steps,
            "cause": "timeout",
            "metric": round(balance, 2),  # ~0 means balanced upright
            "success": balance > -0.5,
        }

    def ghost_sample(self) -> list[float]:
        return [PIVOT[0], PIVOT[1], round(self.theta, 3), 0.0,
                round(abs(self.theta_dot), 1)]
