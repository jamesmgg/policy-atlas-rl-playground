"""Continuous-action Cart-Pole using the canonical control equations.

The dynamics and thresholds follow the classic CartPole benchmark; PPO emits
a bounded continuous force instead of choosing only left or right.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
LENGTH = 0.5                 # half the pole length
POLEMASS_LENGTH = MASS_POLE * LENGTH
FORCE_MAG = 10.0
DT = 0.02
X_THRESHOLD = 2.4
THETA_THRESHOLD = 12.0 * math.pi / 180.0

RAIL_Y = 520.0
RAIL_X0, RAIL_X1 = 150.0, 850.0


def _screen_x(x: float) -> float:
    return 500.0 + x / X_THRESHOLD * 350.0


def scene() -> dict:
    return {
        "kind": "generic",
        "bounds": [1000, 700],
        "primary_shape": "cartpole",
        "statics": [
            {"shape": "line", "points": [[RAIL_X0, RAIL_Y], [RAIL_X1, RAIL_Y]]},
            {"shape": "line", "points": [[RAIL_X0, RAIL_Y - 14], [RAIL_X0, RAIL_Y + 14]],
             "color": "danger"},
            {"shape": "line", "points": [[RAIL_X1, RAIL_Y - 14], [RAIL_X1, RAIL_Y + 14]],
             "color": "danger"},
        ],
    }


@dataclass
class CartPoleEnv:
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)

    obs_dim = 4
    n_continuous = 1
    n_binary = 0
    max_steps = 500
    dt = DT

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> np.ndarray:
        if self.jitter:
            self.x, self.x_dot, self.theta, self.theta_dot = (
                self.rng.uniform(-0.05, 0.05) for _ in range(4)
            )
        else:
            # A fixed small perturbation keeps evaluation/replay deterministic
            # without handing an untrained zero-output policy a solved state.
            self.x = self.x_dot = self.theta_dot = 0.0
            self.theta = 0.05
        self.steps = 0
        self.episode_reward = 0.0
        self.cause = "running"
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        force = float(np.clip(action[0], -1.0, 1.0)) * FORCE_MAG
        costheta, sintheta = math.cos(self.theta), math.sin(self.theta)
        temp = (force + POLEMASS_LENGTH * self.theta_dot ** 2 * sintheta) / TOTAL_MASS
        theta_acc = ((GRAVITY * sintheta - costheta * temp) /
                     (LENGTH * (4.0 / 3.0 - MASS_POLE * costheta ** 2 / TOTAL_MASS)))
        x_acc = temp - POLEMASS_LENGTH * theta_acc * costheta / TOTAL_MASS

        # Semi-implicit Euler is more stable than explicit Euler at the same dt.
        self.x_dot += DT * x_acc
        self.theta_dot += DT * theta_acc
        self.x += DT * self.x_dot
        self.theta += DT * self.theta_dot
        self.steps += 1

        failed = abs(self.x) > X_THRESHOLD or abs(self.theta) > THETA_THRESHOLD
        done = failed or self.steps >= self.max_steps
        reward = 1.0 if not failed else 0.0
        if failed:
            self.cause = "out_of_bounds" if abs(self.x) > X_THRESHOLD else "tipped"
        elif done:
            self.cause = "balanced"
        self.episode_reward += reward
        return self._obs(), reward, done, {"truncated": done and not failed}

    def _obs(self) -> np.ndarray:
        return np.array([
            self.x / X_THRESHOLD,
            self.x_dot / 3.0,
            self.theta / THETA_THRESHOLD,
            self.theta_dot / 3.5,
        ], dtype=np.float32)

    def frame_payload(self) -> dict:
        return {
            "objects": [{"shape": "cartpole", "x": round(_screen_x(self.x), 1),
                         "y": RAIL_Y, "rot": round(self.theta, 4), "len": 150.0}],
            "balance_time": round(self.steps * DT, 2),
        }

    def episode_summary(self) -> dict:
        success = self.cause == "balanced"
        safe_steps = self.steps if success else max(self.steps - 1, 0)
        return {
            "reward": round(self.episode_reward, 2),
            "steps": self.steps,
            "cause": self.cause,
            "metric": round(safe_steps * DT, 2),
            "success": success,
        }

    def ghost_sample(self) -> list[float]:
        return [round(_screen_x(self.x), 1), RAIL_Y, round(self.theta, 4),
                0.0, round(abs(self.x_dot), 2)]
