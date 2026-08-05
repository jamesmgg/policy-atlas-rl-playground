"""2D quadcopter: two rotor thrusts, fly a fixed 5-waypoint course."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

DT = 0.04
G = 50.0
T_MAX = 70.0       # per rotor; hover needs ~0.36 each
TORQUE = 8.0
TIP_OVER = 1.3
CAPTURE_DIST = 25.0

WAYPOINTS: list[tuple[float, float]] = [
    (250, 500), (700, 420), (450, 180), (820, 160), (150, 250),
]
START = (500.0, 560.0)
CANONICAL_START_PROBABILITY = 0.5
TRAINING_START_DISTRIBUTION = (
    "50% canonical full-course start; 50% uniform later waypoint "
    "segments (targets 2-5) from the preceding waypoint"
)


def scene() -> dict:
    return {
        "kind": "generic",
        "bounds": [1000, 700],
        "primary_shape": "drone",
        "statics": [{"shape": "circle", "x": x, "y": y, "r": CAPTURE_DIST,
                     "color": "rgba(77,208,225,0.25)"}
                    for x, y in WAYPOINTS],
    }


@dataclass
class DroneEnv:
    jitter: bool = True
    waypoint_start_curriculum: bool = False
    rng: random.Random = field(default_factory=random.Random)

    obs_dim = 9
    n_continuous = 2
    n_binary = 0
    max_steps = 900
    dt = DT

    def __post_init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        self.x, self.y = START
        self.k = 0
        if (self.waypoint_start_curriculum
                and self.rng.random() >= CANONICAL_START_PROBABILITY):
            self.k = self.rng.randrange(1, len(WAYPOINTS))
            self.x, self.y = WAYPOINTS[self.k - 1]
        if self.jitter:
            self.x += self.rng.uniform(-30.0, 30.0)
        self.vx = self.vy = 0.0
        self.theta = self.omega = 0.0
        self.steps = 0
        self.episode_reward = 0.0
        self.cause = "running"
        self._d_prev = self._dist()
        return self._obs()

    def _target(self) -> tuple[float, float]:
        return WAYPOINTS[min(self.k, len(WAYPOINTS) - 1)]

    def _dist(self) -> float:
        wx, wy = self._target()
        return math.hypot(self.x - wx, self.y - wy)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        t_l = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        t_r = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))

        thrust = (t_l + t_r) * T_MAX
        self.vx += thrust * math.sin(self.theta) * DT
        self.vy += (-thrust * math.cos(self.theta) + G) * DT
        self.omega += (t_r - t_l) * TORQUE * DT
        self.theta += self.omega * DT
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.steps += 1

        d = self._dist()
        reward = (0.05 * (self._d_prev - d) - 0.002 * abs(self.omega)
                  - 0.005 * (t_l ** 2 + t_r ** 2))

        done = False
        if d < CAPTURE_DIST:
            reward += 20.0
            self.k += 1
            if self.k >= len(WAYPOINTS):
                reward += 50.0
                done, self.cause = True, "complete"
            else:
                d = self._dist()
        if not done and (abs(self.theta) > TIP_OVER
                         or self.x < 0 or self.x > 1000
                         or self.y < 0 or self.y > 700):
            reward -= 50.0
            done, self.cause = True, "crash"
        elif not done and self.steps >= self.max_steps:
            done, self.cause = True, "timeout"
        self._d_prev = d

        self.episode_reward += reward
        deadline = done and self.cause == "timeout"
        return self._obs(), reward, done, {
            "truncated": deadline,
            "task_deadline": deadline,
        }

    def _obs(self) -> np.ndarray:
        wx, wy = self._target()
        return np.array([
            (wx - self.x) / 300.0,
            (wy - self.y) / 300.0,
            self.vx / 60.0,
            self.vy / 60.0,
            math.sin(self.theta),
            math.cos(self.theta),
            self.omega / 4.0,
            self.k / len(WAYPOINTS),
            max(0.0, 1.0 - self.steps / self.max_steps),
        ], dtype=np.float32)

    def frame_payload(self) -> dict:
        wx, wy = self._target()
        return {
            "objects": [
                {"shape": "drone", "x": round(self.x, 1), "y": round(self.y, 1),
                 "rot": round(self.theta, 3)},
                {"shape": "target", "x": wx, "y": wy},
            ],
            "waypoints": self.k,
        }

    def episode_summary(self) -> dict:
        return {
            "reward": round(self.episode_reward, 2),
            "steps": self.steps,
            "cause": self.cause,
            "metric": float(self.k),
            "success": self.cause == "complete",
        }

    def ghost_sample(self) -> list[float]:
        return [round(self.x, 1), round(self.y, 1), round(self.theta, 3),
                0.0, round(math.hypot(self.vx, self.vy), 1)]
