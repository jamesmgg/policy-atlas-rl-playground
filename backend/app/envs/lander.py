"""2D lunar lander: main + side thrusters, fixed terrain with a flat pad,
limited fuel (engines die when dry — the ship falls, episode plays out)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
import math

import numpy as np

DT = 0.04
G = 50.0           # world units/s^2 (1000x700 world)
A_MAIN = 90.0
A_SIDE = 18.0
ALPHA_SIDE = 4.0   # rad/s^2
FUEL_RATE = 1.0 / 12.0   # full burn empties the tank in ~12 s

# Piecewise-linear heightfield (y-down: ground at larger y). Pad is flat.
TERRAIN: list[tuple[float, float]] = [
    (0, 520), (90, 480), (180, 540), (270, 500), (360, 560),
    (455, 620), (545, 620),     # landing pad
    (640, 560), (730, 580), (820, 500), (910, 540), (1000, 490),
]
PAD_X0, PAD_X1 = 455.0, 545.0
PAD_Y = 620.0
PAD_CX = (PAD_X0 + PAD_X1) / 2

SAFE_VX, SAFE_VY, SAFE_THETA = 8.0, 14.0, 0.25
FAILURE_REWARD = -100.0

# Half of training episodes retain the evaluation start distribution. A short
# terminal-rehearsal band makes the sparse successful touchdown discoverable;
# a wider approach band connects that skill back toward the full descent.
STANDARD_START_PROBABILITY = 0.5
TOUCHDOWN_START_PROBABILITY = 0.25
APPROACH_START_PROBABILITY = 0.25
TOUCHDOWN_ALTITUDE_MIN, TOUCHDOWN_ALTITUDE_MAX = 5.0, 18.0
TOUCHDOWN_X_OFFSET_MAX = 20.0
TOUCHDOWN_VX_MAX = 3.0
TOUCHDOWN_VY_MIN, TOUCHDOWN_VY_MAX = 0.0, 6.0
TOUCHDOWN_THETA_MAX = 0.08
TOUCHDOWN_OMEGA_MAX = 0.05
APPROACH_ALTITUDE_MIN, APPROACH_ALTITUDE_MAX = 30.0, 500.0
APPROACH_X_OFFSET_MAX = 35.0
APPROACH_VX_MAX = 6.0
APPROACH_VY_MIN, APPROACH_VY_MAX = 2.0, 18.0
APPROACH_THETA_MAX = 0.18
APPROACH_OMEGA_MAX = 0.25
TRAINING_START_DISTRIBUTION = (
    "50% standard high-altitude starts; 25% touchdown rehearsal 5-18 units "
    "above the pad; 25% braking approaches 30-500 units above the pad; all "
    "sampled states expose velocity, tilt, time, and fuel"
)


def wrap_angle(theta: float) -> float:
    """Represent physically equivalent attitudes on [-pi, pi)."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def terrain_y(x: float) -> float:
    pts = TERRAIN
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + (y1 - y0) * t
    return PAD_Y


def scene() -> dict:
    return {
        "kind": "generic",
        "bounds": [1000, 700],
        "primary_shape": "lander",
        "statics": [
            {"shape": "terrain", "points": [[x, y] for x, y in TERRAIN]},
            {"shape": "flag", "x": PAD_X0, "y": PAD_Y},
            {"shape": "flag", "x": PAD_X1, "y": PAD_Y},
        ],
    }


@dataclass
class LanderEnv:
    jitter: bool = True
    approach_curriculum: bool = False
    rng: random.Random = field(default_factory=random.Random)

    obs_dim = 9
    n_continuous = 2
    n_binary = 0
    max_steps = 600
    dt = DT

    def __post_init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        self.x, self.y = 500.0, 120.0
        self.vx = self.rng.uniform(-15.0, 15.0) if self.jitter else 0.0
        self.vy = 0.0
        self.theta = self.rng.uniform(-0.15, 0.15) if self.jitter else 0.0
        self.omega = 0.0
        self.fuel = 1.0
        self.steps = 0
        self.episode_reward = 0.0
        self.landed = False
        self.cause = "running"
        self._u_main = 0.0
        self._start_kind = "standard"
        if self.approach_curriculum:
            start_draw = self.rng.random()
            if start_draw >= STANDARD_START_PROBABILITY:
                if start_draw < (STANDARD_START_PROBABILITY
                                 + TOUCHDOWN_START_PROBABILITY):
                    self._reset_touchdown()
                else:
                    self._reset_approach()
        self._phi_prev = self._phi()
        return self._obs()

    def _reset_touchdown(self) -> None:
        self._reset_sampled_approach(
            kind="touchdown",
            altitude_min=TOUCHDOWN_ALTITUDE_MIN,
            altitude_max=TOUCHDOWN_ALTITUDE_MAX,
            x_offset_max=TOUCHDOWN_X_OFFSET_MAX,
            vx_max=TOUCHDOWN_VX_MAX,
            vy_min=TOUCHDOWN_VY_MIN,
            vy_max=TOUCHDOWN_VY_MAX,
            theta_max=TOUCHDOWN_THETA_MAX,
            omega_max=TOUCHDOWN_OMEGA_MAX,
        )

    def _reset_approach(self) -> None:
        """Sample a fully observed, dynamically plausible landing approach."""
        self._reset_sampled_approach(
            kind="approach",
            altitude_min=APPROACH_ALTITUDE_MIN,
            altitude_max=APPROACH_ALTITUDE_MAX,
            x_offset_max=APPROACH_X_OFFSET_MAX,
            vx_max=APPROACH_VX_MAX,
            vy_min=APPROACH_VY_MIN,
            vy_max=APPROACH_VY_MAX,
            theta_max=APPROACH_THETA_MAX,
            omega_max=APPROACH_OMEGA_MAX,
        )

    def _reset_sampled_approach(
        self, *, kind: str, altitude_min: float, altitude_max: float,
        x_offset_max: float, vx_max: float, vy_min: float, vy_max: float,
        theta_max: float, omega_max: float,
    ) -> None:
        altitude = self.rng.uniform(altitude_min, altitude_max)
        self.x = PAD_CX + self.rng.uniform(-x_offset_max, x_offset_max)
        self.y = PAD_Y - altitude
        self.vx = self.rng.uniform(-vx_max, vx_max)
        self.vy = self.rng.uniform(vy_min, vy_max)
        self.theta = self.rng.uniform(-theta_max, theta_max)
        self.omega = self.rng.uniform(-omega_max, omega_max)

        descent_fraction = (self.y - 120.0) / (PAD_Y - 120.0)
        self.steps = round(180.0 * descent_fraction)
        fuel_high = 1.0 - 0.1 * descent_fraction
        fuel_low = 1.0 - 0.5 * descent_fraction
        self.fuel = self.rng.uniform(fuel_low, fuel_high)
        self._start_kind = kind

    def _phi(self) -> float:
        dist = math.hypot(self.x - PAD_CX, self.y - PAD_Y)
        v = math.hypot(self.vx, self.vy)
        return 0.012 * dist + 0.04 * v + 0.4 * abs(self.theta)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        u_main = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        u_side = float(np.clip(action[1], -1.0, 1.0))
        if self.fuel <= 0.0:
            u_main = u_side = 0.0
        self._u_main = u_main

        self.omega += u_side * ALPHA_SIDE * DT
        self.theta = wrap_angle(self.theta + self.omega * DT)
        # Body-up in y-down coords is (sin t, -cos t).
        self.vx += (u_main * A_MAIN * math.sin(self.theta)
                    + u_side * A_SIDE * math.cos(self.theta)) * DT
        self.vy += (-u_main * A_MAIN * math.cos(self.theta) + G) * DT
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.fuel = max(self.fuel - (u_main + 0.15 * abs(u_side)) * FUEL_RATE * DT, 0.0)
        self.steps += 1

        phi = self._phi()
        reward = self._phi_prev - phi - 0.03 * u_main
        self._phi_prev = phi

        done = False
        if self.y >= terrain_y(self.x):
            done = True
            on_pad = PAD_X0 <= self.x <= PAD_X1
            soft = (abs(self.vx) < SAFE_VX and abs(self.vy) < SAFE_VY
                    and abs(self.theta) < SAFE_THETA)
            if on_pad and soft:
                reward += 100.0
                self.landed = True
                self.cause = "landed"
            else:
                reward += FAILURE_REWARD
                self.cause = "crash"
        elif self.x < 0 or self.x > 1000 or self.y < 0:
            reward += FAILURE_REWARD
            done, self.cause = True, "out_of_bounds"
        elif self.steps >= self.max_steps:
            reward += FAILURE_REWARD
            done, self.cause = True, "timeout"

        self.episode_reward += reward
        deadline = done and self.cause == "timeout"
        return self._obs(), reward, done, {
            "truncated": deadline,
            "task_deadline": deadline,
        }

    def _obs(self) -> np.ndarray:
        return np.array([
            (self.x - PAD_CX) / 300.0,
            (self.y - PAD_Y) / 300.0,
            self.vx / 60.0,
            self.vy / 60.0,
            math.sin(self.theta),
            math.cos(self.theta),
            self.omega / 3.0,
            self.fuel,
            max(0.0, 1.0 - self.steps / self.max_steps),
        ], dtype=np.float32)

    def frame_payload(self) -> dict:
        return {
            "objects": [{"shape": "lander", "x": round(self.x, 1),
                         "y": round(self.y, 1), "rot": round(self.theta, 3),
                         "flame": round(self._u_main, 2)}],
            "fuel": round(self.fuel, 3),
        }

    def episode_summary(self) -> dict:
        dist = math.hypot(self.x - PAD_CX, self.y - PAD_Y)
        return {
            "reward": round(self.episode_reward, 2),
            "steps": self.steps,
            "cause": self.cause,
            # A crash at pad center must never tie a safe touchdown. The fixed
            # failure offset preserves distance ordering among unsuccessful runs.
            "metric": 0.0 if self.landed else round(100.0 + dist, 1),
            "success": self.landed,
        }

    def ghost_sample(self) -> list[float]:
        return [round(self.x, 1), round(self.y, 1), round(self.theta, 3),
                0.0, round(math.hypot(self.vx, self.vy), 1)]
