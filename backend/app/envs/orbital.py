"""Planar Hill/Clohessy-Wiltshire rendezvous around a circular reference orbit.

x is radial, y along-track; metres and seconds. This linear local model omits
attitude, plume/contact dynamics, eccentricity, and perturbations. It is an
educational proximity-control task, not a flight-qualified docking simulator.
"""
from dataclasses import dataclass, field
import math
import random

import numpy as np

MEAN_MOTION = 0.0011
MAX_ACCELERATION = 0.04
DT = 0.5


def integrate_cw(state: np.ndarray, acceleration: np.ndarray, dt: float) -> np.ndarray:
    """Fourth-order Runge-Kutta with constant thrust over the control interval."""
    n = MEAN_MOTION

    def derivative(s):
        x, _, vx, vy = s
        return np.array([vx, vy, 3*n*n*x + 2*n*vy + acceleration[0],
                         -2*n*vx + acceleration[1]], dtype=np.float64)

    k1 = derivative(state)
    k2 = derivative(state + dt*k1/2)
    k3 = derivative(state + dt*k2/2)
    k4 = derivative(state + dt*k3)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def scene():
    return {"kind": "generic", "bounds": [1000, 700], "primary_shape": "spacecraft",
            "statics": [
                {"shape": "circle", "x": 500, "y": 350, "r": 95},
                {"shape": "circle", "x": 500, "y": 350, "r": 190},
                {"shape": "line", "points": [[110, 350], [890, 350]]},
                {"shape": "line", "points": [[500, 65], [500, 635]]},
            ]}


@dataclass
class OrbitalEnv:
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)
    obs_dim = 6
    n_continuous = 2
    n_binary = 0
    max_steps = 600
    dt = DT

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.state = np.array([
            self.rng.uniform(30, 65) if self.jitter else 50,
            self.rng.uniform(-35, 35) if self.jitter else -25,
            self.rng.uniform(-0.1, 0.1) if self.jitter else 0,
            self.rng.uniform(-0.1, 0.1) if self.jitter else 0,
        ], dtype=np.float64)
        self.steps = self.hold_steps = 0
        self.episode_reward = self.delta_v = 0.0
        self.cause = "running"
        self.action = np.zeros(2)
        return self._obs()

    def _obs(self):
        return np.array([self.state[0]/100, self.state[1]/100,
                         self.state[2]/2, self.state[3]/2,
                         self.hold_steps/20, max(0, 1-self.steps/self.max_steps)], np.float32)

    def step(self, action):
        self.action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        acceleration = self.action * MAX_ACCELERATION
        self.state = integrate_cw(self.state, acceleration, self.dt)
        self.steps += 1
        distance, speed = np.linalg.norm(self.state[:2]), np.linalg.norm(self.state[2:])
        self.delta_v += float(np.linalg.norm(acceleration))*self.dt
        self.hold_steps = self.hold_steps+1 if distance < 2 and speed < 0.08 else 0
        success = self.hold_steps >= 20
        escaped = distance > 140
        deadline = self.steps >= self.max_steps and not success and not escaped
        done = success or escaped or deadline
        reward = -0.05 - (distance/60)**2 - 0.2*speed**2 - 0.005*float(self.action @ self.action)
        if success:
            self.cause, reward = "docked", reward + 100
        elif escaped:
            self.cause, reward = "escaped", reward - 600
        elif deadline:
            self.cause = "timeout"
        self.episode_reward += float(reward)
        return self._obs(), float(reward), done, {"truncated": deadline, "task_deadline": deadline}

    def reference_action(self):
        x, _, vx, vy = self.state
        correction = np.array([3*MEAN_MOTION**2*x+2*MEAN_MOTION*vy, -2*MEAN_MOTION*vx])
        acceleration = -0.003*self.state[:2] - 0.11*self.state[2:] - correction
        return np.clip(acceleration/MAX_ACCELERATION, -1, 1).astype(np.float32)

    def frame_payload(self):
        x, y, vx, vy = self.state
        return {"objects": [
            {"shape": "spacecraft", "x": 500+y*3, "y": 350-x*3,
             "rot": math.atan2(-vx, vy), "thrust_x": float(self.action[1]),
             "thrust_y": -float(self.action[0])},
            {"shape": "station", "x": 500, "y": 350},
        ], "distance": round(float(np.linalg.norm(self.state[:2])), 2),
            "speed": round(float(np.linalg.norm(self.state[2:])), 3),
            "hold": round(self.hold_steps*self.dt, 1), "delta_v": round(self.delta_v, 2)}

    def episode_summary(self):
        distance = float(np.linalg.norm(self.state[:2]))
        return {"reward": round(self.episode_reward, 3), "steps": self.steps, "cause": self.cause,
                "metric": round(distance, 3), "success": self.cause == "docked"}

    def ghost_sample(self):
        x, y, vx, vy = self.state
        return [500+y*3, 350-x*3, math.atan2(-vx, vy), 0.0, float(math.hypot(vx, vy))]
