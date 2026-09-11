"""Rolling solid ball on a motorized beam with bounded tilt and angular rate.

Reduced-order no-slip model: a=(5/7)g sin(theta)-drag*v. Rotational inertial
coupling of the beam and motor electrical dynamics are intentionally omitted.
"""
from dataclasses import dataclass, field
import math
import random

import numpy as np

DT, GRAVITY, MAX_TILT = 0.04, 9.81, 0.3


def scene():
    return {"kind": "generic", "bounds": [1000, 700], "primary_shape": "ballbeam",
            "statics": [{"shape": "line", "points": [[475, 455], [500, 420], [525, 455], [475, 455]]},
                        {"shape": "line", "points": [[120, 540], [880, 540]], "color": "danger"}]}


@dataclass
class BallBeamEnv:
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)
    obs_dim = 7
    n_continuous = 1
    n_binary = 0
    max_steps = 500
    dt = DT

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.position = self.rng.uniform(-0.65, 0.65) if self.jitter else -0.55
        self.target = self.rng.uniform(-0.3, 0.3) if self.jitter else 0.25
        self.velocity = self.theta = self.omega = 0.0
        self.steps = self.hold_steps = 0
        self.episode_reward = 0.0
        self.cause = "running"
        return self._obs()

    def _obs(self):
        return np.array([self.position, self.velocity/2, self.theta/MAX_TILT,
                         self.omega/1.2, self.target, self.hold_steps/50,
                         max(0, 1-self.steps/self.max_steps)], np.float32)

    def step(self, action):
        control = float(np.clip(action[0], -1, 1))
        self.omega = float(np.clip(self.omega+self.dt*(5*control-0.8*self.omega), -1.2, 1.2))
        self.theta += self.dt*self.omega
        if abs(self.theta) > MAX_TILT:
            self.theta = float(np.clip(self.theta, -MAX_TILT, MAX_TILT))
            self.omega = 0.0
        self.velocity += self.dt*((5/7)*GRAVITY*math.sin(self.theta)-0.08*self.velocity)
        self.position += self.dt*self.velocity
        self.steps += 1
        error = self.position-self.target
        settled = abs(error) < 0.045 and abs(self.velocity) < 0.06 and abs(self.theta) < 0.035
        self.hold_steps = self.hold_steps+1 if settled else 0
        success, fallen = self.hold_steps >= 50, abs(self.position) > 1
        deadline = self.steps >= self.max_steps and not success and not fallen
        done = success or fallen or deadline
        reward = -2*error**2-0.2*self.velocity**2-0.03*self.theta**2-0.001*control**2
        if success:
            self.cause, reward = "balanced", reward + 30
        elif fallen:
            self.cause, reward = "fell", reward - 200
        elif deadline:
            self.cause = "timeout"
        self.episode_reward += reward
        return self._obs(), float(reward), done, {"truncated": deadline, "task_deadline": deadline}

    def reference_action(self):
        desired_tilt = float(np.clip(-0.8*(self.position-self.target)-0.65*self.velocity, -0.25, 0.25))
        acceleration = 40*(desired_tilt-self.theta)-10*self.omega
        return np.array([np.clip((acceleration+0.8*self.omega)/5, -1, 1)], np.float32)

    def frame_payload(self):
        x, y = 500+self.position*330*math.cos(self.theta), 420+self.position*330*math.sin(self.theta)-13
        return {"objects": [
            {"shape": "ballbeam", "x": x, "y": y, "rot": self.theta},
            {"shape": "target", "x": 500+self.target*330*math.cos(self.theta),
             "y": 420+self.target*330*math.sin(self.theta)-13}],
            "distance": round(abs(self.position-self.target), 3), "hold": round(self.hold_steps*self.dt, 2)}

    def episode_summary(self):
        return {"reward": round(self.episode_reward, 3), "steps": self.steps, "cause": self.cause,
                "metric": round(abs(self.position-self.target), 4), "success": self.cause == "balanced"}

    def ghost_sample(self):
        primary = self.frame_payload()["objects"][0]
        return [primary["x"], primary["y"], self.theta, 0.0, abs(self.velocity)]
