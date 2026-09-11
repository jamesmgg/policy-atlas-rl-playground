"""Two planar links driven by ideal independent damped acceleration servos.

This is a kinematic/servo-control teaching model: no gravity, rigid-body
coupling, contacts, joint limits, or claim of equivalence to MuJoCo Reacher.
"""
from dataclasses import dataclass, field
import math
import random

import numpy as np

L1, L2 = 1.0, 0.75
DT, ACCELERATION, DAMPING = 0.05, 4.0, 1.2
TRACK_RADIUS, TRACK_OMEGA = 0.3, 0.45


def kinematics(q1, q2):
    elbow = np.array([L1*math.cos(q1), L1*math.sin(q1)])
    return elbow, elbow + np.array([L2*math.cos(q1+q2), L2*math.sin(q1+q2)])


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def scene():
    return {"kind": "generic", "bounds": [1000, 700], "primary_shape": "robotarm",
            "statics": [{"shape": "circle", "x": 450, "y": 390, "r": 280, "color": "#29414c"},
                        {"shape": "circle", "x": 450, "y": 390, "r": 40, "color": "#29414c"}]}


@dataclass
class RobotArmEnv:
    jitter: bool = True
    tracking: bool = False
    rng: random.Random = field(default_factory=random.Random)
    obs_dim = 14
    n_continuous = 2
    n_binary = 0
    max_steps = 300
    dt = DT

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.q = np.array([-0.7, 1.2], dtype=np.float64)
        if self.jitter:
            self.q += np.array([self.rng.uniform(-0.25, 0.25) for _ in range(2)])
        self.velocity = np.zeros(2)
        self.phase = self.rng.uniform(-math.pi, math.pi) if self.jitter else 0.0
        if self.tracking:
            self.target, self.target_velocity = self._tracking_target()
        else:
            q1 = self.rng.uniform(0.2, 1.5) if self.jitter else 0.9
            q2 = self.rng.uniform(0.5, 1.7) if self.jitter else 1.1
            self.target = kinematics(q1, q2)[1]
            self.target_velocity = np.zeros(2)
        self.steps = self.hold_steps = self.tracking_hits = 0
        self.recent_error = self.episode_reward = 0.0
        self.cause = "running"
        return self._obs()

    def _tracking_target(self):
        c, s = math.cos(self.phase), math.sin(self.phase)
        return (np.array([0.8+TRACK_RADIUS*c, 0.4+TRACK_RADIUS*s]),
                TRACK_RADIUS*TRACK_OMEGA*np.array([-s, c]))

    def _tip_velocity(self):
        q1, q2 = self.q
        return np.array([
            -L1*math.sin(q1)*self.velocity[0]-L2*math.sin(q1+q2)*sum(self.velocity),
            L1*math.cos(q1)*self.velocity[0]+L2*math.cos(q1+q2)*sum(self.velocity)])

    def _obs(self):
        error = kinematics(*self.q)[1] - self.target
        return np.array([math.cos(self.q[0]), math.sin(self.q[0]),
                         math.cos(self.q[1]), math.sin(self.q[1]),
                         *self.velocity/3, *self.target/1.75,
                         *self.target_velocity, *error/1.75,
                         (self.tracking_hits/100 if self.tracking else self.hold_steps/20),
                         max(0, 1-self.steps/self.max_steps)], np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.velocity = np.clip(self.velocity+self.dt*(ACCELERATION*action-DAMPING*self.velocity), -3, 3)
        self.q = wrap(self.q+self.dt*self.velocity)
        self.steps += 1
        if self.tracking:
            self.phase = float(wrap(self.phase+TRACK_OMEGA*self.dt))
            self.target, self.target_velocity = self._tracking_target()
        distance = float(np.linalg.norm(kinematics(*self.q)[1]-self.target))
        tip_speed = float(np.linalg.norm(self._tip_velocity()-self.target_velocity))
        self.hold_steps = self.hold_steps+1 if distance < 0.08 and tip_speed < 0.1 else 0
        if self.steps > self.max_steps-100:
            self.tracking_hits += int(distance < 0.12)
            self.recent_error += distance
        success = (self.steps >= self.max_steps and self.tracking_hits >= 90) if self.tracking else self.hold_steps >= 20
        deadline = self.steps >= self.max_steps and not success
        done = success or deadline
        reward = -distance**2 - 0.025*tip_speed**2 - 0.005*float(action @ action)
        if success:
            self.cause, reward = ("tracked" if self.tracking else "reached"), reward + 30
        elif deadline:
            self.cause = "timeout"
        self.episode_reward += reward
        return self._obs(), float(reward), done, {"truncated": deadline, "task_deadline": deadline}

    def reference_action(self):
        # Analytic inverse kinematics plus servo PD and target-velocity feedforward.
        x, y = self.target
        c2 = float(np.clip((x*x+y*y-L1*L1-L2*L2)/(2*L1*L2), -1, 1))
        q2 = math.acos(c2)
        q1 = math.atan2(y, x)-math.atan2(L2*math.sin(q2), L1+L2*c2)
        desired = np.array([q1, q2])
        a, b = desired
        jacobian = np.array([[-L1*math.sin(a)-L2*math.sin(a+b), -L2*math.sin(a+b)],
                             [L1*math.cos(a)+L2*math.cos(a+b), L2*math.cos(a+b)]])
        desired_velocity = np.linalg.solve(jacobian, self.target_velocity)
        acceleration = 12*wrap(desired-self.q)+6*(desired_velocity-self.velocity)
        return np.clip((acceleration+DAMPING*self.velocity)/ACCELERATION, -1, 1).astype(np.float32)

    def frame_payload(self):
        tip = kinematics(*self.q)[1]
        return {"objects": [
            {"shape": "robotarm", "x": 450+tip[0]*160, "y": 390-tip[1]*160,
             "rot": float(self.q[0]), "joint2": float(self.q[1])},
            {"shape": "target", "x": 450+self.target[0]*160, "y": 390-self.target[1]*160}],
            "distance": round(float(np.linalg.norm(tip-self.target)), 3),
            "hold": round(self.hold_steps*self.dt, 2), "tracking_hits": self.tracking_hits}

    def episode_summary(self):
        distance = float(np.linalg.norm(kinematics(*self.q)[1]-self.target))
        metric = self.recent_error/max(1, self.steps-(self.max_steps-100)) if self.tracking else distance
        return {"reward": round(self.episode_reward, 3), "steps": self.steps, "cause": self.cause,
                "metric": round(metric, 4), "success": self.cause in ("tracked", "reached")}

    def ghost_sample(self):
        tip = kinematics(*self.q)[1]
        return [450+tip[0]*160, 390-tip[1]*160, float(self.q[0]), float(self.q[1]),
                float(np.linalg.norm(self.velocity))]
