"""Three finite, fully observed arcade tasks with absorbing terminal states.

Reference controllers are diagnostics/demonstration sources only. Environment
steps consume the supplied action; they never substitute a reference controller.
"""
from dataclasses import dataclass, field
import math
import random

import numpy as np


class ArcadeEpisode:
    n_binary = 0

    def _begin(self):
        self.steps = 0
        self.episode_reward = 0.0
        self.cause = "running"

    def _finish(self, reward, success=False, failure=None):
        self.steps += 1
        deadline = self.steps >= self.max_steps and not success and failure is None
        if failure is not None:
            self.cause, reward = failure, reward - 30.0
        elif success:
            self.cause, reward = "completed", reward + 30.0
        elif deadline:
            self.cause, reward = "timeout", reward - 15.0
        self.episode_reward += reward
        return self._obs(), float(reward), self.cause != "running", {
            "truncated": deadline, "task_deadline": deadline}

    def _absorbing(self):
        deadline = self.cause == "timeout"
        return self._obs(), 0.0, True, {"truncated": deadline, "task_deadline": deadline}

    def episode_summary(self):
        return {"reward": round(self.episode_reward, 3), "steps": self.steps,
                "cause": self.cause, "metric": self.progress,
                "success": self.cause == "completed"}

    def ghost_sample(self):
        primary = self.frame_payload()["objects"][0]
        return [primary["x"], primary["y"], primary.get("rot", 0.0), 0.0, 0.0]


def paddle_scene():
    return {"kind": "generic", "bounds": [1000, 700], "primary_shape": "paddle",
            "statics": [{"shape": "line", "points": [[75, 620], [75, 70], [925, 70], [925, 620]]}]}


@dataclass
class PaddleRallyEnv(ArcadeEpisode):
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)
    obs_dim = 9
    n_continuous = 1
    max_steps = 450
    dt = .04
    target_hits = 5
    paddle_half_width = .18
    ball_radius = .035

    def __post_init__(self):
        self.reset()

    def reset(self):
        self._begin()
        self.progress = 0
        self.paddle = self.rng.uniform(-.6, .6) if self.jitter else -.5
        self.ball_x = self.rng.uniform(-.6, .6) if self.jitter else .45
        self.ball_y = 1.25
        self.ball_vx = self.rng.uniform(-1.1, 1.1) if self.jitter else .8
        self.ball_vy = -1.8
        return self._obs()

    def intercept(self):
        # Exact wall-reflected intercept is an explicitly engineered sensor.
        height, floor, edge = 1.465, .12, 1-self.ball_radius
        travel = ((self.ball_y-floor)/abs(self.ball_vy) if self.ball_vy < 0 else
                  (2*height-self.ball_y-floor)/self.ball_vy)
        unfolded = self.ball_x + max(0, travel)*self.ball_vx
        folded = (unfolded+edge) % (4*edge)
        return (folded if folded <= 2*edge else 4*edge-folded)-edge

    def _obs(self):
        intercept = self.intercept()
        return np.array([self.paddle, self.ball_x, self.ball_y/1.5,
                         self.ball_vx/1.2, self.ball_vy/1.8, intercept,
                         intercept-self.paddle, self.progress/self.target_hits,
                         max(0, 1-self.steps/self.max_steps)], np.float32)

    def step(self, action):
        if self.cause != "running":
            return self._absorbing()
        control = float(np.clip(action[0], -1, 1))
        self.paddle = float(np.clip(self.paddle+2.4*self.dt*control, -.82, .82))
        old_y = self.ball_y
        self.ball_x += self.dt*self.ball_vx
        self.ball_y += self.dt*self.ball_vy
        edge = 1-self.ball_radius
        if abs(self.ball_x) > edge:
            self.ball_x = math.copysign(2*edge-abs(self.ball_x), self.ball_x)
            self.ball_vx *= -1
        if self.ball_y > 1.465:
            self.ball_y = 2*1.465-self.ball_y
            self.ball_vy = -abs(self.ball_vy)
        reward, failure = -.015-.03*(self.intercept()-self.paddle)**2, None
        if self.ball_vy < 0 and self.ball_y <= .12:
            # Evaluate the actual swept crossing, not a post-step overshoot.
            crossing_x = self.ball_x - self.ball_vx*(.12-self.ball_y)/abs(self.ball_vy)
            if abs(crossing_x-self.paddle) <= self.paddle_half_width+self.ball_radius:
                self.progress += 1
                reward += 5.0
                self.ball_y = .12+max(0, .12-self.ball_y)
                self.ball_vy = 1.8
                self.ball_vx = self.rng.uniform(-1.15, 1.15) if self.jitter else (-.9 if self.progress % 2 else .8)
            else:
                failure = "missed-ball"
        return self._finish(reward, self.progress >= self.target_hits, failure)

    def reference_action(self):
        return np.array([np.clip(5*(self.intercept()-self.paddle), -1, 1)], np.float32)

    def frame_payload(self):
        return {"objects": [{"shape": "paddle", "x": 500+425*self.paddle, "y": 606,
                             "rot": 0., "width": 153, "height": 16},
                            {"shape": "ball", "x": 500+425*self.ball_x,
                             "y": 620-366.67*self.ball_y, "radius": 13}],
                "hits": self.progress, "target_hits": self.target_hits}


def flappy_scene():
    return {"kind": "generic", "bounds": [1000, 700], "primary_shape": "bird",
            "statics": [{"shape": "line", "points": [[50, 65], [950, 65]], "color": "danger"},
                        {"shape": "line", "points": [[50, 635], [950, 635]], "color": "danger"}]}


@dataclass
class FlappyFlightEnv(ArcadeEpisode):
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)
    obs_dim = 8
    n_continuous = 1
    max_steps = 380
    dt = .04
    target_gates = 6
    gap_half_height = .30
    bird_radius = .045
    pipe_half_width = .09

    def __post_init__(self):
        self.reset()

    def _gap(self):
        return self.rng.uniform(-.46, .46) if self.jitter else (.43 if self.progress % 2 == 0 else -.43)

    def reset(self):
        self._begin()
        self.progress = 0
        self.y = self.rng.uniform(-.3, .3) if self.jitter else -.3
        self.velocity = 0.
        self.pipe_x = 1.25
        self.gap = self._gap()
        self.next_gap = self.rng.uniform(-.46, .46) if self.jitter else -.43
        return self._obs()

    def _obs(self):
        return np.array([self.y, self.velocity/2, self.pipe_x/1.4, self.gap,
                         self.gap-self.y, self.next_gap,
                         self.progress/self.target_gates,
                         max(0, 1-self.steps/self.max_steps)], np.float32)

    def step(self, action):
        if self.cause != "running":
            return self._absorbing()
        control = float(np.clip(action[0], -1, 1))
        self.velocity = float(np.clip(self.velocity+self.dt*(4*control-.7*self.velocity), -2., 2.))
        self.y += self.dt*self.velocity
        self.pipe_x -= .8*self.dt
        reward = -.01-.08*(self.gap-self.y)**2-.002*control**2
        failure = None
        if abs(self.y)+self.bird_radius >= 1:
            failure = "out-of-bounds"
        elif abs(self.pipe_x) <= self.pipe_half_width+self.bird_radius and abs(self.y-self.gap)+self.bird_radius >= self.gap_half_height:
            failure = "pipe-hit"
        if self.pipe_x < -(self.pipe_half_width+self.bird_radius) and failure is None:
            self.progress += 1
            reward += 5.
            if self.progress < self.target_gates:
                self.pipe_x += 1.45
                self.gap = self.next_gap
                self.next_gap = self._gap()
        return self._finish(reward, self.progress >= self.target_gates, failure)

    def reference_action(self):
        return np.array([np.clip((9*(self.gap-self.y)-4*self.velocity)/4, -1, 1)], np.float32)

    def frame_payload(self):
        objects = [{"shape": "bird", "x": 270., "y": 350-285*self.y,
                    "rot": -.22*self.velocity}]
        for distance, gap in ((self.pipe_x, self.gap), (self.pipe_x+1.45, self.next_gap)):
            x = 270+350*distance
            if -80 <= x <= 1080:
                gap_top, gap_bottom = 350-285*(gap+self.gap_half_height), 350-285*(gap-self.gap_half_height)
                objects.extend([
                    {"shape": "pipe", "x": x, "y": (65+gap_top)/2, "width": 63, "height": gap_top-65},
                    {"shape": "pipe", "x": x, "y": (635+gap_bottom)/2, "width": 63, "height": 635-gap_bottom}])
        return {"objects": objects, "gates": self.progress, "target_gates": self.target_gates}


def coin_scene():
    return {"kind": "generic", "bounds": [1000, 700], "primary_shape": "collector",
            "statics": [{"shape": "line", "points": [[200, 50], [800, 50], [800, 650], [200, 650], [200, 50]], "color": "danger"}]}


@dataclass
class CoinCollectorEnv(ArcadeEpisode):
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)
    obs_dim = 11
    n_continuous = 2
    max_steps = 600
    dt = .04
    target_coins = 5
    radius = .045

    def __post_init__(self):
        self.reset()

    def _new_target(self):
        # Each new destination is separated from the last by at least 0.55 m.
        for _ in range(100):
            angle = self.rng.uniform(-math.pi, math.pi) if self.jitter else self.progress*2.4+.6
            target = .68*np.array([math.cos(angle), math.sin(angle)])
            if np.linalg.norm(target-self.position) >= .55:
                return target
        return -.68*self.position/max(.01, np.linalg.norm(self.position))

    def reset(self):
        self._begin()
        self.progress = self.hold = 0
        self.position = np.array([self.rng.uniform(-.2, .2), self.rng.uniform(-.2, .2)]) if self.jitter else np.zeros(2)
        self.velocity = np.zeros(2)
        self.target = self._new_target()
        return self._obs()

    def _obs(self):
        return np.array([*self.position, *self.velocity, *self.target,
                         *(self.target-self.position), self.hold/3,
                         self.progress/self.target_coins,
                         max(0, 1-self.steps/self.max_steps)], np.float32)

    def step(self, action):
        if self.cause != "running":
            return self._absorbing()
        control = np.clip(action, -1, 1)
        old_distance = np.linalg.norm(self.position-self.target)
        self.velocity = np.clip(self.velocity+self.dt*(3*control-1.2*self.velocity), -1.5, 1.5)
        self.position += self.dt*self.velocity
        distance = np.linalg.norm(self.position-self.target)
        speed = np.linalg.norm(self.velocity)
        # Dwell prevents a fast fly-through from being described as collection.
        self.hold = self.hold+1 if distance < .10 and speed < .20 else 0
        failure = "out-of-bounds" if np.max(np.abs(self.position))+self.radius >= 1 else None
        reward = 2*(old_distance-distance)-.02-.03*distance**2-.003*float(control@control)
        if self.hold >= 3 and failure is None:
            self.progress += 1
            reward += 5.
            if self.progress < self.target_coins:
                self.target = self._new_target()
                self.hold = 0
        return self._finish(reward, self.progress >= self.target_coins, failure)

    def reference_action(self):
        return np.clip((8*(self.target-self.position)-4*self.velocity)/3, -1, 1).astype(np.float32)

    def frame_payload(self):
        return {"objects": [{"shape": "collector", "x": 500+300*self.position[0],
                             "y": 350-300*self.position[1], "rot": math.atan2(-self.velocity[1], self.velocity[0]) if np.linalg.norm(self.velocity) > .03 else 0.},
                            {"shape": "coin", "x": 500+300*self.target[0],
                             "y": 350-300*self.target[1], "radius": 16}],
                "coins": self.progress, "target_coins": self.target_coins,
                "speed": round(float(np.linalg.norm(self.velocity)), 3), "capture": self.hold/3}
