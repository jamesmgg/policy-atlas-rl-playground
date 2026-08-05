"""Driving environment: car on track, dense progress reward, drift shaping.

Feature flags turn the same env into every driving scenario: surface grip
zones (wet/ice), a fuel budget (eco), scripted traffic bots (overtaking),
and drift-style scoring (drift trial). Observations include signed curvature
at five arc distances ahead so the agent can brake before corners.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .. import physics
from ..physics import CarState, PhysicsParams
from ..track import Track, heading_at

FRAME_SKIP = 2
DT_AGENT = physics.DT * FRAME_SKIP   # 0.04 s per agent step

MAX_AGENT_STEPS = 1500               # 60 s of sim time
STALL_WINDOW = 300                   # agent steps (12 s)
STALL_MIN_PROGRESS = 12.0            # arc units

LOOKAHEAD = (8.0, 20.0, 40.0, 75.0, 120.0)
GRIP_LOOKAHEAD = (20.0, 75.0, 120.0)
BASE_OBS_DIM = 8 + len(LOOKAHEAD) + 1 + len(GRIP_LOOKAHEAD)

CURV_SCALE = 80.0
CORNER_CURV = 0.012                  # |curvature| above this counts as a corner
CONTACT_DIST = 5.0


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 0.05           # per arc unit
    checkpoint: float = 3.0
    lap: float = 30.0
    time: float = -0.02              # per agent step
    drift_corner: float = 0.01       # per step, scaled by drift intensity
    collision: float = -40.0
    stall: float = -15.0
    wrong_way: float = -20.0
    style_coef: float = 0.0          # drift-trial style points per step
    overtake: float = 0.0
    contact: float = 0.0
    fuel_empty: float = 0.0


@dataclass(frozen=True)
class Zone:
    start_frac: float
    end_frac: float
    grip_scale: float
    color: str


@dataclass(frozen=True)
class FuelConfig:
    rate: float = 0.018              # tank fraction per second at full throttle


@dataclass(frozen=True)
class Bot:
    start_frac: float
    speed: float                     # m/s along centerline
    lat_frac: float                  # lateral offset as fraction of half width


@dataclass(frozen=True)
class DrivingFeatures:
    global_grip: float = 1.0
    zones: tuple[Zone, ...] = ()
    fuel: FuelConfig | None = None
    bots: tuple[Bot, ...] = ()
    metric: str = "lap"              # "lap" | "style" | "tank" | "overtakes"


@dataclass
class DrivingEnv:
    track: Track
    params: PhysicsParams = field(default_factory=lambda: physics.F1)
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    features: DrivingFeatures = field(default_factory=DrivingFeatures)
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)

    n_continuous = 2
    n_binary = 1
    max_steps = MAX_AGENT_STEPS
    dt = DT_AGENT

    def __post_init__(self):
        self.obs_dim = (BASE_OBS_DIM
                        + (1 if self.features.fuel else 0)
                        + (3 * len(self.features.bots)))
        # Per-sample grip from global surface + zones.
        grip = np.full(self.track.n, self.features.global_grip)
        for z in self.features.zones:
            i0 = self.track.index_at_arc(z.start_frac * self.track.total_length)
            i1 = self.track.index_at_arc(z.end_frac * self.track.total_length)
            if i0 <= i1:
                grip[i0:i1 + 1] *= z.grip_scale
            else:
                grip[i0:] *= z.grip_scale
                grip[:i1 + 1] *= z.grip_scale
        self._grip = grip
        self.car: CarState = None  # type: ignore[assignment]
        self.reset()

    def reset(self) -> np.ndarray:
        track = self.track
        idx = 0
        heading = heading_at(track, idx)
        x, y = track.centerline[idx]
        if self.jitter:
            heading += self.rng.uniform(-0.05, 0.05)
            nx, ny = track.normals[idx]
            off = self.rng.uniform(-3.0, 3.0)
            x, y = x + nx * off, y + ny * off
        self.car = CarState(x=float(x), y=float(y), heading=heading,
                            v_long=0.0, v_lat=0.0, omega=0.0, drift=0.0)
        self.idx = idx
        self.s_prev = float(track.arc[idx])
        self.progress = 0.0
        self.peak_progress = 0.0
        self.next_cp = 1
        self.laps = 0
        self.steps = 0
        self.lap_start_step = 0
        self.best_lap_time: float | None = None
        self.last_lap_time: float | None = None
        self.episode_reward = 0.0
        self.fuel = 1.0 if self.features.fuel else None
        self.style = 0.0
        self.overtakes = 0
        self.cause = "running"
        self._progress_log: deque[float] = deque(maxlen=STALL_WINDOW)
        self._bot_arcs = [b.start_frac * track.total_length for b in self.features.bots]
        self._bot_armed = [True] * len(self.features.bots)
        self._bot_prev_gap = [self._bot_gap(i) for i in range(len(self.features.bots))]
        return self._obs()

    # ------------------------------------------------------------------ step

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        cfg = self.reward_cfg
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        drift_on = action[2] > 0.5

        if self.fuel is not None and self.fuel <= 0.0:
            throttle = min(throttle, 0.0)  # tank dry: engine dead, brakes work

        grip_scale = float(self._grip[self.idx])
        for _ in range(FRAME_SKIP):
            self.car = physics.step(self.car, throttle, steer, drift_on,
                                    self.params, grip_scale)
        self.steps += 1

        track = self.track
        self.idx, lat = track.localize(self.car.x, self.car.y, self.idx)
        s_new = float(track.arc[self.idx])
        ds = s_new - self.s_prev
        if ds < -track.total_length / 2:
            ds += track.total_length
        elif ds > track.total_length / 2:
            ds -= track.total_length
        self.s_prev = s_new
        self.progress += ds
        self.peak_progress = max(self.peak_progress, self.progress)
        self._progress_log.append(self.progress)

        reward = cfg.progress * ds + cfg.time

        while self.progress >= self._cp_threshold():
            reward += cfg.checkpoint
            self.next_cp += 1
            if (self.next_cp - 1) % len(track.checkpoint_arcs) == 0:
                self.laps += 1
                lap_time = (self.steps - self.lap_start_step) * DT_AGENT
                self.lap_start_step = self.steps
                self.last_lap_time = lap_time
                if self.best_lap_time is None or lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                reward += cfg.lap

        curv = float(track.curvature[self.idx])
        in_corner = abs(curv) > CORNER_CURV
        if in_corner:
            reward += cfg.drift_corner * self.car.drift
            if cfg.style_coef > 0.0:
                pts = (cfg.style_coef * (self.car.speed / self.params.max_speed)
                       * self.car.drift * min(abs(curv) / CORNER_CURV, 3.0))
                self.style += pts
                reward += pts

        if self.fuel is not None:
            self.fuel = max(self.fuel - self.features.fuel.rate
                            * max(throttle, 0.0) ** 2 * DT_AGENT, 0.0)

        contact = False
        if self.features.bots:
            reward += self._step_bots()
            contact = self._check_contact()

        done = False
        if abs(lat) > track.half_widths[self.idx]:
            reward += cfg.collision
            done, self.cause = True, "collision"
        elif contact:
            reward += cfg.contact
            done, self.cause = True, "contact"
        elif self.fuel is not None and self.fuel <= 0.0 and self.car.speed < 1.0:
            reward += cfg.fuel_empty
            done, self.cause = True, "fuel"
        elif self.progress < self.peak_progress - 25.0:
            reward += cfg.wrong_way
            done, self.cause = True, "wrong_way"
        elif (len(self._progress_log) == STALL_WINDOW
              and self.progress - self._progress_log[0] < STALL_MIN_PROGRESS):
            reward += cfg.stall
            done, self.cause = True, "stall"
        elif self.steps >= MAX_AGENT_STEPS:
            done, self.cause = True, "timeout"

        self.episode_reward += reward
        return self._obs(), reward, done, {"truncated": done and self.cause == "timeout"}

    # ------------------------------------------------------------------ bots

    def _bot_pos(self, i: int) -> tuple[float, float, int]:
        track = self.track
        idx = track.index_at_arc(self._bot_arcs[i])
        bot = self.features.bots[i]
        off = bot.lat_frac * track.half_widths[idx]
        x = track.centerline[idx, 0] + track.normals[idx, 0] * off
        y = track.centerline[idx, 1] + track.normals[idx, 1] * off
        return float(x), float(y), idx

    def _bot_gap(self, i: int) -> float:
        """Arc distance from car to bot i, ahead-positive, in [0, L)."""
        return (self._bot_arcs[i] - self.s_prev) % self.track.total_length

    def _bot_signed_gap(self, i: int) -> float:
        """Shortest signed arc gap, preserving nearby rear hazards."""
        length = self.track.total_length
        return ((self._bot_arcs[i] - self.s_prev + length / 2.0) % length
                - length / 2.0)

    def _step_bots(self) -> float:
        L = self.track.total_length
        reward = 0.0
        for i, bot in enumerate(self.features.bots):
            self._bot_arcs[i] = (self._bot_arcs[i] + bot.speed * DT_AGENT) % L
            g = self._bot_gap(i)
            prev = self._bot_prev_gap[i]
            # Car passes bot: small positive gap wraps to almost-L.
            if self._bot_armed[i] and prev < 30.0 and g > L - 30.0:
                self.overtakes += 1
                reward += self.reward_cfg.overtake
                self._bot_armed[i] = False
            elif not self._bot_armed[i] and 60.0 < g < L * 0.5:
                self._bot_armed[i] = True  # car has fallen back / lapped around
            self._bot_prev_gap[i] = g
        return reward

    def _check_contact(self) -> bool:
        for i in range(len(self.features.bots)):
            bx, by, _ = self._bot_pos(i)
            if math.hypot(self.car.x - bx, self.car.y - by) < CONTACT_DIST:
                return True
        return False

    # ----------------------------------------------------------------- protocol

    def frame_payload(self) -> dict:
        car = self.car
        payload = {
            "car": {
                "x": round(car.x, 1), "y": round(car.y, 1),
                "heading": round(car.heading, 3),
                "speed": round(car.speed, 1),
                "drift": round(car.drift, 2),
                "slip": round(car.slip_angle, 3),
            },
            "laps": self.laps,
            "last_lap": round(self.last_lap_time, 2) if self.last_lap_time else None,
            "best_lap": round(self.best_lap_time, 2) if self.best_lap_time else None,
        }
        if self.fuel is not None:
            payload["fuel"] = round(self.fuel, 3)
        if self.features.bots:
            payload["bots"] = [
                {"x": round(x, 1), "y": round(y, 1),
                 "heading": round(heading_at(self.track, idx), 3)}
                for x, y, idx in (self._bot_pos(i) for i in range(len(self.features.bots)))
            ]
        if self.features.metric == "style":
            payload["style"] = round(self.style, 1)
        if self.features.metric == "overtakes":
            payload["overtakes"] = self.overtakes
        return payload

    def episode_summary(self) -> dict:
        metric: float | None
        kind = self.features.metric
        if kind == "style":
            metric = round(self.style, 1)
        elif kind == "tank":
            metric = round(self.progress / self.track.total_length, 2)
        elif kind == "overtakes":
            metric = float(self.overtakes)
        else:
            metric = round(self.best_lap_time, 2) if self.best_lap_time else None
        if kind == "style":
            success = self.style >= 10.0
        elif kind == "tank":
            success = self.progress >= self.track.total_length
        elif kind == "overtakes":
            success = self.overtakes >= len(self.features.bots)
        else:
            success = self.laps >= 1
        # Reaching a numeric target does not make an episode successful when it
        # subsequently ends in a safety failure. Timeouts and fuel exhaustion
        # remain valid endings for objectives that were already achieved.
        if self.cause in {"collision", "contact", "wrong_way", "stall"}:
            success = False
        return {
            "reward": round(self.episode_reward, 2),
            "steps": self.steps,
            "cause": self.cause,
            "metric": metric,
            "laps": self.laps,
            "best_lap": round(self.best_lap_time, 2) if self.best_lap_time else None,
            "success": success,
        }

    def ghost_sample(self) -> list[float]:
        car = self.car
        return [round(car.x, 1), round(car.y, 1), round(car.heading, 3),
                round(car.drift, 2), round(car.speed, 1)]

    # ------------------------------------------------------------------- obs

    def _cp_threshold(self) -> float:
        track = self.track
        n_cp = len(track.checkpoint_arcs)
        lap, k = divmod(self.next_cp, n_cp)
        return lap * track.total_length + track.checkpoint_arcs[k]

    def _obs(self) -> np.ndarray:
        track = self.track
        car = self.car
        p = self.params
        idx = self.idx
        tangent_heading = heading_at(track, idx)
        he = math.atan2(math.sin(car.heading - tangent_heading),
                        math.cos(car.heading - tangent_heading))
        _, lat = track.localize(car.x, car.y, idx)
        hw = float(track.half_widths[idx])

        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[0] = car.v_long / p.max_speed
        obs[1] = car.v_lat / 25.0
        obs[2] = car.omega / 3.0
        obs[3] = car.drift
        obs[4] = lat / hw
        obs[5] = math.sin(he)
        obs[6] = math.cos(he)
        obs[7] = hw / 16.0
        for i, dist in enumerate(LOOKAHEAD):
            ahead = track.index_ahead(idx, dist)
            obs[8 + i] = float(np.clip(track.curvature[ahead] * CURV_SCALE, -2.5, 2.5))
        cursor = 8 + len(LOOKAHEAD)
        obs[cursor] = float(self._grip[idx])
        cursor += 1
        for dist in GRIP_LOOKAHEAD:
            obs[cursor] = float(self._grip[track.index_ahead(idx, dist)])
            cursor += 1
        if self.fuel is not None:
            obs[cursor] = self.fuel
            cursor += 1
        if self.features.bots:
            for i, bot in enumerate(self.features.bots):
                gap = self._bot_signed_gap(i)
                obs[cursor] = float(np.clip(gap / 150.0, -1.0, 1.0))
                obs[cursor + 1] = (bot.speed - car.v_long) / p.max_speed
                obs[cursor + 2] = bot.lat_frac
                cursor += 3
        return obs
