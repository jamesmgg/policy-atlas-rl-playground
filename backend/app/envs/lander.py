"""2D lunar lander: main + side thrusters, fixed terrain with a flat pad,
limited fuel (engines die when dry — the ship falls, episode plays out)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
import math

import numpy as np

from .base import TrainingCurriculumSpec

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

# The sparse terminal outcome is learned from the pad upward. A fixed-suite
# performance gate, rather than episode count, controls when a harder altitude
# becomes available. Previously mastered lower-altitude starts remain in the
# mixture so advancing the frontier does not abruptly remove positive examples.
CURRICULUM_FRONTIER_ORDER = (4, 3, 2, 1, 0)
CURRICULUM_ACTIVE_FRONTIER_PROBABILITY = 0.5
CURRICULUM_SUCCESS_RATE_THRESHOLD = 0.75
CURRICULUM_SUCCESS_RATE_THRESHOLDS = (
    (4, 0.9),
    (3, 0.75),
    (2, 0.75),
    (1, 0.75),
    (0, 0.75),
)
CURRICULUM_CONSECUTIVE_CONFIRMATIONS = 1
TOUCHDOWN_ALTITUDE_MIN, TOUCHDOWN_ALTITUDE_MAX = 5.0, 18.0
TOUCHDOWN_X_OFFSET_MAX = 20.0
TOUCHDOWN_VX_MAX = 3.0
TOUCHDOWN_VY_MIN, TOUCHDOWN_VY_MAX = 0.0, 6.0
TOUCHDOWN_THETA_MAX = 0.08
TOUCHDOWN_OMEGA_MAX = 0.05
APPROACH_ALTITUDE_MIN, APPROACH_ALTITUDE_MAX = 30.0, 500.0
APPROACH_FRONTIER_ALTITUDES = {
    3: (30.0, 100.0),
    2: (50.0, 200.0),
    1: (100.0, 500.0),
}
APPROACH_X_OFFSET_MAX = 35.0
APPROACH_VX_MAX = 6.0
APPROACH_VY_MIN, APPROACH_VY_MAX = 2.0, 18.0
APPROACH_THETA_MAX = 0.18
APPROACH_OMEGA_MAX = 0.25
TRAINING_START_DISTRIBUTION = (
    "Performance-gated reverse altitude curriculum: begin with 100% "
    "touchdown rehearsals at k4 (5-18 units above the pad); unlock low "
    "k3 (30-100), overlapping k2 (50-200), high k1 (100-500), "
    "then canonical "
    "k0 descents after one >=90% fixed 20-start k4 evaluation and "
    ">=75% at each harder frontier; "
    "thereafter the active frontier receives 50% of resets and mastered "
    "easier frontiers uniformly share the remainder"
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
    forced_start_frontier: int | None = None
    rng: random.Random = field(default_factory=random.Random)
    _curriculum_position: int = field(default=0, init=False, repr=False)
    _curriculum_pass_streak: int = field(default=0, init=False, repr=False)
    _curriculum_complete: bool = field(default=False, init=False, repr=False)
    _curriculum_evaluations: int = field(default=0, init=False, repr=False)
    _curriculum_last_success_rate: float | None = field(
        default=None, init=False, repr=False)
    _curriculum_last_evaluation_episode: int | None = field(
        default=None, init=False, repr=False)

    obs_dim = 9
    n_continuous = 2
    n_binary = 0
    max_steps = 600
    dt = DT

    def __post_init__(self):
        if (self.forced_start_frontier is not None
                and self.forced_start_frontier not in CURRICULUM_FRONTIER_ORDER):
            raise ValueError("forced Lander frontier must be in [0, 4]")
        self.reset()

    def _sample_start_frontier(self) -> int:
        if self.forced_start_frontier is not None:
            return self.forced_start_frontier
        if not self.approach_curriculum:
            return 0
        frontier = CURRICULUM_FRONTIER_ORDER[self._curriculum_position]
        mastered = CURRICULUM_FRONTIER_ORDER[:self._curriculum_position]
        if (not mastered
                or self.rng.random() < CURRICULUM_ACTIVE_FRONTIER_PROBABILITY):
            return frontier
        return self.rng.choice(mastered)

    def training_curriculum_state(self) -> dict:
        """Return the complete JSON-safe state needed for exact continuation."""
        frontier = CURRICULUM_FRONTIER_ORDER[self._curriculum_position]
        return {
            "version": 1,
            "frontier_position": self._curriculum_position,
            "frontier": frontier,
            "mastered": list(
                CURRICULUM_FRONTIER_ORDER[:self._curriculum_position]),
            "pass_streak": self._curriculum_pass_streak,
            "complete": self._curriculum_complete,
            "evaluations": self._curriculum_evaluations,
            "last_success_rate": self._curriculum_last_success_rate,
            "last_evaluation_episode": self._curriculum_last_evaluation_episode,
        }

    def restore_training_curriculum_state(self, state: dict) -> None:
        """Restore the exact, validated gate state from a checkpoint."""
        expected_keys = {
            "version", "frontier_position", "frontier", "mastered",
            "pass_streak", "complete", "evaluations", "last_success_rate",
            "last_evaluation_episode",
        }
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError("invalid Lander curriculum state fields")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("unsupported Lander curriculum state version")
        for key in ("frontier_position", "frontier", "pass_streak", "evaluations"):
            if type(state[key]) is not int:
                raise TypeError(f"Lander curriculum {key} must be an integer")
        if type(state["complete"]) is not bool:
            raise TypeError("Lander curriculum complete must be a boolean")

        position = state["frontier_position"]
        pass_streak = state["pass_streak"]
        evaluations = state["evaluations"]
        complete = state["complete"]
        last_success_rate = state["last_success_rate"]
        last_evaluation_episode = state["last_evaluation_episode"]
        if not 0 <= position < len(CURRICULUM_FRONTIER_ORDER):
            raise ValueError("invalid Lander curriculum frontier position")
        if state["frontier"] != CURRICULUM_FRONTIER_ORDER[position]:
            raise ValueError("Lander curriculum frontier does not match position")
        if state["mastered"] != list(CURRICULUM_FRONTIER_ORDER[:position]):
            raise ValueError("Lander curriculum mastered frontiers are inconsistent")
        if not 0 <= pass_streak <= CURRICULUM_CONSECUTIVE_CONFIRMATIONS:
            raise ValueError("invalid Lander curriculum confirmation streak")
        if evaluations < 0 or evaluations < position:
            raise ValueError("invalid Lander curriculum evaluation count")
        if complete and position != len(CURRICULUM_FRONTIER_ORDER) - 1:
            raise ValueError("only the canonical Lander frontier can be complete")
        if complete and pass_streak != CURRICULUM_CONSECUTIVE_CONFIRMATIONS:
            raise ValueError("complete Lander curriculum lacks its final pass")
        if not complete and pass_streak != 0:
            raise ValueError("non-complete Lander curriculum has a stale pass")
        if last_success_rate is not None:
            if (isinstance(last_success_rate, bool)
                    or not isinstance(last_success_rate, (int, float))):
                raise TypeError("Lander curriculum success rate must be numeric")
            last_success_rate = float(last_success_rate)
            if not math.isfinite(last_success_rate) or not 0.0 <= last_success_rate <= 1.0:
                raise ValueError("invalid Lander curriculum success rate")
        if last_evaluation_episode is not None:
            if type(last_evaluation_episode) is not int:
                raise TypeError("Lander curriculum evaluation episode must be an integer")
            if last_evaluation_episode < 0:
                raise ValueError("invalid Lander curriculum evaluation episode")
        history_empty = (last_success_rate is None
                         and last_evaluation_episode is None)
        history_complete = (last_success_rate is not None
                            and last_evaluation_episode is not None)
        if ((evaluations == 0 and not history_empty)
                or (evaluations > 0 and not history_complete)):
            raise ValueError("Lander curriculum evaluation history is inconsistent")

        self._curriculum_position = position
        self._curriculum_pass_streak = pass_streak
        self._curriculum_complete = complete
        self._curriculum_evaluations = evaluations
        self._curriculum_last_success_rate = last_success_rate
        self._curriculum_last_evaluation_episode = last_evaluation_episode

    def reset_training_curriculum(self) -> None:
        self._curriculum_position = 0
        self._curriculum_pass_streak = 0
        self._curriculum_complete = False
        self._curriculum_evaluations = 0
        self._curriculum_last_success_rate = None
        self._curriculum_last_evaluation_episode = None

    def record_training_curriculum_evaluation(
            self, success_rate: float,
            curriculum: TrainingCurriculumSpec, *,
            evaluation_episode: int | None = None) -> dict:
        """Apply one distinct fixed-frontier result to the advancement gate."""
        if tuple(curriculum.frontier_order) != CURRICULUM_FRONTIER_ORDER:
            raise ValueError("curriculum frontier order does not match Lander")
        if (curriculum.active_frontier_probability
                != CURRICULUM_ACTIVE_FRONTIER_PROBABILITY
                or curriculum.success_rate_threshold
                != CURRICULUM_SUCCESS_RATE_THRESHOLD
                or curriculum.frontier_success_rate_thresholds
                != CURRICULUM_SUCCESS_RATE_THRESHOLDS
                or curriculum.consecutive_confirmations
                != CURRICULUM_CONSECUTIVE_CONFIRMATIONS):
            raise ValueError("curriculum gate does not match Lander")
        if (isinstance(success_rate, bool)
                or not isinstance(success_rate, (int, float))
                or not math.isfinite(float(success_rate))
                or not 0.0 <= float(success_rate) <= 1.0):
            raise ValueError("curriculum success rate must be in [0, 1]")
        success_rate = float(success_rate)
        if evaluation_episode is None:
            evaluation_episode = (
                0 if self._curriculum_last_evaluation_episode is None
                else self._curriculum_last_evaluation_episode + 1)
        if type(evaluation_episode) is not int or evaluation_episode < 0:
            raise ValueError("curriculum evaluation episode must be non-negative")

        before = self.training_curriculum_state()
        success_rate_threshold = curriculum.success_rate_threshold_for(
            before["frontier"])
        passed = success_rate >= success_rate_threshold
        if self._curriculum_last_evaluation_episode == evaluation_episode:
            return {
                "success_rate": success_rate,
                "success_rate_threshold": success_rate_threshold,
                "passed": passed,
                "evaluation_episode": evaluation_episode,
                "ignored_duplicate": True,
                "frontier_before": before["frontier"],
                "frontier_after": before["frontier"],
                "mastered_before": before["mastered"],
                "mastered_after": before["mastered"],
                "confirmation_streak_before": before["pass_streak"],
                "confirmation_streak_after": before["pass_streak"],
                "complete_before": before["complete"],
                "complete_after": before["complete"],
                "unlocked": False,
            }
        if (self._curriculum_last_evaluation_episode is not None
                and evaluation_episode < self._curriculum_last_evaluation_episode):
            raise ValueError("curriculum evaluation episodes must be monotonic")

        self._curriculum_evaluations += 1
        self._curriculum_last_success_rate = success_rate
        self._curriculum_last_evaluation_episode = evaluation_episode
        unlocked = False
        if not self._curriculum_complete:
            self._curriculum_pass_streak = (
                self._curriculum_pass_streak + 1 if passed else 0)
            if self._curriculum_pass_streak >= curriculum.consecutive_confirmations:
                if self._curriculum_position < len(CURRICULUM_FRONTIER_ORDER) - 1:
                    self._curriculum_position += 1
                    self._curriculum_pass_streak = 0
                    unlocked = True
                else:
                    self._curriculum_complete = True
        after = self.training_curriculum_state()
        return {
            "success_rate": success_rate,
            "success_rate_threshold": success_rate_threshold,
            "passed": passed,
            "evaluation_episode": evaluation_episode,
            "ignored_duplicate": False,
            "frontier_before": before["frontier"],
            "frontier_after": after["frontier"],
            "mastered_before": before["mastered"],
            "mastered_after": after["mastered"],
            "confirmation_streak_before": before["pass_streak"],
            "confirmation_streak_after": after["pass_streak"],
            "complete_before": before["complete"],
            "complete_after": after["complete"],
            "unlocked": unlocked,
        }

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
        self._start_frontier = self._sample_start_frontier()
        if self._start_frontier == 4:
            self._reset_touchdown()
        elif self._start_frontier in APPROACH_FRONTIER_ALTITUDES:
            self._reset_approach(self._start_frontier)
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

    def _reset_approach(self, frontier: int | None = None) -> None:
        """Sample a fully observed, dynamically plausible landing approach."""
        altitude_min, altitude_max = (
            APPROACH_FRONTIER_ALTITUDES[frontier]
            if frontier is not None
            else (APPROACH_ALTITUDE_MIN, APPROACH_ALTITUDE_MAX)
        )
        self._reset_sampled_approach(
            kind="approach",
            altitude_min=altitude_min,
            altitude_max=altitude_max,
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


def make_frontier_evaluation_env(frontier: int) -> LanderEnv:
    """Build one fixed-suite env without stochastic frontier selection."""
    if frontier not in CURRICULUM_FRONTIER_ORDER:
        raise ValueError("Lander frontier must be in [0, 4]")
    return LanderEnv(jitter=True, forced_start_frontier=frontier)


TRAINING_CURRICULUM = TrainingCurriculumSpec(
    id="lander-reverse-altitude-v1",
    frontier_order=CURRICULUM_FRONTIER_ORDER,
    active_frontier_probability=CURRICULUM_ACTIVE_FRONTIER_PROBABILITY,
    success_rate_threshold=CURRICULUM_SUCCESS_RATE_THRESHOLD,
    consecutive_confirmations=CURRICULUM_CONSECUTIVE_CONFIRMATIONS,
    evaluation_suite_version="lander-altitude-eval-v2",
    evaluation_episodes=20,
    evaluation_seed_base=300_000,
    segment_seed_stride=1_000,
    start_state_description=(
        "k4 touchdown altitude 5-18 with pad offset <=20, |vx|<=3, vy 0-6, "
        "|tilt|<=0.08, |rate|<=0.05; k3/k2/k1 approach altitude "
        "30-100/50-200/100-500 with pad offset <=35, |vx|<=6, vy 2-18, "
        "|tilt|<=0.18, |rate|<=0.25; k0 canonical x=500, y=120, vy=rate=0, "
        "fuel=1, elapsed=0 with seeded |vx|<=15 and |tilt|<=0.15; "
        "rehearsals preserve altitude-derived elapsed time and fuel"
    ),
    make_evaluation_env=make_frontier_evaluation_env,
    frontier_success_rate_thresholds=CURRICULUM_SUCCESS_RATE_THRESHOLDS,
)
