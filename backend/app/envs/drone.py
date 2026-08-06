"""2D quadcopter: two rotor thrusts, fly a fixed 5-waypoint course."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .base import TrainingCurriculumSpec

DT = 0.04
G = 50.0
T_MAX = 70.0       # per rotor; hover needs ~0.36 each
TORQUE = 8.0
TIP_OVER = 1.3
CAPTURE_DIST = 25.0
HORIZON_STEPS = 900

WAYPOINTS: list[tuple[float, float]] = [
    (250, 500), (700, 420), (450, 180), (820, 160), (150, 250),
]
START = (500.0, 560.0)
_COURSE_POINTS = (START, *WAYPOINTS)
_SEGMENT_DISTANCES = tuple(
    math.dist(start, end)
    for start, end in zip(_COURSE_POINTS, _COURSE_POINTS[1:])
)
_COURSE_DISTANCE = sum(_SEGMENT_DISTANCES)
WAYPOINT_START_STEPS = tuple(
    round(HORIZON_STEPS * sum(_SEGMENT_DISTANCES[:completed])
          / _COURSE_DISTANCE)
    for completed in range(1, len(WAYPOINTS))
)
TRAINING_START_DISTRIBUTION = (
    "Performance-gated reverse waypoint curriculum: start at target 5 "
    "(k4); unlock k3, k2, k1, then canonical k0 after one >=90% fixed "
    "segment evaluation; active frontier receives 100% of resets"
)
CURRICULUM_FRONTIER_ORDER = (4, 3, 2, 1, 0)
CURRICULUM_ACTIVE_FRONTIER_PROBABILITY = 1.0
CURRICULUM_SUCCESS_RATE_THRESHOLD = 0.9
CURRICULUM_CONSECUTIVE_CONFIRMATIONS = 1


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
    forced_start_segment: int | None = None
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
    max_steps = HORIZON_STEPS
    dt = DT

    def __post_init__(self):
        if (self.forced_start_segment is not None
                and self.forced_start_segment not in CURRICULUM_FRONTIER_ORDER):
            raise ValueError("forced Drone segment must be in [0, 4]")
        self.reset()

    def reset(self) -> np.ndarray:
        self.x, self.y = START
        self.k = self._sample_start_segment()
        if self.k > 0:
            self.x, self.y = WAYPOINTS[self.k - 1]
        if self.jitter:
            self.x += self.rng.uniform(-30.0, 30.0)
        self.vx = self.vy = 0.0
        self.theta = self.omega = 0.0
        self.steps = (0 if self.k == 0
                      else WAYPOINT_START_STEPS[self.k - 1])
        self.episode_reward = 0.0
        self.cause = "running"
        self._d_prev = self._dist()
        return self._obs()

    def _sample_start_segment(self) -> int:
        if self.forced_start_segment is not None:
            return self.forced_start_segment
        if not self.waypoint_start_curriculum:
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
        """Restore a validated state saved in the checkpoint RNG payload."""
        position = int(state["frontier_position"])
        pass_streak = int(state["pass_streak"])
        evaluations = int(state["evaluations"])
        complete = bool(state["complete"])
        last_success_rate = state.get("last_success_rate")
        last_evaluation_episode = state.get("last_evaluation_episode")
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported Drone curriculum state version")
        if not 0 <= position < len(CURRICULUM_FRONTIER_ORDER):
            raise ValueError("invalid Drone curriculum frontier position")
        if int(state.get("frontier", -1)) != CURRICULUM_FRONTIER_ORDER[position]:
            raise ValueError("Drone curriculum frontier does not match position")
        if list(state.get("mastered", [])) != list(
                CURRICULUM_FRONTIER_ORDER[:position]):
            raise ValueError("Drone curriculum mastered segments are inconsistent")
        if not 0 <= pass_streak <= CURRICULUM_CONSECUTIVE_CONFIRMATIONS:
            raise ValueError("invalid Drone curriculum confirmation streak")
        if evaluations < 0:
            raise ValueError("invalid Drone curriculum evaluation count")
        if complete and position != len(CURRICULUM_FRONTIER_ORDER) - 1:
            raise ValueError("only the canonical frontier can be complete")
        if last_success_rate is not None:
            last_success_rate = float(last_success_rate)
            if not 0.0 <= last_success_rate <= 1.0:
                raise ValueError("invalid Drone curriculum success rate")
        if last_evaluation_episode is not None:
            last_evaluation_episode = int(last_evaluation_episode)
            if last_evaluation_episode < 0:
                raise ValueError("invalid Drone curriculum evaluation episode")
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
        """Apply one fixed-suite result to the two-confirmation gate."""
        if tuple(curriculum.frontier_order) != CURRICULUM_FRONTIER_ORDER:
            raise ValueError("curriculum frontier order does not match Drone")
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError("curriculum success rate must be in [0, 1]")
        if evaluation_episode is None:
            evaluation_episode = (
                0 if self._curriculum_last_evaluation_episode is None
                else self._curriculum_last_evaluation_episode + 1)
        evaluation_episode = int(evaluation_episode)
        if evaluation_episode < 0:
            raise ValueError("curriculum evaluation episode must be non-negative")
        before = self.training_curriculum_state()
        passed = success_rate >= curriculum.success_rate_threshold
        unlocked = False
        if self._curriculum_last_evaluation_episode == evaluation_episode:
            return {
                "success_rate": float(success_rate),
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
        self._curriculum_last_success_rate = float(success_rate)
        self._curriculum_last_evaluation_episode = evaluation_episode
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
            "success_rate": float(success_rate),
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


def make_segment_evaluation_env(segment: int) -> DroneEnv:
    """Build one fixed-suite env without stochastic training-start sampling."""
    if segment not in CURRICULUM_FRONTIER_ORDER:
        raise ValueError("Drone segment must be in [0, 4]")
    return DroneEnv(jitter=True, forced_start_segment=segment)


TRAINING_CURRICULUM = TrainingCurriculumSpec(
    id="drone-reverse-waypoint-v1",
    frontier_order=CURRICULUM_FRONTIER_ORDER,
    active_frontier_probability=CURRICULUM_ACTIVE_FRONTIER_PROBABILITY,
    success_rate_threshold=CURRICULUM_SUCCESS_RATE_THRESHOLD,
    consecutive_confirmations=CURRICULUM_CONSECUTIVE_CONFIRMATIONS,
    evaluation_suite_version="drone-segment-eval-v1",
    evaluation_episodes=10,
    evaluation_seed_base=200_000,
    segment_seed_stride=1_000,
    start_state_description=(
        "preceding waypoint, zero velocity/attitude/rate, standard seeded "
        "horizontal jitter, cumulative-distance elapsed clock"
    ),
    make_evaluation_env=make_segment_evaluation_env,
)
