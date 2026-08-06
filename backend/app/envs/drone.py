"""2D quadcopter: two rotor thrusts, fly a fixed 5-waypoint course."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .base import (
    EpisodeTrainingPhase,
    EpisodeTrainingScheduleSpec,
    TrainingControlSpec,
    TrainingCurriculumSpec,
)

DT = 0.04
G = 50.0
T_MAX = 70.0       # per rotor; hover needs ~0.36 each
HOVER_ACTION = 2.0 * (G / (2.0 * T_MAX)) - 1.0
TORQUE = 8.0
TIP_OVER = 1.3
CAPTURE_DIST = 25.0
HORIZON_STEPS = 900
HANDOFF_HORIZONTAL_SPEED_MIN = 60.0
HANDOFF_HORIZONTAL_SPEED_MAX = 100.0
MOMENTUM_STAGE_IDS = ("foundation", "bridge", "hard")
MOMENTUM_STAGE_SPEED_RANGES = ((-20.0, 20.0), (20.0, 60.0), (60.0, 100.0))
MOMENTUM_ACTIVE_STAGE_PROBABILITY = 0.75
MOMENTUM_SUCCESS_RATE_THRESHOLD = 0.8
MOMENTUM_CONSECUTIVE_CONFIRMATIONS = 1

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
APPROACH_REMAINING_DISTANCE = 120.0
HALF_SEGMENT_PROGRESS = 0.5
TRAINING_MODES = ("approach", "half", "handoff", "canonical")
TRAINING_START_DISTRIBUTION = (
    "Predeclared all-direction episode schedule: episodes 1-200 use 120-unit "
    "single-segment capture approaches; 201-450 sample 30% approach and 70% "
    "half-segment starts; 451-800 sample 10% approach, 20% half, and 70% full "
    "hard inbound handoffs; from episode 801 onward, 60% of resets use the "
    "unchanged jittered canonical full-course start and 40% retain uniformly "
    "sampled single-segment rehearsals (5% approach, 5% half, 30% handoff). "
    "The fixed schedule never reads evaluation results"
)
CURRICULUM_FRONTIER_ORDER = (4, 3, 2, 1, 0)
CURRICULUM_ACTIVE_FRONTIER_PROBABILITY = 0.5
CURRICULUM_SUCCESS_RATE_THRESHOLD = 0.9
CURRICULUM_CONSECUTIVE_CONFIRMATIONS = 1
TERMINAL_FAILURE_PENALTY = 50.0
ANGULAR_RATE_REGULARIZER = 0.002
THRUST_REGULARIZER = 0.005

# Potential-shaping targets are derived from a bounded braking envelope. At
# 20 units/s^2, slowing from the 80-unit/s cruise target to the 20-unit/s
# waypoint target takes 150 units, well inside every curriculum segment. The
# desired-tilt coefficient is the one-second velocity-error coefficient under
# hover gravity (0.10 * G * 1.0), so the first counter-thrust transition gets
# credit for creating the attitude that will reverse hard inbound momentum.
WAYPOINT_TARGET_SPEED = 20.0
WAYPOINT_CRUISE_SPEED = 80.0
WAYPOINT_COMFORT_DECELERATION = 20.0
VELOCITY_RESPONSE_SECONDS = 1.0
DESIRED_TILT_LIMIT = 0.6
DISTANCE_POTENTIAL_WEIGHT = 0.05
VELOCITY_ERROR_POTENTIAL_WEIGHT = 0.10
DESIRED_TILT_POTENTIAL_WEIGHT = (
    VELOCITY_ERROR_POTENTIAL_WEIGHT * G * VELOCITY_RESPONSE_SECONDS)
VELOCITY_OBSERVATION_SCALE = 100.0


TRAINING_SCHEDULE = EpisodeTrainingScheduleSpec(
    id="drone-all-direction-schedule-v1",
    phases=(
        EpisodeTrainingPhase(
            id="capture_approach", start_episode=1, end_episode=200,
            mode_probabilities=(("approach", 1.0),),
            description=(
                "uniformly sample all five directions 120 units before the "
                "active waypoint at zero velocity and end after one capture"),
        ),
        EpisodeTrainingPhase(
            id="half_segments", start_episode=201, end_episode=450,
            mode_probabilities=(("approach", 0.3), ("half", 0.7)),
            description=(
                "retain capture approaches while expanding every direction "
                "to its half-segment zero-velocity start"),
        ),
        EpisodeTrainingPhase(
            id="full_handoffs", start_episode=451, end_episode=800,
            mode_probabilities=(
                ("approach", 0.1), ("half", 0.2), ("handoff", 0.7)),
            description=(
                "retain easier starts while emphasizing full preceding-waypoint "
                "handoffs with the unchanged inbound 60-100 unit/s range"),
        ),
        EpisodeTrainingPhase(
            id="canonical_consolidation", start_episode=801, end_episode=None,
            mode_probabilities=(
                ("approach", 0.05), ("half", 0.05),
                ("handoff", 0.3), ("canonical", 0.6)),
            description=(
                "train mostly on the unchanged full course while retaining a "
                "minority of all-direction single-segment rehearsals"),
        ),
    ),
    start_state_description=(
        "single-segment modes choose k0-k4 uniformly with seeded x jitter; "
        "approach and half modes start at zero velocity along the segment; "
        "handoff starts use the exact preceding waypoint and, for k1-k4, "
        "the previous segment's horizontal sign at 60-100 units/s; canonical "
        "mode uses the ordinary jittered k0 full-course start at rest"),
    checkpoint_selection_description=(
        "schedule state and demonstrations are training-only; fixed full-course "
        "evaluation remains the checkpoint-selection signal"),
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
    episode_schedule: bool = False
    forced_start_segment: int | None = None
    forced_momentum_stage: str | None = None
    forced_training_mode: str | None = None
    rng: random.Random = field(default_factory=random.Random)
    _training_episode: int = field(default=1, init=False, repr=False)
    _training_phase: str | None = field(default=None, init=False, repr=False)
    _training_mode: str = field(default="canonical", init=False, repr=False)
    _training_segment: int | None = field(default=None, init=False, repr=False)
    _single_segment_episode: bool = field(default=False, init=False, repr=False)
    _training_segment_captured: bool = field(default=False, init=False, repr=False)
    _curriculum_position: int = field(default=0, init=False, repr=False)
    _curriculum_pass_streak: int = field(default=0, init=False, repr=False)
    _curriculum_complete: bool = field(default=False, init=False, repr=False)
    _curriculum_evaluations: int = field(default=0, init=False, repr=False)
    _curriculum_last_success_rate: float | None = field(
        default=None, init=False, repr=False)
    _curriculum_last_evaluation_episode: int | None = field(
        default=None, init=False, repr=False)
    _momentum_position: int = field(default=0, init=False, repr=False)
    _momentum_pass_streak: int = field(default=0, init=False, repr=False)
    _momentum_complete: bool = field(default=False, init=False, repr=False)
    _momentum_evaluations: int = field(default=0, init=False, repr=False)
    _momentum_last_success_rate: float | None = field(
        default=None, init=False, repr=False)
    _momentum_last_evaluation_episode: int | None = field(
        default=None, init=False, repr=False)

    obs_dim = 12
    n_continuous = 2
    n_binary = 0
    max_steps = HORIZON_STEPS
    dt = DT

    def __post_init__(self):
        if self.waypoint_start_curriculum and self.episode_schedule:
            raise ValueError("Drone training curricula are mutually exclusive")
        if (self.forced_start_segment is not None
                and self.forced_start_segment not in CURRICULUM_FRONTIER_ORDER):
            raise ValueError("forced Drone segment must be in [0, 4]")
        if (self.forced_momentum_stage is not None
                and self.forced_momentum_stage not in MOMENTUM_STAGE_IDS):
            raise ValueError("unknown forced Drone momentum stage")
        if (self.forced_momentum_stage is not None
                and self.forced_start_segment != 4):
            raise ValueError("forced momentum stages are defined only for k4")
        if (self.forced_training_mode is not None
                and self.forced_training_mode not in TRAINING_MODES):
            raise ValueError("unknown forced Drone training mode")
        if self.forced_training_mode is not None and not self.episode_schedule:
            raise ValueError("forced Drone training mode requires episode schedule")
        if (self.forced_training_mode == "canonical"
                and self.forced_start_segment not in (None, 0)):
            raise ValueError("canonical Drone training mode starts at k0")
        self.reset()

    def reset(self) -> np.ndarray:
        self._training_segment_captured = False
        if self.episode_schedule:
            self._reset_scheduled_start()
        else:
            self._training_phase = None
            self._training_mode = "canonical"
            self._training_segment = None
            self._single_segment_episode = False
            self.x, self.y = START
            self.k = self._sample_start_segment()
            if self.k > 0:
                self.x, self.y = WAYPOINTS[self.k - 1]
            if self.jitter:
                self.x += self.rng.uniform(-30.0, 30.0)
        self.vx = self.vy = 0.0
        self.theta = self.omega = 0.0
        if self.episode_schedule and self._training_mode == "handoff":
            self._set_handoff_velocity()
        elif not self.episode_schedule and self.k > 0:
            inbound_dx = (
                _COURSE_POINTS[self.k][0] - _COURSE_POINTS[self.k - 1][0])
            momentum_stage = self._momentum_stage_for_reset()
            if momentum_stage == "foundation":
                self.vx = self.rng.uniform(*MOMENTUM_STAGE_SPEED_RANGES[0])
            elif momentum_stage in ("bridge", "hard"):
                stage_position = MOMENTUM_STAGE_IDS.index(momentum_stage)
                speed_min, speed_max = MOMENTUM_STAGE_SPEED_RANGES[stage_position]
                self.vx = math.copysign(
                    self.rng.uniform(speed_min, speed_max),
                    inbound_dx,
                )
            else:
                self.vx = math.copysign(
                    self.rng.uniform(
                        HANDOFF_HORIZONTAL_SPEED_MIN,
                        HANDOFF_HORIZONTAL_SPEED_MAX,
                    ),
                    inbound_dx,
                )
        self.steps = (0 if self.k == 0
                      else WAYPOINT_START_STEPS[self.k - 1])
        self.episode_reward = 0.0
        self._completion_regularizer = 0.0
        self.cause = "running"
        self._shaping_potential_prev = self._shaping_potential()
        return self._obs()

    def reset_for_training_episode(self, episode: int) -> np.ndarray:
        episode = int(episode)
        if episode < 1:
            raise ValueError("Drone training episode must be positive")
        self._training_episode = episode
        return self.reset()

    def _reset_scheduled_start(self) -> None:
        phase, mode = TRAINING_SCHEDULE.sample_mode(
            self.rng, self._training_episode)
        if self.forced_training_mode is not None:
            mode = self.forced_training_mode
        self._training_phase = phase
        self._training_mode = mode
        self._single_segment_episode = mode != "canonical"
        if mode == "canonical":
            self._training_segment = None
            self.k = 0
            self.x, self.y = START
        else:
            self.k = (self.forced_start_segment
                      if self.forced_start_segment is not None
                      else self.rng.choice(CURRICULUM_FRONTIER_ORDER))
            self._training_segment = self.k
            origin = _COURSE_POINTS[self.k]
            target = WAYPOINTS[self.k]
            if mode == "approach":
                remaining = min(
                    APPROACH_REMAINING_DISTANCE, _SEGMENT_DISTANCES[self.k])
                progress = 1.0 - remaining / _SEGMENT_DISTANCES[self.k]
            elif mode == "half":
                progress = HALF_SEGMENT_PROGRESS
            elif mode == "handoff":
                progress = 0.0
            else:
                raise RuntimeError("unsupported Drone training mode")
            self.x = origin[0] + progress * (target[0] - origin[0])
            self.y = origin[1] + progress * (target[1] - origin[1])
        if self.jitter:
            self.x += self.rng.uniform(-30.0, 30.0)

    def _set_handoff_velocity(self) -> None:
        if self.k == 0:
            return
        inbound_dx = _COURSE_POINTS[self.k][0] - _COURSE_POINTS[self.k - 1][0]
        self.vx = math.copysign(
            self.rng.uniform(
                HANDOFF_HORIZONTAL_SPEED_MIN,
                HANDOFF_HORIZONTAL_SPEED_MAX,
            ),
            inbound_dx,
        )

    @property
    def training_phase(self) -> str | None:
        return self._training_phase

    @property
    def training_mode(self) -> str:
        return self._training_mode

    @property
    def training_segment(self) -> int | None:
        return self._training_segment

    @property
    def single_segment_episode(self) -> bool:
        return self._single_segment_episode

    def _momentum_stage_for_reset(self) -> str | None:
        """Choose a k4 stage only while its nested training control is active."""
        if self.forced_momentum_stage is not None:
            return self.forced_momentum_stage
        if (not self.waypoint_start_curriculum
                or self.forced_start_segment is not None
                or self._curriculum_position != 0
                or self.k != 4):
            return None
        active = MOMENTUM_STAGE_IDS[self._momentum_position]
        mastered = MOMENTUM_STAGE_IDS[:self._momentum_position]
        if (not mastered
                or self.rng.random() < MOMENTUM_ACTIVE_STAGE_PROBABILITY):
            return active
        return self.rng.choice(mastered)

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
        if self.episode_schedule:
            return {
                "version": 3,
                "schedule_id": TRAINING_SCHEDULE.id,
                "episode": self._training_episode,
                "phase": self._training_phase,
                "mode": self._training_mode,
                "segment": self._training_segment,
                "single_segment_episode": self._single_segment_episode,
                "segment_captured": self._training_segment_captured,
            }
        frontier = CURRICULUM_FRONTIER_ORDER[self._curriculum_position]
        return {
            "version": 2,
            "frontier_position": self._curriculum_position,
            "frontier": frontier,
            "mastered": list(
                CURRICULUM_FRONTIER_ORDER[:self._curriculum_position]),
            "pass_streak": self._curriculum_pass_streak,
            "complete": self._curriculum_complete,
            "evaluations": self._curriculum_evaluations,
            "last_success_rate": self._curriculum_last_success_rate,
            "last_evaluation_episode": self._curriculum_last_evaluation_episode,
            "momentum": self.training_control_state(),
        }

    def training_control_state(self) -> dict:
        """Return the exact nested k4 momentum-control state."""
        stage = MOMENTUM_STAGE_IDS[self._momentum_position]
        return {
            "version": 1,
            "control_id": "drone-k4-momentum-v1",
            "stage_position": self._momentum_position,
            "stage": stage,
            "mastered": list(MOMENTUM_STAGE_IDS[:self._momentum_position]),
            "pass_streak": self._momentum_pass_streak,
            "complete": self._momentum_complete,
            "evaluations": self._momentum_evaluations,
            "last_success_rate": self._momentum_last_success_rate,
            "last_evaluation_episode": self._momentum_last_evaluation_episode,
        }

    def restore_training_control_state(self, state: dict) -> None:
        """Restore and validate the nested k4 momentum-control state."""
        position = int(state["stage_position"])
        pass_streak = int(state["pass_streak"])
        evaluations = int(state["evaluations"])
        complete = bool(state["complete"])
        last_success_rate = state.get("last_success_rate")
        last_evaluation_episode = state.get("last_evaluation_episode")
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported Drone momentum state version")
        if state.get("control_id") != "drone-k4-momentum-v1":
            raise ValueError("Drone momentum control id does not match")
        if not 0 <= position < len(MOMENTUM_STAGE_IDS):
            raise ValueError("invalid Drone momentum stage position")
        if state.get("stage") != MOMENTUM_STAGE_IDS[position]:
            raise ValueError("Drone momentum stage does not match position")
        if list(state.get("mastered", [])) != list(
                MOMENTUM_STAGE_IDS[:position]):
            raise ValueError("Drone momentum mastered stages are inconsistent")
        if not 0 <= pass_streak <= MOMENTUM_CONSECUTIVE_CONFIRMATIONS:
            raise ValueError("invalid Drone momentum confirmation streak")
        if evaluations < 0:
            raise ValueError("invalid Drone momentum evaluation count")
        if complete and position != len(MOMENTUM_STAGE_IDS) - 1:
            raise ValueError("only the hard momentum stage can be complete")
        if last_success_rate is not None:
            last_success_rate = float(last_success_rate)
            if not 0.0 <= last_success_rate <= 1.0:
                raise ValueError("invalid Drone momentum success rate")
        if last_evaluation_episode is not None:
            last_evaluation_episode = int(last_evaluation_episode)
            if last_evaluation_episode < 0:
                raise ValueError("invalid Drone momentum evaluation episode")
        self._momentum_position = position
        self._momentum_pass_streak = pass_streak
        self._momentum_complete = complete
        self._momentum_evaluations = evaluations
        self._momentum_last_success_rate = last_success_rate
        self._momentum_last_evaluation_episode = last_evaluation_episode

    def restore_training_curriculum_state(self, state: dict) -> None:
        """Restore a validated state saved in the checkpoint RNG payload."""
        if self.episode_schedule:
            self._restore_episode_schedule_state(state)
            return
        position = int(state["frontier_position"])
        pass_streak = int(state["pass_streak"])
        evaluations = int(state["evaluations"])
        complete = bool(state["complete"])
        last_success_rate = state.get("last_success_rate")
        last_evaluation_episode = state.get("last_evaluation_episode")
        if int(state.get("version", -1)) != 2:
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
        momentum = state.get("momentum")
        if not isinstance(momentum, dict):
            raise ValueError("Drone curriculum state lacks momentum control")
        if position > 0 and not bool(momentum.get("complete")):
            raise ValueError("an unlocked Drone frontier requires hard proficiency")
        self.restore_training_control_state(momentum)
        self._curriculum_position = position
        self._curriculum_pass_streak = pass_streak
        self._curriculum_complete = complete
        self._curriculum_evaluations = evaluations
        self._curriculum_last_success_rate = last_success_rate
        self._curriculum_last_evaluation_episode = last_evaluation_episode

    def _restore_episode_schedule_state(self, state: dict) -> None:
        if int(state.get("version", -1)) != 3:
            raise ValueError("unsupported Drone schedule state version")
        if state.get("schedule_id") != TRAINING_SCHEDULE.id:
            raise ValueError("Drone training schedule id does not match")
        episode = int(state["episode"])
        expected_phase = TRAINING_SCHEDULE.phase_for_episode(episode)
        phase = state.get("phase")
        if phase != expected_phase.id:
            raise ValueError("Drone training schedule phase does not match episode")
        mode = state.get("mode")
        allowed_modes = {name for name, _ in expected_phase.mode_probabilities}
        if self.forced_training_mode is not None:
            allowed_modes.add(self.forced_training_mode)
        if mode not in allowed_modes:
            raise ValueError("Drone training mode does not match schedule phase")
        segment = state.get("segment")
        if segment is not None:
            segment = int(segment)
        if mode == "canonical":
            if segment is not None:
                raise ValueError("canonical Drone schedule state has a segment")
        elif segment not in CURRICULUM_FRONTIER_ORDER:
            raise ValueError("single-segment Drone schedule state lacks a segment")
        single_segment = bool(state["single_segment_episode"])
        if single_segment != (mode != "canonical"):
            raise ValueError("Drone single-segment flag does not match mode")
        segment_captured = bool(state["segment_captured"])

        self._training_episode = episode
        self._training_phase = phase
        self._training_mode = str(mode)
        self._training_segment = segment
        self._single_segment_episode = single_segment
        self._training_segment_captured = segment_captured

    def reset_training_curriculum(self) -> None:
        if self.episode_schedule:
            self._training_episode = 1
            self._training_phase = None
            self._training_mode = "canonical"
            self._training_segment = None
            self._single_segment_episode = False
            self._training_segment_captured = False
            return
        self._curriculum_position = 0
        self._curriculum_pass_streak = 0
        self._curriculum_complete = False
        self._curriculum_evaluations = 0
        self._curriculum_last_success_rate = None
        self._curriculum_last_evaluation_episode = None
        self._momentum_position = 0
        self._momentum_pass_streak = 0
        self._momentum_complete = False
        self._momentum_evaluations = 0
        self._momentum_last_success_rate = None
        self._momentum_last_evaluation_episode = None

    def record_training_control_evaluation(
            self, success_rate: float,
            control: TrainingControlSpec, *,
            evaluation_episode: int | None = None) -> dict:
        """Apply one fixed-suite result to the active momentum stage."""
        if tuple(control.stage_ids) != MOMENTUM_STAGE_IDS:
            raise ValueError("momentum stage order does not match Drone")
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError("momentum success rate must be in [0, 1]")
        if evaluation_episode is None:
            evaluation_episode = (
                0 if self._momentum_last_evaluation_episode is None
                else self._momentum_last_evaluation_episode + 1)
        evaluation_episode = int(evaluation_episode)
        if evaluation_episode < 0:
            raise ValueError("momentum evaluation episode must be non-negative")
        before = self.training_control_state()
        passed = success_rate >= control.success_rate_threshold
        if self._momentum_last_evaluation_episode == evaluation_episode:
            return {
                "success_rate": float(success_rate),
                "passed": passed,
                "evaluation_episode": evaluation_episode,
                "ignored_duplicate": True,
                "stage_before": before["stage"],
                "stage_after": before["stage"],
                "mastered_before": before["mastered"],
                "mastered_after": before["mastered"],
                "confirmation_streak_before": before["pass_streak"],
                "confirmation_streak_after": before["pass_streak"],
                "complete_before": before["complete"],
                "complete_after": before["complete"],
                "unlocked": False,
                "completed": False,
            }
        if (self._momentum_last_evaluation_episode is not None
                and evaluation_episode < self._momentum_last_evaluation_episode):
            raise ValueError("momentum evaluation episodes must be monotonic")
        self._momentum_evaluations += 1
        self._momentum_last_success_rate = float(success_rate)
        self._momentum_last_evaluation_episode = evaluation_episode
        unlocked = False
        completed = False
        if not self._momentum_complete:
            self._momentum_pass_streak = (
                self._momentum_pass_streak + 1 if passed else 0)
            if self._momentum_pass_streak >= control.consecutive_confirmations:
                if self._momentum_position < len(MOMENTUM_STAGE_IDS) - 1:
                    self._momentum_position += 1
                    self._momentum_pass_streak = 0
                    unlocked = True
                else:
                    self._momentum_complete = True
                    completed = True
        after = self.training_control_state()
        return {
            "success_rate": float(success_rate),
            "passed": passed,
            "evaluation_episode": evaluation_episode,
            "ignored_duplicate": False,
            "stage_before": before["stage"],
            "stage_after": after["stage"],
            "mastered_before": before["mastered"],
            "mastered_after": after["mastered"],
            "confirmation_streak_before": before["pass_streak"],
            "confirmation_streak_after": after["pass_streak"],
            "complete_before": before["complete"],
            "complete_after": after["complete"],
            "unlocked": unlocked,
            "completed": completed,
        }

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
        prerequisite_met = (
            before["frontier"] != 4 or before["momentum"]["complete"])
        unlocked = False
        if self._curriculum_last_evaluation_episode == evaluation_episode:
            return {
                "success_rate": float(success_rate),
                "passed": passed,
                "prerequisite_met": prerequisite_met,
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
                self._curriculum_pass_streak + 1
                if passed and prerequisite_met else 0)
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
            "prerequisite_met": prerequisite_met,
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

    def _target(self, waypoint_index: int | None = None) -> tuple[float, float]:
        index = (min(self.k, len(WAYPOINTS) - 1)
                 if waypoint_index is None else int(waypoint_index))
        if not 0 <= index < len(WAYPOINTS):
            raise ValueError("Drone waypoint index must be in [0, 4]")
        return WAYPOINTS[index]

    def _dist(self, waypoint_index: int | None = None) -> float:
        wx, wy = self._target(waypoint_index)
        return math.hypot(self.x - wx, self.y - wy)

    @staticmethod
    def _braking_speed_target(distance_to_capture: float) -> float:
        """Desired speed with enough distance to brake at the comfort rate."""
        distance = max(float(distance_to_capture), 0.0)
        target = math.sqrt(
            WAYPOINT_TARGET_SPEED ** 2
            + 2.0 * WAYPOINT_COMFORT_DECELERATION * distance)
        return min(WAYPOINT_CRUISE_SPEED, target)

    def _desired_velocity_target(
            self, waypoint_index: int | None = None) -> tuple[float, float]:
        """Velocity vector toward the active waypoint's braking envelope."""
        wx, wy = self._target(waypoint_index)
        dx, dy = wx - self.x, wy - self.y
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            return 0.0, 0.0
        speed = self._braking_speed_target(
            max(distance - CAPTURE_DIST, 0.0))
        return speed * dx / distance, speed * dy / distance

    def _desired_tilt_target(self, desired_vx: float | None = None) -> float:
        """Tilt that closes horizontal velocity error over one response time."""
        if desired_vx is None:
            desired_vx, _ = self._desired_velocity_target()
        desired_acceleration = (
            (desired_vx - self.vx) / VELOCITY_RESPONSE_SECONDS)
        acceleration_limit = G * math.sin(DESIRED_TILT_LIMIT)
        desired_acceleration = float(np.clip(
            desired_acceleration, -acceleration_limit, acceleration_limit))
        return math.asin(desired_acceleration / G)

    def _shaping_potential(self, waypoint_index: int | None = None) -> float:
        """Return the objective-aligned cost potential for one course segment.

        This is deliberately a training credit-assignment auxiliary, not a
        policy-invariant terminal-zero potential. Failed attempts retain their
        final segment cost so GAE does not see a positive terminal correction
        for crashing from an expensive state.
        """
        desired_vx, desired_vy = self._desired_velocity_target(waypoint_index)
        velocity_error = math.hypot(
            self.vx - desired_vx, self.vy - desired_vy)
        desired_tilt = self._desired_tilt_target(desired_vx)
        cost = (
            DISTANCE_POTENTIAL_WEIGHT * self._dist(waypoint_index)
            + VELOCITY_ERROR_POTENTIAL_WEIGHT * velocity_error
            + DESIRED_TILT_POTENTIAL_WEIGHT
            * abs(self.theta - desired_tilt)
        )
        return -cost

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        active_waypoint_index = min(self.k, len(WAYPOINTS) - 1)
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

        d = self._dist(active_waypoint_index)
        segment_end_potential = self._shaping_potential(active_waypoint_index)
        self._completion_regularizer += (
            ANGULAR_RATE_REGULARIZER * abs(self.omega)
            + THRUST_REGULARIZER * (t_l ** 2 + t_r ** 2)
        )

        done = False
        captured_waypoint = False
        task_reward = 0.0
        if d < CAPTURE_DIST:
            captured_waypoint = True
            task_reward += 20.0
            self.k += 1
            if self._single_segment_episode:
                self._training_segment_captured = True
                done, self.cause = True, "training_segment_complete"
            elif self.k >= len(WAYPOINTS):
                # Energy and attitude are secondary efficiency tie-breakers,
                # not a reason to terminate a failed attempt early. Charging
                # the path regularizer only on success keeps failed returns
                # at the terminal outcome plus a start-state shaping constant,
                # while preserving the original successful-course score.
                task_reward += 50.0 - self._completion_regularizer
                done, self.cause = True, "complete"
        if not done and (abs(self.theta) > TIP_OVER
                         or self.x < 0 or self.x > 1000
                         or self.y < 0 or self.y > 700):
            task_reward -= TERMINAL_FAILURE_PENALTY
            done, self.cause = True, "crash"
        elif not done and self.steps >= self.max_steps:
            task_reward -= TERMINAL_FAILURE_PENALTY
            done, self.cause = True, "timeout"

        # Give local progress credit only within the segment that owned this
        # transition. A waypoint capture starts a fresh baseline for the next
        # target without charging its discontinuous target-switch jump. At a
        # crash, timeout, or completion, retain the physical end potential;
        # forcing it to zero would give high-cost terminal states a spurious
        # positive correction under finite-lambda GAE.
        reward = (segment_end_potential - self._shaping_potential_prev
                  + task_reward)
        self._shaping_potential_prev = (
            self._shaping_potential()
            if captured_waypoint and not done
            else segment_end_potential
        )

        self.episode_reward += reward
        deadline = done and self.cause == "timeout"
        return self._obs(), reward, done, {
            "truncated": deadline,
            "task_deadline": deadline,
        }

    def _obs(self) -> np.ndarray:
        wx, wy = self._target()
        desired_vx, desired_vy = self._desired_velocity_target()
        desired_tilt = self._desired_tilt_target(desired_vx)
        return np.array([
            (wx - self.x) / 300.0,
            (wy - self.y) / 300.0,
            self.vx / VELOCITY_OBSERVATION_SCALE,
            self.vy / VELOCITY_OBSERVATION_SCALE,
            (desired_vx - self.vx) / VELOCITY_OBSERVATION_SCALE,
            (desired_vy - self.vy) / VELOCITY_OBSERVATION_SCALE,
            (desired_tilt - self.theta) / TIP_OVER,
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
        canonical_success = self.cause == "complete"
        training_success = self.cause == "training_segment_complete"
        return {
            "reward": round(self.episode_reward, 2),
            "steps": self.steps,
            "cause": self.cause,
            "metric": (float(self._training_segment_captured)
                       if self._single_segment_episode else float(self.k)),
            "success": canonical_success or training_success,
            "canonical_success": canonical_success,
            "training_mode": self._training_mode,
            "training_segment": self._training_segment,
        }

    def ghost_sample(self) -> list[float]:
        return [round(self.x, 1), round(self.y, 1), round(self.theta, 3),
                0.0, round(math.hypot(self.vx, self.vy), 1)]


def physics_reference_action(env: DroneEnv) -> np.ndarray:
    """Bounded PD controller used only to generate disclosed demonstrations."""
    desired_vx, desired_vy = env._desired_velocity_target()
    ax = float(np.clip((desired_vx - env.vx) / 0.5, -50.0, 50.0))
    ay = float(np.clip((desired_vy - env.vy) / 0.5, -50.0, 50.0))
    upward_force = G - ay
    desired_theta = float(np.clip(
        math.atan2(ax, upward_force), -1.0, 1.0))
    thrust = float(np.clip(math.hypot(ax, upward_force), 0.0, 2.0 * T_MAX))
    thrust_sum = thrust / T_MAX
    angular_acceleration = 25.0 * (desired_theta - env.theta) - 7.0 * env.omega
    thrust_difference = float(np.clip(
        angular_acceleration / TORQUE, -thrust_sum, thrust_sum))
    left = float(np.clip((thrust_sum - thrust_difference) / 2.0, 0.0, 1.0))
    right = float(np.clip((thrust_sum + thrust_difference) / 2.0, 0.0, 1.0))
    return np.asarray([2.0 * left - 1.0, 2.0 * right - 1.0],
                      dtype=np.float32)


def behavior_cloning_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate the frozen-seed canonical expert state/action dataset."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode_index in range(80):
        env = DroneEnv(jitter=True)
        env.rng.seed(600_000 + episode_index)
        obs = env.reset()
        for _ in range(env.max_steps):
            action = physics_reference_action(env)
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, _, done, _ = env.step(action)
            if done:
                break
        if not env.episode_summary()["canonical_success"]:
            raise RuntimeError(
                f"Drone demonstration {episode_index} did not complete")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def make_segment_evaluation_env(segment: int) -> DroneEnv:
    """Build one fixed-suite env without stochastic training-start sampling."""
    if segment not in CURRICULUM_FRONTIER_ORDER:
        raise ValueError("Drone segment must be in [0, 4]")
    return DroneEnv(jitter=True, forced_start_segment=segment)


def make_momentum_evaluation_env(stage: str) -> DroneEnv:
    """Build one deterministic-control k4 env for a disclosed momentum stage."""
    if stage not in MOMENTUM_STAGE_IDS:
        raise ValueError("unknown Drone momentum stage")
    return DroneEnv(
        jitter=True,
        forced_start_segment=4,
        forced_momentum_stage=stage,
    )


MOMENTUM_CONTROL = TrainingControlSpec(
    id="drone-k4-momentum-v1",
    scope_description="outer frontier k4 before it may unlock",
    stage_ids=MOMENTUM_STAGE_IDS,
    stage_start_descriptions=(
        "k4 at waypoint 4 with standard seeded horizontal position jitter, "
        "signed horizontal velocity uniform on [-20, 20] units/s, zero "
        "vertical velocity/attitude/rate, cumulative-distance elapsed clock",
        "k4 at waypoint 4 with standard seeded horizontal position jitter, "
        "inbound horizontal velocity using the previous-segment sign and "
        "magnitude uniform on [20, 60] units/s, zero vertical velocity/attitude/"
        "rate, cumulative-distance elapsed clock",
        "k4 at waypoint 4 with standard seeded horizontal position jitter, "
        "inbound horizontal velocity using the previous-segment sign and "
        "magnitude uniform on [60, 100] units/s, zero vertical velocity/attitude/"
        "rate, cumulative-distance elapsed clock",
    ),
    active_stage_probability=MOMENTUM_ACTIVE_STAGE_PROBABILITY,
    success_rate_threshold=MOMENTUM_SUCCESS_RATE_THRESHOLD,
    consecutive_confirmations=MOMENTUM_CONSECUTIVE_CONFIRMATIONS,
    evaluation_suite_version="drone-momentum-control-v1",
    evaluation_episodes=10,
    evaluation_seed_base=500_000,
    stage_seed_stride=1_000,
    evaluation_start_state_description=(
        "the active stage's disclosed k4 start state"),
    outer_gate_dependency_description=(
        "the k4 segment gate remains locked until momentum control is complete"),
    checkpoint_selection_description=(
        "momentum control is training-only; fixed full-course evaluation remains "
        "the checkpoint-selection signal"),
    make_evaluation_env=make_momentum_evaluation_env,
)


TRAINING_CURRICULUM = TrainingCurriculumSpec(
    id="drone-reverse-waypoint-v1",
    frontier_order=CURRICULUM_FRONTIER_ORDER,
    active_frontier_probability=CURRICULUM_ACTIVE_FRONTIER_PROBABILITY,
    success_rate_threshold=CURRICULUM_SUCCESS_RATE_THRESHOLD,
    consecutive_confirmations=CURRICULUM_CONSECUTIVE_CONFIRMATIONS,
    evaluation_suite_version="drone-segment-eval-v3",
    evaluation_episodes=10,
    evaluation_seed_base=400_000,
    segment_seed_stride=1_000,
    start_state_description=(
        "noncanonical segment at preceding waypoint with standard seeded "
        "horizontal position jitter, inbound horizontal velocity using "
        "the previous-segment sign and magnitude uniform on [60, 100] "
        "units/s, zero vertical velocity/attitude/rate, cumulative-distance "
        "elapsed clock; k0 remains the canonical zero-motion start"
    ),
    make_evaluation_env=make_segment_evaluation_env,
    training_control=MOMENTUM_CONTROL,
)
