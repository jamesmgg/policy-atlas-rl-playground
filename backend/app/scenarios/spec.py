from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..envs.base import Env, TrainingCurriculumSpec
from ..ppo.initialization import ActorInitialization


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    name: str
    group: str                       # Circuits | Weather | Vehicles | Objectives | Classic
    kind: str                        # "driving" | "generic"
    description: str
    metric_label: str                # e.g. "best lap", "style pts", "waypoints"
    metric_mode: str                 # "min" (lap times) | "max" (scores)
    make_env: Callable[[bool], Env]  # arg: jitter (False for deterministic eval)
    scene: Callable[[], dict]        # static geometry for the frontend
    training_factory: Callable[[], Env] | None = None
    training_start_distribution: str = "scenario default starts"
    objective: str = "Maximize expected return."
    success: str = "Complete the task before the time limit."
    observations: tuple[str, ...] = ("agent state", "task state")
    observation_dimensions: tuple[str, ...] = ("agent state",)
    actions: tuple[str, ...] = ("continuous control",)
    reward_terms: tuple[str, ...] = ("task progress changes the return",)
    termination_conditions: tuple[str, ...] = ("success, failure, or horizon",)
    difficulty: str = "Intermediate"
    horizon_steps: int = 1
    horizon_seconds: float | None = 60.0
    gamma: float = 0.995
    actor_initialization: ActorInitialization | None = None
    training_curriculum: TrainingCurriculumSpec | None = None
    # Increment when observations, actions, rewards, optimization, or metric
    # semantics change incompatibly. Old checkpoints remain on disk but are
    # hidden rather than loaded or compared under a different scientific
    # contract.
    checkpoint_schema: int = 1

    def make_training_env(self) -> Env:
        """Build the learning environment without changing evaluation starts."""
        if self.training_factory is not None:
            return self.training_factory()
        return self.make_env(True)

    def info(self) -> dict:
        return {
            "id": self.id, "name": self.name, "group": self.group,
            "kind": self.kind, "description": self.description,
            "metric_label": self.metric_label, "metric_mode": self.metric_mode,
            "objective": self.objective, "success": self.success,
            "observations": list(self.observations), "actions": list(self.actions),
            "observation_dimensions": list(self.observation_dimensions),
            "reward_terms": list(self.reward_terms),
            "termination_conditions": list(self.termination_conditions),
            "training_start_distribution": self.training_start_distribution,
            "difficulty": self.difficulty,
            "horizon_steps": self.horizon_steps,
            "horizon_seconds": self.horizon_seconds,
            "gamma": self.gamma,
        }
