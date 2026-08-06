"""Explicit, serializable initialization contracts for PPO action heads."""
from __future__ import annotations

from dataclasses import dataclass
import math


CONTINUOUS_HEAD_WEIGHT_STD = 0.01
BINARY_HEAD_WEIGHT_STD = 0.01
INITIAL_CONTINUOUS_LOG_STD = -0.5


@dataclass(frozen=True)
class ActorInitialization:
    """A semantic prior for freshly constructed policy action heads.

    Continuous priors are expressed in the bounded action space consumed by
    environments, not in the latent Gaussian space. Binary priors are
    Bernoulli probabilities. The network performs the corresponding inverse
    transforms when it initializes its output biases.
    """

    scope: str
    continuous_action_labels: tuple[str, ...]
    continuous_action_prior: tuple[float, ...]
    continuous_log_std: tuple[float, ...] | None = None
    binary_action_labels: tuple[str, ...] = ()
    binary_probability_prior: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.continuous_action_labels) != len(
                self.continuous_action_prior):
            raise ValueError(
                "continuous action labels and priors must have equal length")
        if self.continuous_log_std is None:
            object.__setattr__(
                self,
                "continuous_log_std",
                (INITIAL_CONTINUOUS_LOG_STD,) * len(
                    self.continuous_action_prior),
            )
        if len(self.continuous_log_std) != len(
                self.continuous_action_prior):
            raise ValueError(
                "continuous log std and priors must have equal length")
        if len(self.binary_action_labels) != len(
                self.binary_probability_prior):
            raise ValueError(
                "binary action labels and priors must have equal length")
        if any(not math.isfinite(value) or not -1.0 < value < 1.0
               for value in self.continuous_action_prior):
            raise ValueError(
                "continuous action priors must be finite and strictly inside "
                "[-1, 1]")
        if any(not math.isfinite(value) for value in self.continuous_log_std):
            raise ValueError("continuous log std values must be finite")
        if any(not math.isfinite(value) or not 0.0 < value < 1.0
               for value in self.binary_probability_prior):
            raise ValueError(
                "binary probability priors must be finite and strictly inside "
                "[0, 1]")

    def validate_dimensions(self, n_continuous: int, n_binary: int) -> None:
        if len(self.continuous_action_prior) != n_continuous:
            raise ValueError(
                "actor initialization defines "
                f"{len(self.continuous_action_prior)} continuous actions; "
                f"environment expects {n_continuous}")
        if len(self.binary_probability_prior) != n_binary:
            raise ValueError(
                "actor initialization defines "
                f"{len(self.binary_probability_prior)} binary actions; "
                f"environment expects {n_binary}")

    @property
    def continuous_latent_bias(self) -> tuple[float, ...]:
        return tuple(math.atanh(value) for value in self.continuous_action_prior)

    @property
    def binary_logit_bias(self) -> tuple[float, ...]:
        return tuple(
            math.log(probability / (1.0 - probability))
            for probability in self.binary_probability_prior
        )

    def protocol(self, n_continuous: int, n_binary: int) -> dict:
        self.validate_dimensions(n_continuous, n_binary)
        return {
            "scope": self.scope,
            "applies_on": "fresh_seeded_policy_construction",
            "continuous_action_labels": list(self.continuous_action_labels),
            "continuous_action_prior": list(self.continuous_action_prior),
            "continuous_latent_bias": list(self.continuous_latent_bias),
            "continuous_log_std": list(self.continuous_log_std),
            "continuous_head_weight_std": CONTINUOUS_HEAD_WEIGHT_STD,
            "binary_action_labels": list(self.binary_action_labels),
            "binary_probability_prior": list(self.binary_probability_prior),
            "binary_logit_bias": list(self.binary_logit_bias),
            "binary_head_weight_std": BINARY_HEAD_WEIGHT_STD,
            "deterministic_zero_observation_action": [
                *self.continuous_action_prior,
                *(1.0 if probability > 0.5 else 0.0
                  for probability in self.binary_probability_prior),
            ],
            "checkpoint_restore_overrides_initialization": True,
        }


def default_actor_initialization(
    n_continuous: int,
    n_binary: int,
) -> ActorInitialization:
    """Preserve the pre-v4 zero-mean / p=.5 generic policy defaults."""
    return ActorInitialization(
        scope="generic_default",
        continuous_action_labels=tuple(
            f"continuous_action_{index}" for index in range(n_continuous)),
        continuous_action_prior=(0.0,) * n_continuous,
        binary_action_labels=tuple(
            f"binary_action_{index}" for index in range(n_binary)),
        binary_probability_prior=(0.5,) * n_binary,
    )
