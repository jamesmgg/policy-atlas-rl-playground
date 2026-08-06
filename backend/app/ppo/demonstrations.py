"""Deterministic, protocol-disclosed demonstration actor warm starts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .network import ActorCritic

if TYPE_CHECKING:
    from .agent import PPOAgent


DatasetBuilder = Callable[[], tuple[np.ndarray, np.ndarray]]
DaggerDatasetBuilder = Callable[
    [ActorCritic, int], tuple[np.ndarray, np.ndarray]
]
_WARM_START_CACHE: dict[tuple, tuple[dict, dict]] = {}


def _dataset_digest(observations: np.ndarray, actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (observations, actions):
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class BehaviorCloningWarmStart:
    """Fixed expert data and optional DAgger actor initialization."""

    id: str
    expert_id: str
    expert_description: str
    dataset_seed_base: int
    dataset_episodes: int
    dataset_start_description: str
    dataset_builder: DatasetBuilder
    initialization_seed: int = 42
    learning_rate: float = 1e-3
    batch_size: int = 1024
    epochs: int = 60
    state_stride: int = 1
    continuous_action_labels: tuple[str, ...] = ()
    continuous_loss_weights: tuple[float, ...] | None = None
    binary_action_labels: tuple[str, ...] = ()
    binary_loss_weight: float = 1.0
    dagger_rounds: int = 0
    dagger_rollout_seed_base: int | None = None
    dagger_round_seed_stride: int = 1_000
    dagger_episodes_per_round: int = 0
    dagger_state_stride: int = 1
    dagger_dataset_builder: DaggerDatasetBuilder | None = None
    dagger_epochs: int | None = None

    def __post_init__(self) -> None:
        if self.dataset_seed_base < 0 or self.dataset_episodes < 1:
            raise ValueError("behavior-cloning dataset seed contract is invalid")
        if self.batch_size < 1 or self.epochs < 1 or self.state_stride < 1:
            raise ValueError("behavior-cloning optimizer contract is invalid")
        if self.learning_rate <= 0.0:
            raise ValueError("behavior-cloning learning rate must be positive")
        if self.continuous_loss_weights is not None:
            if len(self.continuous_loss_weights) != len(
                    self.continuous_action_labels):
                raise ValueError("continuous loss labels and weights differ")
            if any(weight <= 0.0 for weight in self.continuous_loss_weights):
                raise ValueError("continuous loss weights must be positive")
        if self.binary_loss_weight <= 0.0:
            raise ValueError("binary loss weight must be positive")
        has_dagger = self.dagger_rounds > 0
        if has_dagger != (self.dagger_dataset_builder is not None):
            raise ValueError("DAgger rounds require exactly one rollout builder")
        if has_dagger:
            if (self.dagger_rollout_seed_base is None
                    or self.dagger_episodes_per_round < 1
                    or self.dagger_round_seed_stride
                    < self.dagger_episodes_per_round
                    or self.dagger_state_stride < 1):
                raise ValueError("DAgger rollout seed contract is invalid")
            if self.dagger_epochs is not None and self.dagger_epochs < 1:
                raise ValueError("DAgger epochs must be positive")

    def _loss_protocol(self) -> str | dict:
        if self.continuous_loss_weights is None:
            return "mean squared bounded-action error"
        loss = {
            f"{label}_mse_weight": float(weight)
            for label, weight in zip(
                self.continuous_action_labels,
                self.continuous_loss_weights,
            )
        }
        loss.update({
            f"binary_{label}_bce_weight": float(self.binary_loss_weight)
            for label in self.binary_action_labels
        })
        return loss

    def protocol(self) -> dict:
        protocol = {
            "id": self.id,
            "role": (
                "actor DAgger warm start"
                if self.dagger_rounds else "actor behavior-cloning warm start"
            ),
            "pure_model_free_from_scratch": False,
            "expert": {
                "id": self.expert_id,
                "description": self.expert_description,
                "used_at_inference": False,
            },
            "dataset": {
                "seed_base": self.dataset_seed_base,
                "episodes": self.dataset_episodes,
                "seed_formula": "seed_base + episode_index",
                "state_stride": self.state_stride,
                "start_state": self.dataset_start_description,
                "observations": "pre-action environment observations",
                "targets": "bounded expert actions",
            },
            "optimizer": {
                "algorithm": "Adam",
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs_per_initial_fit": self.epochs,
                "epochs_per_dagger_fit": (
                    self.dagger_epochs
                    if self.dagger_epochs is not None else self.epochs
                ),
                "loss": self._loss_protocol(),
                "shuffle": False,
                "initialization_seed": self.initialization_seed,
                "trained_parameters": (
                    "shared torso, continuous mean head, and any binary head"
                ),
            },
            "ppo_fine_tuning": True,
            "checkpoint_restore_overrides_warm_start": True,
        }
        if self.dagger_rounds:
            protocol["dagger"] = {
                "rounds": self.dagger_rounds,
                "rollout_seed_base": self.dagger_rollout_seed_base,
                "round_seed_stride": self.dagger_round_seed_stride,
                "episodes_per_round": self.dagger_episodes_per_round,
                "state_stride": self.dagger_state_stride,
                "expert_role": "labels learned-policy rollout states only",
            }
        return protocol

    @staticmethod
    def _validated_dataset(
            observations: np.ndarray, actions: np.ndarray, *,
            obs_dim: int, act_dim: int) -> tuple[np.ndarray, np.ndarray]:
        observations = np.ascontiguousarray(observations, dtype=np.float32)
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != obs_dim:
            raise ValueError("behavior-cloning observations have wrong shape")
        if actions.shape != (len(observations), act_dim):
            raise ValueError("behavior-cloning actions have wrong shape")
        if not (np.all(np.isfinite(observations))
                and np.all(np.isfinite(actions))):
            raise ValueError("behavior-cloning dataset contains non-finite data")
        if np.any(actions < -1.0) or np.any(actions > 1.0):
            raise ValueError("behavior-cloning targets are outside action bounds")
        return observations, actions

    def _fit_actor(
            self, model: ActorCritic, observations: np.ndarray,
            actions: np.ndarray, *, epochs: int) -> dict:
        parameters = [*model.torso.parameters(), *model.mu.parameters()]
        if model.drift_logit is not None:
            parameters.extend(model.drift_logit.parameters())
        optimizer = torch.optim.Adam(
            parameters, lr=self.learning_rate, eps=1e-5)
        obs_tensor = torch.from_numpy(observations)
        action_tensor = torch.from_numpy(actions)
        for _ in range(epochs):
            for start in range(0, len(observations), self.batch_size):
                stop = min(start + self.batch_size, len(observations))
                hidden = model.torso(obs_tensor[start:stop])
                predicted = torch.tanh(model.mu(hidden))
                target = action_tensor[start:stop, :model.n_continuous]
                if self.continuous_loss_weights is None:
                    loss = F.mse_loss(predicted, target)
                else:
                    loss = sum(
                        weight * F.mse_loss(predicted[:, index], target[:, index])
                        for index, weight in enumerate(
                            self.continuous_loss_weights)
                    )
                if model.drift_logit is not None:
                    binary_target = action_tensor[
                        start:stop, model.n_continuous:]
                    loss = loss + self.binary_loss_weight * (
                        F.binary_cross_entropy_with_logits(
                            model.drift_logit(hidden), binary_target)
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            hidden = model.torso(obs_tensor)
            predicted = torch.tanh(model.mu(hidden))
            continuous_target = action_tensor[:, :model.n_continuous]
            diagnostics = {
                "final_mse": float(F.mse_loss(
                    predicted, continuous_target).item()),
            }
            if model.drift_logit is not None:
                binary_target = action_tensor[:, model.n_continuous:]
                diagnostics["final_binary_bce"] = float(
                    F.binary_cross_entropy_with_logits(
                        model.drift_logit(hidden), binary_target).item())
        return diagnostics

    def apply(self, agent: PPOAgent) -> dict:
        """Load a cached deterministic clone, building it once per process."""
        key = (
            self.id, self.dataset_seed_base, self.dataset_episodes,
            self.initialization_seed, self.learning_rate, self.batch_size,
            self.epochs, self.state_stride, self.continuous_action_labels,
            self.continuous_loss_weights, self.binary_action_labels,
            self.binary_loss_weight, self.dagger_rounds,
            self.dagger_rollout_seed_base, self.dagger_round_seed_stride,
            self.dagger_episodes_per_round, self.dagger_state_stride,
            self.dagger_epochs, agent.obs_dim, agent.n_continuous,
            agent.n_binary, repr(agent.network.actor_initialization),
        )
        cached = _WARM_START_CACHE.get(key)
        if cached is None:
            observations, actions = self._validated_dataset(
                *self.dataset_builder(),
                obs_dim=agent.obs_dim,
                act_dim=agent.act_dim,
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.initialization_seed)
                model = ActorCritic(
                    agent.obs_dim,
                    agent.n_continuous,
                    agent.n_binary,
                    actor_initialization=agent.network.actor_initialization,
                ).cpu()
                fit_diagnostics = self._fit_actor(
                    model, observations, actions, epochs=self.epochs)
                dagger_diagnostics = []
                if self.dagger_dataset_builder is not None:
                    for round_index in range(self.dagger_rounds):
                        new_obs, new_actions = self._validated_dataset(
                            *self.dagger_dataset_builder(model, round_index),
                            obs_dim=agent.obs_dim,
                            act_dim=agent.act_dim,
                        )
                        observations = np.ascontiguousarray(np.concatenate(
                            (observations, new_obs), axis=0))
                        actions = np.ascontiguousarray(np.concatenate(
                            (actions, new_actions), axis=0))
                        fit_diagnostics = self._fit_actor(
                            model,
                            observations,
                            actions,
                            epochs=(
                                self.dagger_epochs
                                if self.dagger_epochs is not None
                                else self.epochs
                            ),
                        )
                        dagger_diagnostics.append({
                            "round": round_index + 1,
                            "samples_added": int(len(new_obs)),
                            "total_samples": int(len(observations)),
                            **fit_diagnostics,
                        })
                state = {
                    "torso": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.torso.state_dict().items()
                    },
                    "mu": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.mu.state_dict().items()
                    },
                    "drift_logit": (
                        {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in model.drift_logit.state_dict().items()
                        }
                        if model.drift_logit is not None else None
                    ),
                }
            diagnostics = {
                "samples": int(len(observations)),
                **fit_diagnostics,
                "dataset_sha256": _dataset_digest(observations, actions),
                "initialization_seed": self.initialization_seed,
                "dagger_rounds": dagger_diagnostics,
            }
            cached = state, diagnostics
            _WARM_START_CACHE[key] = cached

        state, diagnostics = cached
        agent.network.torso.load_state_dict(state["torso"])
        agent.network.mu.load_state_dict(state["mu"])
        if agent.network.drift_logit is not None:
            if state["drift_logit"] is None:
                raise ValueError("warm start lacks the expected binary head")
            agent.network.drift_logit.load_state_dict(state["drift_logit"])
        return dict(diagnostics)
