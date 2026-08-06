"""Deterministic, protocol-disclosed behavior-cloning actor warm starts."""
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
    """Fixed expert dataset and deterministic supervised actor initialization."""

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

    def __post_init__(self) -> None:
        if self.dataset_seed_base < 0 or self.dataset_episodes < 1:
            raise ValueError("behavior-cloning dataset seed contract is invalid")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("behavior-cloning optimizer contract is invalid")
        if self.learning_rate <= 0.0:
            raise ValueError("behavior-cloning learning rate must be positive")

    def protocol(self) -> dict:
        return {
            "id": self.id,
            "role": "actor behavior-cloning warm start",
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
                "start_state": self.dataset_start_description,
                "observations": "pre-action environment observations",
                "targets": "bounded expert rotor actions",
            },
            "optimizer": {
                "algorithm": "Adam",
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "loss": "mean squared bounded-action error",
                "shuffle": False,
                "initialization_seed": self.initialization_seed,
                "trained_parameters": "shared torso and continuous mean head",
            },
            "ppo_fine_tuning": True,
            "checkpoint_restore_overrides_warm_start": True,
        }

    def apply(self, agent: PPOAgent) -> dict:
        """Load a cached deterministic clone, building it once per process."""
        key = (
            self.id,
            self.dataset_seed_base,
            self.dataset_episodes,
            self.initialization_seed,
            self.learning_rate,
            self.batch_size,
            self.epochs,
            agent.obs_dim,
            agent.n_continuous,
            agent.n_binary,
            repr(agent.network.actor_initialization),
        )
        cached = _WARM_START_CACHE.get(key)
        if cached is None:
            observations, actions = self.dataset_builder()
            observations = np.ascontiguousarray(observations, dtype=np.float32)
            actions = np.ascontiguousarray(actions, dtype=np.float32)
            if observations.ndim != 2 or observations.shape[1] != agent.obs_dim:
                raise ValueError("behavior-cloning observations have wrong shape")
            if actions.shape != (len(observations), agent.n_continuous):
                raise ValueError("behavior-cloning actions have wrong shape")
            if not (np.all(np.isfinite(observations))
                    and np.all(np.isfinite(actions))):
                raise ValueError("behavior-cloning dataset contains non-finite data")
            if np.any(actions < -1.0) or np.any(actions > 1.0):
                raise ValueError("behavior-cloning targets are outside action bounds")

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.initialization_seed)
                model = ActorCritic(
                    agent.obs_dim,
                    agent.n_continuous,
                    agent.n_binary,
                    actor_initialization=agent.network.actor_initialization,
                ).cpu()
                parameters = [
                    *model.torso.parameters(),
                    *model.mu.parameters(),
                ]
                optimizer = torch.optim.Adam(
                    parameters, lr=self.learning_rate, eps=1e-5)
                obs_tensor = torch.from_numpy(observations)
                action_tensor = torch.from_numpy(actions)
                for _ in range(self.epochs):
                    for start in range(0, len(observations), self.batch_size):
                        stop = min(start + self.batch_size, len(observations))
                        predicted = torch.tanh(
                            model.mu(model.torso(obs_tensor[start:stop])))
                        loss = F.mse_loss(predicted, action_tensor[start:stop])
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                with torch.no_grad():
                    predicted = torch.tanh(model.mu(model.torso(obs_tensor)))
                    final_mse = float(F.mse_loss(
                        predicted, action_tensor).item())
                state = {
                    "torso": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.torso.state_dict().items()
                    },
                    "mu": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.mu.state_dict().items()
                    },
                }
            diagnostics = {
                "samples": int(len(observations)),
                "final_mse": final_mse,
                "dataset_sha256": _dataset_digest(observations, actions),
                "initialization_seed": self.initialization_seed,
            }
            cached = state, diagnostics
            _WARM_START_CACHE[key] = cached

        state, diagnostics = cached
        agent.network.torso.load_state_dict(state["torso"])
        agent.network.mu.load_state_dict(state["mu"])
        return dict(diagnostics)
