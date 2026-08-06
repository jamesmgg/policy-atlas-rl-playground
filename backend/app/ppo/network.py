"""Actor-critic network: shared MLP torso, Gaussian head for continuous
actions, optional Bernoulli head for binary actions, and a value head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal

from .initialization import (
    ActorInitialization,
    BINARY_HEAD_WEIGHT_STD,
    CONTINUOUS_HEAD_WEIGHT_STD,
    default_actor_initialization,
)


def layer_init(layer: nn.Linear, std: float = 2.0 ** 0.5, bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_continuous: int, n_binary: int,
                 hidden: tuple[int, ...] = (256, 256, 128),
                 actor_initialization: ActorInitialization | None = None):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_binary = n_binary
        self.actor_initialization = (
            actor_initialization
            if actor_initialization is not None
            else default_actor_initialization(n_continuous, n_binary)
        )
        self.actor_initialization.validate_dimensions(n_continuous, n_binary)
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [layer_init(nn.Linear(last, h)), nn.Tanh()]
            last = h
        self.torso = nn.Sequential(*layers)
        self.mu = layer_init(
            nn.Linear(last, n_continuous), std=CONTINUOUS_HEAD_WEIGHT_STD)
        with torch.no_grad():
            self.mu.bias.copy_(torch.tensor(
                self.actor_initialization.continuous_latent_bias,
                dtype=self.mu.bias.dtype,
            ))
        self.log_std = nn.Parameter(torch.tensor(
            self.actor_initialization.continuous_log_std,
            dtype=self.mu.weight.dtype,
        ))
        # Generic binary head; name kept as `drift_logit` so pre-multi-scenario
        # checkpoints (where it really was the drift button) still load.
        self.drift_logit = (layer_init(
            nn.Linear(last, n_binary), std=BINARY_HEAD_WEIGHT_STD)
                            if n_binary > 0 else None)
        if self.drift_logit is not None:
            with torch.no_grad():
                self.drift_logit.bias.copy_(torch.tensor(
                    self.actor_initialization.binary_logit_bias,
                    dtype=self.drift_logit.bias.dtype,
                ))
        self.value = layer_init(nn.Linear(last, 1), std=1.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(self.torso(obs)).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action[..., n_cont+n_bin], log_prob, entropy, value).

        Continuous actions use a tanh-squashed Normal, so the action sent to an
        environment and the action whose density PPO optimizes are identical.
        Binary actions are Bernoulli.
        """
        h = self.torso(obs)
        mu = self.mu(h)
        std = torch.exp(self.log_std.clamp(-3.0, 0.7))
        dist_c = Normal(mu, std)

        if action is not None:
            act_c = action[..., :self.n_continuous]
        elif deterministic:
            act_c = torch.tanh(mu)
        else:
            act_c = torch.tanh(dist_c.sample())
        # Compute both rollout-time and update-time likelihoods from the
        # *executed* float32 action. Very large latents can round tanh to exact
        # +/-1; using the original latent only on the sampling path would then
        # make PPO's stored and recomputed likelihoods disagree catastrophically.
        likelihood_action = act_c.clamp(-1 + 1e-6, 1 - 1e-6)
        raw_c = torch.atanh(likelihood_action)
        # Stable log|d tanh(x) / dx| correction (SAC appendix C).
        log_det = 2.0 * (math_log_two() - raw_c - F.softplus(-2.0 * raw_c))
        log_prob = (dist_c.log_prob(raw_c) - log_det).sum(-1)
        entropy_raw = dist_c.rsample()
        entropy_action = torch.tanh(entropy_raw)
        entropy_log_det = 2.0 * (
            math_log_two() - entropy_raw - F.softplus(-2.0 * entropy_raw))
        entropy = -(dist_c.log_prob(entropy_raw) - entropy_log_det).sum(-1)
        parts = [act_c]

        if self.drift_logit is not None:
            dist_d = Bernoulli(logits=self.drift_logit(h))
            if action is not None:
                act_d = action[..., self.n_continuous:]
            elif deterministic:
                act_d = (dist_d.logits > 0).float()
            else:
                act_d = dist_d.sample()
            log_prob = log_prob + dist_d.log_prob(act_d).sum(-1)
            entropy = entropy + dist_d.entropy().sum(-1)
            parts.append(act_d)

        value = self.value(h).squeeze(-1)
        return torch.cat(parts, dim=-1), log_prob, entropy, value


def math_log_two() -> float:
    # A tiny helper keeps the transform expression readable and avoids creating
    # a device tensor for a scalar constant on every policy call.
    return 0.6931471805599453
