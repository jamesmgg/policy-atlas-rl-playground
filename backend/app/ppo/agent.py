"""PPO agent: clipped surrogate objective, clipped value loss, entropy bonus."""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn

from .buffer import RolloutBuffer
from .initialization import ActorInitialization
from .network import ActorCritic

LR = 3e-4
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.0
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 256
TARGET_KL = 0.03


def deterministic_action(agent, observation: np.ndarray) -> np.ndarray:
    """Action-only fast path, retaining the small policy interface for baselines."""
    predict = getattr(agent, "predict", None)
    if callable(predict):
        return predict(observation)
    return agent.select_action(observation, deterministic=True)[0]


def critic_diagnostics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calibration signals in the critic's disclosed training-reward units."""
    target_variance = float(np.var(targets))
    explained_variance = (
        1.0 - float(np.var(targets - predictions)) / target_variance
        if target_variance > 1e-8 else 0.0
    )
    return {
        "explained_variance": explained_variance,
        "value_bias": float(np.mean(predictions - targets)),
    }


def scale_aware_value_loss(
    new_value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_epsilon: float = CLIP_EPS,
    target_scale: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped critic loss expressed in rollout-return RMS units.

    PPO's advantages are already normalized, but the shared critic previously
    saw raw targets ranging from fractions to thousands across experiments.
    One rollout-level scale is reused for every shuffled minibatch so a rare
    high-return sample cannot change critic weighting merely because of which
    batch it lands in.
    """
    if target_scale is None:
        scale = returns.square().mean().sqrt().detach().clamp_min(1.0)
    else:
        scale = torch.as_tensor(
            target_scale, dtype=returns.dtype, device=returns.device,
        ).detach().clamp_min(1.0)
    clip_width = clip_epsilon * scale
    value_delta = new_value - old_value
    clipped_value = old_value + torch.clamp(
        value_delta, -clip_width, clip_width)
    raw_error = (new_value - returns) / scale
    clipped_error = (clipped_value - returns) / scale
    loss = 0.5 * torch.max(raw_error.square(), clipped_error.square()).mean()
    clip_fraction = (value_delta.detach().abs() > clip_width).float().mean()
    return loss, scale, clip_fraction


class PPOAgent:
    def __init__(self, obs_dim: int, n_continuous: int, n_binary: int,
                 device: torch.device,
                 actor_initialization: ActorInitialization | None = None,
                 learning_rate: float = LR):
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("PPO learning rate must be finite and positive")
        self.device = device
        self.obs_dim = obs_dim
        self.n_continuous = n_continuous
        self.n_binary = n_binary
        self.network = ActorCritic(
            obs_dim,
            n_continuous,
            n_binary,
            actor_initialization=actor_initialization,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=float(learning_rate), eps=1e-5)

    @property
    def act_dim(self) -> int:
        return self.n_continuous + self.n_binary

    @torch.no_grad()
    def exploration_stats(self) -> dict[str, float]:
        """Report latent continuous-action spread without changing sampling."""
        std = torch.exp(self.network.log_std.clamp(-3.0, 0.7))
        stats = {
            "action_std_mean": float(std.mean().item()),
            "action_std_min": float(std.min().item()),
            "action_std_max": float(std.max().item()),
        }
        stats.update({
            f"action_std_{index}": float(value)
            for index, value in enumerate(std.cpu().tolist())
        })
        return stats

    @torch.no_grad()
    def policy_action_diagnostics(
        self,
        observations: np.ndarray,
    ) -> dict[str, float]:
        """Summarize policy tendencies on the states in one rollout."""
        obs = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        h = self.network.torso(obs)
        continuous = torch.tanh(self.network.mu(h))
        stats = {
            f"continuous_action_mean_{index}": float(value)
            for index, value in enumerate(
                continuous.mean(dim=0).detach().cpu().tolist())
        }
        if self.network.drift_logit is not None:
            logits = self.network.drift_logit(h)
            probabilities = torch.sigmoid(logits).mean(dim=0)
            deterministic_on = (logits > 0).float().mean(dim=0)
            stats.update({
                f"binary_probability_mean_{index}": float(value)
                for index, value in enumerate(
                    probabilities.detach().cpu().tolist())
            })
            stats.update({
                f"binary_deterministic_on_fraction_{index}": float(value)
                for index, value in enumerate(
                    deterministic_on.detach().cpu().tolist())
            })
        return stats

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False
                      ) -> tuple[np.ndarray, float, float]:
        t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action, log_prob, _, value = self.network.act(t, deterministic=deterministic)
        return (action.squeeze(0).cpu().numpy(),
                float(log_prob.item()), float(value.item()))

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        return float(self.network.get_value(t).item())

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action only; evaluation needs neither densities nor RNG."""
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        h = self.network.torso(t)
        parts = [torch.tanh(self.network.mu(h))]
        if self.network.drift_logit is not None:
            parts.append((self.network.drift_logit(h) > 0).to(t.dtype))
        return torch.cat(parts, dim=-1).squeeze(0).cpu().numpy()

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        if buffer.ptr == 0:
            raise ValueError("cannot update PPO from an empty rollout")
        pg_losses, v_losses, entropies, kls, clip_fracs = [], [], [], [], []
        value_scales, value_clip_fracs = [], []
        targets = buffer.returns[:buffer.ptr]
        predictions = buffer.values[:buffer.ptr]
        calibration = critic_diagnostics(targets, predictions)
        rollout_value_scale = max(
            float(np.sqrt(np.mean(np.square(targets)))), 1.0)
        for _ in range(UPDATE_EPOCHS):
            stop = False
            for batch in buffer.minibatches(MINIBATCH_SIZE, self.device):
                _, new_log_prob, entropy, new_value = self.network.act(
                    batch["obs"], action=batch["actions"])
                log_ratio = new_log_prob - batch["log_probs"]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fracs.append(((ratio - 1).abs() > CLIP_EPS).float().mean().item())
                kls.append(approx_kl)

                # This batch already exceeds the trust-region budget. Taking
                # another step before stopping moves the policy further away.
                if not np.isfinite(approx_kl):
                    raise FloatingPointError("non-finite PPO KL divergence")
                if approx_kl > TARGET_KL:
                    stop = True
                    break

                adv = batch["advantages"]
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS),
                ).mean()

                v_loss, value_scale, value_clip_fraction = scale_aware_value_loss(
                    new_value, batch["values"], batch["returns"],
                    target_scale=rollout_value_scale)

                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                value_scales.append(value_scale.item())
                value_clip_fracs.append(value_clip_fraction.item())
                entropies.append(entropy.mean().item())
            if stop:
                break

        return {
            "policy_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)),
            "clip_frac": float(np.mean(clip_fracs)),
            "value_scale": float(np.mean(value_scales)) if value_scales else rollout_value_scale,
            "value_clip_frac": float(np.mean(value_clip_fracs)) if value_clip_fracs else 0.0,
            "optimizer_steps": len(pg_losses),
            "kl_early_stopped": int(stop),
            **calibration,
            **self.exploration_stats(),
            **self.policy_action_diagnostics(buffer.obs[:buffer.ptr]),
        }

    def state_dict(self) -> dict:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        incoming = state["network"]
        current = self.network.state_dict()
        if incoming.keys() != current.keys():
            raise RuntimeError("checkpoint network keys do not match this policy")
        for key, value in incoming.items():
            if value.shape != current[key].shape:
                raise RuntimeError(
                    f"checkpoint tensor {key} has shape {tuple(value.shape)}; "
                    f"expected {tuple(current[key].shape)}")

        # PyTorch may copy shape-compatible tensors before raising on a later
        # mismatch. Keep rollback state so a rejected checkpoint can never
        # leave a hybrid in-memory policy or optimizer.
        network_before = copy.deepcopy(current)
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        try:
            self.network.load_state_dict(incoming)
            self.optimizer.load_state_dict(state["optimizer"])
        except Exception:
            self.network.load_state_dict(network_before)
            self.optimizer.load_state_dict(optimizer_before)
            raise
