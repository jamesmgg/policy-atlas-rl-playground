"""PPO agent: clipped surrogate objective, clipped value loss, entropy bonus."""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn

from .buffer import RolloutBuffer
from .network import ActorCritic

LR = 3e-4
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 256
TARGET_KL = 0.03


class PPOAgent:
    def __init__(self, obs_dim: int, n_continuous: int, n_binary: int,
                 device: torch.device):
        self.device = device
        self.obs_dim = obs_dim
        self.n_continuous = n_continuous
        self.n_binary = n_binary
        self.network = ActorCritic(obs_dim, n_continuous, n_binary).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR, eps=1e-5)

    @property
    def act_dim(self) -> int:
        return self.n_continuous + self.n_binary

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

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        pg_losses, v_losses, entropies, kls, clip_fracs = [], [], [], [], []
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

                adv = batch["advantages"]
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS),
                ).mean()

                v_clipped = batch["values"] + torch.clamp(
                    new_value - batch["values"], -CLIP_EPS, CLIP_EPS)
                v_loss = 0.5 * torch.max(
                    (new_value - batch["returns"]) ** 2,
                    (v_clipped - batch["returns"]) ** 2,
                ).mean()

                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.mean().item())
                if approx_kl > TARGET_KL:
                    stop = True
                    break
            if stop:
                break

        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_frac": float(np.mean(clip_fracs)),
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
