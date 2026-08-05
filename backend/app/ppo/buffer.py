"""Fixed-size rollout buffer with GAE."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr >= self.capacity

    def add(self, obs, action, log_prob, reward, done, value) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, last_done: bool,
                    gamma: float = 0.995, gae_lambda: float = 0.95) -> None:
        if self.ptr == 0:
            return
        adv = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_nonterminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                # dones[t] belongs to the transition from state t to t + 1.
                # Looking at dones[t + 1] leaks value/advantage information
                # across episode boundaries and cuts the preceding transition
                # one step too early.
                next_nonterminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + gamma * gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns[:self.ptr] = (self.advantages[:self.ptr]
                                   + self.values[:self.ptr])

    def minibatches(self, batch_size: int, device: torch.device) -> Iterator[dict[str, torch.Tensor]]:
        indices = np.random.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                "obs": torch.as_tensor(self.obs[idx], device=device),
                "actions": torch.as_tensor(self.actions[idx], device=device),
                "log_probs": torch.as_tensor(self.log_probs[idx], device=device),
                "advantages": torch.as_tensor(self.advantages[idx], device=device),
                "returns": torch.as_tensor(self.returns[idx], device=device),
                "values": torch.as_tensor(self.values[idx], device=device),
            }

    def reset(self) -> None:
        self.ptr = 0
