"""Fixed training-only expert labels; inference always uses the neural actor."""
from functools import partial

import numpy as np
import torch

from ..envs.arcade import PaddleRallyEnv, FlappyFlightEnv, CoinCollectorEnv
from .demonstrations import BehaviorCloningWarmStart


FACTORIES = {"paddle-rally": PaddleRallyEnv, "flappy-flight": FlappyFlightEnv,
             "coin-collector": CoinCollectorEnv}
DATASET_SEED_BASE = 710_000
DATASET_EPISODES = 48
DATASET_STRIDE = 2


def build_dataset(scenario_id, episodes=DATASET_EPISODES,
                  seed_base=DATASET_SEED_BASE, model=None, round_index=0):
    observations, targets = [], []
    env = FACTORIES[scenario_id](jitter=True)
    if model is not None:
        seed_base = 720_000 + round_index*1_000
        episodes = 20
    for episode in range(episodes):
        env.rng.seed(seed_base+episode)
        observation = env.reset()
        noise = np.random.default_rng(seed_base+episode)
        for step in range(env.max_steps):
            expert = env.reference_action()
            if step % DATASET_STRIDE == 0:
                observations.append(observation.copy())
                targets.append(expert.copy())
            if model is None:
                # Perturbed rollouts include recoveries, while labels stay expert.
                action = np.clip(expert+noise.normal(0, .16, env.n_continuous), -1, 1)
            else:
                with torch.no_grad():
                    obs = torch.from_numpy(observation).unsqueeze(0)
                    action = torch.tanh(model.mu(model.torso(obs)))[0].numpy()
            observation, _, done, _ = env.step(action)
            if done:
                break
    return np.asarray(observations, np.float32), np.asarray(targets, np.float32)


def _dagger(scenario_id, model, round_index):
    return build_dataset(scenario_id, model=model, round_index=round_index)


def warm_start(scenario_id, action_labels):
    return BehaviorCloningWarmStart(
        id=f"{scenario_id}-expert-clone-v1", expert_id=f"{scenario_id}-pd-v1",
        expert_description="Task-specific proportional/PD controller; labels training states only, never used at inference.",
        dataset_seed_base=DATASET_SEED_BASE, dataset_episodes=DATASET_EPISODES,
        dataset_start_description="Full seeded games; expert rollouts perturbed by Gaussian action noise std0.16 and clipped to [-1,1].",
        dataset_builder=partial(build_dataset, scenario_id),
        initialization_seed=42, epochs=55, state_stride=DATASET_STRIDE,
        continuous_action_labels=action_labels,
        dagger_rounds=2, dagger_rollout_seed_base=720_000,
        dagger_round_seed_stride=1_000, dagger_episodes_per_round=20,
        dagger_state_stride=DATASET_STRIDE, dagger_epochs=35,
        dagger_dataset_builder=partial(_dagger, scenario_id))
