"""Disclosed geometric demonstrations shorten the driving exploration phase."""
from dataclasses import replace

import numpy as np
import torch

from ..envs import driving
from .demonstrations import BehaviorCloningWarmStart


DRIVING_TASKS = (
    "apex-gp", "velocita", "grandville", "thunder-oval", "apex-gp-wet",
    "glacier", "rally-ridge", "kart-sprint", "drift-trial", "eco-gp",
)
DEMONSTRATION_EPISODES = 30
DAGGER_EPISODES = 20
STATE_STRIDE = 2


def _dataset(make_env, seed_base, *, model=None):
    observations, actions = [], []
    episodes = DAGGER_EPISODES if model is not None else DEMONSTRATION_EPISODES
    for index in range(episodes):
        env = make_env(True)
        env.rng.seed(seed_base + index)
        observation = env.reset()
        for step in range(env.max_steps):
            target = driving.guided_reference_action(env)
            if step % STATE_STRIDE == 0:
                observations.append(observation.copy())
                actions.append(target.copy())
            action = target
            if model is not None:
                with torch.no_grad():
                    hidden = model.torso(torch.as_tensor(observation).unsqueeze(0))
                    action = torch.cat((torch.tanh(model.mu(hidden)),
                                        (model.drift_logit(hidden) > 0).float()), dim=-1)[0].numpy()
            observation, _, done, _ = env.step(action)
            if done:
                break
        if model is None and not env.episode_summary()["success"]:
            raise RuntimeError(f"Driving demonstration {index} failed: {env.episode_summary()}")
    return np.asarray(observations, np.float32), np.asarray(actions, np.float32)


def guided_driving_spec(spec):
    """Attach one fixed assistance recipe; retain rewards and evaluation starts."""
    index = DRIVING_TASKS.index(spec.id)
    demonstration_seed = 4_500_000 + 10_000 * index
    dagger_seed = demonstration_seed + 5_000
    warm_start = BehaviorCloningWarmStart(
        id=f"{spec.id}-geometric-dagger-v1",
        expert_id="driving-geometric-pursuit-v1",
        expert_description=(
            "training-only pure pursuit, 18 m + 0.25 * speed lookahead, "
            "95% curvature/grip braking-envelope speed, steering heading "
            "gain 1.6 and lateral damping 0.3; Drift Trial also enables drift "
            "in corners while measured slip is below 8 degrees; teachers "
            "generate labels only and never act during PPO or evaluation"),
        dataset_seed_base=demonstration_seed,
        dataset_episodes=DEMONSTRATION_EPISODES,
        dataset_start_description="ordinary jittered canonical full-course starts, disjoint from selection and validation",
        dataset_builder=lambda: _dataset(spec.make_env, demonstration_seed),
        batch_size=2048,
        epochs=40,
        state_stride=STATE_STRIDE,
        continuous_action_labels=("throttle", "steering"),
        continuous_loss_weights=(1.0, 5.0),
        binary_action_labels=("drift",),
        binary_loss_weight=0.1,
        dagger_rounds=1,
        dagger_rollout_seed_base=dagger_seed,
        dagger_round_seed_stride=1000,
        dagger_episodes_per_round=DAGGER_EPISODES,
        dagger_state_stride=STATE_STRIDE,
        dagger_dataset_builder=lambda model, _: _dataset(spec.make_env, dagger_seed, model=model),
        dagger_epochs=40,
    )
    return replace(
        spec,
        actor_initialization=replace(spec.actor_initialization, continuous_log_std=(-2.0, -2.0)),
        actor_warm_start=warm_start,
        training_learning_rate=3e-5 if spec.id in {"apex-gp", "drift-trial"} else None,
        checkpoint_schema=spec.checkpoint_schema + 1,
    )
