"""Disclosed full-descent demonstrations; never called by policy inference."""
from __future__ import annotations

import math

import numpy as np

from .lander import (
    A_MAIN, ALPHA_SIDE, G, PAD_CX, PAD_Y, LanderEnv,
    make_frontier_evaluation_env, wrap_angle,
)

DATASET_SEED_BASE = 620_000
DATASET_EPISODES = 160


def physics_reference_action(env: LanderEnv) -> np.ndarray:
    """Track a fuel-feasible descent envelope using bounded physics feedback.

    The 60-unit/s cruise brakes at 10 units/s² toward a 5-unit/s touchdown.
    Including the envelope's own deceleration prevents velocity tracking lag
    from turning an apparently safe target into a hard landing.
    """
    altitude = max(0.0, PAD_Y - env.y)
    desired_vy = min(60.0, math.sqrt(25.0 + 20.0 * altitude))
    feed_forward = -10.0 if desired_vy < 60.0 else 0.0
    ay = float(np.clip(4.0 * (desired_vy - env.vy) + feed_forward,
                       -30.0, 45.0))
    desired_vx = float(np.clip(.2 * (PAD_CX - env.x), -15.0, 15.0))
    ax = float(np.clip(1.5 * (desired_vx - env.vx), -20.0, 20.0))
    desired_theta = float(np.clip(math.atan2(ax, G - ay), -.4, .4))
    side = float(np.clip(
        (20.0 * wrap_angle(desired_theta - env.theta) - 7.0 * env.omega)
        / ALPHA_SIDE, -1.0, 1.0))
    main = float(np.clip(
        (G - ay) / (A_MAIN * max(.3, math.cos(env.theta))), 0.0, 1.0))
    return np.array([2.0 * main - 1.0, side], dtype=np.float32)


def behavior_cloning_dataset() -> tuple[np.ndarray, np.ndarray]:
    """80 full-height and 20 per easier frontier, at disjoint fixed seeds."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for index in range(DATASET_EPISODES):
        frontier = 0 if index % 2 == 0 else 1 + (index // 2) % 4
        env = make_frontier_evaluation_env(frontier)
        env.rng.seed(DATASET_SEED_BASE + index)
        observation = env.reset()
        for _ in range(env.max_steps):
            action = physics_reference_action(env)
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, _, done, _ = env.step(action)
            if done:
                break
        if not env.landed:
            raise RuntimeError(f"Lander demonstration {index} did not land")
    # One frozen permutation avoids fitting minibatches in flight-phase order.
    order = np.random.default_rng(DATASET_SEED_BASE).permutation(len(actions))
    return (np.asarray(observations, dtype=np.float32)[order],
            np.asarray(actions, dtype=np.float32)[order])
