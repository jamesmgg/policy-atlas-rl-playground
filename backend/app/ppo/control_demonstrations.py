"""Seeded reference demonstrations for precision tasks with sparse settling success.

Experts generate action targets only. Runtime control always uses the learned
actor; ordinary PPO follows this disclosed supervised initialization.
"""
import math
import numpy as np
import torch

from ..envs.orbital import OrbitalEnv
from ..envs.robot_arm import L1, L2, RobotArmEnv, kinematics
from .demonstrations import BehaviorCloningWarmStart


def make_env(key):
    if key == "orbital-docking":
        return OrbitalEnv(jitter=True)
    if key == "robot-reach":
        return RobotArmEnv(jitter=True, tracking=False)
    raise ValueError(f"unknown demonstration task {key}")


def near_goal_examples(env, seed, count=48):
    """Label corrective states around the target, including late episode clocks.

    Expert transits terminate quickly and otherwise leave the actor with no
    examples of late corrections. These are supervised training states only;
    canonical reset distributions and evaluation dynamics remain unchanged.
    """
    x, y = env.target
    cosine = float(np.clip((x*x + y*y - L1*L1 - L2*L2) / (2*L1*L2), -1, 1))
    elbow = math.acos(cosine)
    shoulder = math.atan2(y, x) - math.atan2(L2*math.sin(elbow), L1+L2*cosine)
    desired = np.asarray([shoulder, elbow])
    rng = np.random.default_rng(seed)
    observations, actions = [], []
    for _ in range(count):
        env.q = desired + rng.uniform(-.2, .2, 2)
        env.velocity = rng.uniform(-.4, .4, 2)
        env.steps = int(rng.integers(0, env.max_steps))
        distance = float(np.linalg.norm(kinematics(*env.q)[1] - env.target))
        settled = distance < .08 and np.linalg.norm(env._tip_velocity()) < .1
        env.hold_steps = int(rng.integers(0, min(env.steps, 19) + 1)) if settled else 0
        observations.append(env._obs().copy())
        actions.append(env.reference_action().copy())
    return observations, actions


def build_dataset(key, *, episodes=80, seed_base=810_000, model=None):
    observations, actions = [], []
    for index in range(episodes):
        env = make_env(key)
        env.rng.seed(seed_base + index)
        obs = env.reset()
        for _ in range(env.max_steps):
            target = env.reference_action()
            observations.append(obs.copy())
            actions.append(target.copy())
            if model is None:
                action = target
            else:
                with torch.no_grad():
                    tensor = torch.from_numpy(obs).unsqueeze(0)
                    action = torch.tanh(model.mu(model.torso(tensor)))[0].numpy()
            obs, _, done, _ = env.step(action)
            if done:
                break
        if key == "robot-reach":
            extra_obs, extra_actions = near_goal_examples(env, seed_base + index)
            observations.extend(extra_obs)
            actions.extend(extra_actions)
    observations, actions = (np.asarray(observations, np.float32),
                              np.asarray(actions, np.float32))
    if key == "robot-reach":
        order = np.random.default_rng(seed_base).permutation(len(actions))
        observations, actions = observations[order], actions[order]
    return observations, actions


def warm_start(key):
    seed_base = 810_000 if key == "orbital-docking" else 820_000
    return BehaviorCloningWarmStart(
        id=f"{key}-reference-dagger-v{2 if key == 'robot-reach' else 1}",
        expert_id=f"{key}-analytic-reference-v1",
        expert_description=("PD rendezvous with cancellation of CW terms" if key == "orbital-docking"
                            else "inverse kinematics plus damped joint PD"),
        dataset_seed_base=seed_base,
        dataset_episodes=80,
        dataset_start_description=(
            "unchanged seeded full-task starts plus 48 near-goal examples per target: "
            "joint perturbations ±0.2 rad, velocities ±0.4 rad/s, sampled valid "
            "dwell/clock states; one fixed permutation per dataset seed; "
            "demonstration seeds excluded from evaluation"
            if key == "robot-reach" else
            "unchanged seeded full-task starts; demonstrations excluded from evaluation seeds"),
        dataset_builder=lambda: build_dataset(key, seed_base=seed_base),
        epochs=120,
        continuous_action_labels=("radial", "along_track") if key == "orbital-docking" else ("shoulder", "elbow"),
        dagger_rounds=2,
        dagger_rollout_seed_base=seed_base+1_000,
        dagger_episodes_per_round=30,
        dagger_dataset_builder=lambda model, round_index: build_dataset(
            key, episodes=30, seed_base=seed_base+1_000+1_000*round_index, model=model),
        dagger_epochs=80,
    )
