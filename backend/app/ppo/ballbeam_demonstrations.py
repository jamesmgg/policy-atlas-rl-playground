"""Training-only reference demonstrations for the ball-and-beam actor."""
import numpy as np

from ..envs.ball_beam import BallBeamEnv
from .demonstrations import BehaviorCloningWarmStart

DATASET_SEED_BASE = 930_000
DATASET_EPISODES = 120


def build_dataset():
    observations, actions = [], []
    for index in range(DATASET_EPISODES):
        env = BallBeamEnv(jitter=True)
        env.rng.seed(DATASET_SEED_BASE + index)
        observation = env.reset()
        for _ in range(env.max_steps):
            action = env.reference_action()
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, _, done, _ = env.step(action)
            if done:
                break
        if not env.episode_summary()["success"]:
            raise RuntimeError(f"Ball and beam demonstration {index} did not settle")
    order = np.random.default_rng(DATASET_SEED_BASE).permutation(len(actions))
    return (np.asarray(observations, np.float32)[order],
            np.asarray(actions, np.float32)[order])


BALLBEAM_WARM_START = BehaviorCloningWarmStart(
    id="ballbeam-reference-demonstrations-v1",
    expert_id="ballbeam-cascaded-pd-v1",
    expert_description=(
        "cascaded ball-position/velocity and beam-attitude PD reference; "
        "generates supervised training targets only, absent at inference"),
    dataset_seed_base=DATASET_SEED_BASE,
    dataset_episodes=DATASET_EPISODES,
    dataset_start_description=(
        "unchanged canonical jittered starts: ball within ±0.65 m, target "
        "within ±0.3 m, level beam; one frozen sample permutation with "
        "dataset seed; disjoint from fixed tests and holdouts"),
    dataset_builder=build_dataset,
    epochs=80,
    continuous_action_labels=("beam_angular_acceleration",),
)
