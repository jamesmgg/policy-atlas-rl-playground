"""Sparse summit reward must not trap a fresh policy at zero motor effort."""
import unittest
from unittest.mock import patch

import torch

from app.envs import mountain_car
from app.ppo.agent import PPOAgent
from app.scenarios import get_spec


class MountainCarWarmStartTests(unittest.TestCase):
    def test_neural_warm_start_reaches_summit_without_teacher_at_inference(self):
        torch.set_num_threads(1)
        spec = get_spec("mountain-car")
        self.assertIsNotNone(spec.actor_warm_start)
        for training_seed in (42, 123):
            torch.manual_seed(training_seed)
            env = spec.make_env(False)
            agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary,
                             torch.device("cpu"),
                             actor_initialization=spec.actor_initialization)
            diagnostics = spec.actor_warm_start.apply(agent)
            self.assertGreater(diagnostics["samples"], 1000)
            self.assertFalse(spec.actor_warm_start.protocol()["expert"]["used_at_inference"])
            with patch.object(mountain_car, "momentum_reference_action",
                              side_effect=AssertionError("teacher called at inference")):
                for evaluation_seed in range(4_250_000, 4_250_020):
                    env = spec.make_env(True)
                    env.rng.seed(evaluation_seed)
                    observation = env.reset()
                    for _ in range(env.max_steps):
                        observation, _, done, _ = env.step(agent.predict(observation))
                        if done:
                            break
                    self.assertTrue(env.episode_summary()["success"],
                                    (training_seed, evaluation_seed, env.episode_summary()))


if __name__ == "__main__":
    unittest.main()
