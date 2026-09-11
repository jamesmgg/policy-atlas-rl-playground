"""Driving assistance must produce standalone neural policies on real courses."""
import unittest
from unittest.mock import patch
from dataclasses import replace

import torch

from app.envs import driving
from app.ppo.agent import PPOAgent
from app.scenarios import get_spec


class GuidedDrivingLearningTests(unittest.TestCase):
    def test_guidance_changes_observations_without_changing_task_dynamics(self):
        spec = get_spec("apex-gp-wet")
        guided = spec.make_env(True)
        original = driving.DrivingEnv(
            guided.track, params=guided.params, reward_cfg=guided.reward_cfg,
            features=replace(guided.features, guidance_observations=False),
            jitter=True, max_steps=guided.max_steps)
        for env in (guided, original):
            env.rng.seed(4_760_000)
            env.reset()
        self.assertEqual(guided.obs_dim, original.obs_dim + 3)
        for _ in range(guided.max_steps):
            action = driving.guided_reference_action(guided)
            _, guided_reward, guided_done, _ = guided.step(action)
            _, original_reward, original_done, _ = original.step(action)
            self.assertEqual(guided.car, original.car)
            self.assertEqual(guided_reward, original_reward)
            self.assertEqual(guided_done, original_done)
            if guided_done:
                break
        self.assertEqual(guided.episode_summary(), original.episode_summary())

    def test_guided_policy_completes_circuit_weather_and_style_without_teacher(self):
        torch.set_num_threads(1)
        for task in ("apex-gp", "glacier", "drift-trial"):
            spec = get_spec(task)
            with self.subTest(task=task):
                self.assertIsNotNone(spec.actor_warm_start)
                env = spec.make_env(False)
                self.assertEqual(len(spec.observation_dimensions), env.obs_dim)
                torch.manual_seed(123)
                agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary,
                                 torch.device("cpu"), actor_initialization=spec.actor_initialization)
                spec.actor_warm_start.apply(agent)
                with patch.object(driving, "guided_reference_action",
                                  side_effect=AssertionError("teacher acted at inference")):
                    successes = 0
                    for seed in range(4_750_000, 4_750_010):
                        env = spec.make_env(True)
                        env.rng.seed(seed)
                        observation = env.reset()
                        for _ in range(env.max_steps):
                            observation, _, done, _ = env.step(agent.predict(observation))
                            if done:
                                break
                        successes += int(env.episode_summary()["success"])
                    self.assertGreaterEqual(successes, 9, task)


if __name__ == "__main__":
    unittest.main()
