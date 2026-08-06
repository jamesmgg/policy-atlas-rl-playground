"""Scientific contract tests for scenario-specific actor initialization."""
from __future__ import annotations

import copy
import inspect
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.ppo.agent import PPOAgent
from app.ppo.initialization import default_actor_initialization
from app.scenarios import get_spec, list_specs
from app.settings import Settings
import app.trainer as trainer_module


EXPECTED_DRIVING_PROTOCOL = {
    "scope": "driving_only",
    "applies_on": "fresh_seeded_policy_construction",
    "continuous_action_labels": ["throttle_brake", "steering"],
    "continuous_action_prior": [0.25, 0.0],
    "continuous_latent_bias": [math.atanh(0.25), 0.0],
    "continuous_log_std": [-0.5, -0.5],
    "continuous_head_weight_std": 0.01,
    "binary_action_labels": ["drift"],
    "binary_probability_prior": [0.05],
    "binary_logit_bias": [math.log(0.05 / 0.95)],
    "binary_head_weight_std": 0.01,
    "deterministic_zero_observation_action": [0.25, 0.0, 0.0],
    "checkpoint_restore_overrides_initialization": True,
}

EXPECTED_LANDER_PROTOCOL = {
    "scope": "lunar_lander_only",
    "applies_on": "fresh_seeded_policy_construction",
    "continuous_action_labels": [
        "main_engine_throttle",
        "side_thruster_command",
    ],
    "continuous_action_prior": [0.0, 0.0],
    "continuous_latent_bias": [0.0, 0.0],
    "continuous_log_std": [-1.2, -1.2],
    "continuous_head_weight_std": 0.01,
    "binary_action_labels": [],
    "binary_probability_prior": [],
    "binary_logit_bias": [],
    "binary_head_weight_std": 0.01,
    "deterministic_zero_observation_action": [0.0, 0.0],
    "checkpoint_restore_overrides_initialization": True,
}


class TestDrivingActorInitialization(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_spec("rally-ridge")
        self.initialization = getattr(self.spec, "actor_initialization", None)

    def test_driving_scenario_declares_semantic_bounded_action_priors(self) -> None:
        self.assertIsNotNone(
            self.initialization,
            "driving scenarios must explicitly declare their actor initialization",
        )
        self.assertEqual(
            self.initialization.continuous_action_labels,
            ("throttle_brake", "steering"),
        )
        self.assertEqual(
            self.initialization.continuous_action_prior,
            (0.25, 0.0),
        )
        self.assertEqual(self.initialization.binary_action_labels, ("drift",))
        self.assertEqual(self.initialization.binary_probability_prior, (0.05,))

    def test_actor_initialization_rejects_semantic_length_mismatches(self) -> None:
        self.assertIsNotNone(self.initialization)
        with self.assertRaisesRegex(ValueError, "continuous action labels"):
            replace(
                self.initialization,
                continuous_action_labels=("throttle_brake",),
            )
        with self.assertRaisesRegex(ValueError, "binary action labels"):
            replace(self.initialization, binary_action_labels=())
        with self.assertRaisesRegex(ValueError, "continuous log std"):
            replace(self.initialization, continuous_log_std=(-0.5,))

    def test_fresh_driving_agent_uses_forward_zero_steer_drift_off_prior(self) -> None:
        self.assertIn(
            "actor_initialization",
            inspect.signature(PPOAgent).parameters,
            "PPOAgent must accept a scenario actor initialization contract",
        )
        env = self.spec.make_env(False)
        agent = PPOAgent(
            env.obs_dim,
            env.n_continuous,
            env.n_binary,
            torch.device("cpu"),
            actor_initialization=self.initialization,
        )

        torch.testing.assert_close(
            agent.network.mu.bias,
            torch.tensor([math.atanh(0.25), 0.0]),
        )
        torch.testing.assert_close(
            agent.network.log_std,
            torch.tensor([-0.5, -0.5]),
        )
        self.assertIsNotNone(agent.network.drift_logit)
        torch.testing.assert_close(
            agent.network.drift_logit.bias,
            torch.tensor([math.log(0.05 / 0.95)]),
        )
        action, _, _ = agent.select_action(
            np.zeros(env.obs_dim, dtype=np.float32), deterministic=True)
        np.testing.assert_allclose(action, [0.25, 0.0, 0.0], atol=1e-7)

    def test_non_driving_agent_keeps_zero_continuous_default(self) -> None:
        spec = get_spec("mountain-car")
        self.assertIsNone(getattr(spec, "actor_initialization", None))
        env = spec.make_env(False)
        agent = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        torch.testing.assert_close(
            agent.network.mu.bias,
            torch.zeros(env.n_continuous),
        )
        action, _, _ = agent.select_action(
            np.zeros(env.obs_dim, dtype=np.float32), deterministic=True)
        np.testing.assert_allclose(action, np.zeros(env.n_continuous), atol=1e-7)

    def test_checkpoint_restore_fully_overrides_fresh_prior_tensors(self) -> None:
        self.assertIn("actor_initialization", inspect.signature(PPOAgent).parameters)
        env = self.spec.make_env(False)
        restored = PPOAgent(
            env.obs_dim,
            env.n_continuous,
            env.n_binary,
            torch.device("cpu"),
            actor_initialization=self.initialization,
        )
        source = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        with torch.no_grad():
            source.network.mu.bias.copy_(torch.tensor([-0.4, 0.3]))
            source.network.drift_logit.bias.fill_(1.5)
        expected = copy.deepcopy(source.state_dict())

        restored.load_state_dict(expected)

        for key, tensor in restored.network.state_dict().items():
            torch.testing.assert_close(tensor, expected["network"][key])

    def test_policy_diagnostics_report_state_conditioned_action_tendencies(self) -> None:
        diagnostics = getattr(PPOAgent, "policy_action_diagnostics", None)
        self.assertTrue(
            callable(diagnostics),
            "PPOAgent must expose state-conditioned policy action diagnostics",
        )
        env = self.spec.make_env(False)
        agent = PPOAgent(
            env.obs_dim,
            env.n_continuous,
            env.n_binary,
            torch.device("cpu"),
            actor_initialization=self.initialization,
        )

        stats = agent.policy_action_diagnostics(
            np.zeros((8, env.obs_dim), dtype=np.float32))

        self.assertAlmostEqual(stats["continuous_action_mean_0"], 0.25, places=6)
        self.assertAlmostEqual(stats["continuous_action_mean_1"], 0.0, places=6)
        self.assertAlmostEqual(stats["binary_probability_mean_0"], 0.05, places=6)
        self.assertEqual(stats["binary_deterministic_on_fraction_0"], 0.0)

    def test_v9_checkpoint_protocol_records_exact_initialization_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = trainer_module.Trainer(Settings(
                port=8901,
                checkpoint_dir=Path(tmp),
                checkpoint_every_n=25,
                max_episodes=10,
                use_gpu=False,
                seed=42,
                eval_episodes=1,
            ))
            trainer._run_eval = lambda: {
                "reward": 0.0,
                "reward_std": 0.0,
                "metric": None,
                "metric_std": None,
                "failure_progress": 0.0,
                "episodes": 1,
                "success_rate": 0.0,
                "success_ci_low": 0.0,
                "success_ci_high": 1.0,
                "evaluation_suite": "test-suite",
                "seed": 42,
                "trajectory": [],
            }
            trainer._save_checkpoint()
            protocol = trainer.registry.list()[0]["protocol"]

        self.assertEqual(protocol["version"], 13)
        self.assertEqual(protocol["actor_initialization"], EXPECTED_DRIVING_PROTOCOL)


class TestLanderExplorationInitialization(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_spec("lunar-lander")
        self.initialization = getattr(self.spec, "actor_initialization", None)

    def test_lander_declares_lower_per_action_latent_log_std(self) -> None:
        self.assertIsNotNone(self.initialization)
        self.assertEqual(self.initialization.scope, "lunar_lander_only")
        self.assertEqual(
            self.initialization.continuous_action_labels,
            ("main_engine_throttle", "side_thruster_command"),
        )
        self.assertEqual(
            self.initialization.continuous_action_prior,
            (0.0, 0.0),
        )
        self.assertEqual(
            self.initialization.continuous_log_std,
            (-1.2, -1.2),
        )

        env = self.spec.make_env(False)
        agent = PPOAgent(
            env.obs_dim,
            env.n_continuous,
            env.n_binary,
            torch.device("cpu"),
            actor_initialization=self.initialization,
        )
        torch.testing.assert_close(agent.network.mu.bias, torch.zeros(2))
        torch.testing.assert_close(
            agent.network.log_std,
            torch.tensor([-1.2, -1.2]),
        )

    def test_every_other_scenario_keeps_initial_log_std_minus_point_five(self) -> None:
        for spec in list_specs():
            if spec.id == self.spec.id:
                continue
            env = spec.make_env(False)
            initialization = (
                spec.actor_initialization
                if spec.actor_initialization is not None
                else default_actor_initialization(
                    env.n_continuous, env.n_binary)
            )
            self.assertEqual(
                initialization.continuous_log_std,
                (-0.5,) * env.n_continuous,
                f"{spec.id} changed its initial exploration",
            )

    def test_checkpoint_restore_overrides_lander_exploration_initialization(self) -> None:
        env = self.spec.make_env(False)
        restored = PPOAgent(
            env.obs_dim,
            env.n_continuous,
            env.n_binary,
            torch.device("cpu"),
            actor_initialization=self.initialization,
        )
        source = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        with torch.no_grad():
            source.network.mu.bias.copy_(torch.tensor([-0.4, 0.3]))
            source.network.log_std.copy_(torch.tensor([-0.25, -0.75]))
        expected = copy.deepcopy(source.state_dict())

        restored.load_state_dict(expected)

        torch.testing.assert_close(
            restored.network.mu.bias,
            torch.tensor([-0.4, 0.3]),
        )
        torch.testing.assert_close(
            restored.network.log_std,
            torch.tensor([-0.25, -0.75]),
        )

    def test_v9_metadata_records_exact_lander_exploration_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario": "lunar-lander"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901,
                checkpoint_dir=root,
                checkpoint_every_n=25,
                max_episodes=10,
                use_gpu=False,
                seed=42,
                eval_episodes=1,
            ))
            trainer._run_eval = lambda: {
                "reward": 0.0,
                "reward_std": 0.0,
                "metric": None,
                "metric_std": None,
                "failure_progress": 0.0,
                "episodes": 1,
                "success_rate": 0.0,
                "success_ci_low": 0.0,
                "success_ci_high": 1.0,
                "evaluation_suite": "test-suite",
                "seed": 42,
                "trajectory": [],
            }
            trainer._save_checkpoint()
            protocol = trainer.registry.list()[0]["protocol"]

        self.assertEqual(protocol["version"], 13)
        self.assertEqual(protocol["actor_initialization"], EXPECTED_LANDER_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
