from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ppo.agent import PPOAgent  # noqa: E402
from app.scenarios import get_spec, list_specs  # noqa: E402
from app.settings import Settings  # noqa: E402
import app.trainer as trainer_module  # noqa: E402


class ThunderInitializationAblationTests(unittest.TestCase):
    def test_only_thunder_removes_the_fresh_throttle_bias(self) -> None:
        thunder = get_spec("thunder-oval").actor_initialization
        self.assertIsNotNone(thunder)
        self.assertEqual(thunder.scope, "thunder_oval_only")
        self.assertEqual(thunder.continuous_action_prior, (0.0, 0.0))
        self.assertEqual(thunder.binary_probability_prior, (0.05,))
        self.assertEqual(thunder.continuous_log_std, (-0.5, -0.5))

        for spec in list_specs():
            if spec.kind == "driving" and spec.id != "thunder-oval":
                with self.subTest(scenario=spec.id):
                    self.assertEqual(
                        spec.actor_initialization.continuous_action_prior,
                        (0.25, 0.0),
                    )

    def test_seeded_network_delta_is_isolated_to_throttle_bias(self) -> None:
        thunder = get_spec("thunder-oval")
        rally = get_spec("rally-ridge")
        env = thunder.make_env(False)
        torch.manual_seed(42)
        baseline = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"),
            actor_initialization=rally.actor_initialization,
        )
        torch.manual_seed(42)
        ablation = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"),
            actor_initialization=thunder.actor_initialization,
        )

        baseline_state = baseline.network.state_dict()
        ablation_state = ablation.network.state_dict()
        for key in baseline_state:
            if key == "mu.bias":
                self.assertAlmostEqual(
                    float(baseline_state[key][0]), math.atanh(0.25), places=6)
                self.assertEqual(float(ablation_state[key][0]), 0.0)
                torch.testing.assert_close(
                    baseline_state[key][1:], ablation_state[key][1:])
            else:
                torch.testing.assert_close(baseline_state[key], ablation_state[key])

    def test_zero_observation_action_and_restore_contract(self) -> None:
        spec = get_spec("thunder-oval")
        env = spec.make_env(False)
        restored = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"),
            actor_initialization=spec.actor_initialization,
        )
        action, _, _ = restored.select_action(
            np.zeros(env.obs_dim, dtype=np.float32), deterministic=True)
        np.testing.assert_allclose(action, [0.0, 0.0, 0.0], atol=1e-7)

        source = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        with torch.no_grad():
            source.network.mu.bias.copy_(torch.tensor([-0.4, 0.3]))
            source.network.drift_logit.bias.fill_(1.5)
        restored.load_state_dict(source.state_dict())
        torch.testing.assert_close(restored.network.mu.bias, torch.tensor([-0.4, 0.3]))
        torch.testing.assert_close(restored.network.drift_logit.bias, torch.tensor([1.5]))

    def test_protocol_v8_records_the_exact_thunder_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text('{"active_scenario":"thunder-oval"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            ))
            trainer._run_eval = lambda: {
                "reward": 0.0, "reward_std": 0.0, "metric": None,
                "metric_std": None, "failure_progress": 0.0, "episodes": 1,
                "success_rate": 0.0, "success_ci_low": 0.0,
                "success_ci_high": 1.0, "evaluation_suite": "test-suite",
                "seed": 42, "trajectory": [],
            }
            trainer._save_checkpoint()
            protocol = trainer.registry.list()[0]["protocol"]

        self.assertEqual(protocol["version"], 8)
        self.assertEqual(protocol["actor_initialization"]["scope"],
                         "thunder_oval_only")
        self.assertEqual(protocol["actor_initialization"][
            "continuous_action_prior"], [0.0, 0.0])
        self.assertEqual(protocol["actor_initialization"][
            "binary_probability_prior"], [0.05])


if __name__ == "__main__":
    unittest.main()
