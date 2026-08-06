from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import physics  # noqa: E402
from app.checkpoints import (CheckpointRegistry,  # noqa: E402
                             IncompatibleCheckpointError)
from app.envs import driving  # noqa: E402
from app.envs.driving import RewardConfig  # noqa: E402
from app.ppo.agent import PPOAgent  # noqa: E402
from app.scenarios import get_spec, list_specs  # noqa: E402
from app.settings import Settings  # noqa: E402
import app.trainer as trainer_module  # noqa: E402


class WetHorizonAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wet = get_spec("apex-gp-wet")

    def test_wet_receives_a_ninety_second_horizon(self) -> None:
        fixed = self.wet.make_env(False)
        training = self.wet.make_training_env()

        self.assertEqual((fixed.max_steps, training.max_steps), (2250, 2250))
        self.assertEqual(
            (self.wet.horizon_steps, self.wet.horizon_seconds),
            (2250, 90.0),
        )
        self.assertIn("90-second horizon", self.wet.termination_conditions)

    def test_only_wet_and_traffic_use_the_extended_driving_horizon(self) -> None:
        extended = {"apex-gp-wet", "traffic-rush"}

        for spec in list_specs():
            if spec.kind != "driving":
                continue
            expected = ((2250, 2250, 90.0)
                        if spec.id in extended else (1500, 1500, 60.0))
            with self.subTest(scenario=spec.id):
                self.assertEqual(
                    (spec.make_env(False).max_steps,
                     spec.horizon_steps, spec.horizon_seconds),
                    expected,
                )

    def test_wet_runtime_timeout_uses_the_extended_horizon(self) -> None:
        env = self.wet.make_env(False)
        env.steps = 1499
        env._stall_steps = 0

        _, _, done, _ = env.step(np.zeros(3, dtype=np.float32))

        self.assertFalse(done)
        self.assertEqual(env.cause, "running")

        env.steps = env.max_steps - 1
        env._stall_steps = 0
        _, _, done, info = env.step(np.zeros(3, dtype=np.float32))

        self.assertTrue(done)
        self.assertEqual(env.cause, "timeout")
        self.assertTrue(info["task_deadline"])

    def test_horizon_only_ablation_preserves_wet_learning_contract(self) -> None:
        fixed = self.wet.make_env(False)
        training = self.wet.make_training_env()

        self.assertEqual(fixed.reward_cfg, RewardConfig())
        self.assertEqual(fixed.params, physics.F1)
        self.assertEqual(training.start_line_probability, 0.75)
        self.assertEqual(training.rolling_checkpoint_indices, (11,))
        self.assertEqual(
            (driving.ROLLING_SPEED_MIN_FRACTION,
             driving.ROLLING_SPEED_MAX_FRACTION),
            (0.70, 0.90),
        )
        self.assertEqual(driving.STALL_WINDOW, 300)
        self.assertEqual(
            self.wet.training_start_distribution,
            "75% canonical start; 25% checkpoint 11 as a rolling state at "
            "70-90% of the curvature/grip backward-braking envelope; clock "
            "integrates an 80% envelope with a 1-second reserve; no reset "
            "reward",
        )
        self.assertEqual(
            self.wet.success,
            "Complete at least one timed lap without leaving the circuit.",
        )

    def test_schema_eight_refuses_a_schema_seven_wet_checkpoint(self) -> None:
        env = self.wet.make_env(False)
        agent = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = CheckpointRegistry(
                root, self.wet.id,
                schema_version=self.wet.checkpoint_schema)
            old = CheckpointRegistry(root, self.wet.id, schema_version=7)
            old.save(25, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": None, "trajectory": [],
            })

            self.assertEqual(self.wet.checkpoint_schema, 8)
            self.assertEqual(current.list(), [])
            with self.assertRaises(IncompatibleCheckpointError):
                current.load_into(25, agent)

    def test_protocol_v10_records_the_exact_wet_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"apex-gp-wet"}')
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

        self.assertEqual(protocol["version"], 10)
        self.assertEqual(protocol["task_horizon_steps"], 2250)
        self.assertEqual(protocol["task_horizon_seconds"], 90.0)


if __name__ == "__main__":
    unittest.main()
