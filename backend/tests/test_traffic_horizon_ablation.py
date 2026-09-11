from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.checkpoints import (CheckpointRegistry,  # noqa: E402
                             IncompatibleCheckpointError)
from app.envs.driving import RewardConfig  # noqa: E402
from app.ppo.agent import PPOAgent  # noqa: E402
from app.scenarios import get_spec, list_specs  # noqa: E402
from app.settings import Settings  # noqa: E402
import app.trainer as trainer_module  # noqa: E402


class TrafficHorizonAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.traffic = get_spec("traffic-rush")

    def test_traffic_retains_its_ninety_second_horizon(self) -> None:
        fixed = self.traffic.make_env(False)
        training = self.traffic.make_training_env()

        self.assertEqual((fixed.max_steps, training.max_steps), (2250, 2250))
        self.assertEqual(
            (self.traffic.horizon_steps, self.traffic.horizon_seconds),
            (2250, 90.0),
        )
        self.assertIn("90-second horizon", self.traffic.termination_conditions)

    def test_runtime_timeout_uses_the_instance_horizon(self) -> None:
        env = self.traffic.make_env(False)
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

    def test_reference_third_overtake_fits_only_the_extended_horizon(self) -> None:
        env = self.traffic.make_env(False)
        fastest = env.features.bots[-1]
        reference_speed = env.track.total_length / env._rolling_lap_seconds
        catch_seconds = (
            fastest.start_frac * env.track.total_length
            / (reference_speed - fastest.speed)
        )

        self.assertGreater(catch_seconds, 60.0)
        self.assertLess(catch_seconds, self.traffic.horizon_seconds)

    def test_ablation_preserves_the_traffic_task_and_curriculum(self) -> None:
        env = self.traffic.make_env(False)

        self.assertEqual(
            env.reward_cfg,
            RewardConfig(
                drift_corner=0.0,
                overtake=8.0,
                contact=-40.0,
                stall=-40.0,
                wrong_way=-40.0,
                timeout=-40.0,
                terminalize_failure_time=True,
                retain_terminal_course_potential=True,
            ),
        )
        self.assertEqual(
            tuple((bot.start_frac, bot.speed, bot.lat_frac)
                  for bot in env.features.bots),
            ((0.25, 18.0, -0.4), (0.50, 24.0, 0.0),
             (0.75, 30.0, 0.4)),
        )
        self.assertIn(
            "episodes 1-200 sample 75% physical checkpoint-11 near-pass",
            self.traffic.training_start_distribution,
        )
        self.assertIn(
            "nested bot3 control advances through 18, 24, and canonical "
            "30 m/s",
            self.traffic.training_start_distribution,
        )
        self.assertEqual(
            self.traffic.success,
            "Overtake all three traffic cars in one episode.",
        )

    def test_schema_eighteen_refuses_a_schema_eight_traffic_checkpoint(self) -> None:
        env = self.traffic.make_env(False)
        agent = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = CheckpointRegistry(
                root, self.traffic.id,
                schema_version=self.traffic.checkpoint_schema)
            old = CheckpointRegistry(root, self.traffic.id, schema_version=8)
            old.save(25, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": 1.0, "trajectory": [],
            })

            self.assertEqual(self.traffic.checkpoint_schema, 18)
            self.assertEqual(current.list(), [])
            with self.assertRaises(IncompatibleCheckpointError):
                current.load_into(
                    25,
                    agent,
                    expected_engine="current-engine",
                    expected_evaluation_suite="current-suite",
                )

    def test_protocol_v18_records_the_exact_task_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"traffic-rush"}')
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
                "metric": 0.0,
                "metric_std": 0.0,
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

        self.assertEqual(protocol["version"], 19)
        self.assertEqual(protocol["task_horizon_steps"], 2250)
        self.assertEqual(protocol["task_horizon_seconds"], 90.0)


if __name__ == "__main__":
    unittest.main()
