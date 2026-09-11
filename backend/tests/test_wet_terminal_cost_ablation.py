"""Contract tests for the Wet Apex terminal-failure cost ablation."""
from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.checkpoints import (CheckpointRegistry,
                             IncompatibleCheckpointError)  # noqa: E402
from app.envs.driving import RewardConfig  # noqa: E402
from app.ppo.agent import PPOAgent  # noqa: E402
from app.scenarios import get_spec  # noqa: E402


class WetTerminalCostAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wet = get_spec("apex-gp-wet")

    def test_wet_makes_all_unsafe_terminal_failures_equally_costly(self) -> None:
        expected = replace(RewardConfig(), stall=-40.0, wrong_way=-40.0)

        self.assertEqual(self.wet.make_env(False).reward_cfg, expected)
        self.assertEqual(self.wet.make_training_env().reward_cfg, expected)
        self.assertEqual(
            {
                self.wet.make_env(False).reward_cfg.collision,
                self.wet.make_env(False).reward_cfg.stall,
                self.wet.make_env(False).reward_cfg.wrong_way,
            },
            {-40.0},
        )

    def test_ablation_preserves_horizon_curriculum_and_other_rewards(self) -> None:
        training = self.wet.make_training_env()

        self.assertEqual((training.max_steps, self.wet.horizon_seconds),
                         (2250, 90.0))
        self.assertEqual(training.start_line_probability, 0.75)
        self.assertEqual(training.rolling_checkpoint_indices, (11,))
        self.assertEqual(get_spec("apex-gp").make_env(False).reward_cfg,
                         RewardConfig())
        self.assertEqual(
            get_spec("traffic-rush").make_env(False).reward_cfg.overtake,
            8.0,
        )

    def test_schema_nine_refuses_schema_eight_wet_checkpoints(self) -> None:
        env = self.wet.make_env(False)
        agent = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = CheckpointRegistry(
                root, self.wet.id,
                schema_version=self.wet.checkpoint_schema)
            old = CheckpointRegistry(root, self.wet.id, schema_version=8)
            old.save(25, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": None, "trajectory": [],
            })

            self.assertEqual(self.wet.checkpoint_schema, 10)
            self.assertEqual(current.list(), [])
            with self.assertRaises(IncompatibleCheckpointError):
                current.load_into(
                    25,
                    agent,
                    expected_engine="current-engine",
                    expected_evaluation_suite="current-suite",
                )


if __name__ == "__main__":
    unittest.main()
