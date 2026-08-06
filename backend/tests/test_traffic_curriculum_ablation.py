from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.checkpoints import (CheckpointRegistry,  # noqa: E402
                             IncompatibleCheckpointError)
from app.envs.driving import RewardConfig  # noqa: E402
from app.ppo.agent import PPOAgent  # noqa: E402
from app.scenarios import get_spec, list_specs  # noqa: E402
from app.settings import Settings  # noqa: E402
import app.trainer as trainer_module  # noqa: E402


class TrafficStageCurriculumAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.traffic = get_spec("traffic-rush")

    def forced_rolling_start(self, checkpoint: int):
        env = self.traffic.make_training_env()
        env.start_line_probability = 0.0
        env.rolling_checkpoint_indices = (checkpoint,)
        env.rng.seed(42)
        env.reset()
        return env

    def test_training_rehearses_each_overtake_stage(self) -> None:
        env = self.traffic.make_training_env()

        self.assertEqual(env.start_line_probability, 0.75)
        self.assertEqual(env.rolling_checkpoint_indices, (4, 10, 11))

        env.start_line_probability = 0.0
        env.rng.seed(42)
        sampled = set()
        for _ in range(240):
            env.reset()
            sampled.add(env.track.checkpoints.index(env.idx))
        self.assertEqual(sampled, {4, 10, 11})

    def test_stage_starts_reconstruct_pass_masks_and_reachable_targets(self) -> None:
        first = self.forced_rolling_start(4)
        second = self.forced_rolling_start(10)
        third = self.forced_rolling_start(11)

        self.assertEqual(first._bot_passed, [False, False, False])
        self.assertLess(first._bot_gap(0), 0.02 * first.track.total_length)

        self.assertEqual(second._bot_passed, [True, False, False])
        self.assertLess(second._bot_gap(1), 0.02 * second.track.total_length)

        self.assertEqual(third._bot_passed, [True, True, False])
        remaining_seconds = (third.max_steps - third.steps) * third.dt
        third_gap = third._bot_gap(2)
        required_speed = third.features.bots[2].speed + (
            third_gap / remaining_seconds
        )
        reference_speed = third.track.total_length / third._rolling_lap_seconds
        self.assertGreater(remaining_seconds, 48.0)
        self.assertLess(required_speed, reference_speed)

    def test_canonical_evaluation_task_and_other_scenarios_are_unchanged(self) -> None:
        env = self.traffic.make_env(False)

        self.assertFalse(env.random_start)
        self.assertEqual(env.idx, 0)
        self.assertEqual(env.steps, 0)
        self.assertEqual(env._bot_passed, [False, False, False])
        self.assertEqual(env.max_steps, 2250)
        self.assertEqual(
            env.reward_cfg, RewardConfig(overtake=8.0, contact=-40.0))
        self.assertEqual(
            tuple((bot.start_frac, bot.speed, bot.lat_frac)
                  for bot in env.features.bots),
            ((0.25, 18.0, -0.4), (0.50, 24.0, 0.0),
             (0.75, 30.0, 0.4)),
        )
        self.assertEqual(
            self.traffic.success,
            "Overtake all three traffic cars in one episode.",
        )
        for spec in list_specs():
            if spec.kind == "driving" and spec.id != self.traffic.id:
                with self.subTest(scenario=spec.id):
                    self.assertNotEqual(
                        spec.make_training_env().rolling_checkpoint_indices,
                        (4, 10, 11),
                    )

    def test_schema_ten_refuses_a_schema_nine_traffic_checkpoint(self) -> None:
        env = self.traffic.make_env(False)
        agent = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = CheckpointRegistry(
                root, self.traffic.id,
                schema_version=self.traffic.checkpoint_schema)
            old = CheckpointRegistry(root, self.traffic.id, schema_version=9)
            old.save(25, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": 2.0, "trajectory": [],
            })

            self.assertEqual(self.traffic.checkpoint_schema, 10)
            self.assertEqual(current.list(), [])
            with self.assertRaises(IncompatibleCheckpointError):
                current.load_into(25, agent)

    def test_protocol_v10_discloses_the_targeted_stage_curriculum(self) -> None:
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

        self.assertEqual(protocol["version"], 10)
        self.assertEqual(protocol["task_horizon_steps"], 2250)
        self.assertEqual(protocol["task_horizon_seconds"], 90.0)
        self.assertEqual(
            protocol["training_start_distribution"],
            "75% canonical start; 25% uniform checkpoints 4,10,11 as rolling "
            "states at 70-90% of the curvature/grip backward-braking "
            "envelope; clock integrates an 80% envelope with a 1-second "
            "reserve; time-advanced traffic and pass masks; no reset reward",
        )


if __name__ == "__main__":
    unittest.main()
