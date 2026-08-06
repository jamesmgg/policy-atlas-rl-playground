from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.envs.driving import RewardConfig, STYLE_TARGET  # noqa: E402
from app.scenarios.registry import list_specs  # noqa: E402


class DriftProgressAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = {spec.id: spec for spec in list_specs()}
        cls.spec = cls.specs["drift-trial"]

    def test_drift_progress_shaping_pays_for_ten_metres_per_second(self) -> None:
        env = self.spec.make_env(False)
        distance_per_step = 10.0 * env.dt
        progress_and_time_reward = (
            env.reward_cfg.progress * distance_per_step
            + env.reward_cfg.time
        )

        self.assertEqual(env.reward_cfg.progress, 0.05)
        self.assertGreaterEqual(progress_and_time_reward, 0.0)

    def test_drift_ablation_changes_only_the_progress_reward_term(self) -> None:
        env = self.spec.make_env(False)

        self.assertEqual(
            env.reward_cfg,
            RewardConfig(
                progress=0.05,
                checkpoint=3.0,
                lap=5.0,
                time=-0.02,
                drift_corner=0.0,
                collision=-40.0,
                stall=-15.0,
                wrong_way=-20.0,
                style_coef=0.12,
                overtake=0.0,
                contact=0.0,
                fuel_empty=0.0,
            ),
        )

    def test_drift_reward_revision_uses_a_fresh_checkpoint_schema(self) -> None:
        self.assertEqual(self.spec.checkpoint_schema, 8)
        self.assertEqual(self.specs["rally-ridge"].checkpoint_schema, 7)
        self.assertEqual(self.specs["traffic-rush"].checkpoint_schema, 18)

    def test_other_driving_reward_contracts_are_explicit(self) -> None:
        expected = {
            "apex-gp": RewardConfig(),
            "rally-ridge": RewardConfig(drift_corner=0.008),
            "eco-gp": RewardConfig(fuel_empty=-5.0),
            "traffic-rush": RewardConfig(
                drift_corner=0.0,
                overtake=8.0,
                contact=-40.0,
                stall=-40.0,
                wrong_way=-40.0,
                timeout=-40.0,
                terminalize_failure_time=True,
                retain_terminal_course_potential=True,
            ),
        }
        for scenario_id, reward in expected.items():
            with self.subTest(scenario=scenario_id):
                self.assertEqual(
                    self.specs[scenario_id].make_env(False).reward_cfg,
                    reward,
                )

    def test_drift_task_and_training_contracts_are_unchanged(self) -> None:
        fixed = self.spec.make_env(False)
        training = self.spec.make_training_env()

        self.assertEqual(fixed.obs_dim, 26)
        self.assertEqual((fixed.n_continuous, fixed.n_binary), (2, 1))
        self.assertEqual((self.spec.horizon_steps, self.spec.horizon_seconds),
                         (1500, 60.0))
        self.assertEqual(fixed.features.metric, "style")
        self.assertEqual(STYLE_TARGET, 10.0)
        self.assertEqual(
            self.spec.success,
            "Score at least 10 style points before the horizon.",
        )
        self.assertFalse(fixed.random_start)
        self.assertTrue(training.random_start)
        self.assertEqual(training.start_line_probability, 0.75)
        self.assertEqual(
            self.spec.actor_initialization.binary_probability_prior,
            (0.05,),
        )
        self.assertEqual(
            self.spec.training_start_distribution,
            "75% canonical start; 25% uniform checkpoints 1..N-1 as "
            "rolling states at 70-90% of the curvature/grip backward-braking "
            "envelope; clock integrates an 80% envelope with a 1-second "
            "reserve; assumed proportional style prefix; no reset reward",
        )


if __name__ == "__main__":
    unittest.main()
