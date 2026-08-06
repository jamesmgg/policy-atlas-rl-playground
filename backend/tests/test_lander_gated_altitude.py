"""Scientific contract tests for the Lander gated altitude curriculum."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1]))

from app import trainer as trainer_module
from app.envs import lander
from app.scenarios import get_spec
from app.settings import Settings


EXPECTED_START_STATE = (
    "k4 touchdown altitude 5-18 with pad offset <=20, |vx|<=3, vy 0-6, "
    "|tilt|<=0.08, |rate|<=0.05; k3/k2/k1 approach altitude "
    "30-100/100-250/250-500 with pad offset <=35, |vx|<=6, vy 2-18, "
    "|tilt|<=0.18, |rate|<=0.25; k0 canonical x=500, y=120, vy=rate=0, "
    "fuel=1, elapsed=0 with seeded |vx|<=15 and |tilt|<=0.15; "
    "rehearsals preserve altitude-derived elapsed time and fuel"
)

EXPECTED_CURRICULUM_PROTOCOL = {
    "id": "lander-reverse-altitude-v1",
    "frontier_order": [4, 3, 2, 1, 0],
    "start_sampling": {
        "active_frontier_probability": 0.5,
        "mastered_later_segments": "uniform remainder",
        "empty_mastered_fallback": "100% active frontier",
    },
    "gate": {
        "success_rate_threshold": 0.9,
        "comparison": ">=",
        "consecutive_confirmations": 1,
        "distinct_checkpoint_episodes": True,
    },
    "segment_evaluation": {
        "suite_version": "lander-altitude-eval-v1",
        "episodes": 20,
        "seed_base": 300_000,
        "segment_seed_stride": 1_000,
        "seed_formula": "seed_base + segment * segment_seed_stride + episode_index",
        "deterministic_policy": True,
        "start_state": EXPECTED_START_STATE,
    },
    "checkpoint_selection": (
        "segment evaluation is training-only diagnostic; fixed full-course "
        "evaluation remains the checkpoint-selection signal"
    ),
}


class ConstantAgent:
    def __init__(self, action: tuple[float, float] = (0.0, 0.0)) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def select_action(self, observation, deterministic=False):
        del observation
        if not deterministic:
            raise AssertionError("curriculum evaluation must be deterministic")
        return self.action.copy(), 0.0, 0.0


class TestLanderGatedAltitudeCurriculum(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_spec("lunar-lander")
        self.curriculum = self.spec.training_curriculum
        self.assertIsNotNone(self.curriculum)

    def pass_frontier(self, env, *, episode: int | None = None) -> dict:
        return env.record_training_curriculum_evaluation(
            0.9, self.curriculum, evaluation_episode=episode)

    def assert_rehearsal_state(self, env, frontier: int) -> None:
        altitude = lander.PAD_Y - env.y
        if frontier == 4:
            altitude_min, altitude_max = 5.0, 18.0
            x_offset, vx_max = 20.0, 3.0
            vy_min, vy_max = 0.0, 6.0
            theta_max, omega_max = 0.08, 0.05
            self.assertEqual(env._start_kind, "touchdown")
        else:
            altitude_min, altitude_max = lander.APPROACH_FRONTIER_ALTITUDES[frontier]
            x_offset, vx_max = 35.0, 6.0
            vy_min, vy_max = 2.0, 18.0
            theta_max, omega_max = 0.18, 0.25
            self.assertEqual(env._start_kind, "approach")
        self.assertGreaterEqual(altitude, altitude_min)
        self.assertLessEqual(altitude, altitude_max)
        self.assertLessEqual(abs(env.x - lander.PAD_CX), x_offset)
        self.assertLessEqual(abs(env.vx), vx_max)
        self.assertGreaterEqual(env.vy, vy_min)
        self.assertLessEqual(env.vy, vy_max)
        self.assertLessEqual(abs(env.theta), theta_max)
        self.assertLessEqual(abs(env.omega), omega_max)
        descent_fraction = (env.y - 120.0) / (lander.PAD_Y - 120.0)
        self.assertEqual(env.steps, round(180.0 * descent_fraction))
        self.assertGreaterEqual(env.fuel, 1.0 - 0.5 * descent_fraction)
        self.assertLessEqual(env.fuel, 1.0 - 0.1 * descent_fraction)

    def test_locked_frontier_starts_only_with_touchdown_rehearsals(self) -> None:
        training = self.spec.make_training_env()
        self.assertTrue(training.approach_curriculum)
        training.rng.seed(42)

        starts = []
        for _ in range(200):
            observation = training.reset()
            starts.append(training._start_frontier)
            self.assert_rehearsal_state(training, 4)
            self.assertEqual(observation.shape, (training.obs_dim,))
            self.assertTrue(np.all(np.isfinite(observation)))

        self.assertEqual(set(starts), {4})
        self.assertEqual(training.training_curriculum_state()["frontier"], 4)

    def test_each_forced_frontier_has_the_exact_altitude_contract(self) -> None:
        for frontier in lander.CURRICULUM_FRONTIER_ORDER:
            with self.subTest(frontier=frontier):
                env = lander.make_frontier_evaluation_env(frontier)
                env.rng.seed(self.curriculum.evaluation_seed(frontier, 0))
                observation = env.reset()
                self.assertEqual(env._start_frontier, frontier)
                self.assertEqual(observation.shape, (env.obs_dim,))
                if frontier == 0:
                    self.assertEqual(env._start_kind, "standard")
                    self.assertEqual((env.x, env.y, env.vy, env.omega,
                                      env.fuel, env.steps),
                                     (500.0, 120.0, 0.0, 0.0, 1.0, 0))
                    self.assertLessEqual(abs(env.vx), 15.0)
                    self.assertLessEqual(abs(env.theta), 0.15)
                else:
                    self.assert_rehearsal_state(env, frontier)

    def test_unlocked_sampling_is_half_frontier_and_uniform_over_mastered(self) -> None:
        training = self.spec.make_training_env()
        for expected in (3, 2, 1):
            transition = self.pass_frontier(training)
            self.assertTrue(transition["unlocked"])
            self.assertEqual(transition["frontier_after"], expected)

        training.rng.seed(2026)
        starts = []
        for _ in range(4_000):
            training.reset()
            starts.append(training._start_frontier)

        self.assertEqual(set(starts), {1, 2, 3, 4})
        self.assertGreaterEqual(starts.count(1), 1_800)
        self.assertLessEqual(starts.count(1), 2_200)
        for mastered in (2, 3, 4):
            self.assertGreaterEqual(starts.count(mastered), 550)
            self.assertLessEqual(starts.count(mastered), 800)
        self.assertNotIn(0, starts)

    def test_gate_requires_threshold_and_distinct_monotonic_checkpoints(self) -> None:
        training = self.spec.make_training_env()
        failed = training.record_training_curriculum_evaluation(
            0.899, self.curriculum, evaluation_episode=25)
        unlocked = self.pass_frontier(training, episode=50)
        duplicate = self.pass_frontier(training, episode=50)

        self.assertFalse(failed["unlocked"])
        self.assertEqual(failed["frontier_after"], 4)
        self.assertTrue(unlocked["unlocked"])
        self.assertEqual(unlocked["frontier_after"], 3)
        self.assertTrue(duplicate["ignored_duplicate"])
        self.assertEqual(duplicate["frontier_after"], 3)
        self.assertEqual(training.training_curriculum_state()["evaluations"], 2)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            self.pass_frontier(training, episode=49)

    def test_final_frontier_completion_is_stable(self) -> None:
        training = self.spec.make_training_env()
        for episode, expected in zip((25, 50, 75, 100), (3, 2, 1, 0)):
            self.assertEqual(
                self.pass_frontier(training, episode=episode)["frontier_after"],
                expected,
            )
        completed = self.pass_frontier(training, episode=125)
        after_failure = training.record_training_curriculum_evaluation(
            0.0, self.curriculum, evaluation_episode=150)

        self.assertTrue(completed["complete_after"])
        self.assertFalse(completed["unlocked"])
        self.assertTrue(after_failure["complete_after"])
        self.assertEqual(after_failure["frontier_after"], 0)
        training.rng.seed(91)
        sampled = []
        for _ in range(2_000):
            training.reset()
            sampled.append(training._start_frontier)
        self.assertEqual(set(sampled), {0, 1, 2, 3, 4})
        self.assertGreaterEqual(sampled.count(0), 900)
        self.assertLessEqual(sampled.count(0), 1_100)

    def test_persisted_gate_state_is_exact_and_validated(self) -> None:
        training = self.spec.make_training_env()
        self.pass_frontier(training, episode=25)
        training.record_training_curriculum_evaluation(
            0.4, self.curriculum, evaluation_episode=50)
        expected = {
            "version": 1,
            "frontier_position": 1,
            "frontier": 3,
            "mastered": [4],
            "pass_streak": 0,
            "complete": False,
            "evaluations": 2,
            "last_success_rate": 0.4,
            "last_evaluation_episode": 50,
        }
        self.assertEqual(training.training_curriculum_state(), expected)

        restored = self.spec.make_training_env()
        restored.restore_training_curriculum_state(copy.deepcopy(expected))
        self.assertEqual(restored.training_curriculum_state(), expected)

        invalid_states = []
        for key, value in (
            ("version", 2), ("frontier_position", 9), ("frontier", 2),
            ("mastered", []), ("evaluations", -1),
            ("last_success_rate", 1.1), ("last_evaluation_episode", -1),
            ("last_success_rate", None), ("last_evaluation_episode", None),
            ("pass_streak", 1),
        ):
            invalid = copy.deepcopy(expected)
            invalid[key] = value
            invalid_states.append(invalid)
        invalid = copy.deepcopy(expected)
        invalid["complete"] = True
        invalid_states.append(invalid)
        for invalid in invalid_states:
            with self.subTest(invalid=invalid):
                with self.assertRaises((KeyError, TypeError, ValueError)):
                    restored.restore_training_curriculum_state(invalid)

    def test_rng_and_gate_state_round_trip_replays_future_starts(self) -> None:
        training = self.spec.make_training_env()
        self.pass_frontier(training)
        self.pass_frontier(training)
        training.rng.seed(718)
        state = trainer_module.capture_rng_state(training)

        def sample_starts() -> list[tuple[int, float, float, int]]:
            result = []
            for _ in range(40):
                training.reset()
                result.append((training._start_frontier, training.x,
                               training.y, training.steps))
            return result

        expected = sample_starts()
        training.record_training_curriculum_evaluation(0.0, self.curriculum)
        trainer_module.restore_rng_state(state, training)

        self.assertEqual(training.training_curriculum_state(),
                         state["training_curriculum"])
        self.assertEqual(sample_starts(), expected)

    def test_checkpoint_resume_restores_gate_and_pending_reset_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"lunar-lander"}')
            settings = Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            )
            first = trainer_module.Trainer(settings)
            self.pass_frontier(first.env, episode=25)
            self.pass_frontier(first.env, episode=50)
            first.env.rng.seed(991)
            first.episode = 50
            first.history = [{
                "episode": 50, "reward": 0.0, "steps": 1,
                "cause": "timeout", "metric": 500.0, "success": False,
            }]
            eval_payload = {
                "reward": 0.0, "reward_std": 0.0, "metric": 500.0,
                "metric_std": 0.0, "failure_progress": None, "episodes": 1,
                "success_rate": 0.0, "success_ci_low": 0.0,
                "success_ci_high": 1.0,
                "evaluation_suite": trainer_module.evaluation_suite_id(1),
                "seed": 42, "trajectory": [],
            }
            first._run_eval = lambda: copy.deepcopy(eval_payload)
            first._run_training_curriculum_eval = lambda: None
            first._save_checkpoint()
            expected_state = first.env.training_curriculum_state()
            expected = []
            for _ in range(25):
                first.env.reset()
                expected.append((first.env._start_frontier, first.env.x,
                                 first.env.y, first.env.steps))

            resumed = trainer_module.Trainer(settings)
            actual_state = resumed.env.training_curriculum_state()
            actual = []
            for _ in range(25):
                resumed.env.reset()
                actual.append((resumed.env._start_frontier, resumed.env.x,
                               resumed.env.y, resumed.env.steps))

        self.assertEqual(resumed.episode, 50)
        self.assertEqual(actual_state, expected_state)
        self.assertEqual(actual, expected)

    def test_new_seeded_run_resets_gate_to_touchdown_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"lunar-lander"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            ))
            self.pass_frontier(trainer.env)
            self.assertEqual(
                trainer.env.training_curriculum_state()["frontier"], 3)
            self.assertTrue(trainer.reset_agent(seed=17))

        state = trainer.env.training_curriculum_state()
        self.assertEqual(state["frontier"], 4)
        self.assertEqual(state["pass_streak"], 0)
        self.assertEqual(state["evaluations"], 0)
        self.assertFalse(state["complete"])
        self.assertEqual(trainer.env._start_frontier, 4)

    def test_fixed_frontier_suite_is_exact_deterministic_and_training_only(self) -> None:
        self.assertEqual(
            [self.curriculum.evaluation_seed(4, i) for i in range(20)],
            list(range(304_000, 304_020)),
        )
        self.assertEqual(self.curriculum.evaluation_suite_id(4),
                         "lander-altitude-eval-v1-k4-n20")
        first_env = self.spec.make_training_env()
        second_env = self.spec.make_training_env()
        first = trainer_module.evaluate_training_curriculum(
            self.curriculum, first_env, ConstantAgent())
        second = trainer_module.evaluate_training_curriculum(
            self.curriculum, second_env, ConstantAgent())
        self.assertEqual(first, second)
        self.assertEqual(first["seeds"], list(range(304_000, 304_020)))
        self.assertEqual(first["episodes"], 20)
        self.assertEqual(first["frontier"], 4)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"lunar-lander"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=2,
            ))
            trainer.agent = ConstantAgent()
            before = trainer._run_eval()
            self.pass_frontier(trainer.env)
            after = trainer._run_eval()
        self.assertEqual(after, before)
        self.assertNotIn("training_curriculum", before)
        self.assertEqual(before["evaluation_suite"], "policy-atlas-eval-v1-n2")

    def test_checkpoint_discloses_protocol_diagnostic_and_post_gate_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"lunar-lander"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            ))
            trainer.episode = 25
            trainer.history = [{
                "episode": 25, "reward": 0.0, "steps": 1,
                "cause": "timeout", "metric": 500.0, "success": False,
            }]
            canonical = {
                "reward": -10.0, "reward_std": 0.0, "metric": 500.0,
                "metric_std": 0.0, "failure_progress": None, "episodes": 1,
                "success_rate": 0.0, "success_ci_low": 0.0,
                "success_ci_high": 1.0, "evaluation_suite": "canonical-test",
                "seed": 42, "trajectory": [],
            }
            frontier = {
                "frontier": 4, "episodes": 20, "successes": 18,
                "success_rate": 0.9,
                "evaluation_suite": "lander-altitude-eval-v1-k4-n20",
                "seeds": list(range(304_000, 304_020)),
            }
            trainer._run_eval = lambda: copy.deepcopy(canonical)
            trainer._run_training_curriculum_eval = lambda: copy.deepcopy(frontier)
            trainer._save_checkpoint()
            meta = trainer.registry.list()[0]
            payload = trainer.registry.load(25)

        self.assertEqual(meta["schema_version"], 6)
        self.assertEqual(meta["protocol"]["version"], 13)
        self.assertEqual(meta["protocol"]["training_curriculum"],
                         EXPECTED_CURRICULUM_PROTOCOL)
        diagnostic = meta["training_diagnostics"]["training_curriculum"]
        self.assertEqual(diagnostic["frontier_before"], 4)
        self.assertEqual(diagnostic["frontier_after"], 3)
        self.assertTrue(diagnostic["unlocked"])
        self.assertEqual(diagnostic["state_after"]["frontier"], 3)
        self.assertEqual(diagnostic["state_after"]["last_evaluation_episode"], 25)
        self.assertEqual(meta["success_rate"], canonical["success_rate"])
        self.assertEqual(meta["eval_metric"], canonical["metric"])
        self.assertEqual(
            payload["rng_state"]["training_curriculum"]["frontier"], 3)

    def test_metadata_and_actor_initialization_are_exact(self) -> None:
        expected_distribution = (
            "Performance-gated reverse altitude curriculum: begin with 100% "
            "touchdown rehearsals at k4 (5-18 units above the pad); unlock low "
            "k3 (30-100), mid k2 (100-250), high k1 (250-500), then canonical "
            "k0 descents after one >=90% fixed 20-start frontier evaluation; "
            "thereafter the active frontier receives 50% of resets and mastered "
            "easier frontiers uniformly share the remainder"
        )
        self.assertEqual(self.spec.training_start_distribution,
                         expected_distribution)
        self.assertEqual(self.curriculum.protocol(), EXPECTED_CURRICULUM_PROTOCOL)
        self.assertEqual(self.spec.checkpoint_schema, 6)
        self.assertEqual(self.spec.actor_initialization.continuous_log_std,
                         (-1.2, -1.2))

    def test_curriculum_provenance_does_not_change_dynamics_or_rewards(self) -> None:
        curriculum = lander.make_frontier_evaluation_env(2)
        canonical = self.spec.make_env(True)
        curriculum.rng.seed(17)
        curriculum.reset()
        dynamic_state = (
            "x", "y", "vx", "vy", "theta", "omega", "fuel", "steps",
            "episode_reward", "landed", "cause", "_u_main", "_phi_prev",
        )
        for name in dynamic_state:
            setattr(canonical, name, getattr(curriculum, name))
        curriculum._start_kind = "approach"
        canonical._start_kind = "standard"
        action = np.array([0.1, -0.2], dtype=np.float32)
        curriculum_result = curriculum.step(action)
        canonical_result = canonical.step(action)
        np.testing.assert_allclose(curriculum_result[0], canonical_result[0])
        self.assertAlmostEqual(curriculum_result[1], canonical_result[1])
        self.assertEqual(curriculum_result[2:], canonical_result[2:])


if __name__ == "__main__":
    unittest.main()
