"""Scientific contract tests for the Drone reverse waypoint curriculum."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).parents[1]))

from app import trainer as trainer_module
from app.envs import drone
from app.scenarios import get_spec, list_specs
from app.settings import Settings


EXPECTED_CURRICULUM_PROTOCOL = {
    "id": "drone-reverse-waypoint-v1",
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
        "suite_version": "drone-segment-eval-v1",
        "episodes": 10,
        "seed_base": 200_000,
        "segment_seed_stride": 1_000,
        "seed_formula": "seed_base + segment * segment_seed_stride + episode_index",
        "deterministic_policy": True,
        "start_state": (
            "preceding waypoint, zero velocity/attitude/rate, standard seeded "
            "horizontal jitter, cumulative-distance elapsed clock"
        ),
    },
    "checkpoint_selection": (
        "segment evaluation is training-only diagnostic; fixed full-course "
        "evaluation remains the checkpoint-selection signal"
    ),
}


def pass_frontier(env) -> dict:
    """Supply the prescribed confirmation and return the transition."""
    spec = get_spec("drone-hover").training_curriculum
    assert spec is not None
    return env.record_training_curriculum_evaluation(0.9, spec)


class ConstantAgent:
    def __init__(self, action: tuple[float, float] = (0.0, 0.0)) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def select_action(self, observation, deterministic=False):
        del observation
        if not deterministic:
            raise AssertionError("curriculum evaluation must be deterministic")
        return self.action.copy(), 0.0, 0.0


class TestDroneReverseCurriculum(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_spec("drone-hover")
        self.curriculum = self.spec.training_curriculum
        self.assertIsNotNone(self.curriculum)

    def test_locked_frontier_starts_only_at_last_segment(self) -> None:
        training = self.spec.make_training_env()
        self.assertTrue(training.waypoint_start_curriculum)
        training.rng.seed(42)

        starts = []
        for _ in range(200):
            training.reset()
            starts.append(training.k)
            self.assertLessEqual(abs(training.x - drone.WAYPOINTS[3][0]), 30.0)
            self.assertEqual(training.y, drone.WAYPOINTS[3][1])
            self.assertEqual(training.steps, drone.WAYPOINT_START_STEPS[3])

        self.assertEqual(set(starts), {4})
        self.assertEqual(training.training_curriculum_state()["frontier"], 4)

    def test_unlocked_sampling_is_half_frontier_and_uniform_over_mastered(self) -> None:
        training = self.spec.make_training_env()
        for expected in (3, 2, 1):
            transition = pass_frontier(training)
            self.assertTrue(transition["unlocked"])
            self.assertEqual(transition["frontier_after"], expected)

        # Frontier 1 is active; 2, 3 and 4 are mastered later segments.
        training.rng.seed(2026)
        starts = []
        for _ in range(4_000):
            training.reset()
            starts.append(training.k)

        self.assertEqual(set(starts), {1, 2, 3, 4})
        self.assertGreaterEqual(starts.count(1), 1_800)
        self.assertLessEqual(starts.count(1), 2_200)
        for mastered in (2, 3, 4):
            self.assertGreaterEqual(starts.count(mastered), 550)
            self.assertLessEqual(starts.count(mastered), 800)
        self.assertNotIn(0, starts)

    def test_gate_unlocks_after_one_threshold_pass(self) -> None:
        training = self.spec.make_training_env()

        failed = training.record_training_curriculum_evaluation(0.89, self.curriculum)
        self.assertEqual(failed["frontier_after"], 4)
        self.assertEqual(failed["confirmation_streak_after"], 0)
        self.assertFalse(failed["unlocked"])

        unlocked = training.record_training_curriculum_evaluation(
            0.9, self.curriculum)
        self.assertEqual(unlocked["frontier_before"], 4)
        self.assertEqual(unlocked["frontier_after"], 3)
        self.assertEqual(unlocked["mastered_after"], [4])
        self.assertEqual(unlocked["confirmation_streak_after"], 0)
        self.assertTrue(unlocked["unlocked"])

    def test_repeated_save_at_one_episode_cannot_double_count_confirmation(self) -> None:
        training = self.spec.make_training_env()
        first = training.record_training_curriculum_evaluation(
            0.9, self.curriculum, evaluation_episode=25)
        duplicate = training.record_training_curriculum_evaluation(
            0.9, self.curriculum, evaluation_episode=25)

        self.assertTrue(first["unlocked"])
        self.assertEqual(first["frontier_after"], 3)
        self.assertEqual(first["confirmation_streak_after"], 0)
        self.assertTrue(duplicate["ignored_duplicate"])
        self.assertEqual(duplicate["frontier_after"], 3)
        self.assertEqual(duplicate["confirmation_streak_after"], 0)
        self.assertEqual(training.training_curriculum_state()["evaluations"], 1)
        self.assertEqual(training.training_curriculum_state()[
            "last_evaluation_episode"], 25)

        next_unlock = training.record_training_curriculum_evaluation(
            0.9, self.curriculum, evaluation_episode=50)
        self.assertTrue(next_unlock["unlocked"])
        self.assertEqual(next_unlock["frontier_after"], 2)

    def test_final_frontier_is_unlocked_and_sampling_stays_stable(self) -> None:
        training = self.spec.make_training_env()
        for expected_frontier in (3, 2, 1, 0):
            self.assertEqual(pass_frontier(training)["frontier_after"],
                             expected_frontier)
        first = training.record_training_curriculum_evaluation(0.89, self.curriculum)
        completed = training.record_training_curriculum_evaluation(0.9, self.curriculum)

        self.assertFalse(first["complete_after"])
        self.assertTrue(completed["complete_after"])
        self.assertFalse(completed["unlocked"])
        self.assertEqual(completed["frontier_after"], 0)

        training.rng.seed(41)
        starts = []
        for _ in range(2_000):
            training.reset()
            starts.append(training.k)
        self.assertEqual(set(starts), {0, 1, 2, 3, 4})
        self.assertGreaterEqual(starts.count(0), 900)
        self.assertLessEqual(starts.count(0), 1_100)

    def test_segment_reset_exposes_the_existing_exact_task_and_clock_state(self) -> None:
        for segment in range(len(drone.WAYPOINTS)):
            env = drone.make_segment_evaluation_env(segment)
            env.rng.seed(self.curriculum.evaluation_seed(segment, 0))
            observation = env.reset()
            origin = drone.START if segment == 0 else drone.WAYPOINTS[segment - 1]
            expected_steps = (0 if segment == 0
                              else drone.WAYPOINT_START_STEPS[segment - 1])

            self.assertLessEqual(abs(env.x - origin[0]), 30.0)
            self.assertEqual(env.y, origin[1])
            self.assertEqual(env.vx, 0.0)
            self.assertEqual(env.vy, 0.0)
            self.assertEqual(env.theta, 0.0)
            self.assertEqual(env.omega, 0.0)
            self.assertEqual(env.k, segment)
            self.assertEqual(env.steps, expected_steps)
            self.assertEqual(env.episode_reward, 0.0)
            self.assertEqual(env.cause, "running")
            self.assertAlmostEqual(float(observation[7]),
                                   segment / len(drone.WAYPOINTS), places=6)
            self.assertAlmostEqual(float(observation[-1]),
                                   1.0 - expected_steps / env.max_steps,
                                   places=6)

    def test_curriculum_provenance_does_not_change_dynamics_or_rewards(self) -> None:
        curriculum = self.spec.make_training_env()
        canonical = self.spec.make_env(True)
        curriculum.rng.seed(17)
        curriculum.reset()

        dynamic_state = (
            "x", "y", "vx", "vy", "theta", "omega", "k", "steps",
            "episode_reward", "cause", "_d_prev",
        )
        for name in dynamic_state:
            setattr(canonical, name, getattr(curriculum, name))

        action = np.array([0.1, -0.2], dtype=np.float32)
        curriculum_result = curriculum.step(action)
        canonical_result = canonical.step(action)

        np.testing.assert_allclose(curriculum_result[0], canonical_result[0])
        self.assertAlmostEqual(curriculum_result[1], canonical_result[1])
        self.assertEqual(curriculum_result[2:], canonical_result[2:])

    def test_fixed_segment_suite_seeds_are_exact_and_repeatable(self) -> None:
        self.assertEqual(
            [self.curriculum.evaluation_seed(4, i) for i in range(10)],
            list(range(204_000, 204_010)),
        )
        self.assertEqual(self.curriculum.evaluation_suite_id(4),
                         "drone-segment-eval-v1-k4-n10")

        first_env = self.spec.make_training_env()
        second_env = self.spec.make_training_env()
        first = trainer_module.evaluate_training_curriculum(
            self.curriculum, first_env, ConstantAgent())
        second = trainer_module.evaluate_training_curriculum(
            self.curriculum, second_env, ConstantAgent())

        self.assertEqual(first, second)
        self.assertEqual(first["evaluation_suite"],
                         "drone-segment-eval-v1-k4-n10")
        self.assertEqual(first["seeds"], list(range(204_000, 204_010)))
        self.assertEqual(first["episodes"], 10)
        self.assertEqual(first["frontier"], 4)

    def test_evaluation_factories_and_canonical_selection_remain_full_course(self) -> None:
        fixed_suite = self.spec.make_env(True)
        canonical = self.spec.make_env(False)
        for env in (fixed_suite, canonical):
            self.assertFalse(env.waypoint_start_curriculum)
            self.assertIsNone(env.forced_start_segment)
            env.rng.seed(7)
            for _ in range(25):
                observation = env.reset()
                self.assertEqual(env.k, 0)
                self.assertEqual(env.y, drone.START[1])
                self.assertEqual(float(observation[7]), 0.0)
                self.assertEqual(env.steps, 0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"drone-hover"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=2,
            ))
            trainer.agent = ConstantAgent()
            before = trainer._run_eval()
            pass_frontier(trainer.env)
            after = trainer._run_eval()

        # Training-only segment diagnostics cannot alter checkpoint selection.
        self.assertEqual(after, before)
        self.assertNotIn("training_curriculum", before)
        self.assertEqual(before["evaluation_suite"], "policy-atlas-eval-v1-n2")

    def test_curriculum_state_round_trip_replays_future_reset_sequence(self) -> None:
        training = self.spec.make_training_env()
        pass_frontier(training)
        training.record_training_curriculum_evaluation(1.0, self.curriculum)
        training.rng.seed(718)
        state = trainer_module.capture_rng_state(training)

        def sample_starts() -> list[tuple[int, float, int]]:
            result = []
            for _ in range(40):
                training.reset()
                result.append((training.k, training.x, training.steps))
            return result

        expected = sample_starts()
        training.record_training_curriculum_evaluation(0.0, self.curriculum)
        trainer_module.restore_rng_state(state, training)

        self.assertEqual(training.training_curriculum_state(),
                         state["training_curriculum"])
        self.assertEqual(sample_starts(), expected)

    def test_checkpoint_resume_restores_gate_state_and_pending_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"drone-hover"}')
            settings = Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            )
            first = trainer_module.Trainer(settings)
            pass_frontier(first.env)
            first.env.record_training_curriculum_evaluation(
                1.0, self.curriculum)
            first.env.rng.seed(991)
            first.episode = 25
            first.history = [{
                "episode": 25, "reward": 0.0, "steps": 1,
                "cause": "timeout", "metric": 4.0, "success": False,
            }]
            eval_payload = {
                "reward": 0.0, "reward_std": 0.0, "metric": 0.0,
                "metric_std": 0.0, "failure_progress": None, "episodes": 1,
                "success_rate": 0.0, "success_ci_low": 0.0,
                "success_ci_high": 1.0, "evaluation_suite": "canonical-test",
                "seed": 42, "trajectory": [],
            }
            first._run_eval = lambda: copy.deepcopy(eval_payload)
            first._run_training_curriculum_eval = lambda: None
            first._save_checkpoint()
            expected_state = first.env.training_curriculum_state()
            expected = []
            for _ in range(25):
                first.env.reset()
                expected.append((first.env.k, first.env.x, first.env.steps))

            resumed = trainer_module.Trainer(settings)
            actual_state = resumed.env.training_curriculum_state()
            actual = []
            for _ in range(25):
                resumed.env.reset()
                actual.append((resumed.env.k, resumed.env.x, resumed.env.steps))

        self.assertEqual(resumed.episode, 25)
        self.assertEqual(actual_state, expected_state)
        self.assertEqual(actual, expected)

    def test_new_seeded_run_resets_the_curriculum_to_k4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"drone-hover"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            ))
            pass_frontier(trainer.env)
            self.assertEqual(trainer.env.training_curriculum_state()["frontier"], 3)

            self.assertTrue(trainer.reset_agent(seed=17))

        state = trainer.env.training_curriculum_state()
        self.assertEqual(state["frontier"], 4)
        self.assertEqual(state["pass_streak"], 0)
        self.assertEqual(state["evaluations"], 0)
        self.assertFalse(state["complete"])
        self.assertEqual(trainer.env.k, 4)

    def test_protocol_and_diagnostics_disclose_the_exact_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"drone-hover"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            ))
            canonical = {
                "reward": 11.0, "reward_std": 0.0, "metric": 2.0,
                "metric_std": 0.0, "failure_progress": 0.4, "episodes": 1,
                "success_rate": 0.0, "success_ci_low": 0.0,
                "success_ci_high": 1.0, "evaluation_suite": "canonical-test",
                "seed": 42, "trajectory": [],
            }
            segment = {
                "frontier": 4,
                "episodes": 10,
                "successes": 9,
                "success_rate": 0.9,
                "evaluation_suite": "drone-segment-eval-v1-k4-n10",
                "seeds": list(range(204_000, 204_010)),
            }
            trainer._run_eval = lambda: copy.deepcopy(canonical)
            trainer._run_training_curriculum_eval = lambda: copy.deepcopy(segment)
            trainer._save_checkpoint()
            meta = trainer.registry.list()[0]
            payload = trainer.registry.load(0)

        self.assertEqual(meta["schema_version"], 6)
        self.assertEqual(meta["protocol"]["version"], 11)
        self.assertEqual(meta["protocol"]["training_curriculum"],
                         EXPECTED_CURRICULUM_PROTOCOL)
        diagnostic = meta["training_diagnostics"]["training_curriculum"]
        self.assertEqual(diagnostic["frontier_before"], 4)
        self.assertEqual(diagnostic["frontier_after"], 3)
        self.assertEqual(diagnostic["confirmation_streak_after"], 0)
        self.assertTrue(diagnostic["unlocked"])
        self.assertEqual(diagnostic["state_after"]["frontier"], 3)
        self.assertEqual(diagnostic["state_after"]["pass_streak"], 0)
        self.assertEqual(diagnostic["state_after"]["evaluations"], 1)
        self.assertEqual(diagnostic["state_after"]["last_evaluation_episode"], 0)
        self.assertEqual(meta["success_rate"], canonical["success_rate"])
        self.assertEqual(meta["eval_metric"], canonical["metric"])
        self.assertEqual(
            payload["rng_state"]["training_curriculum"]["frontier"], 3)

    def test_non_drone_scenarios_keep_their_training_contracts_and_schemas(self) -> None:
        expected_schemas = {
            "apex-gp": 7, "velocita": 7, "grandville": 7,
            "thunder-oval": 7, "apex-gp-wet": 9, "glacier": 7,
            "rally-ridge": 7, "kart-sprint": 7, "drift-trial": 8,
            "eco-gp": 7, "traffic-rush": 9, "lunar-lander": 5,
            "pendulum-swingup": 2, "cartpole-balance": 2,
            "mountain-car": 3,
        }
        non_drone = {spec.id: spec for spec in list_specs()
                     if spec.id != "drone-hover"}

        self.assertEqual({key: spec.checkpoint_schema
                          for key, spec in non_drone.items()}, expected_schemas)
        self.assertTrue(all(spec.training_curriculum is None
                            for spec in non_drone.values()))
        self.assertFalse(any(hasattr(spec.make_training_env(),
                                     "training_curriculum_state")
                             for spec in non_drone.values()))

    def test_curriculum_and_schema_are_explicit_in_scenario_metadata(self) -> None:
        self.assertEqual(
            self.spec.training_start_distribution,
            "Performance-gated reverse waypoint curriculum: start at target 5 "
            "(k4); unlock k3, k2, k1, then canonical k0 after one >=90% fixed "
            "segment evaluation; active frontier receives 50% "
            "of resets and mastered later segments uniformly share the remainder",
        )
        self.assertEqual(self.spec.training_curriculum.protocol(),
                         EXPECTED_CURRICULUM_PROTOCOL)
        self.assertEqual(self.spec.checkpoint_schema, 6)


if __name__ == "__main__":
    unittest.main()
