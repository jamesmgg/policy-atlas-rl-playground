"""Scientific contract tests for the Drone reverse waypoint curriculum."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from collections import Counter
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
        "suite_version": "drone-segment-eval-v3",
        "episodes": 10,
        "seed_base": 400_000,
        "segment_seed_stride": 1_000,
        "seed_formula": "seed_base + segment * segment_seed_stride + episode_index",
        "deterministic_policy": True,
        "start_state": (
            "noncanonical segment at preceding waypoint with standard seeded "
            "horizontal position jitter, inbound horizontal velocity using "
            "the previous-segment sign and magnitude uniform on [60, 100] "
            "units/s, zero vertical velocity/attitude/rate, cumulative-distance "
            "elapsed clock; k0 remains the canonical zero-motion start"
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

    def test_locked_frontier_uses_overlap_momentum_only_at_last_segment(self) -> None:
        training = self.spec.make_training_env()
        self.assertTrue(training.waypoint_start_curriculum)
        training.rng.seed(42)

        starts = []
        inbound = 0
        opposite = 0
        low_speed = 0
        for _ in range(4_000):
            training.reset()
            starts.append(training.k)
            self.assertLessEqual(abs(training.x - drone.WAYPOINTS[3][0]), 30.0)
            self.assertEqual(training.y, drone.WAYPOINTS[3][1])
            self.assertLessEqual(abs(training.vx), 100.0)
            inbound += training.vx > 0.0
            opposite += training.vx < 0.0
            low_speed += abs(training.vx) < 60.0
            self.assertEqual(training.vy, 0.0)
            self.assertEqual(training.theta, 0.0)
            self.assertEqual(training.omega, 0.0)
            self.assertEqual(training.steps, drone.WAYPOINT_START_STEPS[3])

        self.assertEqual(set(starts), {4})
        # 50% hard inbound U[60, 100] + 50% symmetric U[-100, 100]
        # yields approximately 75% inbound, 25% opposite-direction, and 30%
        # below the hard suite's minimum speed.
        self.assertGreaterEqual(inbound, 2_850)
        self.assertLessEqual(inbound, 3_150)
        self.assertGreaterEqual(opposite, 850)
        self.assertLessEqual(opposite, 1_150)
        self.assertGreaterEqual(low_speed, 1_050)
        self.assertLessEqual(low_speed, 1_350)
        self.assertEqual(training.training_curriculum_state()["frontier"], 4)

    def test_unlocked_sampling_rehearses_active_and_mastered_frontiers(self) -> None:
        training = self.spec.make_training_env()
        for expected in (3, 2, 1):
            transition = pass_frontier(training)
            self.assertTrue(transition["unlocked"])
            self.assertEqual(transition["frontier_after"], expected)

        # Frontier 1 is active; 2, 3 and 4 are mastered later segments.
        training.rng.seed(2026)
        starts = []
        inbound = 0
        opposite = 0
        low_speed = 0
        for _ in range(4_000):
            training.reset()
            starts.append(training.k)
            origin = drone.WAYPOINTS[training.k - 1]
            prior = drone.START if training.k == 1 else drone.WAYPOINTS[
                training.k - 2]
            self.assertLessEqual(abs(training.vx), 100.0)
            relative_velocity = training.vx * (origin[0] - prior[0])
            inbound += relative_velocity > 0.0
            opposite += relative_velocity < 0.0
            low_speed += abs(training.vx) < 60.0

        counts = Counter(starts)
        self.assertEqual(set(starts), {1, 2, 3, 4})
        self.assertGreaterEqual(counts[1], 1_800)
        self.assertLessEqual(counts[1], 2_200)
        for mastered in (2, 3, 4):
            self.assertGreaterEqual(counts[mastered], 500)
            self.assertLessEqual(counts[mastered], 850)
        self.assertGreaterEqual(inbound, 2_850)
        self.assertLessEqual(inbound, 3_150)
        self.assertGreaterEqual(opposite, 850)
        self.assertLessEqual(opposite, 1_150)
        self.assertGreaterEqual(low_speed, 1_050)
        self.assertLessEqual(low_speed, 1_350)

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

    def test_final_frontier_keeps_half_of_resets_canonical(self) -> None:
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
            if training.k == 0:
                self.assertEqual(training.vx, 0.0)
            else:
                self.assertLessEqual(abs(training.vx), 100.0)
        counts = Counter(starts)
        self.assertEqual(set(starts), {0, 1, 2, 3, 4})
        self.assertGreaterEqual(counts[0], 900)
        self.assertLessEqual(counts[0], 1_100)

    def test_segment_reset_replays_seeded_inbound_horizontal_momentum(self) -> None:
        for segment in range(len(drone.WAYPOINTS)):
            env = drone.make_segment_evaluation_env(segment)
            seed = self.curriculum.evaluation_seed(segment, 0)
            env.rng.seed(seed)
            observation = env.reset()
            replay = drone.make_segment_evaluation_env(segment)
            replay.rng.seed(seed)
            replay.reset()
            origin = drone.START if segment == 0 else drone.WAYPOINTS[segment - 1]
            expected_steps = (0 if segment == 0
                              else drone.WAYPOINT_START_STEPS[segment - 1])

            self.assertLessEqual(abs(env.x - origin[0]), 30.0)
            self.assertEqual(env.y, origin[1])
            if segment == 0:
                self.assertEqual(env.vx, 0.0)
            else:
                prior = drone.START if segment == 1 else drone.WAYPOINTS[segment - 2]
                inbound_dx = origin[0] - prior[0]
                self.assertGreater(env.vx * inbound_dx, 0.0)
                self.assertGreaterEqual(abs(env.vx), 60.0)
                self.assertLessEqual(abs(env.vx), 100.0)
            self.assertEqual(replay.vx, env.vx)
            self.assertEqual(env.vy, 0.0)
            self.assertEqual(env.theta, 0.0)
            self.assertEqual(env.omega, 0.0)
            self.assertEqual(env.k, segment)
            self.assertEqual(env.steps, expected_steps)
            self.assertEqual(env.episode_reward, 0.0)
            self.assertEqual(env.cause, "running")
            self.assertAlmostEqual(float(observation[7]),
                                   segment / len(drone.WAYPOINTS), places=6)
            self.assertAlmostEqual(float(observation[2]),
                                   env.vx / 60.0, places=6)
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

    def test_undiscounted_timeout_and_crash_have_comparable_failure_costs(self) -> None:
        """Hovering until the deadline must not dominate attempting the course."""
        self.assertEqual(self.spec.gamma, 1.0)
        self.assertEqual(self.spec.info()["gamma"], 1.0)

        def constant_action_return(action: np.ndarray) -> dict:
            env = self.spec.make_env(False)
            env.reset()
            raw_return = 0.0
            discounted_return = 0.0
            terminal_reward = None
            for step in range(env.max_steps):
                _, reward, done, _ = env.step(action)
                raw_return += reward
                discounted_return += (self.spec.gamma ** step) * reward
                if done:
                    terminal_reward = reward
                    break
            self.assertTrue(done)
            return {
                "cause": env.cause,
                "raw_return": raw_return,
                "discounted_return": discounted_return,
                "terminal_reward": terminal_reward,
            }

        hover_throttle = drone.G / (2.0 * drone.T_MAX)
        hover_action = np.full(2, 2.0 * hover_throttle - 1.0,
                               dtype=np.float32)
        timeout = constant_action_return(hover_action)
        crash = constant_action_return(np.full(2, -1.0, dtype=np.float32))

        self.assertEqual(timeout["cause"], "timeout")
        self.assertEqual(crash["cause"], "crash")
        # The terminal transition also carries the final potential delta, so
        # floating-point hover drift may place it infinitesimally above -50.
        self.assertLessEqual(
            timeout["terminal_reward"],
            -drone.TERMINAL_FAILURE_PENALTY + 1e-6,
        )
        self.assertAlmostEqual(timeout["discounted_return"],
                               timeout["raw_return"], places=9)
        self.assertAlmostEqual(crash["discounted_return"],
                               crash["raw_return"], places=9)
        magnitudes = [abs(timeout["discounted_return"]),
                      abs(crash["discounted_return"])]
        self.assertLess(max(magnitudes) / min(magnitudes), 2.0)

    def test_failure_return_does_not_reward_an_earlier_identical_crash(self) -> None:
        """A longer same-state failure must not pay extra path regularizers."""
        hover_throttle = drone.G / (2.0 * drone.T_MAX)
        hover_action = np.full(
            2, 2.0 * hover_throttle - 1.0, dtype=np.float64)
        crash_action = np.full(2, -1.0, dtype=np.float64)

        def crash_after_hover(hover_steps: int) -> tuple[float, float]:
            env = self.spec.make_env(False)
            initial_distance = env._d_prev
            raw_return = 0.0
            for _ in range(hover_steps):
                _, reward, done, _ = env.step(hover_action)
                self.assertFalse(done)
                raw_return += reward
            # Trigger the same zero-thrust terminal transition from the same
            # physical state. The elapsed clock is the only intended change.
            env.theta = drone.TIP_OVER + 0.01
            _, reward, done, _ = env.step(crash_action)
            raw_return += reward
            self.assertTrue(done)
            self.assertEqual(env.cause, "crash")
            potential_and_terminal = (
                0.05 * (initial_distance - env._d_prev)
                - drone.TERMINAL_FAILURE_PENALTY
            )
            return raw_return, potential_and_terminal

        early, early_expected = crash_after_hover(0)
        delayed, delayed_expected = crash_after_hover(100)

        self.assertAlmostEqual(early, early_expected, places=9)
        self.assertAlmostEqual(delayed, delayed_expected, places=9)
        self.assertGreaterEqual(delayed, early - 1e-9)

    def test_efficiency_regularizers_are_charged_on_completion(self) -> None:
        """Successful policies retain the attitude/energy tie-breaker."""
        env = drone.DroneEnv(jitter=False, forced_start_segment=4)
        env.vx = 0.0
        first_distance = env._d_prev
        effort_action = np.array([1.0, -1.0], dtype=np.float64)
        _, first_reward, done, _ = env.step(effort_action)
        self.assertFalse(done)
        first_progress = 0.05 * (first_distance - env._d_prev)
        self.assertAlmostEqual(first_reward, first_progress, places=9)
        accumulated_regularizer = (
            0.002 * abs(env.omega)
            + 0.005
        )

        env.x, env.y = drone.WAYPOINTS[-1]
        env.vx = env.vy = env.theta = env.omega = 0.0
        env._d_prev = env._dist()
        _, completion_reward, done, _ = env.step(
            np.full(2, -1.0, dtype=np.float64))

        self.assertTrue(done)
        self.assertEqual(env.cause, "complete")
        completion_progress = -0.05 * env._d_prev
        self.assertAlmostEqual(
            first_reward + completion_reward,
            first_progress + completion_progress + 70.0
            - accumulated_regularizer,
            places=9,
        )

    def test_fixed_segment_suite_seeds_are_exact_and_repeatable(self) -> None:
        self.assertEqual(
            [self.curriculum.evaluation_seed(4, i) for i in range(10)],
            list(range(404_000, 404_010)),
        )
        self.assertEqual(self.curriculum.evaluation_suite_id(4),
                         "drone-segment-eval-v3-k4-n10")

        first_env = self.spec.make_training_env()
        second_env = self.spec.make_training_env()
        first = trainer_module.evaluate_training_curriculum(
            self.curriculum, first_env, ConstantAgent())
        second = trainer_module.evaluate_training_curriculum(
            self.curriculum, second_env, ConstantAgent())

        self.assertEqual(first, second)
        self.assertEqual(first["evaluation_suite"],
                         "drone-segment-eval-v3-k4-n10")
        self.assertEqual(first["seeds"], list(range(404_000, 404_010)))
        self.assertEqual(first["episodes"], 10)
        self.assertEqual(first["frontier"], 4)

    def test_every_segment_suite_is_disjoint_from_selection_and_holdout(self) -> None:
        named_ranges = {
            f"segment-k{segment}": {
                self.curriculum.evaluation_seed(segment, episode)
                for episode in range(self.curriculum.evaluation_episodes)
            }
            for segment in self.curriculum.frontier_order
        }
        named_ranges["checkpoint-selection"] = {
            trainer_module.evaluation_seed(episode) for episode in range(10)
        }
        named_ranges["default-holdout"] = set(range(200_000, 200_100))

        names = list(named_ranges)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1:]:
                with self.subTest(left=left_name, right=right_name):
                    self.assertTrue(
                        named_ranges[left_name].isdisjoint(
                            named_ranges[right_name]),
                        f"{left_name} overlaps {right_name}",
                    )

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
                self.assertEqual(env.vx, 0.0)
                self.assertEqual(env.vy, 0.0)
                self.assertEqual(env.theta, 0.0)
                self.assertEqual(env.omega, 0.0)
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

        def sample_starts() -> list[tuple[int, float, float, int]]:
            result = []
            for _ in range(40):
                training.reset()
                result.append((training.k, training.x, training.vx, training.steps))
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
                expected.append((
                    first.env.k, first.env.x, first.env.vx, first.env.steps))

            resumed = trainer_module.Trainer(settings)
            actual_state = resumed.env.training_curriculum_state()
            actual = []
            for _ in range(25):
                resumed.env.reset()
                actual.append((
                    resumed.env.k, resumed.env.x, resumed.env.vx,
                    resumed.env.steps))

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
                "evaluation_suite": "drone-segment-eval-v3-k4-n10",
                "seeds": list(range(404_000, 404_010)),
            }
            trainer._run_eval = lambda: copy.deepcopy(canonical)
            trainer._run_training_curriculum_eval = lambda: copy.deepcopy(segment)
            trainer._save_checkpoint()
            meta = trainer.registry.list()[0]
            payload = trainer.registry.load(0)

        self.assertEqual(meta["schema_version"], 13)
        self.assertEqual(meta["protocol"]["version"], 13)
        self.assertEqual(meta["protocol"]["gamma"], 1.0)
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
            "thunder-oval": 7, "apex-gp-wet": 7, "glacier": 7,
            "rally-ridge": 7, "kart-sprint": 7, "drift-trial": 8,
            "eco-gp": 7, "traffic-rush": 8, "lunar-lander": 5,
            "pendulum-swingup": 2, "cartpole-balance": 2,
            "mountain-car": 3,
        }
        non_drone = {spec.id: spec for spec in list_specs()
                     if spec.id != "drone-hover"}

        self.assertEqual({key: spec.checkpoint_schema
                          for key, spec in non_drone.items()}, expected_schemas)
        self.assertTrue(all(spec.gamma == 0.995
                            for spec in non_drone.values()))
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
            "segment evaluation; active frontier receives 50% of resets and "
            "mastered later segments uniformly share the remainder; "
            "noncanonical training starts mix 50% hard inbound horizontal "
            "velocity with the previous-segment sign and magnitude uniform on "
            "[60, 100] units/s and 50% overlap velocity uniform on [-100, 100] "
            "units/s; fixed segment gates retain only the hard inbound starts",
        )
        self.assertEqual(self.spec.training_curriculum.protocol(),
                         EXPECTED_CURRICULUM_PROTOCOL)
        self.assertEqual(
            self.spec.reward_terms,
            (
                "+0.05 × decrease in waypoint distance",
                "+20 per waypoint and +50 course completion",
                "angular-rate and squared-thrust regularizers charged only "
                "on successful course completion",
                "−50 crash or timeout",
            ),
        )
        self.assertEqual(self.spec.checkpoint_schema, 13)


if __name__ == "__main__":
    unittest.main()
