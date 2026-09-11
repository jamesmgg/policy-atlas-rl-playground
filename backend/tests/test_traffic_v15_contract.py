"""Scientific contract for Traffic's course reward and reverse stages."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import trainer as trainer_module  # noqa: E402
from app.envs import driving  # noqa: E402
from app.scenarios import get_spec, list_specs  # noqa: E402
from app.settings import Settings  # noqa: E402


EXPECTED_FRONTIERS = (11, 9, 3, 0)
EXPECTED_THRESHOLDS = {11: 0.8, 9: 0.8, 3: 0.8, 0: 0.9}


class ConstantAgent:
    def __init__(self, action=(0.0, 0.0, 0.0)) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def select_action(self, observation, deterministic=False):
        del observation
        if not deterministic:
            raise AssertionError("curriculum evaluation must be deterministic")
        return self.action.copy(), 0.0, 0.0


class TrafficV15ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_spec("traffic-rush")

    def curriculum(self):
        curriculum = self.spec.training_curriculum
        self.assertIsNotNone(curriculum)
        return curriculum

    def training_env(self):
        env_type = getattr(driving, "TrafficCurriculumEnv", None)
        self.assertIsNotNone(env_type)
        env = self.spec.make_training_env()
        self.assertIsInstance(env, env_type)
        return env

    @staticmethod
    def checkpoint_index(env) -> int:
        return env.track.checkpoints.index(env.idx)

    def pass_frontier(self, env, first_episode: int) -> tuple[dict, dict]:
        curriculum = self.curriculum()
        frontier = env.training_curriculum_state()["frontier"]
        if frontier == 11:
            self.complete_speed_control(env)
        threshold = EXPECTED_THRESHOLDS[frontier]
        first = env.record_training_curriculum_evaluation(
            threshold, curriculum, evaluation_episode=first_episode)
        second = env.record_training_curriculum_evaluation(
            threshold, curriculum, evaluation_episode=first_episode + 25)
        return first, second

    def complete_speed_control(self, env) -> None:
        curriculum = self.curriculum()
        control = getattr(curriculum, "training_control", None)
        if control is None:
            return
        while not env.training_control_state()["complete"]:
            state = env.training_control_state()
            episode = (
                0 if state["last_evaluation_episode"] is None
                else state["last_evaluation_episode"] + 1
            )
            env.record_training_control_evaluation(
                control.success_rate_threshold,
                control,
                evaluation_episode=episode,
            )

    # -------------------------------------------------------------- reward

    def test_traffic_retains_terminal_course_potential_without_drift_bonus(self) -> None:
        env = self.spec.make_env(False)
        cfg = env.reward_cfg

        self.assertFalse(cfg.terminal_zero_course_potential)
        self.assertTrue(getattr(
            cfg, "retain_terminal_course_potential", False))
        self.assertEqual(cfg.drift_corner, 0.0)
        self.assertEqual(cfg.progress, 0.05)
        self.assertEqual(cfg.checkpoint, 3.0)
        self.assertEqual(cfg.lap, 30.0)
        self.assertTrue(any(
            "retained terminal course potential" in term
            and "episode sum = potential(end) - potential(start)" in term
            for term in self.spec.reward_terms
        ))
        self.assertFalse(any("corner slip" in term
                             for term in self.spec.reward_terms))

    def test_course_potential_formula_and_retained_delta_are_exact(self) -> None:
        delta = getattr(driving, "retained_course_potential_delta", None)
        self.assertTrue(callable(delta))
        env = self.spec.make_env(False)
        potential = getattr(env, "course_reward_potential", None)
        self.assertTrue(callable(potential))

        env.progress = 100.0
        env.next_cp = 4
        env.laps = 1
        self.assertAlmostEqual(potential(), 0.05 * 100.0 + 3.0 * 3 + 30.0)

        paths = (
            (0.0, (10.0, 30.0, 80.0)),
            (0.0, (2.0, 4.0)),
            (37.5, (50.0, 200.0, 800.0, 900.0)),
            (104.0, (90.0, 20.0)),
        )
        for start, states in paths:
            with self.subTest(start=start, states=states):
                total = 0.0
                previous = start
                for current in states:
                    total += delta(previous, current)
                    previous = current
                self.assertAlmostEqual(total, states[-1] - start)

    def test_canonical_full_returns_order_clean_overtakes_then_success(self) -> None:
        env = self.spec.make_env(False)
        cfg = env.reward_cfg
        potential = getattr(env, "course_reward_potential", None)
        self.assertTrue(callable(potential))
        self.assertEqual(potential(), 0.0)

        failure_clock = cfg.time * env.max_steps
        partial_returns = [
            failure_clock + cfg.collision + count * cfg.overtake
            for count in range(3)
        ]
        slowest_success = cfg.time * env.max_steps + 3 * cfg.overtake

        self.assertEqual(partial_returns, [-85.0, -77.0, -69.0])
        self.assertEqual(slowest_success, -21.0)
        self.assertLess(partial_returns[0], partial_returns[1])
        self.assertLess(partial_returns[1], partial_returns[2])
        self.assertLess(partial_returns[2], slowest_success)

    def test_runtime_zero_progress_failure_has_only_clock_and_safety_cost(self) -> None:
        env = self.spec.make_env(False)
        done = False
        while not done:
            _, _, done, _ = env.step(np.zeros(3, dtype=np.float32))

        self.assertEqual(env.cause, "stall")
        self.assertEqual(env.overtakes, 0)
        self.assertAlmostEqual(env.episode_reward, -85.0)

    def test_pass_candidates_commit_exactly_once_only_after_safe_transition(self) -> None:
        env = self.spec.make_env(False)
        commit = getattr(env, "_commit_bot_passes", None)
        self.assertTrue(callable(commit))

        self.assertEqual(commit([0]), 8.0)
        self.assertEqual(env.overtakes, 1)
        self.assertEqual(commit([0]), 0.0)
        self.assertEqual(env.overtakes, 1)

        for unsafe_cause in ("contact", "collision"):
            with self.subTest(cause=unsafe_cause):
                penalized = self.spec.make_env(False)
                baseline = copy.deepcopy(penalized)
                if unsafe_cause == "collision":
                    for candidate in (penalized, baseline):
                        candidate.car.x = -10_000.0
                contact = unsafe_cause == "contact"
                with patch.object(
                        penalized, "_advance_bots", return_value=[0]), patch.object(
                        baseline, "_advance_bots", return_value=[]), patch.object(
                        penalized, "_check_contact", return_value=contact), patch.object(
                        baseline, "_check_contact", return_value=contact):
                    _, candidate_reward, candidate_done, _ = penalized.step(
                        np.zeros(3, dtype=np.float32))
                    _, baseline_reward, baseline_done, _ = baseline.step(
                        np.zeros(3, dtype=np.float32))

                self.assertTrue(candidate_done and baseline_done)
                self.assertEqual(penalized.cause, unsafe_cause)
                self.assertEqual(penalized.overtakes, 0)
                self.assertEqual(penalized._bot_passed, [False, False, False])
                self.assertAlmostEqual(candidate_reward, baseline_reward)

    # ----------------------------------------------------------- curriculum

    def test_curriculum_contract_uses_reverse_frontiers_and_per_stage_gates(self) -> None:
        curriculum = self.curriculum()

        self.assertEqual(curriculum.frontier_order, EXPECTED_FRONTIERS)
        self.assertEqual(curriculum.active_frontier_probability, 0.8)
        self.assertEqual(curriculum.consecutive_confirmations, 2)
        self.assertEqual(curriculum.evaluation_episodes, 10)
        for frontier, expected in EXPECTED_THRESHOLDS.items():
            with self.subTest(frontier=frontier):
                self.assertEqual(
                    curriculum.success_rate_threshold_for(frontier), expected)

    def test_initial_frontier_is_checkpoint_eleven_with_two_reconstructed_passes(self) -> None:
        env = self.training_env()
        env.rng.seed(42)

        for _ in range(100):
            observation = env.reset()
            self.assertEqual(self.checkpoint_index(env), 11)
            self.assertEqual(env._bot_passed, [True, True, False])
            self.assertEqual(env.overtakes, 2)
            self.assertEqual(env.episode_reward, 0.0)
            self.assertEqual(env.cause, "running")
            self.assertGreater(env.steps, 0)
            self.assertAlmostEqual(
                env._course_reward_potential_prev,
                env.course_reward_potential(),
            )
            self.assertAlmostEqual(float(observation[24]), 2.0 / 3.0)

    def test_sampling_is_eighty_percent_frontier_and_twenty_percent_mastered(self) -> None:
        env = self.training_env()
        first, unlocked = self.pass_frontier(env, 25)
        self.assertFalse(first["unlocked"])
        self.assertEqual(first["confirmation_streak_after"], 1)
        self.assertTrue(unlocked["unlocked"])
        self.assertEqual(unlocked["frontier_after"], 9)

        env.rng.seed(2026)
        starts = []
        for _ in range(4_000):
            env.reset()
            starts.append(self.checkpoint_index(env))
        self.assertEqual(set(starts), {9, 11})
        self.assertGreaterEqual(starts.count(9), 3_000)
        self.assertLessEqual(starts.count(9), 3_400)

        self.pass_frontier(env, 75)
        self.pass_frontier(env, 125)
        self.assertEqual(env.training_curriculum_state()["frontier"], 0)
        env.rng.seed(2027)
        starts = []
        for _ in range(8_000):
            env.reset()
            starts.append(self.checkpoint_index(env))
        self.assertEqual(set(starts), {0, 3, 9, 11})
        self.assertGreaterEqual(starts.count(0), 6_100)
        self.assertLessEqual(starts.count(0), 6_700)
        for mastered in (3, 9, 11):
            self.assertGreaterEqual(starts.count(mastered), 400)
            self.assertLessEqual(starts.count(mastered), 700)

    def test_gate_requires_two_distinct_checkpoints_and_ninety_percent_at_zero(self) -> None:
        env = self.training_env()
        curriculum = self.curriculum()

        below = env.record_training_curriculum_evaluation(
            0.79, curriculum, evaluation_episode=25)
        duplicate = env.record_training_curriculum_evaluation(
            1.0, curriculum, evaluation_episode=25)
        self.assertEqual(below["confirmation_streak_after"], 0)
        self.assertTrue(duplicate["ignored_duplicate"])
        self.assertEqual(duplicate["frontier_after"], 11)

        self.pass_frontier(env, 50)
        self.pass_frontier(env, 100)
        self.pass_frontier(env, 150)
        self.assertEqual(env.training_curriculum_state()["frontier"], 0)

        first_low = env.record_training_curriculum_evaluation(
            0.89, curriculum, evaluation_episode=200)
        second_low = env.record_training_curriculum_evaluation(
            0.89, curriculum, evaluation_episode=225)
        self.assertFalse(first_low["complete_after"])
        self.assertFalse(second_low["complete_after"])
        first_pass = env.record_training_curriculum_evaluation(
            0.9, curriculum, evaluation_episode=250)
        complete = env.record_training_curriculum_evaluation(
            0.9, curriculum, evaluation_episode=275)
        self.assertEqual(first_pass["confirmation_streak_after"], 1)
        self.assertTrue(complete["complete_after"])

    def test_fixed_stage_envs_are_physical_repeatable_and_grant_no_reset_reward(self) -> None:
        curriculum = self.curriculum()
        expected_masks = {
            11: [True, True, False],
            9: [True, False, False],
            3: [False, False, False],
            0: [False, False, False],
        }
        for frontier in EXPECTED_FRONTIERS:
            with self.subTest(frontier=frontier):
                first = curriculum.make_evaluation_env(frontier)
                second = curriculum.make_evaluation_env(frontier)
                seed = curriculum.evaluation_seed(frontier, 0)
                first.rng.seed(seed)
                second.rng.seed(seed)
                first_obs = first.reset()
                second_obs = second.reset()

                np.testing.assert_allclose(first_obs, second_obs)
                self.assertEqual(self.checkpoint_index(first), frontier)
                self.assertEqual(first._bot_passed, expected_masks[frontier])
                self.assertEqual(first.overtakes, sum(expected_masks[frontier]))
                self.assertEqual(first.episode_reward, 0.0)
                self.assertEqual(first.cause, "running")
                self.assertEqual(first.steps == 0, frontier == 0)

    def test_training_control_seeds_are_exact_and_disjoint(self) -> None:
        curriculum = self.curriculum()
        for frontier in EXPECTED_FRONTIERS:
            expected = list(range(
                700_000 + frontier * 1_000,
                700_010 + frontier * 1_000,
            ))
            self.assertEqual(
                [curriculum.evaluation_seed(frontier, i) for i in range(10)],
                expected,
            )
        all_control = {
            curriculum.evaluation_seed(frontier, i)
            for frontier in EXPECTED_FRONTIERS
            for i in range(10)
        }
        self.assertTrue(all_control.isdisjoint(range(100_000, 100_010)))
        self.assertTrue(all_control.isdisjoint(range(200_000, 200_100)))
        self.assertTrue(all_control.isdisjoint(range(600_000, 600_100)))

    def test_curriculum_state_round_trip_and_validation_are_exact(self) -> None:
        env = self.training_env()
        self.pass_frontier(env, 25)
        env.rng.seed(718)
        state = trainer_module.capture_rng_state(env)

        def sample_starts() -> list[tuple[int, tuple[bool, ...], int]]:
            samples = []
            for _ in range(40):
                env.reset()
                samples.append((
                    self.checkpoint_index(env), tuple(env._bot_passed), env.steps,
                ))
            return samples

        expected = sample_starts()
        self.pass_frontier(env, 75)
        trainer_module.restore_rng_state(state, env)
        self.assertEqual(env.training_curriculum_state(),
                         state["training_curriculum"])
        self.assertEqual(sample_starts(), expected)

        base = env.training_curriculum_state()
        invalid_states = []
        for key, value in (
            ("id", "other-curriculum"),
            ("version", 999),
            ("frontier", 3),
            ("mastered", [9]),
        ):
            invalid = copy.deepcopy(base)
            invalid[key] = value
            invalid_states.append(invalid)
        for invalid in invalid_states:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    env.restore_training_curriculum_state(invalid)

    def test_checkpoint_resume_restores_gate_and_pending_reset_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"traffic-rush"}')
            settings = Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            )
            first = trainer_module.Trainer(settings)
            self.pass_frontier(first.env, 25)
            first.env.rng.seed(991)
            first.episode = 50
            first.history = [{
                "episode": 50, "reward": -77.0, "steps": 10,
                "cause": "timeout", "metric": 1.0, "success": False,
            }]
            payload = {
                "reward": 0.0, "reward_std": 0.0, "metric": 0.0,
                "metric_std": 0.0, "failure_progress": 0.0, "episodes": 1,
                "success_rate": 0.0, "success_ci_low": 0.0,
                "success_ci_high": 1.0,
                "evaluation_suite": trainer_module.evaluation_suite_id(1),
                "seed": 42, "trajectory": [],
            }
            first._run_eval = lambda: copy.deepcopy(payload)
            first._run_training_curriculum_eval = lambda: None
            first._save_checkpoint()
            expected_state = first.env.training_curriculum_state()
            expected = []
            for _ in range(25):
                first.env.reset()
                expected.append((
                    self.checkpoint_index(first.env),
                    tuple(first.env._bot_passed),
                    first.env.steps,
                ))

            resumed = trainer_module.Trainer(settings)
            actual_state = resumed.env.training_curriculum_state()
            actual = []
            for _ in range(25):
                resumed.env.reset()
                actual.append((
                    self.checkpoint_index(resumed.env),
                    tuple(resumed.env._bot_passed),
                    resumed.env.steps,
                ))

        self.assertEqual(resumed.episode, 50)
        self.assertEqual(actual_state, expected_state)
        self.assertEqual(actual, expected)

    def test_canonical_checkpoint_selection_environment_is_unchanged(self) -> None:
        curriculum_env_type = getattr(driving, "TrafficCurriculumEnv", None)
        self.assertIsNotNone(curriculum_env_type)
        for env in (self.spec.make_env(True), self.spec.make_env(False)):
            self.assertIs(type(env), driving.DrivingEnv)
            self.assertNotIsInstance(env, curriculum_env_type)
            self.assertFalse(env.random_start)
            self.assertEqual(self.checkpoint_index(env), 0)
            self.assertEqual(env.steps, 0)
            self.assertEqual(env._bot_passed, [False, False, False])
            self.assertFalse(hasattr(env, "training_curriculum_state"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"traffic-rush"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=2,
            ))
            trainer.agent = ConstantAgent()
            before = trainer._run_eval()
            self.pass_frontier(trainer.env, 25)
            after = trainer._run_eval()

        self.assertEqual(after, before)
        self.assertEqual(before["evaluation_suite"], "policy-atlas-eval-v1-n2")
        self.assertNotIn("training_curriculum", before)

    def test_schema_protocol_and_diagnostics_disclose_the_exact_contract(self) -> None:
        curriculum = self.curriculum()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"traffic-rush"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            ))
            canonical = {
                "reward": -85.0, "reward_std": 0.0, "metric": 0.0,
                "metric_std": 0.0, "failure_progress": 0.0, "episodes": 1,
                "success_rate": 0.0, "success_ci_low": 0.0,
                "success_ci_high": 1.0, "evaluation_suite": "canonical-test",
                "seed": 42, "trajectory": [],
            }
            segment = {
                "frontier": 11, "episodes": 10, "successes": 8,
                "success_rate": 0.8,
                "evaluation_suite": "traffic-stage-eval-v1-k11-n10",
                "seeds": list(range(711_000, 711_010)),
            }
            self.complete_speed_control(trainer.env)
            trainer._run_eval = lambda: copy.deepcopy(canonical)
            trainer._run_training_control_eval = lambda: None
            trainer._run_training_curriculum_eval = lambda: copy.deepcopy(segment)
            trainer._save_checkpoint()
            meta = trainer.registry.list()[0]

        self.assertEqual(self.spec.checkpoint_schema, 18)
        self.assertEqual(meta["schema_version"], 18)
        self.assertEqual(meta["protocol"]["version"], 19)
        self.assertEqual(meta["protocol"]["training_curriculum"],
                         curriculum.protocol())
        self.assertEqual(
            meta["protocol"].get("course_reward_potential"),
            {
                "enabled": True,
                "state_potential": (
                    "0.05 * signed_progress + 3 * completed_checkpoints + "
                    "30 * completed_laps"
                ),
                "live_transition": "potential(next_state) - potential(state)",
                "terminal_potential": "retained physical end potential",
                "episode_sum": (
                    "potential(end_state) - potential(start_state)"
                ),
                "discount_factor": 1.0,
            },
        )
        diagnostic = meta["training_diagnostics"]["training_curriculum"]
        self.assertEqual(diagnostic["frontier_before"], 11)
        self.assertEqual(diagnostic["frontier_after"], 11)
        self.assertEqual(diagnostic["confirmation_streak_after"], 1)
        self.assertFalse(diagnostic["unlocked"])

    def test_other_driving_scenarios_keep_rng_curricula_and_schemas(self) -> None:
        curriculum_env_type = getattr(driving, "TrafficCurriculumEnv", None)
        self.assertIsNotNone(curriculum_env_type)
        expected_schemas = {
            "apex-gp": 7, "velocita": 7, "grandville": 7,
            "thunder-oval": 7, "apex-gp-wet": 9, "glacier": 7,
            "rally-ridge": 7, "kart-sprint": 7, "drift-trial": 8,
            "eco-gp": 7,
        }
        driving_specs = {spec.id: spec for spec in list_specs()
                         if spec.kind == "driving"}
        for scenario_id, expected_schema in expected_schemas.items():
            with self.subTest(scenario=scenario_id):
                spec = driving_specs[scenario_id]
                self.assertEqual(spec.checkpoint_schema, expected_schema)
                self.assertIsNone(spec.training_curriculum)
                self.assertNotIsInstance(spec.make_training_env(),
                                         curriculum_env_type)
                self.assertFalse(getattr(
                    spec.make_env(False).reward_cfg,
                    "terminal_zero_course_potential",
                    False,
                ))


if __name__ == "__main__":
    unittest.main()
