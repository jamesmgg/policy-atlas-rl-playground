"""Scientific contract for Traffic's nested cp11 bot-speed curriculum."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scenarios import get_spec  # noqa: E402
from app import trainer as trainer_module  # noqa: E402
from app.settings import Settings  # noqa: E402


SPEED_STAGES = ("speed-18", "speed-24", "speed-30")


class ConstantAgent:
    def __init__(self, action=(0.0, 0.0, 0.0)) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def select_action(self, observation, deterministic=False):
        del observation
        if not deterministic:
            raise AssertionError("control evaluation must be deterministic")
        return self.action.copy(), 0.0, 0.0


class TrafficV16ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_spec("traffic-rush")
        self.curriculum = self.spec.training_curriculum
        self.assertIsNotNone(self.curriculum)

    @staticmethod
    def checkpoint_index(env) -> int:
        return env.track.checkpoints.index(env.idx)

    def pass_speed_stage(self, env, first_episode: int) -> tuple[dict, dict]:
        control = self.curriculum.training_control
        first = env.record_training_control_evaluation(
            0.8, control, evaluation_episode=first_episode)
        second = env.record_training_control_evaluation(
            0.8, control, evaluation_episode=first_episode + 25)
        return first, second

    def build_hard_checkpoint_trainer(self, root: Path):
        (root / "state.json").write_text(
            '{"active_scenario":"traffic-rush"}')
        trainer = trainer_module.Trainer(Settings(
            port=8901, checkpoint_dir=root, checkpoint_every_n=25,
            max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
        ))
        self.pass_speed_stage(trainer.env, 25)
        self.pass_speed_stage(trainer.env, 75)
        trainer.env.record_training_control_evaluation(
            0.8, self.curriculum.training_control,
            evaluation_episode=125,
        )
        trainer.episode = 150
        trainer.history = [{
            "episode": 150, "reward": -69.0, "steps": 100,
            "cause": "timeout", "metric": 2.0, "success": False,
        }]
        trainer._run_eval = lambda: {
            "reward": -85.0, "reward_std": 0.0, "metric": 0.0,
            "metric_std": 0.0, "failure_progress": 0.0, "episodes": 1,
            "success_rate": 0.0, "success_ci_low": 0.0,
            "success_ci_high": 1.0, "evaluation_suite": "canonical-test",
            "seed": 42, "trajectory": [],
        }
        trainer._run_training_control_eval = lambda: {
            "stage": "speed-30", "episodes": 10, "successes": 8,
            "success_rate": 0.8,
            "evaluation_suite": (
                "traffic-cp11-bot3-speed-control-v1-speed-30-n10"),
            "seeds": list(range(802_000, 802_010)),
        }
        trainer._run_training_curriculum_eval = lambda: {
            "frontier": 11, "episodes": 10, "successes": 8,
            "success_rate": 0.8,
            "evaluation_suite": "traffic-stage-eval-v1-k11-n10",
            "seeds": list(range(711_000, 711_010)),
        }
        return trainer

    def test_nested_control_contract_and_initial_cp11_stage(self) -> None:
        control = getattr(self.curriculum, "training_control", None)
        self.assertIsNotNone(control)
        self.assertEqual(control.stage_ids, SPEED_STAGES)
        self.assertEqual(control.active_stage_probability, 0.8)
        self.assertEqual(control.success_rate_threshold, 0.8)
        self.assertEqual(control.consecutive_confirmations, 2)
        self.assertEqual(control.evaluation_episodes, 10)

        env = self.spec.make_training_env()
        state = env.training_control_state()
        self.assertEqual(state["stage"], "speed-18")
        self.assertEqual(state["speed_mps"], 18.0)
        for _ in range(30):
            env.reset_for_training_episode(401)
            self.assertEqual(self.checkpoint_index(env), 11)
            self.assertEqual(env.features.bots[2].speed, 18.0)
            self.assertEqual(env._bot_passed, [True, True, False])
            self.assertEqual(env.episode_reward, 0.0)

    def test_speed_gate_requires_two_distinct_confirmations_and_rehearses(self) -> None:
        env = self.spec.make_training_env()
        control = self.curriculum.training_control
        record = getattr(env, "record_training_control_evaluation", None)
        self.assertTrue(callable(record))

        failed = record(0.79, control, evaluation_episode=25)
        first = record(0.8, control, evaluation_episode=50)
        duplicate = record(1.0, control, evaluation_episode=50)
        unlocked = record(0.8, control, evaluation_episode=75)
        self.assertFalse(failed["passed"])
        self.assertEqual(first["confirmation_streak_after"], 1)
        self.assertTrue(duplicate["ignored_duplicate"])
        self.assertEqual(duplicate["confirmation_streak_after"], 1)
        self.assertTrue(unlocked["unlocked"])
        self.assertEqual(unlocked["stage_after"], "speed-24")

        env.rng.seed(2026)
        speeds = []
        for _ in range(4_000):
            env.reset_for_training_episode(401)
            speeds.append(env.features.bots[2].speed)
        self.assertEqual(set(speeds), {18.0, 24.0})
        self.assertGreaterEqual(speeds.count(24.0), 3_000)
        self.assertLessEqual(speeds.count(24.0), 3_400)

        record(0.8, control, evaluation_episode=100)
        bridge = record(0.8, control, evaluation_episode=125)
        self.assertTrue(bridge["unlocked"])
        self.assertEqual(bridge["stage_after"], "speed-30")

        env.rng.seed(2027)
        speeds = []
        for _ in range(8_000):
            env.reset_for_training_episode(401)
            speeds.append(env.features.bots[2].speed)
        self.assertEqual(set(speeds), {18.0, 24.0, 30.0})
        self.assertGreaterEqual(speeds.count(30.0), 6_100)
        self.assertLessEqual(speeds.count(30.0), 6_700)
        for mastered in (18.0, 24.0):
            self.assertGreaterEqual(speeds.count(mastered), 650)
            self.assertLessEqual(speeds.count(mastered), 950)

        record(0.8, control, evaluation_episode=150)
        completed = record(0.8, control, evaluation_episode=175)
        self.assertTrue(completed["completed"])
        self.assertTrue(env.training_control_state()["complete"])
        for _ in range(100):
            env.reset()
            self.assertEqual(env.features.bots[2].speed, 30.0)

    def test_fixed_speed_stages_reconstruct_physics_and_improve_catchability(self) -> None:
        control = self.curriculum.training_control
        probe = control.make_evaluation_env("speed-18")
        self.assertTrue(callable(getattr(
            probe, "bot3_catchup_diagnostics", None)))
        expected = {
            "speed-18": (18.0, 319.4233, 24.5671),
            "speed-24": (24.0, 567.5833, 35.6691),
            "speed-30": (30.0, 815.7433, 46.7710),
        }
        diagnostics = []
        for stage, (speed, gap, required_speed) in expected.items():
            with self.subTest(stage=stage):
                seed = control.evaluation_seed(stage, 0)
                first = control.make_evaluation_env(stage)
                second = control.make_evaluation_env(stage)
                first.rng.seed(seed)
                second.rng.seed(seed)
                first_obs = first.reset()
                second_obs = second.reset()
                np.testing.assert_allclose(first_obs, second_obs)

                diagnostic = first.bot3_catchup_diagnostics()
                diagnostics.append(diagnostic)
                self.assertEqual(self.checkpoint_index(first), 11)
                self.assertEqual(first.features.bots[2].speed, speed)
                self.assertEqual(first._bot_passed, [True, True, False])
                self.assertEqual(first.episode_reward, 0.0)
                self.assertAlmostEqual(diagnostic["gap_m"], gap, places=3)
                self.assertAlmostEqual(
                    diagnostic["remaining_seconds"], 48.64, places=2)
                self.assertAlmostEqual(
                    diagnostic["required_average_speed_mps"],
                    required_speed,
                    places=3,
                )

                elapsed = first.steps * first.dt
                bot = first.features.bots[2]
                expected_unwrapped = (
                    bot.start_frac * first.track.total_length
                    + speed * elapsed
                )
                self.assertAlmostEqual(
                    first._bot_arcs[2],
                    expected_unwrapped % first.track.total_length,
                )
                self.assertEqual(
                    first._bot_passed[2],
                    first.progress > expected_unwrapped,
                )

        gaps = [item["gap_m"] for item in diagnostics]
        required = [item["required_average_speed_mps"]
                    for item in diagnostics]
        self.assertEqual(gaps, sorted(gaps))
        self.assertEqual(required, sorted(required))
        self.assertGreater(required[-1] - required[0], 22.0)
        self.assertLess(required[0], required[-1] * 0.55)

    def test_outer_cp11_gate_is_blocked_until_hard_speed_proficiency(self) -> None:
        env = self.spec.make_training_env()
        first = env.record_training_curriculum_evaluation(
            1.0, self.curriculum, evaluation_episode=25)
        self.assertIn("prerequisite_met", first)
        second = env.record_training_curriculum_evaluation(
            1.0, self.curriculum, evaluation_episode=50)
        self.assertTrue(first["passed"] and second["passed"])
        self.assertFalse(first["prerequisite_met"])
        self.assertFalse(second["prerequisite_met"])
        self.assertFalse(first["unlocked"] or second["unlocked"])
        self.assertEqual(env.training_curriculum_state()["frontier"], 11)
        self.assertEqual(env.training_curriculum_state()["pass_streak"], 0)

        self.pass_speed_stage(env, 75)
        self.pass_speed_stage(env, 125)
        self.pass_speed_stage(env, 175)
        self.assertTrue(env.training_control_state()["complete"])

        canonical_first = env.record_training_curriculum_evaluation(
            0.8, self.curriculum, evaluation_episode=225)
        canonical_second = env.record_training_curriculum_evaluation(
            0.8, self.curriculum, evaluation_episode=250)
        self.assertTrue(canonical_first["prerequisite_met"])
        self.assertFalse(canonical_first["unlocked"])
        self.assertTrue(canonical_second["unlocked"])
        self.assertEqual(canonical_second["frontier_after"], 9)

        env.rng.seed(2028)
        checkpoints = set()
        for _ in range(1_000):
            env.reset()
            checkpoints.add(self.checkpoint_index(env))
            self.assertEqual(
                env.features.bots[2].speed,
                30.0,
                "nested speeds must not leak into cp9 or mastered cp11 resets",
            )
        self.assertEqual(checkpoints, {9, 11})

    def test_nested_state_and_rng_restore_are_exact_and_cross_layer_atomic(self) -> None:
        env = self.spec.make_training_env()
        self.pass_speed_stage(env, 25)
        env.rng.seed(991)
        outer_state = env.training_curriculum_state()
        self.assertIn("bot3_speed", outer_state)

        snapshot = trainer_module.capture_rng_state(env)

        def sample_resets() -> list[tuple[float, float, int]]:
            samples = []
            for _ in range(40):
                env.reset_for_training_episode(401)
                samples.append((
                    env.features.bots[2].speed,
                    env._bot_gap(2),
                    env.steps,
                ))
            return samples

        expected = sample_resets()
        self.pass_speed_stage(env, 75)
        trainer_module.restore_rng_state(snapshot, env)
        self.assertEqual(
            env.training_curriculum_state(), snapshot["training_curriculum"])
        self.assertEqual(sample_resets(), expected)

        before = copy.deepcopy(env.training_curriculum_state())
        invalid = copy.deepcopy(before)
        invalid.update({
            "frontier_position": 1,
            "frontier": 9,
            "mastered": [11],
        })
        with self.assertRaisesRegex(ValueError, "30 m/s proficiency"):
            env.restore_training_curriculum_state(invalid)
        self.assertEqual(env.training_curriculum_state(), before)

    def test_malformed_nested_restore_is_rejected_without_partial_mutation(self) -> None:
        env = self.spec.make_training_env()
        self.pass_speed_stage(env, 25)
        before = copy.deepcopy(env.training_curriculum_state())
        invalid_states = []
        invalid_mastered = copy.deepcopy(before)
        invalid_mastered["bot3_speed"]["mastered"] = None
        invalid_states.append(invalid_mastered)
        impossible_streak = copy.deepcopy(before)
        impossible_streak["bot3_speed"]["pass_streak"] = 2
        invalid_states.append(impossible_streak)

        for invalid in invalid_states:
            with self.subTest(invalid=invalid):
                env.restore_training_curriculum_state(before)
                caught = None
                try:
                    env.restore_training_curriculum_state(invalid)
                except Exception as exc:
                    caught = exc

                self.assertIsInstance(caught, ValueError)
                self.assertEqual(env.training_curriculum_state(), before)

    def test_control_suites_use_exact_disjoint_seeds_and_defer_outer_gate(self) -> None:
        evaluate = getattr(trainer_module, "evaluate_training_control", None)
        self.assertTrue(callable(evaluate))
        control = self.curriculum.training_control
        expected_ranges = {
            "speed-18": range(800_000, 800_010),
            "speed-24": range(801_000, 801_010),
            "speed-30": range(802_000, 802_010),
        }
        nested_seeds = set()
        for stage, expected in expected_ranges.items():
            seeds = [control.evaluation_seed(stage, i) for i in range(10)]
            self.assertEqual(seeds, list(expected))
            nested_seeds.update(seeds)
        outer_seeds = {
            self.curriculum.evaluation_seed(frontier, i)
            for frontier in self.curriculum.frontier_order
            for i in range(self.curriculum.evaluation_episodes)
        }
        self.assertTrue(nested_seeds.isdisjoint(outer_seeds))
        self.assertTrue(nested_seeds.isdisjoint(range(100_000, 100_100)))
        self.assertTrue(nested_seeds.isdisjoint(range(600_000, 600_100)))

        first_env = self.spec.make_training_env()
        second_env = self.spec.make_training_env()
        first = evaluate(control, first_env, ConstantAgent())
        second = evaluate(control, second_env, ConstantAgent())
        self.assertEqual(first, second)
        self.assertEqual(first["stage"], "speed-18")
        self.assertEqual(
            first["evaluation_suite"],
            "traffic-cp11-bot3-speed-control-v1-speed-18-n10",
        )
        self.assertEqual(first["seeds"], list(range(800_000, 800_010)))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"traffic-rush"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            ))
            trainer.agent = ConstantAgent()
            self.assertIsNotNone(trainer._run_training_control_eval())
            self.assertIsNone(trainer._run_training_curriculum_eval())
            self.pass_speed_stage(trainer.env, 25)
            self.pass_speed_stage(trainer.env, 75)
            self.pass_speed_stage(trainer.env, 125)
            self.assertIsNone(trainer._run_training_control_eval())
            outer = trainer._run_training_curriculum_eval()

        self.assertIsNotNone(outer)
        self.assertEqual(outer["frontier"], 11)
        self.assertEqual(outer["seeds"], list(range(711_000, 711_010)))

    def test_checkpoint_atomically_records_control_then_outer_gate_and_protocol(self) -> None:
        self.assertEqual(self.spec.checkpoint_schema, 18)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.build_hard_checkpoint_trainer(Path(tmp))
            trainer._save_checkpoint()
            meta = trainer.registry.list()[0]
            payload = trainer.registry.load(150)

        self.assertEqual(meta["schema_version"], 18)
        self.assertEqual(meta["protocol"]["version"], 18)
        self.assertEqual(
            meta["protocol"]["training_control"],
            self.curriculum.training_control.protocol(),
        )
        self.assertEqual(
            meta["protocol"]["training_curriculum"],
            self.curriculum.protocol(),
        )
        control = meta["training_diagnostics"]["training_control"]
        outer = meta["training_diagnostics"]["training_curriculum"]
        self.assertTrue(control["completed"])
        self.assertTrue(control["state_after"]["complete"])
        self.assertEqual(outer["frontier_before"], 11)
        self.assertEqual(outer["frontier_after"], 11)
        self.assertTrue(outer["prerequisite_met"])
        self.assertEqual(outer["confirmation_streak_after"], 1)
        saved = payload["rng_state"]["training_curriculum"]
        self.assertTrue(saved["bot3_speed"]["complete"])
        self.assertEqual(saved["pass_streak"], 1)

    def test_failed_checkpoint_persistence_rolls_back_both_gate_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.build_hard_checkpoint_trainer(Path(tmp))
            before = copy.deepcopy(trainer.env.training_curriculum_state())

            def fail_save(*args, **kwargs):
                del args, kwargs
                raise OSError("persistence failed")

            trainer.registry.save = fail_save
            with self.assertRaisesRegex(OSError, "persistence failed"):
                trainer._save_checkpoint()

        self.assertEqual(trainer.env.training_curriculum_state(), before)


if __name__ == "__main__":
    unittest.main()
