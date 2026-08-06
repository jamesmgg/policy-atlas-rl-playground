from __future__ import annotations

import copy
import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.envs import driving  # noqa: E402
from app.scenarios.registry import list_specs  # noqa: E402
from app.track import heading_at  # noqa: E402

DT_AGENT = driving.DT_AGENT
ROLLING_HEADING_JITTER = getattr(driving, "ROLLING_HEADING_JITTER", 0.0)
ROLLING_LATERAL_JITTER_FRACTION = getattr(
    driving, "ROLLING_LATERAL_JITTER_FRACTION", 0.0)
ROLLING_LATERAL_JITTER_MAX = getattr(
    driving, "ROLLING_LATERAL_JITTER_MAX", 0.0)
ROLLING_REFERENCE_THROTTLE = getattr(
    driving, "ROLLING_REFERENCE_THROTTLE", 0.0)
ROLLING_SPEED_MAX_FRACTION = getattr(
    driving, "ROLLING_SPEED_MAX_FRACTION", 0.0)
STYLE_TARGET = getattr(driving, "STYLE_TARGET", 10.0)
HAS_ROLLING_CURRICULUM = all(hasattr(driving, name) for name in (
    "ROLLING_HEADING_JITTER",
    "ROLLING_LATERAL_JITTER_FRACTION",
    "ROLLING_LATERAL_JITTER_MAX",
    "ROLLING_REFERENCE_THROTTLE",
    "ROLLING_SPEED_MAX_FRACTION",
    "STYLE_TARGET",
)) and hasattr(driving.DrivingEnv, "rolling_speed_limit")


class DrivingCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = {spec.id: spec for spec in list_specs()}

    def rolling_start(self, scenario_id: str, seed: int = 7):
        env = self.specs[scenario_id].make_training_env()
        env.start_line_probability = 0.0
        env.rng.seed(seed)
        observation = env.reset()
        self.assertNotEqual(env.idx, 0, "seed must exercise a rolling start")
        return env, observation

    def test_rolling_curriculum_contract_exists(self) -> None:
        self.assertTrue(
            HAS_ROLLING_CURRICULUM,
            "DrivingEnv has no dynamically plausible rolling-start contract",
        )

    @unittest.skipUnless(HAS_ROLLING_CURRICULUM, "curriculum API not implemented")
    def test_fixed_evaluation_keeps_the_canonical_start_and_rng_sequence(self) -> None:
        for jitter in (False, True):
            with self.subTest(jitter=jitter):
                env = self.specs["rally-ridge"].make_env(jitter)
                env.rng.seed(31)
                observation = env.reset()

                expected_rng = random.Random(31)
                expected_heading = heading_at(env.track, 0)
                expected_x, expected_y = map(float, env.track.centerline[0])
                if jitter:
                    expected_heading += expected_rng.uniform(-0.05, 0.05)
                    nx, ny = env.track.normals[0]
                    offset = expected_rng.uniform(-3.0, 3.0)
                    expected_x += float(nx) * offset
                    expected_y += float(ny) * offset

                self.assertEqual(env.idx, 0)
                self.assertAlmostEqual(env.car.x, expected_x)
                self.assertAlmostEqual(env.car.y, expected_y)
                self.assertAlmostEqual(env.car.heading, expected_heading)
                self.assertEqual(env.car.v_long, 0.0)
                self.assertEqual(env.progress, 0.0)
                self.assertEqual(env.next_cp, 1)
                self.assertEqual(env.steps, 0)
                self.assertEqual(env.laps, 0)
                self.assertTrue(np.array_equal(observation, env._obs()))

    @unittest.skipUnless(HAS_ROLLING_CURRICULUM, "curriculum API not implemented")
    def test_traffic_fixed_evaluation_remains_canonical(self) -> None:
        spec = self.specs["traffic-rush"]
        self.assertEqual(spec.checkpoint_schema, 10)

        env = spec.make_env(False)
        env.rng.seed(41)
        observation = env.reset()

        self.assertFalse(env.random_start)
        self.assertEqual(env.idx, 0)
        self.assertEqual(env.car.v_long, 0.0)
        self.assertEqual(env.progress, 0.0)
        self.assertEqual(env.steps, 0)
        self.assertEqual(env._bot_passed, [False, False, False])
        self.assertTrue(np.array_equal(observation, env._obs()))

    @unittest.skipUnless(HAS_ROLLING_CURRICULUM, "curriculum API not implemented")
    def test_traffic_training_uses_only_audited_rolling_checkpoints(self) -> None:
        env = self.specs["traffic-rush"].make_training_env()
        self.assertEqual(env.start_line_probability, 0.75)
        env.start_line_probability = 0.0
        env.rng.seed(42)

        sampled = set()
        for _ in range(240):
            env.reset()
            sampled.add(env.track.checkpoints.index(env.idx))

        self.assertEqual(sampled, {4, 10, 11})

    @unittest.skipUnless(HAS_ROLLING_CURRICULUM, "curriculum API not implemented")
    def test_generic_driving_training_still_uses_every_rolling_checkpoint(self) -> None:
        env = self.specs["rally-ridge"].make_training_env()
        env.start_line_probability = 0.0
        env.rng.seed(42)

        sampled = set()
        for _ in range(480):
            env.reset()
            sampled.add(env.track.checkpoints.index(env.idx))

        self.assertEqual(sampled, set(range(1, len(env.track.checkpoints))))

    @unittest.skipUnless(HAS_ROLLING_CURRICULUM, "curriculum API not implemented")
    def test_rolling_start_reconstructs_an_observed_remaining_lap_state(self) -> None:
        env, observation = self.rolling_start("rally-ridge")
        spec = self.specs["rally-ridge"]
        dimensions = list(spec.observation_dimensions)
        checkpoint_index = env.track.checkpoints.index(env.idx)
        checkpoint_progress = env.track.checkpoint_arcs[checkpoint_index]
        speed_limit = env.rolling_speed_limit(env.idx)

        self.assertGreater(env.car.v_long, 0.0)
        self.assertLessEqual(
            env.car.v_long,
            speed_limit * ROLLING_SPEED_MAX_FRACTION + 1e-9,
        )
        self.assertAlmostEqual(env.progress, checkpoint_progress)
        self.assertAlmostEqual(env.peak_progress, checkpoint_progress)
        self.assertAlmostEqual(env._stall_anchor_progress, checkpoint_progress)
        self.assertEqual(env.next_cp, checkpoint_index + 1)
        self.assertGreater(env.steps, 0)
        self.assertEqual(env.lap_start_step, 0)

        immediate_threshold = (
            env.track.checkpoint_arcs[checkpoint_index + 1]
            if checkpoint_index + 1 < len(env.track.checkpoint_arcs)
            else env.track.total_length
        )
        self.assertAlmostEqual(env._cp_threshold(), immediate_threshold)
        remaining = env.track.total_length - checkpoint_progress
        self.assertLess(remaining, env.track.total_length)
        original_next_cp = env.next_cp
        env.next_cp = len(env.track.checkpoint_arcs)
        self.assertAlmostEqual(env._cp_threshold() - env.progress, remaining)
        env.next_cp = original_next_cp
        self.assertLessEqual(
            env.steps + env.rolling_remaining_steps(env.idx),
            env.max_steps,
            "clock reconstruction must leave enough reference time to finish",
        )

        expected_next = env._cp_threshold() / env.track.total_length
        self.assertAlmostEqual(
            float(observation[dimensions.index(
                "next checkpoint course progress / lap length")]),
            expected_next,
            places=5,
        )
        self.assertAlmostEqual(
            float(observation[dimensions.index(
                "forward course progress / lap length")]),
            checkpoint_progress / env.track.total_length,
            places=5,
        )
        self.assertAlmostEqual(
            float(observation[dimensions.index("objective completion fraction")]),
            checkpoint_progress / env.track.total_length,
            places=5,
        )
        self.assertAlmostEqual(
            float(observation[dimensions.index("remaining horizon fraction")]),
            1.0 - env.steps / env.max_steps,
            places=5,
        )

        _, lateral_offset = env.track.localize(env.car.x, env.car.y, env.idx)
        lateral_limit = min(
            ROLLING_LATERAL_JITTER_MAX,
            float(env.track.half_widths[env.idx])
            * ROLLING_LATERAL_JITTER_FRACTION,
        )
        self.assertLessEqual(abs(lateral_offset), lateral_limit + 1e-6)
        tangent = heading_at(env.track, env.idx)
        heading_error = math.atan2(
            math.sin(env.car.heading - tangent),
            math.cos(env.car.heading - tangent),
        )
        self.assertLessEqual(abs(heading_error), ROLLING_HEADING_JITTER + 1e-9)

    @unittest.skipUnless(HAS_ROLLING_CURRICULUM, "curriculum API not implemented")
    def test_curriculum_provenance_does_not_change_a_twin_state_transition(self) -> None:
        rolling, _ = self.rolling_start("rally-ridge", seed=23)
        self.assertEqual(rolling.episode_reward, 0.0)
        twin = copy.deepcopy(rolling)
        twin.random_start = False
        action = np.array([0.35, -0.1, 0.0], dtype=np.float32)

        rolling_result = rolling.step(action)
        twin_result = twin.step(action)

        self.assertTrue(np.array_equal(rolling_result[0], twin_result[0]))
        self.assertEqual(rolling_result[1:], twin_result[1:])

    @unittest.skipUnless(HAS_ROLLING_CURRICULUM, "curriculum API not implemented")
    def test_every_driving_rolling_speed_respects_curvature_grip_braking_envelope(self) -> None:
        driving_specs = [spec for spec in self.specs.values()
                         if spec.kind == "driving"]
        for spec in driving_specs:
            env = spec.make_training_env()
            env.start_line_probability = 0.0
            env.rng.seed(103)
            for _ in range(32):
                env.reset()
                with self.subTest(scenario=spec.id, checkpoint=env.idx):
                    self.assertIn(env.idx, env.track.checkpoints[1:])
                    self.assertGreater(env.car.v_long, 0.0)
                    self.assertLessEqual(
                        env.car.v_long,
                        env.rolling_speed_limit(env.idx)
                        * ROLLING_SPEED_MAX_FRACTION + 1e-9,
                    )
                    curvature = abs(float(env.track.curvature[env.idx]))
                    if curvature > 1e-9:
                        lateral_cap = (
                            env.params.a_lat_grip_normal * env._grip[env.idx]
                        )
                        self.assertLessEqual(
                            env.rolling_speed_limit(env.idx) ** 2 * curvature,
                            lateral_cap + 1e-6,
                        )

    @unittest.skipUnless(HAS_ROLLING_CURRICULUM, "curriculum API not implemented")
    def test_fuel_style_and_traffic_resume_from_observed_consistent_state(self) -> None:
        eco, eco_observation = self.rolling_start("eco-gp", seed=11)
        elapsed = eco.steps * DT_AGENT
        expected_fuel = max(
            1.0
            - eco.features.fuel.rate
            * ROLLING_REFERENCE_THROTTLE ** 2
            * elapsed,
            0.0,
        )
        self.assertAlmostEqual(eco.fuel, expected_fuel)
        eco_dimensions = list(self.specs["eco-gp"].observation_dimensions)
        self.assertAlmostEqual(
            float(eco_observation[eco_dimensions.index("fuel fraction")]),
            eco.fuel,
            places=6,
        )

        drift, drift_observation = self.rolling_start("drift-trial", seed=13)
        expected_style = STYLE_TARGET * drift.progress / drift.track.total_length
        self.assertAlmostEqual(drift.style, expected_style)
        drift_dimensions = list(
            self.specs["drift-trial"].observation_dimensions)
        self.assertAlmostEqual(
            float(drift_observation[drift_dimensions.index(
                "objective completion fraction")]),
            expected_style / STYLE_TARGET,
            places=6,
        )

        traffic, traffic_observation = self.rolling_start(
            "traffic-rush", seed=17)
        elapsed = traffic.steps * DT_AGENT
        for index, bot in enumerate(traffic.features.bots):
            unwrapped_bot = (
                bot.start_frac * traffic.track.total_length
                + bot.speed * elapsed
            )
            expected_arc = unwrapped_bot % traffic.track.total_length
            expected_passed = traffic.progress > unwrapped_bot
            with self.subTest(bot=index):
                self.assertAlmostEqual(traffic._bot_arcs[index], expected_arc)
                self.assertEqual(traffic._bot_passed[index], expected_passed)
        self.assertEqual(traffic.overtakes, sum(traffic._bot_passed))
        traffic_dimensions = list(
            self.specs["traffic-rush"].observation_dimensions)
        for index, passed in enumerate(traffic._bot_passed):
            self.assertEqual(
                float(traffic_observation[traffic_dimensions.index(
                    f"traffic {index + 1} already passed")]),
                float(passed),
            )

    @unittest.skipUnless(HAS_ROLLING_CURRICULUM, "curriculum API not implemented")
    def test_generic_scenario_metadata_discloses_the_exact_curriculum(self) -> None:
        spec = self.specs["rally-ridge"]
        disclosure = spec.training_start_distribution
        self.assertEqual(spec.info()["training_start_distribution"], disclosure)
        self.assertEqual(
            disclosure,
            "75% canonical start; 25% uniform checkpoints 1..N-1 as "
            "rolling states at 70-90% of the curvature/grip backward-braking "
            "envelope; clock integrates an 80% envelope with a 1-second "
            "reserve; no reset reward",
        )

    def test_traffic_protocol_discloses_the_exact_curriculum(self) -> None:
        spec = self.specs["traffic-rush"]
        disclosure = spec.training_start_distribution
        self.assertEqual(spec.info()["training_start_distribution"], disclosure)
        self.assertEqual(
            disclosure,
            "75% canonical start; 25% uniform checkpoints 4,10,11 as rolling "
            "states at 70-90% of the curvature/grip backward-braking "
            "envelope; clock integrates an 80% envelope with a 1-second "
            "reserve; time-advanced traffic and pass masks; no reset reward",
        )


if __name__ == "__main__":
    unittest.main()
