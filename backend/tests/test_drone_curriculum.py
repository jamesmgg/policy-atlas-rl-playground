"""Scientific contract tests for the Drone waypoint-start curriculum."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1]))

from app.envs import drone
from app.scenarios import get_spec


class TestDroneWaypointCurriculum(unittest.TestCase):
    def test_training_factory_mixes_full_course_and_later_segments(self) -> None:
        spec = get_spec("drone-hover")
        training = spec.make_training_env()

        self.assertTrue(
            getattr(training, "waypoint_start_curriculum", False),
            "Drone training must opt into waypoint-segment resets",
        )

        training.rng.seed(42)
        starts: list[int] = []
        for _ in range(800):
            training.reset()
            starts.append(training.k)

            if training.k == 0:
                origin = drone.START
            else:
                origin = drone.WAYPOINTS[training.k - 1]
            self.assertLessEqual(abs(training.x - origin[0]), 30.0)
            self.assertEqual(training.y, origin[1])

        canonical_count = starts.count(0)
        self.assertGreaterEqual(canonical_count, 320)
        self.assertLessEqual(canonical_count, 480)
        self.assertEqual(set(starts), set(range(len(drone.WAYPOINTS))))

    def test_evaluation_factories_always_start_the_full_course(self) -> None:
        spec = get_spec("drone-hover")
        fixed_suite = spec.make_env(True)
        canonical = spec.make_env(False)

        self.assertFalse(getattr(fixed_suite, "waypoint_start_curriculum", False))
        self.assertFalse(getattr(canonical, "waypoint_start_curriculum", False))

        for env in (fixed_suite, canonical):
            env.rng.seed(7)
            for _ in range(100):
                observation = env.reset()
                self.assertEqual(env.k, 0)
                self.assertEqual(env.y, drone.START[1])
                self.assertEqual(float(observation[7]), 0.0)

        self.assertEqual((canonical.x, canonical.y), drone.START)

    def test_segment_reset_exposes_complete_task_and_clock_state(self) -> None:
        training = get_spec("drone-hover").make_training_env()
        self.assertTrue(getattr(training, "waypoint_start_curriculum", False))
        training.rng.seed(9)

        observation = training.reset()
        for _ in range(100):
            if training.k > 0:
                break
            observation = training.reset()

        self.assertGreater(training.k, 0, "seed must exercise a segment reset")
        target_x, target_y = drone.WAYPOINTS[training.k]
        self.assertAlmostEqual(float(observation[0]),
                               (target_x - training.x) / 300.0, places=6)
        self.assertAlmostEqual(float(observation[1]),
                               (target_y - training.y) / 300.0, places=6)
        self.assertAlmostEqual(float(observation[7]),
                               training.k / len(drone.WAYPOINTS), places=6)
        self.assertEqual(float(observation[-1]), 1.0)
        self.assertEqual(training.steps, 0)
        self.assertEqual(training.episode_reward, 0.0)
        self.assertEqual(training.cause, "running")

    def test_start_source_does_not_change_transition_or_reward(self) -> None:
        curriculum = get_spec("drone-hover").make_training_env()
        canonical = get_spec("drone-hover").make_env(True)
        curriculum.rng.seed(17)
        for _ in range(100):
            curriculum.reset()
            if curriculum.k > 0:
                break
        else:
            self.fail("seeded curriculum did not produce a segment start")

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

    def test_curriculum_and_schema_are_explicit_in_the_scenario_contract(self) -> None:
        spec = get_spec("drone-hover")

        self.assertEqual(
            getattr(spec, "training_start_distribution", None),
            "50% canonical full-course start; 50% uniform later waypoint "
            "segments (targets 2-5) from the preceding waypoint",
        )
        self.assertEqual(spec.checkpoint_schema, 3)


if __name__ == "__main__":
    unittest.main()
