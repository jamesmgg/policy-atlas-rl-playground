"""Invariant scientific tests shared by every Lander curriculum revision."""
from __future__ import annotations

import unittest

import numpy as np

from app.envs import lander
from app.scenarios import list_specs


class TestLanderScientificContract(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = {spec.id: spec for spec in list_specs()}["lunar-lander"]

    def test_training_factory_does_not_change_evaluation_starts(self) -> None:
        training = self.spec.make_training_env()
        randomized_evaluation = self.spec.make_env(True)
        canonical_evaluation = self.spec.make_env(False)

        self.assertTrue(training.approach_curriculum)
        self.assertFalse(randomized_evaluation.approach_curriculum)
        self.assertFalse(canonical_evaluation.approach_curriculum)
        self.assertIsNone(randomized_evaluation.forced_start_frontier)
        self.assertIsNone(canonical_evaluation.forced_start_frontier)
        self.assertEqual(
            self.spec.training_start_distribution,
            lander.TRAINING_START_DISTRIBUTION,
        )

        randomized_evaluation.rng.seed(2026)
        for _ in range(40):
            randomized_evaluation.reset()
            self.assertEqual(randomized_evaluation._start_kind, "standard")
            self.assertEqual(randomized_evaluation._start_frontier, 0)
            self.assertEqual(randomized_evaluation.x, 500.0)
            self.assertEqual(randomized_evaluation.y, 120.0)
            self.assertEqual(randomized_evaluation.vy, 0.0)
            self.assertEqual(randomized_evaluation.omega, 0.0)
            self.assertEqual(randomized_evaluation.fuel, 1.0)
            self.assertEqual(randomized_evaluation.steps, 0)

        canonical = canonical_evaluation.reset()
        np.testing.assert_allclose(
            canonical,
            np.array([0.0, -5.0 / 3.0, 0.0, 0.0, 0.0, 1.0,
                      0.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_altitude_frontiers_overlap_from_low_approach_to_canonical(self) -> None:
        self.assertEqual(lander.TOUCHDOWN_ALTITUDE_MIN, 5.0)
        self.assertEqual(lander.TOUCHDOWN_ALTITUDE_MAX, 18.0)
        self.assertEqual(lander.APPROACH_FRONTIER_ALTITUDES, {
            3: (30.0, 100.0),
            2: (50.0, 200.0),
            1: (100.0, lander.PAD_Y - 120.0),
        })
        self.assertLess(
            lander.APPROACH_FRONTIER_ALTITUDES[2][0],
            lander.APPROACH_FRONTIER_ALTITUDES[3][1],
        )
        self.assertLess(
            lander.APPROACH_FRONTIER_ALTITUDES[1][0],
            lander.APPROACH_FRONTIER_ALTITUDES[2][1],
        )
        self.assertEqual(lander.APPROACH_ALTITUDE_MIN, 30.0)
        self.assertEqual(lander.APPROACH_ALTITUDE_MAX,
                         lander.PAD_Y - 120.0)

    def test_every_rehearsal_exposes_clock_fuel_and_dynamic_state(self) -> None:
        for frontier in (4, 3, 2, 1):
            with self.subTest(frontier=frontier):
                env = lander.make_frontier_evaluation_env(frontier)
                env.rng.seed(20260805 + frontier)
                observation = env.reset()
                descent_fraction = (env.y - 120.0) / (lander.PAD_Y - 120.0)
                self.assertEqual(env.steps, round(180.0 * descent_fraction))
                self.assertGreaterEqual(
                    env.fuel, 1.0 - 0.5 * descent_fraction)
                self.assertLessEqual(
                    env.fuel, 1.0 - 0.1 * descent_fraction)
                np.testing.assert_allclose(observation, np.array([
                    (env.x - lander.PAD_CX) / 300.0,
                    (env.y - lander.PAD_Y) / 300.0,
                    env.vx / 60.0,
                    env.vy / 60.0,
                    np.sin(env.theta),
                    np.cos(env.theta),
                    env.omega / 3.0,
                    env.fuel,
                    1.0 - env.steps / env.max_steps,
                ], dtype=np.float32))

    def test_touchdown_rehearsal_exposes_the_success_outcome(self) -> None:
        env = lander.make_frontier_evaluation_env(4)
        env.rng.seed(7)
        successes = 0
        for _ in range(50):
            env.reset()
            for _ in range(env.max_steps - env.steps):
                _, _, done, _ = env.step(np.zeros(2, dtype=np.float32))
                if done:
                    break
            successes += int(env.landed)
        self.assertGreaterEqual(successes, 35)

    def test_start_source_does_not_change_transition_or_reward(self) -> None:
        approach = lander.make_frontier_evaluation_env(2)
        standard = self.spec.make_env(True)
        approach.rng.seed(7)
        approach.reset()

        dynamic_state = (
            "x", "y", "vx", "vy", "theta", "omega", "fuel", "steps",
            "episode_reward", "landed", "cause", "_u_main", "_phi_prev",
        )
        for name in dynamic_state:
            setattr(standard, name, getattr(approach, name))
        approach._start_kind = "approach"
        standard._start_kind = "standard"

        action = np.array([0.1, -0.2], dtype=np.float32)
        approach_result = approach.step(action)
        standard_result = standard.step(action)

        np.testing.assert_allclose(approach_result[0], standard_result[0])
        self.assertAlmostEqual(approach_result[1], standard_result[1])
        self.assertEqual(approach_result[2:], standard_result[2:])

    def test_every_unsuccessful_terminal_cause_has_the_same_failure_reward(self) -> None:
        failure_reward = getattr(lander, "FAILURE_REWARD", None)
        self.assertEqual(failure_reward, -100.0)

        def terminal_component(env: lander.LanderEnv) -> tuple[float, dict]:
            action = np.array([-1.0, 0.0], dtype=np.float32)
            env._phi_prev = env._phi()
            phi_before = env._phi_prev
            _, reward, done, info = env.step(action)
            self.assertTrue(done)
            self.assertEqual(env._phi_prev, 0.0)
            shaping = phi_before
            return reward - shaping, info

        crash = lander.LanderEnv(jitter=False)
        crash.x, crash.y = lander.PAD_CX, lander.PAD_Y
        crash.vx, crash.vy = 0.0, lander.SAFE_VY + 1.0
        crash_component, crash_info = terminal_component(crash)

        out_of_bounds = lander.LanderEnv(jitter=False)
        out_of_bounds.x = -1.0
        bounds_component, bounds_info = terminal_component(out_of_bounds)

        timeout = lander.LanderEnv(jitter=False)
        timeout.steps = timeout.max_steps - 1
        timeout_component, timeout_info = terminal_component(timeout)

        self.assertAlmostEqual(crash_component, failure_reward)
        self.assertAlmostEqual(bounds_component, failure_reward)
        self.assertAlmostEqual(timeout_component, failure_reward)
        self.assertEqual(crash.cause, "crash")
        self.assertEqual(out_of_bounds.cause, "out_of_bounds")
        self.assertEqual(timeout.cause, "timeout")
        self.assertFalse(crash_info["task_deadline"])
        self.assertFalse(bounds_info["task_deadline"])
        self.assertTrue(timeout_info["truncated"])
        self.assertTrue(timeout_info["task_deadline"])
        self.assertFalse(timeout.episode_summary()["success"])

    def test_soft_touchdown_reward_and_schema_remain_explicit(self) -> None:
        env = lander.LanderEnv(jitter=False)
        env.x, env.y = lander.PAD_CX, lander.PAD_Y
        env.vx = env.vy = env.theta = env.omega = 0.0
        env._phi_prev = env._phi()
        phi_before = env._phi_prev

        _, reward, done, info = env.step(
            np.array([-1.0, 0.0], dtype=np.float32))
        shaping = phi_before

        self.assertTrue(done)
        self.assertEqual(env._phi_prev, 0.0)
        self.assertAlmostEqual(reward - shaping, 100.0)
        self.assertTrue(env.episode_summary()["success"])
        self.assertFalse(info["truncated"])
        self.assertFalse(info["task_deadline"])
        self.assertEqual(self.spec.checkpoint_schema, 12)


if __name__ == "__main__":
    unittest.main()
