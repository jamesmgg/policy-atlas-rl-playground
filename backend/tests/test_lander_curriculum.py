from __future__ import annotations

import unittest

import numpy as np

from app.envs import lander
from app.scenarios import list_specs


class TestLanderApproachCurriculum(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = {spec.id: spec for spec in list_specs()}["lunar-lander"]

    def test_training_factory_does_not_change_evaluation_starts(self) -> None:
        training = self.spec.make_training_env()
        randomized_evaluation = self.spec.make_env(True)
        canonical_evaluation = self.spec.make_env(False)

        self.assertTrue(training.approach_curriculum)
        self.assertFalse(randomized_evaluation.approach_curriculum)
        self.assertFalse(canonical_evaluation.approach_curriculum)
        self.assertEqual(
            self.spec.training_start_distribution,
            lander.TRAINING_START_DISTRIBUTION,
        )

        randomized_evaluation.rng.seed(2026)
        for _ in range(40):
            randomized_evaluation.reset()
            self.assertEqual(randomized_evaluation._start_kind, "standard")
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

    def test_training_mixture_covers_standard_touchdown_and_approach_states(self) -> None:
        self.assertEqual(lander.STANDARD_START_PROBABILITY, 0.5)
        self.assertEqual(lander.TOUCHDOWN_START_PROBABILITY, 0.25)
        self.assertEqual(lander.APPROACH_START_PROBABILITY, 0.25)
        self.assertEqual(
            lander.STANDARD_START_PROBABILITY
            + lander.TOUCHDOWN_START_PROBABILITY
            + lander.APPROACH_START_PROBABILITY,
            1.0,
        )

        env = self.spec.make_training_env()
        env.rng.seed(42)
        starts = {"standard": 0, "touchdown": 0, "approach": 0}

        for _ in range(900):
            observation = env.reset()
            starts[env._start_kind] += 1
            self.assertEqual(observation.shape, (env.obs_dim,))
            self.assertTrue(np.all(np.isfinite(observation)))

            if env._start_kind == "standard":
                self.assertEqual((env.x, env.y, env.vy, env.omega, env.fuel, env.steps),
                                 (500.0, 120.0, 0.0, 0.0, 1.0, 0))
                continue

            altitude = lander.PAD_Y - env.y
            prefix = ("TOUCHDOWN" if env._start_kind == "touchdown"
                      else "APPROACH")
            self.assertGreaterEqual(
                altitude, getattr(lander, f"{prefix}_ALTITUDE_MIN"))
            self.assertLessEqual(
                altitude, getattr(lander, f"{prefix}_ALTITUDE_MAX"))
            self.assertLessEqual(
                abs(env.x - lander.PAD_CX),
                getattr(lander, f"{prefix}_X_OFFSET_MAX"))
            self.assertLessEqual(abs(env.vx), getattr(lander, f"{prefix}_VX_MAX"))
            self.assertGreaterEqual(env.vy, getattr(lander, f"{prefix}_VY_MIN"))
            self.assertLessEqual(env.vy, getattr(lander, f"{prefix}_VY_MAX"))
            self.assertLessEqual(
                abs(env.theta), getattr(lander, f"{prefix}_THETA_MAX"))
            self.assertLessEqual(
                abs(env.omega), getattr(lander, f"{prefix}_OMEGA_MAX"))
            descent_fraction = (env.y - 120.0) / (lander.PAD_Y - 120.0)
            self.assertEqual(env.steps, round(180.0 * descent_fraction))
            self.assertGreaterEqual(env.steps, 0)
            self.assertGreater(env.fuel, 0.0)
            self.assertLessEqual(env.fuel, 1.0)

            # Every curriculum-varying quantity that affects future dynamics is
            # represented in the observation; the reset source itself is not
            # used by transition or reward logic.
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

        self.assertGreaterEqual(starts["standard"], 400)
        self.assertLessEqual(starts["standard"], 500)
        self.assertGreaterEqual(starts["touchdown"], 180)
        self.assertLessEqual(starts["touchdown"], 270)
        self.assertGreaterEqual(starts["approach"], 180)
        self.assertLessEqual(starts["approach"], 270)

    def test_approach_band_bridges_to_the_canonical_start_altitude(self) -> None:
        self.assertEqual(lander.APPROACH_ALTITUDE_MIN, 30.0)
        self.assertEqual(
            lander.APPROACH_ALTITUDE_MAX,
            lander.PAD_Y - 120.0,
        )

    def test_seeded_approach_samples_cover_old_and_extended_altitudes(self) -> None:
        env = self.spec.make_training_env()
        env.rng.seed(20260805)
        altitudes = []

        for _ in range(400):
            env.reset()
            if env._start_kind == "approach":
                altitudes.append(lander.PAD_Y - env.y)

        self.assertGreater(len(altitudes), 50)
        self.assertTrue(all(30.0 <= altitude <= 500.0 for altitude in altitudes))
        self.assertTrue(any(altitude <= 180.0 for altitude in altitudes))
        self.assertTrue(any(altitude > 180.0 for altitude in altitudes))

    def test_extended_approach_preserves_clock_fuel_observation_and_dynamics(self) -> None:
        approach = self.spec.make_training_env()
        approach.rng.seed(20260805)
        for _ in range(400):
            observation = approach.reset()
            altitude = lander.PAD_Y - approach.y
            if approach._start_kind == "approach" and altitude > 180.0:
                break
        else:
            self.fail("seeded curriculum did not produce an extended approach start")

        descent_fraction = (approach.y - 120.0) / (lander.PAD_Y - 120.0)
        self.assertEqual(approach.steps, round(180.0 * descent_fraction))
        self.assertGreaterEqual(approach.fuel, 1.0 - 0.5 * descent_fraction)
        self.assertLessEqual(approach.fuel, 1.0 - 0.1 * descent_fraction)
        np.testing.assert_allclose(observation, np.array([
            (approach.x - lander.PAD_CX) / 300.0,
            (approach.y - lander.PAD_Y) / 300.0,
            approach.vx / 60.0,
            approach.vy / 60.0,
            np.sin(approach.theta),
            np.cos(approach.theta),
            approach.omega / 3.0,
            approach.fuel,
            1.0 - approach.steps / approach.max_steps,
        ], dtype=np.float32))

        standard = self.spec.make_env(True)
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

    def test_v6_curriculum_protocol_text_is_exact(self) -> None:
        expected = (
            "50% standard high-altitude starts; 25% touchdown rehearsal 5-18 units "
            "above the pad; 25% braking approaches 30-500 units above the pad; all "
            "sampled states expose velocity, tilt, time, and fuel"
        )
        self.assertEqual(lander.TRAINING_START_DISTRIBUTION, expected)
        self.assertEqual(self.spec.training_start_distribution, expected)

    def test_touchdown_rehearsal_exposes_the_success_outcome(self) -> None:
        env = self.spec.make_training_env()
        env.rng.seed(7)
        successes = 0
        sampled = 0

        for _ in range(1000):
            env.reset()
            if env._start_kind != "touchdown":
                continue
            sampled += 1
            for _ in range(env.max_steps - env.steps):
                _, _, done, _ = env.step(np.zeros(2, dtype=np.float32))
                if done:
                    break
            successes += int(env.landed)
            if sampled == 50:
                break

        self.assertEqual(sampled, 50)
        self.assertGreaterEqual(successes, 35)

    def test_start_source_does_not_change_transition_or_reward(self) -> None:
        approach = self.spec.make_training_env()
        standard = self.spec.make_env(True)
        approach.rng.seed(7)
        for _ in range(100):
            approach.reset()
            if approach._start_kind == "approach":
                break
        else:
            self.fail("seeded curriculum did not produce an approach start")

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
            shaping = phi_before - env._phi()
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
        shaping = phi_before - env._phi()

        self.assertTrue(done)
        self.assertAlmostEqual(reward - shaping, 100.0)
        self.assertTrue(env.episode_summary()["success"])
        self.assertFalse(info["truncated"])
        self.assertFalse(info["task_deadline"])
        self.assertEqual(self.spec.checkpoint_schema, 4)


if __name__ == "__main__":
    unittest.main()
