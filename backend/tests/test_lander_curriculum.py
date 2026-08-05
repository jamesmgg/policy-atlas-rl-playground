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
            self.assertGreater(env.steps, 0)
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


if __name__ == "__main__":
    unittest.main()
