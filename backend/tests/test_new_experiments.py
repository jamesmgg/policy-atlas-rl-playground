"""Physical invariants and seeded task contracts for the expanded control lab."""
import importlib
import json
import math
import unittest

import numpy as np

from app.scenarios import get_spec, list_specs


IDS = ("orbital-docking", "robot-reach", "robot-tracking", "ball-beam")


class NewExperimentTests(unittest.TestCase):
    def specs(self):
        available = {spec.id for spec in list_specs()}
        self.assertTrue(set(IDS) <= available, "four new control tasks must be registered")
        return [get_spec(key) for key in IDS]

    def test_contracts_have_complete_observations_and_serializable_visuals(self):
        for spec in self.specs():
            with self.subTest(scenario=spec.id):
                env = spec.make_env(False)
                self.assertEqual(len(spec.observation_dimensions), env.obs_dim)
                self.assertEqual(len(spec.actions), env.n_continuous + env.n_binary)
                self.assertEqual(spec.horizon_steps, env.max_steps)
                self.assertEqual(len(env.reset()), env.obs_dim)
                self.assertTrue(callable(getattr(env, "reference_action", None)))
                json.dumps(spec.info(), allow_nan=False)
                json.dumps(spec.scene(), allow_nan=False)
                json.dumps(env.frame_payload(), allow_nan=False)
                self.assertFalse(env.episode_summary()["success"])

    def test_seeded_rollouts_are_reproducible_finite_and_reach_a_terminal(self):
        for spec in self.specs():
            left, right = spec.make_env(True), spec.make_env(True)
            left.rng.seed(314)
            right.rng.seed(314)
            np.testing.assert_array_equal(left.reset(), right.reset())
            actions = np.random.default_rng(2718)
            for _ in range(left.max_steps):
                action = actions.uniform(-1, 1, left.n_continuous).astype(np.float32)
                a, b = left.step(action), right.step(action)
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1:], b[1:])
                self.assertTrue(np.isfinite(a[0]).all())
                self.assertEqual(a[0].dtype, np.float32)
                if a[2]:
                    break
            self.assertTrue(a[2], spec.id)
            self.assertLessEqual(left.steps, left.max_steps)
            json.dumps(left.frame_payload(), allow_nan=False)
            json.dumps(left.episode_summary(), allow_nan=False)

    def test_reference_controllers_solve_seeded_tasks_without_neural_policies(self):
        for spec in self.specs():
            for seed in range(10):
                with self.subTest(scenario=spec.id, seed=seed):
                    env = spec.make_env(True)
                    env.rng.seed(seed)
                    env.reset()
                    for _ in range(env.max_steps):
                        action = env.reference_action()
                        self.assertTrue(np.all(np.abs(action) <= 1))
                        _, _, done, _ = env.step(action)
                        if done:
                            break
                    self.assertTrue(env.episode_summary()["success"], env.episode_summary())

    def test_zero_action_does_not_trivially_solve_the_canonical_tasks(self):
        for spec in self.specs():
            env = spec.make_env(False)
            for _ in range(env.max_steps):
                _, _, done, _ = env.step(np.zeros(env.n_continuous, np.float32))
                if done:
                    break
            self.assertFalse(env.episode_summary()["success"], spec.id)

    def test_orbital_integrator_matches_unforced_analytic_solution(self):
        self.specs()
        orbit = importlib.import_module("app.envs.orbital")
        initial = np.array([40.0, -15.0, 0.1, -0.2], dtype=np.float64)
        state = initial.copy()
        for _ in range(100):
            state = orbit.integrate_cw(state, np.zeros(2), 0.5)
        n, t = orbit.MEAN_MOTION, 50.0
        s, c = math.sin(n*t), math.cos(n*t)
        x, y, vx, vy = initial
        expected = [(4-3*c)*x+s/n*vx+2*(1-c)/n*vy,
                    6*(s-n*t)*x+y-2*(1-c)/n*vx+(4*s-3*n*t)/n*vy,
                    3*n*s*x+c*vx+2*s*vy,
                    -6*n*(1-c)*x-2*s*vx+(4*c-3)*vy]
        np.testing.assert_allclose(state, expected, rtol=1e-9, atol=1e-9)

    def test_robot_forward_kinematics_preserve_link_lengths(self):
        self.specs()
        robot = importlib.import_module("app.envs.robot_arm")
        for joints in ((0, 0), (math.pi/2, 0), (-0.5, 1.5)):
            elbow, tip = robot.kinematics(*joints)
            self.assertAlmostEqual(np.linalg.norm(elbow), robot.L1)
            self.assertAlmostEqual(np.linalg.norm(tip-elbow), robot.L2)

    def test_ball_accelerates_downhill_and_level_beam_has_no_gravity_drift(self):
        self.specs()
        env = get_spec("ball-beam").make_env(False)
        env.theta, env.omega, env.velocity = 0.1, 0.0, 0.0
        env.step(np.zeros(1))
        self.assertGreater(env.velocity, 0)
        env.theta, env.omega, env.velocity = 0.0, 0.0, 0.0
        position = env.position
        env.step(np.zeros(1))
        self.assertEqual(env.velocity, 0)
        self.assertEqual(env.position, position)


if __name__ == "__main__":
    unittest.main()
