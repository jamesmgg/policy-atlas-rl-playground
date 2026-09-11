"""Success describes a completed task, never an empty or partial attempt."""
import unittest

import numpy as np

from app.envs.pendulum import PendulumEnv
from app.envs import drone


class PendulumTerminalReportingTests(unittest.TestCase):
    def test_reset_does_not_report_a_successful_balance(self):
        env = PendulumEnv(jitter=False)
        self.assertFalse(env.episode_summary()["success"])
        self.assertEqual(env.episode_summary()["cause"], "running")

    def test_short_upright_attempt_is_not_a_completed_hold(self):
        env = PendulumEnv(jitter=False)
        env.theta = env.theta_dot = 0.0
        for _ in range(100):
            _, _, done, _ = env.step(np.zeros(1, dtype=np.float32))
        self.assertFalse(done)
        self.assertFalse(env.episode_summary()["success"])

    def test_finished_upright_hold_reports_success(self):
        env = PendulumEnv(jitter=False)
        env.theta = env.theta_dot = 0.0
        for _ in range(env.max_steps):
            _, _, done, _ = env.step(np.zeros(1, dtype=np.float32))
        self.assertTrue(done)
        self.assertTrue(env.episode_summary()["success"])
        self.assertEqual(env.episode_summary()["cause"], "balanced")

    def test_finished_hanging_attempt_reports_timeout(self):
        env = PendulumEnv(jitter=False)
        for _ in range(env.max_steps):
            env.step(np.zeros(1, dtype=np.float32))
        self.assertFalse(env.episode_summary()["success"])
        self.assertEqual(env.episode_summary()["cause"], "timeout")


class DroneTerminalReportingTests(unittest.TestCase):
    def final_waypoint(self, theta):
        env = drone.DroneEnv(jitter=False)
        env.k = len(drone.WAYPOINTS) - 1
        env.x, env.y = drone.WAYPOINTS[-1]
        env.vx = env.vy = env.omega = 0.0
        env.theta = theta
        env._shaping_potential_prev = env._shaping_potential()
        return env

    def test_tipped_drone_cannot_win_by_touching_final_waypoint(self):
        env = self.final_waypoint(drone.TIP_OVER + 0.01)
        _, _, done, _ = env.step(np.full(2, drone.HOVER_ACTION))
        self.assertTrue(done)
        self.assertFalse(env.episode_summary()["success"])
        self.assertEqual(env.episode_summary()["cause"], "crash")

    def test_safe_final_waypoint_still_completes(self):
        env = self.final_waypoint(0.0)
        _, _, done, _ = env.step(np.full(2, drone.HOVER_ACTION))
        self.assertTrue(done)
        self.assertTrue(env.episode_summary()["success"])
        self.assertEqual(env.episode_summary()["cause"], "complete")


if __name__ == "__main__":
    unittest.main()
