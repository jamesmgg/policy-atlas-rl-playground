"""Reward and physics regressions retained after the v17 gate was retired."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1]))

from app.envs import drone
from app.scenarios import get_spec


class DroneRewardAndPhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_spec("drone-hover")

    def test_schedule_provenance_does_not_change_a_canonical_transition(self) -> None:
        scheduled = drone.DroneEnv(
            jitter=False,
            episode_schedule=True,
            forced_training_mode="canonical",
        )
        canonical = drone.DroneEnv(jitter=False)
        scheduled.rng.seed(17)
        scheduled.reset_for_training_episode(900)
        canonical.rng.seed(17)
        canonical.reset()

        action = np.array([0.1, -0.2], dtype=np.float32)
        scheduled_result = scheduled.step(action)
        canonical_result = canonical.step(action)

        np.testing.assert_allclose(scheduled_result[0], canonical_result[0])
        self.assertAlmostEqual(scheduled_result[1], canonical_result[1])
        self.assertEqual(scheduled_result[2:], canonical_result[2:])

    def test_segment_local_failure_return_preserves_terminal_cost(self) -> None:
        self.assertEqual(self.spec.gamma, 1.0)

        def terminal_transition(x: float, steps: int) -> tuple[float, float]:
            env = drone.DroneEnv(jitter=False, forced_start_segment=4)
            env.x, env.y = drone.WAYPOINTS[3]
            env.vx = env.vy = env.theta = env.omega = 0.0
            baseline = env._shaping_potential()
            env.x = x
            env.y = drone.WAYPOINTS[-1][1] - 80.0
            env.theta = drone.TIP_OVER + 0.01
            env.steps = steps
            env._shaping_potential_prev = baseline
            _, reward, done, _ = env.step(
                np.full(2, drone.HOVER_ACTION, dtype=np.float64))
            self.assertTrue(done)
            self.assertEqual(env.cause, "crash")
            terminal_potential = env._shaping_potential_prev
            self.assertAlmostEqual(
                reward,
                terminal_potential - baseline
                - drone.TERMINAL_FAILURE_PENALTY,
                places=9,
            )
            self.assertLess(
                reward, -baseline - drone.TERMINAL_FAILURE_PENALTY)
            return reward, terminal_potential

        near_reward, near_potential = terminal_transition(
            drone.WAYPOINTS[-1][0], 100)
        far_reward, far_potential = terminal_transition(900.0, 100)
        self.assertGreater(near_potential, far_potential)
        self.assertGreater(near_reward, far_reward)

        delayed_reward, delayed_potential = terminal_transition(
            drone.WAYPOINTS[-1][0], 500)
        self.assertAlmostEqual(delayed_potential, near_potential, places=9)
        self.assertAlmostEqual(delayed_reward, near_reward, places=9)

    def test_waypoint_capture_resets_baseline_without_target_jump(self) -> None:
        env = drone.DroneEnv(jitter=False, forced_start_segment=3)
        env.x, env.y = drone.WAYPOINTS[3]
        env.vx = env.vy = env.theta = env.omega = 0.0
        old_target_potential = env._shaping_potential()
        env._shaping_potential_prev = old_target_potential

        _, reward, done, _ = env.step(
            np.full(2, drone.HOVER_ACTION, dtype=np.float64))

        self.assertFalse(done)
        self.assertEqual(env.k, 4)
        captured = env._shaping_potential(waypoint_index=3)
        next_baseline = env._shaping_potential(waypoint_index=4)
        self.assertAlmostEqual(
            reward, captured - old_target_potential + 20.0, places=9)
        self.assertAlmostEqual(
            env._shaping_potential_prev, next_baseline, places=9)
        self.assertNotAlmostEqual(
            reward, next_baseline - old_target_potential + 20.0, places=6)

    def test_efficiency_regularizers_are_charged_only_on_completion(self) -> None:
        env = drone.DroneEnv(jitter=False, forced_start_segment=4)
        env.vx = 0.0
        env._shaping_potential_prev = env._shaping_potential()
        first_potential = env._shaping_potential_prev
        _, first_reward, done, _ = env.step(
            np.array([1.0, -1.0], dtype=np.float64))
        self.assertFalse(done)
        next_potential = env._shaping_potential_prev
        self.assertAlmostEqual(
            first_reward, next_potential - first_potential, places=9)
        accumulated_regularizer = 0.002 * abs(env.omega) + 0.005

        env.x, env.y = drone.WAYPOINTS[-1]
        env.vx = env.vy = env.theta = env.omega = 0.0
        env._shaping_potential_prev = env._shaping_potential()
        completion_start = env._shaping_potential_prev
        _, completion_reward, done, _ = env.step(
            np.full(2, -1.0, dtype=np.float64))

        self.assertTrue(done)
        self.assertEqual(env.cause, "complete")
        completion_end = env._shaping_potential_prev
        self.assertAlmostEqual(
            completion_reward,
            completion_end - completion_start + 70.0
            - accumulated_regularizer,
            places=9,
        )

    def test_braking_envelope_is_physical_and_segment_local(self) -> None:
        env = drone.DroneEnv(jitter=False, forced_start_segment=4)
        speeds = [env._braking_speed_target(distance)
                  for distance in (0.0, 25.0, 150.0, 700.0)]
        self.assertEqual(speeds[0], drone.WAYPOINT_TARGET_SPEED)
        self.assertTrue(all(left <= right
                            for left, right in zip(speeds, speeds[1:])))
        self.assertLessEqual(max(speeds), drone.WAYPOINT_CRUISE_SPEED)
        self.assertAlmostEqual(
            speeds[1] ** 2,
            drone.WAYPOINT_TARGET_SPEED ** 2
            + 2.0 * drone.WAYPOINT_COMFORT_DECELERATION * 25.0,
        )

        env.vx = 80.0
        desired_vx, _ = env._desired_velocity_target()
        self.assertLess(desired_vx, 0.0)
        self.assertLess(env._desired_tilt_target(desired_vx), 0.0)

    def test_braking_potential_prefers_counter_thrust(self) -> None:
        def reward_for(action: np.ndarray) -> float:
            env = drone.DroneEnv(jitter=False, forced_start_segment=4)
            env.vx = 80.0
            env.vy = env.theta = env.omega = 0.0
            env._shaping_potential_prev = env._shaping_potential()
            _, reward, done, _ = env.step(action)
            self.assertFalse(done)
            return reward

        correct = reward_for(np.array([1.0, -1.0], dtype=np.float64))
        hover = reward_for(np.full(
            2, drone.HOVER_ACTION, dtype=np.float64))
        wrong = reward_for(np.array([-1.0, 1.0], dtype=np.float64))
        self.assertGreater(correct, hover)
        self.assertGreater(hover, wrong)


if __name__ == "__main__":
    unittest.main()
