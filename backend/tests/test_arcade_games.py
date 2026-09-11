"""Arcade objectives, replay state, reproducibility and terminal contracts."""
import importlib.util
import json
import unittest

import numpy as np


class ArcadeTests(unittest.TestCase):
    def specs(self):
        self.assertIsNotNone(importlib.util.find_spec("app.scenarios.arcade"),
                             "the arcade scenario contracts must exist")
        from app.scenarios.arcade import ARCADE_SPECS
        return ARCADE_SPECS

    def test_seeded_contracts_and_terminal_states(self):
        for spec in self.specs():
            with self.subTest(scenario=spec.id):
                left, right = spec.make_env(True), spec.make_env(True)
                left.rng.seed(1234)
                right.rng.seed(1234)
                np.testing.assert_array_equal(left.reset(), right.reset())
                self.assertEqual(len(spec.observation_dimensions), left.obs_dim)
                self.assertEqual(len(spec.actions), left.n_continuous + left.n_binary)
                self.assertEqual(spec.horizon_steps, left.max_steps)
                self.assertFalse(left.episode_summary()["success"])
                json.dumps(spec.info(), allow_nan=False)
                json.dumps(spec.scene(), allow_nan=False)
                actions = np.random.default_rng(123)
                for _ in range(left.max_steps):
                    action = actions.uniform(-1, 1, left.n_continuous)
                    a, b = left.step(action), right.step(action)
                    np.testing.assert_array_equal(a[0], b[0])
                    self.assertEqual(a[1:], b[1:])
                    self.assertTrue(np.isfinite(a[0]).all())
                    if a[2]:
                        break
                self.assertTrue(a[2])
                final = left.frame_payload()
                self.assertGreaterEqual(len(final["objects"]), 2)
                self.assertEqual(len(left.ghost_sample()), 5)
                json.dumps(final, allow_nan=False)
                reward = left.episode_reward
                state, extra_reward, done, info = left.step(np.ones(left.n_continuous))
                self.assertTrue(done)
                self.assertEqual(extra_reward, 0)
                self.assertEqual(left.episode_reward, reward)
                self.assertEqual(left.frame_payload(), final)

    def test_expert_solves_seeded_full_games_and_zero_action_does_not(self):
        for spec in self.specs():
            for seed in range(25):
                with self.subTest(scenario=spec.id, seed=seed):
                    env = spec.make_env(True)
                    env.rng.seed(seed)
                    env.reset()
                    for _ in range(env.max_steps):
                        _, _, done, _ = env.step(env.reference_action())
                        if done:
                            break
                    self.assertTrue(env.episode_summary()["success"], env.episode_summary())
            env = spec.make_env(False)
            for _ in range(env.max_steps):
                _, _, done, _ = env.step(np.zeros(env.n_continuous))
                if done:
                    break
            self.assertFalse(env.episode_summary()["success"], spec.id)

    def test_failures_cannot_count_as_a_win(self):
        specs = {spec.id: spec for spec in self.specs()}
        paddle = specs["paddle-rally"].make_env(False)
        paddle.ball_x, paddle.ball_y, paddle.ball_vy, paddle.paddle = .9, .12, -2., -.7
        self.assertTrue(paddle.step(np.zeros(1))[2])
        self.assertEqual(paddle.episode_summary()["cause"], "missed-ball")
        bird = specs["flappy-flight"].make_env(False)
        bird.y, bird.velocity = .98, 1.
        self.assertTrue(bird.step(np.zeros(1))[2])
        self.assertEqual(bird.episode_summary()["cause"], "out-of-bounds")
        coin = specs["coin-collector"].make_env(False)
        coin.position[:] = [.98, .5]
        coin.velocity[:] = [1., 0.]
        self.assertTrue(coin.step(np.array([1., 0.]))[2])
        self.assertEqual(coin.episode_summary()["cause"], "out-of-bounds")
        for env in (paddle, bird, coin):
            self.assertFalse(env.episode_summary()["success"])

    def test_demonstrations_are_disclosed_and_reproduce_pre_action_targets(self):
        for spec in self.specs():
            self.assertIsNotNone(spec.actor_warm_start)
            protocol = spec.actor_warm_start.protocol()
            self.assertFalse(protocol["pure_model_free_from_scratch"])
            self.assertFalse(protocol["expert"]["used_at_inference"])
        from app.ppo.arcade_demonstrations import build_dataset
        for spec in self.specs():
            a = build_dataset(spec.id, episodes=2, seed_base=710_000)
            b = build_dataset(spec.id, episodes=2, seed_base=710_000)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            env = spec.make_env(True)
            env.rng.seed(710_000)
            np.testing.assert_array_equal(a[0][0], env.reset())
            np.testing.assert_allclose(a[1][0], env.reference_action())
            self.assertTrue(np.all(np.abs(a[1]) <= 1))
            self.assertTrue(np.isfinite(a[0]).all())

    def test_deadline_is_observed_and_is_a_failure(self):
        for spec in self.specs():
            env = spec.make_env(False)
            env.steps = env.max_steps-1
            self.assertAlmostEqual(env._obs()[-1], 1/env.max_steps)
            observation, _, done, info = env.step(env.reference_action())
            self.assertTrue(done)
            self.assertTrue(info["task_deadline"])
            self.assertEqual(observation[-1], 0)
            self.assertEqual(env.cause, "timeout")
            self.assertFalse(env.episode_summary()["success"])

    def test_coin_requires_braking_and_contiguous_dwell(self):
        spec = next(s for s in self.specs() if s.id == "coin-collector")
        env = spec.make_env(False)
        env.position = env.target.copy()
        env.velocity[:] = [.5, 0]
        env.step(np.zeros(2))
        self.assertEqual(env.progress, 0)
        self.assertEqual(env.hold, 0)
        env.position = env.target.copy()
        env.velocity[:] = 0
        for _ in range(2):
            env.step(np.zeros(2))
        self.assertEqual(env.progress, 0)
        env.step(np.zeros(2))
        self.assertEqual(env.progress, 1)
        self.assertGreaterEqual(np.linalg.norm(env.position-env.target), .55)

    def test_pipe_collision_includes_bird_radius_and_counts_only_cleared_gates(self):
        spec = next(s for s in self.specs() if s.id == "flappy-flight")
        bird = spec.make_env(False)
        bird.pipe_x = .03
        bird.y = bird.gap+bird.gap_half_height-bird.bird_radius+.001
        _, _, done, _ = bird.step(np.zeros(1))
        self.assertTrue(done)
        self.assertEqual(bird.cause, "pipe-hit")
        self.assertEqual(bird.progress, 0)
        bird.reset()
        bird.pipe_x, bird.y = 0., bird.gap
        bird.step(np.zeros(1))
        self.assertEqual(bird.progress, 0)
        while bird.pipe_x <= 0 and bird.cause == "running":
            bird.step(np.zeros(1))
        self.assertEqual(bird.progress, 1)

    def test_game_steps_do_not_call_the_reference_controller(self):
        for spec in self.specs():
            env = spec.make_env(False)
            def forbidden():
                raise AssertionError("reference controller was used at inference")
            env.reference_action = forbidden
            observation, _, _, _ = env.step(np.zeros(env.n_continuous))
            self.assertTrue(np.isfinite(observation).all())


if __name__ == "__main__":
    unittest.main()
