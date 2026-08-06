"""Scientific regression tests for the RL experiment engine.

These tests intentionally focus on invariants that can silently invalidate an
experiment: transition boundaries, action-density consistency, deterministic
seeding, evaluation aggregation, and the shared scenario contract.
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.ppo.buffer import RolloutBuffer
from app.ppo.network import ActorCritic
from app.ppo.agent import PPOAgent
from app.ppo import agent as agent_module
from app.checkpoints import (
    CheckpointIntegrityError,
    CheckpointRegistry,
    IncompatibleCheckpointError,
    _metadata_sha256,
    _sha256,
)
from app.settings import Settings
from app.scenarios import list_specs
from app.track import build_track
from app import track as track_defs
import app.trainer as trainer_module


class TestRolloutBoundaries(unittest.TestCase):
    def test_gae_stops_at_the_terminal_transition_that_was_stored(self) -> None:
        buffer = RolloutBuffer(capacity=3, obs_dim=1, act_dim=1)
        for reward, done, value in (
            (1.0, False, 0.0),
            (2.0, True, 0.0),
            (3.0, False, 10.0),
        ):
            buffer.add([0.0], [0.0], 0.0, reward, done, value)

        buffer.compute_gae(last_value=5.0, last_done=False,
                           gamma=1.0, gae_lambda=1.0)

        np.testing.assert_allclose(buffer.advantages, [3.0, 2.0, -2.0])
        np.testing.assert_allclose(buffer.returns, [3.0, 2.0, 8.0])

    def test_intrinsic_deadline_is_terminal_but_external_truncation_bootstraps(self) -> None:
        bootstrap = getattr(trainer_module, "bootstrap_time_limit", None)
        self.assertTrue(callable(bootstrap), "bootstrap_time_limit is missing")
        self.assertEqual(
            bootstrap(2.0, next_value=10.0, done=True,
                      info={"truncated": True}, gamma=0.9),
            11.0,
        )
        self.assertEqual(
            bootstrap(2.0, next_value=10.0, done=True,
                      info={"truncated": False}, gamma=0.9),
            2.0,
        )
        self.assertEqual(
            bootstrap(2.0, next_value=10.0, done=True,
                      info={"truncated": True, "task_deadline": True}, gamma=0.9),
            2.0,
            "a declared puzzle deadline has no unobserved continuation value",
        )

    def test_training_reward_scale_is_explicit_and_precedes_bootstrapping(self) -> None:
        training_reward = getattr(trainer_module, "training_reward", None)
        self.assertTrue(callable(training_reward), "training_reward is missing")
        self.assertAlmostEqual(
            training_reward(250.0, next_value=3.0, done=False, info={}),
            2.5,
        )
        self.assertAlmostEqual(
            training_reward(
                250.0,
                next_value=3.0,
                done=True,
                info={"truncated": True, "task_deadline": True},
            ),
            2.5,
            msg="intrinsic deadlines must remain terminal after reward scaling",
        )
        self.assertAlmostEqual(
            training_reward(
                250.0,
                next_value=3.0,
                done=True,
                info={"truncated": True, "task_deadline": False},
            ),
            2.5 + trainer_module.GAMMA * 3.0,
            msg="an external truncation bootstraps in scaled critic units",
        )

    def test_every_fixed_horizon_is_marked_as_an_intrinsic_deadline(self) -> None:
        for spec in list_specs():
            with self.subTest(scenario=spec.id):
                env = spec.make_env(False)
                env.steps = env.max_steps - 1
                action = np.zeros(env.n_continuous + env.n_binary)
                _, _, done, info = env.step(action)
                self.assertTrue(done)
                self.assertTrue(info.get("task_deadline"))

    def test_partial_rollout_uses_only_the_samples_that_were_collected(self) -> None:
        buffer = RolloutBuffer(capacity=8, obs_dim=1, act_dim=1)
        for i in range(3):
            buffer.add([float(i)], [0.0], 0.0, 1.0, False, 0.0)
        buffer.compute_gae(last_value=0.0, last_done=False)
        batches = list(buffer.minibatches(batch_size=8, device=torch.device("cpu")))
        self.assertEqual(sum(len(batch["obs"]) for batch in batches), 3)
        self.assertTrue(np.all(np.isfinite(buffer.advantages[:buffer.ptr])))

    def test_checkpoint_frequency_does_not_change_ppo_update_boundaries(self) -> None:
        class ThreeEpisodeEnv:
            obs_dim = 1
            n_continuous = 1
            n_binary = 0
            max_steps = 1
            dt = 0.1

            def __init__(self):
                self.completed = 0
                self.stop_event = None

            def reset(self):
                self.episode_reward = 0.0
                return np.array([0.0], dtype=np.float32)

            def step(self, action):
                self.episode_reward = 1.0
                self.completed += 1
                if self.completed == 1 and self.stop_event is not None:
                    self.stop_event.set()
                return np.array([0.0], dtype=np.float32), 1.0, True, {
                    "truncated": False,
                }

            def episode_summary(self):
                return {"reward": 1.0, "steps": 1, "cause": "done",
                        "metric": 1.0, "success": True}

            def frame_payload(self):
                return {}

        class RecordingAgent:
            act_dim = 1

            def __init__(self):
                self.batch_sizes = []

            def select_action(self, observation, deterministic=False):
                return np.array([0.0], dtype=np.float32), 0.0, 0.0

            def update(self, buffer):
                self.batch_sizes.append(buffer.ptr)
                return {"policy_loss": 0.0, "value_loss": 0.0,
                        "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}

            def get_value(self, observation):
                return 0.0

        def update_boundaries(checkpoint_every: int, pause_after_first: bool = False) -> list[int]:
            trainer = object.__new__(trainer_module.Trainer)
            trainer.env = ThreeEpisodeEnv()
            trainer.agent = RecordingAgent()
            trainer.spec = SimpleNamespace(id="three-step", metric_mode="max",
                                           kind="generic", metric_label="score")
            trainer.episode = trainer.total_steps = trainer.update_count = 0
            trainer.sps = 0.0
            trainer.history = []
            trainer.best_reward = trainer.best_metric = None
            trainer.ghost = trainer._learning = None
            trainer.seed = 42
            trainer.settings = SimpleNamespace(eval_episodes=1)
            trainer.device = torch.device("cpu")
            trainer.max_episodes = 3
            trainer.checkpoint_every_n = checkpoint_every
            trainer._stop = trainer_module.threading.Event()
            if pause_after_first:
                trainer.env.stop_event = trainer._stop
            trainer._thread = None
            trainer.emit = lambda message: None
            trainer._save_checkpoint = lambda: None
            trainer._run()
            return trainer.agent.batch_sizes

        self.assertEqual(update_boundaries(1), update_boundaries(100))
        self.assertEqual(update_boundaries(100, pause_after_first=True),
                         update_boundaries(100))


class TestBoundedPolicy(unittest.TestCase):
    def test_continuous_exploration_has_no_forced_entropy_pressure(self) -> None:
        self.assertEqual(agent_module.ENT_COEF, 0.0)
        agent = PPOAgent(3, 2, 0, torch.device("cpu"))
        exploration_stats = getattr(agent, "exploration_stats", None)
        self.assertTrue(callable(exploration_stats), "exploration diagnostics missing")
        stats = exploration_stats()
        expected = math.exp(-0.5)
        self.assertAlmostEqual(stats["action_std_mean"], expected, places=6)
        self.assertAlmostEqual(stats["action_std_min"], expected, places=6)
        self.assertAlmostEqual(stats["action_std_max"], expected, places=6)
        self.assertAlmostEqual(stats["action_std_0"], expected, places=6)
        self.assertAlmostEqual(stats["action_std_1"], expected, places=6)

    def test_critic_diagnostics_report_calibration_and_bias(self) -> None:
        diagnostics = getattr(agent_module, "critic_diagnostics", None)
        self.assertTrue(callable(diagnostics), "critic diagnostics missing")
        stats = diagnostics(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, 1.0, 1.5]),
        )
        self.assertAlmostEqual(stats["value_bias"], -1.0)
        self.assertLess(stats["explained_variance"], 1.0)

    def test_value_loss_is_invariant_to_reward_units(self) -> None:
        value_loss = getattr(agent_module, "scale_aware_value_loss", None)
        self.assertTrue(callable(value_loss), "scale_aware_value_loss is missing")
        old = torch.tensor([0.0, 2.0, 4.0, 6.0])
        new = torch.tensor([1.0, 0.0, 7.0, 2.0], requires_grad=True)
        returns = torch.tensor([0.0, 10.0, 20.0, 30.0])

        base_loss, base_scale, base_clip_fraction = value_loss(
            new, old, returns, clip_epsilon=0.2)
        scaled_loss, scaled_scale, scaled_clip_fraction = value_loss(
            new * 100.0, old * 100.0, returns * 100.0, clip_epsilon=0.2)

        torch.testing.assert_close(base_loss, scaled_loss)
        torch.testing.assert_close(scaled_scale, base_scale * 100.0)
        torch.testing.assert_close(base_clip_fraction, scaled_clip_fraction)
        _, constant_scale, _ = value_loss(
            torch.full((4,), 10.0), torch.zeros(4), torch.full((4,), 100.0),
            clip_epsilon=0.2,
        )
        torch.testing.assert_close(constant_scale, torch.tensor(100.0))
        base_loss.backward()
        self.assertTrue(torch.all(torch.isfinite(new.grad)).item())

    def test_value_loss_accepts_one_normalizer_for_the_whole_rollout(self) -> None:
        first_loss, first_scale, _ = agent_module.scale_aware_value_loss(
            torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0]),
            target_scale=25.0,
        )
        rare_loss, rare_scale, _ = agent_module.scale_aware_value_loss(
            torch.tensor([90.0]), torch.tensor([80.0]), torch.tensor([100.0]),
            target_scale=25.0,
        )

        torch.testing.assert_close(first_scale, torch.tensor(25.0))
        torch.testing.assert_close(rare_scale, torch.tensor(25.0))
        self.assertTrue(torch.isfinite(first_loss))
        self.assertTrue(torch.isfinite(rare_loss))

    def test_continuous_actions_are_bounded_and_log_prob_is_recomputable(self) -> None:
        torch.manual_seed(41)
        policy = ActorCritic(obs_dim=4, n_continuous=2, n_binary=1,
                             hidden=(32, 32))
        obs = torch.randn(4096, 4)

        action, sampled_log_prob, _, _ = policy.act(obs)
        _, recomputed_log_prob, _, _ = policy.act(obs, action=action)

        self.assertTrue(torch.all(action[:, :2] <= 1.0).item())
        self.assertTrue(torch.all(action[:, :2] >= -1.0).item())
        torch.testing.assert_close(sampled_log_prob, recomputed_log_prob,
                                   atol=2e-5, rtol=2e-5)

    def test_saturated_float32_action_keeps_the_same_policy_likelihood(self) -> None:
        """Numerical tanh saturation must not corrupt PPO's old/new ratio."""
        policy = ActorCritic(obs_dim=2, n_continuous=1, n_binary=0,
                             hidden=(16,))
        obs = torch.zeros(1, 2)
        with torch.no_grad():
            policy.mu.weight.zero_()
            policy.mu.bias.fill_(20.0)

        action, sampled_log_prob, _, _ = policy.act(obs, deterministic=True)
        _, recomputed_log_prob, _, _ = policy.act(obs, action=action)

        self.assertEqual(float(action.item()), 1.0)
        torch.testing.assert_close(sampled_log_prob, recomputed_log_prob,
                                   atol=1e-4, rtol=1e-4)

    def test_reported_entropy_falls_when_squashed_actions_saturate(self) -> None:
        torch.manual_seed(4)
        policy = ActorCritic(obs_dim=2, n_continuous=1, n_binary=0,
                             hidden=(16,))
        obs = torch.zeros(4096, 2)
        with torch.no_grad():
            policy.log_std.fill_(-2.0)
            policy.mu.weight.zero_()
            policy.mu.bias.zero_()
        torch.manual_seed(8)
        _, _, centered_entropy, _ = policy.act(obs)
        with torch.no_grad():
            policy.mu.bias.fill_(5.0)
        torch.manual_seed(8)
        _, _, saturated_entropy, _ = policy.act(obs)

        self.assertLess(float(saturated_entropy.mean()),
                        float(centered_entropy.mean()) - 2.0)

    def test_failed_checkpoint_load_does_not_leave_a_hybrid_policy(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        before = {key: value.detach().clone()
                  for key, value in agent.network.state_dict().items()}
        bad = copy.deepcopy(agent.state_dict())
        bad["network"]["torso.0.bias"].fill_(123.0)
        bad["network"]["torso.0.weight"] = torch.zeros(1, 1)

        with self.assertRaises(RuntimeError):
            agent.load_state_dict(bad)

        for key, value in agent.network.state_dict().items():
            torch.testing.assert_close(value, before[key])


class TestExperimentContract(unittest.TestCase):
    def test_every_scenario_explains_the_experiment_surface(self) -> None:
        required = {
            "id", "name", "group", "kind", "description", "metric_label",
            "metric_mode", "objective", "success", "observations", "actions",
            "observation_dimensions", "reward_terms", "termination_conditions", "difficulty",
            "horizon_steps", "horizon_seconds",
        }
        for spec in list_specs():
            with self.subTest(spec=spec.id):
                info = spec.info()
                self.assertTrue(required <= set(info), required - set(info))
                self.assertGreaterEqual(len(info["observations"]), 2)
                self.assertGreaterEqual(len(info["actions"]), 1)
                self.assertEqual(len(info["observation_dimensions"]),
                                 spec.make_env(False).obs_dim)
                self.assertGreaterEqual(len(info["reward_terms"]), 1)
                self.assertGreaterEqual(len(info["termination_conditions"]), 1)
                self.assertGreater(info["horizon_steps"], 0)
                if info["horizon_seconds"] is not None:
                    self.assertGreater(info["horizon_seconds"], 0)

        mountain = {spec.id: spec for spec in list_specs()}["mountain-car"].info()
        self.assertIsNone(
            mountain["horizon_seconds"],
            "canonical MountainCar has discrete control steps, not physical seconds",
        )

    def test_finite_horizon_state_exposes_remaining_time(self) -> None:
        for spec in list_specs():
            with self.subTest(scenario=spec.id):
                env = spec.make_env(False)
                initial = env.reset()
                self.assertIn("remaining", spec.observation_dimensions[-1])
                self.assertAlmostEqual(float(initial[-1]), 1.0, places=6)
                env.steps = env.max_steps // 2
                midpoint = env._obs()
                self.assertAlmostEqual(float(midpoint[-1]), 0.5, places=2)

    def test_classic_control_library_includes_cartpole_and_mountain_car(self) -> None:
        specs = {spec.id: spec for spec in list_specs()}
        self.assertIn("cartpole-balance", specs)
        self.assertIn("mountain-car", specs)

        cart = specs["cartpole-balance"].make_env(False)
        mountain = specs["mountain-car"].make_env(False)
        self.assertEqual(cart.reset().shape, (cart.obs_dim,))
        self.assertEqual(mountain.reset().shape, (mountain.obs_dim,))
        self.assertEqual(len(cart.ghost_sample()), 5)
        self.assertEqual(len(mountain.ghost_sample()), 5)
        self.assertTrue(np.all(np.isfinite(cart.step(np.array([0.0]))[0])))
        self.assertTrue(np.all(np.isfinite(mountain.step(np.array([0.0]))[0])))

    def test_glacier_description_does_not_claim_drift_is_mandatory(self) -> None:
        glacier = {spec.id: spec for spec in list_specs()}["glacier"]

        self.assertNotIn("only way", glacier.description.lower())
        self.assertIn("controlled slides", glacier.description.lower())

    def test_cartpole_canonical_start_is_nontrivial_and_failure_metric_is_distinct(self) -> None:
        cart = {s.id: s for s in list_specs()}["cartpole-balance"].make_env(False)
        observation = cart.reset()
        self.assertNotEqual(float(observation[2]), 0.0,
                            "the fixed replay must not start at perfect equilibrium")

        cart.steps = cart.max_steps
        cart.cause = "tipped"
        failed_metric = cart.episode_summary()["metric"]
        cart.cause = "balanced"
        successful_metric = cart.episode_summary()["metric"]
        self.assertLess(failed_metric, successful_metric)

    def test_mountain_car_primary_metric_rewards_efficient_success_only(self) -> None:
        spec = {s.id: s for s in list_specs()}["mountain-car"]
        self.assertEqual(
            spec.checkpoint_schema,
            3,
            "changed metric and finite-horizon observations require a fresh policy",
        )
        mountain = spec.make_env(False)
        mountain.control_effort = 3.25
        mountain.peak_position = 0.31
        mountain.cause = "timeout"
        failed = mountain.episode_summary()
        self.assertIsNone(failed["metric"])
        self.assertEqual(failed["failure_progress"], 0.31)

        mountain.cause = "summit"
        successful = mountain.episode_summary()
        self.assertEqual(successful["metric"], 3.25)
        self.assertIsNone(successful["failure_progress"])

    def test_track_arc_queries_use_measured_arc_length(self) -> None:
        track = build_track(track_defs.APEX_GP)
        segments = np.diff(np.r_[track.arc, track.total_length])
        tolerance = float(segments.max()) + 1e-6

        for s in np.linspace(0.0, track.total_length, 101, endpoint=False):
            idx = track.index_at_arc(float(s))
            error = abs(float(track.arc[idx]) - float(s))
            error = min(error, track.total_length - error)
            self.assertLessEqual(error, tolerance)

        for idx in range(0, track.n, max(track.n // 25, 1)):
            ahead = track.index_ahead(idx, 120.0)
            expected = (float(track.arc[idx]) + 120.0) % track.total_length
            error = abs(float(track.arc[ahead]) - expected)
            error = min(error, track.total_length - error)
            self.assertLessEqual(error, tolerance)

    def test_crashed_lander_cannot_report_the_optimal_landing_metric(self) -> None:
        from app.envs.lander import LanderEnv, PAD_CX, PAD_Y

        env = LanderEnv(jitter=False)
        env.x, env.y = PAD_CX, PAD_Y
        env.landed = False
        env.cause = "crash"
        self.assertGreater(env.episode_summary()["metric"], 0.0)

    def test_lander_wraps_equivalent_angles_before_reward_and_touchdown(self) -> None:
        from app.envs.lander import LanderEnv, PAD_CX, PAD_Y

        env = LanderEnv(jitter=False)
        env.x, env.y = PAD_CX, PAD_Y
        env.vx = env.vy = env.omega = 0.0
        env.theta = 2.0 * math.pi

        _, _, done, _ = env.step(np.array([-1.0, 0.0]))

        self.assertTrue(done)
        self.assertTrue(env.landed, "2*pi is the same upright attitude as zero")
        self.assertGreaterEqual(env.theta, -math.pi)
        self.assertLess(env.theta, math.pi)

    def test_lander_rejects_high_speed_contact_in_either_vertical_direction(self) -> None:
        from app.envs.lander import LanderEnv, PAD_CX, PAD_Y

        env = LanderEnv(jitter=False)
        env.x, env.y = PAD_CX, PAD_Y + 5.0
        env.vx, env.vy = 0.0, -100.0
        env.theta = env.omega = 0.0

        _, _, done, _ = env.step(np.array([-1.0, 0.0]))

        self.assertTrue(done)
        self.assertFalse(env.landed)
        self.assertEqual(env.cause, "crash")

    def test_pendulum_catalog_horizon_matches_runtime(self) -> None:
        spec = {spec.id: spec for spec in list_specs()}["pendulum-swingup"]
        env = spec.make_env(False)

        self.assertEqual(spec.horizon_seconds, env.max_steps * env.dt)
        self.assertTrue(any(
            "16-second horizon" in condition
            for condition in spec.termination_conditions
        ))

    def test_driving_observation_exposes_surface_and_every_traffic_car(self) -> None:
        specs = {spec.id: spec for spec in list_specs()}
        dry = specs["apex-gp"].make_env(False)
        wet = specs["apex-gp-wet"].make_env(False)
        traffic = specs["traffic-rush"].make_env(False)
        self.assertEqual(specs["traffic-rush"].checkpoint_schema, 12)

        self.assertGreaterEqual(dry.obs_dim, 17)  # base state + grip profile
        self.assertEqual(wet.obs_dim, dry.obs_dim)
        self.assertEqual(traffic.obs_dim - dry.obs_dim, 12)  # 4 values x 3 bots

        first_bot = dry.obs_dim - 1  # traffic fields precede the shared time field
        traffic._bot_arcs[0] = (traffic.s_prev - 2.0) % traffic.track.total_length
        self.assertLess(
            float(traffic._obs()[first_bot]), 0.0,
            "a nearby rear traffic car must not look like distant clear road",
        )
        traffic._bot_passed[0] = True
        self.assertEqual(float(traffic._obs()[first_bot + 3]), 1.0)

    def test_driving_observation_exposes_reward_and_termination_state(self) -> None:
        spec = {spec.id: spec for spec in list_specs()}["drift-trial"]
        env = spec.make_env(False)
        dimensions = list(spec.observation_dimensions)
        required = (
            "sin track phase",
            "cos track phase",
            "forward course progress / lap length",
            "wrong-way margin / 25",
            "progress since stall anchor / threshold",
            "stall counter / limit",
            "objective completion fraction",
        )
        for dimension in required:
            self.assertIn(dimension, dimensions)

        env.s_prev = env.track.total_length * 0.25
        env.progress = env.track.total_length * 0.4
        env.peak_progress = env.progress + 12.5
        env._stall_anchor_progress = env.progress - 6.0
        env._stall_steps = 150
        env.style = 5.0
        observation = env._obs()

        self.assertAlmostEqual(
            float(observation[dimensions.index("sin track phase")]), 1.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(observation[dimensions.index("cos track phase")]), 0.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(observation[dimensions.index(
                "forward course progress / lap length")]), 0.4,
            places=5,
        )
        self.assertAlmostEqual(
            float(observation[dimensions.index("wrong-way margin / 25")]), -0.5,
            places=5,
        )
        self.assertAlmostEqual(
            float(observation[dimensions.index(
                "progress since stall anchor / threshold")]), 0.5,
            places=5,
        )
        self.assertAlmostEqual(
            float(observation[dimensions.index("stall counter / limit")]), 0.5,
            places=5,
        )
        self.assertAlmostEqual(
            float(observation[dimensions.index("objective completion fraction")]),
            0.5,
            places=5,
        )

        eco_spec = {spec.id: spec for spec in list_specs()}["eco-gp"]
        eco = eco_spec.make_env(False)
        eco.progress = 1.5 * eco.track.total_length
        eco_observation = eco._obs()
        self.assertAlmostEqual(
            float(eco_observation[list(eco_spec.observation_dimensions).index(
                "forward course progress / lap length")]),
            1.5,
            places=5,
            msg="multi-lap Eco state must not collapse to the one-lap boundary",
        )

    def test_driving_stall_rule_is_an_explicit_observed_counter(self) -> None:
        from app.envs.driving import STALL_WINDOW

        spec = {spec.id: spec for spec in list_specs()}["apex-gp"]
        env = spec.make_env(False)
        self.assertFalse(hasattr(env, "_progress_log"))
        env._stall_steps = STALL_WINDOW - 1

        _, _, done, _ = env.step(np.zeros(3))

        self.assertTrue(done)
        self.assertEqual(env.cause, "stall")

    def test_driving_curriculum_keeps_start_line_exposure_and_track_coverage(self) -> None:
        training = {spec.id: spec for spec in list_specs()}[
            "rally-ridge"
        ].make_training_env()
        self.assertTrue(
            hasattr(training, "random_start"),
            "DrivingEnv has no training start-distribution control",
        )
        self.assertEqual(training.start_line_probability, 0.75)
        training.rng.seed(42)

        starts: list[int] = []
        for _ in range(240):
            training.reset()
            starts.append(training.idx)

        self.assertTrue(set(starts).issubset(set(training.track.checkpoints)))
        start_line_count = starts.count(0)
        self.assertGreaterEqual(start_line_count, 150)
        self.assertLessEqual(start_line_count, 210)
        self.assertGreaterEqual(
            len(set(starts) - {0}), len(training.track.checkpoints) // 2,
        )

    def test_randomized_traffic_start_advances_world_state_consistently(self) -> None:
        traffic = {spec.id: spec for spec in list_specs()}[
            "traffic-rush"
        ].make_env(False)
        traffic.random_start = True
        traffic.rng.seed(7)
        traffic.reset()
        self.assertNotEqual(traffic.idx, 0, "test seed must exercise a moved start")

        elapsed = traffic.steps * traffic.dt
        for index, bot in enumerate(traffic.features.bots):
            bot_arc = (
                bot.start_frac * traffic.track.total_length
                + bot.speed * elapsed
            ) % traffic.track.total_length
            with self.subTest(bot=index):
                self.assertAlmostEqual(
                    traffic._bot_gap(index),
                    (bot_arc - traffic.s_prev) % traffic.track.total_length,
                    places=6,
                )

    def test_only_driving_training_factory_enables_random_starts(self) -> None:
        specs = {spec.id: spec for spec in list_specs()}
        driving = specs["rally-ridge"]
        make_training_env = getattr(driving, "make_training_env", None)
        self.assertTrue(callable(make_training_env), "training factory is missing")

        training = make_training_env()
        fixed_suite = driving.make_env(True)
        canonical = driving.make_env(False)

        self.assertTrue(training.random_start)
        self.assertFalse(fixed_suite.random_start)
        self.assertFalse(canonical.random_start)
        self.assertFalse(
            getattr(specs["cartpole-balance"].make_training_env(),
                    "random_start", False),
        )

    def test_terminal_driving_failure_cannot_be_reported_as_success(self) -> None:
        specs = {spec.id: spec for spec in list_specs()}
        circuit = specs["apex-gp"].make_env(False)
        circuit.laps = 1
        circuit.best_lap_time = 45.0
        circuit.cause = "collision"
        self.assertFalse(circuit.episode_summary()["success"])

        traffic = specs["traffic-rush"].make_env(False)
        traffic.overtakes = len(traffic.features.bots)
        traffic.cause = "contact"
        self.assertFalse(traffic.episode_summary()["success"])

    def test_stationary_drift_cannot_earn_a_corner_bonus(self) -> None:
        from app.envs.driving import drift_slip_quality

        spec = {spec.id: spec for spec in list_specs()}["rally-ridge"]
        rally = spec.make_env(False)

        self.assertEqual(drift_slip_quality(0.0), 0.0)
        self.assertEqual(drift_slip_quality(math.radians(15.0)), 1.0)
        self.assertEqual(drift_slip_quality(math.radians(35.0)), 0.0)
        self.assertEqual(
            drift_slip_quality(-math.radians(15.0)),
            1.0,
            "left and right drifts should be symmetric",
        )
        self.assertLessEqual(
            rally.reward_cfg.drift_corner,
            rally.reward_cfg.progress * 0.2,
            "drift shaping must stay secondary to task progress",
        )
        self.assertGreater(abs(float(rally.track.curvature[rally.idx])), 0.012)
        _, reward, done, _ = rally.step(np.array([0.0, 0.0, 1.0]))

        self.assertFalse(done)
        self.assertGreater(rally.car.drift, 0.0)
        self.assertAlmostEqual(
            reward,
            rally.reward_cfg.time,
            places=8,
            msg="the drift control alone is not physical drifting",
        )

    def test_drift_trial_style_requires_forward_measured_slip(self) -> None:
        from app.physics import CarState
        from app.track import heading_at

        spec = {spec.id: spec for spec in list_specs()}["drift-trial"]
        trial = spec.make_env(False)
        idx = int(np.argmax(np.abs(trial.track.curvature)))
        x, y = trial.track.centerline[idx]
        heading = heading_at(trial.track, idx)

        def place(v_long: float, v_lat: float) -> None:
            trial.car = CarState(
                x=float(x), y=float(y), heading=heading,
                v_long=v_long, v_lat=v_lat, omega=0.0, drift=1.0,
            )
            trial.idx = idx
            trial.s_prev = float(trial.track.arc[idx])
            trial.style = 0.0

        place(20.0, 0.0)
        trial.step(np.array([0.0, 0.0, 1.0]))
        self.assertEqual(trial.style, 0.0, "a drift button is not tire slip")

        place(-20.0, math.tan(math.radians(15.0)) * 20.0)
        trial.step(np.array([0.0, 0.0, 1.0]))
        self.assertEqual(trial.style, 0.0, "reverse travel cannot farm style")

        place(20.0, math.tan(math.radians(15.0)) * 20.0)
        trial.step(np.array([0.0, 0.0, 1.0]))
        self.assertGreater(trial.style, 0.0)

    def test_traffic_rewards_each_bot_identity_only_once(self) -> None:
        traffic = {spec.id: spec for spec in list_specs()}["traffic-rush"].make_env(False)
        length = traffic.track.total_length
        far = traffic.s_prev + length * 0.5
        traffic._bot_arcs = [traffic.s_prev + 10.0, far, far + 50.0]
        traffic._bot_prev_gap = [traffic._bot_gap(i) for i in range(3)]
        traffic.s_prev += 25.0

        self.assertGreater(traffic._step_bots(), 0.0)
        self.assertEqual(traffic.overtakes, 1)

        traffic._bot_arcs[0] = traffic.s_prev + 100.0
        traffic._bot_prev_gap[0] = 100.0
        traffic._step_bots()  # the legacy implementation re-arms bot zero
        traffic._bot_arcs[0] = traffic.s_prev + 10.0
        traffic._bot_prev_gap[0] = 10.0
        traffic.s_prev += 25.0
        self.assertEqual(traffic._step_bots(), 0.0)
        self.assertEqual(traffic.overtakes, 1)

    def test_completed_driving_objective_ends_before_a_later_crash(self) -> None:
        specs = {spec.id: spec for spec in list_specs()}
        apex = specs["apex-gp"].make_env(False)
        apex.laps = 1

        _, _, done, _ = apex.step(np.zeros(3))

        self.assertTrue(done)
        self.assertEqual(apex.cause, "complete")
        self.assertTrue(apex.episode_summary()["success"])

    def test_rally_failure_reports_normalized_peak_progress(self) -> None:
        spec = {spec.id: spec for spec in list_specs()}["rally-ridge"]
        self.assertEqual(
            spec.checkpoint_schema,
            7,
            "changed reward, evaluation, and observation semantics require a fresh policy",
        )
        rally = spec.make_env(False)
        rally.peak_progress = rally.track.total_length * 0.42
        rally.cause = "stall"

        failed = rally.episode_summary()
        self.assertEqual(failed["failure_progress"], 0.42)

        rally.laps = 1
        rally.cause = "timeout"
        successful = rally.episode_summary()
        self.assertTrue(successful["success"])
        self.assertIsNone(successful["failure_progress"])


class TestEvaluationProtocol(unittest.TestCase):
    def test_checkpoint_protocol_discloses_learning_scale_and_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = trainer_module.Trainer(Settings(
                port=8901,
                checkpoint_dir=Path(tmp),
                checkpoint_every_n=25,
                max_episodes=10,
                use_gpu=False,
                seed=42,
                eval_episodes=1,
            ))
            trainer._run_eval = lambda: {
                "reward": 0.0, "reward_std": 0.0,
                "metric": None, "metric_std": None,
                "failure_progress": 0.0, "episodes": 1,
                "success_rate": 0.0,
                "success_ci_low": 0.0, "success_ci_high": 1.0,
                "evaluation_suite": "test-suite", "seed": 42,
                "trajectory": [],
            }
            trainer._save_checkpoint()
            protocol = trainer.registry.list()[0]["protocol"]

        self.assertEqual(protocol["version"], 12)
        self.assertEqual(protocol["training_reward_scale"], 0.01)
        self.assertEqual(protocol["entropy_coefficient"], 0.0)
        self.assertEqual(protocol["value_loss_scale"], "rollout return RMS")
        self.assertEqual(
            protocol["training_start_distribution"],
            trainer.spec.training_start_distribution,
        )

    def test_trainer_uses_training_factory_for_driving_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = trainer_module.Trainer(Settings(
                port=8901,
                checkpoint_dir=Path(tmp),
                checkpoint_every_n=25,
                max_episodes=10,
                use_gpu=False,
                seed=42,
                eval_episodes=1,
            ))

        self.assertEqual(trainer.spec.id, "apex-gp")
        self.assertTrue(trainer.env.random_start)

    def test_evaluation_summary_reports_dispersion_and_success_rate(self) -> None:
        aggregate = getattr(trainer_module, "aggregate_evaluations", None)
        self.assertTrue(callable(aggregate), "aggregate_evaluations is missing")
        result = aggregate([
            {"reward": 10.0, "metric": 4.0, "success": True},
            {"reward": 14.0, "metric": 6.0, "success": False,
             "failure_progress": 0.2},
            {"reward": 12.0, "metric": None, "success": True},
        ], metric_mode="min")

        self.assertEqual(result["episodes"], 3)
        self.assertEqual(result["reward_mean"], 12.0)
        self.assertAlmostEqual(result["reward_std"], math.sqrt(8 / 3), places=6)
        self.assertEqual(result["metric"], 5.0)
        self.assertAlmostEqual(result["success_rate"], 2 / 3)
        self.assertLess(result["success_ci_low"], result["success_rate"])
        self.assertGreater(result["success_ci_high"], result["success_rate"])
        self.assertEqual(result["failure_progress"], 0.2)

    def test_evaluation_suite_is_versioned_and_independent_of_training_seed(self) -> None:
        seed_for_episode = getattr(trainer_module, "evaluation_seed", None)
        suite_id = getattr(trainer_module, "evaluation_suite_id", None)
        self.assertTrue(callable(seed_for_episode), "evaluation_seed is missing")
        self.assertTrue(callable(suite_id), "evaluation_suite_id is missing")
        self.assertEqual([seed_for_episode(i) for i in range(3)],
                         [100_000, 100_001, 100_002])
        self.assertEqual(suite_id(10), "policy-atlas-eval-v1-n10")
        self.assertNotEqual(suite_id(5), suite_id(10))

    def test_engine_source_digest_is_recordable_with_results(self) -> None:
        source_digest = getattr(trainer_module, "source_digest", None)
        self.assertTrue(callable(source_digest), "source_digest is missing")
        digest = source_digest()
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in digest))

    def test_seed_helper_replays_python_numpy_and_torch_streams(self) -> None:
        seed_everything = getattr(trainer_module, "seed_everything", None)
        self.assertTrue(callable(seed_everything), "seed_everything is missing")

        def sample() -> tuple[float, float, float]:
            return random.random(), float(np.random.random()), float(torch.rand(1))

        seed_everything(2026)
        first = sample()
        seed_everything(2026)
        second = sample()
        self.assertEqual(first, second)

    def test_checkpoint_sidecar_preserves_evaluation_protocol(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            registry = CheckpointRegistry(Path(tmp), "test")
            registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 12.0,
                "reward_std": 2.5,
                "metric": 4.0,
                "metric_std": 0.75,
                "episodes": 5,
                "success_rate": 0.6,
                "success_ci_low": 0.23,
                "success_ci_high": 0.88,
                "evaluation_suite": "test-suite-v1",
                "failure_progress": 0.37,
                "seed": 2026,
                "update_count": 17,
                "total_steps": 4096,
                "protocol": {"algorithm": "PPO", "gamma": 0.995,
                             "rollout_steps": 2048},
                "training_diagnostics": {
                    "explained_variance": 0.42,
                    "value_bias": -0.17,
                    "action_std_mean": 0.31,
                },
                "trajectory": [],
            })
            meta = registry.list()[0]
        self.assertEqual(meta["eval_episodes"], 5)
        self.assertEqual(meta["eval_reward_std"], 2.5)
        self.assertEqual(meta["eval_metric_std"], 0.75)
        self.assertEqual(meta["success_rate"], 0.6)
        self.assertEqual(meta["success_ci_low"], 0.23)
        self.assertEqual(meta["success_ci_high"], 0.88)
        self.assertEqual(meta["evaluation_suite"], "test-suite-v1")
        self.assertEqual(meta["eval_failure_progress"], 0.37)
        self.assertEqual(meta["seed"], 2026)
        self.assertEqual(meta["update_count"], 17)
        self.assertEqual(meta["total_steps"], 4096)
        self.assertEqual(meta["protocol"]["gamma"], 0.995)
        self.assertEqual(meta["training_diagnostics"]["explained_variance"], 0.42)

    def test_status_exposes_latest_optimizer_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = trainer_module.Trainer(Settings(
                port=8901,
                checkpoint_dir=Path(tmp),
                checkpoint_every_n=25,
                max_episodes=10,
                use_gpu=False,
                seed=42,
                eval_episodes=1,
            ))
            trainer.latest_update_metrics = {
                "explained_variance": 0.25,
                "value_bias": -0.5,
                "action_std_mean": 0.4,
            }
            status = trainer.status()

        self.assertEqual(status["ppo_diagnostics"]["explained_variance"], 0.25)

    def test_tampered_sidecar_is_hidden_and_refused_on_load(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            registry = CheckpointRegistry(Path(tmp), "test")
            registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 12.0,
                "metric": 4.0,
                "episodes": 1,
                "success_rate": 1.0,
                "trajectory": [],
                "protocol": {"algorithm": "PPO", "engine_source_sha256": "a" * 64},
            })
            sidecar = Path(tmp) / "test" / "checkpoint_ep000005.json"
            payload = json.loads(sidecar.read_text())
            payload["eval_metric"] = 999.0
            payload["protocol"]["engine_source_sha256"] = "b" * 64
            sidecar.write_text(json.dumps(payload))

            self.assertEqual(registry.list(), [])
            with self.assertRaises(CheckpointIntegrityError):
                registry.load(5)

    def test_same_schema_checkpoint_signed_before_optional_field_is_added(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            registry = CheckpointRegistry(Path(tmp), "test", schema_version=1)
            registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 12.0, "metric": 4.0, "trajectory": [],
            })
            sidecar = registry._json(5)
            tensor = registry._pt(5)

            payload = json.loads(sidecar.read_text())
            payload.pop("training_diagnostics")
            payload["metadata_sha256"] = _metadata_sha256(payload)
            data = torch.load(tensor, map_location="cpu", weights_only=False)
            data["meta"].pop("training_diagnostics")
            data["meta"]["metadata_sha256"] = _metadata_sha256(data["meta"])
            torch.save(data, tensor)
            payload["checkpoint_sha256"] = _sha256(tensor)
            sidecar.write_text(json.dumps(payload))

            listed = registry.list()
            self.assertEqual(len(listed), 1)
            self.assertIsNone(listed[0]["training_diagnostics"])
            loaded = registry.load(5)
            self.assertNotIn("training_diagnostics", loaded["meta"])

    def test_incompatible_checkpoint_schema_is_hidden_and_refused(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            # Construct the newer reader first so this test still exercises
            # its load-time guard independently of startup migration.
            new_registry = CheckpointRegistry(Path(tmp), "driving", schema_version=2)
            old_registry = CheckpointRegistry(Path(tmp), "driving", schema_version=1)
            old_registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": 1.0, "trajectory": [],
            })

            self.assertEqual(new_registry.list(), [])
            with self.assertRaises(IncompatibleCheckpointError):
                new_registry.load_into(5, agent)

    def test_schema_upgrade_archives_incompatible_active_checkpoint_before_reuse(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_registry = CheckpointRegistry(root, "driving", schema_version=1)
            old_registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": 1.0, "trajectory": [],
            })
            old_checkpoint = old_registry._pt(5).read_bytes()
            old_sidecar = old_registry._json(5).read_bytes()

            new_registry = CheckpointRegistry(root, "driving", schema_version=2)

            self.assertEqual(new_registry.list(), [])
            archives = list((new_registry.dir / "archive").glob("schema-1-to-2-*"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(
                (archives[0] / "checkpoint_ep000005.pt").read_bytes(),
                old_checkpoint,
            )
            self.assertEqual(
                (archives[0] / "checkpoint_ep000005.json").read_bytes(),
                old_sidecar,
            )
            listed = new_registry.list_archives()
            self.assertEqual(len(listed), 1)
            self.assertFalse(listed[0]["compatible"])
            self.assertEqual(listed[0]["schema_version"], 1)
            self.assertFalse(new_registry.restore_archive(listed[0]["id"]))

            # Reusing an episode number in the new schema must not overwrite
            # the preserved experiment.
            new_registry.save(5, agent, [{"reward": 2.0}], {
                "reward": 2.0, "metric": 2.0, "trajectory": [],
            })
            self.assertEqual([meta["episode"] for meta in new_registry.list()], [5])
            self.assertEqual(
                (archives[0] / "checkpoint_ep000005.pt").read_bytes(),
                old_checkpoint,
            )

    def test_new_seeded_run_archives_old_checkpoints_recoverably(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            registry = CheckpointRegistry(Path(tmp), "test")
            registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": 1.0, "trajectory": [],
            })
            archive = registry.archive_current()
            self.assertEqual(registry.list(), [])
            self.assertIsNotNone(archive)
            self.assertTrue((archive / "checkpoint_ep000005.json").exists())
            self.assertTrue((archive / "checkpoint_ep000005.pt").exists())

    def test_archived_run_can_be_listed_and_restored(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            registry = CheckpointRegistry(Path(tmp), "test")
            registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": 1.0, "seed": 9, "trajectory": [],
            })
            archive = registry.archive_current()
            archive_id = archive.name

            runs = registry.list_archives()
            self.assertEqual(runs[0]["id"], archive_id)
            self.assertEqual(runs[0]["latest_episode"], 5)
            self.assertEqual(runs[0]["checkpoints"], 1)
            self.assertTrue(runs[0]["compatible"])
            self.assertEqual(runs[0]["schema_version"], 1)
            self.assertTrue(registry.restore_archive(archive_id))
            self.assertEqual([m["episode"] for m in registry.list()], [5])
            self.assertEqual(registry.list_archives(), [])

    def test_malformed_unsigned_archive_is_ignored_without_hiding_other_runs(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            registry = CheckpointRegistry(Path(tmp), "test")
            registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": 1.0, "trajectory": [],
            })
            valid_archive = registry.archive_current()
            malformed = registry.dir / "archive" / "malformed"
            malformed.mkdir()
            (malformed / "checkpoint_ep000010.json").write_text("{}")
            (malformed / "checkpoint_ep000010.pt").write_bytes(b"not-a-checkpoint")

            runs = registry.list_archives()

            self.assertEqual([run["id"] for run in runs], [valid_archive.name])

    def test_corrupt_archive_is_rejected_before_the_active_branch_moves(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            registry = CheckpointRegistry(Path(tmp), "test")
            registry.save(5, agent, [{"reward": 5.0}], {
                "reward": 5.0, "metric": 5.0, "trajectory": [],
            })
            archive = registry.archive_current()
            archive_id = archive.name
            (archive / "checkpoint_ep000005.pt").write_bytes(b"corrupt")

            registry.save(10, agent, [{"reward": 10.0}], {
                "reward": 10.0, "metric": 10.0, "trajectory": [],
            })

            self.assertFalse(registry.restore_archive(archive_id))
            self.assertEqual([meta["episode"] for meta in registry.list()], [10])
            self.assertTrue((archive / "checkpoint_ep000005.pt").exists())

    def test_resuming_an_older_checkpoint_archives_its_newer_descendants(self) -> None:
        agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            registry = CheckpointRegistry(Path(tmp), "test")
            for episode in (5, 10, 15):
                registry.save(episode, agent, [{"reward": float(episode)}], {
                    "reward": float(episode), "metric": float(episode),
                    "trajectory": [],
                })

            archive = registry.archive_after(5)

            self.assertEqual([m["episode"] for m in registry.list()], [5])
            self.assertIsNotNone(archive)
            self.assertTrue((archive / "checkpoint_ep000010.pt").exists())
            self.assertTrue((archive / "checkpoint_ep000015.json").exists())

    def test_restore_quarantines_a_corrupt_latest_checkpoint_and_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                port=8901, checkpoint_dir=Path(tmp), checkpoint_every_n=5,
                max_episodes=100, use_gpu=False, seed=42, eval_episodes=1,
            )
            first = trainer_module.Trainer(settings)
            for episode in (5, 10):
                first.registry.save(episode, first.agent, [
                    {"episode": episode, "reward": float(episode), "steps": 1,
                     "cause": "done", "metric": None, "success": False},
                ], {"reward": float(episode), "metric": None,
                    "trajectory": [], "update_count": episode})
            first.registry._pt(10).write_bytes(b"not a torch checkpoint")

            restored = trainer_module.Trainer(settings)

            self.assertEqual(restored.episode, 5)
            self.assertEqual([m["episode"] for m in restored.registry.list()], [5])
            quarantined = list(
                (Path(tmp) / restored.spec.id / "archive").rglob(
                    "checkpoint_ep000010.json"))
            self.assertEqual(len(quarantined), 1)

    def test_learning_lens_payload_is_finite_and_bounded_in_size(self) -> None:
        build_payload = getattr(trainer_module, "learning_payload", None)
        self.assertTrue(callable(build_payload), "learning_payload is missing")
        payload = build_payload(
            observation=np.arange(20, dtype=np.float32),
            action=np.array([-1.0, 0.25, 1.0], dtype=np.float32),
            reward=-0.125,
        )
        self.assertEqual(payload["observation"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(payload["action"], [-1.0, 0.25, 1.0])
        self.assertEqual(payload["reward"], -0.125)

    def test_learning_lens_shows_the_observation_that_produced_the_action(self) -> None:
        """The explanatory transition must not pair an action with next_obs."""
        class OneStepEnv:
            obs_dim = 2
            n_continuous = 1
            n_binary = 0
            max_steps = 1
            dt = 0.1
            episode_reward = 0.0

            def reset(self):
                self.episode_reward = 0.0
                return np.array([1.0, 2.0], dtype=np.float32)

            def step(self, action):
                self.episode_reward = 3.0
                return np.array([9.0, 10.0], dtype=np.float32), 3.0, True, {
                    "truncated": False,
                }

            def episode_summary(self):
                return {"reward": 3.0, "steps": 1, "cause": "done",
                        "metric": 1.0, "success": True}

            def frame_payload(self):
                return {}

        class OneStepAgent:
            act_dim = 1

            def select_action(self, observation, deterministic=False):
                return np.array([0.25], dtype=np.float32), 0.0, 0.0

            def update(self, buffer):
                return {"policy_loss": 0.0, "value_loss": 0.0,
                        "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}

            def get_value(self, observation):
                return 0.0

        trainer = object.__new__(trainer_module.Trainer)
        trainer.env = OneStepEnv()
        trainer.agent = OneStepAgent()
        trainer.spec = SimpleNamespace(id="one-step", metric_mode="max",
                                       kind="generic", metric_label="score")
        trainer.episode = 0
        trainer.total_steps = 0
        trainer.update_count = 0
        trainer.sps = 0.0
        trainer.history = []
        trainer.best_reward = None
        trainer.best_metric = None
        trainer.ghost = None
        trainer._learning = None
        trainer.seed = 42
        trainer.settings = SimpleNamespace(eval_episodes=1)
        trainer.device = torch.device("cpu")
        trainer.max_episodes = 1
        trainer.checkpoint_every_n = 2
        trainer._stop = trainer_module.threading.Event()
        trainer._thread = None
        messages = []
        trainer.emit = messages.append
        trainer._save_checkpoint = lambda: None

        trainer._run()

        self.assertEqual(trainer._learning["observation"], [1.0, 2.0])
        episode_end_index = next(
            i for i, message in enumerate(messages)
            if message["type"] == "episode_end"
        )
        terminal_frame_index = next(
            i for i, message in enumerate(messages)
            if message["type"] == "frame" and message.get("terminal")
        )
        self.assertLess(terminal_frame_index, episode_end_index)
        self.assertEqual(messages[terminal_frame_index]["cause"], "done")

    def test_rng_state_round_trip_replays_all_training_streams(self) -> None:
        capture = getattr(trainer_module, "capture_rng_state", None)
        restore = getattr(trainer_module, "restore_rng_state", None)
        self.assertTrue(callable(capture), "capture_rng_state is missing")
        self.assertTrue(callable(restore), "restore_rng_state is missing")
        env = {s.id: s for s in list_specs()}["pendulum-swingup"].make_env(True)
        trainer_module.seed_everything(77)
        env.rng.seed(77)
        state = capture(env)

        def sample() -> tuple[float, float, float, float]:
            return (random.random(), float(np.random.random()),
                    float(torch.rand(1)), env.rng.random())

        expected = sample()
        restore(state, env)
        self.assertEqual(sample(), expected)

    def test_training_resets_receive_the_restored_next_episode_number(self) -> None:
        """Curriculum phase must derive from persisted completed episodes."""
        class EpisodeAwareOneStepEnv:
            obs_dim = 1
            n_continuous = 1
            n_binary = 0
            max_steps = 1
            dt = 0.1

            def __init__(self):
                self.episode_reward = 0.0
                self.training_episodes = []

            def reset(self):
                self.fail("trainer bypassed the episode-aware reset")

            def fail(self, message):
                raise AssertionError(message)

            def reset_for_training_episode(self, episode):
                self.training_episodes.append(episode)
                self.episode_reward = 0.0
                return np.array([float(episode)], dtype=np.float32)

            def step(self, action):
                self.episode_reward = 1.0
                return np.array([0.0], dtype=np.float32), 1.0, True, {
                    "truncated": False,
                }

            def episode_summary(self):
                return {"reward": 1.0, "steps": 1, "cause": "done",
                        "metric": 1.0, "success": True}

            def frame_payload(self):
                return {}

        class Agent:
            act_dim = 1

            def select_action(self, observation, deterministic=False):
                return np.array([0.0], dtype=np.float32), 0.0, 0.0

            def update(self, buffer):
                return {"policy_loss": 0.0, "value_loss": 0.0,
                        "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}

            def get_value(self, observation):
                return 0.0

        trainer = object.__new__(trainer_module.Trainer)
        trainer.env = EpisodeAwareOneStepEnv()
        trainer.agent = Agent()
        trainer.spec = SimpleNamespace(id="episode-aware", metric_mode="max",
                                       kind="generic", metric_label="score")
        trainer.episode = 499
        trainer.total_steps = trainer.update_count = 0
        trainer.sps = 0.0
        trainer.history = []
        trainer.best_reward = trainer.best_metric = None
        trainer.ghost = trainer._learning = None
        trainer.seed = 42
        trainer.settings = SimpleNamespace(eval_episodes=1)
        trainer.device = torch.device("cpu")
        trainer.max_episodes = 501
        trainer.checkpoint_every_n = 10_000
        trainer._stop = trainer_module.threading.Event()
        trainer._thread = None
        trainer.emit = lambda message: None
        trainer._save_checkpoint = lambda: None

        trainer._run()

        self.assertEqual(trainer.env.training_episodes, [500, 501])

    def test_checkpoint_rng_precedes_the_next_episode_reset(self) -> None:
        """A restored checkpoint must reproduce the pending next start."""
        class SeededOneStepEnv:
            obs_dim = 1
            n_continuous = 1
            n_binary = 0
            max_steps = 1
            dt = 0.1

            def __init__(self):
                self.rng = random.Random(77)
                self.episode_reward = 0.0

            def reset(self):
                self.episode_reward = 0.0
                return np.array([self.rng.random()], dtype=np.float32)

            def step(self, action):
                self.episode_reward = 1.0
                return np.array([0.0], dtype=np.float32), 1.0, True, {
                    "truncated": False,
                }

            def episode_summary(self):
                return {"reward": 1.0, "steps": 1, "cause": "done",
                        "metric": 1.0, "success": True}

            def frame_payload(self):
                return {}

        class Agent:
            act_dim = 1

            def select_action(self, observation, deterministic=False):
                return np.array([0.0], dtype=np.float32), 0.0, 0.0

            def update(self, buffer):
                return {"policy_loss": 0.0, "value_loss": 0.0,
                        "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}

            def get_value(self, observation):
                return 0.0

        class Registry:
            rng_state = None

            def save(self, episode, agent, history, evaluation):
                self.rng_state = evaluation["rng_state"]["environment"]
                return SimpleNamespace(episode=episode, eval_reward=0.0,
                                       eval_metric=None)

            def list(self):
                return []

        env = SeededOneStepEnv()
        registry = Registry()
        trainer = object.__new__(trainer_module.Trainer)
        trainer.env = env
        trainer.agent = Agent()
        trainer.registry = registry
        trainer.spec = SimpleNamespace(id="seeded", metric_mode="max",
                                       kind="generic", metric_label="score")
        trainer.episode = trainer.total_steps = trainer.update_count = 0
        trainer.sps = 0.0
        trainer.history = []
        trainer.best_reward = trainer.best_metric = None
        trainer.ghost = trainer._learning = None
        trainer.seed = 77
        trainer.settings = SimpleNamespace(eval_episodes=1)
        trainer.device = torch.device("cpu")
        trainer.max_episodes = 1
        trainer.checkpoint_every_n = 1
        trainer._stop = trainer_module.threading.Event()
        trainer._thread = None
        trainer.emit = lambda message: None
        trainer._run_eval = lambda: {
            "reward": 0.0, "reward_std": 0.0, "metric": None,
            "metric_std": None, "episodes": 1, "success_rate": 0.0,
            "seed": 77, "trajectory": [],
        }

        trainer._run()

        expected_rng = random.Random(77)
        expected_rng.random()  # initial episode start
        expected_next_start = expected_rng.random()
        restored_rng = random.Random()
        restored_rng.setstate(registry.rng_state)
        self.assertEqual(restored_rng.random(), expected_next_start)


if __name__ == "__main__":
    unittest.main()
