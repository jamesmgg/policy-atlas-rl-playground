from __future__ import annotations

import copy
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.checkpoints import (CheckpointRegistry,  # noqa: E402
                             IncompatibleCheckpointError)
import app.envs.driving as driving_module  # noqa: E402
from app.envs.driving import RewardConfig  # noqa: E402
from app.ppo.agent import PPOAgent  # noqa: E402
from app.ppo.buffer import RolloutBuffer  # noqa: E402
from app.scenarios import get_spec, list_specs  # noqa: E402
from app.settings import Settings  # noqa: E402
import app.trainer as trainer_module  # noqa: E402


class TrafficStageCurriculumAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.traffic = get_spec("traffic-rush")

    def forced_rolling_start(self, checkpoint: int):
        env = self.traffic.training_curriculum.make_evaluation_env(checkpoint)
        env.rng.seed(42)
        env.reset()
        return env

    def test_training_rehearses_each_overtake_stage(self) -> None:
        env = self.traffic.make_training_env()

        self.assertEqual(env.start_line_probability, 0.0)
        self.assertEqual(env.rolling_checkpoint_indices, (11,))
        self.assertEqual(
            env.training_curriculum_state()["frontier"], 11)

        env.rng.seed(42)
        sampled = set()
        for _ in range(240):
            env.reset()
            sampled.add(env.track.checkpoints.index(env.idx))
        self.assertEqual(sampled, {11})

    def test_stage_starts_allow_reaction_time_and_reconstruct_task_state(self) -> None:
        first = self.forced_rolling_start(3)
        second = self.forced_rolling_start(9)
        third = self.forced_rolling_start(11)

        self.assertEqual(first._bot_passed, [False, False, False])
        self.assertEqual(second._bot_passed, [True, False, False])
        self.assertEqual(third._bot_passed, [True, True, False])
        for env, bot_index in ((first, 0), (second, 1)):
            gap = env._bot_gap(bot_index)
            closing_speed = env.car.speed - env.features.bots[bot_index].speed
            self.assertGreater(gap, 100.0)
            self.assertGreater(gap / closing_speed, 3.5)

        remaining_seconds = (third.max_steps - third.steps) * third.dt
        third_gap = third._bot_gap(2)
        required_speed = third.features.bots[2].speed + (
            third_gap / remaining_seconds
        )
        reference_speed = third.track.total_length / third._rolling_lap_seconds
        self.assertGreater(remaining_seconds, 48.0)
        self.assertLess(required_speed, reference_speed)

    def test_canonical_evaluation_task_and_other_scenarios_are_unchanged(self) -> None:
        env = self.traffic.make_env(False)

        self.assertFalse(env.random_start)
        self.assertEqual(env.idx, 0)
        self.assertEqual(env.steps, 0)
        self.assertEqual(env._bot_passed, [False, False, False])
        self.assertEqual(env.max_steps, 2250)
        self.assertEqual(
            env.reward_cfg,
            RewardConfig(
                drift_corner=0.0,
                overtake=8.0,
                contact=-40.0,
                stall=-40.0,
                wrong_way=-40.0,
                timeout=-40.0,
                terminalize_failure_time=True,
                retain_terminal_course_potential=True,
            ),
        )
        # The live differences remain dense and retain real local progress at
        # failure, so later successful transitions are not erased at terminal.
        self.assertEqual(
            (
                env.reward_cfg.progress,
                env.reward_cfg.checkpoint,
                env.reward_cfg.lap,
                env.reward_cfg.overtake,
            ),
            (0.05, 3.0, 30.0, 8.0),
        )
        self.assertEqual(
            tuple((bot.start_frac, bot.speed, bot.lat_frac)
                  for bot in env.features.bots),
            ((0.25, 18.0, -0.4), (0.50, 24.0, 0.0),
             (0.75, 30.0, 0.4)),
        )
        self.assertEqual(
            self.traffic.success,
            "Overtake all three traffic cars in one episode.",
        )
        for spec in list_specs():
            if spec.kind == "driving" and spec.id != self.traffic.id:
                with self.subTest(scenario=spec.id):
                    self.assertNotEqual(
                        spec.make_training_env().rolling_checkpoint_indices,
                        (3, 9, 11),
                    )

    def test_failure_clock_component_is_timing_invariant_for_every_cause(self) -> None:
        clock_cost = getattr(
            driving_module, "terminal_failure_time_cost", None)
        self.assertTrue(
            callable(clock_cost),
            "Traffic needs an explicit, testable failure clock contract",
        )
        env = self.traffic.make_env(False)
        cfg = env.reward_cfg
        self.assertEqual(cfg.time, -0.02)

        for cause in ("collision", "contact", "stall", "wrong_way", "timeout"):
            for terminal_step in (1, 2, env.max_steps - 1, env.max_steps):
                with self.subTest(cause=cause, terminal_step=terminal_step):
                    elapsed_clock = cfg.time * terminal_step
                    correction = clock_cost(
                        cfg,
                        cause=cause,
                        steps=terminal_step,
                        max_steps=env.max_steps,
                    )
                    self.assertAlmostEqual(
                        elapsed_clock + correction,
                        cfg.time * env.max_steps,
                    )

    def test_successful_completion_keeps_elapsed_time_pressure(self) -> None:
        clock_cost = getattr(
            driving_module, "terminal_failure_time_cost", None)
        self.assertTrue(callable(clock_cost))
        env = self.traffic.make_env(False)
        cfg = env.reward_cfg

        for terminal_step in (1, 400, env.max_steps):
            with self.subTest(terminal_step=terminal_step):
                correction = clock_cost(
                    cfg,
                    cause="complete",
                    steps=terminal_step,
                    max_steps=env.max_steps,
                )
                self.assertEqual(correction, 0.0)
                self.assertEqual(
                    cfg.time * terminal_step + correction,
                    cfg.time * terminal_step,
                )

    def test_runtime_failure_correction_uses_post_step_clock_without_off_by_one(self) -> None:
        penalized = self.traffic.make_env(False)
        self.assertTrue(
            getattr(penalized.reward_cfg, "terminalize_failure_time", False))
        baseline = copy.deepcopy(penalized)
        object.__setattr__(
            baseline.reward_cfg, "terminalize_failure_time", False)
        for env in (penalized, baseline):
            env._stall_steps = driving_module.STALL_WINDOW - 1
            env._stall_anchor_progress = env.progress

        _, penalized_reward, penalized_done, _ = penalized.step(
            np.zeros(3, dtype=np.float32))
        _, baseline_reward, baseline_done, _ = baseline.step(
            np.zeros(3, dtype=np.float32))

        self.assertTrue(penalized_done and baseline_done)
        self.assertEqual(penalized.cause, "stall")
        self.assertEqual(penalized.steps, 1)
        self.assertAlmostEqual(
            penalized_reward - baseline_reward,
            penalized.reward_cfg.time * (penalized.max_steps - 1),
        )

    def test_failure_clock_regularizer_is_disclosed_and_traffic_only(self) -> None:
        for spec in list_specs():
            if spec.kind != "driving":
                continue
            with self.subTest(scenario=spec.id):
                enabled = getattr(
                    spec.make_env(False).reward_cfg,
                    "terminalize_failure_time",
                    False,
                )
                self.assertEqual(enabled, spec.id == self.traffic.id)

        self.assertTrue(any(
            "canonical failures pay the full -45 horizon time budget" in term
            for term in self.traffic.reward_terms
        ))

    def test_timeout_carries_the_same_failure_cost_as_unsafe_termination(self) -> None:
        """Waiting out the clock must not dominate an immediate failed attempt."""
        env = self.traffic.make_env(False)
        gamma = getattr(
            self.traffic, "training_discount_factor", trainer_module.GAMMA)
        timeout_cost = getattr(env.reward_cfg, "timeout", 0.0)
        unsafe_costs = (
            env.reward_cfg.collision,
            env.reward_cfg.contact,
            env.reward_cfg.stall,
            env.reward_cfg.wrong_way,
        )

        self.assertEqual(gamma, 1.0)
        self.assertEqual(timeout_cost, -40.0)
        self.assertTrue(all(cost == timeout_cost for cost in unsafe_costs))
        self.assertIn(
            "-40 task-deadline timeout penalty",
            self.traffic.reward_terms,
        )

        immediate_failure = unsafe_costs[0]
        delayed_timeout = gamma ** (env.max_steps - 1) * timeout_cost
        self.assertAlmostEqual(delayed_timeout, immediate_failure)

        legacy_delayed = (
            trainer_module.GAMMA ** (env.max_steps - 1) * immediate_failure
        )
        self.assertGreater(
            legacy_delayed,
            immediate_failure,
            "the discounted v11 objective makes a delayed equal-cost failure safer",
        )

    def test_runtime_timeout_applies_the_declared_terminal_cost(self) -> None:
        penalized = self.traffic.make_env(False)
        baseline = copy.deepcopy(penalized)
        object.__setattr__(penalized.reward_cfg, "timeout", -40.0)
        object.__setattr__(baseline.reward_cfg, "timeout", 0.0)
        for env in (penalized, baseline):
            env.steps = env.max_steps - 1
            env._stall_steps = 0

        _, penalized_reward, penalized_done, penalized_info = penalized.step(
            np.zeros(3, dtype=np.float32))
        _, baseline_reward, baseline_done, _ = baseline.step(
            np.zeros(3, dtype=np.float32))

        self.assertTrue(penalized_done and baseline_done)
        self.assertEqual(penalized.cause, "timeout")
        self.assertTrue(penalized_info["task_deadline"])
        self.assertAlmostEqual(penalized_reward - baseline_reward, -40.0)

    def test_scenario_discount_is_wired_through_reward_bootstrap_and_gae(self) -> None:
        self.assertIn(
            "gamma", inspect.signature(trainer_module.training_reward).parameters)
        self.assertAlmostEqual(
            trainer_module.training_reward(
                250.0,
                next_value=3.0,
                done=True,
                info={"truncated": True, "task_deadline": False},
                gamma=1.0,
            ),
            5.5,
        )

        reward_gammas: list[float | None] = []
        gae_gammas: list[float] = []

        class OneStepDeadlineEnv:
            obs_dim = 1
            n_continuous = 1
            n_binary = 0
            max_steps = 1
            dt = 0.1

            def reset(self):
                self.episode_reward = 0.0
                return np.array([0.0], dtype=np.float32)

            def step(self, action):
                del action
                self.episode_reward = -40.0
                return np.array([0.0], dtype=np.float32), -40.0, True, {
                    "truncated": True,
                    "task_deadline": True,
                }

            def episode_summary(self):
                return {"reward": -40.0, "steps": 1, "cause": "timeout",
                        "metric": 0.0, "success": False}

            def frame_payload(self):
                return {}

        class RecordingAgent:
            act_dim = 1

            def select_action(self, observation, deterministic=False):
                del observation, deterministic
                return np.array([0.0], dtype=np.float32), 0.0, 0.0

            def get_value(self, observation):
                del observation
                return 7.0

            def update(self, buffer):
                del buffer
                return {"policy_loss": 0.0, "value_loss": 0.0,
                        "entropy": 0.0, "approx_kl": 0.0,
                        "clip_frac": 0.0}

        class RecordingBuffer(RolloutBuffer):
            def compute_gae(self, last_value, last_done, gamma=0.995,
                            gae_lambda=0.95):
                gae_gammas.append(gamma)
                return super().compute_gae(
                    last_value, last_done, gamma=gamma,
                    gae_lambda=gae_lambda)

        def record_training_reward(reward, next_value, done, info, gamma=None):
            del next_value, done, info
            reward_gammas.append(gamma)
            return reward * trainer_module.TRAINING_REWARD_SCALE

        trainer = object.__new__(trainer_module.Trainer)
        trainer.env = OneStepDeadlineEnv()
        trainer.agent = RecordingAgent()
        trainer.spec = SimpleNamespace(
            id="traffic-rush", metric_mode="max", kind="driving",
            metric_label="overtakes", training_discount_factor=1.0,
        )
        trainer.episode = trainer.total_steps = trainer.update_count = 0
        trainer.sps = 0.0
        trainer.history = []
        trainer.best_reward = trainer.best_metric = None
        trainer.ghost = trainer._learning = None
        trainer.seed = 42
        trainer.settings = SimpleNamespace(eval_episodes=1)
        trainer.device = torch.device("cpu")
        trainer.max_episodes = 1
        trainer.checkpoint_every_n = 100
        trainer._stop = trainer_module.threading.Event()
        trainer._thread = None
        trainer.emit = lambda message: None
        trainer._save_checkpoint = lambda: None

        with patch.object(
                trainer_module, "training_reward",
                side_effect=record_training_reward), patch.object(
                    trainer_module, "RolloutBuffer", RecordingBuffer):
            trainer._run()

        self.assertEqual(reward_gammas, [1.0])
        self.assertEqual(gae_gammas, [1.0])

    def test_undiscounted_training_objective_is_explicitly_scoped(self) -> None:
        self.assertEqual(
            self.traffic.info().get("training_discount_factor"), 1.0)
        for spec in list_specs():
            with self.subTest(scenario=spec.id):
                expected = 1.0 if spec.id in {
                    "traffic-rush", "lunar-lander", "drone-hover", "orbital-docking",
                } else 0.995
                self.assertEqual(spec.training_discount_factor, expected)

    def test_schema_eighteen_refuses_a_schema_seventeen_traffic_checkpoint(self) -> None:
        env = self.traffic.make_env(False)
        agent = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = CheckpointRegistry(
                root, self.traffic.id,
                schema_version=self.traffic.checkpoint_schema)
            old = CheckpointRegistry(root, self.traffic.id, schema_version=17)
            old.save(25, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": 2.0, "trajectory": [],
            })

            self.assertEqual(self.traffic.checkpoint_schema, 18)
            self.assertEqual(current.list(), [])
            with self.assertRaises(IncompatibleCheckpointError):
                current.load_into(
                    25,
                    agent,
                    expected_engine="current-engine",
                    expected_evaluation_suite="current-suite",
                )

    def test_protocol_v18_discloses_the_terminalized_failure_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"traffic-rush"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901,
                checkpoint_dir=root,
                checkpoint_every_n=25,
                max_episodes=10,
                use_gpu=False,
                seed=42,
                eval_episodes=1,
            ))
            trainer._run_eval = lambda: {
                "reward": 0.0,
                "reward_std": 0.0,
                "metric": 0.0,
                "metric_std": 0.0,
                "failure_progress": 0.0,
                "episodes": 1,
                "success_rate": 0.0,
                "success_ci_low": 0.0,
                "success_ci_high": 1.0,
                "evaluation_suite": "test-suite",
                "seed": 42,
                "trajectory": [],
            }
            trainer._save_checkpoint()
            protocol = trainer.registry.list()[0]["protocol"]

        self.assertEqual(
            protocol.get("failure_clock_regularizer"),
            {
                "enabled": True,
                "time_per_step": -0.02,
                "failure_causes": [
                    "collision", "contact", "stall", "wrong_way", "timeout",
                ],
                "remaining_cost_formula": (
                    "time_per_step * max(horizon_steps - terminal_step, 0)"
                ),
                "canonical_failure_clock_total": -45.0,
                "rolling_start_semantics": (
                    "constant over the remaining suffix from each sampled "
                    "start; no reset reward"
                ),
                "successful_completion": "elapsed live-step time cost only",
            },
        )
        self.assertEqual(protocol["version"], 19)
        self.assertEqual(protocol["gamma"], 1.0)
        self.assertEqual(protocol["task_horizon_steps"], 2250)
        self.assertEqual(protocol["task_horizon_seconds"], 90.0)
        self.assertEqual(
            protocol["training_start_distribution"],
            self.traffic.training_start_distribution,
        )


if __name__ == "__main__":
    unittest.main()
