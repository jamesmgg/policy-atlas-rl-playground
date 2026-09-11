"""Scientific contract for Traffic's learnable v17 training design."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.envs import driving  # noqa: E402
from app.ppo.agent import PPOAgent  # noqa: E402
from app.scenarios import get_spec  # noqa: E402
from app.settings import Settings  # noqa: E402
from app import trainer as trainer_module  # noqa: E402


class TrafficV17ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_spec("traffic-rush")

    def test_failure_terminal_retains_physical_course_potential(self) -> None:
        delta = getattr(driving, "retained_course_potential_delta", None)
        self.assertTrue(callable(delta), "retained potential delta is missing")

        env = self.spec.make_env(False)
        cfg = env.reward_cfg
        self.assertTrue(getattr(cfg, "retain_terminal_course_potential", False))
        self.assertFalse(cfg.terminal_zero_course_potential)

        paths = (
            (0.0, (10.0, 30.0, 80.0)),
            (37.5, (50.0, 200.0, 800.0, 900.0)),
            (104.0, (90.0, 20.0)),
        )
        for start, states in paths:
            with self.subTest(start=start, states=states):
                total = 0.0
                previous = start
                for current in states:
                    total += delta(previous, current)
                    previous = current
                self.assertAlmostEqual(total, states[-1] - start)

        protocol = env.course_reward_potential_protocol()
        self.assertEqual(protocol["terminal_potential"],
                         "retained physical end potential")
        self.assertEqual(protocol["episode_sum"],
                         "potential(end_state) - potential(start_state)")
        self.assertEqual(protocol["discount_factor"], 1.0)

    def test_predeclared_episode_schedule_rehearses_a_physical_near_pass(self) -> None:
        schedule = getattr(self.spec, "training_schedule", None)
        self.assertIsNotNone(schedule, "Traffic episode schedule is undisclosed")
        schedule_protocol = schedule.protocol()
        self.assertEqual(schedule_protocol["advancement"], {
            "signal": "one-based completed-training-episode schedule",
            "evaluation_conditioned": False,
            "fixed_before_training": True,
        })
        self.assertEqual(
            [(phase["id"], phase["episodes"], phase["sampling"])
             for phase in schedule_protocol["phases"]],
            [
                ("near_pass_bootstrap", [1, 200], {
                    "near_pass_rehearsal": 0.75,
                    "nested_speed_control": 0.25,
                }),
                ("near_pass_bridge", [201, 400], {
                    "near_pass_rehearsal": 0.25,
                    "nested_speed_control": 0.75,
                }),
                ("canonical_speed_consolidation", [401, None], {
                    "nested_speed_control": 1.0,
                }),
            ],
        )
        self.assertEqual(
            getattr(driving, "TRAFFIC_BOT3_REHEARSAL_SPEED", None), 12.0)
        self.assertEqual(
            getattr(driving, "TRAFFIC_BOT3_REHEARSAL_SCHEDULE", None),
            ((1, 200, 0.75), (201, 400, 0.25), (401, None, 0.0)),
        )

        expected = {
            100: (12.0, (1_400, 1_600)),
            300: (12.0, (400, 600)),
            401: (18.0, (2_000, 2_000)),
        }
        for episode, (scheduled_speed, bounds) in expected.items():
            with self.subTest(episode=episode):
                env = self.spec.make_training_env()
                env.rng.seed(17_000 + episode)
                speeds = []
                for _ in range(2_000):
                    env.reset_for_training_episode(episode)
                    speeds.append(env.features.bots[2].speed)
                count = speeds.count(scheduled_speed)
                self.assertGreaterEqual(count, bounds[0])
                self.assertLessEqual(count, bounds[1])
                self.assertTrue(set(speeds).issubset({12.0, 18.0}))

        probe = self.spec.make_training_env()
        probe.rng.seed(17_100)
        while True:
            probe.reset_for_training_episode(100)
            if probe.features.bots[2].speed == 12.0:
                break
        diagnostic = probe.bot3_catchup_diagnostics()
        self.assertEqual(probe._bot_passed, [True, True, False])
        self.assertEqual(probe.episode_reward, 0.0)
        self.assertAlmostEqual(diagnostic["gap_m"], 71.2633, places=3)
        self.assertAlmostEqual(
            diagnostic["required_average_speed_mps"], 13.4652, places=3)

        with self.assertRaisesRegex(ValueError, "one-based"):
            probe.reset_for_training_episode(0)

    def test_canonical_observation_discloses_strategy_relevant_guidance(self) -> None:
        env = self.spec.make_env(False)
        guidance = getattr(env, "traffic_guidance", None)
        self.assertTrue(callable(guidance), "Traffic guidance is missing")
        observation = env.reset()

        self.assertEqual(env.obs_dim, 41)
        self.assertEqual(len(observation), 41)
        self.assertEqual(
            self.spec.observation_dimensions[-4:],
            (
                "Traffic pursuit target lane fraction",
                "Traffic pursuit heading error / max steering angle",
                "Traffic physics speed target / max speed",
                "remaining horizon fraction",
            ),
        )
        initial = guidance()
        np.testing.assert_allclose(
            observation[-4:-1],
            np.array([
                initial["target_lane_fraction"],
                initial["heading_error_normalized"],
                initial["speed_target_fraction"],
            ], dtype=np.float32),
        )

        env._bot_passed = [True, True, False]
        env._bot_arcs[2] = (env.s_prev + 50.0) % env.track.total_length
        nearby = guidance()
        self.assertEqual(nearby["target_bot_index"], 3)
        self.assertEqual(nearby["target_lane_fraction"], -0.72)
        self.assertLessEqual(nearby["speed_target_mps"], 62.0)

        env._bot_passed[2] = True
        cleared = guidance()
        self.assertIsNone(cleared["target_bot_index"])
        self.assertEqual(cleared["target_lane_fraction"], 0.0)

    def test_demonstration_policy_starts_with_narrow_residual_exploration(self) -> None:
        initialization = self.spec.actor_initialization
        self.assertIsNotNone(initialization)
        self.assertEqual(initialization.continuous_log_std, (-2.0, -2.0))
        self.assertAlmostEqual(float(np.exp(-2.0)), 0.13533528, places=7)

    def test_disclosed_training_teacher_solves_its_demonstration_seeds(self) -> None:
        teacher = getattr(driving, "traffic_reference_action", None)
        self.assertTrue(callable(teacher), "Traffic training teacher is missing")

        probe = self.spec.make_env(True)
        probe.rng.seed(900_000)
        probe.reset()
        guidance = probe.traffic_guidance()
        action = teacher(probe)
        speed_error = guidance["speed_target_mps"] - probe.car.v_long
        expected_throttle = np.clip(
            speed_error / (8.0 if speed_error >= 0.0 else 10.0),
            -1.0,
            1.0,
        )
        expected_steering = np.clip(
            1.6 * guidance["heading_error_normalized"]
            - 0.3 * probe.car.v_lat / 25.0,
            -1.0,
            1.0,
        )
        np.testing.assert_allclose(
            action,
            np.array([expected_throttle, expected_steering, 0.0],
                     dtype=np.float32),
        )

        outcomes = []
        for index in range(10):
            env = self.spec.make_env(True)
            env.rng.seed(900_000 + index)
            observation = env.reset()
            del observation
            done = False
            while not done:
                _, _, done, _ = env.step(teacher(env))
            outcomes.append(env.episode_summary())

        self.assertTrue(all(item["success"] for item in outcomes))
        self.assertTrue(all(item["metric"] == 3.0 for item in outcomes))
        self.assertLess(max(item["steps"] for item in outcomes), 1_650)

    def test_dagger_warm_start_is_disjoint_disclosed_and_learned_at_inference(self) -> None:
        warm_start = getattr(self.spec, "actor_warm_start", None)
        self.assertIsNotNone(warm_start, "Traffic DAgger warm start is missing")
        protocol = warm_start.protocol()
        self.assertEqual(protocol["id"], "traffic-pure-pursuit-dagger-v1")
        self.assertEqual(protocol["role"], "actor DAgger warm start")
        self.assertFalse(protocol["pure_model_free_from_scratch"])
        self.assertFalse(protocol["expert"]["used_at_inference"])
        self.assertEqual(protocol["dataset"]["seed_base"], 900_000)
        self.assertEqual(protocol["dataset"]["episodes"], 30)
        self.assertEqual(protocol["dataset"]["state_stride"], 2)
        self.assertEqual(protocol["dagger"], {
            "rounds": 2,
            "rollout_seed_base": 920_000,
            "round_seed_stride": 1_000,
            "episodes_per_round": 40,
            "state_stride": 2,
            "expert_role": "labels learned-policy rollout states only",
        })
        self.assertEqual(protocol["optimizer"]["loss"], {
            "throttle_mse_weight": 1.0,
            "steering_mse_weight": 5.0,
            "binary_drift_bce_weight": 0.1,
        })

        training_seeds = set(range(900_000, 900_030))
        for round_index in range(2):
            start = 920_000 + round_index * 1_000
            training_seeds.update(range(start, start + 40))
        protected = (
            set(range(100_000, 100_100))
            | set(range(600_000, 600_100))
            | {700_000 + frontier * 1_000 + index
               for frontier in (11, 9, 3, 0) for index in range(10)}
            | {800_000 + stage * 1_000 + index
               for stage in range(3) for index in range(10)}
        )
        self.assertTrue(training_seeds.isdisjoint(protected))

        torch.manual_seed(42)
        env = self.spec.make_env(False)
        agent = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary,
            torch.device("cpu"),
            actor_initialization=self.spec.actor_initialization,
        )
        diagnostics = warm_start.apply(agent)
        self.assertGreater(diagnostics["samples"], 60_000)
        self.assertRegex(diagnostics["dataset_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(diagnostics["dagger_rounds"]), 2)

        def evaluate(seeds: range) -> list[dict]:
            results = []
            for seed in seeds:
                evaluation = self.spec.make_env(True)
                evaluation.rng.seed(seed)
                observation = evaluation.reset()
                done = False
                while not done:
                    action, _, _ = agent.select_action(
                        observation, deterministic=True)
                    observation, _, done, _ = evaluation.step(action)
                results.append(evaluation.episode_summary())
            return results

        with patch.object(
                driving, "traffic_reference_action",
                side_effect=AssertionError("expert leaked into inference")):
            fixed = evaluate(range(100_000, 100_100))
            fresh = evaluate(range(960_000, 960_100))
        self.assertTrue(all(item["success"] for item in fixed))
        self.assertTrue(all(item["success"] for item in fresh))
        self.assertTrue(all(item["metric"] == 3.0
                            for item in fixed + fresh))

    def test_checkpoint_discloses_assistance_and_restore_overrides_warm_start(self) -> None:
        warm_start = self.spec.actor_warm_start
        realized = {
            "samples": 123,
            "final_mse": 0.001,
            "final_binary_bce": 0.002,
            "dataset_sha256": "a" * 64,
            "initialization_seed": 42,
            "dagger_rounds": [{"round": 1}, {"round": 2}],
        }

        def cheap_apply(_contract, agent):
            with torch.no_grad():
                agent.network.mu.bias.fill_(0.123)
            return dict(realized)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
                type(warm_start), "apply", cheap_apply):
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"traffic-rush"}')
            settings = Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            )
            trainer = trainer_module.Trainer(settings)
            self.assertEqual(trainer.actor_warm_start_diagnostics, realized)
            with torch.no_grad():
                trainer.agent.network.mu.bias.fill_(0.777)
            trainer.episode = 25
            trainer._run_eval = lambda: {
                "reward": 0.0, "reward_std": 0.0,
                "metric": 0.0, "metric_std": 0.0,
                "failure_progress": 0.0, "episodes": 1,
                "success_rate": 0.0, "success_ci_low": 0.0,
                "success_ci_high": 1.0,
                "evaluation_suite": trainer_module.evaluation_suite_id(1),
                "seed": 42, "trajectory": [],
            }
            trainer._run_training_control_eval = lambda: None
            trainer._run_training_curriculum_eval = lambda: None
            trainer._save_checkpoint()
            meta = trainer.registry.list()[0]

            self.assertEqual(meta["schema_version"], 18)
            self.assertEqual(meta["protocol"]["version"], 20)
            self.assertEqual(meta["protocol"]["algorithm"],
                             "demonstration-assisted PPO")
            self.assertEqual(meta["protocol"]["actor_warm_start"], {
                "contract": warm_start.protocol(),
                "realized": realized,
            })
            self.assertFalse(
                meta["protocol"]["actor_warm_start"]["contract"]
                ["pure_model_free_from_scratch"])

            restored = trainer_module.Trainer(settings)
            self.assertEqual(restored.episode, 25)
            np.testing.assert_allclose(
                restored.agent.network.mu.bias.detach().cpu().numpy(),
                np.array([0.777, 0.777], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
