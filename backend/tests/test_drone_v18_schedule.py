"""Scientific contract for the Drone v18 demonstration + schedule revision."""
from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).parents[1]))

from app.checkpoints import CheckpointRegistry
from app.envs import drone
from app.ppo.agent import PPOAgent
from app.scenarios import get_spec, list_specs
from app.settings import Settings
from app.trainer import capture_rng_state, restore_rng_state
import app.trainer as trainer_module


def course_origin(segment: int) -> tuple[float, float]:
    return drone.START if segment == 0 else drone.WAYPOINTS[segment - 1]


def evaluate_policy(agent: PPOAgent, seeds: range) -> list[dict]:
    spec = get_spec("drone-hover")
    results = []
    for seed in seeds:
        env = spec.make_env(True)
        env.rng.seed(seed)
        obs = env.reset()
        for _ in range(env.max_steps):
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done:
                break
        results.append(env.episode_summary())
    return results


class DroneV18ScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = get_spec("drone-hover")
        self.schedule = getattr(self.spec, "training_schedule", None)
        self.warm_start = getattr(self.spec, "actor_warm_start", None)

    def test_schedule_is_predeclared_and_never_evaluation_conditioned(self) -> None:
        self.assertIsNone(self.spec.training_curriculum)
        self.assertIsNotNone(self.schedule)
        protocol = self.schedule.protocol()
        self.assertEqual(protocol["id"], "drone-all-direction-schedule-v1")
        self.assertEqual(protocol["advancement"], {
            "signal": "one-based completed-training-episode schedule",
            "evaluation_conditioned": False,
            "fixed_before_training": True,
        })
        self.assertEqual(
            [(phase["id"], phase["episodes"])
             for phase in protocol["phases"]],
            [
                ("capture_approach", [1, 200]),
                ("half_segments", [201, 450]),
                ("full_handoffs", [451, 800]),
                ("canonical_consolidation", [801, None]),
            ],
        )
        self.assertEqual(protocol["phases"][0]["sampling"], {"approach": 1.0})
        self.assertEqual(protocol["phases"][1]["sampling"], {
            "approach": 0.3, "half": 0.7,
        })
        self.assertEqual(protocol["phases"][2]["sampling"], {
            "approach": 0.1, "half": 0.2, "handoff": 0.7,
        })
        self.assertEqual(protocol["phases"][3]["sampling"], {
            "approach": 0.05, "half": 0.05,
            "handoff": 0.3, "canonical": 0.6,
        })
        self.assertIn("fixed full-course", protocol["checkpoint_selection"])

    def test_scenario_contract_keeps_canonical_training_discount_api(self) -> None:
        self.assertEqual(self.spec.training_discount_factor, 1.0)
        self.assertEqual(self.spec.info()["training_discount_factor"], 1.0)
        self.assertFalse(hasattr(self.spec, "gamma"))

    def test_episode_bands_sample_the_disclosed_mixtures_and_all_directions(self) -> None:
        training = self.spec.make_training_env()
        self.assertTrue(training.episode_schedule)

        cases = (
            (100, {"approach": 1.0}),
            (300, {"approach": 0.3, "half": 0.7}),
            (600, {"approach": 0.1, "half": 0.2, "handoff": 0.7}),
            (1200, {"approach": 0.05, "half": 0.05,
                    "handoff": 0.3, "canonical": 0.6}),
        )
        for episode, expected in cases:
            training.rng.seed(episode)
            modes: Counter[str] = Counter()
            segments: Counter[int] = Counter()
            for _ in range(5_000):
                training.reset_for_training_episode(episode)
                modes[training.training_mode] += 1
                if training.training_segment is not None:
                    segments[training.training_segment] += 1
            for mode, probability in expected.items():
                self.assertAlmostEqual(
                    modes[mode] / 5_000, probability, delta=0.025,
                    msg=f"episode {episode} mode {mode}",
                )
            unexpected = set(modes) - set(expected)
            self.assertEqual(unexpected, set())
            if set(expected) != {"canonical"}:
                self.assertEqual(set(segments), set(range(5)))

    def test_single_segment_start_geometry_expands_without_changing_handoff(self) -> None:
        for segment in range(5):
            target = drone.WAYPOINTS[segment]
            origin = course_origin(segment)
            segment_distance = math.dist(origin, target)

            approach = drone.DroneEnv(
                jitter=False, episode_schedule=True,
                forced_training_mode="approach", forced_start_segment=segment,
            )
            self.assertAlmostEqual(
                math.dist((approach.x, approach.y), target),
                min(drone.APPROACH_REMAINING_DISTANCE, segment_distance),
                places=9,
            )
            self.assertEqual((approach.vx, approach.vy), (0.0, 0.0))

            half = drone.DroneEnv(
                jitter=False, episode_schedule=True,
                forced_training_mode="half", forced_start_segment=segment,
            )
            self.assertAlmostEqual(
                math.dist((half.x, half.y), target),
                segment_distance * (1.0 - drone.HALF_SEGMENT_PROGRESS),
                places=9,
            )
            self.assertEqual((half.vx, half.vy), (0.0, 0.0))

            handoff = drone.DroneEnv(
                jitter=False, episode_schedule=True,
                forced_training_mode="handoff", forced_start_segment=segment,
            )
            self.assertEqual((handoff.x, handoff.y), origin)
            if segment == 0:
                self.assertEqual(handoff.vx, 0.0)
            else:
                inbound_dx = course_origin(segment)[0] - course_origin(segment - 1)[0]
                self.assertGreaterEqual(abs(handoff.vx), 60.0)
                self.assertLessEqual(abs(handoff.vx), 100.0)
                self.assertGreater(handoff.vx * inbound_dx, 0.0)
            self.assertTrue(approach.single_segment_episode)
            self.assertTrue(half.single_segment_episode)
            self.assertTrue(handoff.single_segment_episode)
            expected_steps = (0 if segment == 0
                              else drone.WAYPOINT_START_STEPS[segment - 1])
            self.assertEqual(approach.steps, expected_steps)
            self.assertEqual(half.steps, expected_steps)
            self.assertEqual(handoff.steps, expected_steps)

    def test_training_segment_capture_is_not_reported_as_canonical_completion(self) -> None:
        env = drone.DroneEnv(
            jitter=False, episode_schedule=True,
            forced_training_mode="approach", forced_start_segment=2,
        )
        env.x, env.y = drone.WAYPOINTS[2]
        env.vx = env.vy = env.theta = env.omega = 0.0
        env._shaping_potential_prev = env._shaping_potential()
        _, reward, done, _ = env.step(
            np.full(2, drone.HOVER_ACTION, dtype=np.float64))

        summary = env.episode_summary()
        self.assertTrue(done)
        self.assertEqual(summary["cause"], "training_segment_complete")
        self.assertTrue(summary["success"])
        self.assertFalse(summary["canonical_success"])
        self.assertEqual(summary["metric"], 1.0)
        self.assertAlmostEqual(reward, 20.0, places=9)

    def test_canonical_task_start_horizon_and_completion_are_unchanged(self) -> None:
        canonical = self.spec.make_env(False)
        self.assertFalse(canonical.episode_schedule)
        self.assertFalse(canonical.single_segment_episode)
        self.assertEqual((canonical.x, canonical.y), drone.START)
        self.assertEqual((canonical.vx, canonical.vy), (0.0, 0.0))
        self.assertEqual(canonical.k, 0)
        self.assertEqual(canonical.steps, 0)
        self.assertEqual(canonical.max_steps, 900)

        final = drone.DroneEnv(jitter=False, forced_start_segment=4)
        final.x, final.y = drone.WAYPOINTS[-1]
        final.vx = final.vy = final.theta = final.omega = 0.0
        final._shaping_potential_prev = final._shaping_potential()
        _, reward, done, _ = final.step(
            np.full(2, drone.HOVER_ACTION, dtype=np.float64))
        self.assertTrue(done)
        self.assertEqual(final.cause, "complete")
        self.assertTrue(final.episode_summary()["canonical_success"])
        self.assertGreater(reward, 69.0)

    def test_observation_exposes_in_scale_physics_control_errors(self) -> None:
        env = drone.DroneEnv(jitter=False, forced_start_segment=4)
        env.vx, env.vy = 100.0, -35.0
        env.theta, env.omega = 0.25, -0.4
        obs = env._obs()
        desired_vx, desired_vy = env._desired_velocity_target()
        desired_tilt = env._desired_tilt_target(desired_vx)

        self.assertEqual(env.obs_dim, 12)
        self.assertEqual(obs.shape, (12,))
        self.assertAlmostEqual(float(obs[2]), 1.0, places=6)
        self.assertAlmostEqual(float(obs[3]), -0.35, places=6)
        self.assertAlmostEqual(
            float(obs[4]), (desired_vx - env.vx) / 100.0, places=6)
        self.assertAlmostEqual(
            float(obs[5]), (desired_vy - env.vy) / 100.0, places=6)
        self.assertAlmostEqual(
            float(obs[6]), (desired_tilt - env.theta) / drone.TIP_OVER,
            places=6,
        )
        self.assertLessEqual(abs(float(obs[2])), 1.0)

    def test_schedule_state_and_rng_restore_replay_pending_resets_exactly(self) -> None:
        original = self.spec.make_training_env()
        original.rng.seed(123)
        original.reset_for_training_episode(800)
        state = capture_rng_state(original)
        before = copy.deepcopy(original.training_curriculum_state())

        expected = []
        for episode in range(801, 821):
            obs = original.reset_for_training_episode(episode)
            expected.append((
                original.training_phase,
                original.training_mode,
                original.training_segment,
                obs.copy(),
            ))

        restored = self.spec.make_training_env()
        restored.rng.seed(999)
        restore_rng_state(state, restored)
        self.assertEqual(restored.training_curriculum_state(), before)
        actual = []
        for episode in range(801, 821):
            obs = restored.reset_for_training_episode(episode)
            actual.append((
                restored.training_phase,
                restored.training_mode,
                restored.training_segment,
                obs.copy(),
            ))
        for left, right in zip(expected, actual):
            self.assertEqual(left[:3], right[:3])
            np.testing.assert_array_equal(left[3], right[3])

    def test_rejected_schedule_restore_is_atomic(self) -> None:
        env = self.spec.make_training_env()
        env.rng.seed(7)
        env.reset_for_training_episode(600)
        before = copy.deepcopy(env.training_curriculum_state())
        invalid = copy.deepcopy(before)
        invalid["phase"] = "canonical_consolidation"
        with self.assertRaisesRegex(ValueError, "phase"):
            env.restore_training_curriculum_state(invalid)
        self.assertEqual(env.training_curriculum_state(), before)

    def test_demo_contract_is_disjoint_reproducible_and_not_called_scratch_ppo(self) -> None:
        self.assertIsNotNone(self.warm_start)
        protocol = self.warm_start.protocol()
        self.assertEqual(protocol["id"], "drone-physics-demonstrations-v1")
        self.assertEqual(protocol["role"], "actor behavior-cloning warm start")
        self.assertFalse(protocol["pure_model_free_from_scratch"])
        self.assertEqual(protocol["expert"]["id"], "drone-physics-pd-v1")
        self.assertEqual(protocol["dataset"]["seed_base"], 600_000)
        self.assertEqual(protocol["dataset"]["episodes"], 80)
        self.assertEqual(
            protocol["dataset"]["targets"],
            "bounded expert rotor actions",
        )
        self.assertNotIn("state_stride", protocol["dataset"])
        self.assertEqual(protocol["optimizer"]["epochs"], 60)
        self.assertNotIn("epochs_per_initial_fit", protocol["optimizer"])
        demo_seeds = set(range(600_000, 600_080))
        selection = set(range(100_000, 100_010))
        holdout = set(range(200_000, 200_100))
        self.assertFalse(demo_seeds & selection)
        self.assertFalse(demo_seeds & holdout)

        env = self.spec.make_env(False)
        agent = PPOAgent(
            env.obs_dim, env.n_continuous, env.n_binary,
            torch.device("cpu"),
            actor_initialization=self.spec.actor_initialization,
        )
        diagnostics = self.warm_start.apply(agent)
        self.assertGreater(diagnostics["samples"], 60_000)
        # Canonical completion below is the acceptance criterion; this bound
        # catches a clearly failed clone without overfitting to a proxy loss.
        self.assertLess(diagnostics["final_mse"], 2.5e-4)
        self.assertRegex(diagnostics["dataset_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("dagger_rounds", diagnostics)

        fixed = evaluate_policy(agent, range(100_000, 100_100))
        fresh = evaluate_policy(agent, range(960_000, 960_020))
        self.assertTrue(all(item["canonical_success"] for item in fixed))
        self.assertTrue(all(item["canonical_success"] for item in fresh))
        self.assertTrue(all(item["steps"] <= drone.HORIZON_STEPS
                            for item in fixed + fresh))

    def test_checkpoint_discloses_v18_and_rejects_v17_policy_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state.json").write_text(
                '{"active_scenario":"drone-hover"}')
            trainer = trainer_module.Trainer(Settings(
                port=8901, checkpoint_dir=root, checkpoint_every_n=25,
                max_episodes=10, use_gpu=False, seed=42, eval_episodes=1,
            ))
            trainer._run_eval = lambda: {
                "reward": 0.0, "reward_std": 0.0,
                "metric": 0.0, "metric_std": 0.0,
                "failure_progress": None, "episodes": 1,
                "success_rate": 0.0, "success_ci_low": 0.0,
                "success_ci_high": 1.0, "evaluation_suite": "test",
                "seed": 42, "trajectory": [],
            }
            trainer._save_checkpoint()
            meta = trainer.registry.list()[0]

            self.assertEqual(meta["schema_version"], 17)
            self.assertEqual(meta["protocol"]["version"], 19)
            self.assertEqual(meta["protocol"]["training_schedule"],
                             self.schedule.protocol())
            warm = meta["protocol"]["actor_warm_start"]
            self.assertEqual(warm["contract"], self.warm_start.protocol())
            self.assertLess(warm["realized"]["final_mse"], 2.5e-4)
            self.assertEqual(
                meta["protocol"]["actor_initialization"][
                    "continuous_log_std"], [-2.0, -2.0])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_env = drone.DroneEnv(jitter=False)
            old_agent = PPOAgent(
                9, old_env.n_continuous, old_env.n_binary,
                torch.device("cpu"),
            )
            old = CheckpointRegistry(root, "drone-hover", 16)
            old.save(25, old_agent, [], {
                "reward": 0.0, "metric": 0.0, "trajectory": [],
                "protocol": {"version": 16},
            })
            upgraded = CheckpointRegistry(root, "drone-hover", 17)
            self.assertEqual(upgraded.list(), [])
            archives = upgraded.list_archives()
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0]["schema_version"], 16)

    def test_only_assisted_scenarios_gain_demo_and_episode_schedules(self) -> None:
        for spec in list_specs():
            if spec.id in {"drone-hover", "traffic-rush"}:
                continue
            self.assertIsNone(getattr(spec, "training_schedule", None), spec.id)
            self.assertIsNone(getattr(spec, "actor_warm_start", None), spec.id)
        traffic = get_spec("traffic-rush")
        self.assertIsNotNone(traffic.training_schedule)
        self.assertIsNotNone(traffic.actor_warm_start)


if __name__ == "__main__":
    unittest.main()
