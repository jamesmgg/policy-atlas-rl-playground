"""Full-height success must come from learned actions, with expert use disclosed."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from app.envs import lander
from app.ppo.agent import PPOAgent
from app.scenarios import list_specs
from app.settings import Settings
from app.trainer import Trainer


class TestLanderDemonstrations(unittest.TestCase):
    def test_lander_initial_actor_lands_from_full_height_on_unseen_starts(self):
        spec = next(s for s in list_specs() if s.id == "lunar-lander")
        warm_start = spec.actor_warm_start
        self.assertIsNotNone(warm_start,
                             "full-descent demonstrations must seed the actor")
        protocol = warm_start.protocol()
        self.assertFalse(protocol["pure_model_free_from_scratch"])
        self.assertFalse(protocol["expert"]["used_at_inference"])
        torch.set_num_threads(1)
        for seed in (42, 123):
            torch.manual_seed(seed)
            agent = PPOAgent(9, 2, 0, torch.device("cpu"),
                             actor_initialization=spec.actor_initialization)
            warm_start.apply(agent)
            successes = 0
            for index in range(50):
                env = lander.LanderEnv(jitter=True)
                env.rng.seed(850_000 + index)
                obs = env.reset()
                for _ in range(env.max_steps):
                    obs, _, done, _ = env.step(agent.predict(obs))
                    if done:
                        break
                successes += int(env.landed)
                if env.landed:
                    self.assertLess(abs(env.vx), 8)
                    self.assertLess(abs(env.vy), 14)
                    self.assertLess(abs(env.theta), .25)
            self.assertEqual(successes, 50, f"initialization seed {seed}")

    def test_first_successful_ppo_checkpoint_generalizes_to_full_descents(self):
        # Selection uses only the ordinary fixed10-start checkpoint suite.
        # The50 starts below are disjoint from the selection and demo seeds.
        for seed in (42, 123):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "state.json").write_text(
                    json.dumps({"active_scenario": "lunar-lander"}))
                trainer = Trainer(Settings(
                    port=8901, checkpoint_dir=root, checkpoint_every_n=50,
                    max_episodes=100, use_gpu=False, seed=seed,
                    eval_episodes=10, cpu_threads=1))
                trainer.start(100, 50)
                trainer._thread.join(timeout=60)
                self.assertFalse(trainer.running)
                self.assertIsNone(trainer.last_error)
                candidates = sorted(
                    (c for c in trainer.registry.list()
                     if c["success_rate"] == 1.0), key=lambda c: c["episode"])
                self.assertTrue(candidates, f"training seed {seed}")
                selected = trainer.registry.load(candidates[0]["episode"])
                trainer.agent.load_state_dict(selected["agent"])
                successes = 0
                for index in range(50):
                    env = lander.LanderEnv(jitter=True)
                    env.rng.seed(1_950_000 + index)
                    obs = env.reset()
                    for _ in range(env.max_steps):
                        obs, _, done, _ = env.step(trainer.agent.predict(obs))
                        if done:
                            break
                    successes += int(env.landed)
                self.assertEqual(successes, 50, f"training seed {seed}")


if __name__ == "__main__":
    unittest.main()
