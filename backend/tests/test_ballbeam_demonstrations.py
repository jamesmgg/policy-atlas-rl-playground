"""A qualified ball-balancing policy must settle across the full start range."""
import json
from pathlib import Path
import tempfile
import unittest

from app.scenarios import get_spec
from app.settings import Settings
from app.trainer import Trainer


class BallBeamDemonstrationTests(unittest.TestCase):
    def test_guidance_is_disclosed_without_changing_the_task(self):
        spec = get_spec("ball-beam")
        self.assertIsNotNone(spec.actor_warm_start)
        contract = spec.actor_warm_start.protocol()
        self.assertFalse(contract["pure_model_free_from_scratch"])
        self.assertFalse(contract["expert"]["used_at_inference"])
        self.assertGreaterEqual(spec.checkpoint_schema, 2)
        self.assertEqual(spec.make_env(False).max_steps, 500)

    def test_first_qualified_ppo_policy_settles_fifty_unseen_starts_per_seed(self):
        for seed in (42, 123):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "state.json").write_text(json.dumps(
                    {"active_scenario": "ball-beam"}))
                trainer = Trainer(Settings(
                    port=8901, checkpoint_dir=root, checkpoint_every_n=50,
                    max_episodes=300, use_gpu=False, seed=seed,
                    eval_episodes=10, cpu_threads=1))
                trainer.start(300, 50, True)
                trainer._thread.join(timeout=60)
                self.assertFalse(trainer.running)
                self.assertIsNone(trainer.last_error)
                self.assertEqual(trainer.pause_reason, "fixed_test_success")
                successes = 0
                for index in range(50):
                    env = trainer.spec.make_env(True)
                    env.rng.seed(5_400_000 + index)
                    observation = env.reset()
                    for _ in range(env.max_steps):
                        observation, _, done, _ = env.step(
                            trainer.agent.predict(observation))
                        if done:
                            break
                    successes += int(env.episode_summary()["success"])
                self.assertEqual(successes, 50, f"training seed {seed}")


if __name__ == "__main__":
    unittest.main()
