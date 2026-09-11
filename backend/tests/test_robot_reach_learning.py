"""Precision PPO fine-tuning must retain the cloned arm controller."""
import json
from pathlib import Path
import tempfile
import unittest

from app.scenarios import get_spec
from app.settings import Settings
from app.trainer import Trainer


class RobotReachLearningTests(unittest.TestCase):
    def test_smaller_learning_rate_is_explicit_and_reach_only(self):
        reach = get_spec("robot-reach")
        self.assertEqual(reach.training_learning_rate, 3e-5)
        self.assertEqual(reach.info()["training_learning_rate"], 3e-5)
        self.assertEqual(reach.checkpoint_schema, 3)
        self.assertIsNone(get_spec("robot-tracking").training_learning_rate)
        self.assertEqual(get_spec("robot-tracking").checkpoint_schema, 1)

    def test_first_successful_policy_reaches_fifty_new_targets_per_seed(self):
        for seed in (42, 123):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "state.json").write_text(json.dumps(
                    {"active_scenario": "robot-reach"}))
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
                    env.rng.seed(5_600_000 + index)
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
