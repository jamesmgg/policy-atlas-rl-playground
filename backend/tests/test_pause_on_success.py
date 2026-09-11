import random
import threading
from types import SimpleNamespace
import unittest
from app.trainer import Trainer


class PauseOnSuccessTests(unittest.TestCase):
    def trainer(self, *, enabled, successes=1.0, episodes=10, canonical=True):
        trainer = Trainer.__new__(Trainer)
        trainer.env = SimpleNamespace(rng=random.Random(42))
        trainer.spec = SimpleNamespace(id="test")
        trainer.pause_on_success = enabled
        trainer.pause_reason = None
        trainer.latest_replay_success = canonical
        trainer._stop = threading.Event()
        trainer.emit = lambda message: None
        trainer.registry = SimpleNamespace(list=lambda: [])
        trainer._save_checkpoint_transaction = lambda rng: SimpleNamespace(
            episode=25, eval_reward=100, eval_metric=0, success_rate=successes,
            eval_episodes=episodes)
        trainer.replayed = []
        trainer._activate_saved_checkpoint = trainer.replayed.append
        return trainer

    def test_complete_fixed_suite_freezes_saved_policy_and_opens_its_replay(self):
        trainer = self.trainer(enabled=True)
        trainer._save_checkpoint()
        self.assertTrue(trainer._stop.is_set())
        self.assertEqual(trainer.pause_reason, "fixed_test_success")
        self.assertEqual(trainer.replayed, [25])

    def test_partial_small_or_disabled_evaluation_cannot_auto_pause(self):
        for args in ({"enabled": False}, {"enabled": True, "successes": 0.9},
                     {"enabled": True, "episodes": 1}, {"enabled": True, "canonical": False}):
            trainer = self.trainer(**args)
            trainer._save_checkpoint()
            self.assertFalse(trainer._stop.is_set())
            self.assertEqual(trainer.replayed, [])


if __name__ == "__main__":
    unittest.main()
