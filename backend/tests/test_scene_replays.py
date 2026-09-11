import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from app.checkpoints import CheckpointRegistry
from app.ppo.agent import PPOAgent
from app.trainer import Trainer, checkpoint_origin


class TinyGame:
    max_steps = 2
    dt = 0.05

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.episode_reward = 0.0
        return np.zeros(2, np.float32)

    def step(self, action):
        self.steps += 1
        self.episode_reward += 1
        return np.zeros(2, np.float32), 1.0, self.steps == 2, {}

    def episode_summary(self):
        return dict(reward=self.episode_reward, steps=self.steps,
                    metric=self.steps, success=self.steps == 2, cause="collected" if self.steps == 2 else "running")

    def ghost_sample(self):
        return [float(self.steps), 2.0, 0.0, 0.0, 0.0]

    def frame_payload(self):
        return {"objects": [{"shape": "collector", "x": self.steps, "y": 2},
                            {"shape": "coin", "x": 8, "y": 7}], "coins": self.steps}


class SceneReplayTests(unittest.TestCase):
    def test_saved_generic_replays_preserve_targets_counters_and_terminal_outcome(self):
        trainer = Trainer.__new__(Trainer)
        trainer.spec = SimpleNamespace(id="tiny-game", kind="generic", metric_mode="max", make_env=lambda jitter: TinyGame())
        trainer.settings = SimpleNamespace(eval_episodes=2)
        trainer.agent = PPOAgent(2, 1, 0, torch.device("cpu"))
        trainer.seed, trainer.episode = 42, 5
        result = trainer._run_eval()
        self.assertEqual(len(result["replay_frames"]), len(result["trajectory"]))
        self.assertEqual(result["replay_frames"][0]["coins"], 1)
        self.assertEqual(result["replay_frames"][-1]["objects"][1]["shape"], "coin")
        self.assertTrue(result["replay_frames"][-1]["terminal"])
        self.assertEqual(result["replay_frames"][-1]["cause"], "collected")
        with tempfile.TemporaryDirectory() as root:
            trainer.registry = CheckpointRegistry(Path(root), "tiny-game")
            trainer.registry.save(5, trainer.agent, [], result)
            trainer.env = TinyGame()
            trainer._lock = threading.Lock()
            messages = []
            trainer.emit = messages.append
            self.assertTrue(trainer.set_ghost(5))
            self.assertEqual(messages[0]["frames"], result["replay_frames"])
            self.assertEqual(messages[0]["scenario_id"], "tiny-game")
            archive = trainer.registry.archive_current()
            self.assertEqual(trainer.registry.list(), [])
            trainer._lock = threading.Lock()
            before = {key: value.clone() for key, value in trainer.agent.network.state_dict().items()}
            self.assertFalse(trainer.set_archive_ghost(archive.name, 5, "different-game"))
            self.assertTrue(trainer.set_archive_ghost(archive.name, 5, "tiny-game"))
            self.assertEqual(trainer.ghost["archive_id"], archive.name)
            for key, value in trainer.agent.network.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            saved = trainer.registry.load_archive(archive.name, 5)
            self.assertEqual(saved["canonical_summary"]["success"], True)
            self.assertEqual(saved["replay_frames"], result["replay_frames"])
            self.assertEqual(trainer.registry.list(), [])
            for invalid in ("..", "../other", "bad\\name", "invalid-quarantine"):
                with self.assertRaises(ValueError):
                    trainer.registry.load_archive(invalid, 5)

    def test_imported_lineage_survives_subsequent_checkpoint_protocols(self):
        origin = {"seed": 42, "episode": 57, "engine_source_sha256": "original-engine"}
        self.assertEqual(checkpoint_origin({"meta": {"protocol": {"origin": origin}}}), origin)
        self.assertEqual(checkpoint_origin({"meta": {"protocol": {"inherited_policy_origin": origin}}}), origin)
        self.assertIsNone(checkpoint_origin({}))


if __name__ == "__main__":
    unittest.main()
