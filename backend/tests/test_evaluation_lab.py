import importlib.util
import random
import unittest

import numpy as np
import torch

from app.scenarios import get_spec


class EvaluationLabTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("app.evaluation"), "comparison lab is missing")
        from app import evaluation
        return evaluation

    def test_comparison_uses_paired_seeds_and_preserves_rng(self):
        module = self.module()

        class ZeroPolicy:
            def predict(self, obs):
                return np.zeros(2, np.float32)

        random.seed(12)
        np.random.seed(13)
        torch.manual_seed(14)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        result = module.compare_controllers(get_spec("orbital-docking"), ZeroPolicy(), episodes=3)
        self.assertEqual([r["controller"] for r in result["results"]], ["policy", "zero", "random", "reference"])
        for row in result["results"]:
            self.assertEqual([t["seed"] for t in row["trials"]], [3000000, 3000001, 3000002])
            self.assertEqual(row["episodes"], 3)
        self.assertEqual(result["results"][0]["success_rate"], 0)
        self.assertEqual(result["results"][-1]["success_rate"], 1)
        self.assertEqual(random.getstate(), python_before)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_before)
        self.assertFalse(result["is_holdout"])

    def test_reference_replay_contains_complete_scene_frames(self):
        module = self.module()
        result = module.reference_replay(get_spec("robot-reach"))
        self.assertTrue(result["summary"]["success"])
        self.assertEqual(result["controller"], "reference")
        self.assertEqual(result["scenario_id"], "robot-reach")
        self.assertEqual(len(result["frames"]), result["summary"]["steps"])
        self.assertEqual(result["frames"][0]["objects"][0]["shape"], "robotarm")
        self.assertTrue(result["frames"][-1]["terminal"])

    def test_invalid_evaluation_budgets_are_rejected(self):
        module = self.module()
        for episodes in (0, -1, 51):
            with self.assertRaises(ValueError):
                module.compare_controllers(get_spec("robot-reach"), None, episodes=episodes)


if __name__ == "__main__":
    unittest.main()
