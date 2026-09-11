"""Assisted policies can use a smaller disclosed PPO step without global drift."""
import unittest
from dataclasses import replace

import torch

from app.ppo.agent import LR, PPOAgent
from app.scenarios import get_spec


class ScenarioLearningRateTests(unittest.TestCase):
    def test_optional_learning_rate_reaches_optimizer_without_changing_default(self):
        default = PPOAgent(3, 1, 0, torch.device("cpu"))
        assisted = PPOAgent(3, 1, 0, torch.device("cpu"), learning_rate=3e-5)
        self.assertEqual(default.optimizer.param_groups[0]["lr"], LR)
        self.assertEqual(assisted.optimizer.param_groups[0]["lr"], 3e-5)

    def test_invalid_learning_rates_are_rejected(self):
        for rate in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                PPOAgent(3, 1, 0, torch.device("cpu"), learning_rate=rate)

    def test_scenario_cannot_silently_fall_back_from_invalid_rate(self):
        for rate in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                replace(get_spec("apex-gp"), training_learning_rate=rate)


if __name__ == "__main__":
    unittest.main()
