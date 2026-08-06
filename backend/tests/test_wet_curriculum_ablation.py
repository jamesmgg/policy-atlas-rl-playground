from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scenarios.registry import list_specs  # noqa: E402


class WetApproachCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = {spec.id: spec for spec in list_specs()}

    def test_wet_training_focuses_only_the_three_water_approaches(self) -> None:
        env = self.specs["apex-gp-wet"].make_training_env()

        self.assertEqual(env.start_line_probability, 0.75)
        self.assertEqual(env.rolling_checkpoint_indices, (1, 6, 9))
        env.start_line_probability = 0.0
        env.rng.seed(42)
        sampled = set()
        for _ in range(180):
            env.reset()
            sampled.add(env.track.checkpoints.index(env.idx))
        self.assertEqual(sampled, {1, 6, 9})

    def test_wet_disclosure_names_every_sparse_checkpoint(self) -> None:
        disclosure = self.specs["apex-gp-wet"].training_start_distribution

        self.assertIn("checkpoints 1,6,9", disclosure)
        self.assertNotIn("checkpoints 1..9", disclosure)

    def test_wet_evaluation_and_glacier_training_remain_unchanged(self) -> None:
        wet_eval = self.specs["apex-gp-wet"].make_env(False)
        glacier_training = self.specs["glacier"].make_training_env()

        self.assertFalse(wet_eval.random_start)
        self.assertEqual(wet_eval.steps, 0)
        self.assertEqual(wet_eval.idx, 0)
        self.assertIsNone(glacier_training.rolling_checkpoint_indices)


if __name__ == "__main__":
    unittest.main()
