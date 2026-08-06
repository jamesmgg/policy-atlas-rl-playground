from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scenarios.registry import list_specs  # noqa: E402


class WetTerminalSectorCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs = {spec.id: spec for spec in list_specs()}

    def test_wet_training_focuses_only_the_terminal_sector(self) -> None:
        env = self.specs["apex-gp-wet"].make_training_env()

        self.assertEqual(env.start_line_probability, 0.75)
        self.assertEqual(env.rolling_checkpoint_indices, (11,))
        env.start_line_probability = 0.0
        env.rng.seed(42)
        sampled = set()
        for _ in range(180):
            env.reset()
            sampled.add(env.track.checkpoints.index(env.idx))
        self.assertEqual(sampled, {11})

    def test_wet_disclosure_names_the_single_terminal_checkpoint_exactly(self) -> None:
        disclosure = self.specs["apex-gp-wet"].training_start_distribution

        self.assertEqual(
            disclosure,
            "75% canonical start; 25% checkpoint 11 as a rolling state at "
            "70-90% of the curvature/grip backward-braking envelope; clock "
            "integrates an 80% envelope with a 1-second reserve; no reset "
            "reward",
        )

    def test_wet_evaluation_and_glacier_training_remain_unchanged(self) -> None:
        wet_eval = self.specs["apex-gp-wet"].make_env(False)
        glacier_training = self.specs["glacier"].make_training_env()

        self.assertEqual(self.specs["apex-gp-wet"].checkpoint_schema, 9)
        self.assertFalse(wet_eval.random_start)
        self.assertEqual(wet_eval.steps, 0)
        self.assertEqual(wet_eval.idx, 0)
        self.assertIsNone(glacier_training.rolling_checkpoint_indices)

        other_driving_starts = {
            scenario_id: self.specs[scenario_id].make_training_env()
            .rolling_checkpoint_indices
            for scenario_id in self.specs
            if self.specs[scenario_id].kind == "driving"
            and scenario_id != "apex-gp-wet"
        }
        self.assertEqual(other_driving_starts["traffic-rush"], (11,))
        self.assertTrue(all(
            starts is None
            for scenario_id, starts in other_driving_starts.items()
            if scenario_id != "traffic-rush"
        ))


if __name__ == "__main__":
    unittest.main()
