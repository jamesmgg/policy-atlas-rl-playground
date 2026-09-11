import unittest
import numpy as np
from app.scenarios import get_spec


class ControlDemonstrationTests(unittest.TestCase):
    def test_reach_demonstrations_include_late_near_goal_corrective_actions(self):
        from app.ppo.control_demonstrations import build_dataset
        observations, actions = build_dataset("robot-reach", episodes=2,
                                             seed_base=820_000)
        late = observations[:, -1] < .2
        outside_target = np.linalg.norm(observations[:, 10:12] * 1.75, axis=1) > .08
        corrective = np.max(np.abs(actions), axis=1) > .1
        self.assertTrue(np.any(late & outside_target & corrective),
                        "late near-goal states must still ask the actor to correct")

    def test_unsolved_precision_tasks_disclose_training_only_demonstrations(self):
        for key in ("orbital-docking", "robot-reach"):
            with self.subTest(task=key):
                spec = get_spec(key)
                self.assertIsNotNone(spec.actor_warm_start)
                protocol = spec.actor_warm_start.protocol()
                self.assertFalse(protocol["pure_model_free_from_scratch"])
                self.assertFalse(protocol["expert"]["used_at_inference"])
                self.assertGreaterEqual(spec.checkpoint_schema, 2)

    def test_demo_targets_are_bounded_reproducible_pre_action_states(self):
        from app.ppo.control_demonstrations import build_dataset
        for key in ("orbital-docking", "robot-reach"):
            left = build_dataset(key, episodes=2, seed_base=810_000)
            right = build_dataset(key, episodes=2, seed_base=810_000)
            for a, b in zip(left, right):
                np.testing.assert_array_equal(a, b)
                self.assertTrue(np.isfinite(a).all())
            env = get_spec(key).make_env(True)
            env.rng.seed(810_000)
            start = env.reset()
            matching = np.flatnonzero(np.all(left[0] == start, axis=1))
            self.assertGreater(len(matching), 0)
            np.testing.assert_allclose(left[1][matching[0]], env.reference_action())
            self.assertTrue(np.all(np.abs(left[1]) <= 1))
            self.assertGreater(len(left[0]), 20)


if __name__ == "__main__":
    unittest.main()
