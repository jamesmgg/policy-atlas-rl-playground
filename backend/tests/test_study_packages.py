import json
from pathlib import Path
import tempfile
import unittest

import torch

from app.checkpoints import CheckpointRegistry
from app.ppo.agent import PPOAgent
from app.scenarios import get_spec
from app.trainer import source_digest
from package_study_policies import install, package, safe_copy_pair


class StudyPackageTests(unittest.TestCase):
    def test_import_rejects_path_components_in_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed, tensor = "../escape", "a"*64
            relative = Path("ball-beam")/"archive"/f"study-20260910-seed-{seed}-ep-1-{tensor[:8]}"/"checkpoint_ep000001.json"
            (root/"manifest.json").write_text(json.dumps([{
                "scenario_id": "ball-beam", "seed": seed, "episode": 1,
                "checkpoint_sha256": tensor, "file": relative.as_posix(),
            }]))
            with self.assertRaisesRegex(ValueError, "seed"):
                install(root, root/"live")
            self.assertFalse((root/"live").exists())

    def test_existing_different_tensor_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/"source.json", Path(tmp)/"target.json"
            source.write_text("{}")
            source.with_suffix(".pt").write_bytes(b"new")
            target.with_suffix(".pt").write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                safe_copy_pair(source, target)
            self.assertEqual(target.with_suffix(".pt").read_bytes(), b"original")
            self.assertFalse(target.exists())

    def test_verified_import_is_idempotent_and_leaves_active_files_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = get_spec("ball-beam")
            env = spec.make_env(False)
            agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
            registry = CheckpointRegistry(root/"source", spec.id, spec.checkpoint_schema)
            registry.save(3, agent, [], {
                "reward": 0, "metric": 0, "trajectory": [], "seed": 42,
                "evaluation_suite": "policy-atlas-eval-v1-n10",
                "protocol": {"engine_source_sha256": source_digest()},
            })
            selected = registry.list()[0]
            report = {"runs": [{"scenario_id": spec.id, "seed": 42,
                "selection": {"checkpoint": selected},
                "holdout": {"state": "complete", "successes": 50, "episodes": 50}}]}
            package(report, root/"source", root/"package")
            live = root/"live"
            (live/spec.id).mkdir(parents=True)
            sentinel = live/spec.id/"checkpoint_ep000003.pt"
            sentinel.write_bytes(b"active policy")
            first = install(root/"package", live)
            second = install(root/"package", live)
            self.assertEqual(first, second)
            self.assertEqual(sentinel.read_bytes(), b"active policy")
            self.assertEqual(len(list((live/spec.id/"archive").rglob("*.pt"))), 1)


if __name__ == "__main__":
    unittest.main()
