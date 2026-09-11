"""Qualification packages must be verifiable, archive-only, and preflighted."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from app.checkpoints import CheckpointRegistry
from app.ppo.agent import PPOAgent
from app.scenarios import get_spec
from app.trainer import source_digest
import package_qualified_policies as packaging


class QualifiedPackageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.qualified = self.root / "qualification"
        self.engine = source_digest()
        self.report = {
            "state": "complete", "all_passed": True, "requested_policies": 2,
            "engine_source_sha256": self.engine, "holdout_seed_base": 5_200_000,
            "holdout_episodes": 50, "policies": [],
        }
        spec = get_spec("ball-beam")
        env = spec.make_env(False)
        self.agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary, torch.device("cpu"))
        for index, seed in enumerate((42, 123)):
            canonical = {"success": True, "cause": "balanced"}
            origin = {"scenario_id": spec.id, "seed": seed, "episode": 23,
                      "update_count": 4, "checkpoint_sha256": "a" * 64}
            protocol = {
                "algorithm": "frozen-policy requalification", "engine_source_sha256": self.engine,
                "origin": origin, "network_weights_sha256": "b" * 64,
                "expert_used_at_inference": False,
                "additional_training_episodes": 0, "additional_training_steps": 0,
                "additional_optimizer_updates": 0,
                "requalification": {
                    "origin": origin, "criteria": {
                        "fixed_successes": 10, "fixed_episodes": 10,
                        "holdout_min_successes": 50, "holdout_episodes": 50,
                        "canonical_success_required": True,
                    }, "holdout_seed_base": 5_200_000,
                    "holdout_successes": 50, "holdout_episodes": 50,
                    "canonical_summary": canonical,
                },
            }
            registry = CheckpointRegistry(self.qualified / f"policy-{index:03d}", spec.id, spec.checkpoint_schema)
            meta = registry.save(0, self.agent, [], {
                "reward": 1, "metric": 0, "trajectory": [[1, 2, 0, 0, 0]],
                "canonical_summary": canonical, "seed": seed,
                "episodes": 10, "success_rate": 1.0,
                "evaluation_suite": "policy-atlas-eval-v1-n10", "protocol": protocol,
            })
            self.report["policies"].append({
                "scenario_id": spec.id, "seed": seed, "accepted": True,
                "checkpoint": registry._json(0).relative_to(self.qualified).as_posix(),
                "export": asdict(meta), "origin": origin,
                "network_weights_sha256": "b" * 64,
                "fixed": {"episodes": 10, "success_rate": 1.0},
                "holdout": {"episodes": 50, "successes": 50}, "canonical": canonical,
            })
        self.write_report()

    def write_report(self):
        (self.qualified / "qualification-report.json").write_text(json.dumps(self.report))

    def test_archive_install_is_idempotent_and_never_constructs_mutating_registry(self):
        package = self.root / "package"
        live = self.root / "live"
        (live / "ball-beam").mkdir(parents=True)
        active = live / "ball-beam" / "checkpoint_ep000007.pt"
        active.write_bytes(b"active policy")
        with patch.object(CheckpointRegistry, "__init__", side_effect=AssertionError("must not archive active files")):
            manifest = packaging.package(self.qualified, package)
            self.assertEqual(packaging.install(package, live), manifest)
            self.assertEqual(packaging.install(package, live), manifest)
        self.assertEqual(active.read_bytes(), b"active policy")
        self.assertEqual(len(list(live.rglob("archive/*/*.pt"))), 2)
        self.assertEqual(len(manifest), 2)
        for item in manifest:
            self.assertEqual(item["episode"], 0)
            self.assertIn(f"qualified-20260911-seed-{item['seed']}-", item["file"])
            self.assertEqual((live / item["file"]).read_bytes(), (package / item["file"]).read_bytes())

    def test_report_must_be_complete_unique_and_pass_all_gates(self):
        original = deepcopy(self.report)
        for invalid in ("incomplete", "count", "duplicate", "fixed", "holdout", "canonical", "engine", "hash"):
            self.report = deepcopy(original)
            record = self.report["policies"][1]
            if invalid == "incomplete": self.report["all_passed"] = False
            elif invalid == "count": self.report["requested_policies"] = 46
            elif invalid == "duplicate": self.report["policies"][1] = deepcopy(self.report["policies"][0])
            elif invalid == "fixed": record["fixed"]["success_rate"] = .9
            elif invalid == "holdout": record["holdout"]["successes"] = 49
            elif invalid == "canonical": record["canonical"]["success"] = False
            elif invalid == "engine": self.report["engine_source_sha256"] = "c" * 64
            else: record["export"]["metadata_sha256"] = "d" * 64
            self.write_report()
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                packaging.package(self.qualified, self.root / "rejected")
            self.assertFalse((self.root / "rejected").exists())

    def test_signed_evidence_and_root_containment_are_required(self):
        original = deepcopy(self.report)
        record = self.report["policies"][0]
        source = self.qualified / record["checkpoint"]
        outside = self.root / "outside.json"
        outside.write_bytes(source.read_bytes())
        outside.with_suffix(".pt").write_bytes(source.with_suffix(".pt").read_bytes())
        record["checkpoint"] = "../outside.json"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "root|relative"):
            packaging.package(self.qualified, self.root / "escape")
        self.report = original
        record = self.report["policies"][0]
        protocol = deepcopy(record["export"]["protocol"])
        protocol["requalification"]["holdout_successes"] = 49
        spec = get_spec("ball-beam")
        registry = CheckpointRegistry(source.parent.parent, spec.id, spec.checkpoint_schema)
        meta = registry.save(0, self.agent, [], {
            "reward": 1, "metric": 0, "trajectory": [], "seed": 42,
            "episodes": 10, "success_rate": 1.0,
            "evaluation_suite": "policy-atlas-eval-v1-n10", "protocol": protocol,
        })
        record["export"] = asdict(meta)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "qualification|holdout"):
            packaging.package(self.qualified, self.root / "unsigned-claim")
        self.assertFalse((self.root / "unsigned-claim").exists())

    def test_install_preflights_all_sources_and_conflicts_before_any_copy(self):
        package = self.root / "package"
        manifest = packaging.package(self.qualified, package)
        second = package / manifest[1]["file"]
        tensor = second.with_suffix(".pt")
        original = tensor.read_bytes()
        tensor.write_bytes(b"corrupt")
        with self.assertRaises(Exception):
            packaging.install(package, self.root / "corrupt-target")
        self.assertFalse((self.root / "corrupt-target").exists())
        tensor.write_bytes(original)
        live = self.root / "conflict-target"
        conflict = live / manifest[1]["file"]
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"different existing sidecar")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            packaging.install(package, live)
        self.assertFalse((live / manifest[0]["file"]).parent.exists())
        self.assertFalse(conflict.with_suffix(".pt").exists())
        self.assertEqual(conflict.read_bytes(), b"different existing sidecar")


if __name__ == "__main__":
    unittest.main()
