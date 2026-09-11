"""Frozen-policy exports must preserve origin and reject invalid evidence."""
from dataclasses import replace
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from app.checkpoints import CheckpointRegistry
from app.ppo.agent import PPOAgent
from app.scenarios import get_spec
from app.trainer import aggregate_evaluations, source_digest
import requalify_policies as qualification


class RequalificationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = get_spec("ball-beam")
        env = spec.make_env(False)
        agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary,
                         torch.device("cpu"))
        registry = CheckpointRegistry(self.root / "original", spec.id,
                                      spec.checkpoint_schema)
        meta = registry.save(23, agent, [{"reward": 7}], {
            "reward": 7, "metric": .01, "trajectory": [], "seed": 42,
            "episodes": 10, "success_rate": 1.0,
            "update_count": 4, "total_steps": 9000,
            "evaluation_suite": "policy-atlas-eval-v1-n10",
            "protocol": {"algorithm": "PPO", "engine_source_sha256": "a" * 64},
        })
        self.path = registry._json(23)
        self.item = {
            "scenario_id": spec.id, "seed": 42,
            "checkpoint": str(self.path), "source_schema": spec.checkpoint_schema,
            "source_engine_source_sha256": "a" * 64,
            "checkpoint_sha256": meta.checkpoint_sha256,
            "metadata_sha256": meta.metadata_sha256,
        }

    def successful_evidence(self):
        trials = [{"seed": 5_200_000 + i, "reward": 10,
                   "metric": 0, "success": True, "cause": "balanced"}
                  for i in range(50)]
        return {
            "fixed": {"reward": 10, "reward_std": 0, "metric": 0,
                      "metric_std": 0, "failure_progress": None,
                      "episodes": 10, "success_rate": 1.0,
                      "success_ci_low": .72, "success_ci_high": 1.0,
                      "evaluation_suite": "policy-atlas-eval-v1-n10",
                      "seed": 42, "trajectory": [[1, 2, 3, 0, 0]],
                      "canonical_summary": {"success": True, "cause": "balanced"},
                      "replay_frames": [{"terminal": True, "cause": "balanced"}]},
            "holdout": {**aggregate_evaluations(trials, "min"),
                        "successes": 50, "trials": trials},
            "canonical": {"success": True, "cause": "balanced"},
        }

    def test_export_preserves_source_and_creates_zero_training_lineage(self):
        before = {suffix: self.path.with_suffix(suffix).read_bytes()
                  for suffix in (".pt", ".json")}
        with patch.object(qualification, "evaluate_candidate",
                          return_value=self.successful_evidence()), patch(
                              "app.ppo.demonstrations.BehaviorCloningWarmStart.apply",
                              side_effect=AssertionError("qualification must not run BC")):
            report = qualification.requalify_manifest(
                {"policies": [self.item]}, self.root / "qualified")
        self.assertTrue(report["all_passed"])
        result = report["policies"][0]
        output = self.root / "qualified" / result["checkpoint"]
        metadata = json.loads(output.read_text())
        payload = torch.load(output.with_suffix(".pt"), weights_only=False)
        self.assertEqual(metadata["episode"], 0)
        self.assertEqual(metadata["total_steps"], 0)
        self.assertEqual(metadata["update_count"], 0)
        self.assertEqual(payload["history"], [])
        self.assertEqual(payload["agent"]["optimizer"]["state"], {})
        self.assertEqual(len(payload["replay_frames"]), 1)
        protocol = metadata["protocol"]
        self.assertEqual(protocol["algorithm"], "frozen-policy requalification")
        self.assertEqual(protocol["engine_source_sha256"], source_digest())
        self.assertFalse(protocol["exact_training_continuation"])
        self.assertEqual(protocol["additional_training_episodes"], 0)
        self.assertEqual(protocol["origin"]["engine_source_sha256"], "a" * 64)
        self.assertEqual(protocol["origin"]["episode"], 23)
        self.assertEqual(protocol["origin"]["update_count"], 4)
        self.assertEqual(protocol["origin"]["total_steps"], 9000)
        self.assertEqual(protocol["origin"]["checkpoint_sha256"],
                         self.item["checkpoint_sha256"])
        old = torch.load(self.path.with_suffix(".pt"), weights_only=False)
        for key, tensor in old["agent"]["network"].items():
            torch.testing.assert_close(payload["agent"]["network"][key], tensor,
                                       rtol=0, atol=0)
        for suffix, contents in before.items():
            self.assertEqual(self.path.with_suffix(suffix).read_bytes(), contents)

    def test_corrupt_source_is_rejected_before_export(self):
        self.path.with_suffix(".pt").write_bytes(b"corrupt")
        report = qualification.requalify_manifest(
            {"policies": [self.item]}, self.root / "qualified")
        self.assertFalse(report["all_passed"])
        self.assertIn("SHA-256", report["policies"][0]["error"])
        self.assertEqual(list((self.root / "qualified").rglob("*.pt")), [])

    def test_schema_change_requires_explicit_opt_in(self):
        newer = replace(get_spec("ball-beam"),
                        checkpoint_schema=self.item["source_schema"] + 1)
        with patch.object(qualification, "get_spec", return_value=newer):
            with self.assertRaisesRegex(ValueError, "schema change"):
                qualification.validate_source(self.item)
            accepted = qualification.validate_source({
                **self.item, "allow_source_schema_change": True})
        self.assertEqual(accepted["origin"]["schema_version"],
                         self.item["source_schema"])

    def test_network_shape_rejection_leaves_source_and_target_unchanged(self):
        source = qualification.validate_source(self.item)
        current = get_spec("ball-beam").make_env(False)
        wrong = PPOAgent(current.obs_dim + 1, 1, 0, torch.device("cpu"))
        source["data"]["agent"]["network"] = wrong.network.state_dict()
        from types import SimpleNamespace
        target = SimpleNamespace(agent=PPOAgent(current.obs_dim, 1, 0,
                                               torch.device("cpu")))
        before = {key: value.clone() for key, value
                  in target.agent.network.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "incompatible"):
            qualification.transfer_network(source, target)
        for key, value in target.agent.network.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_nonfinite_network_weights_are_rejected(self):
        source = qualification.validate_source(self.item)
        network = source["data"]["agent"]["network"]
        network[next(iter(network))].fill_(float("nan"))
        from types import SimpleNamespace
        target = SimpleNamespace(agent=PPOAgent(7, 1, 0, torch.device("cpu")))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            qualification.transfer_network(source, target)

    def test_partial_progress_cannot_claim_all_policies_passed(self):
        snapshots = []
        with patch.object(qualification, "evaluate_candidate",
                          return_value=self.successful_evidence()), patch.object(
                              qualification, "_write_report",
                              side_effect=lambda _, report: snapshots.append(
                                  json.loads(json.dumps(report)))):
            qualification.requalify_manifest({"policies": [self.item, self.item]},
                                             self.root / "qualified")
        self.assertFalse(snapshots[0]["all_passed"])
        self.assertEqual(snapshots[0]["state"], "running")
        self.assertTrue(snapshots[-1]["all_passed"])
        self.assertEqual(snapshots[-1]["state"], "complete")

    def test_engine_seed_and_manifest_hash_mismatches_are_rejected(self):
        for key, value in (("source_engine_source_sha256", "b" * 64),
                           ("checkpoint_sha256", "c" * 64),
                           ("metadata_sha256", "d" * 64), ("seed", 123)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                qualification.validate_source({**self.item, key: value})

    def test_failure_does_not_export_and_does_not_block_next_policy(self):
        failed = self.successful_evidence()
        failed["canonical"] = {"success": False, "cause": "fell"}
        with patch.object(qualification, "evaluate_candidate",
                          side_effect=[failed, self.successful_evidence()]):
            report = qualification.requalify_manifest(
                {"policies": [self.item, self.item]}, self.root / "qualified")
        self.assertFalse(report["all_passed"])
        self.assertEqual([r["accepted"] for r in report["policies"]], [False, True])
        self.assertEqual(len(list((self.root / "qualified").rglob("*.pt"))), 1)

    def test_all_three_acceptance_gates_are_required(self):
        for changed in ("fixed", "holdout", "canonical"):
            evidence = self.successful_evidence()
            if changed == "canonical":
                evidence[changed]["success"] = False
            elif changed == "fixed":
                evidence[changed]["success_rate"] = .9
            else:
                evidence[changed]["successes"] = 49
                evidence[changed]["success_rate"] = .98
            self.assertFalse(qualification.qualifies(evidence), changed)

    def test_existing_output_is_never_mutated(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "active.pt"
        sentinel.write_bytes(b"live")
        with self.assertRaises(FileExistsError):
            qualification.requalify_manifest({"policies": [self.item]}, output)
        self.assertEqual(sentinel.read_bytes(), b"live")

    def test_cli_returns_failure_status_when_any_policy_fails(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"policies": [self.item]}))
        evidence = self.successful_evidence()
        evidence["canonical"]["success"] = False
        with patch.object(qualification, "evaluate_candidate", return_value=evidence), \
                contextlib.redirect_stdout(io.StringIO()):
            status = qualification.main([
                "--manifest", str(manifest), "--output", str(self.root / "qualified")])
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
