from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
import io
import json
from pathlib import Path
import tempfile
import unittest

from app import trainer as trainer_module


def _reverify_module():
    spec = find_spec("reverify_holdouts")
    if spec is None:
        return None
    return import_module("reverify_holdouts")


def _successful_holdout(run: dict, **kwargs) -> dict:
    del kwargs
    return {
        "state": "complete",
        "protocol_role": "post_selection_holdout",
        "influences_selection": False,
        "selected_checkpoint_episode": run["selection"]["checkpoint"]["episode"],
        "episodes": 4,
        "seed_base": 200_000,
        "seed_end": 200_003,
        "selection_seed_base": 100_000,
        "selection_seed_end": 100_009,
        "seed_range_disjoint_from_selection": True,
        "engine_source_sha256": run["engine_source_sha256"],
        "successes": 4,
        "success_rate": 1.0,
        "success_ci_low": 0.72,
        "success_ci_high": 1.0,
        "trials": [],
    }


def _report(engine: str = "current-engine") -> dict:
    def run(scenario_id: str) -> dict:
        return {
            "scenario_id": scenario_id,
            "seed": 42,
            "state": "solved",
            "engine_source_sha256": engine,
            "selection": {
                "protocol_role": "checkpoint_selection",
                "reason": "earliest_confirmed_solve",
                "suite": "policy-atlas-eval-v1-n10",
                "seed_base": 100_000,
                "checkpoint": {
                    "episode": 75,
                    "evaluation_episodes": 10,
                    "metadata_sha256": f"{scenario_id}-metadata",
                    "checkpoint_sha256": f"{scenario_id}-checkpoint",
                },
            },
            "final_checkpoint": {"episode": 100, "metric": 1.0},
            "checkpoint_trace": [{"episode": 25}, {"episode": 50}],
            "holdout": {
                "state": "failed",
                "error": "interrupted evaluator",
                "requested_episodes": 4,
            },
        }

    return {
        "schema_version": 1,
        "state": "incomplete",
        "created_at": "2026-08-01T00:00:00+00:00",
        "updated_at": "2026-08-01T01:00:00+00:00",
        "selected_scenarios": ["mountain-car", "cartpole-balance"],
        "seeds": [42],
        "solve_contract": {
            "min_success_rate": 0.9,
            "min_ci_low": 0.7,
            "min_eval_episodes": 4,
            "confirmations": 2,
        },
        "holdout_protocol": {
            "role": "post_selection_only",
            "episodes": 4,
            "seed_base": 200_000,
            "checkpoint_root": "/old/checkpoints",
            "influences_selection": False,
        },
        "runs": [run("mountain-car"), run("cartpole-balance")],
        "verification": {
            "all_verified": False,
            "expected_runs": 2,
            "observed_runs": 2,
            "verified_runs": 0,
            "require_holdout": True,
            "failures": [{"scenario_id": "mountain-car"}],
        },
        "experiment_history": [{"event": "training_completed"}],
    }


class SourceDigestTests(unittest.TestCase):
    def test_digest_normalizes_lf_crlf_and_cr_without_decoding_source(self) -> None:
        helper = getattr(trainer_module, "source_digest_for_root", None)
        self.assertTrue(callable(helper), "source_digest_for_root is missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = [root / name for name in ("lf", "crlf", "cr")]
            for candidate in roots:
                (candidate / "nested").mkdir(parents=True)
            (roots[0] / "nested" / "engine.py").write_bytes(
                b"value = '\xff'\nnext_value = 2\n")
            (roots[1] / "nested" / "engine.py").write_bytes(
                b"value = '\xff'\r\nnext_value = 2\r\n")
            (roots[2] / "nested" / "engine.py").write_bytes(
                b"value = '\xff'\rnext_value = 2\r")

            digests = [helper(candidate) for candidate in roots]

        self.assertEqual(digests[0], digests[1])
        self.assertEqual(digests[1], digests[2])

    def test_digest_still_hashes_relative_path_and_every_non_newline_byte(self) -> None:
        helper = getattr(trainer_module, "source_digest_for_root", None)
        self.assertTrue(callable(helper), "source_digest_for_root is missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            by_path = root / "by-path"
            renamed = root / "renamed"
            changed = root / "changed"
            for candidate in (by_path, renamed, changed):
                candidate.mkdir()
            (by_path / "alpha.py").write_bytes(b"value = 1\n")
            (renamed / "beta.py").write_bytes(b"value = 1\r\n")
            (changed / "alpha.py").write_bytes(b"value = 2\n")

            baseline = helper(by_path)

            self.assertNotEqual(baseline, helper(renamed))
            self.assertNotEqual(baseline, helper(changed))


class ReportReverificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _reverify_module()

    def test_reverification_replaces_only_filtered_holdout_and_preserves_selection(self) -> None:
        self.assertIsNotNone(self.module, "reverify_holdouts CLI is missing")
        report = _report()
        original = deepcopy(report)

        revised = self.module.reverify_report(
            report,
            checkpoint_root=Path("readonly-checkpoints"),
            scenarios=["mountain-car"],
            engine_digest="current-engine",
            reverified_at="2026-08-06T10:00:00+00:00",
            evaluator=_successful_holdout,
        )

        self.assertEqual(report, original, "input report was mutated")
        self.assertEqual(revised["runs"][0]["selection"],
                         original["runs"][0]["selection"])
        self.assertEqual(revised["runs"][0]["final_checkpoint"],
                         original["runs"][0]["final_checkpoint"])
        self.assertEqual(revised["runs"][0]["checkpoint_trace"],
                         original["runs"][0]["checkpoint_trace"])
        self.assertEqual(revised["runs"][1], original["runs"][1])
        self.assertEqual(revised["runs"][0]["holdout"]["state"], "complete")
        self.assertEqual(revised["solve_contract"]["confirmations"], 2)
        self.assertEqual(revised["experiment_history"],
                         original["experiment_history"])
        archived = revised["holdout_reverification_history"][-1]
        self.assertEqual(archived["previous_state"], "incomplete")
        self.assertEqual(archived["runs"][0]["holdout"],
                         original["runs"][0]["holdout"])
        self.assertEqual(revised["reverified_at"],
                         "2026-08-06T10:00:00+00:00")
        self.assertEqual(
            revised["holdout_reverification_protocol"]["tool_protocol"],
            "policy-atlas-holdout-reverify-v1",
        )

    def test_final_verdict_uses_stored_contract_and_expected_run_count(self) -> None:
        self.assertIsNotNone(self.module, "reverify_holdouts CLI is missing")
        report = _report()
        report["solve_contract"]["min_success_rate"] = 0.99
        report["verification"]["expected_runs"] = 3

        def below_stored_threshold(run: dict, **kwargs) -> dict:
            holdout = _successful_holdout(run, **kwargs)
            holdout["success_rate"] = 0.95
            return holdout

        revised = self.module.reverify_report(
            report,
            checkpoint_root=Path("readonly-checkpoints"),
            scenarios=None,
            engine_digest="current-engine",
            reverified_at="2026-08-06T10:00:00+00:00",
            evaluator=below_stored_threshold,
        )

        self.assertEqual(revised["state"], "incomplete")
        self.assertFalse(revised["verification"]["all_verified"])
        self.assertEqual(revised["verification"]["expected_runs"], 3)
        reasons = revised["verification"]["failures"]
        self.assertTrue(any("success rate is below" in reason
                            for failure in reasons
                            for reason in failure["reasons"]))
        self.assertTrue(any("observed 2 of 3" in reason
                            for failure in reasons
                            for reason in failure["reasons"]))

    def test_engine_mismatch_fails_before_evaluation(self) -> None:
        self.assertIsNotNone(self.module, "reverify_holdouts CLI is missing")
        calls: list[str] = []

        with self.assertRaisesRegex(RuntimeError, "engine.*does not match"):
            self.module.reverify_report(
                _report(engine="old-engine"),
                checkpoint_root=Path("readonly-checkpoints"),
                scenarios=None,
                engine_digest="current-engine",
                reverified_at="2026-08-06T10:00:00+00:00",
                evaluator=lambda run, **kwargs: calls.append(run["scenario_id"]),
            )

        self.assertEqual(calls, [])

    def test_same_path_write_is_atomic_and_mismatch_leaves_file_untouched(self) -> None:
        self.assertIsNotNone(self.module, "reverify_holdouts CLI is missing")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign.json"
            path.write_text(json.dumps(_report()), encoding="utf-8")

            exit_code = self.module.run_reverification(
                input_path=path,
                output_path=path,
                checkpoint_root=Path(tmp) / "checkpoints",
                scenarios=["mountain-car", "cartpole-balance"],
                evaluator=_successful_holdout,
                engine_digest="current-engine",
                reverified_at="2026-08-06T10:00:00+00:00",
            )

            self.assertEqual(exit_code, 0)
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["state"], "verified")
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])
            before_failure = path.read_bytes()

            with self.assertRaisesRegex(RuntimeError, "engine.*does not match"):
                self.module.run_reverification(
                    input_path=path,
                    output_path=path,
                    checkpoint_root=Path(tmp) / "checkpoints",
                    scenarios=None,
                    evaluator=_successful_holdout,
                    engine_digest="different-engine",
                    reverified_at="2026-08-06T11:00:00+00:00",
                )

            self.assertEqual(path.read_bytes(), before_failure)

    def test_cli_requires_explicit_input_output_and_checkpoint_root(self) -> None:
        self.assertIsNotNone(self.module, "reverify_holdouts CLI is missing")
        parser = self.module._parser()

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args([
            "--input", "in.json",
            "--output", "out.json",
            "--checkpoint-root", "checkpoints",
            "--scenarios", "mountain-car,cartpole-balance",
        ])
        self.assertEqual(args.input, Path("in.json"))
        self.assertEqual(args.output, Path("out.json"))
        self.assertEqual(args.checkpoint_root, Path("checkpoints"))


if __name__ == "__main__":
    unittest.main()
