from __future__ import annotations

import unittest
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import torch

from app.checkpoints import CheckpointRegistry
from app.ppo.agent import PPOAgent
from app.scenarios import get_spec
from app.trainer import source_digest

from benchmark_all import (
    BenchmarkRunner,
    SolveCriteria,
    _parser,
    campaign_verdict,
    checkpoint_result,
    evaluate_holdout,
    evaluate_selected_checkpoint,
    find_confirmed_solve,
    inventory_report,
    load_agent_readonly,
    parse_seeds,
    resolve_scenarios,
    validate_args,
    write_json_atomic,
)


def checkpoint(
    episode: int,
    *,
    rate: float,
    ci_low: float,
    engine: str = "engine-a",
    suite: str = "suite-n10",
    seed: int = 42,
    eval_episodes: int = 10,
) -> dict:
    return {
        "episode": episode,
        "eval_reward": 12.5,
        "eval_reward_std": 1.25,
        "eval_metric": 0.75,
        "eval_metric_std": 0.05,
        "eval_failure_progress": 0.8,
        "eval_episodes": eval_episodes,
        "success_rate": rate,
        "success_ci_low": ci_low,
        "success_ci_high": 1.0,
        "evaluation_suite": suite,
        "seed": seed,
        "total_steps": episode * 100,
        "update_count": episode // 4,
        "training_diagnostics": {
            "explained_variance": 0.42,
            "value_bias": -0.17,
            "action_std_mean": 0.31,
        },
        "schema_version": 5,
        "obs_dim": 24,
        "n_continuous": 2,
        "n_binary": 1,
        "metadata_sha256": "meta-hash",
        "checkpoint_sha256": "checkpoint-hash",
        "protocol": {
            "algorithm": "PPO", "version": 3,
            "engine_source_sha256": engine,
        },
    }


class SolveDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.criteria = SolveCriteria(
            min_success_rate=0.9,
            min_ci_low=0.7,
            min_eval_episodes=10,
            confirmations=2,
        )

    def test_reports_first_checkpoint_only_after_consecutive_confirmation(self) -> None:
        checkpoints = [
            checkpoint(25, rate=1.0, ci_low=0.72),
            checkpoint(50, rate=0.8, ci_low=0.49),
            checkpoint(75, rate=1.0, ci_low=0.72),
            checkpoint(100, rate=1.0, ci_low=0.72),
        ]

        solve = find_confirmed_solve(
            checkpoints,
            self.criteria,
            expected_engine="engine-a",
            expected_suite="suite-n10",
            expected_seed=42,
        )

        self.assertEqual(solve["earliest_episode"], 75)
        self.assertEqual(solve["confirmed_episode"], 100)

    def test_rejects_checkpoint_from_another_protocol_or_too_small_suite(self) -> None:
        checkpoints = [
            checkpoint(25, rate=1.0, ci_low=0.72, engine="old-engine"),
            checkpoint(50, rate=1.0, ci_low=0.72, eval_episodes=5),
            checkpoint(75, rate=1.0, ci_low=0.72, suite="other-suite"),
            checkpoint(100, rate=1.0, ci_low=0.72, seed=7),
        ]

        self.assertIsNone(find_confirmed_solve(
            checkpoints,
            self.criteria,
            expected_engine="engine-a",
            expected_suite="suite-n10",
            expected_seed=42,
        ))


class ReportShapeTests(unittest.TestCase):
    def test_checkpoint_result_keeps_reproducibility_and_outcome_fields(self) -> None:
        result = checkpoint_result(checkpoint(100, rate=1.0, ci_low=0.72))

        self.assertEqual(result, {
            "episode": 100,
            "total_steps": 10_000,
            "updates": 25,
            "seed": 42,
            "evaluation_suite": "suite-n10",
            "evaluation_episodes": 10,
            "success_rate": 1.0,
            "success_ci_low": 0.72,
            "success_ci_high": 1.0,
            "eval_reward": 12.5,
            "eval_reward_std": 1.25,
            "metric": 0.75,
            "metric_std": 0.05,
            "failure_progress": 0.8,
            "training_diagnostics": {
                "explained_variance": 0.42,
                "value_bias": -0.17,
                "action_std_mean": 0.31,
            },
            "schema_version": 5,
            "obs_dim": 24,
            "n_continuous": 2,
            "n_binary": 1,
            "metadata_sha256": "meta-hash",
            "checkpoint_sha256": "checkpoint-hash",
            "protocol": {
                "algorithm": "PPO", "version": 3,
                "engine_source_sha256": "engine-a",
            },
            "engine_source_sha256": "engine-a",
        })

    def test_inventory_report_includes_every_catalog_scenario(self) -> None:
        catalog = {
            "active": "one",
            "scenarios": [
                {"id": "one", "name": "One", "kind": "generic",
                 "progress": {"episode": 50, "checkpoints": 2,
                              "mean_reward": 3.0, "best_metric": 1.0}},
                {"id": "two", "name": "Two", "kind": "driving",
                 "progress": None},
            ],
        }
        status = {
            "engine_source_sha256": "engine-a",
            "evaluation_suite": "suite-n10",
            "eval_episodes": 10,
            "training": True,
            "episode": 57,
            "total_steps": 12_345,
            "update_count": 8,
            "seed": 42,
        }

        report = inventory_report(catalog, status)

        self.assertEqual(report["scenario_count"], 2)
        self.assertEqual(report["scenarios"][0]["episode"], 50)
        self.assertEqual(report["scenarios"][1]["episode"], 0)
        self.assertEqual(report["engine_source_sha256"], "engine-a")
        self.assertEqual(report["active_run"], {
            "training": True,
            "episode": 57,
            "total_steps": 12_345,
            "updates": 8,
            "seed": 42,
        })

    def test_campaign_verdict_requires_selection_and_disjoint_holdout(self) -> None:
        criteria = SolveCriteria()
        verified_run = {
            "scenario_id": "one",
            "seed": 42,
            "state": "solved",
            "holdout": {
                "state": "complete",
                "episodes": 100,
                "success_rate": 0.95,
                "success_ci_low": 0.88,
                "seed_range_disjoint_from_selection": True,
            },
        }
        verdict = campaign_verdict(
            [verified_run], criteria=criteria, require_holdout=True,
            expected_runs=1,
        )
        self.assertTrue(verdict["all_verified"])
        self.assertEqual(verdict["verified_runs"], 1)

        for bad_run in (
            {**verified_run, "state": "budget_exhausted"},
            {**verified_run, "holdout": {"state": "not_run"}},
            {**verified_run, "holdout": {
                **verified_run["holdout"], "success_rate": 0.5,
            }},
        ):
            with self.subTest(run=bad_run):
                failed = campaign_verdict(
                    [bad_run], criteria=criteria, require_holdout=True,
                    expected_runs=1,
                )
                self.assertFalse(failed["all_verified"])
                self.assertTrue(failed["failures"])

    def test_campaign_verdict_rejects_missing_requested_runs(self) -> None:
        verdict = campaign_verdict(
            [], criteria=SolveCriteria(), require_holdout=False,
            expected_runs=2,
        )
        self.assertFalse(verdict["all_verified"])
        self.assertEqual(verdict["expected_runs"], 2)


class ScriptedApi:
    def __init__(self, statuses: list[dict], checkpoint_sets: list[list[dict]]):
        self.statuses = iter(statuses)
        self.checkpoint_sets = iter(checkpoint_sets)
        self.posts: list[tuple[str, dict | None]] = []
        self.active_scenario = "test-scenario"
        self.seed = 42

    def get(self, path: str) -> dict:
        if path == "/api/training/status":
            payload = next(self.statuses)
            payload.setdefault("scenario_id", self.active_scenario)
            payload.setdefault("engine_source_sha256", "engine-a")
            payload.setdefault("evaluation_suite", "suite-n10")
            payload.setdefault("seed", self.seed)
            return payload
        if path == "/api/checkpoints":
            return {"scenario_id": self.active_scenario,
                    "checkpoints": next(self.checkpoint_sets)}
        raise AssertionError(path)

    def post(self, path: str, payload: dict | None = None) -> dict:
        self.posts.append((path, payload))
        if path == "/api/scenario" and payload is not None:
            self.active_scenario = str(payload["id"])
        if path == "/api/training/reset" and payload is not None:
            self.seed = int(payload["seed"])
        if path != "/api/training/stop":
            return {"scenario_id": self.active_scenario,
                    "training": path == "/api/training/start",
                    "engine_source_sha256": "engine-a",
                    "evaluation_suite": "suite-n10",
                    "seed": self.seed}
        return {"ok": True}


class RunnerTests(unittest.TestCase):
    def test_runner_rejects_same_scenario_when_run_contract_changes(self) -> None:
        api = ScriptedApi(
            statuses=[
                {"training": False},
                {"training": False, "eval_episodes": 10,
                 "evaluation_seed_base": 100_000},
                {"training": True, "episode": 1, "seed": 99},
            ],
            checkpoint_sets=[[]],
        )
        runner = BenchmarkRunner(
            api, criteria=SolveCriteria(), max_episodes=10,
            checkpoint_every_n=5, poll_seconds=0, stop_on_solve=True,
            sleep=lambda _: None,
        )

        with self.assertRaisesRegex(RuntimeError, "run contract changed.*seed"):
            runner.run_scenario("test-scenario", seed=42)

    def test_runner_rejects_an_external_scenario_switch_during_campaign(self) -> None:
        api = ScriptedApi(
            statuses=[
                {"scenario_id": "test-scenario", "training": False},
                {"scenario_id": "test-scenario", "training": False,
                 "engine_source_sha256": "engine-a",
                 "evaluation_suite": "suite-n10", "eval_episodes": 10,
                 "evaluation_seed_base": 100_000},
                {"scenario_id": "other-scenario", "training": False,
                 "episode": 1},
            ],
            checkpoint_sets=[[]],
        )
        runner = BenchmarkRunner(
            api, criteria=SolveCriteria(), max_episodes=10,
            checkpoint_every_n=5, poll_seconds=0, stop_on_solve=True,
            sleep=lambda _: None,
        )

        with self.assertRaisesRegex(RuntimeError, "scenario changed"):
            runner.run_scenario("test-scenario", seed=42)

    def test_final_selection_ignores_stale_protocol_checkpoints(self) -> None:
        matching = checkpoint(25, rate=0.0, ci_low=0.0)
        stale = checkpoint(50, rate=1.0, ci_low=0.72, engine="old-engine")
        api = ScriptedApi(
            statuses=[
                {"scenario_id": "test-scenario", "training": False},
                {"scenario_id": "test-scenario", "training": False,
                 "engine_source_sha256": "engine-a",
                 "evaluation_suite": "suite-n10", "eval_episodes": 10,
                 "evaluation_seed_base": 100_000},
                {"scenario_id": "test-scenario", "training": False,
                 "episode": 2000},
            ],
            checkpoint_sets=[[matching, stale]],
        )
        runner = BenchmarkRunner(
            api, criteria=SolveCriteria(), max_episodes=2000,
            checkpoint_every_n=25, poll_seconds=0, stop_on_solve=True,
            sleep=lambda _: None,
        )

        result = runner.run_scenario("test-scenario", seed=42)

        self.assertEqual(result["state"], "budget_exhausted")
        self.assertEqual(result["final_checkpoint"]["episode"], 25)
        self.assertEqual(result["selection"]["checkpoint"]["episode"], 25)

    def test_runner_never_switches_scenario_when_shared_trainer_is_busy(self) -> None:
        api = ScriptedApi(
            statuses=[{"training": True, "episode": 12}],
            checkpoint_sets=[],
        )
        runner = BenchmarkRunner(
            api,
            criteria=SolveCriteria(),
            max_episodes=2000,
            checkpoint_every_n=25,
            poll_seconds=0,
            stop_on_solve=True,
            sleep=lambda _: None,
        )

        with self.assertRaisesRegex(RuntimeError, "already running"):
            runner.run_scenario("test-scenario", seed=42)

        self.assertEqual(api.posts, [])

    def test_runner_stops_after_confirmed_solve_and_reports_final_checkpoint(self) -> None:
        first = checkpoint(25, rate=1.0, ci_low=0.72)
        second = checkpoint(50, rate=1.0, ci_low=0.72)
        final = checkpoint(51, rate=1.0, ci_low=0.72)
        api = ScriptedApi(
            statuses=[
                {"training": False, "episode": 0},
                {"training": False, "episode": 0,
                 "engine_source_sha256": "engine-a",
                 "evaluation_suite": "suite-n10", "eval_episodes": 10,
                 "evaluation_seed_base": 100_000},
                {"training": True, "episode": 25, "total_steps": 2500,
                 "update_count": 6},
                {"training": True, "episode": 50, "total_steps": 5000,
                 "update_count": 12},
                {"training": False, "episode": 51, "total_steps": 5100,
                 "update_count": 13},
            ],
            checkpoint_sets=[[first], [first, second], [first, second, final]],
        )
        runner = BenchmarkRunner(
            api,
            criteria=SolveCriteria(confirmations=2),
            max_episodes=2000,
            checkpoint_every_n=25,
            poll_seconds=0,
            stop_on_solve=True,
            sleep=lambda _: None,
        )

        result = runner.run_scenario("test-scenario", seed=42)

        self.assertEqual(result["state"], "solved")
        self.assertEqual(result["earliest_solve"]["earliest_episode"], 25)
        self.assertEqual(result["earliest_solve"]["confirmed_episode"], 50)
        self.assertEqual(result["final_checkpoint"]["episode"], 51)
        self.assertEqual(result["selection"]["reason"], "earliest_confirmed_solve")
        self.assertEqual(result["selection"]["checkpoint"]["episode"], 25)
        self.assertEqual(result["selection"]["seed_base"], 100_000)
        self.assertIn(("/api/training/stop", None), api.posts)
        self.assertIn(("/api/training/start", {
            "max_episodes": 2000, "checkpoint_every_n": 25,
        }), api.posts)

    def test_full_budget_mode_does_not_stop_when_solve_is_detected(self) -> None:
        first = checkpoint(25, rate=1.0, ci_low=0.72)
        second = checkpoint(50, rate=1.0, ci_low=0.72)
        final = checkpoint(2000, rate=1.0, ci_low=0.72)
        api = ScriptedApi(
            statuses=[
                {"training": False, "episode": 0},
                {"training": False, "episode": 0,
                 "engine_source_sha256": "engine-a",
                 "evaluation_suite": "suite-n10", "eval_episodes": 10,
                 "evaluation_seed_base": 100_000},
                {"training": True, "episode": 50},
                {"training": False, "episode": 2000},
            ],
            checkpoint_sets=[[first, second], [first, second, final]],
        )
        runner = BenchmarkRunner(
            api,
            criteria=SolveCriteria(confirmations=2),
            max_episodes=2000,
            checkpoint_every_n=25,
            poll_seconds=0,
            stop_on_solve=False,
            sleep=lambda _: None,
        )

        result = runner.run_scenario("test-scenario", seed=42)

        self.assertEqual(result["state"], "solved")
        self.assertNotIn(("/api/training/stop", None), api.posts)
        self.assertEqual(result["completed_episodes"], 2000)


class OneStepHoldoutEnv:
    max_steps = 1

    def __init__(self, seen_seeds: list[int]):
        self.rng = random.Random()
        self.seen_seeds = seen_seeds
        self.episode_reward = 0.0
        self.sample = 0

    def reset(self):
        self.sample = self.rng.randrange(1_000_000)
        self.seen_seeds.append(self.sample)
        return [0.0]

    def step(self, action):
        del action
        self.episode_reward = float(self.sample % 10)
        return [0.0], self.episode_reward, True, {}

    def episode_summary(self):
        return {
            "metric": float(self.sample % 5),
            "failure_progress": (self.sample % 100) / 100,
            "success": self.sample % 2 == 0,
            "steps": 1,
            "cause": "success" if self.sample % 2 == 0 else "failure",
        }


class DeterministicAgent:
    def __init__(self):
        self.flags: list[bool] = []

    def select_action(self, observation, deterministic=False):
        del observation
        self.flags.append(deterministic)
        return [0.0], 0.0, 0.0


class HoldoutTests(unittest.TestCase):
    def test_holdout_uses_distinct_seed_range_and_records_raw_trials(self) -> None:
        seen_samples: list[int] = []
        spec = SimpleNamespace(
            metric_mode="max",
            make_env=lambda jitter: OneStepHoldoutEnv(seen_samples),
        )
        agent = DeterministicAgent()

        result = evaluate_holdout(
            spec,
            agent,
            episodes=4,
            seed_base=200_000,
            selection_seed_base=100_000,
            selection_episodes=10,
            engine_digest="engine-a",
        )

        expected_samples = [random.Random(200_000 + i).randrange(1_000_000)
                            for i in range(4)]
        self.assertEqual(seen_samples, expected_samples)
        self.assertEqual([row["seed"] for row in result["trials"]],
                         [200_000, 200_001, 200_002, 200_003])
        self.assertEqual(result["protocol_role"], "post_selection_holdout")
        self.assertFalse(result["influences_selection"])
        self.assertEqual(result["episodes"], 4)
        self.assertEqual(result["engine_source_sha256"], "engine-a")
        self.assertTrue(result["seed_range_disjoint_from_selection"])
        self.assertTrue(all(agent.flags))
        self.assertIn("success_ci_low", result)
        self.assertIn("failure_progress", result)

    def test_holdout_rejects_any_overlap_with_selection_starts(self) -> None:
        spec = SimpleNamespace(metric_mode="max", make_env=lambda jitter: None)

        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_holdout(
                spec,
                DeterministicAgent(),
                episodes=100,
                seed_base=100_005,
                selection_seed_base=100_000,
                selection_episodes=10,
                engine_digest="engine-a",
            )

    def test_checkpoint_loader_validates_without_mutating_registry(self) -> None:
        spec = get_spec("mountain-car")
        engine = source_digest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = spec.make_env(False)
            agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary,
                             torch.device("cpu"))
            registry = CheckpointRegistry(
                root, spec.id, schema_version=spec.checkpoint_schema)
            registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 1.0,
                "metric": 1.0,
                "trajectory": [],
                "protocol": {"engine_source_sha256": engine},
            })
            before = {path.relative_to(root): path.read_bytes()
                      for path in root.rglob("*") if path.is_file()}

            loaded, metadata = load_agent_readonly(
                root, spec, 5, expected_engine=engine)

            self.assertIsInstance(loaded, PPOAgent)
            self.assertEqual(metadata["episode"], 5)
            holdout = evaluate_selected_checkpoint(
                {
                    "scenario_id": spec.id,
                    "engine_source_sha256": engine,
                    "selection": {
                        "checkpoint": {
                            "episode": 5,
                            "evaluation_episodes": 1,
                            "metadata_sha256": metadata["metadata_sha256"],
                            "checkpoint_sha256": metadata["checkpoint_sha256"],
                        },
                        "seed_base": 100_000,
                        "suite": "selection-n1",
                    },
                },
                checkpoint_root=root,
                episodes=2,
                seed_base=200_000,
            )
            self.assertEqual(holdout["state"], "complete")
            self.assertEqual(holdout["selected_checkpoint_episode"], 5)
            self.assertEqual(holdout["episodes"], 2)
            after = {path.relative_to(root): path.read_bytes()
                     for path in root.rglob("*") if path.is_file()}
            self.assertEqual(after, before)

    def test_holdout_refuses_a_different_artifact_at_selected_episode(self) -> None:
        spec = get_spec("mountain-car")
        engine = source_digest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = spec.make_env(False)
            agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary,
                             torch.device("cpu"))
            registry = CheckpointRegistry(
                root, spec.id, schema_version=spec.checkpoint_schema)
            registry.save(5, agent, [{"reward": 1.0}], {
                "reward": 1.0, "metric": 1.0, "trajectory": [],
                "protocol": {"engine_source_sha256": engine},
            })

            with self.assertRaisesRegex(RuntimeError, "artifact hash"):
                evaluate_selected_checkpoint(
                    {
                        "scenario_id": spec.id,
                        "engine_source_sha256": engine,
                        "selection": {
                            "checkpoint": {
                                "episode": 5,
                                "evaluation_episodes": 1,
                                "metadata_sha256": "selected-meta",
                                "checkpoint_sha256": "selected-tensor",
                            },
                            "seed_base": 100_000,
                            "suite": "selection-n1",
                        },
                    },
                    checkpoint_root=root,
                    episodes=2,
                    seed_base=200_000,
                )


class CliContractTests(unittest.TestCase):
    def test_holdout_cli_defaults_to_100_distinct_episodes(self) -> None:
        args = _parser().parse_args([])

        self.assertEqual(args.holdout_episodes, 100)
        self.assertEqual(args.holdout_seed_base, 200_000)
        self.assertIsNone(args.checkpoint_root)

    def test_all_and_selected_scenario_resolution_preserves_catalog_order(self) -> None:
        catalog = {"scenarios": [{"id": "one"}, {"id": "two"}, {"id": "three"}]}

        self.assertEqual(resolve_scenarios("all", catalog), ["one", "two", "three"])
        self.assertEqual(resolve_scenarios("three, one", catalog), ["three", "one"])
        with self.assertRaisesRegex(ValueError, "unknown scenario"):
            resolve_scenarios("missing", catalog)

    def test_seed_parser_rejects_out_of_range_and_deduplicates(self) -> None:
        self.assertEqual(parse_seeds("42, 7,42"), [42, 7])
        with self.assertRaisesRegex(ValueError, "seed"):
            parse_seeds("-1")

    def test_cli_rejects_invalid_evaluation_and_holdout_seed_ranges(self) -> None:
        args = _parser().parse_args(["--min-eval-episodes", "0"])
        with self.assertRaisesRegex(ValueError, "minimum evaluation episodes"):
            validate_args(args)

        args = _parser().parse_args(["--holdout-seed-base", "-1"])
        with self.assertRaisesRegex(ValueError, "holdout seed range"):
            validate_args(args)

        args = _parser().parse_args([
            "--holdout-seed-base", str(2 ** 32 - 50),
            "--holdout-episodes", "100",
        ])
        with self.assertRaisesRegex(ValueError, "holdout seed range"):
            validate_args(args)

    def test_atomic_json_write_leaves_complete_machine_readable_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_json_atomic(path, {"state": "running", "episode": 25})

            self.assertEqual(json.loads(path.read_text()), {
                "state": "running", "episode": 25,
            })
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
