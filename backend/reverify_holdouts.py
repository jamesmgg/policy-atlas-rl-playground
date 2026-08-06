#!/usr/bin/env python3
"""Recompute benchmark holdouts without touching training or checkpoint state."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from app.trainer import source_digest
from benchmark_all import (
    SolveCriteria,
    campaign_verdict,
    evaluate_selected_checkpoint,
    write_json_atomic,
)


TOOL_PROTOCOL = "policy-atlas-holdout-reverify-v1"
HoldoutEvaluator = Callable[..., dict[str, Any]]


def _solve_criteria(report: dict[str, Any]) -> SolveCriteria:
    contract = report.get("solve_contract")
    if not isinstance(contract, dict):
        raise ValueError("report is missing its solve_contract")
    try:
        return SolveCriteria(
            min_success_rate=float(contract["min_success_rate"]),
            min_ci_low=float(contract["min_ci_low"]),
            min_eval_episodes=int(contract["min_eval_episodes"]),
            confirmations=int(contract["confirmations"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("report has an invalid solve_contract") from exc


def _holdout_parameters(report: dict[str, Any]) -> tuple[int, int]:
    protocol = report.get("holdout_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("report is missing its holdout_protocol")
    try:
        episodes = int(protocol["episodes"])
        seed_base = int(protocol["seed_base"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("report has an invalid holdout_protocol") from exc
    if episodes < 1:
        raise ValueError("report holdout protocol must request at least one episode")
    max_seed = 2 ** 32 - 1
    if seed_base < 0 or seed_base + episodes - 1 > max_seed:
        raise ValueError("report holdout seed range must fit unsigned 32-bit seeds")
    return episodes, seed_base


def _expected_runs(report: dict[str, Any]) -> int:
    verification = report.get("verification")
    stored = (verification.get("expected_runs")
              if isinstance(verification, dict) else None)
    if stored is None:
        scenarios = report.get("selected_scenarios")
        seeds = report.get("seeds")
        if not isinstance(scenarios, list) or not isinstance(seeds, list):
            raise ValueError("report is missing its expected run dimensions")
        stored = len(scenarios) * len(seeds)
    try:
        expected = int(stored)
    except (TypeError, ValueError) as exc:
        raise ValueError("report has an invalid expected run count") from exc
    if expected < 1:
        raise ValueError("report expected run count must be positive")
    return expected


def _selected_scenarios(
    runs: list[dict[str, Any]], scenarios: Sequence[str] | None,
) -> list[str]:
    available = list(dict.fromkeys(
        str(run.get("scenario_id")) for run in runs if run.get("scenario_id")
    ))
    if scenarios is None:
        selected = available
    else:
        selected = list(dict.fromkeys(str(value) for value in scenarios))
        unknown = [value for value in selected if value not in available]
        if unknown:
            raise ValueError(
                "scenario filter is not present in report runs: "
                + ", ".join(unknown))
    if not selected:
        raise ValueError("report has no runs selected for holdout re-verification")
    return selected


def _archive_previous_evidence(
    report: dict[str, Any], targeted_runs: list[dict[str, Any]], timestamp: str,
) -> None:
    history = report.setdefault("holdout_reverification_history", [])
    if not isinstance(history, list):
        raise ValueError("report holdout_reverification_history must be a list")
    history.append({
        "superseded_at": timestamp,
        "tool_protocol": TOOL_PROTOCOL,
        "previous_state": report.get("state"),
        "previous_updated_at": report.get("updated_at"),
        "previous_verification": deepcopy(report.get("verification")),
        "runs": [
            {
                "scenario_id": run.get("scenario_id"),
                "seed": run.get("seed"),
                "holdout": deepcopy(run.get("holdout")),
            }
            for run in targeted_runs
        ],
    })


def reverify_report(
    report: dict[str, Any],
    *,
    checkpoint_root: Path,
    scenarios: Sequence[str] | None,
    engine_digest: str,
    reverified_at: str,
    evaluator: HoldoutEvaluator = evaluate_selected_checkpoint,
) -> dict[str, Any]:
    """Return a report with fresh holdouts while preserving frozen selection."""
    if not isinstance(report, dict):
        raise ValueError("benchmark report must be a JSON object")
    revised = deepcopy(report)
    runs = revised.get("runs")
    if not isinstance(runs, list):
        raise ValueError("report runs must be a list")
    selected_scenarios = _selected_scenarios(runs, scenarios)
    targeted_runs = [
        run for run in runs if run.get("scenario_id") in selected_scenarios
    ]

    # Preflight the entire requested batch before evaluating a single policy.
    # This prevents a mixed-engine report and ensures a mismatch cannot produce
    # a partially rewritten output.
    for run in targeted_runs:
        run_engine = run.get("engine_source_sha256")
        if run_engine != engine_digest:
            raise RuntimeError(
                f"{run.get('scenario_id')} seed={run.get('seed')}: report engine "
                f"{run_engine!r} does not match current engine {engine_digest!r}")

    criteria = _solve_criteria(revised)
    expected_runs = _expected_runs(revised)
    episodes, seed_base = _holdout_parameters(revised)
    _archive_previous_evidence(revised, targeted_runs, reverified_at)

    for run in targeted_runs:
        run["holdout"] = evaluator(
            run,
            checkpoint_root=Path(checkpoint_root),
            episodes=episodes,
            seed_base=seed_base,
        )

    verdict = campaign_verdict(
        runs,
        criteria=criteria,
        require_holdout=True,
        expected_runs=expected_runs,
        expected_holdout_episodes=episodes,
        expected_holdout_seed_base=seed_base,
    )
    revised["verification"] = verdict
    revised["state"] = "verified" if verdict["all_verified"] else "incomplete"
    revised["updated_at"] = reverified_at
    revised["reverified_at"] = reverified_at
    revised["holdout_reverification_protocol"] = {
        "tool_protocol": TOOL_PROTOCOL,
        "engine_source_sha256": engine_digest,
        "checkpoint_root": str(checkpoint_root),
        "scenarios": selected_scenarios,
        "episodes": episodes,
        "seed_base": seed_base,
        "influences_selection": False,
        "checkpoint_access": "read_only",
    }
    return revised


def run_reverification(
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_root: Path,
    scenarios: Sequence[str] | None,
    evaluator: HoldoutEvaluator = evaluate_selected_checkpoint,
    engine_digest: str | None = None,
    reverified_at: str | None = None,
) -> int:
    """Load, re-verify, then perform one atomic report replacement."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    report = json.loads(input_path.read_text(encoding="utf-8"))
    timestamp = reverified_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    revised = reverify_report(
        report,
        checkpoint_root=Path(checkpoint_root),
        scenarios=scenarios,
        engine_digest=engine_digest or source_digest(),
        reverified_at=timestamp,
        evaluator=evaluator,
    )
    write_json_atomic(output_path, revised)
    return 0 if revised["verification"]["all_verified"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="existing benchmark JSON report")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination JSON report; may equal --input")
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="read-only root containing scenario checkpoints")
    parser.add_argument(
        "--scenarios", default="all",
        help="comma-separated report scenario ids, or all",
    )
    return parser


def _parse_scenarios(value: str) -> list[str] | None:
    if value.strip().lower() == "all":
        return None
    selected = [part.strip() for part in value.split(",") if part.strip()]
    if not selected:
        raise ValueError("at least one scenario is required")
    return list(dict.fromkeys(selected))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        scenarios = _parse_scenarios(args.scenarios)
        result = run_reverification(
            input_path=args.input,
            output_path=args.output,
            checkpoint_root=args.checkpoint_root,
            scenarios=scenarios,
        )
        print(json.dumps({
            "output": str(args.output),
            "state": "verified" if result == 0 else "incomplete",
            "tool_protocol": TOOL_PROTOCOL,
        }, sort_keys=True))
        return result
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"holdout re-verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
