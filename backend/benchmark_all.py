#!/usr/bin/env python3
"""Reproducible sequential benchmark runner for the Policy Atlas API.

The default mode is read-only inventory. Pass ``--execute`` to archive each
selected scenario's active run, start a fresh seeded run, and train it up to
the requested episode budget. Results are flushed to JSON after every poll so
an interrupted campaign remains auditable.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import fmean, pstdev
import sys
import time
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


REPORT_SCHEMA_VERSION = 2
REPORT_PROTOCOL = "policy-atlas-benchmark-v2"


@dataclass(frozen=True)
class SolveCriteria:
    """Evidence required before a policy is labelled solved."""

    min_success_rate: float = 0.9
    min_ci_low: float = 0.7
    min_eval_episodes: int = 10
    confirmations: int = 1


def solve_contract(criteria: SolveCriteria) -> dict[str, Any]:
    """Serialize the checkpoint-selection rule stored with every report."""
    noun = "checkpoint" if criteria.confirmations == 1 else "checkpoints"
    return {
        "definition": (
            "Selection chooses the first checkpoint in a streak of "
            f"{criteria.confirmations} consecutive qualifying {noun}. A "
            "qualifier must match the run protocol and meet the configured "
            "fixed-suite success-rate, Wilson-bound, and suite-size thresholds. "
            "Independent holdout evidence is evaluated only after selection."
        ),
        "min_success_rate": criteria.min_success_rate,
        "min_ci_low": criteria.min_ci_low,
        "min_eval_episodes": criteria.min_eval_episodes,
        "confirmations": criteria.confirmations,
    }


def _engine_digest(checkpoint: dict[str, Any]) -> str | None:
    protocol = checkpoint.get("protocol") or {}
    return protocol.get("engine_source_sha256")


def checkpoint_result(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Select stable scientific fields from a checkpoint sidecar."""
    return {
        "episode": checkpoint.get("episode"),
        "total_steps": checkpoint.get("total_steps"),
        "updates": checkpoint.get("update_count"),
        "seed": checkpoint.get("seed"),
        "evaluation_suite": checkpoint.get("evaluation_suite"),
        "evaluation_episodes": checkpoint.get("eval_episodes"),
        "success_rate": checkpoint.get("success_rate"),
        "success_ci_low": checkpoint.get("success_ci_low"),
        "success_ci_high": checkpoint.get("success_ci_high"),
        "eval_reward": checkpoint.get("eval_reward"),
        "eval_reward_std": checkpoint.get("eval_reward_std"),
        "metric": checkpoint.get("eval_metric"),
        "metric_std": checkpoint.get("eval_metric_std"),
        "failure_progress": checkpoint.get("eval_failure_progress"),
        "training_diagnostics": checkpoint.get("training_diagnostics"),
        "schema_version": checkpoint.get("schema_version"),
        "obs_dim": checkpoint.get("obs_dim"),
        "n_continuous": checkpoint.get("n_continuous"),
        "n_binary": checkpoint.get("n_binary"),
        "metadata_sha256": checkpoint.get("metadata_sha256"),
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "protocol": checkpoint.get("protocol"),
        "engine_source_sha256": _engine_digest(checkpoint),
    }


def checkpoint_trace_result(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Keep the learning curve without repeating immutable run metadata."""
    return {
        "episode": checkpoint.get("episode"),
        "total_steps": checkpoint.get("total_steps"),
        "updates": checkpoint.get("update_count"),
        "mean_training_reward": checkpoint.get("mean_reward"),
        "success_rate": checkpoint.get("success_rate"),
        "success_ci_low": checkpoint.get("success_ci_low"),
        "success_ci_high": checkpoint.get("success_ci_high"),
        "eval_reward": checkpoint.get("eval_reward"),
        "eval_reward_std": checkpoint.get("eval_reward_std"),
        "metric": checkpoint.get("eval_metric"),
        "metric_std": checkpoint.get("eval_metric_std"),
        "failure_progress": checkpoint.get("eval_failure_progress"),
        "training_diagnostics": checkpoint.get("training_diagnostics"),
        "metadata_sha256": checkpoint.get("metadata_sha256"),
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
    }


def _qualifies(
    checkpoint: dict[str, Any],
    criteria: SolveCriteria,
    *,
    expected_engine: str,
    expected_suite: str,
    expected_seed: int,
) -> bool:
    rate = checkpoint.get("success_rate")
    ci_low = checkpoint.get("success_ci_low")
    return bool(
        _engine_digest(checkpoint) == expected_engine
        and checkpoint.get("evaluation_suite") == expected_suite
        and checkpoint.get("seed") == expected_seed
        and int(checkpoint.get("eval_episodes") or 0) >= criteria.min_eval_episodes
        and rate is not None
        and float(rate) >= criteria.min_success_rate
        and ci_low is not None
        and float(ci_low) >= criteria.min_ci_low
    )


def _matches_run_contract(
    checkpoint: dict[str, Any],
    *,
    expected_engine: str,
    expected_suite: str,
    expected_seed: int,
) -> bool:
    return bool(
        _engine_digest(checkpoint) == expected_engine
        and checkpoint.get("evaluation_suite") == expected_suite
        and checkpoint.get("seed") == expected_seed
    )


def _assert_scenario(payload: dict[str, Any], expected: str, context: str) -> None:
    actual = payload.get("scenario_id")
    if actual != expected:
        raise RuntimeError(
            f"{expected}: scenario changed during {context}; active={actual!r}")


def _assert_run_contract(
    payload: dict[str, Any],
    *,
    expected_scenario: str,
    expected_engine: str,
    expected_suite: str,
    expected_seed: int,
    context: str,
) -> None:
    """Reject external resets/restarts that keep the same scenario id."""
    _assert_scenario(payload, expected_scenario, context)
    expected = {
        "engine_source_sha256": expected_engine,
        "evaluation_suite": expected_suite,
        "seed": expected_seed,
    }
    for field, wanted in expected.items():
        actual = payload.get(field)
        if actual != wanted:
            raise RuntimeError(
                f"{expected_scenario}: run contract changed during {context}; "
                f"{field} expected {wanted!r}, observed {actual!r}")


def find_confirmed_solve(
    checkpoints: list[dict[str, Any]],
    criteria: SolveCriteria,
    *,
    expected_engine: str,
    expected_suite: str,
    expected_seed: int,
) -> dict[str, Any] | None:
    """Return the first policy in a consecutive, protocol-matched solve streak."""
    streak: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoints, key=lambda item: int(item["episode"])):
        if _qualifies(
            checkpoint,
            criteria,
            expected_engine=expected_engine,
            expected_suite=expected_suite,
            expected_seed=expected_seed,
        ):
            streak.append(checkpoint)
            if len(streak) >= criteria.confirmations:
                return {
                    "earliest_episode": streak[0]["episode"],
                    "confirmed_episode": checkpoint["episode"],
                    "checkpoint": checkpoint_result(streak[0]),
                }
        else:
            streak = []
    return None


def campaign_verdict(
    runs: list[dict[str, Any]],
    *,
    criteria: SolveCriteria,
    require_holdout: bool,
    expected_runs: int,
    expected_holdout_episodes: int | None = None,
    expected_holdout_seed_base: int | None = None,
) -> dict[str, Any]:
    """Decide whether every requested run has independent solve evidence."""
    failures: list[dict[str, Any]] = []
    verified_runs = 0
    for run in runs:
        reasons: list[str] = []
        if run.get("state") != "solved":
            reasons.append(f"selection state is {run.get('state', 'missing')}")
        if require_holdout:
            holdout = run.get("holdout") or {}
            if holdout.get("state") != "complete":
                reasons.append(
                    f"holdout state is {holdout.get('state', 'missing')}")
            else:
                selection = run.get("selection") or {}
                selected = selection.get("checkpoint") or {}
                selected_engine = run.get("engine_source_sha256")
                selected_tensor = selected.get("checkpoint_sha256")
                if not isinstance(selected_engine, str) or not selected_engine:
                    reasons.append("selection engine digest is missing")
                elif (holdout.get("engine_source_sha256")
                      != selected_engine):
                    reasons.append("holdout engine digest does not match selection")
                if not isinstance(selected_tensor, str) or not selected_tensor:
                    reasons.append("selected tensor hash is missing")
                elif (holdout.get("selected_checkpoint_sha256")
                      != selected_tensor):
                    reasons.append("holdout tensor hash does not match selection")
                if (holdout.get("selected_checkpoint_episode")
                        != selected.get("episode")):
                    reasons.append("holdout checkpoint episode does not match selection")
                if not holdout.get("seed_range_disjoint_from_selection", False):
                    reasons.append("holdout seeds overlap checkpoint selection")
                if not holdout.get(
                    "seed_range_disjoint_from_training_curriculum", False
                ):
                    reasons.append("holdout seeds overlap training curriculum")
                observed_episodes = int(holdout.get("episodes") or 0)
                if observed_episodes < criteria.min_eval_episodes:
                    reasons.append("holdout evaluation is too small")
                if expected_holdout_episodes is not None:
                    if observed_episodes != expected_holdout_episodes:
                        reasons.append("holdout episode count differs from request")
                    if (holdout.get("requested_episodes")
                            != expected_holdout_episodes):
                        reasons.append("holdout request count is inconsistent")
                if (expected_holdout_seed_base is not None
                        and expected_holdout_episodes is not None):
                    expected_end = (expected_holdout_seed_base
                                    + expected_holdout_episodes - 1)
                    if (holdout.get("seed_base") != expected_holdout_seed_base
                            or holdout.get("seed_end") != expected_end):
                        reasons.append("holdout seed range differs from request")
                    if (holdout.get("requested_seed_base")
                            != expected_holdout_seed_base
                            or holdout.get("requested_seed_end") != expected_end):
                        reasons.append("holdout requested seed range is inconsistent")
                rate = holdout.get("success_rate")
                if rate is None or float(rate) < criteria.min_success_rate:
                    reasons.append("holdout success rate is below threshold")
                ci_low = holdout.get("success_ci_low")
                if ci_low is None or float(ci_low) < criteria.min_ci_low:
                    reasons.append("holdout Wilson lower bound is below threshold")
        if reasons:
            failures.append({
                "scenario_id": run.get("scenario_id"),
                "seed": run.get("seed"),
                "reasons": reasons,
            })
        else:
            verified_runs += 1

    if len(runs) != expected_runs:
        failures.append({
            "scenario_id": None,
            "seed": None,
            "reasons": [
                f"observed {len(runs)} of {expected_runs} requested runs"
            ],
        })
    return {
        "all_verified": not failures and verified_runs == expected_runs,
        "expected_runs": expected_runs,
        "observed_runs": len(runs),
        "verified_runs": verified_runs,
        "require_holdout": require_holdout,
        "failures": failures,
    }


def inventory_report(catalog: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    scenarios = []
    for scenario in catalog.get("scenarios", []):
        progress = scenario.get("progress") or {}
        scenarios.append({
            "id": scenario["id"],
            "name": scenario["name"],
            "kind": scenario["kind"],
            "episode": int(progress.get("episode") or 0),
            "checkpoints": int(progress.get("checkpoints") or 0),
            "mean_reward": progress.get("mean_reward"),
            "best_metric": progress.get("best_metric"),
        })
    return {
        "active_scenario": catalog.get("active"),
        "active_run": {
            "training": bool(status.get("training")),
            "episode": int(status.get("episode") or 0),
            "total_steps": int(status.get("total_steps") or 0),
            "updates": int(status.get("update_count") or 0),
            "seed": status.get("seed"),
        },
        "scenario_count": len(scenarios),
        "engine_source_sha256": status.get("engine_source_sha256"),
        "evaluation_suite": status.get("evaluation_suite"),
        "evaluation_episodes": status.get("eval_episodes"),
        "scenarios": scenarios,
    }


def _seed_ranges_overlap(first_base: int, first_count: int,
                         second_base: int, second_count: int) -> bool:
    first_end = first_base + first_count
    second_end = second_base + second_count
    return first_base < second_end and second_base < first_end


def training_curriculum_seed_ranges(
    protocol: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Expand every fixed outer and nested training-control seed suite."""
    if not protocol:
        return []
    curriculum = protocol.get("training_curriculum")
    if curriculum is None:
        return []
    if not isinstance(curriculum, dict):
        raise ValueError("training curriculum protocol must be an object")
    segments = curriculum.get("frontier_order")
    evaluation = curriculum.get("segment_evaluation")
    if not isinstance(segments, list) or not segments:
        raise ValueError("training curriculum frontier_order must be non-empty")
    if not isinstance(evaluation, dict):
        raise ValueError("training curriculum segment evaluation is missing")

    def required_integer(field: str, *, minimum: int = 0) -> int:
        value = evaluation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"training curriculum {field} must be an integer >= {minimum}")
        return value

    seed_base = required_integer("seed_base")
    stride = required_integer("segment_seed_stride", minimum=1)
    episodes = required_integer("episodes", minimum=1)
    ranges: list[dict[str, Any]] = []
    seen: set[int] = set()
    max_seed = 2 ** 32 - 1
    for raw_segment in segments:
        if isinstance(raw_segment, bool) or not isinstance(raw_segment, int):
            raise ValueError("training curriculum segments must be integers")
        if raw_segment in seen:
            raise ValueError("training curriculum segments must be distinct")
        seen.add(raw_segment)
        start = seed_base + raw_segment * stride
        end = start + episodes - 1
        if not 0 <= start <= end <= max_seed:
            raise ValueError("training curriculum seed range is outside uint32")
        ranges.append({
            "range_kind": "training_curriculum",
            "curriculum_id": curriculum.get("id"),
            "segment": raw_segment,
            "episodes": episodes,
            "seed_base": start,
            "seed_end": end,
        })

    control = curriculum.get("training_control")
    if control is None:
        return ranges
    if not isinstance(control, dict):
        raise ValueError("training control protocol must be an object")
    stages = control.get("stages")
    control_evaluation = control.get("control_evaluation")
    if not isinstance(stages, list) or not stages:
        raise ValueError("training control stages must be non-empty")
    if not isinstance(control_evaluation, dict):
        raise ValueError("training control evaluation is missing")

    def required_control_integer(field: str, *, minimum: int = 0) -> int:
        value = control_evaluation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"training control {field} must be an integer >= {minimum}")
        return value

    control_seed_base = required_control_integer("seed_base")
    control_stride = required_control_integer("stage_seed_stride", minimum=1)
    control_episodes = required_control_integer("episodes", minimum=1)
    if control_stride < control_episodes:
        raise ValueError("training control stage seed suites overlap")

    seen_stages: set[str] = set()
    for stage_position, raw_stage in enumerate(stages):
        if isinstance(raw_stage, str):
            stage = raw_stage
        elif isinstance(raw_stage, dict):
            stage = raw_stage.get("id")
        else:
            stage = None
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("training control stage ids must be non-empty strings")
        if stage in seen_stages:
            raise ValueError("training control stage ids must be distinct")
        seen_stages.add(stage)
        start = control_seed_base + stage_position * control_stride
        end = start + control_episodes - 1
        if not 0 <= start <= end <= max_seed:
            raise ValueError("training control seed range is outside uint32")
        ranges.append({
            "range_kind": "training_control",
            "curriculum_id": curriculum.get("id"),
            "control_id": control.get("id"),
            "stage": stage,
            "stage_position": stage_position,
            "episodes": control_episodes,
            "seed_base": start,
            "seed_end": end,
        })
    return ranges


def _assert_holdout_disjoint_from_training_curriculum(
    seed_base: int,
    episodes: int,
    ranges: list[dict[str, Any]],
) -> None:
    for seed_range in ranges:
        if _seed_ranges_overlap(
            seed_base,
            episodes,
            int(seed_range["seed_base"]),
            int(seed_range["episodes"]),
        ):
            if seed_range.get("range_kind") == "training_control":
                source = (
                    "training control "
                    f"{seed_range.get('control_id')} stage "
                    f"{seed_range['stage']}"
                )
            else:
                source = (
                    "training curriculum segment "
                    f"{seed_range['segment']}"
                )
            raise ValueError(
                f"holdout seed range overlaps {source} seeds "
                f"{seed_range['seed_base']}-{seed_range['seed_end']}"
            )


def _wilson_interval(successes: int, total: int, z: float = 1.96
                     ) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total))
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean_and_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(fmean(values)), float(pstdev(values))


def evaluate_holdout(
    spec: Any,
    agent: Any,
    *,
    episodes: int,
    seed_base: int,
    selection_seed_base: int,
    selection_episodes: int,
    engine_digest: str,
    curriculum_seed_ranges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen selected policy on starts never used for selection."""
    from app.ppo.agent import deterministic_action
    if episodes < 1:
        raise ValueError("holdout episodes must be positive")
    if _seed_ranges_overlap(
        seed_base, episodes, selection_seed_base, selection_episodes
    ):
        raise ValueError("holdout seed range overlaps the selection suite")
    curriculum_seed_ranges = curriculum_seed_ranges or []
    _assert_holdout_disjoint_from_training_curriculum(
        seed_base, episodes, curriculum_seed_ranges)

    trials: list[dict[str, Any]] = []
    for index in range(episodes):
        seed = seed_base + index
        env = spec.make_env(True)
        if not hasattr(env, "rng"):
            raise RuntimeError("holdout environment does not expose a seeded RNG")
        env.rng.seed(seed)
        observation = env.reset()
        reward_total = 0.0
        done = False
        for _ in range(env.max_steps):
            action = deterministic_action(agent, observation)
            observation, reward, done, _ = env.step(action)
            reward_total += float(reward)
            if done:
                break
        summary = env.episode_summary()
        trials.append({
            "index": index,
            "seed": seed,
            "reward": reward_total,
            "metric": summary.get("metric"),
            "failure_progress": summary.get("failure_progress"),
            "success": bool(summary.get("success", False)),
            "steps": summary.get("steps"),
            "cause": summary.get("cause"),
        })

    rewards = [float(trial["reward"]) for trial in trials]
    metrics = [float(trial["metric"]) for trial in trials
               if trial["metric"] is not None]
    progress = [float(trial["failure_progress"]) for trial in trials
                if trial["failure_progress"] is not None]
    successes = sum(bool(trial["success"]) for trial in trials)
    reward_mean, reward_std = _mean_and_std(rewards)
    metric_mean, metric_std = _mean_and_std(metrics)
    progress_mean, progress_std = _mean_and_std(progress)
    ci_low, ci_high = _wilson_interval(successes, episodes)
    return {
        "protocol_role": "post_selection_holdout",
        "influences_selection": False,
        "suite": (
            f"policy-atlas-holdout-v1-n{episodes}-"
            f"seeds{seed_base}-{seed_base + episodes - 1}"
        ),
        "episodes": episodes,
        "seed_base": seed_base,
        "seed_end": seed_base + episodes - 1,
        "selection_seed_base": selection_seed_base,
        "selection_seed_end": selection_seed_base + selection_episodes - 1,
        "seed_range_disjoint_from_selection": True,
        "seed_range_disjoint_from_training_curriculum": True,
        "training_curriculum_seed_ranges": curriculum_seed_ranges,
        "deterministic_policy": True,
        "engine_source_sha256": engine_digest,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "metric": metric_mean,
        "metric_std": metric_std,
        "failure_progress": progress_mean,
        "failure_progress_std": progress_std,
        "successes": successes,
        "success_rate": successes / episodes,
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "trials": trials,
    }


def load_agent_readonly(
    checkpoint_root: Path,
    spec: Any,
    episode: int,
    *,
    expected_engine: str,
    expected_evaluation_suite: str,
) -> tuple[Any, dict[str, Any]]:
    """Validate and load a policy without constructing a mutating registry."""
    import torch

    from app.checkpoints import CheckpointRegistry
    from app.ppo.agent import PPOAgent
    from app.trainer import source_digest

    local_engine = source_digest()
    if local_engine != expected_engine:
        raise RuntimeError(
            "holdout evaluator engine does not match the selected checkpoint: "
            f"local={local_engine}, selected={expected_engine}")

    # CheckpointRegistry.__init__ performs schema migration. A read-only
    # evaluator deliberately bypasses it, then uses the registry's validated
    # load path without creating, archiving, or renaming any file.
    registry = CheckpointRegistry.__new__(CheckpointRegistry)
    registry.dir = Path(checkpoint_root) / spec.id
    registry.schema_version = spec.checkpoint_schema
    env = spec.make_env(False)
    agent = PPOAgent(env.obs_dim, env.n_continuous, env.n_binary,
                     torch.device("cpu"))
    data = registry.load_into(
        episode,
        agent,
        expected_engine=expected_engine,
        expected_evaluation_suite=expected_evaluation_suite,
    )
    metadata = json.loads(
        (registry.dir / f"checkpoint_ep{episode:06d}.json").read_text())
    checkpoint_engine = (metadata.get("protocol") or {}).get(
        "engine_source_sha256")
    if checkpoint_engine != expected_engine:
        raise RuntimeError(
            "selected checkpoint engine does not match the campaign protocol")
    return agent, metadata


def evaluate_selected_checkpoint(
    run_result: dict[str, Any],
    *,
    checkpoint_root: Path | None,
    episodes: int = 100,
    seed_base: int = 200_000,
) -> dict[str, Any]:
    """Run the post-selection holdout without changing checkpoint selection."""
    selection = run_result.get("selection")
    if selection is None:
        return {"state": "not_run", "reason": "no checkpoint was selected"}
    if episodes == 0:
        return {"state": "not_run", "reason": "holdout disabled"}
    if checkpoint_root is None:
        return {
            "state": "not_run",
            "reason": "checkpoint root not provided",
            "requested_episodes": episodes,
        }

    checkpoint = selection["checkpoint"]
    checkpoint_episode = int(checkpoint["episode"])
    curriculum_ranges = training_curriculum_seed_ranges(
        checkpoint.get("protocol"))
    if _seed_ranges_overlap(
        seed_base,
        episodes,
        int(selection["seed_base"]),
        int(checkpoint["evaluation_episodes"]),
    ):
        raise ValueError("holdout seed range overlaps the selection suite")
    _assert_holdout_disjoint_from_training_curriculum(
        seed_base, episodes, curriculum_ranges)
    from app.scenarios import get_spec

    spec = get_spec(run_result["scenario_id"])
    agent, metadata = load_agent_readonly(
        checkpoint_root,
        spec,
        checkpoint_episode,
        expected_engine=run_result["engine_source_sha256"],
        expected_evaluation_suite=str(selection["suite"]),
    )
    selected_metadata_hash = checkpoint.get("metadata_sha256")
    selected_checkpoint_hash = checkpoint.get("checkpoint_sha256")
    if not selected_metadata_hash or not selected_checkpoint_hash:
        raise RuntimeError(
            "selected checkpoint is missing its immutable artifact hashes")
    if (metadata.get("metadata_sha256") != selected_metadata_hash
            or metadata.get("checkpoint_sha256") != selected_checkpoint_hash):
        raise RuntimeError(
            "loaded checkpoint artifact hash does not match checkpoint selection")
    holdout = evaluate_holdout(
        spec,
        agent,
        episodes=episodes,
        seed_base=seed_base,
        selection_seed_base=int(selection["seed_base"]),
        selection_episodes=int(checkpoint["evaluation_episodes"]),
        engine_digest=run_result["engine_source_sha256"],
        curriculum_seed_ranges=curriculum_ranges,
    )
    return {
        "state": "complete",
        "requested_episodes": episodes,
        "requested_seed_base": seed_base,
        "requested_seed_end": seed_base + episodes - 1,
        "selected_checkpoint_episode": checkpoint_episode,
        "selected_checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "selection_reason": selection.get("reason"),
        "selection_suite": selection.get("suite"),
        **holdout,
    }


class Api(Protocol):
    def get(self, path: str) -> dict[str, Any]: ...

    def post(self, path: str, payload: dict[str, Any] | None = None
             ) -> dict[str, Any]: ...


class BenchmarkRunner:
    """Drive one fresh run at a time through the public training API."""

    def __init__(
        self,
        api: Api,
        *,
        criteria: SolveCriteria,
        max_episodes: int,
        checkpoint_every_n: int,
        poll_seconds: float,
        stop_on_solve: bool,
        sleep: Callable[[float], None] = time.sleep,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.api = api
        self.criteria = criteria
        self.max_episodes = max_episodes
        self.checkpoint_every_n = checkpoint_every_n
        self.poll_seconds = poll_seconds
        self.stop_on_solve = stop_on_solve
        self.sleep = sleep
        self.on_progress = on_progress or (lambda _: None)

    def run_scenario(self, scenario_id: str, *, seed: int) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        preflight_status = self.api.get("/api/training/status")
        if preflight_status.get("training"):
            raise RuntimeError(f"{scenario_id}: trainer is already running")
        switch_status = self.api.post("/api/scenario", {"id": scenario_id})
        _assert_scenario(switch_status, scenario_id, "switch")
        protocol_status = self.api.get("/api/training/status")
        _assert_scenario(protocol_status, scenario_id, "protocol preflight")
        if protocol_status.get("training"):
            raise RuntimeError(f"{scenario_id}: trainer is already running")
        engine = str(protocol_status["engine_source_sha256"])
        suite = str(protocol_status["evaluation_suite"])
        selection_seed_base = int(protocol_status["evaluation_seed_base"])
        if int(protocol_status.get("eval_episodes") or 0) < self.criteria.min_eval_episodes:
            raise RuntimeError(
                f"{scenario_id}: evaluation suite has only "
                f"{protocol_status.get('eval_episodes')} starts; "
                f"need {self.criteria.min_eval_episodes}")

        # A reset archives rather than deletes the active run, preserving every
        # pre-benchmark policy while guaranteeing an independent seeded start.
        reset_status = self.api.post("/api/training/reset", {"seed": seed})
        _assert_run_contract(
            reset_status,
            expected_scenario=scenario_id,
            expected_engine=engine,
            expected_suite=suite,
            expected_seed=seed,
            context="reset",
        )
        start_status = self.api.post("/api/training/start", {
            "max_episodes": self.max_episodes,
            "checkpoint_every_n": self.checkpoint_every_n,
        })
        _assert_run_contract(
            start_status,
            expected_scenario=scenario_id,
            expected_engine=engine,
            expected_suite=suite,
            expected_seed=seed,
            context="start",
        )

        stop_sent = False
        solve = None
        checkpoints: list[dict[str, Any]] = []
        all_checkpoints: list[dict[str, Any]] = []
        status: dict[str, Any] = {}
        while True:
            status = self.api.get("/api/training/status")
            _assert_run_contract(
                status,
                expected_scenario=scenario_id,
                expected_engine=engine,
                expected_suite=suite,
                expected_seed=seed,
                context="status poll",
            )
            checkpoint_response = self.api.get("/api/checkpoints")
            _assert_scenario(checkpoint_response, scenario_id, "checkpoint poll")
            all_checkpoints = checkpoint_response.get("checkpoints", [])
            checkpoints = [
                checkpoint for checkpoint in all_checkpoints
                if _matches_run_contract(
                    checkpoint,
                    expected_engine=engine,
                    expected_suite=suite,
                    expected_seed=seed,
                )
            ]
            solve = find_confirmed_solve(
                checkpoints,
                self.criteria,
                expected_engine=engine,
                expected_suite=suite,
                expected_seed=seed,
            )
            partial = {
                "scenario_id": scenario_id,
                "seed": seed,
                "episode": int(status.get("episode") or 0),
                "total_steps": int(status.get("total_steps") or 0),
                "updates": int(status.get("update_count") or 0),
                "training": bool(status.get("training")),
                "earliest_solve": solve,
            }
            self.on_progress(partial)

            if (solve is not None and self.stop_on_solve
                    and status.get("training") and not stop_sent):
                self.api.post("/api/training/stop")
                stop_sent = True
            if not status.get("training"):
                break
            self.sleep(self.poll_seconds)

        final_checkpoint = (
            checkpoint_result(max(
                checkpoints, key=lambda checkpoint: int(checkpoint["episode"])))
            if checkpoints else None
        )
        checkpoint_trace = [
            checkpoint_trace_result(checkpoint)
            for checkpoint in sorted(
                checkpoints, key=lambda checkpoint: int(checkpoint["episode"])
            )
        ]
        completed = int(status.get("episode") or 0)
        if solve is not None:
            state = "solved"
        elif completed >= self.max_episodes:
            state = "budget_exhausted"
        else:
            state = "interrupted"
        if solve is not None:
            selection = {
                "protocol_role": "checkpoint_selection",
                "reason": "earliest_confirmed_solve",
                "suite": suite,
                "seed_base": selection_seed_base,
                "checkpoint": solve["checkpoint"],
            }
        elif final_checkpoint is not None:
            selection = {
                "protocol_role": "checkpoint_selection",
                "reason": "final_budget_checkpoint",
                "suite": suite,
                "seed_base": selection_seed_base,
                "checkpoint": final_checkpoint,
            }
        else:
            selection = None
        return {
            "scenario_id": scenario_id,
            "seed": seed,
            "state": state,
            "max_episodes": self.max_episodes,
            "completed_episodes": completed,
            "total_steps": int(status.get("total_steps") or 0),
            "updates": int(status.get("update_count") or 0),
            "checkpoint_every_n": self.checkpoint_every_n,
            "stop_on_solve": self.stop_on_solve,
            "solve_criteria": solve_contract(self.criteria),
            "engine_source_sha256": engine,
            "evaluation_suite": suite,
            "earliest_solve": solve,
            "selection": selection,
            "final_checkpoint": final_checkpoint,
            "checkpoint_trace": checkpoint_trace,
            "checkpoints_observed": len(checkpoints),
            "checkpoints_seen_total": len(all_checkpoints),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


class HttpApi:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str,
                 payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path}: HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"{method} {path}: {exc.reason}") from exc

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any] | None = None
             ) -> dict[str, Any]:
        return self._request("POST", path, payload)


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for token in value.split(","):
        try:
            seed = int(token.strip())
        except ValueError as exc:
            raise ValueError(f"invalid seed: {token!r}") from exc
        if not 0 <= seed <= 2 ** 32 - 1:
            raise ValueError(f"seed out of range: {seed}")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def resolve_scenarios(value: str, catalog: dict[str, Any]) -> list[str]:
    available = [scenario["id"] for scenario in catalog.get("scenarios", [])]
    if value.strip().lower() == "all":
        return available
    selected = [token.strip() for token in value.split(",") if token.strip()]
    unknown = [scenario_id for scenario_id in selected
               if scenario_id not in available]
    if unknown:
        raise ValueError(f"unknown scenario(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("at least one scenario is required")
    return list(dict.fromkeys(selected))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8900")
    parser.add_argument("--execute", action="store_true",
                        help="run fresh training campaigns; otherwise inventory only")
    parser.add_argument("--scenarios", default="all",
                        help="comma-separated scenario ids, or all")
    parser.add_argument("--seeds", default="42",
                        help="comma-separated independent training seeds")
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--request-timeout", type=float, default=180.0,
        help="API timeout in seconds; checkpoint evaluation can occupy the trainer",
    )
    parser.add_argument("--full-budget", action="store_true",
                        help="continue to max episodes after a confirmed solve")
    parser.add_argument("--min-success-rate", type=float, default=0.9)
    parser.add_argument("--min-ci-low", type=float, default=0.7)
    parser.add_argument("--min-eval-episodes", type=int, default=10)
    parser.add_argument("--confirmations", type=int, default=1)
    parser.add_argument("--holdout-episodes", type=int, default=100,
                        help="post-selection evaluation starts; 0 disables")
    parser.add_argument("--holdout-seed-base", type=int, default=200_000)
    parser.add_argument("--checkpoint-root", type=Path,
                        help="read-only checkpoint volume used for holdout")
    parser.add_argument("--output", type=Path,
                        default=Path("data/benchmarks/latest.json"))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI constraints before any API or filesystem mutation."""
    if args.max_episodes < 1 or args.checkpoint_every < 1:
        raise ValueError("episode budget and checkpoint interval must be positive")
    if args.poll_seconds < 0 or args.confirmations < 1:
        raise ValueError(
            "poll interval must be non-negative and confirmations positive")
    if args.request_timeout <= 0:
        raise ValueError("request timeout must be positive")
    if args.min_eval_episodes < 1:
        raise ValueError("minimum evaluation episodes must be positive")
    if args.holdout_episodes < 0:
        raise ValueError("holdout episodes must be non-negative")
    max_seed = 2 ** 32 - 1
    holdout_end = args.holdout_seed_base + max(args.holdout_episodes - 1, 0)
    if not (0 <= args.holdout_seed_base <= max_seed
            and holdout_end <= max_seed):
        raise ValueError("holdout seed range must fit unsigned 32-bit seeds")
    if not 0 <= args.min_success_rate <= 1 or not 0 <= args.min_ci_low <= 1:
        raise ValueError("success thresholds must be in [0, 1]")
    if (args.execute and args.holdout_episodes > 0
            and args.checkpoint_root is None):
        raise ValueError(
            "--checkpoint-root is required for the default post-selection "
            "holdout; pass --holdout-episodes 0 to run selection only")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    api = HttpApi(args.base_url, timeout=args.request_timeout)
    catalog = api.get("/api/scenarios")
    status = api.get("/api/training/status")
    inventory = inventory_report(catalog, status)
    selected = resolve_scenarios(args.scenarios, catalog)
    seeds = parse_seeds(args.seeds)
    criteria = SolveCriteria(
        min_success_rate=args.min_success_rate,
        min_ci_low=args.min_ci_low,
        min_eval_episodes=args.min_eval_episodes,
        confirmations=args.confirmations,
    )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_protocol": REPORT_PROTOCOL,
        "state": "inventory" if not args.execute else "running",
        "created_at": now,
        "updated_at": now,
        "base_url": args.base_url,
        "selected_scenarios": selected,
        "seeds": seeds,
        "max_episodes": args.max_episodes,
        "checkpoint_every_n": args.checkpoint_every,
        "full_budget": args.full_budget,
        "solve_contract": solve_contract(criteria),
        "holdout_protocol": {
            "role": "post_selection_only",
            "episodes": args.holdout_episodes,
            "seed_base": args.holdout_seed_base,
            "seed_end": (args.holdout_seed_base + args.holdout_episodes - 1
                         if args.holdout_episodes > 0 else None),
            "checkpoint_root": (str(args.checkpoint_root)
                                if args.checkpoint_root is not None else None),
            "influences_selection": False,
        },
        "inventory_before": inventory,
        "runs": [],
        "current": None,
    }
    write_json_atomic(args.output, report)
    if not args.execute:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if status.get("training"):
        raise SystemExit("trainer is currently running; stop it before a campaign")

    def progress(partial: dict[str, Any]) -> None:
        report["current"] = partial
        report["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json_atomic(args.output, report)
        print(
            f"[{partial['scenario_id']} seed={partial['seed']}] "
            f"episode={partial['episode']}/{args.max_episodes} "
            f"steps={partial['total_steps']} updates={partial['updates']}",
            file=sys.stderr,
            flush=True,
        )

    runner = BenchmarkRunner(
        api,
        criteria=criteria,
        max_episodes=args.max_episodes,
        checkpoint_every_n=args.checkpoint_every,
        poll_seconds=args.poll_seconds,
        stop_on_solve=not args.full_budget,
        on_progress=progress,
    )
    try:
        for scenario_id in selected:
            for seed in seeds:
                result = runner.run_scenario(scenario_id, seed=seed)
                try:
                    result["holdout"] = evaluate_selected_checkpoint(
                        result,
                        checkpoint_root=args.checkpoint_root,
                        episodes=args.holdout_episodes,
                        seed_base=args.holdout_seed_base,
                    )
                except Exception as exc:
                    result["holdout"] = {
                        "state": "failed",
                        "error": str(exc),
                        "requested_episodes": args.holdout_episodes,
                        "requested_seed_base": args.holdout_seed_base,
                        "requested_seed_end": (
                            args.holdout_seed_base + args.holdout_episodes - 1
                            if args.holdout_episodes > 0 else None
                        ),
                    }
                report["runs"].append(result)
                report["current"] = None
                report["updated_at"] = result["finished_at"]
                write_json_atomic(args.output, report)
    except KeyboardInterrupt:
        api.post("/api/training/stop")
        report["state"] = "interrupted"
        report["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json_atomic(args.output, report)
        return 130
    except Exception as exc:
        report["state"] = "failed"
        report["error"] = str(exc)
        report["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json_atomic(args.output, report)
        raise

    verdict = campaign_verdict(
        report["runs"],
        criteria=criteria,
        require_holdout=args.holdout_episodes > 0,
        expected_runs=len(selected) * len(seeds),
        expected_holdout_episodes=(
            args.holdout_episodes if args.holdout_episodes > 0 else None),
        expected_holdout_seed_base=(
            args.holdout_seed_base if args.holdout_episodes > 0 else None),
    )
    report["verification"] = verdict
    report["state"] = "verified" if verdict["all_verified"] else "incomplete"
    report["inventory_after"] = inventory_report(
        api.get("/api/scenarios"), api.get("/api/training/status"))
    report["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if verdict["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
