"""Combine disjoint completed runs after an interrupted benchmark campaign."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from benchmark_all import SolveCriteria, campaign_verdict


def combine(parts):
    combined = copy.deepcopy(parts[0])
    runs, seen = [], set()
    scenarios = list(dict.fromkeys(s for p in parts for s in p["selected_scenarios"]))
    for part in parts:
        for key in ("report_protocol", "seeds", "max_episodes", "checkpoint_every_n",
                    "full_budget", "solve_contract"):
            if part[key] != combined[key]:
                raise ValueError(f"incompatible campaign field: {key}")
        for key in ("episodes", "seed_base", "seed_end", "influences_selection"):
            if part["holdout_protocol"][key] != combined["holdout_protocol"][key]:
                raise ValueError(f"incompatible holdout field: {key}")
        for run in part["runs"]:
            identity = (run["scenario_id"], run["seed"])
            if identity in seen:
                raise ValueError(f"duplicate scenario/seed: {identity}")
            if run["state"] not in ("solved", "budget_exhausted"):
                raise ValueError(f"unfinished training run: {identity}")
            if run["holdout"].get("state") != "complete":
                raise ValueError(f"unfinished holdout: {identity}")
            if runs and run["engine_source_sha256"] != runs[0]["engine_source_sha256"]:
                raise ValueError("engine fingerprints differ")
            runs.append(copy.deepcopy(run))
            seen.add(identity)
    expected = {(s, seed) for s in scenarios for seed in combined["seeds"]}
    if seen != expected:
        raise ValueError(f"missing or unexpected runs: {seen ^ expected}")
    rules = combined["solve_contract"]
    criteria = SolveCriteria(**{k: rules[k] for k in (
        "min_success_rate", "min_ci_low", "min_eval_episodes", "confirmations")})
    holdout = combined["holdout_protocol"]
    verification = campaign_verdict(runs, criteria=criteria, require_holdout=True,
        expected_runs=len(expected), expected_holdout_episodes=holdout["episodes"],
        expected_holdout_seed_base=holdout["seed_base"])
    combined.update(runs=runs, selected_scenarios=scenarios, current=None,
                    verification=verification, state="verified" if verification["all_verified"] else "incomplete",
                    updated_at=datetime.now(timezone.utc).isoformat())
    combined.pop("error", None)
    combined["inventory_after"] = parts[-1]["inventory_after"]
    combined["composition_note"] = (
        "Only disjoint completed training/holdout runs were combined. An interrupted "
        "partial run was restarted from its declared seed; source reports preserve the interruption.")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = combine([json.loads(path.read_text(encoding="utf-8")) for path in args.reports])
    result["source_reports"] = [
        {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in args.reports]
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
