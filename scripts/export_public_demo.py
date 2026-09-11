"""Export qualified policy replays into a static, browser-safe demo bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from app.scenarios.registry import list_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path, help="Qualified policy package directory")
    parser.add_argument("output", type=Path, help="Destination demo.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    specs = {spec.id: spec for spec in list_specs()}
    policies: dict[str, list[dict]] = {scenario_id: [] for scenario_id in specs}

    for entry in manifest:
        scenario_id = entry["scenario_id"]
        if scenario_id not in specs:
            raise ValueError(f"manifest contains unknown scenario {scenario_id!r}")
        metadata_path = args.package / entry["file"]
        checkpoint = torch.load(
            metadata_path.with_suffix(".pt"), map_location="cpu", weights_only=False,
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames = checkpoint.get("replay_frames") or []
        trajectory = checkpoint.get("trajectory") or []
        if not frames or not trajectory or not frames[-1].get("terminal"):
            raise ValueError(f"{scenario_id} seed {entry['seed']} lacks a terminal replay")
        summary = entry["canonical_summary"]
        if not summary.get("success"):
            raise ValueError(f"{scenario_id} seed {entry['seed']} is not successful")

        origin = metadata.get("protocol", {}).get("origin", {})
        policy = {
            "id": f"{scenario_id}-qualified-seed-{entry['seed']}",
            "seed": entry["seed"],
            "origin_episode": origin.get("episode", entry["episode"]),
            "timestamp": metadata.get("timestamp"),
            "holdout_successes": entry["holdout_successes"],
            "holdout_episodes": entry["holdout_episodes"],
            "canonical_summary": summary,
            "evaluation": {
                key: metadata.get(key) for key in (
                    "eval_reward", "eval_reward_std", "eval_metric", "eval_metric_std",
                    "eval_episodes", "success_rate", "success_ci_low", "success_ci_high",
                    "evaluation_suite", "schema_version", "total_steps", "update_count",
                )
            },
            "dt": specs[scenario_id].make_env(False).dt,
            "trajectory": trajectory,
            "frames": frames,
        }
        policies[scenario_id].append(policy)

    expected = set(specs)
    complete = {scenario_id for scenario_id, items in policies.items() if len(items) == 2}
    if complete != expected:
        missing = sorted(expected - complete)
        raise ValueError(f"expected exactly two qualified policies per scenario; missing={missing}")

    scenarios = []
    scenes = {}
    for spec in list_specs():
        info = spec.info()
        info["progress"] = {
            "episode": max(item["origin_episode"] for item in policies[spec.id]),
            "mean_reward": 0,
            "best_metric": policies[spec.id][0]["canonical_summary"].get("metric"),
            "checkpoints": len(policies[spec.id]),
        }
        # Reference controllers are backend-generated. Static Pages replays use
        # only the independently qualified neural policies in this bundle.
        info["reference_controller"] = None
        scenarios.append(info)
        scenes[spec.id] = {"scenario_id": spec.id, **spec.scene()}
        policies[spec.id].sort(key=lambda item: item["seed"])

    engine_hashes = {entry["engine_source_sha256"] for entry in manifest}
    if len(engine_hashes) != 1:
        raise ValueError("qualified package mixes engine source revisions")

    result = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_source_sha256": engine_hashes.pop(),
        "qualification": {
            "scenarios": len(scenarios),
            "policies": len(manifest),
            "starts_per_policy": 50,
            "successful_starts": sum(item["holdout_successes"] for item in manifest),
            "total_starts": sum(item["holdout_episodes"] for item in manifest),
        },
        "scenarios": scenarios,
        "scenes": scenes,
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False), encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {len(scenarios)} scenarios, {len(manifest)} policies, "
        f"{args.output.stat().st_size / 1_000_000:.1f} MB"
    )


if __name__ == "__main__":
    main()
