"""Package a completed qualification report; install only additional archives.

Every pair and its hash-bound acceptance evidence is validated before copying.
Neither command constructs a checkpoint registry, changes an active policy,
trains, or removes files. Export requires a new output directory; installation
is idempotent and refuses conflicting files anywhere in the prepared batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.scenarios import get_spec
from app.trainer import source_digest
from package_study_policies import safe_copy_pair, validate_identity, validate_pair


CRITERIA = {
    "fixed_successes": 10, "fixed_episodes": 10,
    "holdout_min_successes": 50, "holdout_episodes": 50,
    "canonical_success_required": True,
}


def _relative_pair(root, relative):
    """Check both files, including independently symlinked tensor files."""
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
        raise ValueError("checkpoint must be a relative JSON path within its root")
    source = root / relative
    for suffix in (".json", ".pt"):
        if not source.with_suffix(suffix).resolve().is_relative_to(root):
            raise ValueError("checkpoint pair escapes its root")
    return source


def _archive_path(scenario_id, seed, tensor):
    return (Path(scenario_id) / "archive"
            / f"qualified-20260911-seed-{seed}-{tensor[:8]}"
            / "checkpoint_ep000000.json")


def _validate_evidence(source, scenario_id, seed, tensor, expected_metadata):
    validate_identity(seed, 0, tensor)
    metadata = validate_pair(source, scenario_id, 0, tensor)
    if (metadata.get("seed") != seed or metadata.get("episode") != 0
            or metadata.get("metadata_sha256") != expected_metadata):
        raise ValueError("qualified export identity or metadata hash differs from the record")
    protocol = metadata.get("protocol") or {}
    qualified = protocol.get("requalification") or {}
    if (protocol.get("algorithm") != "frozen-policy requalification"
            or metadata.get("eval_episodes") != 10 or metadata.get("success_rate") != 1.0
            or qualified.get("criteria") != CRITERIA
            or qualified.get("holdout_successes") != 50
            or qualified.get("holdout_episodes") != 50
            or qualified.get("canonical_summary", {}).get("success") is not True):
        raise ValueError("signed qualification evidence does not pass all acceptance gates")
    seed_base = qualified.get("holdout_seed_base")
    if type(seed_base) is not int or not 5_200_000 <= seed_base <= 2**32 - 50:
        raise ValueError("invalid qualification holdout seed range")
    origin = protocol.get("origin") or {}
    if (origin.get("scenario_id") != scenario_id or origin.get("seed") != seed
            or qualified.get("origin") != {k: v for k, v in origin.items() if k != "protocol"}):
        raise ValueError("qualification origin does not match the declared scenario and seed")
    if (protocol.get("expert_used_at_inference") is not False
            or any(protocol.get(key) != 0 for key in (
                "additional_training_episodes", "additional_training_steps",
                "additional_optimizer_updates"))):
        raise ValueError("export is not a frozen neural-policy qualification")
    return metadata


def _identity(item, seen):
    spec = get_spec(item["scenario_id"])
    validate_identity(item["seed"], item.get("episode", 0), item["checkpoint_sha256"])
    if item.get("episode", 0) != 0:
        raise ValueError("qualified policies must use episode zero")
    identity = (spec.id, item["seed"])
    if identity in seen:
        raise ValueError("duplicate scenario and seed in qualification package")
    seen.add(identity)
    return spec


def _check_destinations(prepared, destination_root):
    """Preflight the whole batch, including a sidecar-only conflict."""
    root = Path(destination_root).resolve()
    for source, destination in prepared:
        for suffix in (".pt", ".json"):
            target = destination.with_suffix(suffix)
            if not target.resolve().is_relative_to(root):
                raise ValueError("destination escapes the archive root")
            if target.exists() and (not target.is_file() or hashlib.sha256(
                    target.read_bytes()).digest() != hashlib.sha256(
                        source.with_suffix(suffix).read_bytes()).digest()):
                raise ValueError(f"refusing to overwrite {target}")


def package(qualification_root, output):
    qualification_root, output = Path(qualification_root), Path(output)
    report = json.loads((qualification_root / "qualification-report.json").read_text())
    policies = report.get("policies")
    requested = report.get("requested_policies")
    if (report.get("state") != "complete" or report.get("all_passed") is not True
            or type(requested) is not int or requested < 1
            or not isinstance(policies, list) or len(policies) != requested):
        raise ValueError("qualification report must be complete with all requested policies passed")
    if report.get("engine_source_sha256") != source_digest():
        raise ValueError("qualification report engine differs from the current engine")
    manifest, prepared, seen = [], [], set()
    for record in policies:
        exported = record["export"]
        item = {"scenario_id": record["scenario_id"], "seed": record["seed"],
                "episode": exported["episode"],
                "checkpoint_sha256": exported["checkpoint_sha256"],
                "metadata_sha256": exported["metadata_sha256"]}
        spec = _identity(item, seen)
        if (record.get("accepted") is not True
                or record.get("fixed", {}).get("episodes") != 10
                or record.get("fixed", {}).get("success_rate") != 1.0
                or record.get("holdout", {}).get("episodes") != 50
                or record.get("holdout", {}).get("successes") != 50
                or record.get("canonical", {}).get("success") is not True):
            raise ValueError("report policy does not pass all qualification gates")
        source = _relative_pair(qualification_root, record["checkpoint"])
        metadata = _validate_evidence(source, spec.id, item["seed"],
                                      item["checkpoint_sha256"], item["metadata_sha256"])
        protocol = metadata["protocol"]
        qualified = protocol["requalification"]
        if (exported != metadata
                or record.get("origin") != protocol["origin"]
                or record.get("network_weights_sha256") != protocol.get("network_weights_sha256")
                or record["canonical"] != qualified["canonical_summary"]
                or report.get("holdout_seed_base") != qualified["holdout_seed_base"]
                or report.get("holdout_episodes") != qualified["holdout_episodes"]):
            raise ValueError("report differs from the signed qualification evidence")
        relative = _archive_path(spec.id, item["seed"], item["checkpoint_sha256"])
        item.update({"file": relative.as_posix(),
                     "engine_source_sha256": report["engine_source_sha256"],
                     "holdout_seed_base": qualified["holdout_seed_base"],
                     "holdout_successes": 50, "holdout_episodes": 50,
                     "canonical_summary": qualified["canonical_summary"]})
        manifest.append(item)
        prepared.append((source, output / relative))
    _check_destinations(prepared, output)
    output.mkdir(parents=True, exist_ok=False)
    for source, destination in prepared:
        safe_copy_pair(source, destination)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def install(package_root, live_root):
    package_root, live_root = Path(package_root), Path(live_root)
    manifest = json.loads((package_root / "manifest.json").read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("package must contain a nonempty manifest")
    prepared, seen = [], set()
    for item in manifest:
        spec = _identity(item, seen)
        relative = _archive_path(spec.id, item["seed"], item["checkpoint_sha256"])
        if Path(item["file"]) != relative:
            raise ValueError("package path does not match the declared archived policy")
        source = _relative_pair(package_root, relative)
        metadata = _validate_evidence(source, spec.id, item["seed"],
                                      item["checkpoint_sha256"], item["metadata_sha256"])
        qualified = metadata["protocol"]["requalification"]
        if (item.get("engine_source_sha256") != source_digest()
                or any(item.get(key) != qualified[key] for key in (
                    "holdout_seed_base", "holdout_successes", "holdout_episodes", "canonical_summary"))):
            raise ValueError("manifest differs from the signed qualification evidence")
        prepared.append((source, live_root / relative))
    _check_destinations(prepared, live_root)
    for source, destination in prepared:
        safe_copy_pair(source, destination)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--qualification-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    apply = commands.add_parser("install")
    apply.add_argument("--package", type=Path, required=True)
    apply.add_argument("--live-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (package(args.qualification_root, args.output)
              if args.command == "export" else install(args.package, args.live_root))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
