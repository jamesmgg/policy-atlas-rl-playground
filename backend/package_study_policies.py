"""Package selected policies, or install a package as additional archived branches.

Neither operation changes an active policy, starts training, or removes files.
Packages contain only the policies identified by completed study records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from app.checkpoints import CheckpointRegistry, _assert_resume_compatible
from app.scenarios import get_spec
from app.trainer import source_digest


def validate_identity(seed, episode, tensor):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if type(episode) is not int or episode < 0:
        raise ValueError("episode must be a nonnegative integer")
    if not isinstance(tensor, str) or len(tensor) != 64 or any(c not in "0123456789abcdef" for c in tensor):
        raise ValueError("invalid checkpoint SHA-256")


def validate_pair(path, scenario_id, episode, expected_tensor):
    spec = get_spec(scenario_id)
    registry = CheckpointRegistry.__new__(CheckpointRegistry)
    registry.dir = path.parent
    registry.schema_version = spec.checkpoint_schema
    _, metadata = registry._validated_pair(path, path.with_suffix(".pt"), episode)
    _assert_resume_compatible(metadata, expected_engine=source_digest(),
                              expected_evaluation_suite="policy-atlas-eval-v1-n10")
    if metadata["checkpoint_sha256"] != expected_tensor:
        raise ValueError("selected artifact does not match its study record")
    return metadata


def safe_copy_pair(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish the sidecar last: the running lab never observes a sidecar pointing
    # to a partly written tensor. Existing different branches are never replaced.
    for suffix in (".pt", ".json"):
        src, dst = source.with_suffix(suffix), destination.with_suffix(suffix)
        if dst.exists():
            if hashlib.sha256(src.read_bytes()).digest() != hashlib.sha256(dst.read_bytes()).digest():
                raise ValueError(f"refusing to overwrite {dst}")
        else:
            temporary = dst.with_name("."+dst.name+".importing")
            shutil.copy2(src, temporary)
            temporary.rename(dst)


def package(report, checkpoint_root, output):
    manifest = []
    for run in report["runs"]:
        if run.get("holdout", {}).get("state") != "complete":
            continue
        spec = get_spec(run["scenario_id"])
        selected = run["selection"]["checkpoint"]
        episode, tensor = selected["episode"], selected["checkpoint_sha256"]
        validate_identity(run["seed"], episode, tensor)
        matches = []
        for sidecar in (checkpoint_root/spec.id).rglob(f"checkpoint_ep{episode:06d}.json"):
            if json.loads(sidecar.read_text()).get("checkpoint_sha256") == tensor:
                matches.append(sidecar)
        if len(matches) != 1:
            raise ValueError(f"expected one selected checkpoint for {spec.id}, seed {run['seed']}")
        metadata = validate_pair(matches[0], spec.id, episode, tensor)
        if metadata.get("seed") != run["seed"] or metadata["episode"] != episode:
            raise ValueError("checkpoint seed or episode differs from the report")
        if metadata["metadata_sha256"] != selected["metadata_sha256"]:
            raise ValueError("selected metadata hash differs from the report")
        branch = f"study-20260910-seed-{run['seed']}-ep-{episode}-{tensor[:8]}"
        relative = Path(spec.id)/"archive"/branch/matches[0].name
        safe_copy_pair(matches[0], output/relative)
        manifest.append({"scenario_id": spec.id, "seed": run["seed"], "episode": episode,
                         "file": relative.as_posix(), "checkpoint_sha256": tensor,
                         "holdout_successes": run["holdout"]["successes"],
                         "holdout_episodes": run["holdout"]["episodes"]})
    output.mkdir(parents=True, exist_ok=True)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def install(package_root, live_root):
    manifest = json.loads((package_root/"manifest.json").read_text())
    prepared = []
    for item in manifest:
        spec = get_spec(item["scenario_id"])
        validate_identity(item["seed"], item["episode"], item["checkpoint_sha256"])
        relative = Path(item["file"])
        expected_branch = f"study-20260910-seed-{item['seed']}-ep-{item['episode']}-{item['checkpoint_sha256'][:8]}"
        expected = Path(spec.id)/"archive"/expected_branch/f"checkpoint_ep{item['episode']:06d}.json"
        if relative != expected:
            raise ValueError("package path does not match the declared archived policy")
        source = package_root/relative
        destination = live_root/relative
        if not source.resolve().is_relative_to(package_root.resolve()):
            raise ValueError("package source escapes its root")
        if not destination.resolve().is_relative_to((live_root/spec.id/"archive").resolve()):
            raise ValueError("destination escapes the scenario archive")
        metadata = validate_pair(source, spec.id, item["episode"], item["checkpoint_sha256"])
        if metadata.get("seed") != item["seed"] or metadata["episode"] != item["episode"]:
            raise ValueError("checkpoint seed or episode differs from the manifest")
        prepared.append((source, destination))
    for source, destination in prepared:
        safe_copy_pair(source, destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--report", type=Path, required=True)
    export.add_argument("--checkpoint-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    apply = commands.add_parser("install")
    apply.add_argument("--package", type=Path, required=True)
    apply.add_argument("--live-root", type=Path, required=True)
    args = parser.parse_args()
    result = (package(json.loads(args.report.read_text()), args.checkpoint_root, args.output)
              if args.command == "export" else install(args.package, args.live_root))
    print(json.dumps(result))
