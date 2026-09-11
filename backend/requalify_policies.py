"""Re-evaluate explicitly selected frozen weights under the current engine.

This creates a new episode-zero lineage with a fresh optimizer and RNG. It is
neither historical retraining nor exact continuation of the original run.
Original checkpoint files are read-only; output must be a new directory.
Export requires 10/10 fixed tests, 50/50 predeclared qualification starts, and
a successful canonical replay. Individual failures are reported; other entries
continue, and any failed policy produces a nonzero command exit status.

Manifest: {"policies": [{"scenario_id": "lunar-lander", "seed": 42,
"checkpoint": "/path/checkpoint_ep000057.json", "source_schema": 12,
"source_engine_source_sha256": "...", "checkpoint_sha256": "...",
"metadata_sha256": "...", "allow_source_schema_change": false}]}.
Relative checkpoint paths resolve against the manifest's directory. A changed
schema requires an explicit per-policy opt-in; dimensions and neural tensors
must still match. Only local checkpoint files already trusted by the operator
should be supplied (the project's checkpoint format uses torch serialization).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from app.checkpoints import CheckpointRegistry
from app.scenarios import get_spec
from app.settings import Settings
from app.trainer import (
    EVALUATION_SEED_BASE, Trainer, aggregate_evaluations, capture_rng_state,
    source_digest,
)

HOLDOUT_SEED_BASE = 5_200_000
HOLDOUT_EPISODES = 50


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def validate_source(item, base_dir=None):
    """Verify the original signed pair without constructing a mutating registry."""
    if not isinstance(item, dict):
        raise ValueError("each policy must be an object")
    spec = get_spec(item["scenario_id"])
    seed, schema = item["seed"], item["source_schema"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if type(schema) is not int or schema < 1:
        raise ValueError("source_schema must be a positive integer")
    if (schema != spec.checkpoint_schema
            and item.get("allow_source_schema_change") is not True):
        raise ValueError("source schema change requires explicit per-policy opt-in")
    expected_engine = _digest(item["source_engine_source_sha256"], "source engine")
    expected_tensor = _digest(item["checkpoint_sha256"], "checkpoint hash")
    expected_metadata = _digest(item["metadata_sha256"], "metadata hash")
    path = Path(item["checkpoint"])
    if not path.is_absolute():
        path = Path(base_dir or Path.cwd()) / path
    path = path.resolve(strict=True)
    if path.suffix != ".json":
        raise ValueError("checkpoint must identify the original JSON sidecar")
    raw = json.loads(path.read_text())
    episode = raw.get("episode")
    if type(episode) is not int or episode < 0:
        raise ValueError("source episode must be a nonnegative integer")
    if raw.get("metadata_sha256") != expected_metadata:
        raise ValueError("manifest metadata hash does not match original sidecar")
    if raw.get("checkpoint_sha256") != expected_tensor:
        raise ValueError("manifest checkpoint hash does not match original sidecar")
    # __init__ archives incompatible active files; validation must never do that.
    registry = CheckpointRegistry.__new__(CheckpointRegistry)
    registry.dir, registry.schema_version = path.parent, schema
    data, metadata = registry._validated_pair(path, path.with_suffix(".pt"), episode)
    if metadata.get("seed") != seed:
        raise ValueError("manifest seed does not match original checkpoint")
    if (metadata.get("protocol") or {}).get("engine_source_sha256") != expected_engine:
        raise ValueError("manifest engine does not match original training engine")
    env = spec.make_env(False)
    for key in ("obs_dim", "n_continuous", "n_binary"):
        if metadata.get(key) != getattr(env, key):
            raise ValueError(f"current scenario {key} differs from source checkpoint")
    origin = {
        "checkpoint_path": str(path), "scenario_id": spec.id,
        "scenario_binding": "operator-declared selected manifest entry",
        "seed": seed, "episode": episode,
        "schema_version": schema, "timestamp": metadata.get("timestamp"),
        "engine_source_sha256": expected_engine,
        "checkpoint_sha256": expected_tensor,
        "metadata_sha256": expected_metadata,
        "sidecar_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "update_count": metadata.get("update_count", 0),
        "total_steps": metadata.get("total_steps"),
        "protocol": metadata.get("protocol"),
    }
    return {"spec": spec, "data": data, "metadata": metadata, "origin": origin}


def transfer_network(source, trainer):
    """Transfer exactly matching finite tensors, retaining the fresh optimizer."""
    incoming = source["data"]["agent"]["network"]
    current = trainer.agent.network.state_dict()
    if incoming.keys() != current.keys():
        raise ValueError("source network keys are incompatible with current actor")
    for key, value in incoming.items():
        expected = current[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"source network {key} is not a tensor")
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"source network {key} shape or dtype is incompatible")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"source network {key} contains non-finite weights")
    trainer.agent.network.load_state_dict(incoming, strict=True)
    digest = hashlib.sha256()
    for key in sorted(incoming):
        tensor = incoming[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def evaluate_candidate(trainer, seed_base):
    """Observe the unchanged learned policy; never call update/start/reference."""
    fixed = trainer._run_eval()
    canonical = fixed.get("canonical_summary")
    if canonical is None:
        raise RuntimeError("current engine must return its canonical replay summary")
    trials = []
    for index in range(HOLDOUT_EPISODES):
        env = trainer.spec.make_env(True)
        env.rng.seed(seed_base + index)
        observation = env.reset()
        for _ in range(env.max_steps):
            observation, _, done, _ = env.step(trainer.agent.predict(observation))
            if done:
                break
        trials.append({"seed": seed_base + index, **env.episode_summary()})
    return {
        "fixed": fixed, "canonical": canonical,
        "holdout": {**aggregate_evaluations(trials, trainer.spec.metric_mode),
                    "successes": sum(bool(t["success"]) for t in trials),
                    "trials": trials},
    }


def qualifies(evidence):
    return bool(
        evidence["fixed"]["episodes"] == 10
        and evidence["fixed"]["success_rate"] == 1.0
        and evidence["holdout"]["episodes"] == HOLDOUT_EPISODES
        and evidence["holdout"]["successes"] == HOLDOUT_EPISODES
        and evidence["canonical"].get("success") is True
    )


def _write_report(path, report):
    temporary = path.with_name(".qualification-report.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(path)


def requalify_manifest(manifest, output, *, base_dir=None,
                       holdout_seed_base=HOLDOUT_SEED_BASE):
    """Continue after individual failures; only accepted policies are exported."""
    items = manifest.get("policies") if isinstance(manifest, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("manifest must contain a nonempty policies list")
    if (type(holdout_seed_base) is not int
            or not HOLDOUT_SEED_BASE <= holdout_seed_base <= 2**32 - HOLDOUT_EPISODES):
        raise ValueError("qualification holdout seeds must start at or above 5200000")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": "frozen-policy-requalification-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine_source_sha256": source_digest(),
        "runtime": {"device": "cpu", "cpu_threads": 1,
                    "torch_version": str(torch.__version__)},
        "claim": "current-runtime qualification of historically selected frozen weights",
        "new_training_performed": False,
        "holdout_seed_base": holdout_seed_base,
        "holdout_episodes": HOLDOUT_EPISODES,
        "holdout_role": (
            "predeclared current-engine acceptance gate; not a new training run, "
            "repeated invocations reuse this suite and are not new independent evidence"),
        "acceptance": {"fixed_successes": 10, "fixed_episodes": 10,
                       "holdout_min_successes": 50, "holdout_episodes": 50,
                       "canonical_success_required": True},
        "requested_policies": len(items), "state": "running",
        "policies": [], "all_passed": False,
    }
    for index, item in enumerate(items):
        result = {"input": item, "accepted": False,
                  "scenario_id": item.get("scenario_id") if isinstance(item, dict) else None,
                  "seed": item.get("seed") if isinstance(item, dict) else None}
        try:
            source = validate_source(item, base_dir)
            spec, origin = source["spec"], source["origin"]
            result.update({"scenario_id": spec.id, "seed": origin["seed"],
                           "origin": origin})
            with tempfile.TemporaryDirectory(prefix="rl-requalification-") as folder:
                isolated = Path(folder)
                (isolated / "state.json").write_text(json.dumps(
                    {"active_scenario": spec.id}))
                trainer = Trainer(Settings(
                    port=8901, checkpoint_dir=isolated, checkpoint_every_n=50,
                    max_episodes=1, use_gpu=False, seed=origin["seed"],
                    eval_episodes=10, cpu_threads=1), initialize_actor=False)
                weights_digest = transfer_network(source, trainer)
                evidence = evaluate_candidate(trainer, holdout_seed_base)
                json.dumps(evidence, allow_nan=False)
                result.update({
                    "network_weights_sha256": weights_digest,
                    "fixed": {k: v for k, v in evidence["fixed"].items()
                              if k not in {"trajectory", "replay_frames"}},
                    "holdout": evidence["holdout"],
                    "canonical": evidence["canonical"],
                    "accepted": qualifies(evidence),
                })
                if result["accepted"]:
                    relative = Path(f"policy-{index:03d}") / spec.id
                    registry = CheckpointRegistry(output / relative.parent,
                                                  spec.id, spec.checkpoint_schema)
                    protocol = {
                        "algorithm": "frozen-policy requalification",
                        "version": 1, "engine_source_sha256": source_digest(),
                        "device": str(trainer.device),
                        "torch_version": str(torch.__version__),
                        "cpu_threads": torch.get_num_threads(),
                        "engine_role": "qualification and future-training runtime only",
                        "origin": origin,
                        "exact_training_continuation": False,
                        "transfer": "network weights only; optimizer and RNG freshly initialized",
                        "additional_training_episodes": 0,
                        "additional_training_steps": 0, "additional_optimizer_updates": 0,
                        "network_weights_sha256": weights_digest,
                        "expert_used_at_inference": False,
                        "evaluation_suite": evidence["fixed"]["evaluation_suite"],
                        "evaluation_seed_base": EVALUATION_SEED_BASE,
                        "deterministic_evaluation": True,
                        "requalification": {
                            "origin": {k: v for k, v in origin.items()
                                       if k != "protocol"},
                            "criteria": report["acceptance"],
                            "holdout_seed_base": holdout_seed_base,
                            "holdout_successes": evidence["holdout"]["successes"],
                            "holdout_episodes": HOLDOUT_EPISODES,
                            "canonical_summary": evidence["canonical"],
                        },
                    }
                    meta = registry.save(0, trainer.agent, [], {
                        **evidence["fixed"], "protocol": protocol,
                        "update_count": 0, "total_steps": 0,
                        "rng_state": capture_rng_state(trainer.env),
                    })
                    result["checkpoint"] = (
                        relative / "checkpoint_ep000000.json").as_posix()
                    result["export"] = asdict(meta)
                else:
                    result["error"] = "policy failed one or more declared acceptance gates"
        except Exception as exc:
            result["accepted"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
        report["policies"].append(result)
        complete = len(report["policies"]) == len(items)
        report["state"] = "complete" if complete else "running"
        report["all_passed"] = complete and all(
            p["accepted"] for p in report["policies"])
        _write_report(output / "qualification-report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new isolated directory; never an active lab root")
    parser.add_argument("--holdout-seed-base", type=int, default=HOLDOUT_SEED_BASE)
    args = parser.parse_args(argv)
    report = requalify_manifest(
        json.loads(args.manifest.read_text()), args.output,
        base_dir=args.manifest.resolve().parent,
        holdout_seed_base=args.holdout_seed_base)
    print(json.dumps({"all_passed": report["all_passed"], "policies": [
        {k: p.get(k) for k in ("scenario_id", "seed", "accepted", "error", "checkpoint")}
        for p in report["policies"]]}, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
