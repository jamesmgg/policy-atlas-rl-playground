"""Checkpoint registry, one subdirectory per scenario.

Each checkpoint is a torch .pt (weights + optimizer + reward history + the
evaluation ghost-lap trajectory) plus a small .json sidecar so listing and
leaderboards never load tensors.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from .ppo.agent import PPOAgent

log = logging.getLogger("checkpoints")


class IncompatibleCheckpointError(RuntimeError):
    """Raised before model mutation when a checkpoint contract is obsolete."""


class CheckpointIntegrityError(RuntimeError):
    """Raised when a checkpoint and its committed sidecar do not match."""


@dataclass
class CheckpointMeta:
    episode: int
    timestamp: str
    mean_reward: float            # mean over the trailing 50 episodes
    eval_reward: float
    eval_metric: float | None     # scenario metric from the eval episode
    eval_reward_std: float = 0.0
    eval_metric_std: float | None = None
    eval_failure_progress: float | None = None
    eval_episodes: int = 1
    success_rate: float | None = None
    success_ci_low: float | None = None
    success_ci_high: float | None = None
    evaluation_suite: str | None = None
    seed: int | None = None
    update_count: int = 0
    total_steps: int | None = None
    schema_version: int = 1
    obs_dim: int | None = None
    n_continuous: int | None = None
    n_binary: int | None = None
    protocol: dict | None = None
    training_diagnostics: dict | None = None
    metadata_sha256: str | None = None
    checkpoint_sha256: str | None = None


def _normalize_meta(m: dict) -> dict:
    """Read legacy sidecars (pre-multi-scenario) into the current shape."""
    if "eval_metric" not in m:
        m["eval_metric"] = m.get("best_lap_time")
    m.setdefault("eval_reward_std", 0.0)
    m.setdefault("eval_metric_std", None)
    m.setdefault("eval_failure_progress", None)
    m.setdefault("eval_episodes", 1)
    m.setdefault("success_rate", None)
    m.setdefault("success_ci_low", None)
    m.setdefault("success_ci_high", None)
    m.setdefault("evaluation_suite", None)
    m.setdefault("seed", None)
    m.setdefault("update_count", 0)
    m.setdefault("total_steps", None)
    m.setdefault("schema_version", 1)
    m.setdefault("obs_dim", None)
    m.setdefault("n_continuous", None)
    m.setdefault("n_binary", None)
    m.setdefault("protocol", None)
    m.setdefault("training_diagnostics", None)
    m.setdefault("metadata_sha256", None)
    m.setdefault("checkpoint_sha256", None)
    return m


class CheckpointRegistry:
    def __init__(self, root: Path, scenario_id: str, schema_version: int = 1):
        self.dir = root / scenario_id
        self.schema_version = schema_version
        self.dir.mkdir(parents=True, exist_ok=True)
        self._archive_incompatible_active()

    def _pt(self, episode: int) -> Path:
        return self.dir / f"checkpoint_ep{episode:06d}.pt"

    def _json(self, episode: int) -> Path:
        return self.dir / f"checkpoint_ep{episode:06d}.json"

    def save(self, episode: int, agent: PPOAgent, history: list[dict],
             eval_result: dict) -> CheckpointMeta:
        recent = [h["reward"] for h in history[-50:]]
        meta = CheckpointMeta(
            episode=episode,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            mean_reward=round(sum(recent) / max(len(recent), 1), 2),
            eval_reward=round(eval_result["reward"], 2),
            eval_metric=eval_result["metric"],
            eval_reward_std=round(eval_result.get("reward_std", 0.0), 3),
            eval_metric_std=(round(eval_result["metric_std"], 3)
                             if eval_result.get("metric_std") is not None else None),
            eval_failure_progress=(round(eval_result["failure_progress"], 3)
                                   if eval_result.get("failure_progress") is not None else None),
            eval_episodes=int(eval_result.get("episodes", 1)),
            success_rate=(round(eval_result["success_rate"], 4)
                          if eval_result.get("success_rate") is not None else None),
            success_ci_low=(round(eval_result["success_ci_low"], 4)
                            if eval_result.get("success_ci_low") is not None else None),
            success_ci_high=(round(eval_result["success_ci_high"], 4)
                             if eval_result.get("success_ci_high") is not None else None),
            evaluation_suite=eval_result.get("evaluation_suite"),
            seed=eval_result.get("seed"),
            update_count=int(eval_result.get("update_count", 0)),
            total_steps=eval_result.get("total_steps"),
            schema_version=self.schema_version,
            obs_dim=agent.obs_dim,
            n_continuous=agent.n_continuous,
            n_binary=agent.n_binary,
            protocol=eval_result.get("protocol"),
            training_diagnostics=eval_result.get("training_diagnostics"),
        )
        meta.metadata_sha256 = _metadata_sha256(asdict(meta))
        final_pt = self._pt(episode)
        final_json = self._json(episode)
        token = uuid.uuid4().hex
        temp_pt = self.dir / f".{final_pt.name}.{token}.tmp"
        temp_json = self.dir / f".{final_json.name}.{token}.tmp"
        try:
            torch.save({
                "agent": agent.state_dict(),
                "history": history,
                "trajectory": eval_result["trajectory"],
                "update_count": int(eval_result.get("update_count", 0)),
                "total_steps": eval_result.get("total_steps"),
                "rng_state": eval_result.get("rng_state"),
                "meta": asdict(meta),
            }, temp_pt)
            meta.checkpoint_sha256 = _sha256(temp_pt)
            temp_json.write_text(json.dumps(asdict(meta)))
            _sync_file(temp_pt)
            _sync_file(temp_json)
            os.replace(temp_pt, final_pt)
            os.replace(temp_json, final_json)
        finally:
            temp_pt.unlink(missing_ok=True)
            temp_json.unlink(missing_ok=True)
        return meta

    def list(self) -> list[dict]:
        metas = []
        for path in sorted(self.dir.glob("checkpoint_ep*.json")):
            try:
                raw_meta = json.loads(path.read_text())
                # Verify the exact shape that was originally signed before
                # adding defaults introduced by a newer reader.
                if not _metadata_is_valid(raw_meta):
                    continue
                meta = _normalize_meta(raw_meta)
                episode = int(meta["episode"])
                if (int(meta["schema_version"]) == self.schema_version
                        and self._pt(episode).exists()):
                    metas.append(meta)
            except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
                continue
        return metas

    def latest_meta(self) -> dict | None:
        metas = self.list()
        return metas[-1] if metas else None

    def load(self, episode: int) -> dict:
        return self._load_pair(self._json(episode), self._pt(episode), episode)

    def _load_pair(self, sidecar_path: Path, checkpoint_path: Path,
                   episode: int) -> dict:
        """Validate and load one checkpoint pair without mutating an agent."""
        data, _ = self._validated_pair(sidecar_path, checkpoint_path, episode)
        return data

    def _validated_pair(self, sidecar_path: Path, checkpoint_path: Path,
                        episode: int) -> tuple[dict, dict]:
        """Return the tensor payload and its validated, normalized sidecar."""
        raw_meta = json.loads(sidecar_path.read_text())
        if not _metadata_is_valid(raw_meta):
            raise CheckpointIntegrityError(
                f"checkpoint episode {episode} metadata failed its SHA-256 check")
        meta = _normalize_meta(raw_meta)
        if int(meta["schema_version"]) != self.schema_version:
            raise IncompatibleCheckpointError(
                f"checkpoint schema {meta['schema_version']} is incompatible "
                f"with scenario schema {self.schema_version}")
        expected_hash = meta.get("checkpoint_sha256")
        if expected_hash is not None and _sha256(checkpoint_path) != expected_hash:
            raise CheckpointIntegrityError(
                f"checkpoint episode {episode} failed its SHA-256 check")
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        embedded = data.get("meta")
        if isinstance(embedded, dict):
            raw_embedded = dict(embedded)
            if not _metadata_is_valid(raw_embedded):
                raise CheckpointIntegrityError(
                    f"checkpoint episode {episode} embedded metadata is invalid")
            embedded = _normalize_meta(raw_embedded)
            if _comparable_metadata(meta) != _comparable_metadata(embedded):
                raise CheckpointIntegrityError(
                    f"checkpoint episode {episode} sidecar metadata does not "
                    "match the tensor payload")
        return data, meta

    def load_into(
        self,
        episode: int,
        agent: PPOAgent,
        *,
        expected_engine: str,
        expected_evaluation_suite: str,
    ) -> dict:
        """Load resumable state only after the experiment contract matches."""
        data, meta = self._validated_pair(
            self._json(episode), self._pt(episode), episode)
        _assert_resume_compatible(
            meta,
            expected_engine=expected_engine,
            expected_evaluation_suite=expected_evaluation_suite,
        )
        agent.load_state_dict(data["agent"])
        return data

    def archive_current(self) -> Path | None:
        """Move the active run aside so reset starts an isolated, recoverable run."""
        files = sorted(self.dir.glob("checkpoint_ep*"))
        return self._archive(files)

    def _archive_incompatible_active(self) -> Path | None:
        """Preserve obsolete active pairs before their episode names are reused.

        Schema-incompatible checkpoints are deliberately absent from ``list``.
        Without this startup migration, however, a fresh run could silently
        replace an old ``checkpoint_epNNNNNN`` pair with the same episode
        number. Keep the files recoverable for the older experiment engine.
        """
        files: set[Path] = set()
        old_schemas: set[int] = set()
        for sidecar in sorted(self.dir.glob("checkpoint_ep*.json")):
            try:
                meta = _normalize_meta(json.loads(sidecar.read_text()))
                old_schema = int(meta["schema_version"])
            except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
                continue
            if old_schema == self.schema_version:
                continue
            old_schemas.add(old_schema)
            files.add(sidecar)
            checkpoint = sidecar.with_suffix(".pt")
            if checkpoint.exists():
                files.add(checkpoint)
        if not files:
            return None
        versions = "-".join(str(version) for version in sorted(old_schemas))
        archive = self._archive(
            sorted(files), prefix=f"schema-{versions}-to-{self.schema_version}-")
        log.warning(
            "archived %d schema-incompatible checkpoint files in %s",
            len(files), archive,
        )
        return archive

    def archive_after(self, episode: int) -> Path | None:
        """Preserve descendants before branching from an older checkpoint."""
        files = [
            path for path in sorted(self.dir.glob("checkpoint_ep*"))
            if int(path.stem.rsplit("ep", 1)[1]) > episode
        ]
        return self._archive(files)

    def quarantine_episode(self, episode: int) -> Path | None:
        """Remove a corrupt pair from the active catalog without deleting it."""
        files = [path for path in (self._pt(episode), self._json(episode))
                 if path.exists()]
        return self._archive(files, prefix="invalid-")

    def list_archives(self) -> list[dict]:
        """Summarize recoverable branches without exposing invalid quarantine."""
        root = self.dir / "archive"
        if not root.exists():
            return []
        runs = []
        for directory in sorted(root.iterdir(), reverse=True):
            if not directory.is_dir() or directory.name.startswith("invalid-"):
                continue
            metas = []
            for sidecar in sorted(directory.glob("checkpoint_ep*.json")):
                try:
                    raw_meta = json.loads(sidecar.read_text())
                    pt = directory / sidecar.with_suffix(".pt").name
                    # Validate the exact historical shape before adding modern
                    # defaults; otherwise a new optional field would alter an
                    # older sidecar's committed metadata hash.
                    if _metadata_is_valid(raw_meta) and pt.exists():
                        normalized = _normalize_meta(raw_meta)
                        normalized["episode"] = int(normalized["episode"])
                        metas.append(normalized)
                except (json.JSONDecodeError, OSError, KeyError, ValueError):
                    continue
            if metas:
                latest = max(metas, key=lambda item: item["episode"])
                schemas = {int(meta["schema_version"]) for meta in metas}
                compatible = schemas == {self.schema_version}
                runs.append({
                    "id": directory.name,
                    "latest_episode": latest["episode"],
                    "checkpoints": len(metas),
                    "seed": latest.get("seed"),
                    "timestamp": latest.get("timestamp"),
                    "schema_version": int(latest["schema_version"]),
                    "compatible": compatible,
                })
        return runs

    def restore_archive(self, archive_id: str) -> bool:
        """Swap a recoverable branch into the active checkpoint directory."""
        if Path(archive_id).name != archive_id:
            return False
        target = self.dir / "archive" / archive_id
        runs = {
            run["id"] for run in self.list_archives()
            if run.get("compatible", False)
        }
        if archive_id not in runs or not target.is_dir():
            return False
        files = sorted(target.glob("checkpoint_ep*"))
        if not files:
            return False
        try:
            sidecars = sorted(target.glob("checkpoint_ep*.json"))
            if not sidecars:
                return False
            for sidecar in sidecars:
                episode = int(sidecar.stem.rsplit("ep", 1)[1])
                checkpoint = target / sidecar.with_suffix(".pt").name
                self._load_pair(sidecar, checkpoint, episode)
        except (CheckpointIntegrityError, IncompatibleCheckpointError,
                FileNotFoundError, json.JSONDecodeError, KeyError, OSError,
                RuntimeError, TypeError, ValueError):
            log.exception("refusing invalid archived run %s", archive_id)
            return False
        self.archive_current()
        for path in files:
            path.rename(self.dir / path.name)
        try:
            target.rmdir()
        except OSError:
            pass
        return True

    def _archive(self, files: list[Path], prefix: str = "") -> Path | None:
        if not files:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = self.dir / "archive" / f"{prefix}{stamp}"
        target.mkdir(parents=True, exist_ok=False)
        for path in files:
            path.rename(target / path.name)
        return target


def migrate_flat_layout(root: Path, scenario_id: str = "apex-gp") -> None:
    """Move pre-multi-scenario flat checkpoint files into their scenario dir."""
    flat = sorted(root.glob("checkpoint_ep*"))
    if not flat:
        return
    target = root / scenario_id
    target.mkdir(parents=True, exist_ok=True)
    for path in flat:
        path.rename(target / path.name)
    log.info("migrated %d flat checkpoint files into %s/", len(flat), scenario_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_sha256(meta: dict) -> str:
    content = {
        key: value for key, value in meta.items()
        if key not in {"checkpoint_sha256", "metadata_sha256"}
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _metadata_is_valid(meta: dict) -> bool:
    expected = meta.get("metadata_sha256")
    return expected is None or expected == _metadata_sha256(meta)


def _comparable_metadata(meta: dict) -> dict:
    return {key: value for key, value in meta.items()
            if key != "checkpoint_sha256"}


def _assert_resume_compatible(
    meta: dict,
    *,
    expected_engine: str,
    expected_evaluation_suite: str,
) -> None:
    """Reject scientifically different policies before any live-state mutation."""
    protocol = meta.get("protocol")
    checkpoint_engine = (
        protocol.get("engine_source_sha256")
        if isinstance(protocol, dict) else None
    )
    checkpoint_suite = meta.get("evaluation_suite")
    mismatches = []
    if checkpoint_engine != expected_engine:
        mismatches.append(
            f"engine {checkpoint_engine!r} does not match {expected_engine!r}")
    if checkpoint_suite != expected_evaluation_suite:
        mismatches.append(
            "evaluation suite "
            f"{checkpoint_suite!r} does not match {expected_evaluation_suite!r}")
    if mismatches:
        raise IncompatibleCheckpointError(
            "checkpoint cannot be resumed: " + "; ".join(mismatches)
            + "; replay remains available"
        )


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
