"""Filesystem persistence for summarization-stage artifacts.

This module is purely observational: it serialises stage outputs to JSONL +
JSON without changing any extraction logic. Persistence is OFF by default —
it is activated by passing ``artifact_root`` to :class:`SummarizationRunner`.

Layout::

    runs/{run_id}/
        manifest.json
        map/{pmcid}/findings.jsonl
        map/{pmcid}/chunks.jsonl
        map/{pmcid}/rejected_findings.jsonl
        map/{pmcid}/bad_findings.jsonl              (if collected)
        map/{pmcid}/enum_observations.jsonl         (if collected)
        normalize/{pmcid}/normal_findings.jsonl
        normalize/{pmcid}/entity_links.jsonl
        normalize/{pmcid}/dedup_trace.jsonl
        group/{pmcid}/groups.jsonl
        group/{pmcid}/non_groupable.jsonl
        canonicalize/{pmcid}/canonical_rules.jsonl
        relate/{pmcid}/relations.jsonl
        relate/{pmcid}/raw_pairs.jsonl
        relate/corpus/relations.jsonl               (when corpus relate emits)
        resolve/{pmcid}/final_rules.jsonl
        resolve/{pmcid}/score_trace.jsonl
        logs/

Lineage caveats (TODO):
- MAP ``Finding`` has no stable ``finding_id``. v1 stores
  ``(pmcid, chunk_id, position_in_chunk, evidence_refs)`` as the MAP coordinate.
- NORMALIZE ``source_finding_ids`` field is reserved but currently unpopulated;
  ``dedup_trace.jsonl`` records evidence-span coordinates instead.
- Relate skipped/blocking trace is currently not exposed by RelateStage. v1
  only records counts in the manifest summary block (TODO: full per-pair trace).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ── JSON coercion ────────────────────────────────────────────────────────────

def _to_jsonable(obj: Any) -> Any:
    """Coerce an arbitrary object into a JSON-safe structure.

    Handles Pydantic v2 models, dataclasses, enums, sets/frozensets, datetimes,
    Paths, and plain containers. Falls back to ``repr()`` for unknown types so
    serialisation never crashes — telemetry must never break the pipeline.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    # Pydantic v2
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_jsonable(model_dump())
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_to_jsonable(v) for v in sorted(obj, key=lambda x: str(x))]
    # Best-effort fallback: object's __dict__, then repr
    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable({k: v for k, v in vars(obj).items()
                                 if not k.startswith("_")})
        except Exception:
            pass
    return repr(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False, default=str)


# ── Low-level writers ────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    """Bulk-write records as JSONL via temp-file + rename. Returns row count.

    Required output — raises on failure (caller decides whether to swallow).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    n = 0
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(_dumps(rec))
                fh.write("\n")
                n += 1
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return n


def append_jsonl(path: Path, record: Any) -> None:
    """Append one record; create parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_dumps(record))
        fh.write("\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(_to_jsonable(obj), fh, ensure_ascii=False, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> int:
    """Best-effort CSV writer. Returns rows written; logs and returns 0 on failure."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: _csv_value(row.get(k)) for k in columns})
        return len(rows)
    except Exception as exc:
        logger.warning("CSV write failed at %s: %s", path, exc)
        return 0


def _csv_value(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return _dumps(v)


# ── Manifest ─────────────────────────────────────────────────────────────────

@dataclass
class RunManifest:
    """In-memory manifest with stable on-disk shape.

    Fields are intentionally permissive — anything optional may be left None
    or empty. The writer never blocks on missing data.
    """
    run_id: str
    artifact_root: Path
    status: str = "running"               # running | completed | failed
    timestamp_start: str = ""
    timestamp_end: str | None = None
    papers: list[str] = field(default_factory=list)
    stages_attempted: list[str] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)
    schema_version: str | None = None
    prompt_version: str | None = None
    cascade_signature: str | None = None
    git_commit: str | None = None
    config: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    chunk_size: int | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "papers": list(self.papers),
            "stages_attempted": list(self.stages_attempted),
            "stages_completed": list(self.stages_completed),
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "cascade_signature": self.cascade_signature,
            "git_commit": self.git_commit,
            "config": self.config,
            "models": self.models,
            "thresholds": self.thresholds,
            "chunk_size": self.chunk_size,
            "error": self.error,
            "extra": self.extra,
        }


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _try_git_commit() -> str | None:
    """Best-effort git HEAD lookup. Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


# ── Top-level writer ─────────────────────────────────────────────────────────

class RunArtifactWriter:
    """Per-run filesystem writer for summarization-stage outputs.

    All required outputs (``write_stage_jsonl`` with required=True) raise on
    failure. Optional/trace data is logged and swallowed so missing optional
    fields never crash the pipeline.

    When a run directory already contains a ``manifest.json`` (e.g. when a
    batch of papers shares a ``run_id``), the existing manifest is loaded and
    merged with the supplied one so cumulative state (papers, stages_completed)
    isn't clobbered.
    """

    def __init__(
        self,
        run_id: str,
        root_dir: Path | str = "runs",
        manifest: RunManifest | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_dir: Path = Path(root_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

        new_manifest = manifest or RunManifest(
            run_id=run_id,
            artifact_root=Path(root_dir),
            timestamp_start=_now_iso(),
            git_commit=_try_git_commit(),
        )
        existing = self._load_existing_manifest()
        if existing is not None:
            new_manifest = self._merge_manifests(existing, new_manifest)
        self._manifest: RunManifest = new_manifest
        self.write_manifest()

    def _load_existing_manifest(self) -> dict | None:
        path = self.run_dir / "manifest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse existing manifest %s: %s", path, exc)
            return None

    @staticmethod
    def _merge_manifests(prior: dict, current: RunManifest) -> RunManifest:
        """Carry forward cumulative state from a prior manifest on disk.

        Papers, stages_attempted, stages_completed, timestamp_start are taken
        from the prior manifest (so a multi-paper batch run keeps growing the
        roster). Per-paper config/threshold fields fall through to the new
        manifest because they may have been refined since the previous call.
        """
        merged = RunManifest(
            run_id=current.run_id,
            artifact_root=current.artifact_root,
            status="running",
            timestamp_start=prior.get("timestamp_start") or current.timestamp_start,
            timestamp_end=None,
            papers=list(dict.fromkeys([*prior.get("papers", []), *current.papers])),
            stages_attempted=list(dict.fromkeys([
                *prior.get("stages_attempted", []), *current.stages_attempted,
            ])),
            stages_completed=list(dict.fromkeys([
                *prior.get("stages_completed", []), *current.stages_completed,
            ])),
            schema_version=current.schema_version or prior.get("schema_version"),
            prompt_version=current.prompt_version or prior.get("prompt_version"),
            cascade_signature=current.cascade_signature or prior.get("cascade_signature"),
            git_commit=current.git_commit or prior.get("git_commit"),
            config=current.config or prior.get("config", {}),
            models=current.models or prior.get("models", {}),
            thresholds=current.thresholds or prior.get("thresholds", {}),
            chunk_size=current.chunk_size if current.chunk_size is not None
                       else prior.get("chunk_size"),
            error=None,
            extra={**prior.get("extra", {}), **current.extra},
        )
        return merged

    # ── manifest ────────────────────────────────────────────────────────────

    @property
    def manifest(self) -> RunManifest:
        return self._manifest

    def update_manifest(self, **fields: Any) -> None:
        for key, value in fields.items():
            if hasattr(self._manifest, key):
                setattr(self._manifest, key, value)
            else:
                self._manifest.extra[key] = value
        self.write_manifest()

    def mark_stage_attempted(self, stage: str) -> None:
        if stage not in self._manifest.stages_attempted:
            self._manifest.stages_attempted.append(stage)
        self.write_manifest()

    def mark_stage_completed(self, stage: str) -> None:
        self.mark_stage_attempted(stage)
        if stage not in self._manifest.stages_completed:
            self._manifest.stages_completed.append(stage)
        self.write_manifest()

    def add_paper(self, pmcid: str) -> None:
        if pmcid and pmcid not in self._manifest.papers:
            self._manifest.papers.append(pmcid)
            self.write_manifest()

    def finalize(self, status: str, error: str | None = None) -> None:
        self._manifest.status = status
        self._manifest.timestamp_end = _now_iso()
        if error is not None:
            self._manifest.error = error
        self.write_manifest()

    def write_manifest(self) -> None:
        try:
            write_json(self.run_dir / "manifest.json", self._manifest.to_dict())
        except Exception as exc:
            logger.error("Failed to write manifest: %s", exc, exc_info=True)
            raise

    # ── stage path helpers ──────────────────────────────────────────────────

    def stage_dir(self, stage: str, pmcid: str | None = None) -> Path:
        d = self.run_dir / stage
        if pmcid:
            d = d / pmcid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stage_path(self, stage: str, pmcid: str | None, name: str) -> Path:
        return self.stage_dir(stage, pmcid) / name

    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    # ── stage writers ───────────────────────────────────────────────────────

    def write_stage_jsonl(
        self,
        stage: str,
        name: str,
        records: Iterable[Any],
        *,
        pmcid: str | None = None,
        required: bool = False,
    ) -> int:
        """Write a JSONL artifact. ``required=True`` raises on failure."""
        path = self.stage_path(stage, pmcid, name)
        try:
            n = write_jsonl(path, records)
            logger.debug("[%s/%s] wrote %d records to %s", stage, pmcid or "-", n, path)
            return n
        except Exception as exc:
            if required:
                logger.error("Required artifact failed: %s — %s", path, exc)
                raise
            logger.warning("Optional artifact failed: %s — %s", path, exc)
            return 0

    def write_stage_json(
        self,
        stage: str,
        name: str,
        obj: Any,
        *,
        pmcid: str | None = None,
        required: bool = False,
    ) -> None:
        path = self.stage_path(stage, pmcid, name)
        try:
            write_json(path, obj)
        except Exception as exc:
            if required:
                logger.error("Required artifact failed: %s — %s", path, exc)
                raise
            logger.warning("Optional artifact failed: %s — %s", path, exc)

    def write_error(self, stage: str, error: str, *, pmcid: str | None = None) -> None:
        """Append a stage-level error to the run's logs/ directory."""
        path = self.logs_dir() / "stage_errors.jsonl"
        try:
            append_jsonl(path, {
                "ts": _now_iso(),
                "stage": stage,
                "pmcid": pmcid,
                "error": error,
            })
        except Exception as exc:
            logger.warning("Failed to write stage error log: %s", exc)

    def write_stage_csv(
        self,
        stage: str,
        name: str,
        rows: list[dict],
        columns: list[str],
        *,
        pmcid: str | None = None,
    ) -> int:
        """Best-effort CSV summary; never raises."""
        path = self.stage_path(stage, pmcid, name)
        return write_csv(path, rows, columns)
