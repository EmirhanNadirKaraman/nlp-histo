"""
Disk-backed cache for MAP, REDUCE, and RULE outputs.

The MAP cache key is keyed on (input_hash, schema_version, prompt_version,
stage_name, cascade_signature) so a schema/prompt bump or a cascade reshuffle
naturally invalidates stale entries. Each MAP entry stores the run metadata
alongside the result (``{"metadata": {...}, "result": {...}}``).

REDUCE / RULE caches are unrelated legacy caches; they retain the old
key shape. TODO(task-1 follow-up): version-tag those too once their prompts
are bumped.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from .models import (
    AuditableSummary,
    ConsolidatedSummary,
    ExtractedRules,
    MAP_PROMPT_VERSION,
    MAP_SCHEMA_VERSION,
    MAP_STAGE_NAME,
    MapRunMetadata,
)

logger = logging.getLogger(__name__)


class PipelineCache:
    """
    Three-tier cache (map / reduce / rule) persisted as a single JSON file.

    MAP key components
    ------------------
    schema_version + prompt_version + stage_name + cascade_signature + input_hash

    where ``input_hash = sha256(sorted(text_element_ids))`` is the chunk
    identity. The cascade signature is a stable hash of the ordered
    (provider, model) sequence used at L1+L2+L3 (see
    ``models.compute_cascade_signature``).

    REDUCE → pmcid + sorted chunk_ids of inputs
    RULE   → pmcid + sorted text_element_ids from the consolidated summary
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._map: dict[str, dict] = {}
        self._reduce: dict[str, dict] = {}
        self._rule: dict[str, dict] = {}
        self._hits = {"map": 0, "reduce": 0, "rule": 0}
        self._misses = {"map": 0, "reduce": 0, "rule": 0}
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._map = data.get("map", {})
            self._reduce = data.get("reduce", {})
            self._rule = data.get("rule", {})
            logger.info(
                "Cache loaded: %d map / %d reduce / %d rule entries",
                len(self._map), len(self._reduce), len(self._rule),
            )
        except Exception as exc:
            logger.warning("Could not load cache from %s: %s", self.path, exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"map": self._map, "reduce": self._reduce, "rule": self._rule}),
            encoding="utf-8",
        )

    # ── Key builders ───────────────────────────────────────────────────────────

    @staticmethod
    def _input_hash(sentences: list[dict]) -> str:
        ids = sorted(s.get("text_element_id", 0) for s in sentences)
        payload = ",".join(map(str, ids)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @staticmethod
    def _map_key(sentences: list[dict], metadata: MapRunMetadata) -> str:
        """Deterministic versioned key for one MAP chunk.

        Format: ``{schema_version}|{prompt_version}|{stage}|{cascade}|{input_hash}``
        """
        return "|".join((
            metadata.schema_version,
            metadata.prompt_version,
            metadata.stage_name,
            metadata.cascade_signature,
            PipelineCache._input_hash(sentences),
        ))

    @staticmethod
    def _reduce_key(summaries: list, pmcid: str) -> str:
        ids = []
        for s in summaries:
            if hasattr(s, "chunk_id"):
                ids.append(s.chunk_id)
            elif hasattr(s, "pmcid"):
                ids.append(f"reduced:{s.audit_trail.chunks_processed}")
            elif isinstance(s, dict):
                ids.append(s.get("chunk_id", s.get("concept", "?")))
        return f"{pmcid}|" + ";".join(sorted(ids))

    @staticmethod
    def _rule_key(summary: ConsolidatedSummary, pmcid: str) -> str:
        te_ids = sorted(summary.audit_trail.unique_text_element_ids)
        return f"{pmcid}|rule|" + ",".join(map(str, te_ids))

    # ── MAP ────────────────────────────────────────────────────────────────────

    def get_map(
        self,
        sentences: list[dict],
        metadata: MapRunMetadata,
    ) -> Optional[AuditableSummary]:
        """Versioned MAP cache lookup.

        Returns a hit only when the stored entry's ``metadata`` matches the
        caller's ``metadata`` exactly. Old un-wrapped entries are always
        treated as misses (they will be overwritten on the next ``set_map``).
        """
        key = self._map_key(sentences, metadata)
        entry = self._map.get(key)
        if entry is None or "result" not in entry or "metadata" not in entry:
            self._misses["map"] += 1
            return None
        # Belt-and-braces: the key includes versions, but if a future bug
        # writes mismatched metadata under the same key, fail closed.
        stored_meta = entry["metadata"]
        for field in ("schema_version", "prompt_version", "stage_name",
                      "cascade_signature"):
            if stored_meta.get(field) != getattr(metadata, field):
                self._misses["map"] += 1
                return None
        self._hits["map"] += 1
        return AuditableSummary(**entry["result"])

    def set_map(
        self,
        sentences: list[dict],
        result: AuditableSummary,
        metadata: MapRunMetadata,
    ) -> None:
        self._map[self._map_key(sentences, metadata)] = {
            "metadata": metadata.model_dump(),
            "result":   result.model_dump(),
        }

    # ── REDUCE ─────────────────────────────────────────────────────────────────

    def get_reduce(self, summaries: list, pmcid: str) -> Optional[ConsolidatedSummary]:
        key = self._reduce_key(summaries, pmcid)
        if key in self._reduce:
            self._hits["reduce"] += 1
            return ConsolidatedSummary(**self._reduce[key])
        self._misses["reduce"] += 1
        return None

    def set_reduce(self, summaries: list, pmcid: str, result: ConsolidatedSummary) -> None:
        self._reduce[self._reduce_key(summaries, pmcid)] = result.model_dump()

    # ── RULE ───────────────────────────────────────────────────────────────────

    def get_rule(self, summary: ConsolidatedSummary, pmcid: str) -> Optional[ExtractedRules]:
        key = self._rule_key(summary, pmcid)
        if key in self._rule:
            self._hits["rule"] += 1
            return ExtractedRules(**self._rule[key])
        self._misses["rule"] += 1
        return None

    def set_rule(self, summary: ConsolidatedSummary, pmcid: str, result: ExtractedRules) -> None:
        self._rule[self._rule_key(summary, pmcid)] = result.model_dump()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats_str(self) -> str:
        lines = ["Cache stats:"]
        total_h = total_m = 0
        for tier in ("map", "reduce", "rule"):
            h, m = self._hits[tier], self._misses[tier]
            total_h += h
            total_m += m
            rate = f"{h/(h+m):.0%}" if (h + m) else "n/a"
            lines.append(f"  {tier:6}: {h} hits / {m} misses ({rate})")
        overall = f"{total_h/(total_h+total_m):.0%}" if (total_h + total_m) else "n/a"
        lines.append(f"  overall: {overall}")
        return "\n".join(lines)
