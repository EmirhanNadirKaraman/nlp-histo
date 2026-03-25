"""
Disk-backed cache for MAP, REDUCE, and RULE outputs.

Keys are derived from text_element_ids (deterministic, collision-free).
The cache survives kernel restarts and avoids redundant LLM calls when
rerunning the pipeline on the same data.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .models import AuditableSummary, ConsolidatedSummary, ExtractedRules

logger = logging.getLogger(__name__)


class PipelineCache:
    """
    Three-tier cache (map / reduce / rule) persisted as a single JSON file.

    Cache key design
    ----------------
    MAP   → sorted text_element_ids of the chunk (unique by DB constraint)
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
    def _map_key(sentences: list[dict]) -> str:
        ids = sorted(s.get("text_element_id", 0) for s in sentences)
        return ",".join(map(str, ids))

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

    def get_map(self, sentences: list[dict]) -> Optional[AuditableSummary]:
        key = self._map_key(sentences)
        if key in self._map:
            self._hits["map"] += 1
            return AuditableSummary(**self._map[key])
        self._misses["map"] += 1
        return None

    def set_map(self, sentences: list[dict], result: AuditableSummary) -> None:
        self._map[self._map_key(sentences)] = result.model_dump()

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
