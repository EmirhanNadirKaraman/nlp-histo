"""
REDUCE stage: collapses a list of AuditableSummary objects into a single
ConsolidatedSummary via recursive tree reduction, with caching.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Union

from .cache import PipelineCache
from .models import AuditableSummary, ConsolidatedSummary
from .prompts import build_reduce_chain

logger = logging.getLogger(__name__)


def _minify(summaries: list) -> str:
    """Compact JSON to minimise token usage in reduce prompts."""
    return json.dumps(
        [s.model_dump(exclude_none=True) for s in summaries],
        separators=(",", ":"),
    )


class ReduceStage:
    """
    Recursive tree-reduce over chunk summaries.

    Groups are reduced in batches of `batch_size`.  The recursion
    continues until a single ConsolidatedSummary remains.

    Parameters
    ----------
    llm:
        LLM for reduce calls (typically the smart model, e.g. gpt-4o).
    batch_size:
        Max number of summaries to combine in one reduce call.
    """

    def __init__(self, llm, batch_size: int = 10) -> None:
        self._chain = build_reduce_chain(llm)
        self.batch_size = batch_size

    # ── Public API ─────────────────────────────────────────────────────────────

    def reduce(
        self,
        summaries: list[Union[AuditableSummary, ConsolidatedSummary]],
        pmcid: str,
        cache: Optional[PipelineCache] = None,
    ) -> ConsolidatedSummary:
        """
        Recursively reduce `summaries` to a single ConsolidatedSummary.

        Each intermediate result is cached separately so that partial work
        survives restarts.
        """
        current = list(summaries)

        while len(current) > 1:
            groups = [
                current[i : i + self.batch_size]
                for i in range(0, len(current), self.batch_size)
            ]
            uncached_groups = []
            uncached_indices = []
            placeholders: list[Optional[ConsolidatedSummary]] = []

            for group in groups:
                if cache:
                    hit = cache.get_reduce(group, pmcid)
                    if hit:
                        placeholders.append(hit)
                        continue
                placeholders.append(None)
                uncached_groups.append(group)
                uncached_indices.append(len(placeholders) - 1)

            if uncached_groups:
                inputs = [
                    {
                        "pmcid": pmcid,
                        "num_chunks": len(g),
                        "summaries": _minify(g),
                    }
                    for g in uncached_groups
                ]
                new_results: list[ConsolidatedSummary] = self._chain.batch(inputs)
                for i, (group, result) in enumerate(zip(uncached_groups, new_results)):
                    placeholders[uncached_indices[i]] = result
                    if cache:
                        cache.set_reduce(group, pmcid, result)

            current = placeholders  # type: ignore[assignment]

        # Exactly one summary left — might still be an AuditableSummary
        # if the input had only one chunk.
        final = current[0]
        if isinstance(final, AuditableSummary):
            final = self._reduce_single(final, pmcid, cache)
        return final

    # ── Internals ──────────────────────────────────────────────────────────────

    def _reduce_single(
        self,
        summary: AuditableSummary,
        pmcid: str,
        cache: Optional[PipelineCache],
    ) -> ConsolidatedSummary:
        group = [summary]
        if cache:
            hit = cache.get_reduce(group, pmcid)
            if hit:
                return hit
        result: ConsolidatedSummary = self._chain.invoke(
            {
                "pmcid": pmcid,
                "num_chunks": 1,
                "summaries": _minify(group),
            }
        )
        if cache:
            cache.set_reduce(group, pmcid, result)
        return result
