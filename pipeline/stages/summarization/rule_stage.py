"""
RULE EXTRACTION stage: ConsolidatedSummary → ExtractedRules, with caching.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from .cache import PipelineCache
from .models import ConsolidatedSummary, ExtractedRules
from .prompts import build_rule_chain

logger = logging.getLogger(__name__)


class RuleStage:
    """
    Extract IF-THEN clinical rules from a ConsolidatedSummary.

    Parameters
    ----------
    llm:
        LLM for rule extraction (typically the smart model).
    """

    def __init__(self, llm) -> None:
        self._chain = build_rule_chain(llm)

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(
        self,
        summary: ConsolidatedSummary,
        pmcid: str,
        cache: Optional[PipelineCache] = None,
    ) -> ExtractedRules:
        if cache:
            hit = cache.get_rule(summary, pmcid)
            if hit:
                return hit

        result: ExtractedRules = self._chain.invoke(
            {
                "pmcid": pmcid,
                "summary": json.dumps(summary.model_dump(exclude_none=True), separators=(",", ":")),
            }
        )

        if cache:
            cache.set_rule(summary, pmcid, result)

        return result
