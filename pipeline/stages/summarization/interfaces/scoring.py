"""Scoring DTOs shared across the agreement interface and its implementations."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChunkDecision(str, Enum):
    """Decision produced by a MapOutputScorer for a single MAP chunk."""

    KEEP = "keep"        # Accept best voter output; skip escalation LLM.
    REJECT = "reject"    # Discard chunk entirely (e.g. all voters empty).
    ESCALATE = "escalate"  # Hand off to the escalation LLM.


@dataclass
class ScoreBundle:
    """
    Aggregates all scorer outputs for a single MAP chunk.

    Individual scorers populate only their own field(s) and leave the rest as
    None.  A composite scorer (e.g. CascadedCompositeScorer) combines them and
    sets ``confidence`` and ``decision``.

    AgreementChecker applies a fallback theta decision when ``decision`` is
    still None after the scorer returns.
    """

    embedding_agreement: float | None = None
    judge_agreement: float | None = None
    entity_overlap: float | None = None
    confidence: float | None = None
    decision: ChunkDecision | None = None
