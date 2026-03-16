"""MapOutputScorer — protocol for MAP-stage voter-agreement scoring."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pipeline.stages.summarization.models import AuditableSummary
from pipeline.stages.summarization.interfaces.scoring import ScoreBundle


@runtime_checkable
class MapOutputScorer(Protocol):
    """
    Scores agreement between N voter outputs for a single MAP chunk.

    Implementations receive the voter AuditableSummary objects and optionally
    the original formatted source text, and return a populated ScoreBundle.

    Contract
    --------
    - Must populate at least one score field (embedding_agreement,
      judge_agreement, etc.).
    - May optionally set ScoreBundle.decision; if not set, AgreementChecker
      falls back to comparing the primary score against theta.
    - Must be stateless across calls (safe for concurrent use).
    """

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,
    ) -> ScoreBundle: ...
