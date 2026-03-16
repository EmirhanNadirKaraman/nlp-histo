"""AgreementChecker — thin orchestrator: scorer + theta fallback + best()."""
from __future__ import annotations

import logging

from pipeline.stages.summarization.interfaces.agreement import MapOutputScorer
from pipeline.stages.summarization.interfaces.scoring import ChunkDecision, ScoreBundle
from pipeline.stages.summarization.models import AuditableSummary

logger = logging.getLogger(__name__)


class AgreementChecker:
    """
    Wraps a MapOutputScorer with a theta fallback and the best() helper.

    Responsibilities
    ----------------
    - Calls scorer.compute() and ensures ScoreBundle.decision is always set.
    - When the scorer does not set decision (leaf scorers like EmbeddingScorer,
      CategoryJaccardScorer), applies the theta threshold against the primary
      score field (confidence → embedding_agreement → 0.0).
    - CascadedCompositeScorer sets decision internally; theta is ignored.

    Parameters
    ----------
    scorer:
        Any object satisfying the MapOutputScorer protocol.
    theta:
        Fallback threshold.  Only used when scorer does not set a decision.
    """

    def __init__(self, scorer: MapOutputScorer, theta: float = 0.7) -> None:
        self._scorer = scorer
        self.theta = theta

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,
    ) -> ScoreBundle:
        """Run the scorer and guarantee ScoreBundle.decision is populated."""
        if len(outputs) < 2:
            return ScoreBundle(confidence=1.0, decision=ChunkDecision.KEEP)

        bundle = self._scorer.compute(outputs, source_text)

        if bundle.decision is None:
            primary = bundle.confidence or bundle.embedding_agreement or 0.0
            bundle.decision = (
                ChunkDecision.KEEP if primary >= self.theta else ChunkDecision.ESCALATE
            )

        logger.debug(
            "AgreementChecker [%s] emb=%s judge=%s conf=%s → %s",
            type(self._scorer).__name__,
            _fmt(bundle.embedding_agreement),
            _fmt(bundle.judge_agreement),
            _fmt(bundle.confidence),
            bundle.decision.value,
        )
        return bundle

    def best(self, outputs: list[AuditableSummary]) -> AuditableSummary:
        """Return the voter output with the most findings."""
        return max(outputs, key=lambda o: len(o.findings))


def _fmt(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "—"
