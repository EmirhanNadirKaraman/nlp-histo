"""CascadedCompositeScorer — embedding first, LLM judge only in uncertain band."""
from __future__ import annotations

import logging

from pipeline.stages.summarization.interfaces.agreement import MapOutputScorer
from pipeline.stages.summarization.interfaces.scoring import ChunkDecision, ScoreBundle
from pipeline.stages.summarization.models import AuditableSummary

logger = logging.getLogger(__name__)


class CascadedCompositeScorer:
    """
    Two-stage cascade that calls the LLM judge only when the embedding score
    falls in an uncertain middle band, saving LLM calls on easy chunks.

    Decision logic
    --------------
    embedding >= high_threshold  → KEEP   (confident without LLM judge)
    embedding <= low_threshold   → ESCALATE (clearly uncertain, skip judge)
    otherwise                    → call judge, combine:
                                   confidence = 0.6 * embedding + 0.4 * judge
                                   confidence >= 0.75 → KEEP else ESCALATE

    Parameters
    ----------
    embedding_scorer:
        Scorer that populates ScoreBundle.embedding_agreement.
    judge_scorer:
        Scorer that populates ScoreBundle.judge_agreement.  Only called when
        the embedding score falls in (low_threshold, high_threshold).
    high_threshold:
        Embedding score above which voters are accepted without the judge.
    low_threshold:
        Embedding score at or below which the chunk is immediately escalated.
    """

    def __init__(
        self,
        embedding_scorer: MapOutputScorer,
        judge_scorer: MapOutputScorer,
        high_threshold: float = 0.88,
        low_threshold: float = 0.45,
    ) -> None:
        self._embedding = embedding_scorer
        self._judge = judge_scorer
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,
    ) -> ScoreBundle:
        bundle = self._embedding.compute(outputs, source_text)
        emb = bundle.embedding_agreement or 0.0

        if emb >= self.high_threshold:
            bundle.confidence = emb
            bundle.decision = ChunkDecision.KEEP
            logger.debug("Cascade KEEP (emb=%.2f >= %.2f)", emb, self.high_threshold)
            return bundle

        if emb <= self.low_threshold:
            bundle.confidence = emb
            bundle.decision = ChunkDecision.ESCALATE
            logger.debug("Cascade ESCALATE (emb=%.2f <= %.2f)", emb, self.low_threshold)
            return bundle

        judge_bundle = self._judge.compute(outputs, source_text)
        bundle.judge_agreement = judge_bundle.judge_agreement
        judge = bundle.judge_agreement or 0.0

        bundle.confidence = 0.6 * emb + 0.4 * judge
        bundle.decision = (
            ChunkDecision.KEEP if bundle.confidence >= 0.75 else ChunkDecision.ESCALATE
        )
        logger.debug(
            "Cascade %s (emb=%.2f judge=%.2f conf=%.2f)",
            bundle.decision, emb, judge, bundle.confidence,
        )
        return bundle
