"""NERScorer — pairwise entity Jaccard overlap using scispaCy."""
from __future__ import annotations

import logging
import threading
from itertools import combinations

from pipeline.stages.summarization.interfaces.scoring import AgreementContext, ScoreBundle
from pipeline.stages.summarization.models import AuditableSummary

logger = logging.getLogger(__name__)

_nlp = None
_nlp_lock = threading.Lock()


def _get_nlp():
    """Lazy-load scispaCy model (module-level singleton, thread-safe)."""
    global _nlp
    if _nlp is None:
        with _nlp_lock:
            if _nlp is None:  # second check inside lock
                import scispacy  # noqa: F401 — registers scispacy pipeline components
                import spacy
                _nlp = spacy.load(
                    "en_core_sci_lg",
                    disable=["parser", "attribute_ruler", "lemmatizer"],
                )
    return _nlp


def _extract_entities(claims: list[str]) -> frozenset[str]:
    """Extract lowercased entity spans from a list of claim strings."""
    if not claims:
        return frozenset()
    nlp = _get_nlp()
    doc = nlp(" ".join(claims))
    return frozenset(ent.text.lower() for ent in doc.ents)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NERScorer:
    """
    Measures voter agreement via pairwise biomedical entity Jaccard overlap.

    For each voter output, entities are extracted from the claim strings using
    scispaCy (en_core_sci_lg).  Mean pairwise Jaccard similarity is stored in
    ScoreBundle.entity_overlap.

    The scispaCy model is lazy-loaded on first use and shared across instances.
    """

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,  # noqa: ARG002
        context: AgreementContext | None = None,  # noqa: ARG002
    ) -> ScoreBundle:
        entity_sets = [
            _extract_entities([f.claim for f in o.findings]) for o in outputs
        ]

        if len(entity_sets) < 2:
            return ScoreBundle(entity_overlap=1.0)

        scores = [_jaccard(a, b) for a, b in combinations(entity_sets, 2)]
        mean_score = sum(scores) / len(scores)

        logger.debug("NERScorer entity_overlap=%.3f", mean_score)
        return ScoreBundle(entity_overlap=mean_score)
