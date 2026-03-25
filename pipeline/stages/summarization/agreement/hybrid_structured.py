"""
HybridStructuredSimilarity — claim-centered multi-signal voter similarity (v1).

V1 design principle
-------------------
The production v1 agreement path is claim-centered.  All active signals
operate on ``Finding.claim``, ``Finding.category``, and ``Finding.evidence``.
In this v1 path, ``AuditableSummary.summary_text`` is not used.

V1 active signals
-----------------
category_overlap  (weight 0.25) — Jaccard of Finding.category sets
claim_embedding   (weight 0.40) — soft-alignment cosine of Finding.claim embeddings
entity_overlap    (weight 0.25) — Jaccard of biomedical entity sets from Finding.claim
evidence_overlap  (weight 0.10) — Jaccard of text_element_id sets from Finding.evidence

Numeric and polarity signals are NOT included in v1.

Future design space (not implemented)
--------------------------------------
A ``w_summary_text`` weight (default 0.0) is reserved for an optional
summary-prose similarity signal that could be added as an auxiliary component
in a future version — for example, as a fallback when claim extraction is
sparse, or as a separate ablation in evaluation.  It is intentionally absent
from v1 to keep the agreement object cleanly claim-centered.
"""
from __future__ import annotations

import numpy as np

from pipeline.stages.summarization.models import AuditableSummary
from .category_jaccard import _category_set
from .embedding import _align, _claims
from .embedding_similarity import EmbeddingSimilarityStrategy
from .ner_scorer import _extract_entities, _jaccard
from .providers import EmbedFn, OpenAIEmbedder


def _evidence_jaccard(a: AuditableSummary, b: AuditableSummary) -> float:
    """Jaccard similarity of te_id sets extracted from Finding.evidence citations."""

    def _te_ids(summary: AuditableSummary) -> frozenset[str]:
        ids: set[str] = set()
        for f in summary.findings:
            for ev in f.evidence:
                parts = ev.split("|")
                if len(parts) == 3:
                    ids.add(parts[2])
        return frozenset(ids)

    a_ids = _te_ids(a)
    b_ids = _te_ids(b)
    if not a_ids and not b_ids:
        return 1.0
    if not a_ids or not b_ids:
        return 0.0
    return len(a_ids & b_ids) / len(a_ids | b_ids)


def _cat_jaccard(a: AuditableSummary, b: AuditableSummary) -> float:
    cat_a = _category_set(a)
    cat_b = _category_set(b)
    union = cat_a | cat_b
    return len(cat_a & cat_b) / len(union) if union else 1.0


class HybridStructuredSimilarity:
    """
    Weighted combination of four structured similarity signals (v1).

    Weights are keyword arguments for tuning.  The ``compute_matrix()`` method
    batches the embedding component across all voters in a single API call,
    then computes the remaining three signals (all CPU-only) per pair.

    Parameters
    ----------
    embed_fn:
        Callable that maps a list of strings to embedding vectors.
        Defaults to OpenAIEmbedder (text-embedding-3-small).
    w_category, w_embedding, w_entity, w_evidence:
        Signal weights.  Must sum to 1.0 for a calibrated score in [0, 1].
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        w_category: float = 0.25,
        w_embedding: float = 0.40,
        w_entity: float = 0.25,
        w_evidence: float = 0.10,
    ) -> None:
        self._embed: EmbedFn = embed_fn or OpenAIEmbedder()
        self._w_category = w_category
        self._w_embedding = w_embedding
        self._w_entity = w_entity
        self._w_evidence = w_evidence

    def similarity(self, a: AuditableSummary, b: AuditableSummary) -> float:
        """Single-pair weighted combination (for unit tests and one-off calls)."""
        score = 0.0
        if self._w_category > 0.0:
            score += self._w_category * _cat_jaccard(a, b)
        if self._w_embedding > 0.0:
            score += self._w_embedding * _align(self._embed, _claims(a), _claims(b))
        if self._w_entity > 0.0:
            score += self._w_entity * _jaccard(
                _extract_entities([f.claim for f in a.findings]),
                _extract_entities([f.claim for f in b.findings]),
            )
        if self._w_evidence > 0.0:
            score += self._w_evidence * _evidence_jaccard(a, b)
        return score

    def compute_matrix(self, outputs: list[AuditableSummary]) -> np.ndarray:
        """
        Build N×N matrix with a single embedding API call.

        The embedding component is batched via EmbeddingSimilarityStrategy.
        Category, entity, and evidence signals are computed CPU-only per pair.
        """
        n = len(outputs)

        # Pre-compute per-voter signals (only for non-zero weights).
        emb_matrix = (
            EmbeddingSimilarityStrategy(self._embed).compute_matrix(outputs)
            if self._w_embedding > 0.0
            else None
        )
        cat_sets = (
            [_category_set(o) for o in outputs]
            if self._w_category > 0.0
            else None
        )
        ent_sets = (
            [_extract_entities([f.claim for f in o.findings]) for o in outputs]
            if self._w_entity > 0.0
            else None
        )

        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                score = 0.0
                if self._w_category > 0.0 and cat_sets is not None:
                    cat_a, cat_b = cat_sets[i], cat_sets[j]
                    cat_union = cat_a | cat_b
                    score += self._w_category * (
                        len(cat_a & cat_b) / len(cat_union) if cat_union else 1.0
                    )
                if self._w_embedding > 0.0 and emb_matrix is not None:
                    score += self._w_embedding * float(emb_matrix[i, j])
                if self._w_entity > 0.0 and ent_sets is not None:
                    score += self._w_entity * _jaccard(ent_sets[i], ent_sets[j])
                if self._w_evidence > 0.0:
                    score += self._w_evidence * _evidence_jaccard(outputs[i], outputs[j])
                matrix[i, j] = score
                matrix[j, i] = score

        return matrix
