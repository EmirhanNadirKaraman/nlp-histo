"""Category Jaccard agreement strategy."""
from __future__ import annotations

from itertools import combinations

from pipeline.stages.summarization.interfaces.agreement import AgreementStrategy
from pipeline.stages.summarization.models import AuditableSummary


class CategoryJaccardAgreement:
    """
    Mean pairwise Jaccard similarity on the category sets each voter extracted.

    Fast and deterministic — requires no API calls.  Less sensitive than
    EmbeddingAgreement: two voters can extract very different claims within
    the same category and still score 1.0.
    """

    def compute(self, outputs: list[AuditableSummary]) -> float:
        cat_sets = [self._category_set(o) for o in outputs]
        scores = []
        for a, b in combinations(cat_sets, 2):
            union = a | b
            scores.append(len(a & b) / len(union) if union else 1.0)
        return sum(scores) / len(scores)

    @staticmethod
    def _category_set(output: AuditableSummary) -> frozenset[str]:
        return frozenset(f.category for f in output.findings)
