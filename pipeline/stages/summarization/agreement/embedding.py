"""EmbeddingScorer — soft-alignment cosine similarity on claim embeddings."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from pipeline.stages.summarization.interfaces.scoring import ScoreBundle
from pipeline.stages.summarization.models import AuditableSummary
from .providers import EmbedFn, OpenAIEmbedder


class EmbeddingScorer:
    """
    Mean pairwise soft-alignment cosine similarity on claim embeddings.

    For each pair of voters, every claim in voter A is matched to its nearest
    neighbour in voter B (and vice versa); the mean of both directed averages
    gives a symmetric score stored in ScoreBundle.embedding_agreement.

    Parameters
    ----------
    embed_fn:
        Callable that maps a list of strings to embedding vectors.
        Defaults to OpenAIEmbedder (text-embedding-3-small).
    """

    def __init__(self, embed_fn: EmbedFn | None = None) -> None:
        self._embed: EmbedFn = embed_fn or OpenAIEmbedder()

    def compute(
        self,
        outputs: list[AuditableSummary],
        source_text: str | None = None,  # noqa: ARG002
    ) -> ScoreBundle:
        claim_lists = [_claims(o) for o in outputs]
        scores = [_align(self._embed, a, b) for a, b in combinations(claim_lists, 2)]
        score = sum(scores) / len(scores) if scores else 1.0
        return ScoreBundle(embedding_agreement=score)


# ── Pure helpers (no self, easy to test in isolation) ──────────────────────────

def _claims(output: AuditableSummary) -> list[str]:
    return [f.claim for f in output.findings]


def _align(embed: EmbedFn, a_claims: list[str], b_claims: list[str]) -> float:
    """Symmetric soft-alignment cosine similarity between two claim lists."""
    if not a_claims and not b_claims:
        return 1.0
    if not a_claims or not b_claims:
        return 0.0

    embs = embed(a_claims + b_claims)
    emb_a = np.array(embs[: len(a_claims)])
    emb_b = np.array(embs[len(a_claims):])

    emb_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    emb_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)

    sim = emb_a @ emb_b.T
    a_to_b = float(sim.max(axis=1).mean())
    b_to_a = float(sim.max(axis=0).mean())
    return (a_to_b + b_to_a) / 2.0
