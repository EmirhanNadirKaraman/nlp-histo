"""Embedding-based agreement strategy."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from pipeline.stages.summarization.interfaces.agreement import AgreementStrategy
from pipeline.stages.summarization.models import AuditableSummary
from itertools import combinations

EmbedFn = Callable[[list[str]], list[list[float]]]


def _openai_embed(texts: list[str]) -> list[list[float]]:
    """Default embed function using OpenAI text-embedding-3-small."""
    import openai  # imported here — optional dependency

    response = openai.OpenAI().embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


class EmbeddingAgreement:
    """
    Mean pairwise soft-alignment cosine similarity on claim embeddings.

    For each pair of voters, every claim in voter A is matched to its nearest
    neighbour in voter B (and vice versa); the mean of both directed averages
    gives a symmetric alignment score.

    Parameters
    ----------
    embed_fn:
        Callable that maps a list of strings to embedding vectors.
        Defaults to OpenAI text-embedding-3-small.
    """

    def __init__(self, embed_fn: Optional[EmbedFn] = None) -> None:
        self._embed = embed_fn or _openai_embed

    def compute(self, outputs: list[AuditableSummary]) -> float:
        claim_lists = [self._claims(o) for o in outputs]
        scores = [self._align(a, b) for a, b in combinations(claim_lists, 2)]
        return sum(scores) / len(scores)

    def _align(self, a_claims: list[str], b_claims: list[str]) -> float:
        if not a_claims and not b_claims:
            return 1.0
        if not a_claims or not b_claims:
            return 0.0

        embs = self._embed(a_claims + b_claims)
        emb_a = np.array(embs[: len(a_claims)])
        emb_b = np.array(embs[len(a_claims):])

        emb_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
        emb_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)

        sim = emb_a @ emb_b.T
        a_to_b = float(sim.max(axis=1).mean())
        b_to_a = float(sim.max(axis=0).mean())
        return (a_to_b + b_to_a) / 2.0

    @staticmethod
    def _claims(output: AuditableSummary) -> list[str]:
        return [f.claim for f in output.findings]
