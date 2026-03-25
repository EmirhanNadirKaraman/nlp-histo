"""EmbeddingSimilarityStrategy — batched claim embedding similarity."""
from __future__ import annotations

import numpy as np

from pipeline.stages.summarization.models import AuditableSummary
from .embedding import _align, _claims
from .providers import EmbedFn, OpenAIEmbedder


class EmbeddingSimilarityStrategy:
    """
    Pairwise soft-alignment cosine similarity on claim embeddings.

    Wraps the ``_align()`` helper from ``embedding.py`` for single-pair use,
    and adds ``compute_matrix()`` for batched N-voter evaluation that issues a
    single embedding API call regardless of N.

    Parameters
    ----------
    embed_fn:
        Callable that maps a list of strings to embedding vectors.
        Defaults to OpenAIEmbedder (text-embedding-3-small).
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        max_claims_per_batch: int = 2048,
    ) -> None:
        self._embed: EmbedFn = embed_fn or OpenAIEmbedder()
        self._max_batch = max_claims_per_batch

    def similarity(self, a: AuditableSummary, b: AuditableSummary) -> float:
        """Single-pair similarity (for unit tests and single-call use)."""
        return _align(self._embed, _claims(a), _claims(b))

    def compute_matrix(self, outputs: list[AuditableSummary]) -> np.ndarray:
        """
        Build the full N×N similarity matrix in a single embedding API call.

        All claims from all voters are concatenated, embedded in one request,
        then split back by voter to compute soft-alignment scores.  This avoids
        O(N²) separate API calls that a naive pairwise approach would require.

        Returns
        -------
        np.ndarray
            N×N float matrix where ``M[i, j]`` is the soft-alignment cosine
            similarity between voter i and voter j.  Diagonal entries are 1.0.
        """
        n = len(outputs)
        claim_lists = [_claims(o) for o in outputs]
        all_claims = [c for claims in claim_lists for c in claims]

        if not all_claims:
            return np.eye(n)

        # One API call for every claim across every voter (chunked when > max_batch).
        if len(all_claims) <= self._max_batch:
            raw_embs = np.array(self._embed(all_claims), dtype=float)
        else:
            chunks = [
                all_claims[i : i + self._max_batch]
                for i in range(0, len(all_claims), self._max_batch)
            ]
            raw_embs = np.concatenate(
                [np.array(self._embed(c), dtype=float) for c in chunks]
            )
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        all_embs = raw_embs / norms

        # Split normalised embeddings back by voter.
        voter_embs: list[np.ndarray] = []
        start = 0
        for claims in claim_lists:
            count = len(claims)
            voter_embs.append(all_embs[start : start + count])
            start += count

        # Build symmetric matrix.
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                emb_a, emb_b = voter_embs[i], voter_embs[j]
                if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
                    score = 0.0
                else:
                    sim = emb_a @ emb_b.T
                    a_to_b = float(sim.max(axis=1).mean())
                    b_to_a = float(sim.max(axis=0).mean())
                    score = (a_to_b + b_to_a) / 2.0
                matrix[i, j] = score
                matrix[j, i] = score

        return matrix
