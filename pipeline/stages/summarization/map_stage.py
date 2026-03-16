"""
MAP stage with Agreement-Based Cascading (ABC).

For each chunk of sentences:
  Level 1  — run each voter LLM in parallel (one call per voter).
             Voters should be distinct models / providers so disagreement
             reflects genuine architectural differences, not just sampling
             noise (e.g. gpt-4o-mini + claude-haiku).  When using same-
             provider voters, use temperature diversity and lower theta.
             If pairwise claim-embedding alignment >= theta, accept the
             best result.
  Level 2  — escalate to the smart model (single call) for chunks where
             voters disagree.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .agreement import AgreementChecker, AgreementStrategy
from .cache import PipelineCache
from .models import AuditableSummary
from .prompts import build_map_chain

logger = logging.getLogger(__name__)


def _format_sentences(chunk: list[dict]) -> str:
    """Tag each sentence with a citation ID: [S{i}|{PMCID}|{te_id}]."""
    lines = []
    for i, item in enumerate(chunk, 1):
        pmcid = item.get("pmcid", "UNKNOWN")
        te_id = item.get("text_element_id", 0)
        text = item.get("sentence", "").strip()
        lines.append(f"[S{i}|{pmcid}|{te_id}] {text}")
    return "\n".join(lines)


class MapStage:
    """
    MAP stage: chunks → AuditableSummary list, with ABC cascading.

    Parameters
    ----------
    voter_llms:
        List of LLMs used as Level-1 voters.  Use models from different
        providers (e.g. [ChatOpenAI(...), ChatAnthropic(...)]) so that
        disagreement reflects genuine uncertainty rather than sampling noise.
        Must have at least one entry.
    escalation_llm:
        LLM for Level-2 escalation.  Only called when voters disagree below
        theta.  Typically a larger / more capable model (e.g. gpt-4o).
    theta:
        Agreement threshold in [0, 1].  Higher = more escalations.
    chunk_size:
        Number of sentences per chunk.
    agreement_strategy:
        Strategy used to score voter agreement.  Defaults to
        EmbeddingAgreement (OpenAI text-embedding-3-small).
        Pass CategoryJaccardAgreement() for a fast, API-free alternative.
    """

    def __init__(
        self,
        voter_llms: list,
        escalation_llm,
        theta: float = 0.7,
        chunk_size: int = 10,
        agreement_strategy: Optional[AgreementStrategy] = None,
    ) -> None:
        if not voter_llms:
            raise ValueError("voter_llms must contain at least one LLM.")
        self._voter_chains = [build_map_chain(llm) for llm in voter_llms]
        self._escalation_chain = build_map_chain(escalation_llm)
        self._agreement = AgreementChecker(strategy=agreement_strategy, theta=theta)
        self.chunk_size = chunk_size

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        sentences: list[dict],
        concept_name: str,
        cache: Optional[PipelineCache] = None,
    ) -> list[AuditableSummary]:
        """
        Map all sentences for one concept, returning one AuditableSummary per chunk.

        Chunks that hit the cache are returned immediately.  Remaining chunks
        run through the ABC cascade.
        """
        chunks = self._make_chunks(sentences)
        results: list[Optional[AuditableSummary]] = []
        uncached: list[tuple[int, list[dict]]] = []

        for idx, chunk in enumerate(chunks):
            if cache:
                hit = cache.get_map(chunk)
                if hit:
                    results.append(hit)
                    continue
            results.append(None)
            uncached.append((idx, chunk))

        for idx, chunk in uncached:
            chunk_id = f"C{idx + 1}"
            result = self._cascade(chunk, concept_name, chunk_id)
            results[idx] = result
            if cache:
                cache.set_map(chunk, result)

        return results  # type: ignore[return-value]

    # ── ABC cascade ────────────────────────────────────────────────────────────

    def _cascade(self, chunk: list[dict], concept_name: str, chunk_id: str) -> AuditableSummary:
        inp = {
            "concept_name": concept_name,
            "chunk_id": chunk_id,
            "text": _format_sentences(chunk),
        }

        # Level 1: all voter chains in parallel (one thread per chain)
        voters = self._run_voters(inp)
        agreement = self._agreement.compute(voters)

        if agreement >= self._agreement.theta:
            logger.debug(
                "Chunk %s: Level-1 agreement %.2f >= %.2f — voters accepted (%d providers)",
                chunk_id, agreement, self._agreement.theta, len(voters),
            )
            return self._agreement.best(voters)

        # Level 2: single escalation-model call
        logger.debug(
            "Chunk %s: Level-1 agreement %.2f < %.2f — escalating",
            chunk_id, agreement, self._agreement.theta,
        )
        return self._escalation_chain.invoke(inp)

    def _run_voters(self, inp: dict) -> list[AuditableSummary]:
        """Invoke each voter chain concurrently, return results in chain order."""
        results: list[Optional[AuditableSummary]] = [None] * len(self._voter_chains)
        with ThreadPoolExecutor(max_workers=len(self._voter_chains)) as pool:
            future_to_idx = {
                pool.submit(chain.invoke, inp): i
                for i, chain in enumerate(self._voter_chains)
            }
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
        return results  # type: ignore[return-value]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_chunks(self, sentences: list[dict]) -> list[list[dict]]:
        return [
            sentences[i : i + self.chunk_size]
            for i in range(0, len(sentences), self.chunk_size)
        ]
