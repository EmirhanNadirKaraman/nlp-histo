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
from .agreement import AgreementChecker, EmbeddingScorer, MapOutputScorer
from .cache import PipelineCache
from .models import AuditableSummary
from .prompts import build_map_chain
from .routing import MapOutputRouter
from .routing.routing_dataset import RoutingDataset, RoutingRecord
from pipeline.stages.summarization.interfaces.scoring import ChunkDecision

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
    scorer:
        MapOutputScorer used to score voter agreement.  Defaults to
        EmbeddingScorer.  Pass CascadedCompositeScorer for embedding +
        LLM judge cascade, or CategoryJaccardScorer for a fast, API-free
        alternative.
    """

    def __init__(
        self,
        voter_llms: list,
        escalation_llm,
        theta: float = 0.7,
        chunk_size: int = 10,
        scorer: MapOutputScorer | None = None,
        router: MapOutputRouter | None = None,
        routing_collector: RoutingDataset | None = None,
    ) -> None:
        if not voter_llms:
            raise ValueError("voter_llms must contain at least one LLM.")
        self._voter_chains = [build_map_chain(llm) for llm in voter_llms]
        self._escalation_chain = build_map_chain(escalation_llm)
        self._agreement = AgreementChecker(scorer=scorer or EmbeddingScorer(), theta=theta)
        self._router = router
        self._routing_collector = routing_collector
        self.chunk_size = chunk_size

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        sentences: list[dict],
        pmcid: str,
        cache: PipelineCache | None = None,
    ) -> list[AuditableSummary]:
        """
        Map all sentences for one paper, returning one AuditableSummary per chunk.

        Chunks that hit the cache are returned immediately.  Remaining chunks
        run through the ABC cascade.
        """
        chunks = self._make_chunks(sentences)
        results: list[AuditableSummary | None] = []
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
            result = self._cascade(chunk, pmcid, chunk_id)
            results[idx] = result
            if cache and result is not None:
                cache.set_map(chunk, result)

        return [r for r in results if r is not None]  # type: ignore[return-value]

    # ── ABC cascade ────────────────────────────────────────────────────────────

    def _cascade(
        self,
        chunk: list[dict],
        pmcid: str,
        chunk_id: str,
    ) -> AuditableSummary | None:
        inp = {
            "pmcid": pmcid,
            "chunk_id": chunk_id,
            "text": _format_sentences(chunk),
        }

        # Level 1: all voter chains in parallel (one thread per chain)
        voters = self._run_voters(inp)

        # ── Grounding-first router path ─────────────────────────────────────
        if self._router is not None:
            decision = self._router.route(
                voters,
                chunk=chunk,
                pmcid=pmcid,
                source_text=inp["text"],
            )
            logger.debug(
                "Chunk %s: router → %s (gate=%s reasons=%s)",
                chunk_id,
                decision.decision.value,
                decision.gate_origin.value,
                [r.value for r in decision.reason_codes],
            )
            if decision.decision == ChunkDecision.KEEP:
                valid = (
                    [voters[i] for i in decision.valid_voter_indices]
                    if decision.valid_voter_indices
                    else voters
                )
                result = self._agreement.best(valid, bundle=decision.agreement_details)
                best_eligible: AuditableSummary | None = result
                best_eligible_idx: int | None = (
                    decision.valid_voter_indices[decision.agreement_details.best_index]
                    if decision.valid_voter_indices
                    and decision.agreement_details
                    and decision.agreement_details.best_index is not None
                    and decision.agreement_details.best_index
                    < len(decision.valid_voter_indices)
                    else None
                )
            else:
                if decision.decision == ChunkDecision.REJECT:
                    logger.info(
                        "Chunk %s rejected by router — escalating to strong model.",
                        chunk_id,
                    )
                result = self._escalation_chain.invoke(inp)
                if decision.valid_voter_indices:
                    valid = [voters[i] for i in decision.valid_voter_indices]
                    best_eligible = self._agreement.best(
                        valid, bundle=decision.agreement_details
                    )
                    best_eligible_idx = (
                        decision.valid_voter_indices[
                            decision.agreement_details.best_index
                        ]
                        if decision.agreement_details
                        and decision.agreement_details.best_index is not None
                        and decision.agreement_details.best_index
                        < len(decision.valid_voter_indices)
                        else decision.valid_voter_indices[0]
                    )
                else:
                    best_eligible = None
                    best_eligible_idx = None

            if self._routing_collector is not None and result is not None:
                self._routing_collector.append(
                    RoutingRecord(
                        pmcid=pmcid,
                        chunk_id=chunk_id,
                        chunk_text=inp["text"],
                        n_voters=len(voters),
                        n_eligible=len(decision.valid_voter_indices or []),
                        gate_origin=decision.gate_origin.value,
                        decision=decision.decision.value,
                        reason_codes=[r.value for r in decision.reason_codes],
                        escalated=(decision.decision != ChunkDecision.KEEP),
                        used_agreement_gate=(
                            decision.gate_origin.value == "agreement_gate"
                        ),
                        deferral_score=(
                            decision.agreement_details.confidence
                            if decision.agreement_details
                            else None
                        ),
                        agreement_scorer=type(self._agreement._scorer).__name__,
                        output=result.model_dump(),
                        best_eligible_output=(
                            best_eligible.model_dump()
                            if best_eligible is not None
                            else None
                        ),
                        best_eligible_exists=(best_eligible is not None),
                        best_eligible_voter_index=best_eligible_idx,
                    )
                )
            return result

        # ── Legacy path (no router) ─────────────────────────────────────────
        bundle = self._agreement.compute(voters, source_text=inp["text"])

        if bundle.decision == ChunkDecision.KEEP:
            logger.debug("Chunk %s: %s → voters accepted", chunk_id, bundle.decision)
            return self._agreement.best(voters)

        # Fix: REJECT was dead code in the legacy path — both REJECT and
        # ESCALATE send to the escalation model.
        logger.debug("Chunk %s: %s → escalating", chunk_id, bundle.decision)
        return self._escalation_chain.invoke(inp)

    def _run_voters(self, inp: dict) -> list[AuditableSummary]:
        """Invoke each voter chain concurrently, return results in chain order.

        A single voter chain exception is logged and that voter is excluded
        from the result list.  The router handles the reduced count correctly
        (N_eligible < 2 → ESCALATE or REJECT as appropriate).
        """
        results: list[AuditableSummary | None] = [None] * len(self._voter_chains)
        with ThreadPoolExecutor(max_workers=len(self._voter_chains)) as pool:
            future_to_idx = {
                pool.submit(chain.invoke, inp): i
                for i, chain in enumerate(self._voter_chains)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.warning(
                        "Voter %d failed for chunk %s: %s — excluded from agreement",
                        idx,
                        inp.get("chunk_id", "?"),
                        exc,
                    )
        return [r for r in results if r is not None]  # type: ignore[return-value]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_chunks(self, sentences: list[dict]) -> list[list[dict]]:
        return [
            sentences[i : i + self.chunk_size]
            for i in range(0, len(sentences), self.chunk_size)
        ]
