"""
MAP stage with Agreement-Based Cascading (ABC).

For each chunk of sentences:
  Level 1  — run each voter LLM in parallel (one call per voter).
             Voters should be distinct models / providers so disagreement
             reflects genuine architectural differences, not just sampling
             noise (e.g. gemini-flash + deepseek + mistral).  When using
             same-provider voters, use temperature diversity and lower theta.
             If pairwise claim-embedding alignment >= theta, accept the
             best result.
  Level 2  — escalate to mid-tier voter LLMs (one call per voter, in
             parallel) for chunks where Level 1 voters disagree.  If these
             agree, accept the best result without calling Level 3.
  Level 3  — escalate to the strong model (single call) for chunks where
             both Level 1 and Level 2 voters disagree.

Note: the router path (when self._router is not None) bypasses Level 2 and
escalates directly from Level 1 to Level 3.
"""
from __future__ import annotations

import logging
import time
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


def _voter_grounding(v: AuditableSummary) -> tuple[float, float]:
    """Fallback grounding signals when no AgreementContext is available."""
    if not v.findings:
        return 1.0, 0.0
    pf = sum(1 for f in v.findings if f.evidence) / len(v.findings)
    me = sum(len(f.evidence) for f in v.findings) / len(v.findings)
    return pf, me


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
        level2_voter_llms: list,
        escalation_llm,
        theta: float = 0.7,
        chunk_size: int = 10,
        scorer: MapOutputScorer | None = None,
        router: MapOutputRouter | None = None,
        routing_collector: RoutingDataset | None = None,
    ) -> None:
        if not voter_llms:
            raise ValueError("voter_llms must contain at least one LLM.")
        if not level2_voter_llms:
            raise ValueError("level2_voter_llms must contain at least one LLM.")
        self._voter_chains = [build_map_chain(llm) for llm in voter_llms]
        self._level2_voter_chains = [build_map_chain(llm) for llm in level2_voter_llms]
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
        collector=None,  # TraceCollector | None
    ) -> list[AuditableSummary]:
        """
        Map all sentences for one paper, returning one AuditableSummary per chunk.

        Chunks that hit the cache are returned immediately.  Remaining chunks
        run through the ABC cascade.
        """
        from .observability.models import ChunkTrace

        chunks = self._make_chunks(sentences)
        results: list[AuditableSummary | None] = []
        uncached: list[tuple[int, list[dict]]] = []
        cache_hit_indices: list[int] = []

        for idx, chunk in enumerate(chunks):
            if cache:
                hit = cache.get_map(chunk)
                if hit:
                    results.append(hit)
                    cache_hit_indices.append(idx)
                    continue
            results.append(None)
            uncached.append((idx, chunk))

        # Record cache-hit chunk traces (lightweight — no voter/agreement data)
        if collector is not None:
            for idx in cache_hit_indices:
                chunk = chunks[idx]
                te_ids = sorted({item.get("text_element_id", 0) for item in chunk})
                text = _format_sentences(chunk)
                collector.add_chunk_trace(ChunkTrace(
                    chunk_id=f"C{idx + 1}",
                    run_id=collector.run_id,
                    pmcid=pmcid,
                    te_ids=te_ids,
                    sentence_count=len(chunk),
                    text_preview=text[:200],
                    cache_hit=True,
                    voters=[],
                    agreement=None,
                    selected_voter_index=None,
                    escalated=False,
                ))

        for idx, chunk in uncached:
            chunk_id = f"C{idx + 1}"
            result = self._cascade(chunk, pmcid, chunk_id, collector=collector)
            results[idx] = result
            if cache and result is not None:
                cache.set_map(chunk, result)

        live_results = [r for r in results if r is not None]

        # Record MAP-stage summary
        if collector is not None:
            n_cache = len(cache_hit_indices)
            n_miss = len(uncached)
            chunk_traces = collector.chunk_traces
            escalations = sum(1 for ct in chunk_traces if ct.escalated and not ct.cache_hit)
            l2_escalations = sum(
                1 for ct in chunk_traces
                if not ct.cache_hit and ct.escalation_level >= 2
            )
            keeps = sum(
                1 for ct in chunk_traces
                if not ct.cache_hit and not ct.escalated
            )
            # rejects = cache-missed chunks that aren't keeps or escalations
            # (edge case: all non-cache chunks are either kept or escalated in practice)
            rejects = n_miss - escalations - keeps
            collector.record_map_stage(
                total_chunks=len(chunks),
                cache_hits=n_cache,
                cache_misses=n_miss,
                escalations=escalations,
                l2_escalations=l2_escalations,
                keeps=keeps,
                rejects=max(rejects, 0),
                total_findings_out=sum(len(r.findings) for r in live_results),
            )

        return live_results  # type: ignore[return-value]

    # ── ABC cascade ────────────────────────────────────────────────────────────

    def _cascade(
        self,
        chunk: list[dict],
        pmcid: str,
        chunk_id: str,
        collector=None,  # TraceCollector | None
    ) -> AuditableSummary | None:
        inp = {
            "pmcid": pmcid,
            "chunk_id": chunk_id,
            "text": _format_sentences(chunk),
        }

        # Level 1: all voter chains in parallel (one thread per chain)
        voters, voter_timings = self._run_voters(inp)

        # Slot to hold the agreement bundle and grounding contexts for trace building
        _trace_bundle = None
        _escalated = False
        _escalation_level = 1  # 1=L1 kept, 2=L2 kept, 3=L3 used
        _voter_contexts = None  # list[VoterContext] | None — from router

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
            _trace_bundle = decision.agreement_details
            _voter_contexts = decision.voter_grounding_contexts

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
                # Router path escalates L1→L3 directly (no L2 step)
                _escalated = True
                _escalation_level = 3
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

        # ── Legacy path (no router): 3-level cascade ───────────────────────
        else:
            # Level 1: cheapest voters
            bundle = self._agreement.compute(voters, source_text=inp["text"])
            _trace_bundle = bundle

            if bundle.decision == ChunkDecision.KEEP:
                logger.debug("Chunk %s: L1 %s → voters accepted", chunk_id, bundle.decision)
                result = self._agreement.best(voters, bundle=bundle)
            else:
                # Level 2: mid-tier voters
                logger.debug("Chunk %s: L1 %s → escalating to L2", chunk_id, bundle.decision)
                l2_voters, l2_timings = self._run_voters(inp, self._level2_voter_chains)
                l2_bundle = self._agreement.compute(l2_voters, source_text=inp["text"])
                _trace_bundle = l2_bundle
                voters = l2_voters
                voter_timings = l2_timings
                _escalation_level = 2

                if l2_bundle.decision == ChunkDecision.KEEP:
                    logger.debug("Chunk %s: L2 %s → voters accepted", chunk_id, l2_bundle.decision)
                    result = self._agreement.best(l2_voters, bundle=l2_bundle)
                else:
                    # Level 3: final escalation model
                    _escalated = True
                    _escalation_level = 3
                    logger.debug("Chunk %s: L2 %s → escalating to L3", chunk_id, l2_bundle.decision)
                    result = self._escalation_chain.invoke(inp)

        # ── Build chunk trace ───────────────────────────────────────────────
        if collector is not None:
            self._record_chunk_trace(
                collector=collector,
                chunk=chunk,
                chunk_id=chunk_id,
                pmcid=pmcid,
                text=inp["text"],
                voters=voters,
                voter_timings=voter_timings,
                bundle=_trace_bundle,
                escalated=_escalated,
                voter_contexts=_voter_contexts,
                escalation_level=_escalation_level,
            )

        return result

    def _record_chunk_trace(
        self,
        collector,
        chunk: list[dict],
        chunk_id: str,
        pmcid: str,
        text: str,
        voters: list[AuditableSummary],
        voter_timings: dict[int, float | None],
        bundle,  # ScoreBundle | None
        escalated: bool,
        voter_contexts=None,  # list[VoterContext] | None — from router
        escalation_level: int = 1,
    ) -> None:
        from .observability.models import (
            AgreementTrace,
            ChunkTrace,
            PairwiseScore,
            VoterTrace,
        )

        # Voter traces — use validated router grounding when available, else fallback.
        voter_traces = []
        grounding_source = "validated" if voter_contexts is not None else "fallback"
        for i, v in enumerate(voters):
            if voter_contexts is not None and i < len(voter_contexts):
                vc = voter_contexts[i]
                pf = vc.grounding_pass_fraction
                me = vc.mean_evidence_length
            else:
                pf, me = _voter_grounding(v)
            voter_traces.append(VoterTrace(
                voter_index=i,
                finding_count=len(v.findings),
                grounding_pass_fraction=round(pf, 4),
                mean_evidence_length=round(me, 4),
                latency_ms=round(voter_timings.get(i), 1) if voter_timings.get(i) is not None else None,
                grounding_source=grounding_source,
            ))

        # Agreement trace — built from ScoreBundle.score_details when available
        agreement_trace = None
        selected_voter_index: int | None = None

        if bundle is not None:
            sd = bundle.score_details or {}
            eligible = sd.get("eligible_voter_indices", list(range(len(voters))))
            avg_sim_raw = sd.get("avg_sim", [])
            # Build PairwiseScore with breakdown fields when present in score_details.
            _BREAKDOWN_KEYS = (
                "claim_count_a", "claim_count_b",
                "coverage_a_to_b", "coverage_b_to_a", "base",
                "count_factor", "reuse_factor",
                "polarity_contradiction_ratio", "numeric_contradiction_ratio",
                "contradiction_ratio", "contradiction_factor",
                "pre_grounding_score", "grounding_factor",
            )
            pairwise = [
                PairwiseScore(
                    voter_i=p["voter_i"],
                    voter_j=p["voter_j"],
                    score=round(p["score"], 6),
                    **{k: p[k] for k in _BREAKDOWN_KEYS if k in p},
                )
                for p in sd.get("pairwise_upper", [])
            ]

            score = bundle.confidence if bundle.confidence is not None else (
                bundle.embedding_agreement or 0.0
            )
            decision_val = bundle.decision.value if bundle.decision else "unknown"

            # Human-readable reason
            theta = self._agreement.theta
            reject_theta = self._agreement.reject_theta
            if not escalated:
                reason = (
                    f"Deferral score {score:.3f} ≥ theta {theta:.2f}; "
                    f"voter {bundle.best_index} selected"
                )
                selected_voter_index = bundle.best_index
            elif score <= reject_theta and score > 0.0:
                reason = (
                    f"Deferral score {score:.3f} ≤ reject_theta {reject_theta:.2f} — hard reject"
                )
            elif len(eligible) < 2:
                reason = (
                    f"Only {len(eligible)} non-empty voter(s) eligible — too few for agreement"
                )
            else:
                reason = (
                    f"Deferral score {score:.3f} in ({reject_theta:.2f}, {theta:.2f}) — escalated"
                )

            agreement_trace = AgreementTrace(
                eligible_voter_indices=eligible,
                avg_sim=[round(s, 4) for s in avg_sim_raw],
                pairwise_scores=pairwise,
                deferral_score=round(score, 4),
                theta=theta,
                reject_theta=reject_theta,
                decision=decision_val,
                reason=reason,
                selected_voter_index=selected_voter_index,
            )

        te_ids = sorted({item.get("text_element_id", 0) for item in chunk})
        collector.add_chunk_trace(ChunkTrace(
            chunk_id=chunk_id,
            run_id=collector.run_id,
            pmcid=pmcid,
            te_ids=te_ids,
            sentence_count=len(chunk),
            text_preview=text[:200],
            cache_hit=False,
            voters=voter_traces,
            agreement=agreement_trace,
            selected_voter_index=selected_voter_index,
            escalated=escalated,
            escalation_level=escalation_level,
        ))

    def _run_voters(
        self, inp: dict, chains: list | None = None
    ) -> tuple[list[AuditableSummary], dict[int, float | None]]:
        """Invoke each voter chain concurrently; return results and per-voter latency.

        A single voter chain exception is logged and that voter is excluded
        from the result list.  The router handles the reduced count correctly
        (N_eligible < 2 → ESCALATE or REJECT as appropriate).

        Parameters
        ----------
        chains:
            Voter chains to run.  Defaults to self._voter_chains (Level 1).
            Pass self._level2_voter_chains to run Level 2 voters.
        """
        target = chains if chains is not None else self._voter_chains
        results: list[AuditableSummary | None] = [None] * len(target)
        timings: dict[int, float | None] = {}

        def _timed_invoke(chain, i: int):
            t0 = time.monotonic()
            out = chain.invoke(inp)
            timings[i] = (time.monotonic() - t0) * 1000.0
            return out

        with ThreadPoolExecutor(max_workers=len(target)) as pool:
            future_to_idx = {
                pool.submit(_timed_invoke, chain, i): i
                for i, chain in enumerate(target)
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
                    timings[idx] = None
        return [r for r in results if r is not None], timings  # type: ignore[return-value]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_chunks(self, sentences: list[dict]) -> list[list[dict]]:
        return [
            sentences[i : i + self.chunk_size]
            for i in range(0, len(sentences), self.chunk_size)
        ]
