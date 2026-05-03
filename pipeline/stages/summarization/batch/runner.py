"""
BatchSummarizationRunner — asynchronous batch variant of SummarizationRunner.

Workflow
--------
Each call to ``advance()`` checks whether the current batch phase is done
and, if so, runs agreement scoring and submits the next phase.
The handle is persisted to disk between calls so the process can exit freely.

    runner = BatchSummarizationRunner(l1_voters=[...], ...)

    handle = runner.submit(file_data)       # L1 jobs submitted; handle saved
    handle = runner.advance(handle)         # call again until COMPLETE
    result = runner.finalize(handle)        # REDUCE + RULES (sync)

Typical script usage::

    handle = runner.load_or_submit(file_data)
    while handle.phase != BatchPhase.COMPLETE:
        handle = runner.advance(handle)
        if handle.phase != BatchPhase.COMPLETE:
            print("Jobs still running — try again later.")
            break
    else:
        result = runner.finalize(handle)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..agreement import AgreementChecker, EmbeddingScorer
from ..config import SummarizationConfig
from ..helpers.contradiction_detector import ContradictionDetector
from ..helpers.grounding_filter import GroundingFilter
from ..interfaces.scoring import ChunkDecision
from ..current_stages.map_stage import _format_sentences
from ..models import AuditableSummary
from ..old_stages.reduce_stage import ReduceStage
from ..old_stages.rule_stage import RuleStage
from .dispatch import OPENAI_MAP_TOOL, build_providers, build_requests, parse_result, submit_level
from .models import BatchHandle, BatchPhase, BatchResult, ProviderJob, VoterBatchConfig

logger = logging.getLogger(__name__)


class BatchSummarizationRunner:
    """
    Asynchronous batch pipeline runner with 50 % cost discounts.

    Parameters
    ----------
    l1_voters, l2_voters:
        Ordered list of VoterBatchConfig for each level.  Provider must be
        "azure", "claude", or "vertex_gemini".
    l3_model:
        Single VoterBatchConfig for the Level-3 escalation model.
    escalation_llm:
        Synchronous LangChain LLM for REDUCE and RULE EXTRACTION (called once
        per paper at the end, after all MAP batches complete).
    config:
        All numeric/boolean pipeline knobs.  Defaults to SummarizationConfig().
        Only map and grounding sub-configs are used by the batch runner.
    output_dir:
        Directory for final ``{pmcid}.json`` result files.
    handle_dir:
        Directory for ``{pmcid}.batch.json`` state files.
        Defaults to ``output_dir / "batch_handles"``.
    """

    def __init__(
        self,
        l1_voters: list[VoterBatchConfig],
        l2_voters: list[VoterBatchConfig],
        l3_model: VoterBatchConfig,
        escalation_llm,
        config: SummarizationConfig | None = None,
        output_dir: Path = Path("out/summaries"),
        handle_dir: Path | None = None,
    ) -> None:
        cfg = config or SummarizationConfig()
        self._l1 = l1_voters
        self._l2 = l2_voters
        self._l3 = l3_model
        self._chunk_size = cfg.map.chunk_size
        self._agreement = AgreementChecker(
            scorer=EmbeddingScorer(),
            theta=cfg.map.theta,
            reject_theta=cfg.map.reject_theta,
        )
        self._reduce = ReduceStage(escalation_llm)
        self._rules = RuleStage(escalation_llm)
        self._grounding = (
            GroundingFilter(cfg.grounding.threshold)
            if cfg.grounding.threshold is not None else None
        )
        self._contradiction = (
            ContradictionDetector(
                escalation_llm,
                similarity_threshold=cfg.contradiction_similarity_threshold,
            )
            if cfg.contradiction_similarity_threshold is not None
            else None
        )
        self._output_dir = output_dir
        self._summaries_dir = output_dir / "summaries"
        self._summaries_dir.mkdir(parents=True, exist_ok=True)
        self._handle_dir = handle_dir or (output_dir / "batch_handles")
        self._handle_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────────

    def handle_path(self, pmcid: str) -> Path:
        return self._handle_dir / f"{pmcid}.batch.json"

    def load_or_submit(self, file_data: dict) -> BatchHandle:
        """Load existing handle from disk, or submit a new L1 batch."""
        path = self.handle_path(file_data["pmcid"])
        if path.exists():
            handle = BatchHandle.load(path)
            logger.info("[%s] Loaded existing handle (phase=%s)", handle.pmcid, handle.phase.value)
            return handle
        return self.submit(file_data)

    def submit(self, file_data: dict) -> BatchHandle:
        """Chunk the paper, submit L1 batch jobs, and persist the handle."""
        pmcid = file_data["pmcid"]
        sentences = file_data["sentences_with_provenance"]
        chunk_map = self._make_chunk_map(sentences)

        handle = BatchHandle(
            pmcid=pmcid,
            phase=BatchPhase.L1_SUBMITTED,
            sentences=sentences,
            chunk_map=chunk_map,
            l1_strip=[cfg.strip_thinking for cfg in self._l1],
            l2_strip=[cfg.strip_thinking for cfg in self._l2],
            l3_strip=self._l3.strip_thinking,
        )
        handle.jobs = submit_level(chunk_map, pmcid, self._l1, level="l1")
        handle.save(self.handle_path(pmcid))

        total_reqs = sum(j.request_count for j in handle.jobs)
        logger.info(
            "[%s] L1 batch submitted: %d chunks × %d voters = %d requests across %d job(s)",
            pmcid, len(chunk_map), len(self._l1), total_reqs, len(handle.jobs),
        )
        return handle

    def advance(self, handle: BatchHandle) -> BatchHandle:
        """
        Refresh job statuses.  If all jobs in the current phase are done,
        collect results, run agreement, and submit the next phase (or mark COMPLETE).

        Always persists the updated handle to disk.
        """
        if handle.phase == BatchPhase.COMPLETE:
            return handle

        providers = build_providers({j.provider for j in handle.jobs})

        # Refresh statuses
        for job in handle.jobs:
            p = providers.get(job.provider)
            if p and job.status not in ("completed", "failed"):
                p.check(job)

        pending = [j for j in handle.jobs if j.status not in ("completed", "failed")]
        if pending:
            logger.info(
                "[%s] %s: %d/%d job(s) still running",
                handle.pmcid, handle.phase.value, len(pending), len(handle.jobs),
            )
            handle.save(self.handle_path(handle.pmcid))
            return handle

        # Collect all results from completed jobs
        raw_results: list[BatchResult] = []
        for job in handle.jobs:
            if job.status == "failed":
                logger.warning("[%s] Job %s failed — skipping its results", handle.pmcid, job.job_id)
                continue
            p = providers[job.provider]
            raw_results.extend(p.retrieve(job))

        if handle.phase == BatchPhase.L1_SUBMITTED:
            self._process_level(handle, raw_results, level="l1", strip_flags=handle.l1_strip,
                                next_voters=self._l2, next_level="l2",
                                next_strip=handle.l2_strip, next_phase=BatchPhase.L2_SUBMITTED,
                                escalated_attr="l2_chunk_ids")
        elif handle.phase == BatchPhase.L2_SUBMITTED:
            self._process_level(handle, raw_results, level="l2", strip_flags=handle.l2_strip,
                                next_voters=[self._l3], next_level="l3",
                                next_strip=[handle.l3_strip], next_phase=BatchPhase.L3_SUBMITTED,
                                escalated_attr="l3_chunk_ids",
                                chunk_ids_to_process=handle.l2_chunk_ids)
        elif handle.phase == BatchPhase.L3_SUBMITTED:
            self._collect_l3(handle, raw_results)

        handle.save(self.handle_path(handle.pmcid))
        return handle

    def finalize(self, handle: BatchHandle) -> dict:
        """
        Run REDUCE → RULES synchronously on the batch MAP results.
        Only call when ``handle.phase == BatchPhase.COMPLETE``.
        """
        if handle.phase != BatchPhase.COMPLETE:
            raise RuntimeError(
                f"Cannot finalize: handle phase is {handle.phase.value!r}, not 'complete'."
            )

        pmcid = handle.pmcid
        chunk_summaries = [
            AuditableSummary.model_validate(v) for v in handle.finalized.values()
        ]

        if self._grounding is not None:
            chunk_summaries = [self._grounding.filter_findings(cs) for cs in chunk_summaries]

        logger.info("[%s] REDUCE — %d chunks", pmcid, len(chunk_summaries))
        master = self._reduce.reduce(chunk_summaries, pmcid)

        logger.info("[%s] RULES", pmcid)
        rules = self._rules.extract(master, pmcid)

        if self._grounding is not None:
            rules = self._grounding.filter_rules(rules)

        contradiction_report = None
        if self._contradiction is not None:
            contradiction_report = self._contradiction.detect(rules)

        result = {
            "status": "success",
            "pmcid": pmcid,
            "summary": master.narrative_summary,
            "rules": [r.model_dump() for r in rules.rules],
            "contradiction_report": contradiction_report.model_dump() if contradiction_report else None,
            "audit_trail": {
                "map_chunks": [cs.model_dump() for cs in chunk_summaries],
                "master_summary": master.model_dump(),
                "rules_provenance": rules.model_dump(),
            },
        }
        out_path = self._summaries_dir / f"{pmcid}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[%s] Result saved to %s", pmcid, out_path)
        return result

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _make_chunk_map(self, sentences: list[dict]) -> dict[str, list[dict]]:
        chunk_map: dict[str, list[dict]] = {}
        for i in range(0, len(sentences), self._chunk_size):
            chunk_id = f"C{i // self._chunk_size + 1}"
            chunk_map[chunk_id] = sentences[i : i + self._chunk_size]
        return chunk_map

    def _process_level(
        self,
        handle: BatchHandle,
        raw_results: list[BatchResult],
        level: str,
        strip_flags: list[bool],
        next_voters: list[VoterBatchConfig],
        next_level: str,
        next_strip: list[bool],
        next_phase: BatchPhase,
        escalated_attr: str,
        chunk_ids_to_process: list[str] | None = None,
    ) -> None:
        """
        Parse batch results for one level, run agreement, and either:
        - Finalise kept chunks
        - Submit the next level for escalated chunks
        - Mark COMPLETE if nothing needs escalation
        """
        pmcid = handle.pmcid
        targets = chunk_ids_to_process or list(handle.chunk_map.keys())

        # Group raw results by chunk_id
        by_chunk: dict[str, list[BatchResult]] = {}
        for res in raw_results:
            parts = res.custom_id.split("__")
            chunk_id = parts[1] if len(parts) > 1 else "unknown"
            by_chunk.setdefault(chunk_id, []).append(res)

        # Store raw content for debugging
        raw_store = getattr(handle, f"{level}_raw")
        for res in raw_results:
            if res.content:
                raw_store[res.custom_id] = res.content

        escalated: list[str] = []

        for chunk_id in targets:
            results = by_chunk.get(chunk_id, [])
            results_sorted = sorted(results, key=lambda r: r.custom_id)

            voters: list[AuditableSummary] = []
            for res in results_sorted:
                vi_part = res.custom_id.split("__")[-1]
                vi = int(vi_part) if vi_part.isdigit() else 0
                strip = strip_flags[vi] if vi < len(strip_flags) else False
                parsed = parse_result(res, strip_thinking=strip)
                if parsed is not None:
                    voters.append(parsed)

            if not voters:
                logger.warning("[%s] Chunk %s: all %s voters failed — escalating", pmcid, chunk_id, level)
                escalated.append(chunk_id)
                continue

            source_text = _format_sentences(handle.chunk_map[chunk_id])
            bundle = self._agreement.compute(voters, source_text=source_text)

            if bundle.decision == ChunkDecision.KEEP:
                best = self._agreement.best(voters, bundle=bundle)
                handle.finalized[chunk_id] = best.model_dump()
            else:
                escalated.append(chunk_id)

        kept = len(targets) - len(escalated)
        logger.info(
            "[%s] %s complete: %d kept, %d escalated to %s",
            pmcid, level.upper(), kept, len(escalated), next_level.upper(),
        )

        if escalated:
            setattr(handle, escalated_attr, escalated)
            handle.phase = next_phase
            handle.jobs = submit_level(
                handle.chunk_map, pmcid, next_voters,
                level=next_level, chunk_ids=escalated,
            )
        else:
            handle.phase = BatchPhase.COMPLETE
            logger.info("[%s] No escalations — batch MAP complete.", pmcid)

    def _collect_l3(self, handle: BatchHandle, raw_results: list[BatchResult]) -> None:
        """Parse L3 (escalation model) results and finalise remaining chunks."""
        pmcid = handle.pmcid
        targets = handle.l3_chunk_ids

        by_chunk: dict[str, list[BatchResult]] = {}
        for res in raw_results:
            parts = res.custom_id.split("__")
            chunk_id = parts[1] if len(parts) > 1 else "unknown"
            by_chunk.setdefault(chunk_id, []).append(res)

        for res in raw_results:
            if res.content:
                handle.l3_raw[res.custom_id] = res.content

        for chunk_id in targets:
            results = by_chunk.get(chunk_id, [])
            if not results:
                logger.warning("[%s] Chunk %s: no L3 result — skipping", pmcid, chunk_id)
                continue
            parsed = parse_result(results[0], strip_thinking=handle.l3_strip)
            if parsed is not None:
                handle.finalized[chunk_id] = parsed.model_dump()
            else:
                logger.warning("[%s] Chunk %s: L3 parse failed — skipping", pmcid, chunk_id)

        logger.info(
            "[%s] L3 complete. Total finalised: %d/%d chunks.",
            pmcid, len(handle.finalized), len(handle.chunk_map),
        )
        handle.phase = BatchPhase.COMPLETE
