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

from datetime import datetime, timezone

from ..agreement import AgreementChecker, EmbeddingScorer
from ..config import SummarizationConfig
from ..current_stages.canonicalize_stage import CanonicalizeStage
from ..current_stages.group_stage import GroupStage, is_groupable
from ..current_stages.normalize_stage import NormalizeStage
from ..current_stages.relate_stage import RelateStage
from ..current_stages.resolve_stage import ResolveStage
from ..helpers.contradiction_detector import ContradictionDetector
from ..helpers.grounding_filter import GroundingFilter
from ..interfaces.scoring import ChunkDecision
from ..current_stages.map_stage import _format_sentences
from ..models import (
    AuditableSummary,
    MAP_PROMPT_VERSION,
    MAP_SCHEMA_VERSION,
    MAP_STAGE_NAME,
    NormalFinding,
    compute_cascade_signature,
)
from ..old_stages.reduce_stage import ReduceStage
from ..old_stages.rule_stage import RuleStage
from ..persistence import (
    RunArtifactWriter,
    RunManifest,
    persist_canonicalize_artifacts,
    persist_group_artifacts,
    persist_map_artifacts,
    persist_normalize_artifacts,
    persist_relate_artifacts,
    persist_resolve_artifacts,
)
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
        embed_fn=None,
        cascade_profile: str = "custom",
        artifact_root: Path | None = None,
        artifact_run_id: str | None = None,
        run_modern_pipeline: bool = True,
        run_reduce: bool = False,
    ) -> None:
        from ..agreement.providers import OpenAIEmbedder
        cfg = config or SummarizationConfig()
        self._l1 = l1_voters
        self._l2 = l2_voters
        self._l3 = l3_model
        self._chunk_size = cfg.map.chunk_size
        self._embed_fn = embed_fn or OpenAIEmbedder()
        self._agreement = AgreementChecker(
            scorer=EmbeddingScorer(self._embed_fn),
            theta=cfg.map.theta,
            reject_theta=cfg.map.reject_theta,
        )
        # ReduceStage/RuleStage call `llm.with_structured_output(...)` in their
        # ctors, so we only construct them when the legacy REDUCE/RULES block
        # is enabled. Otherwise the runner can be built with any (or no) LLM —
        # convenient for tests.
        self._escalation_llm = escalation_llm
        self._reduce: ReduceStage | None = (
            ReduceStage(escalation_llm) if run_reduce else None
        )
        self._rules: RuleStage | None = (
            RuleStage(escalation_llm) if run_reduce else None
        )
        self._grounding = (
            GroundingFilter(cfg.grounding.threshold)
            if cfg.grounding.threshold is not None else None
        )
        # ContradictionDetector is only useful with REDUCE/RULES output, so it's
        # also lazy. Avoids needing a structured-output-capable LLM in tests.
        self._contradiction = (
            ContradictionDetector(
                escalation_llm,
                similarity_threshold=cfg.contradiction_similarity_threshold,
            )
            if (run_reduce and cfg.contradiction_similarity_threshold is not None)
            else None
        )
        self._output_dir = output_dir
        self._summaries_dir = output_dir / "summaries"
        self._summaries_dir.mkdir(parents=True, exist_ok=True)
        self._handle_dir = handle_dir or (output_dir / "batch_handles")
        self._handle_dir.mkdir(parents=True, exist_ok=True)

        # Run-artifact provenance — kept on the runner so submit() and
        # finalize() emit consistent metadata regardless of who calls them.
        self._cascade_profile = cascade_profile
        self._cascade_signature = compute_cascade_signature(
            [(v.provider, v.model) for v in self._l1]
            + [(v.provider, v.model) for v in self._l2]
            + [(self._l3.provider, self._l3.model)]
        )

        # Modern post-MAP chain. Stateless stages — instantiate once.
        # When ``run_modern_pipeline`` is False, finalize() falls back to the
        # legacy REDUCE/RULES-only behaviour.
        self._cfg_full = cfg
        self._run_modern_pipeline = run_modern_pipeline
        self._run_reduce = run_reduce
        self._normalize = NormalizeStage()
        self._group = GroupStage()
        self._canonicalize = CanonicalizeStage()
        self._relate = RelateStage(
            entailment_threshold=cfg.relate.entailment_threshold,
            contradiction_threshold=cfg.relate.contradiction_threshold,
        )
        self._resolve = ResolveStage(cfg.resolve)

        # Filesystem artifact persistence — opt-in (mirror of sync runner).
        self._artifact_root: Path | None = (
            Path(artifact_root) if artifact_root is not None else None
        )
        self._artifact_run_id_override: str | None = artifact_run_id

        # Snapshot of config + cascade for the manifest, mirrors sync runner shape.
        import dataclasses
        self._config_snapshot = {
            **dataclasses.asdict(cfg),
            "cascade_profile":   cascade_profile,
            "cascade_signature": self._cascade_signature,
            "voter_models":         [v.model for v in self._l1],
            "level2_voter_models":  [v.model for v in self._l2],
            "escalation_model":     self._l3.model,
        }

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
            schema_version=MAP_SCHEMA_VERSION,
            prompt_version=MAP_PROMPT_VERSION,
            stage_name=MAP_STAGE_NAME,
            cascade_profile=self._cascade_profile,
            cascade_signature=self._cascade_signature,
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
        Run the modern post-MAP chain (and optionally REDUCE → RULES) on the
        batch MAP output, persist filesystem artifacts when configured, and
        return the assembled result dict.

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

        # ── Filesystem persistence setup (opt-in) ─────────────────────────────
        run_id = self._artifact_run_id_override or self._make_run_id(pmcid)
        writer = self._make_artifact_writer(run_id, pmcid, handle.cascade_signature)

        try:
            # ── Grounding (matches sync ordering) ─────────────────────────────
            grounding_rejected: list = []
            if self._grounding is not None:
                new_summaries = []
                for cs in chunk_summaries:
                    kept_cs, dropped = self._grounding.filter_findings_with_scores(cs)
                    new_summaries.append(kept_cs)
                    for f in dropped:
                        grounding_rejected.append((cs.chunk_id, f))
                chunk_summaries = new_summaries

            # ── MAP artifact persistence ──────────────────────────────────────
            persist_map_artifacts(writer, pmcid, chunk_summaries, grounding_rejected)

            # ── Optional modern chain — produces canonical/relate/final ──────
            normal_findings: list = []
            finding_groups:  list = []
            canonical_rules: list = []
            relations:       list = []
            relate_raw_pairs: list = []
            final_rules:     list = []

            if self._run_modern_pipeline:
                import time as _time  # noqa: PLC0415
                all_findings = [f for cs in chunk_summaries for f in cs.findings]
                logger.info("[%s] NORMALIZE — start (%d findings)", pmcid, len(all_findings))
                t0 = _time.perf_counter()
                normal_findings = self._normalize.normalize(all_findings, pmcid)
                logger.info("[%s] NORMALIZE — done [%.1fs] → %d normal findings",
                            pmcid, _time.perf_counter() - t0, len(normal_findings))
                persist_normalize_artifacts(writer, pmcid, normal_findings)

                groupable     = [nf for nf in normal_findings if is_groupable(nf)]
                non_groupable = [nf for nf in normal_findings if not is_groupable(nf)]
                logger.info("[%s] GROUP — start (%d groupable, %d non-groupable)",
                            pmcid, len(groupable), len(non_groupable))
                t0 = _time.perf_counter()
                finding_groups = self._group.group(groupable, pmcid)
                logger.info("[%s] GROUP — done [%.1fs] → %d groups",
                            pmcid, _time.perf_counter() - t0, len(finding_groups))
                persist_group_artifacts(writer, pmcid, finding_groups, non_groupable)

                logger.info("[%s] CANONICALIZE — start (%d groups)", pmcid, len(finding_groups))
                t0 = _time.perf_counter()
                nf_by_id: dict[str, NormalFinding] = {nf.normal_id: nf for nf in normal_findings}
                canonical_rules = self._canonicalize.canonicalize(
                    finding_groups, nf_by_id, pmcid
                )
                logger.info("[%s] CANONICALIZE — done [%.1fs] → %d canonical rules",
                            pmcid, _time.perf_counter() - t0, len(canonical_rules))
                # UMLS enrichment is a no-op when scispacy is unavailable but
                # the import itself can take 5-30s on first use.
                logger.info("[%s] UMLS enrichment — start (loading scispacy may be slow)", pmcid)
                t0 = _time.perf_counter()
                from ..helpers.entity_linker import enrich_rules_with_cuis  # noqa: PLC0415
                enrich_rules_with_cuis(canonical_rules)
                logger.info("[%s] UMLS enrichment — done [%.1fs]",
                            pmcid, _time.perf_counter() - t0)
                persist_canonicalize_artifacts(writer, pmcid, canonical_rules)

                logger.info("[%s] RELATE — start (%d canonical rules)",
                            pmcid, len(canonical_rules))
                t0 = _time.perf_counter()
                if len(canonical_rules) < 2:
                    logger.info("[%s] RELATE — skipped (need ≥2 rules to compare)", pmcid)
                relations, relate_raw_pairs = self._relate.relate(canonical_rules, pmcid)
                logger.info("[%s] RELATE — done [%.1fs] → %d relations, %d raw pairs",
                            pmcid, _time.perf_counter() - t0,
                            len(relations), len(relate_raw_pairs))
                persist_relate_artifacts(writer, pmcid, relations, relate_raw_pairs)

                logger.info("[%s] RESOLVE — start (%d canonical rules, %d relations)",
                            pmcid, len(canonical_rules), len(relations))
                t0 = _time.perf_counter()
                final_rules = self._resolve.resolve(canonical_rules, relations, pmcid)
                logger.info("[%s] RESOLVE — done [%.1fs] → %d final rules",
                            pmcid, _time.perf_counter() - t0, len(final_rules))
                persist_resolve_artifacts(writer, pmcid, final_rules, relations)

            # ── Optional REDUCE → RULES (legacy) ─────────────────────────────
            master = None
            rules  = None
            contradiction_report = None
            if self._run_reduce:
                if self._reduce is None or self._rules is None:
                    raise RuntimeError(
                        "run_reduce=True but ReduceStage/RuleStage were not built. "
                        "Reconstruct the runner with run_reduce=True."
                    )
                logger.info("[%s] REDUCE — %d chunks", pmcid, len(chunk_summaries))
                master = self._reduce.reduce(chunk_summaries, pmcid)
                logger.info("[%s] RULES", pmcid)
                rules = self._rules.extract(master, pmcid)
                if self._grounding is not None:
                    rules = self._grounding.filter_rules(rules)
                if self._contradiction is not None:
                    contradiction_report = self._contradiction.detect(rules)

            result = {
                "status": "success",
                "run_id": run_id,
                "pmcid":  pmcid,
                "summary": master.narrative_summary if master else None,
                "rules":   [r.model_dump() for r in rules.rules] if rules else [],
                "contradiction_report":
                    contradiction_report.model_dump() if contradiction_report else None,
                "canonical_rules":  [cr.model_dump() for cr in canonical_rules],
                "relations":        [r.model_dump()  for r in relations],
                "relate_raw_pairs": [p.model_dump()  for p in relate_raw_pairs],
                "final_rules":      [fr.model_dump() for fr in final_rules],
                "audit_trail": {
                    "map_chunks": [cs.model_dump() for cs in chunk_summaries],
                    "master_summary":   master.model_dump() if master else None,
                    "rules_provenance": rules.model_dump()  if rules else None,
                },
                "map_run_metadata": {
                    "schema_version":    handle.schema_version or MAP_SCHEMA_VERSION,
                    "prompt_version":    handle.prompt_version or MAP_PROMPT_VERSION,
                    "stage_name":        handle.stage_name or MAP_STAGE_NAME,
                    "cascade_profile":   handle.cascade_profile or self._cascade_profile,
                    "cascade_signature": handle.cascade_signature or self._cascade_signature,
                    "l1_voters": [{"provider": v.provider, "model": v.model} for v in self._l1],
                    "l2_voters": [{"provider": v.provider, "model": v.model} for v in self._l2],
                    "l3_voter":  {"provider": self._l3.provider, "model": self._l3.model},
                },
            }
            out_path = self._summaries_dir / f"{pmcid}.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("[%s] Result saved to %s", pmcid, out_path)

            if writer is not None:
                writer.finalize("completed")
            return result

        except Exception as exc:
            if writer is not None:
                writer.write_error(stage="batch_finalize", error=str(exc), pmcid=pmcid)
                writer.finalize("failed", error=str(exc))
            raise

    # ── Filesystem artifact helpers ────────────────────────────────────────────

    def _make_run_id(self, pmcid: str) -> str:
        """Mirror SummarizationRunner._make_run_id format for consistency."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{pmcid}_{ts}"

    def _make_artifact_writer(
        self, run_id: str, pmcid: str, cascade_signature: str | None = None,
    ) -> RunArtifactWriter | None:
        """Build the per-run filesystem writer when ``artifact_root`` is set.

        Returns None when persistence is disabled. Failure is logged but never
        fatal — filesystem artifacts are observational.
        """
        if self._artifact_root is None:
            return None
        try:
            cfg = self._cfg_full
            thresholds = {
                "grounding_threshold": (
                    self._grounding.threshold if self._grounding is not None else None
                ),
                "entailment_threshold":   cfg.relate.entailment_threshold,
                "contradiction_threshold": cfg.relate.contradiction_threshold,
                "map_theta":              cfg.map.theta,
                "map_reject_theta":       cfg.map.reject_theta,
                "contradiction_similarity_threshold": cfg.contradiction_similarity_threshold,
            }
            models = {
                "voter_models":        [v.model for v in self._l1],
                "level2_voter_models": [v.model for v in self._l2],
                "escalation_model":    self._l3.model,
            }
            manifest = RunManifest(
                run_id=run_id,
                artifact_root=self._artifact_root,
                timestamp_start=datetime.now(tz=timezone.utc).isoformat(),
                papers=[pmcid],
                schema_version=MAP_SCHEMA_VERSION,
                prompt_version=MAP_PROMPT_VERSION,
                cascade_signature=cascade_signature or self._cascade_signature,
                config=self._config_snapshot,
                models=models,
                thresholds=thresholds,
                chunk_size=cfg.map.chunk_size,
            )
            return RunArtifactWriter(
                run_id=run_id, root_dir=self._artifact_root, manifest=manifest,
            )
        except Exception as exc:
            logger.warning("[%s] artifact writer setup failed: %s", pmcid, exc)
            return None

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

        # Store raw content for debugging and accumulate token usage per model
        raw_store = getattr(handle, f"{level}_raw")
        current_voters = self._l1 if level == "l1" else self._l2
        level_usage = handle.token_usage.setdefault(level, {})
        for res in raw_results:
            if res.content:
                raw_store[res.custom_id] = res.content
            parts = res.custom_id.split("__")
            voter_idx = int(parts[-1]) if len(parts) >= 4 and parts[-1].isdigit() else 0
            model_id = current_voters[voter_idx].model if voter_idx < len(current_voters) else "unknown"
            m = level_usage.setdefault(model_id, {"input": 0, "output": 0})
            m["input"]  += res.input_tokens
            m["output"] += res.output_tokens

        # ── Pass 1: parse all voter outputs for every chunk ───────────────────
        chunk_voters: dict[str, list[AuditableSummary]] = {}
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
            chunk_voters[chunk_id] = voters

        # ── Pass 2: batch-embed all unique claims upfront ─────────────────────
        from ..agreement.embedding import _claims as _extract_claims
        all_texts: list[str] = list({
            c
            for voters in chunk_voters.values()
            for v in voters
            for c in _extract_claims(v)
        })
        embed_cache: dict[str, list[float]] = {}
        if all_texts:
            raw_embed = self._embed_fn
            logger.info(
                "[%s] Pre-embedding %d unique claim strings across %d chunks",
                pmcid, len(all_texts), len(targets),
            )
            embs = raw_embed(all_texts)
            embed_cache = dict(zip(all_texts, embs))

        def _cached_embed(texts: list[str]) -> list[list[float]]:
            result = []
            misses: list[tuple[int, str]] = []
            for idx, t in enumerate(texts):
                if t in embed_cache:
                    result.append(embed_cache[t])
                else:
                    misses.append((idx, t))
                    result.append(None)  # type: ignore[arg-type]
            if misses:
                new_embs = raw_embed([t for _, t in misses])
                for (idx, t), e in zip(misses, new_embs):
                    embed_cache[t] = e
                    result[idx] = e
            return result  # type: ignore[return-value]

        agreement = AgreementChecker(
            scorer=EmbeddingScorer(_cached_embed),
            theta=self._agreement.theta,
            reject_theta=self._agreement.reject_theta,
        )

        # ── Pass 3: agreement scoring from cache — no API calls ───────────────
        escalated: list[str] = []

        for chunk_id in targets:
            voters = chunk_voters[chunk_id]

            if not voters:
                logger.warning("[%s] Chunk %s: all %s voters failed — escalating", pmcid, chunk_id, level)
                escalated.append(chunk_id)
                continue

            source_text = _format_sentences(handle.chunk_map[chunk_id])
            bundle = agreement.compute(voters, source_text=source_text)

            if bundle.decision == ChunkDecision.KEEP:
                best = agreement.best(voters, bundle=bundle)
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

        l3_model = self._l3.model
        l3_level = handle.token_usage.setdefault("l3", {})
        l3_m = l3_level.setdefault(l3_model, {"input": 0, "output": 0})
        for res in raw_results:
            if res.content:
                handle.l3_raw[res.custom_id] = res.content
            l3_m["input"]  += res.input_tokens
            l3_m["output"] += res.output_tokens

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
