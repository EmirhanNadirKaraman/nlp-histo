"""
SummarizationRunner — orchestrates MAP → REDUCE → RULES → NORMALIZE → GROUP
→ CANONICALIZE → RELATE → RESOLVE with ABC cascading.

Usage example
-------------
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pipeline.stages.summarization import SummarizationRunner

runner = SummarizationRunner(
    voter_llms=[                                                   # Level 1: cheapest
        AzureChatOpenAI(model="DeepSeek-V3.2-Speciale",          temperature=0.1),
        VertexAI(model="gemini-2.5-flash-lite-preview-06-17",     temperature=0.1),
        AzureChatOpenAI(model="mistral-large-3",                  temperature=0.1),
    ],
    level2_voter_llms=[                                            # Level 2: mid-tier
        VertexAI(model="gemini-2.5-flash",                        temperature=0.1),
        AzureChatOpenAI(model="kimi-k2.5",                        temperature=0.1),
        ChatAnthropic(model="claude-haiku-4-5-20251001",          temperature=0.1),
    ],
    escalation_llm=ChatAnthropic(model="claude-sonnet-4-6", temperature=0),  # Level 3
    theta=0.7,
    output_dir=Path("out/summaries"),
    trace_enabled=True,   # ← enable structured JSONL traces
)

# Single paper
file_data = SummarizationRunner.load_paper_from_db("PMC10047158")
result = runner.process(file_data)

# Batch
pmcids = ["PMC10047158", "PMC10047213", "PMC10047408"]
results = runner.process_batch([SummarizationRunner.load_paper_from_db(p) for p in pmcids])

# After a batch, export CSV summaries from the JSONL traces:
from pipeline.stages.summarization.observability import export_all_csv
counts = export_all_csv(runner.trace_dir)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .cache import PipelineCache
from .canonicalize_stage import CanonicalizeStage
from .contradiction_detector import ContradictionDetector
from .grounding_filter import GroundingFilter, score_findings
from .group_stage import GroupStage, is_groupable
from .map_stage import MapStage
from .models import (
    CanonicalRule,
    ConsolidatedSummary,
    ContradictionReport,
    ExtractedRules,
    FinalRule,
    Finding,
    FindingGroup,
    NormalFinding,
    Relation,
)
from .normalize_stage import NormalizeStage
from .observability import TraceCollector, flush_collector
from .reduce_stage import ReduceStage
from .relate_stage import RelateStage
from .resolve_stage import ResolveStage
from .rule_stage import RuleStage
from pipeline.stages.summarization.interfaces import (
    ContradictionChecker,
    GroundingChecker,
    MapOutputScorer,
)

logger = logging.getLogger(__name__)


class SummarizationRunner:
    """
    Full pipeline runner: MAP (with ABC cascade) → REDUCE → RULES.

    Parameters
    ----------
    voter_llms:
        List of LangChain chat models used as Level-1 voters in the MAP stage.
        Use cheap models from different providers for genuine independence
        (e.g. [DeepSeek, Gemini-Flash-Lite, Mistral-Large]).
    level2_voter_llms:
        List of LangChain chat models used as Level-2 voters.  Called only when
        Level-1 voters disagree.  Use mid-tier models from different providers
        (e.g. [Gemini-Flash, kimi-k2.5, Haiku]).
    escalation_llm:
        LLM for MAP Level-3 (final) escalations, REDUCE, and RULES.
        Typically the most capable model (e.g. Sonnet 4.6).
    theta:
        Agreement threshold for the ABC cascade in the MAP stage.
    chunk_size:
        Sentences per MAP chunk.
    scorer:
        MapOutputScorer used to score voter agreement in the MAP stage.
        Defaults to EmbeddingScorer.  Pass CascadedCompositeScorer for
        embedding + LLM judge cascade.
    grounding_threshold:
        Minimum NLI entailment score to consider a claim grounded.  Set to
        None to disable grounding filtering entirely.
    contradiction_similarity_threshold:
        Cosine similarity threshold for candidate rule pairs sent to the LLM
        judge in contradiction detection.  Set to None to disable.
    output_dir:
        Where to write per-concept result JSON files.
    cache_path:
        Override for the cache file location.  Defaults to
        ``output_dir/pipeline_cache.json``.
    trace_enabled:
        When True, structured JSONL traces are written to ``trace_dir`` for
        every processed paper.  Traces answer "why was this chunk escalated?",
        "which pairwise score was low?", etc.
    trace_dir:
        Directory for JSONL trace files.  Defaults to ``output_dir/traces``.
        Files: ``runs.jsonl``, ``chunks.jsonl``.
    """

    def __init__(
        self,
        voter_llms: list,
        level2_voter_llms: list,
        escalation_llm,
        theta: float = 0.7,
        chunk_size: int = 10,
        scorer: MapOutputScorer | None = None,
        grounding_threshold: float | None = 0.5,
        contradiction_similarity_threshold: float | None = 0.7,
        canonicalize_with_llm: bool = True,
        nli_entailment_threshold: float = 0.50,
        nli_contradiction_threshold: float = 0.50,
        output_dir: Path = Path("langchain-summarization/summarization_results"),
        cache_path: Path | None = None,
        trace_enabled: bool = False,
        trace_dir: Path | None = None,
        db=None,
        force_rerun: bool = False,
        run_ner: bool = True,
    ) -> None:
        self._output_dir = output_dir
        self._summaries_dir = output_dir / "summaries"
        self._summaries_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_path or (output_dir / "pipeline_cache.json")
        self._cache = PipelineCache(cache_file)

        self._map = MapStage(voter_llms, level2_voter_llms, escalation_llm, theta=theta, chunk_size=chunk_size, scorer=scorer)
        self._normalize = NormalizeStage()
        self._group = GroupStage()
        self._canonicalize = CanonicalizeStage(
            llm=escalation_llm if canonicalize_with_llm else None
        )
        self._relate = RelateStage(
            entailment_threshold=nli_entailment_threshold,
            contradiction_threshold=nli_contradiction_threshold,
        )
        self._resolve = ResolveStage()
        self._reduce = ReduceStage(escalation_llm)
        self._rules = RuleStage(escalation_llm)
        self._grounding: GroundingChecker | None = (
            GroundingFilter(grounding_threshold) if grounding_threshold is not None else None
        )
        self._contradiction: ContradictionChecker | None = (
            ContradictionDetector(escalation_llm, similarity_threshold=contradiction_similarity_threshold)
            if contradiction_similarity_threshold is not None else None
        )

        self._db = db  # DatabaseConnection | None — persistence is fully optional
        self._force_rerun = force_rerun
        self._run_ner = run_ner

        self._trace_enabled = trace_enabled
        self.trace_dir: Path = trace_dir or (output_dir / "traces")

        # Stores all post-MAP findings with grounding_score written in-place,
        # mirroring exactly what flows into REDUCE. NORMALIZE reads this.
        self._scored_map_findings: dict[str, list[Finding]] = {}
        # Stores NORMALIZE output per pmcid. Input to GROUP.
        self._normal_findings: dict[str, list[NormalFinding]] = {}
        # Stores GROUP output per pmcid. Input to Phase 4 CANONICALIZE.
        self._finding_groups: dict[str, list[FindingGroup]] = {}
        # Stores CANONICALIZE output per pmcid. Input to Phase 5 RELATE.
        self._canonical_rules: dict[str, list[CanonicalRule]] = {}
        # Stores RELATE output per pmcid. Input to Phase 6 RESOLVE.
        self._relations: dict[str, list[Relation]] = {}
        # Stores RESOLVE output per pmcid. Final knowledge base.
        self._final_rules: dict[str, list[FinalRule]] = {}

        # Snapshot of config for traces (model introspection is best-effort)
        self._config_snapshot = {
            "theta": theta,
            "chunk_size": chunk_size,
            "scorer": type(scorer).__name__ if scorer else "EmbeddingScorer",
            "grounding_threshold": grounding_threshold,
            "contradiction_similarity_threshold": contradiction_similarity_threshold,
            "voter_model_count": len(voter_llms),
            "voter_models": [_model_name(m) for m in voter_llms],
            "level2_voter_model_count": len(level2_voter_llms),
            "level2_voter_models": [_model_name(m) for m in level2_voter_llms],
            "escalation_model": _model_name(escalation_llm),
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, file_data: dict) -> dict:
        """
        Run the full pipeline for one paper.

        ``file_data`` must have the shape produced by ``load_paper_from_db``:
        keys: pmcid, sentences_with_provenance (list of dicts).

        Returns a result dict with keys:
            status, pmcid, summary, rules, audit_trail
        or on failure:
            status='error', pmcid, error
        """
        pmcid = file_data["pmcid"]

        # Skip if already fully processed (bypass with force_rerun=True)
        if not self._force_rerun:
            existing = self._load_result(pmcid)
            if existing is not None:
                logger.info("[%s] skipped (cached result on disk)", pmcid)
                if self._trace_enabled:
                    collector = self._make_collector(pmcid)
                    flush_collector(collector, self.trace_dir, status="skipped")
                return existing

        run_id = self._make_run_id(pmcid)
        pipeline_run_db_id = self._create_pipeline_run(pmcid, run_id)

        collector: TraceCollector | None = (
            self._make_collector(pmcid, run_id) if self._trace_enabled else None
        )

        try:
            t_total = time.perf_counter()
            sentences = file_data["sentences_with_provenance"]

            # Record ingestion
            if collector is not None:
                te_ids = {s.get("text_element_id", 0) for s in sentences}
                collector.record_ingestion(
                    sentence_count=len(sentences),
                    te_count=len(te_ids),
                )

            # 1. MAP (ABC cascade per chunk)
            logger.info("[%s] MAP — %d sentences", pmcid, len(sentences))
            t0 = time.perf_counter()
            chunk_summaries = self._map.process(
                sentences, pmcid, cache=self._cache, collector=collector
            )
            logger.info("[%s] MAP done [%.1fs] — %d chunks, %d raw findings",
                        pmcid, time.perf_counter() - t0,
                        len(chunk_summaries),
                        sum(len(cs.findings) for cs in chunk_summaries))

            # Record chunking info (chunk_size is on MapStage)
            if collector is not None:
                collector.record_chunking(
                    total_chunks=len(chunk_summaries),
                    chunk_size=self._map.chunk_size,
                )

            # 1a. Grounding filter — drop ungrounded findings before REDUCE
            if self._grounding is not None:
                findings_before = sum(len(cs.findings) for cs in chunk_summaries)
                logger.info("[%s] GROUNDING (filter) — %d findings", pmcid, findings_before)
                t0 = time.perf_counter()
                chunk_summaries = [
                    self._grounding.filter_findings(cs) for cs in chunk_summaries
                ]
                findings_after = sum(len(cs.findings) for cs in chunk_summaries)
                logger.info(
                    "[%s] GROUNDING (filter) done [%.1fs] — %d/%d findings kept",
                    pmcid, time.perf_counter() - t0, findings_after, findings_before,
                )
                if collector is not None:
                    collector.record_grounding(
                        stage="map_findings",
                        items_before=findings_before,
                        items_after=findings_after,
                    )

                # Score all surviving findings in-place
                logger.info("[%s] GROUNDING (score) — %d findings", pmcid, findings_after)
                t0 = time.perf_counter()
                all_findings = [f for cs in chunk_summaries for f in cs.findings]
                score_findings(all_findings, nli_pipe=self._grounding._pipe)
                self._scored_map_findings[pmcid] = all_findings
                logger.info("[%s] GROUNDING (score) done [%.1fs]",
                            pmcid, time.perf_counter() - t0)
                logger.info("[%s] chunk_ids before persist: %s", pmcid, [cs.chunk_id for cs in chunk_summaries])
                self._persist_map_findings(pipeline_run_db_id, pmcid, chunk_summaries)

            # 1b. NORMALIZE — entity normalization + conditional dedup
            n_scored = len(self._scored_map_findings.get(pmcid, []))
            logger.info("[%s] NORMALIZE — %d scored findings", pmcid, n_scored)
            t0 = time.perf_counter()
            self._normal_findings[pmcid] = self._normalize.normalize(
                self._scored_map_findings.get(pmcid, []), pmcid
            )
            logger.info("[%s] NORMALIZE done [%.1fs] — %d NormalFindings",
                        pmcid, time.perf_counter() - t0, len(self._normal_findings[pmcid]))
            _nf_db_id_map = self._persist_normal_findings(
                pipeline_run_db_id, pmcid, self._normal_findings[pmcid]
            )

            # 1c. GROUP — bucket groupable NormalFindings by (subject, outcome, category)
            all_normal = self._normal_findings[pmcid]
            groupable = [nf for nf in all_normal if is_groupable(nf)]
            non_groupable_count = len(all_normal) - len(groupable)
            logger.info(
                "[%s] GROUP — %d groupable, %d non-groupable",
                pmcid, len(groupable), non_groupable_count,
            )
            t0 = time.perf_counter()
            self._finding_groups[pmcid] = self._group.group(groupable, pmcid)
            logger.info("[%s] GROUP done [%.1fs] — %d groups",
                        pmcid, time.perf_counter() - t0, len(self._finding_groups[pmcid]))
            _fg_db_id_map = self._persist_finding_groups(
                pipeline_run_db_id, pmcid, self._finding_groups[pmcid], _nf_db_id_map
            )

            # 1d. CANONICALIZE — FindingGroup[] → CanonicalRule[]
            logger.info("[%s] CANONICALIZE — %d groups", pmcid, len(self._finding_groups[pmcid]))
            t0 = time.perf_counter()
            nf_by_id: dict[str, NormalFinding] = {
                nf.normal_id: nf for nf in all_normal
            }
            self._canonical_rules[pmcid] = self._canonicalize.canonicalize(
                self._finding_groups[pmcid], nf_by_id, pmcid
            )
            logger.info("[%s] CANONICALIZE done [%.1fs] — %d CanonicalRules",
                        pmcid, time.perf_counter() - t0, len(self._canonical_rules[pmcid]))

            # Enrich canonical rules with UMLS CUIs for cross-paper entity matching.
            # No-ops silently if scispacy is unavailable.
            from .entity_linker import enrich_rules_with_cuis  # noqa: PLC0415
            enrich_rules_with_cuis(self._canonical_rules[pmcid])
            _cr_db_id_map = self._persist_canonical_rules(
                pipeline_run_db_id, pmcid, self._canonical_rules[pmcid], _fg_db_id_map
            )
            self._corpus_relate_incremental(pmcid, self._canonical_rules[pmcid])

            # 1e. RELATE — CanonicalRule[] → Relation[]
            logger.info(
                "[%s] RELATE — %d canonical rules", pmcid, len(self._canonical_rules[pmcid])
            )
            t0 = time.perf_counter()
            self._relations[pmcid] = self._relate.relate(
                self._canonical_rules[pmcid], pmcid
            )
            logger.info("[%s] RELATE done [%.1fs] — %d relations",
                        pmcid, time.perf_counter() - t0, len(self._relations[pmcid]))
            self._persist_relations(
                pipeline_run_db_id, pmcid, self._relations[pmcid], _cr_db_id_map
            )

            # 1f. RESOLVE — CanonicalRule[] + Relation[] → FinalRule[]
            logger.info(
                "[%s] RESOLVE — %d canonical rules, %d relations",
                pmcid, len(self._canonical_rules[pmcid]), len(self._relations[pmcid]),
            )
            t0 = time.perf_counter()
            self._final_rules[pmcid] = self._resolve.resolve(
                self._canonical_rules[pmcid], self._relations[pmcid], pmcid
            )
            logger.info("[%s] RESOLVE done [%.1fs] — %d FinalRules",
                        pmcid, time.perf_counter() - t0, len(self._final_rules[pmcid]))
            self._persist_final_rules(
                pipeline_run_db_id, pmcid, self._final_rules[pmcid], _cr_db_id_map
            )

            # 2. REDUCE (recursive tree collapse)
            logger.info("[%s] REDUCE — %d chunks", pmcid, len(chunk_summaries))
            t0 = time.perf_counter()
            master: ConsolidatedSummary = self._reduce.reduce(
                chunk_summaries, pmcid, cache=self._cache, collector=collector
            )
            logger.info("[%s] REDUCE done [%.1fs]", pmcid, time.perf_counter() - t0)

            # 3. RULE EXTRACTION
            logger.info("[%s] RULES", pmcid)
            t0 = time.perf_counter()
            rules: ExtractedRules = self._rules.extract(
                master, pmcid, cache=self._cache, collector=collector
            )
            logger.info("[%s] RULES done [%.1fs] — %d rules",
                        pmcid, time.perf_counter() - t0, len(rules.rules))

            # 3a. Grounding filter — drop ungrounded rules
            if self._grounding is not None:
                rules_before = len(rules.rules)
                logger.info("[%s] GROUNDING (rules) — %d rules", pmcid, rules_before)
                t0 = time.perf_counter()
                rules = self._grounding.filter_rules(rules)
                rules_after = len(rules.rules)
                logger.info("[%s] GROUNDING (rules) done [%.1fs] — %d/%d rules kept",
                            pmcid, time.perf_counter() - t0, rules_after, rules_before)
                if collector is not None:
                    collector.record_grounding(
                        stage="rules",
                        items_before=rules_before,
                        items_after=rules_after,
                    )

            # 4. Contradiction detection
            contradiction_report: ContradictionReport | None = None
            if self._contradiction is not None:
                logger.info("[%s] CONTRADICTION DETECTION", pmcid)
                t0 = time.perf_counter()
                contradiction_report = self._contradiction.detect(rules)
                logger.info("[%s] CONTRADICTION DETECTION done [%.1fs]",
                            pmcid, time.perf_counter() - t0)

            # NER + UMLS linking (optional)
            if self._run_ner and self._db is not None:
                logger.info("[%s] NER — running entity extraction + UMLS linking", pmcid)
                t0 = time.perf_counter()
                try:
                    from named_entity_recognition.ner import run_ner_on_db
                    run_ner_on_db(pmcid, save_to_db=True, force=False)
                    logger.info("[%s] NER done [%.1fs]", pmcid, time.perf_counter() - t0)
                except Exception as ner_exc:
                    logger.warning("[%s] NER failed (non-fatal): %s", pmcid, ner_exc)

            logger.info("[%s] Pipeline complete [%.1fs total]",
                        pmcid, time.perf_counter() - t_total)

            result = {
                "status": "success",
                "run_id": run_id,
                "pmcid": pmcid,
                "summary": master.narrative_summary,
                "rules": [r.model_dump() for r in rules.rules],
                "contradiction_report": contradiction_report.model_dump() if contradiction_report else None,
                # Phase 4–6 knowledge base
                "canonical_rules": [
                    r.model_dump() for r in self._canonical_rules.get(pmcid, [])
                ],
                "relations": [
                    r.model_dump() for r in self._relations.get(pmcid, [])
                ],
                "final_rules": [
                    r.model_dump() for r in self._final_rules.get(pmcid, [])
                ],
                "audit_trail": {
                    "map_chunks": [cs.model_dump() for cs in chunk_summaries],
                    "master_summary": master.model_dump(),
                    "rules_provenance": rules.model_dump(),
                },
            }
            self._finish_pipeline_run(
                pipeline_run_db_id, "success",
                narrative_summary=master.narrative_summary,
            )
            self._save_result(result)
            self._cache.save()

            if collector is not None:
                result_path = str(self._result_path(pmcid))
                collector.add_artifact(result_path, "result_json")
                flush_collector(collector, self.trace_dir, status="success")

            return result

        except KeyboardInterrupt:
            logger.warning("[%s] Pipeline interrupted (KeyboardInterrupt)", pmcid)
            self._finish_pipeline_run(pipeline_run_db_id, "interrupted", error="KeyboardInterrupt")
            if collector is not None:
                flush_collector(collector, self.trace_dir, status="interrupted", error="KeyboardInterrupt")
            raise
        except Exception as exc:
            logger.exception("[%s] Pipeline failed: %s", pmcid, exc)
            self._finish_pipeline_run(pipeline_run_db_id, "failed", error=str(exc))
            if collector is not None:
                flush_collector(collector, self.trace_dir, status="error", error=str(exc))
            return {"status": "error", "run_id": run_id, "pmcid": pmcid, "error": str(exc)}

    def process_batch(self, file_data_list: list[dict]) -> list[dict]:
        """
        Run the pipeline over a list of papers and return all results.
        Logs a summary at the end.
        """
        results = []
        for i, fd in enumerate(file_data_list, 1):
            logger.info("--- [%d/%d] %s ---", i, len(file_data_list), fd.get("pmcid", "?"))
            results.append(self.process(fd))

        n_ok = sum(1 for r in results if r["status"] == "success")
        n_err = sum(1 for r in results if r["status"] == "error")
        n_skip = len(results) - n_ok - n_err
        logger.info(
            "Batch complete: %d ok / %d skipped (cached) / %d errors",
            n_ok, n_skip, n_err,
        )
        logger.info(self._cache.stats_str())
        return results

    def corpus_relate(
        self,
        source_dir: Path | None = None,
        output_path: Path | None = None,
        entailment_threshold: float | None = None,
        contradiction_threshold: float | None = None,
        run_selection: str = "latest_per_pmcid",
        manifest: list[Path] | None = None,
    ) -> list:
        """
        Run the corpus-level relation stage over all per-paper JSONs in source_dir.

        Pools canonical_rules from every per-paper JSON, runs the same NLI-based
        pairwise comparison used by the per-paper RELATE stage, and writes a
        single corpus_relations.json artifact.  Each relation is labeled as
        "intra_paper" or "cross_paper" based on whether both rules came from the
        same paper.

        This is a post-hoc analytical step.  It does NOT modify any per-paper
        output and does NOT affect ResolveStage scoring (which continues to use
        the per-paper ``relations`` key).

        Parameters
        ----------
        source_dir:
            Directory containing per-paper JSON files.
            Defaults to ``self._summaries_dir``.
        output_path:
            Destination for corpus_relations.json.
            Defaults to ``self._output_dir / "corpus_relations.json"``.
        entailment_threshold:
            NLI entailment threshold forwarded to RelateStage.
            Defaults to the threshold used by the per-paper RELATE stage.
        contradiction_threshold:
            NLI contradiction threshold.  Defaults similarly.
        run_selection:
            "latest_per_pmcid" (default) — one representative run per PMCID.
            "all" — every valid file (status=success, non-empty canonical_rules).
            Ignored when manifest is provided.
        manifest:
            Explicit list of JSON paths to load.  When provided, source_dir and
            run_selection are both ignored.

        Returns
        -------
        List of CorpusRelation objects written to output_path.
        """
        from .corpus_relate import CorpusRelateStage  # noqa: PLC0415 — lazy import

        src = source_dir or self._summaries_dir
        out = output_path or (self._output_dir / "corpus_relations.json")

        stage = CorpusRelateStage(
            entailment_threshold=(
                entailment_threshold
                if entailment_threshold is not None
                else self._relate._entailment_threshold
            ),
            contradiction_threshold=(
                contradiction_threshold
                if contradiction_threshold is not None
                else self._relate._contradiction_threshold
            ),
        )
        return stage.relate_from_dir(src, out, run_selection=run_selection, manifest=manifest)

    # ── Disk I/O ───────────────────────────────────────────────────────────────

    @staticmethod
    def load_paper_from_db(pmcid: str, db_url: str | None = None) -> dict:
        """
        Load all text elements for ``pmcid`` from the database and split them
        into sentence-level provenance dicts ready for ``process()``.

        Args:
            pmcid:   PubMed Central ID (e.g. "PMC10047158").
            db_url:  Optional SQLAlchemy database URL.  Defaults to the value
                     in the project .env file.

        Returns:
            dict with keys ``pmcid`` and ``sentences_with_provenance``.
        """
        import spacy  # type: ignore
        from database import get_db_connection, Document, TextElement  # type: ignore

        nlp = spacy.load("en_core_sci_sm")
        db = get_db_connection(database_url=db_url)

        with db.session_scope() as session:
            doc = session.query(Document).filter_by(pmcid=pmcid).first()
            if doc is None:
                raise ValueError(f"PMCID {pmcid!r} not found in database")
            rows = (
                session.query(TextElement)
                .filter_by(document_id=doc.id)
                .order_by(TextElement.position_in_section)
                .all()
            )
            sentences = []
            for te in rows:
                for sent in nlp(te.text_content).sents:
                    text = sent.text.strip()
                    if text:
                        sentences.append({
                            "pmcid": pmcid,
                            "text_element_id": te.id,
                            "sentence": text,
                        })

        return {"pmcid": pmcid, "sentences_with_provenance": sentences}

    def _make_run_id(self, pmcid: str) -> str:
        """Generate a human-readable run identifier: {pmcid}_{YYYYMMDDTHHmmss}."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{pmcid}_{ts}"

    def _make_collector(self, pmcid: str, run_id: str | None = None) -> TraceCollector:
        if run_id is None:
            run_id = self._make_run_id(pmcid)
        return TraceCollector(
            run_id=run_id,
            pmcid=pmcid,
            config_snapshot=self._config_snapshot,
        )

    def _create_pipeline_run(self, pmcid: str, run_id: str) -> int | None:
        """
        Insert a pipeline_runs row with status='running'.

        Returns the integer row id on success, None if db=None or on any error.
        Persistence failures are logged as warnings and never crash the pipeline.
        """
        if self._db is None:
            return None
        try:
            from database import Document, PipelineRun  # noqa: PLC0415 — lazy import
            with self._db.session_scope() as session:
                doc = session.query(Document).filter_by(pmcid=pmcid).first()
                run = PipelineRun(
                    run_id=run_id,
                    pmcid=pmcid,
                    document_id=doc.id if doc else None,
                    status="running",
                    config_snapshot=self._config_snapshot,
                    started_at=datetime.now(tz=timezone.utc),
                )
                session.add(run)
                session.flush()
                return run.id
        except Exception as exc:
            logger.warning("[%s] DB: failed to create pipeline_run: %s", pmcid, exc)
            return None

    def _finish_pipeline_run(
        self,
        db_id: int | None,
        status: str,
        error: str | None = None,
        narrative_summary: str | None = None,
    ) -> None:
        """
        Update a pipeline_runs row to its final status.

        No-op when db_id is None (db=None or creation failed).
        Persistence failures are logged as warnings and never crash the pipeline.
        """
        if db_id is None:
            return
        try:
            from database import PipelineRun  # noqa: PLC0415 — lazy import
            with self._db.session_scope() as session:
                run = session.query(PipelineRun).filter_by(id=db_id).first()
                if run is not None:
                    run.status = status
                    run.finished_at = datetime.now(tz=timezone.utc)
                    run.error = error
                    if narrative_summary is not None:
                        run.narrative_summary = narrative_summary
        except Exception as exc:
            logger.warning("DB: failed to update pipeline_run id=%s: %s", db_id, exc)

    # ── Phase 2: DB persistence ────────────────────────────────────────────────

    def _clear_normalized_run_data(self, db_id: int, pmcid: str) -> None:
        """
        Delete all post-MAP rows for (pipeline_run_id, pmcid) in FK-safe order
        (children before parents).  Called before re-inserting NORMALIZE onwards
        so that re-runs don't hit unique-constraint or FK violations.

        MAP findings are handled separately in _persist_map_findings (which also
        does delete-before-insert), so they are not touched here.
        """
        try:
            from database.models import (
                SumFinalRule, SumRelation, SumCanonicalRule,
                SumGroupMember, SumFindingGroup,
                SumNormalFindingSpan, SumNormalFinding,
            )
            with self._db.engine.begin() as conn:
                t = SumFinalRule.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
                t = SumRelation.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
                t = SumCanonicalRule.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
                # SumGroupMember has no pmcid column — join via finding_group_id
                fg_table = SumFindingGroup.__table__
                subq = fg_table.select().where(
                    (fg_table.c.pipeline_run_id == db_id) & (fg_table.c.pmcid == pmcid)
                ).with_only_columns(fg_table.c.id)
                t = SumGroupMember.__table__
                conn.execute(t.delete().where(t.c.finding_group_id.in_(subq)))
                t = SumFindingGroup.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
                # SumNormalFindingSpan has no pmcid — join via normal_finding_id
                nf_table = SumNormalFinding.__table__
                subq2 = nf_table.select().where(
                    (nf_table.c.pipeline_run_id == db_id) & (nf_table.c.pmcid == pmcid)
                ).with_only_columns(nf_table.c.id)
                t = SumNormalFindingSpan.__table__
                conn.execute(t.delete().where(t.c.normal_finding_id.in_(subq2)))
                t = SumNormalFinding.__table__
                conn.execute(t.delete().where(
                    (t.c.pipeline_run_id == db_id) & (t.c.pmcid == pmcid)))
        except Exception as exc:
            logger.warning(
                "[%s] DB: failed to clear normalized run data (run_id=%d): %s",
                pmcid, db_id, exc, exc_info=True,
            )

    def _persist_map_findings(
        self,
        db_id: int | None,
        pmcid: str,
        chunk_summaries: list,
    ) -> None:
        """
        Persist scored MAP findings (post-grounding) to sum_map_findings.
        Entities are pre-normalization.  No-op when db_id is None.
        """
        if db_id is None:
            return
        try:
            from database.models import SumMapFinding
            from .models import FindingScope
            rows = []
            for cs in chunk_summaries:
                for pos, f in enumerate(cs.findings):
                    scope = f.scope or FindingScope()
                    rows.append({
                        "pipeline_run_id":       db_id,
                        "pmcid":                 pmcid,
                        "chunk_id":              cs.chunk_id,
                        "position_in_chunk":     pos,
                        "category":              f.category,
                        "claim":                 f.claim,
                        "confidence":            f.confidence,
                        "verbatim_support":      f.verbatim_support,
                        "subject_entity":        f.subject_entity,
                        "outcome_entity":        f.outcome_entity,
                        "relation_type":         f.relation_type.value,
                        "direction":             f.direction.value if f.direction else None,
                        "grounding_score":       f.grounding_score,
                        "evidence_refs":         list(f.evidence) if f.evidence else [],
                        "scope_disease_subtype":   scope.disease_subtype,
                        "scope_cohort_n":          scope.cohort_n,
                        "scope_assay_method":      scope.assay_method,
                        "scope_biomarker_cutoff":  scope.biomarker_cutoff,
                        "scope_tissue_site":       scope.tissue_site,
                        "scope_treatment_context": scope.treatment_context,
                        "scope_endpoint":          scope.endpoint,
                        "scope_study_design":      scope.study_design,
                    })
            if not rows:
                return
            with self._db.engine.begin() as conn:
                conn.execute(
                    SumMapFinding.__table__.delete().where(
                        SumMapFinding.__table__.c.pipeline_run_id == db_id
                    )
                )
                conn.execute(SumMapFinding.__table__.insert(), rows)
            logger.info(
                "[%s] DB: persisted %d map findings (run_id=%d)",
                pmcid, len(rows), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist map findings: %s", pmcid, exc, exc_info=True)

    def _persist_normal_findings(
        self,
        db_id: int | None,
        pmcid: str,
        normal_findings: list,
    ) -> dict[str, int]:
        """
        Persist NormalFindings + evidence spans to sum_normal_findings /
        sum_normal_finding_spans.  Returns {normal_id: db_row_id} for later
        phases.  No-op (returns {}) when db_id is None.
        """
        if db_id is None:
            return {}
        self._clear_normalized_run_data(db_id, pmcid)
        nf_id_map: dict[str, int] = {}
        try:
            from database.models import SumNormalFinding, SumNormalFindingSpan
            from .models import FindingScope
            with self._db.session_scope() as session:
                for nf in normal_findings:
                    scope = nf.scope or FindingScope()
                    row = SumNormalFinding(
                        pipeline_run_id     = db_id,
                        pmcid               = pmcid,
                        normal_id           = nf.normal_id,
                        subject_entity      = nf.subject_entity,
                        outcome_entity      = nf.outcome_entity,
                        relation_type       = nf.relation_type.value,
                        direction           = nf.direction.value if nf.direction else None,
                        category            = nf.category,
                        predicate_text      = nf.predicate_text,
                        mean_grounding_score= nf.mean_grounding_score,
                        pmcids              = list(nf.pmcids) if nf.pmcids else [],
                        scope_disease_subtype   = scope.disease_subtype,
                        scope_cohort_n          = scope.cohort_n,
                        scope_assay_method      = scope.assay_method,
                        scope_biomarker_cutoff  = scope.biomarker_cutoff,
                        scope_tissue_site       = scope.tissue_site,
                        scope_treatment_context = scope.treatment_context,
                        scope_endpoint          = scope.endpoint,
                        scope_study_design      = scope.study_design,
                    )
                    session.add(row)
                    session.flush()           # get row.id before adding spans
                    nf_id_map[nf.normal_id] = row.id
                    for span in (nf.evidence or []):
                        session.add(SumNormalFindingSpan(
                            normal_finding_id = row.id,
                            sentence_id       = span.sentence_id,
                            pmcid             = span.pmcid,
                            text_element_id   = span.text_element_id or None,
                            verbatim          = span.verbatim,
                        ))
            logger.info(
                "[%s] DB: persisted %d normal findings + spans (run_id=%d)",
                pmcid, len(nf_id_map), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist normal findings: %s", pmcid, exc)
            return {}  # partial map is invalid after rollback — don't pass stale IDs downstream
        return nf_id_map

    def _persist_finding_groups(
        self,
        db_id: int | None,
        pmcid: str,
        finding_groups: list,
        nf_db_id_map: dict[str, int],
    ) -> dict[str, int]:
        """
        Persist FindingGroups + member links to sum_finding_groups /
        sum_group_members.

        nf_db_id_map maps normal_id -> DB integer id from _persist_normal_findings.
        Member rows are still created when nf_db_id_map is empty — normal_id
        is stored as a string backup; normal_finding_id FK will be NULL.

        Returns {group_id: db_row_id} for Phase 4 (CANONICALIZE) persistence.
        No-op (returns {}) when db_id is None.
        """
        if db_id is None:
            return {}
        group_id_map: dict[str, int] = {}
        try:
            from database.models import SumFindingGroup, SumGroupMember
            with self._db.session_scope() as session:
                for fg in finding_groups:
                    row = SumFindingGroup(
                        pipeline_run_id     = db_id,
                        pmcid               = pmcid,
                        group_id            = fg.group_id,
                        subject_entity      = fg.subject_entity,
                        outcome_entity      = fg.outcome_entity,
                        relation_type       = fg.relation_type.value,
                        category            = fg.category,
                        scope_heterogeneity = fg.scope_heterogeneity,
                        direction_counts    = dict(fg.direction_counts),
                    )
                    session.add(row)
                    session.flush()          # get row.id before adding members
                    group_id_map[fg.group_id] = row.id
                    missing = 0
                    for normal_id in fg.member_ids:
                        nf_db_id = nf_db_id_map.get(normal_id)
                        if nf_db_id is None:
                            missing += 1
                        session.add(SumGroupMember(
                            finding_group_id  = row.id,
                            normal_finding_id = nf_db_id,   # None when Phase 2 skipped
                            normal_id         = normal_id,
                        ))
                    if missing:
                        logger.debug(
                            "[%s] group %s: %d member(s) have no NF DB id (Phase 2 incomplete)",
                            pmcid, fg.group_id, missing,
                        )
            logger.info(
                "[%s] DB: persisted %d finding groups (run_id=%d)",
                pmcid, len(group_id_map), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist finding groups: %s", pmcid, exc)
            return {}
        return group_id_map

    def _persist_canonical_rules(
        self,
        db_id: int | None,
        pmcid: str,
        canonical_rules: list,
        fg_db_id_map: dict[str, int],
    ) -> dict[str, int]:
        """
        Persist CanonicalRules to sum_canonical_rules.
        Returns {canonical_id: db_row_id} for Phase 5/6 persistence.
        """
        if db_id is None:
            return {}
        cr_id_map: dict[str, int] = {}
        try:
            from database.models import SumCanonicalRule
            rows = []
            for cr in canonical_rules:
                fg_db_id = fg_db_id_map.get(cr.group_id)
                row = SumCanonicalRule(
                    pipeline_run_id      = db_id,
                    pmcid                = pmcid,
                    canonical_id         = cr.canonical_id,
                    finding_group_id     = fg_db_id,
                    group_id             = cr.group_id,
                    subject_entity       = cr.subject_entity,
                    outcome_entity       = cr.outcome_entity,
                    relation_type        = cr.relation_type.value,
                    direction            = cr.direction.value if cr.direction else None,
                    predicate_text       = cr.predicate_text,
                    canonical_scope      = cr.canonical_scope.value,
                    category             = cr.category,
                    pmcids               = list(cr.supporting_pmcids) if cr.supporting_pmcids else [],
                    member_normal_ids    = list(cr.member_normal_ids) if cr.member_normal_ids else [],
                    mean_grounding_score = cr.mean_grounding_score,
                    finding_count        = cr.finding_count,
                    subject_cui          = cr.subject_cui,
                    outcome_cui          = cr.outcome_cui,
                )
                rows.append(row)
            
            with self._db.session_scope() as session:
                for row in rows:
                    session.add(row)
                session.flush()
                for row in rows:
                    cr_id_map[row.canonical_id] = row.id
            
            logger.info(
                "[%s] DB: persisted %d canonical rules (run_id=%d)",
                pmcid, len(cr_id_map), db_id,
            )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist canonical rules: %s", pmcid, exc)
            return {}
        return cr_id_map

    def _corpus_relate_incremental(
        self,
        pmcid: str,
        canonical_rules: list,
    ) -> None:
        """
        Run incremental cross-paper corpus relate for a newly processed paper.

        No-ops when DB is unavailable or when fewer than 2 PMCIDs exist in DB.
        Errors are logged as warnings and never propagate — corpus relate is
        analytical and must not block per-paper pipeline completion.
        """
        if self._db is None:
            return
        try:
            from .corpus_relate import CorpusRelateStage  # noqa: PLC0415
            stage = CorpusRelateStage(
                entailment_threshold=self._relate._entailment_threshold,
                contradiction_threshold=self._relate._contradiction_threshold,
            )
            relations = stage.relate_incremental(pmcid, canonical_rules, self._db)
            logger.info(
                "[%s] CORPUS RELATE incremental: %d cross-paper relations",
                pmcid, len(relations),
            )
        except Exception as exc:
            logger.warning(
                "[%s] CORPUS RELATE incremental failed (non-fatal): %s", pmcid, exc,
                exc_info=True,
            )

    def _persist_relations(
        self,
        db_id: int | None,
        pmcid: str,
        relations: list,
        cr_db_id_map: dict[str, int],
    ) -> None:
        """Persist Relations to sum_relations."""
        if db_id is None:
            return
        try:
            from database.models import SumRelation
            rows = []
            for rel in relations:
                rows.append(SumRelation(
                    pipeline_run_id     = db_id,
                    pmcid               = pmcid,
                    rule_id_a           = rel.rule_id_a,
                    rule_id_b           = rel.rule_id_b,
                    canonical_rule_a_id = cr_db_id_map.get(rel.rule_id_a),
                    canonical_rule_b_id = cr_db_id_map.get(rel.rule_id_b),
                    relation_type       = rel.relation_type.value,
                    nli_score_a_to_b    = rel.nli_score_a_to_b,
                    nli_score_b_to_a    = rel.nli_score_b_to_a,
                ))
            if rows:
                with self._db.session_scope() as session:
                    session.bulk_save_objects(rows)
                logger.info(
                    "[%s] DB: persisted %d relations (run_id=%d)",
                    pmcid, len(rows), db_id,
                )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist relations: %s", pmcid, exc)

    def _persist_final_rules(
        self,
        db_id: int | None,
        pmcid: str,
        final_rules: list,
        cr_db_id_map: dict[str, int],
    ) -> None:
        """Persist FinalRules (scored canonical rules) to sum_final_rules."""
        if db_id is None:
            return
        try:
            from database.models import SumFinalRule
            rows = []
            for fr in final_rules:
                rows.append(SumFinalRule(
                    pipeline_run_id      = db_id,
                    pmcid                = pmcid,
                    final_id             = fr.final_id,
                    canonical_rule_id    = cr_db_id_map.get(fr.canonical_id),
                    canonical_id         = fr.canonical_id,
                    subject_entity       = fr.subject_entity,
                    outcome_entity       = fr.outcome_entity,
                    relation_type        = fr.relation_type.value,
                    direction            = fr.direction.value if fr.direction else None,
                    predicate_text       = fr.predicate_text,
                    category             = fr.category,
                    final_score          = fr.final_score,
                    support_count        = fr.support_count,
                    contradict_count     = fr.contradict_count,
                    scope_qualify_count  = fr.scope_qualify_count,
                    is_contradicted      = fr.is_contradicted,
                    contradicted_by      = list(fr.contradicted_by) if fr.contradicted_by else [],
                ))
            if rows:
                with self._db.session_scope() as session:
                    session.bulk_save_objects(rows)
                logger.info(
                    "[%s] DB: persisted %d final rules (run_id=%d)",
                    pmcid, len(rows), db_id,
                )
        except Exception as exc:
            logger.warning("[%s] DB: failed to persist final rules: %s", pmcid, exc)

    def _result_path(self, pmcid: str) -> Path:
        return self._summaries_dir / f"{pmcid}.json"

    def _load_result(self, pmcid: str) -> dict | None:
        p = self._result_path(pmcid)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _save_result(self, result: dict) -> None:
        p = self._result_path(result["pmcid"])
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def _model_name(llm) -> str:
    """Best-effort extraction of model name for config snapshots."""
    for attr in ("model_name", "model", "model_id"):
        val = getattr(llm, attr, None)
        if val:
            return str(val)
    return type(llm).__name__
