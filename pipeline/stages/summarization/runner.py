"""
SummarizationRunner — orchestrates MAP → REDUCE → RULES with ABC cascading.

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
from datetime import datetime, timezone
from pathlib import Path

from .cache import PipelineCache
from .contradiction_detector import ContradictionDetector
from .grounding_filter import GroundingFilter
from .map_stage import MapStage
from .models import ConsolidatedSummary, ContradictionReport, ExtractedRules
from .observability import TraceCollector, flush_collector
from .reduce_stage import ReduceStage
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
        output_dir: Path = Path("langchain-summarization/summarization_results"),
        cache_path: Path | None = None,
        trace_enabled: bool = False,
        trace_dir: Path | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._summaries_dir = output_dir / "summaries"
        self._summaries_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_path or (output_dir / "pipeline_cache.json")
        self._cache = PipelineCache(cache_file)

        self._map = MapStage(voter_llms, level2_voter_llms, escalation_llm, theta=theta, chunk_size=chunk_size, scorer=scorer)
        self._reduce = ReduceStage(escalation_llm)
        self._rules = RuleStage(escalation_llm)
        self._grounding: GroundingChecker | None = (
            GroundingFilter(grounding_threshold) if grounding_threshold is not None else None
        )
        self._contradiction: ContradictionChecker | None = (
            ContradictionDetector(escalation_llm, similarity_threshold=contradiction_similarity_threshold)
            if contradiction_similarity_threshold is not None else None
        )

        self._trace_enabled = trace_enabled
        self.trace_dir: Path = trace_dir or (output_dir / "traces")

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

        # Skip if already fully processed
        existing = self._load_result(pmcid)
        if existing is not None:
            logger.info("[%s] skipped (cached result on disk)", pmcid)
            if self._trace_enabled:
                collector = self._make_collector(pmcid)
                flush_collector(collector, self.trace_dir, status="skipped")
            return existing

        collector: TraceCollector | None = (
            self._make_collector(pmcid) if self._trace_enabled else None
        )

        try:
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
            chunk_summaries = self._map.process(
                sentences, pmcid, cache=self._cache, collector=collector
            )

            # Record chunking info (chunk_size is on MapStage)
            if collector is not None:
                collector.record_chunking(
                    total_chunks=len(chunk_summaries),
                    chunk_size=self._map.chunk_size,
                )

            # 1a. Grounding filter — drop ungrounded findings before REDUCE
            if self._grounding is not None:
                findings_before = sum(len(cs.findings) for cs in chunk_summaries)
                chunk_summaries = [
                    self._grounding.filter_findings(cs) for cs in chunk_summaries
                ]
                findings_after = sum(len(cs.findings) for cs in chunk_summaries)
                logger.info(
                    "[%s] Grounding (MAP): %d findings remaining across %d chunks",
                    pmcid,
                    findings_after,
                    len(chunk_summaries),
                )
                if collector is not None:
                    collector.record_grounding(
                        stage="map_findings",
                        items_before=findings_before,
                        items_after=findings_after,
                    )

            # 2. REDUCE (recursive tree collapse)
            logger.info("[%s] REDUCE — %d chunks", pmcid, len(chunk_summaries))
            master: ConsolidatedSummary = self._reduce.reduce(
                chunk_summaries, pmcid, cache=self._cache, collector=collector
            )

            # 3. RULE EXTRACTION
            logger.info("[%s] RULES", pmcid)
            rules: ExtractedRules = self._rules.extract(
                master, pmcid, cache=self._cache, collector=collector
            )

            # 3a. Grounding filter — drop ungrounded rules
            if self._grounding is not None:
                rules_before = len(rules.rules)
                rules = self._grounding.filter_rules(rules)
                rules_after = len(rules.rules)
                logger.info("[%s] Grounding (RULES): %d rules remaining", pmcid, rules_after)
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
            self._save_result(result)
            self._cache.save()

            if collector is not None:
                result_path = str(self._result_path(pmcid))
                collector.add_artifact(result_path, "result_json")
                flush_collector(collector, self.trace_dir, status="success")

            return result

        except Exception as exc:
            logger.exception("[%s] Pipeline failed: %s", pmcid, exc)
            if collector is not None:
                flush_collector(collector, self.trace_dir, status="error", error=str(exc))
            return {"status": "error", "pmcid": pmcid, "error": str(exc)}

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

    def _make_collector(self, pmcid: str) -> TraceCollector:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_id = f"{pmcid}_{ts}"
        return TraceCollector(
            run_id=run_id,
            pmcid=pmcid,
            config_snapshot=self._config_snapshot,
        )

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
