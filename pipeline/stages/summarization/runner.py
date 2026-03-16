"""
SummarizationRunner — orchestrates MAP → REDUCE → RULES with ABC cascading.

Usage example
-------------
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pipeline.stages.summarization import SummarizationRunner

runner = SummarizationRunner(
    voter_llms=[
        ChatOpenAI(model="gpt-4o-mini",                    temperature=0.1),
        ChatAnthropic(model="claude-haiku-4-5-20251001",   temperature=0.1),
    ],
    escalation_llm=ChatOpenAI(model="gpt-4o", temperature=0),
    theta=0.7,
    output_dir=Path("langchain-summarization/summarization_results"),
)

result = runner.process(file_data)   # file_data = dict from load_json_files_with_provenance
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .agreement import AgreementStrategy
from .cache import PipelineCache
from .contradiction_detector import ContradictionDetector
from .grounding_filter import GroundingFilter
from .map_stage import MapStage
from .models import ConsolidatedSummary, ContradictionReport, ExtractedRules
from .reduce_stage import ReduceStage
from .rule_stage import RuleStage

logger = logging.getLogger(__name__)


class SummarizationRunner:
    """
    Full pipeline runner: MAP (with ABC cascade) → REDUCE → RULES.

    Parameters
    ----------
    voter_llms:
        List of LangChain chat models used as Level-1 voters in the MAP stage.
        Use models from different providers for genuine independence
        (e.g. [ChatOpenAI("gpt-4o-mini"), ChatAnthropic("claude-haiku-...")]).
    escalation_llm:
        LLM for MAP Level-2 escalations, REDUCE, and RULES.
        Typically a larger model (e.g. gpt-4o).
    theta:
        Agreement threshold for the ABC cascade in the MAP stage.
    chunk_size:
        Sentences per MAP chunk.
    agreement_strategy:
        Strategy used to score voter agreement in the MAP stage.  Defaults
        to EmbeddingAgreement.  Pass CategoryJaccardAgreement() for a fast,
        API-free alternative.
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
    """

    def __init__(
        self,
        voter_llms: list,
        escalation_llm,
        theta: float = 0.7,
        chunk_size: int = 10,
        agreement_strategy: Optional[AgreementStrategy] = None,
        grounding_threshold: Optional[float] = 0.5,
        contradiction_similarity_threshold: Optional[float] = 0.7,
        output_dir: Path = Path("langchain-summarization/summarization_results"),
        cache_path: Optional[Path] = None,
    ) -> None:
        self._output_dir = output_dir
        self._summaries_dir = output_dir / "summaries"
        self._summaries_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_path or (output_dir / "pipeline_cache.json")
        self._cache = PipelineCache(cache_file)

        self._map = MapStage(voter_llms, escalation_llm, theta=theta, chunk_size=chunk_size, agreement_strategy=agreement_strategy)
        self._reduce = ReduceStage(escalation_llm)
        self._rules = RuleStage(escalation_llm)
        self._grounding = GroundingFilter(grounding_threshold) if grounding_threshold is not None else None
        self._contradiction = (
            ContradictionDetector(escalation_llm, similarity_threshold=contradiction_similarity_threshold)
            if contradiction_similarity_threshold is not None else None
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, file_data: dict) -> dict:
        """
        Run the full pipeline for one concept file.

        ``file_data`` must have the shape produced by
        ``load_json_files_with_provenance`` in the notebook:
        keys: cui, concept_name, sentences_with_provenance (list of dicts).

        Returns a result dict with keys:
            status, cui, concept_name, summary, rules, audit_trail
        or on failure:
            status='error', cui, concept_name, error
        """
        cui = file_data["cui"]
        concept_name = file_data["concept_name"]

        # Skip if already fully processed
        existing = self._load_result(cui)
        if existing is not None:
            logger.info("[%s] %s — skipped (cached result on disk)", cui, concept_name)
            return existing

        try:
            sentences = file_data["sentences_with_provenance"]

            # 1. MAP (ABC cascade per chunk)
            logger.info("[%s] MAP — %d sentences", cui, len(sentences))
            chunk_summaries = self._map.process(sentences, concept_name, cache=self._cache)

            # 1a. Grounding filter — drop ungrounded findings before REDUCE
            if self._grounding is not None:
                chunk_summaries = [
                    self._grounding.filter_findings(cs) for cs in chunk_summaries
                ]
                logger.info(
                    "[%s] Grounding (MAP): %d findings remaining across %d chunks",
                    cui,
                    sum(len(cs.findings) for cs in chunk_summaries),
                    len(chunk_summaries),
                )

            # 2. REDUCE (recursive tree collapse)
            logger.info("[%s] REDUCE — %d chunks", cui, len(chunk_summaries))
            master: ConsolidatedSummary = self._reduce.reduce(
                chunk_summaries, concept_name, cache=self._cache
            )

            # 3. RULE EXTRACTION
            logger.info("[%s] RULES", cui)
            rules: ExtractedRules = self._rules.extract(master, concept_name, cache=self._cache)

            # 3a. Grounding filter — drop ungrounded rules
            if self._grounding is not None:
                rules = self._grounding.filter_rules(rules)
                logger.info("[%s] Grounding (RULES): %d rules remaining", cui, len(rules.rules))

            # 4. Contradiction detection
            contradiction_report: Optional[ContradictionReport] = None
            if self._contradiction is not None:
                logger.info("[%s] CONTRADICTION DETECTION", cui)
                contradiction_report = self._contradiction.detect(rules)

            result = {
                "status": "success",
                "cui": cui,
                "concept_name": concept_name,
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
            return result

        except Exception as exc:
            logger.exception("[%s] Pipeline failed: %s", cui, exc)
            return {"status": "error", "cui": cui, "concept_name": concept_name, "error": str(exc)}

    def process_batch(self, file_data_list: list[dict]) -> list[dict]:
        """
        Run the pipeline over a list of concept files and return all results.
        Logs a summary at the end.
        """
        results = []
        for i, fd in enumerate(file_data_list, 1):
            logger.info("--- [%d/%d] %s ---", i, len(file_data_list), fd.get("concept_name", "?"))
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

    def _result_path(self, cui: str) -> Path:
        return self._summaries_dir / f"{cui}.json"

    def _load_result(self, cui: str) -> Optional[dict]:
        p = self._result_path(cui)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _save_result(self, result: dict) -> None:
        p = self._result_path(result["cui"])
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
