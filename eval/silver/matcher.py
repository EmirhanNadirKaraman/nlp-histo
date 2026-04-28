"""
Match pipeline findings against silver findings.

Matching algorithm:
1. Build embedding matrix for all silver claims and all pipeline claims.
2. Compute cosine similarity matrix.
3. Greedy one-to-one matching: repeatedly pick the highest-similarity pair
   above threshold (default 0.55), mark both as used.
4. Record field mismatches for matched pairs.
5. Compute precision / recall / F1.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from .schemas import (
    EvalMetrics,
    FieldMismatch,
    MatchedPair,
    MatchResult,
    PipelineCaseOutput,
    SilverCaseResult,
)

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.55

# Fields to compare after matching (both silver and pipeline have these)
_SCOPE_FIELD_PAIRS = [
    ("scope.disease_subtype",   "scope_disease_subtype"),
    ("scope.cohort_n",          "scope_cohort_n"),
    ("scope.assay_method",      "scope_assay_method"),
    ("scope.biomarker_cutoff",  "scope_biomarker_cutoff"),
    ("scope.tissue_site",       "scope_tissue_site"),
    ("scope.treatment_context", "scope_treatment_context"),
    ("scope.endpoint",          "scope_endpoint"),
    ("scope.study_design",      "scope_study_design"),
]
_FLAT_FIELD_PAIRS = [
    ("category",     "category"),
    ("relation_type","relation_type"),
    ("direction",    "direction"),
]


def _normalize(text: str | None) -> str:
    return (text or "").lower().strip()


def _get_embeddings(texts: list[str], api_key: str) -> "list[list[float]]":
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _field_mismatches(silver_finding, pipeline_finding) -> list[FieldMismatch]:
    mismatches = []
    for sf, pf in _FLAT_FIELD_PAIRS:
        sv = _normalize(getattr(silver_finding, sf, None))
        pv = _normalize(getattr(pipeline_finding, pf, None))
        if sv and pv and sv != pv:
            mismatches.append(FieldMismatch(
                field=sf,
                silver=sv or None,
                pipeline=pv or None,
            ))
    # Scope fields
    silver_scope = getattr(silver_finding, "scope", None)
    for sf, pf in _SCOPE_FIELD_PAIRS:
        scope_attr = sf.split(".")[-1]
        sv = _normalize(str(getattr(silver_scope, scope_attr, None) or ""))
        pv = _normalize(str(getattr(pipeline_finding, pf, None) or ""))
        if sv and pv and sv != pv:
            mismatches.append(FieldMismatch(
                field=sf,
                silver=sv or None,
                pipeline=pv or None,
            ))
    return mismatches


def match_case(
    silver: SilverCaseResult,
    pipeline: PipelineCaseOutput,
    openai_api_key: str,
) -> MatchResult:
    silver_claims = [f.claim for f in silver.findings]
    pipeline_claims = [f.claim for f in pipeline.findings]

    if not silver_claims or not pipeline_claims:
        return MatchResult(
            case_id=silver.case_id,
            pmcid=silver.pmcid,
            te_id=silver.te_id,
            matched=[],
            unmatched_silver=silver_claims,
            unmatched_pipeline=pipeline_claims,
        )

    all_texts = silver_claims + pipeline_claims
    embeddings = _get_embeddings(all_texts, openai_api_key)
    s_embs = embeddings[:len(silver_claims)]
    p_embs = embeddings[len(silver_claims):]

    # Build similarity matrix
    sim = [
        [_cosine(s_embs[i], p_embs[j]) for j in range(len(pipeline_claims))]
        for i in range(len(silver_claims))
    ]

    used_s: set[int] = set()
    used_p: set[int] = set()
    matched: list[MatchedPair] = []

    while True:
        best_score = SIMILARITY_THRESHOLD
        best_i = best_j = -1
        for i in range(len(silver_claims)):
            if i in used_s:
                continue
            for j in range(len(pipeline_claims)):
                if j in used_p:
                    continue
                if sim[i][j] > best_score:
                    best_score = sim[i][j]
                    best_i, best_j = i, j
        if best_i == -1:
            break
        used_s.add(best_i)
        used_p.add(best_j)
        mismatches = _field_mismatches(silver.findings[best_i], pipeline.findings[best_j])
        matched.append(MatchedPair(
            silver_claim=silver_claims[best_i],
            pipeline_claim=pipeline_claims[best_j],
            similarity=best_score,
            field_mismatches=mismatches,
        ))

    unmatched_silver = [silver_claims[i] for i in range(len(silver_claims)) if i not in used_s]
    unmatched_pipeline = [pipeline_claims[j] for j in range(len(pipeline_claims)) if j not in used_p]

    return MatchResult(
        case_id=silver.case_id,
        pmcid=silver.pmcid,
        te_id=silver.te_id,
        matched=matched,
        unmatched_silver=unmatched_silver,
        unmatched_pipeline=unmatched_pipeline,
    )


def compute_metrics(
    match_results: list[MatchResult],
    silver_results: list[SilverCaseResult],
    pipeline_outputs: list[PipelineCaseOutput],
    prompt_version: str,
    model: str,
) -> EvalMetrics:
    n_silver = sum(len(s.findings) for s in silver_results)
    n_pipeline = sum(len(p.findings) for p in pipeline_outputs)
    n_matched = sum(len(r.matched) for r in match_results)
    all_sims = [pair.similarity for r in match_results for pair in r.matched]

    precision = n_matched / n_pipeline if n_pipeline else 0.0
    recall = n_matched / n_silver if n_silver else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0

    run_ids = list({p.pipeline_run_id for p in pipeline_outputs})

    return EvalMetrics(
        n_cases=len(match_results),
        n_silver_findings=n_silver,
        n_pipeline_findings=n_pipeline,
        n_matched=n_matched,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        avg_similarity=round(avg_sim, 4),
        prompt_version=prompt_version,
        model=model,
        pipeline_run_ids=sorted(run_ids),
    )
