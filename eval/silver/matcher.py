"""
Match pipeline findings against silver findings.

Matching algorithm:
1. Build embedding input for each finding from claim + subject + outcome +
   relation_type + direction + category (richer signal than claim alone).
2. Compute cosine similarity matrix per case.
3. Greedy one-to-one matching: repeatedly pick the highest-similarity pair
   above threshold (default 0.55), mark both as used.
4. Record field mismatches for matched pairs.
5. Compute precision / recall / F1 (loose) and strict_F1.

Embedding results are cached on disk (keyed by sha256 of model + normalised
input) so threshold sweeps run without additional API calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from .schemas import (
    EvalMetrics,
    FieldMismatch,
    MatchedPair,
    MatchResult,
    PipelineCaseOutput,
    PipelineFinding,
    SilverCaseResult,
    SilverFinding,
)

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.55
EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_CACHE_PATH = Path("eval/data/embedding_cache_openai.json")
DEFAULT_GEMINI_CACHE_PATH = Path("eval/data/embedding_cache_gemini.json")

# Fields compared for field-mismatch detection (both silver and pipeline have these)
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
    ("category",      "category"),
    ("relation_type", "relation_type"),
    ("direction",     "direction"),
]

# "Important" fields for strict-F1: a mismatch on any of these downgrades a
# matched pair to a partial true positive (0.5).
STRICT_FIELDS = {"category", "relation_type", "direction"}


# ── Embedding input ────────────────────────────────────────────────────────────

def _embedding_input(finding: Union[SilverFinding, PipelineFinding]) -> str:
    """Build a richer embedding input from multiple structured fields."""
    parts = [
        finding.claim or "",
        finding.subject_entity or "",
        finding.outcome_entity or "",
        finding.relation_type or "",
        finding.direction or "",
        finding.category or "",
    ]
    return " | ".join(p.strip() for p in parts if p.strip())


# ── Embedding cache ────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Disk-backed embedding cache.

    Cache key: sha256(embedding_model + '\\0' + normalised_input_text).
    This is stable across runs and across machines — the key does not depend
    on file ordering or Python object identity.
    """

    def __init__(self, path: Path, embedding_model: str = EMBEDDING_MODEL) -> None:
        self.path = path
        self.embedding_model = embedding_model
        self._entries: dict[str, list[float]] = {}
        self._created_at: str = ""
        self._dirty = False
        self._load()

    def _make_key(self, text: str) -> str:
        normalised = text.lower().strip()
        raw = f"{self.embedding_model}\x00{normalised}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self._entries.get(self._make_key(text))

    def set(self, text: str, embedding: list[float]) -> None:
        self._entries[self._make_key(text)] = embedding
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.embedding_model,
            "created_at": self._created_at or datetime.now(timezone.utc).isoformat(),
            "entries": self._entries,
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self._dirty = False
        logger.debug("Saved embedding cache (%d entries) → %s", len(self._entries), self.path)

    def _load(self) -> None:
        if not self.path.exists():
            self._created_at = datetime.now(timezone.utc).isoformat()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("embedding_model") != self.embedding_model:
                logger.warning(
                    "Embedding cache model mismatch (%s vs %s) — ignoring cache",
                    data.get("embedding_model"), self.embedding_model,
                )
                return
            self._entries = data.get("entries", {})
            self._created_at = data.get("created_at", "")
            logger.info("Loaded embedding cache: %d entries from %s", len(self._entries), self.path)
        except Exception as exc:
            logger.warning("Could not load embedding cache: %s", exc)

    def __len__(self) -> int:
        return len(self._entries)


# ── Embedding fetch ────────────────────────────────────────────────────────────

def get_embeddings(
    texts: list[str],
    embedder: object,  # any callable: (list[str]) -> list[list[float]]
    cache: EmbeddingCache,
) -> list[list[float]]:
    """Return embeddings for texts, using cache for hits and batching misses."""
    results: list[list[float] | None] = [None] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            results[i] = cached
        else:
            miss_indices.append(i)
            miss_texts.append(t)

    if miss_texts:
        logger.info("Fetching %d embedding(s) from API (cache has %d)",
                    len(miss_texts), len(cache))
        fetched = embedder(miss_texts)  # type: ignore[operator]
        for idx, text, emb in zip(miss_indices, miss_texts, fetched):
            cache.set(text, emb)
            results[idx] = emb
        cache.save()

    return results  # type: ignore[return-value]


# ── Cosine similarity ──────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Field mismatch detection ───────────────────────────────────────────────────

def _normalize(text: str | None) -> str:
    return (text or "").lower().strip()


def _field_mismatches(silver_finding: SilverFinding, pipeline_finding: PipelineFinding) -> list[FieldMismatch]:
    mismatches = []
    for sf, pf in _FLAT_FIELD_PAIRS:
        sv = _normalize(getattr(silver_finding, sf, None))
        pv = _normalize(getattr(pipeline_finding, pf, None))
        if sv and pv and sv != pv:
            mismatches.append(FieldMismatch(field=sf, silver=sv or None, pipeline=pv or None))
    silver_scope = getattr(silver_finding, "scope", None)
    for sf, pf in _SCOPE_FIELD_PAIRS:
        scope_attr = sf.split(".")[-1]
        sv = _normalize(str(getattr(silver_scope, scope_attr, None) or ""))
        pv = _normalize(str(getattr(pipeline_finding, pf, None) or ""))
        if sv and pv and sv != pv:
            mismatches.append(FieldMismatch(field=sf, silver=sv or None, pipeline=pv or None))
    return mismatches


# ── Similarity matrix ──────────────────────────────────────────────────────────

def compute_sim_matrix(
    silver: SilverCaseResult,
    pipeline: PipelineCaseOutput,
    embedder: object,  # any callable: (list[str]) -> list[list[float]]
    cache: EmbeddingCache,
) -> tuple[list[list[float]], list[str], list[str]]:
    """
    Returns (sim_matrix, silver_texts, pipeline_texts).

    sim_matrix[i][j] = cosine(silver[i], pipeline[j]).
    Embeddings are resolved from cache when available.
    """
    silver_texts = [_embedding_input(f) for f in silver.findings]
    pipeline_texts = [_embedding_input(f) for f in pipeline.findings]

    if not silver_texts or not pipeline_texts:
        return [], silver_texts, pipeline_texts

    all_texts = silver_texts + pipeline_texts
    all_embs = get_embeddings(all_texts, embedder, cache)
    s_embs = all_embs[:len(silver_texts)]
    p_embs = all_embs[len(silver_texts):]

    sim = [
        [_cosine(s_embs[i], p_embs[j]) for j in range(len(pipeline_texts))]
        for i in range(len(silver_texts))
    ]
    return sim, silver_texts, pipeline_texts


# ── Greedy matching from pre-computed matrix ───────────────────────────────────

def match_from_matrix(
    silver: SilverCaseResult,
    pipeline: PipelineCaseOutput,
    sim_matrix: list[list[float]],
    threshold: float,
) -> MatchResult:
    """Apply greedy one-to-one matching to a pre-computed similarity matrix."""
    n_s = len(silver.findings)
    n_p = len(pipeline.findings)
    silver_claims = [f.claim for f in silver.findings]
    pipeline_claims = [f.claim for f in pipeline.findings]

    if not sim_matrix:
        return MatchResult(
            case_id=silver.case_id,
            pmcid=silver.pmcid,
            te_id=silver.te_id,
            matched=[],
            unmatched_silver=silver_claims,
            unmatched_pipeline=pipeline_claims,
        )

    used_s: set[int] = set()
    used_p: set[int] = set()
    matched: list[MatchedPair] = []

    while True:
        best_score = threshold
        best_i = best_j = -1
        for i in range(n_s):
            if i in used_s:
                continue
            for j in range(n_p):
                if j in used_p:
                    continue
                if sim_matrix[i][j] > best_score:
                    best_score = sim_matrix[i][j]
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

    unmatched_silver = [silver_claims[i] for i in range(n_s) if i not in used_s]
    unmatched_pipeline = [pipeline_claims[j] for j in range(n_p) if j not in used_p]

    return MatchResult(
        case_id=silver.case_id,
        pmcid=silver.pmcid,
        te_id=silver.te_id,
        matched=matched,
        unmatched_silver=unmatched_silver,
        unmatched_pipeline=unmatched_pipeline,
    )


# ── Convenience wrapper (backward-compatible) ──────────────────────────────────

def match_case(
    silver: SilverCaseResult,
    pipeline: PipelineCaseOutput,
    embedder: object,
    *,
    cache: EmbeddingCache | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> MatchResult:
    """Match one case. Uses/updates cache if provided."""
    if cache is None:
        cache = EmbeddingCache(DEFAULT_CACHE_PATH)
    sim, _, _ = compute_sim_matrix(silver, pipeline, embedder, cache)
    return match_from_matrix(silver, pipeline, sim, threshold)


# ── Metrics ────────────────────────────────────────────────────────────────────

def _strict_tp(match_results: list[MatchResult]) -> float:
    """Strict TP: full credit (1.0) if no important field mismatches, else 0.5."""
    total = 0.0
    for r in match_results:
        for pair in r.matched:
            important_mismatch = any(m.field in STRICT_FIELDS for m in pair.field_mismatches)
            total += 0.5 if important_mismatch else 1.0
    return total


def compute_metrics(
    match_results: list[MatchResult],
    silver_results: list[SilverCaseResult],
    pipeline_outputs: list[PipelineCaseOutput],
    prompt_version: str,
    model: str,
    *,
    split: str = "all",
    split_seed: int = 42,
    dev_fraction: float = 0.8,
    threshold: float = SIMILARITY_THRESHOLD,
) -> EvalMetrics:
    n_silver = sum(len(s.findings) for s in silver_results)
    n_pipeline = sum(len(p.findings) for p in pipeline_outputs)
    n_matched = sum(len(r.matched) for r in match_results)
    all_sims = [pair.similarity for r in match_results for pair in r.matched]

    precision = n_matched / n_pipeline if n_pipeline else 0.0
    recall = n_matched / n_silver if n_silver else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0

    # Strict F1
    stp = _strict_tp(match_results)
    strict_precision = stp / n_pipeline if n_pipeline else 0.0
    strict_recall = stp / n_silver if n_silver else 0.0
    strict_f1 = (
        (2 * strict_precision * strict_recall / (strict_precision + strict_recall))
        if (strict_precision + strict_recall) else 0.0
    )

    n_field_mismatches = sum(
        len(pair.field_mismatches)
        for r in match_results for pair in r.matched
    )

    run_ids = sorted({p.pipeline_run_id for p in pipeline_outputs})
    evaluated_case_ids = sorted(r.case_id for r in match_results)

    return EvalMetrics(
        n_cases=len(match_results),
        n_silver_findings=n_silver,
        n_pipeline_findings=n_pipeline,
        n_matched=n_matched,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        avg_similarity=round(avg_sim, 4),
        strict_f1=round(strict_f1, 4),
        n_field_mismatches=n_field_mismatches,
        prompt_version=prompt_version,
        model=model,
        pipeline_run_ids=run_ids,
        similarity_threshold=threshold,
        split=split,
        split_seed=split_seed,
        dev_fraction=dev_fraction,
        evaluated_case_ids=evaluated_case_ids,
    )
