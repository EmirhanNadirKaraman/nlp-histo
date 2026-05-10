"""ILP-based selectors for the related / diverse / hard buckets.

PuLP-backed alternatives to the greedy selectors in :mod:`selectors`. Each
function returns the same ``(papers, rationale)`` shape so the rest of the
pipeline (export, validation) is strategy-agnostic.

PuLP is an optional dependency — call sites can fall back to greedy when it is
not installed; see :func:`pulp_available`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

from .metrics import Hardness, Relatedness
from .models import HardnessBreakdown, PaperFingerprint, SelectionResult
from .selectors import (
    SelectionConfig,
    _eligible_short,
    select_calibration_set,
)

logger = logging.getLogger(__name__)


# ── PuLP availability ────────────────────────────────────────────────────────

try:
    import pulp as _pulp  # type: ignore
    _PULP_IMPORT_ERROR: Exception | None = None
except ImportError as _e:  # pragma: no cover — exercised only without PuLP
    _pulp = None  # type: ignore
    _PULP_IMPORT_ERROR = _e


def pulp_available() -> bool:
    """True iff PuLP can be imported in the current environment."""
    return _pulp is not None


def _require_pulp() -> None:
    if _pulp is None:
        raise ImportError(
            "PuLP is required for ILP-based selection. "
            "Install with `pip install pulp`, or pass --strategy greedy."
        ) from _PULP_IMPORT_ERROR


# ── ILP-specific knobs ───────────────────────────────────────────────────────

@dataclass
class ILPConfig:
    """Numeric weights and limits specific to the ILP selectors.

    These do not replace :class:`SelectionConfig`; they extend it. Weights are
    chosen so each objective is roughly normalised to the same magnitude as
    the corresponding greedy heuristic — small enough to keep solver time
    short, large enough to discriminate tied candidates.
    """
    # Candidate-pool limit — applied per-bucket before building the ILP.
    candidate_limit: int = 200
    # Solver time budget (seconds, per-bucket). None = no limit.
    time_limit_seconds: float | None = None

    # Related ILP weights
    related_quality_weight: float = 0.05   # × paper_quality_score(i)
    related_length_penalty: float = 0.02   # × normalised n_sentences
    # Diverse ILP weights — one per concept type. A None weight disables the type.
    diverse_concept_weights: dict[str, float] = field(
        default_factory=lambda: {
            "disease":   0.30,
            "biomarker": 0.20,
            "gene":      0.10,
            "tissue":    0.10,
            "method":    0.10,
            "outcome":   0.05,
            "cui":       0.10,
        }
    )
    diverse_pairwise_penalty: float = 0.40
    diverse_near_duplicate_penalty: float = 0.20
    diverse_near_duplicate_threshold: float = 0.65

    # Hard ILP — pool sizes used for the >= 2 / >= 2 / >= 1 constraints
    hard_top_normalized_pool: int = 30
    hard_top_absolute_pool:   int = 30
    hard_medium_pool_low_pct: float = 0.40
    hard_medium_pool_high_pct: float = 0.65


# ── Solver helper ────────────────────────────────────────────────────────────

def _solve(prob: "_pulp.LpProblem", time_limit_seconds: float | None) -> int:
    """Run CBC (PuLP's bundled solver) with optional time budget. Returns status."""
    solver = _pulp.PULP_CBC_CMD(
        msg=0,
        timeLimit=time_limit_seconds if time_limit_seconds and time_limit_seconds > 0 else None,
    )
    return prob.solve(solver)


def _selected_pmcids(papers: list[PaperFingerprint],
                     x: dict[str, "_pulp.LpVariable"]) -> list[str]:
    chosen: list[str] = []
    for p in papers:
        v = x.get(p.pmcid)
        if v is None:
            continue
        val = _pulp.value(v)
        if val is not None and val > 0.5:
            chosen.append(p.pmcid)
    return chosen


# ── Candidate pre-scoring ────────────────────────────────────────────────────

def _related_prescore(p: PaperFingerprint) -> float:
    """Favour short, entity-rich papers with disease/biomarker/gene anchors."""
    bucket_richness = (
        len(p.disease_entities) + len(p.biomarker_entities) + len(p.gene_entities)
    )
    length = max(p.n_sentences, 1)
    return bucket_richness / math.log1p(length + 10)


def _diverse_prescore(p: PaperFingerprint) -> float:
    """Favour entity-rich, manageable-size papers."""
    vocab = (
        len(p.disease_entities) + len(p.method_entities)
        + len(p.biomarker_entities) + len(p.outcome_entities)
        + len(p.tissue_entities)
    )
    length = max(p.n_sentences, 1)
    return vocab / math.log1p(length / 50 + 1)


def _hard_prescore(p: PaperFingerprint, hardness: Hardness) -> float:
    return hardness.breakdown(p).absolute_hardness


def _limit_candidates(
    papers: list[PaperFingerprint],
    *,
    limit: int,
    key,
) -> list[PaperFingerprint]:
    if limit <= 0 or len(papers) <= limit:
        return list(papers)
    scored = sorted(papers, key=lambda p: (-key(p), p.pmcid))
    return scored[:limit]


# ── Related ILP ──────────────────────────────────────────────────────────────

def _paper_quality_score(p: PaperFingerprint) -> float:
    """Crude, deterministic quality signal in roughly [0, 1].

    Rewards papers that have well-rounded structured-entity coverage (each
    bucket contributes at most 0.2). Independent of length.
    """
    buckets = (
        p.disease_entities, p.biomarker_entities, p.gene_entities,
        p.tissue_entities, p.method_entities, p.outcome_entities,
    )
    return sum(min(len(b) / 5.0, 0.2) for b in buckets)


def _length_penalty(p: PaperFingerprint, ref_sentences: int = 350) -> float:
    """Normalised length penalty in [0, 1+]."""
    return min(p.n_sentences / max(ref_sentences, 1), 2.0)


def select_related_papers_ilp(
    fingerprints: list[PaperFingerprint],
    *,
    k: int = 5,
    config: SelectionConfig | None = None,
    ilp_config: ILPConfig | None = None,
    exclude_pmcids: set[str] | None = None,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """ILP version of related-paper selection.

    Decision variables::

        x_i ∈ {0,1}                — paper i selected
        y_ij ∈ {0,1}, i<j          — both i and j selected

    Constraints::

        Σ x_i = k
        y_ij ≤ x_i, y_ij ≤ x_j, y_ij ≥ x_i + x_j − 1   (linking)

    Objective (maximise)::

        Σ rel(i,j)·y_ij
        + α · Σ paper_quality(i)·x_i
        − β · Σ length_penalty(i)·x_i
    """
    _require_pulp()
    cfg = config or SelectionConfig()
    ilp = ilp_config or ILPConfig()
    excluded = set(exclude_pmcids or [])

    eligible = [p for p in _eligible_short(fingerprints, cfg, cfg.max_sentences_related)
                if p.pmcid not in excluded]
    pool = _limit_candidates(eligible, limit=ilp.candidate_limit, key=_related_prescore)

    if len(pool) < k:
        logger.warning("select_related_ilp: only %d candidates after limiting (need %d)",
                       len(pool), k)
        if not pool:
            return [], {}
        # Cannot enforce =k when |pool| < k — return whatever we have.
        return pool[:k], {p.pmcid: {"rank": i + 1,
                                     "selection_reason": "insufficient ILP pool, returned without optimisation"}
                          for i, p in enumerate(pool[:k])}

    rel_metric = cfg.relatedness if isinstance(cfg.relatedness, Relatedness) else Relatedness()
    pmcids = [p.pmcid for p in pool]
    pidx = {pid: i for i, pid in enumerate(pmcids)}
    quality = {p.pmcid: _paper_quality_score(p) for p in pool}
    length  = {p.pmcid: _length_penalty(p) for p in pool}
    pair_rel: dict[tuple[str, str], float] = {}
    for i, a in enumerate(pool):
        for j in range(i + 1, len(pool)):
            b = pool[j]
            pair_rel[(a.pmcid, b.pmcid)] = rel_metric.score(a, b)

    prob = _pulp.LpProblem("related_ilp", _pulp.LpMaximize)
    x = {pid: _pulp.LpVariable(f"x_{pid}", cat=_pulp.LpBinary) for pid in pmcids}
    y = {pair: _pulp.LpVariable(f"y_{pair[0]}_{pair[1]}", cat=_pulp.LpBinary)
         for pair in pair_rel}

    prob += (
        _pulp.lpSum(pair_rel[pair] * y[pair] for pair in pair_rel)
        + ilp.related_quality_weight * _pulp.lpSum(quality[pid] * x[pid] for pid in pmcids)
        - ilp.related_length_penalty * _pulp.lpSum(length[pid] * x[pid] for pid in pmcids)
    )
    prob += _pulp.lpSum(x[pid] for pid in pmcids) == k

    for (a, b) in pair_rel:
        prob += y[(a, b)] <= x[a]
        prob += y[(a, b)] <= x[b]
        prob += y[(a, b)] >= x[a] + x[b] - 1

    status = _solve(prob, ilp.time_limit_seconds)
    chosen_ids = _selected_pmcids(pool, x)
    by_pid = {p.pmcid: p for p in pool}
    chosen = [by_pid[pid] for pid in chosen_ids]
    chosen.sort(key=lambda p: pidx[p.pmcid])

    obj = _pulp.value(prob.objective)
    rationale: dict[str, dict] = {}
    for rank, p in enumerate(chosen, start=1):
        rationale[p.pmcid] = {
            "rank": rank,
            "selection_reason": (
                f"ILP related: quality={quality[p.pmcid]:.3f}, "
                f"length_penalty={length[p.pmcid]:.3f}"
            ),
            "ilp_status": _pulp.LpStatus[status],
            "ilp_objective": float(obj) if obj is not None else None,
            "ilp_pool_size": len(pool),
        }
    return chosen, rationale


# ── Diverse ILP ──────────────────────────────────────────────────────────────

# Concepts pulled from each fingerprint, prefixed by bucket. Prefixing keeps
# concept identifiers globally unique and lets us weight per-bucket directly.
_DIVERSE_BUCKETS = (
    ("disease",   "disease_entities"),
    ("biomarker", "biomarker_entities"),
    ("gene",      "gene_entities"),
    ("tissue",    "tissue_entities"),
    ("method",    "method_entities"),
    ("outcome",   "outcome_entities"),
)


def _concepts_for(p: PaperFingerprint, *,
                  include_cuis: bool) -> dict[str, set[str]]:
    """Bucket → set of concept tokens for one paper."""
    out: dict[str, set[str]] = {}
    for bucket, attr in _DIVERSE_BUCKETS:
        out[bucket] = {f"{bucket}::{e.lower()}" for e in getattr(p, attr)}
    if include_cuis:
        out["cui"] = {f"cui::{c}" for c in p.umls_cuis}
    return out


def select_diverse_papers_ilp(
    fingerprints: list[PaperFingerprint],
    *,
    k: int = 5,
    config: SelectionConfig | None = None,
    ilp_config: ILPConfig | None = None,
    exclude_pmcids: set[str] | None = None,
    include_cuis: bool = True,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """ILP version of diverse-paper selection.

    Variables::

        x_i ∈ {0,1}, y_ij ∈ {0,1}, z_c ∈ {0,1}

    Constraints::

        Σ x_i = k
        z_c ≤ Σ_{i: c ∈ p_i} x_i              — coverage activation
        y_ij linked to x_i, x_j (only required for pairs that contribute to
        the objective — i.e. relatedness > 0 or near-duplicate pairs)

    Objective (maximise)::

        Σ_c w(type(c)) · z_c
        − pair_w · Σ rel(i,j) · y_ij
        − near_dup_w · Σ_{rel(i,j) ≥ τ} y_ij
    """
    _require_pulp()
    cfg = config or SelectionConfig()
    ilp = ilp_config or ILPConfig()
    excluded = set(exclude_pmcids or [])

    eligible = [p for p in _eligible_short(fingerprints, cfg, cfg.max_sentences_diverse)
                if p.pmcid not in excluded]
    pool = _limit_candidates(eligible, limit=ilp.candidate_limit, key=_diverse_prescore)
    if len(pool) < k:
        logger.warning("select_diverse_ilp: only %d candidates after limiting (need %d)",
                       len(pool), k)
        if not pool:
            return [], {}
        return pool[:k], {p.pmcid: {"rank": i + 1,
                                     "selection_reason": "insufficient ILP pool, returned without optimisation"}
                          for i, p in enumerate(pool[:k])}

    rel_metric = cfg.relatedness if isinstance(cfg.relatedness, Relatedness) else Relatedness()

    # Concept inventory per bucket
    paper_concepts: dict[str, dict[str, set[str]]] = {
        p.pmcid: _concepts_for(p, include_cuis=include_cuis) for p in pool
    }
    concept_to_papers: dict[str, list[str]] = {}
    concept_bucket: dict[str, str] = {}
    for pid, by_bucket in paper_concepts.items():
        for bucket, concepts in by_bucket.items():
            for c in concepts:
                concept_to_papers.setdefault(c, []).append(pid)
                concept_bucket[c] = bucket

    # Pair scores — only build a y_ij when the pair actually shows up in the
    # objective (rel > 0 or marked near-duplicate). Cuts variable count for
    # sparsely-related pools.
    pair_rel: dict[tuple[str, str], float] = {}
    for i, a in enumerate(pool):
        for j in range(i + 1, len(pool)):
            b = pool[j]
            s = rel_metric.score(a, b)
            if s > 0 or s >= ilp.diverse_near_duplicate_threshold:
                pair_rel[(a.pmcid, b.pmcid)] = s

    prob = _pulp.LpProblem("diverse_ilp", _pulp.LpMaximize)
    x = {p.pmcid: _pulp.LpVariable(f"x_{p.pmcid}", cat=_pulp.LpBinary) for p in pool}
    y = {pair: _pulp.LpVariable(f"y_{pair[0]}_{pair[1]}", cat=_pulp.LpBinary)
         for pair in pair_rel}
    z = {c: _pulp.LpVariable(f"z_{idx}", cat=_pulp.LpBinary)
         for idx, c in enumerate(concept_to_papers)}

    cw = ilp.diverse_concept_weights

    coverage_term = _pulp.lpSum(
        cw.get(concept_bucket[c], 0.0) * z[c] for c in concept_to_papers
    )
    pair_penalty = _pulp.lpSum(
        ilp.diverse_pairwise_penalty * pair_rel[pair] * y[pair] for pair in pair_rel
    )
    near_dupes = [pair for pair, s in pair_rel.items()
                  if s >= ilp.diverse_near_duplicate_threshold]
    near_dup_penalty = _pulp.lpSum(
        ilp.diverse_near_duplicate_penalty * y[pair] for pair in near_dupes
    )
    prob += coverage_term - pair_penalty - near_dup_penalty

    prob += _pulp.lpSum(x[p.pmcid] for p in pool) == k

    for c, owners in concept_to_papers.items():
        prob += z[c] <= _pulp.lpSum(x[pid] for pid in owners)

    for (a, b) in pair_rel:
        prob += y[(a, b)] <= x[a]
        prob += y[(a, b)] <= x[b]
        prob += y[(a, b)] >= x[a] + x[b] - 1

    status = _solve(prob, ilp.time_limit_seconds)
    chosen_ids = _selected_pmcids(pool, x)
    by_pid = {p.pmcid: p for p in pool}
    chosen = [by_pid[pid] for pid in chosen_ids]
    chosen.sort(key=lambda p: p.pmcid)

    obj = _pulp.value(prob.objective)
    rationale: dict[str, dict] = {}
    for rank, p in enumerate(chosen, start=1):
        contributed = sum(1 for c, owners in concept_to_papers.items() if p.pmcid in owners)
        rationale[p.pmcid] = {
            "rank": rank,
            "selection_reason": f"ILP diverse: contributes {contributed} concepts to coverage",
            "ilp_status": _pulp.LpStatus[status],
            "ilp_objective": float(obj) if obj is not None else None,
            "ilp_pool_size": len(pool),
            "ilp_n_concepts": len(concept_to_papers),
            "ilp_n_near_dup_pairs": len(near_dupes),
        }
    return chosen, rationale


# ── Hard ILP ─────────────────────────────────────────────────────────────────

def select_hard_papers_ilp(
    fingerprints: list[PaperFingerprint],
    *,
    k: int = 5,
    config: SelectionConfig | None = None,
    ilp_config: ILPConfig | None = None,
    exclude_pmcids: set[str] | None = None,
    allow_overlap: bool = False,
) -> tuple[list[PaperFingerprint], dict[str, dict]]:
    """ILP version of hard-paper selection.

    Composition constraints::

        Σ x_i = k
        Σ_{i ∈ TopNorm}    x_i ≥ hard_high_normalized   (default 2)
        Σ_{i ∈ TopAbs}     x_i ≥ hard_high_absolute     (default 2)
        Σ_{i ∈ MediumBand} x_i ≥ hard_medium_control    (default 1)

    Objective: maximise Σ absolute_hardness(i) · x_i. Empty / near-empty
    papers are filtered before constraints are built.
    """
    _require_pulp()
    cfg = config or SelectionConfig()
    ilp = ilp_config or ILPConfig()
    excluded = set() if allow_overlap else set(exclude_pmcids or [])
    hardness = cfg.hardness

    candidates = [p for p in fingerprints
                  if p.pmcid not in excluded and not p.is_empty()]
    if not candidates:
        return [], {}

    # Pre-prune by absolute hardness so the ILP is small even on 980-paper inputs.
    pool = _limit_candidates(
        candidates,
        limit=ilp.candidate_limit,
        key=lambda p: _hard_prescore(p, hardness),
    )

    breakdowns: dict[str, HardnessBreakdown] = {p.pmcid: hardness.breakdown(p) for p in pool}
    by_norm = sorted(pool, key=lambda p: -breakdowns[p.pmcid].normalized_hardness)
    by_abs  = sorted(pool, key=lambda p: -breakdowns[p.pmcid].absolute_hardness)
    by_med  = sorted(pool, key=lambda p:  breakdowns[p.pmcid].normalized_hardness)

    top_norm = {p.pmcid for p in by_norm[: ilp.hard_top_normalized_pool]}
    top_abs  = {p.pmcid for p in by_abs[:  ilp.hard_top_absolute_pool]}

    n = len(by_med)
    lo = int(n * ilp.hard_medium_pool_low_pct)
    hi = max(int(n * ilp.hard_medium_pool_high_pct), lo + 1)
    medium_band = {p.pmcid for p in by_med[lo:hi]}

    if len(pool) < k:
        logger.warning("select_hard_ilp: only %d candidates after limiting (need %d)",
                       len(pool), k)
        return pool[:k], {p.pmcid: {"rank": i + 1,
                                     "selection_reason": "insufficient ILP pool, returned without optimisation"}
                          for i, p in enumerate(pool[:k])}

    prob = _pulp.LpProblem("hard_ilp", _pulp.LpMaximize)
    x = {p.pmcid: _pulp.LpVariable(f"x_{p.pmcid}", cat=_pulp.LpBinary) for p in pool}

    prob += _pulp.lpSum(breakdowns[p.pmcid].absolute_hardness * x[p.pmcid] for p in pool)
    prob += _pulp.lpSum(x[p.pmcid] for p in pool) == k

    if cfg.hard_high_normalized > 0 and top_norm:
        prob += _pulp.lpSum(x[pid] for pid in top_norm) >= min(
            cfg.hard_high_normalized, len(top_norm)
        ), "min_top_normalized"
    if cfg.hard_high_absolute > 0 and top_abs:
        prob += _pulp.lpSum(x[pid] for pid in top_abs) >= min(
            cfg.hard_high_absolute, len(top_abs)
        ), "min_top_absolute"
    if cfg.hard_medium_control > 0 and medium_band:
        prob += _pulp.lpSum(x[pid] for pid in medium_band) >= min(
            cfg.hard_medium_control, len(medium_band)
        ), "min_medium_band"

    status = _solve(prob, ilp.time_limit_seconds)
    chosen_ids = _selected_pmcids(pool, x)
    by_pid = {p.pmcid: p for p in pool}
    chosen = [by_pid[pid] for pid in chosen_ids]
    chosen.sort(key=lambda p: -breakdowns[p.pmcid].absolute_hardness)

    obj = _pulp.value(prob.objective)
    rationale: dict[str, dict] = {}
    for rank, p in enumerate(chosen, start=1):
        b = breakdowns[p.pmcid]
        bucket_tags: list[str] = []
        if p.pmcid in top_norm:    bucket_tags.append("top_normalized")
        if p.pmcid in top_abs:     bucket_tags.append("top_absolute")
        if p.pmcid in medium_band: bucket_tags.append("medium_band")
        rationale[p.pmcid] = {
            "rank": rank,
            "selection_reason": (
                f"ILP hard: absolute={b.absolute_hardness:.3f}, "
                f"normalized={b.normalized_hardness:.3f}, "
                f"sub-pools=[{','.join(bucket_tags) or 'none'}]"
            ),
            "ilp_status": _pulp.LpStatus[status],
            "ilp_objective": float(obj) if obj is not None else None,
            "ilp_pool_size": len(pool),
            "hardness_breakdown": {
                "normalized_hardness": b.normalized_hardness,
                "absolute_hardness":   b.absolute_hardness,
                "content_complexity":  b.content_complexity,
                "layout_complexity":   b.layout_complexity,
                "relation_complexity": b.relation_complexity,
                "workload_factor":     b.workload_factor,
                "reasons":             list(b.reasons),
            },
        }
    return chosen, rationale


# ── Calibration set (ILP-end-to-end) ─────────────────────────────────────────

def select_calibration_set_ilp(
    fingerprints: list[PaperFingerprint],
    *,
    config: SelectionConfig | None = None,
    ilp_config: ILPConfig | None = None,
    allow_overlap: bool = False,
    fallback_to_greedy: bool = False,
) -> SelectionResult:
    """ILP-driven 15-paper calibration set.

    Order: related → diverse (excluding related) → hard (excluding both),
    matching the greedy version. With ``fallback_to_greedy=True``, falls back
    silently if PuLP is missing.
    """
    if not pulp_available():
        if fallback_to_greedy:
            logger.warning("PuLP not installed — falling back to greedy selectors.")
            return select_calibration_set(fingerprints, config=config,
                                          allow_overlap=allow_overlap)
        _require_pulp()  # raises with the install hint

    cfg = config or SelectionConfig()
    ilp = ilp_config or ILPConfig()

    related, related_r = select_related_papers_ilp(
        fingerprints, k=cfg.k_related, config=cfg, ilp_config=ilp,
    )
    rel_pmcids = {p.pmcid for p in related}

    div_excl = set() if allow_overlap else rel_pmcids
    diverse, diverse_r = select_diverse_papers_ilp(
        fingerprints, k=cfg.k_diverse, config=cfg, ilp_config=ilp,
        exclude_pmcids=div_excl,
    )
    div_pmcids = {p.pmcid for p in diverse}

    hard_excl = set() if allow_overlap else (rel_pmcids | div_pmcids)
    hard, hard_r = select_hard_papers_ilp(
        fingerprints, k=cfg.k_hard, config=cfg, ilp_config=ilp,
        exclude_pmcids=hard_excl, allow_overlap=allow_overlap,
    )

    result = SelectionResult(
        related=[p.pmcid for p in related],
        diverse=[p.pmcid for p in diverse],
        hard=[p.pmcid for p in hard],
        rationale={},
    )
    for pmcid, info in related_r.items():
        result.rationale[pmcid] = {**info, "bucket": "related", "strategy": "ilp"}
    for pmcid, info in diverse_r.items():
        if pmcid in result.rationale and allow_overlap:
            result.rationale[pmcid]["diverse_info"] = info
        else:
            result.rationale[pmcid] = {**info, "bucket": "diverse", "strategy": "ilp"}
    for pmcid, info in hard_r.items():
        if pmcid in result.rationale and allow_overlap:
            result.rationale[pmcid]["hard_info"] = info
        else:
            result.rationale[pmcid] = {**info, "bucket": "hard", "strategy": "ilp"}

    return result


__all__ = [
    "ILPConfig",
    "pulp_available",
    "select_calibration_set_ilp",
    "select_diverse_papers_ilp",
    "select_hard_papers_ilp",
    "select_related_papers_ilp",
]
