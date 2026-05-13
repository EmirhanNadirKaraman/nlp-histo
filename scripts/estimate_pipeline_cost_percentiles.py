"""
Estimate end-to-end summarization-pipeline cost for P80/P90 papers
ordered by text-element count.

Outputs a markdown report to stdout (and to ``out/cost_percentile_report.md``).

Reproducibility
---------------
- Percentile method: nearest-rank, ``index = ceil(p * n) - 1`` over the set of
  papers with at least one text element in the ``text_elements`` table.
- Per-chunk token baseline is taken from the only existing trace we have on
  disk (PMC10047158, escalation_report_20260511T234134.json) — one L3 chunk
  with input=4733, output=2521 tokens for claude-sonnet-4-6.  All other
  voters are assumed to consume similar input tokens (same prompt) and
  similar output tokens (same structured AuditableSummary schema).
- Sentence count per paper is *estimated* from total word count using the
  observed ratio from PMC10047158 (78 sentences / 1335 words ≈ 0.0584).
- Chunk count = ceil(n_sentences / stride) with stride = chunk_size - overlap.

Cost coverage
-------------
The active 6-stage cascade only spends LLM API tokens in **MAP**.  GROUNDING
and RELATE run a local NLI model; NORMALIZE, GROUP, CANONICALIZE, and
RESOLVE are deterministic UMLS / embedding stages.  Their LLM API cost is
$0.  Compute time / GPU is out of scope here.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import func, distinct

from database import get_db_connection
from database.models import Document, TextElement, Figure, Table
from pipeline.stages.summarization.costing import PriceBook
from pipeline.stages.summarization.batch.voter_configs import (
    CascadeProfile, get_profile, list_profiles,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reference numbers
# ─────────────────────────────────────────────────────────────────────────────

# Observed per-chunk tokens (from PMC10047158 escalation_report L3 call).
# Used as the canonical "one LLM call on one MAP chunk" estimate for every
# tier (L1, L2, L3).  Conservative: structured-output overhead is similar
# across providers.
OBS_INPUT_TOK_PER_CHUNK = 4733
OBS_OUTPUT_TOK_PER_CHUNK = 2521

# Sentence-from-words ratio observed in PMC10047158.
OBS_SENT_PER_WORD = 78 / 1335  # ≈ 0.0584

# MAP chunking — must match pipeline/stages/summarization/config.py
CHUNK_SIZE = 10
CHUNK_OVERLAP = 2
STRIDE = CHUNK_SIZE - CHUNK_OVERLAP  # 8


# Profile model lists are pulled from voter_configs at runtime — no local
# copy. `cheap` is structurally a 2-tier cascade (L3 mirrors L2); `real` is
# the production 3-tier cascade. See
# pipeline/stages/summarization/batch/voter_configs.py for the source of truth.
def _profile_models(p: CascadeProfile) -> dict[str, list[str] | str]:
    return {
        "l1": [v.model for v in p.l1_voters],
        "l2": [v.model for v in p.l2_voters],
        "l3": p.l3_voter.model,
    }


PROFILES: dict[str, dict[str, list[str] | str]] = {
    name: _profile_models(get_profile(name)) for name in list_profiles()
}

# Escalation rate scenarios (fraction of chunks that escalate past each tier).
# expected_l2 and expected_l3 are calibrated to the only observed run
# (PMC10047158: 1/8 chunks reached L3, no chunks stopped at L2).
SCENARIOS = {
    "best":     {"l2_rate": 0.05, "l3_rate": 0.01},
    "expected": {"l2_rate": 0.20, "l3_rate": 0.10},
    "worst":    {"l2_rate": 1.00, "l3_rate": 1.00},  # every chunk reaches L3
}


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperMeta:
    pmcid: str
    title: str | None
    n_te: int
    n_words: int
    n_chars: int
    n_figures: int
    n_tables: int
    n_distinct_paths: int

    @property
    def est_sentences(self) -> int:
        return max(1, int(round(self.n_words * OBS_SENT_PER_WORD)))

    @property
    def est_chunks(self) -> int:
        if self.est_sentences <= 0:
            return 0
        # ceil((n - overlap) / stride) but at least 1
        return max(1, math.ceil((self.est_sentences) / STRIDE))


def rank_papers() -> list[PaperMeta]:
    db = get_db_connection()
    out: list[PaperMeta] = []
    with db.session_scope() as s:
        rows = (
            s.query(
                Document.pmcid,
                Document.title,
                func.count(TextElement.id).label("n_te"),
                func.coalesce(
                    func.sum(func.length(TextElement.text_content)), 0
                ).label("n_chars"),
            )
            .join(TextElement, TextElement.document_id == Document.id)
            .group_by(Document.id)
            .order_by(func.count(TextElement.id))
            .all()
        )
        # We need word counts and figure/table counts. Issue a second pass
        # only for the percentile rows — but it's cheap to compute words from
        # chars via avg-word-length 5.6 (English).  We will refine for the
        # selected papers below.
        for r in rows:
            out.append(PaperMeta(
                pmcid=r.pmcid, title=r.title,
                n_te=r.n_te, n_words=int(round(r.n_chars / 5.6)),
                n_chars=r.n_chars,
                n_figures=0, n_tables=0, n_distinct_paths=0,
            ))
    return out


def enrich(paper: PaperMeta) -> PaperMeta:
    db = get_db_connection()
    with db.session_scope() as s:
        doc = s.query(Document).filter_by(pmcid=paper.pmcid).first()
        if doc is None:
            return paper
        figs = s.query(func.count(Figure.id)).filter_by(document_id=doc.id).scalar() or 0
        tabs = s.query(func.count(Table.id)).filter_by(document_id=doc.id).scalar() or 0
        n_paths = (
            s.query(func.count(distinct(TextElement.path_string)))
            .filter_by(document_id=doc.id)
            .scalar() or 0
        )
        # Exact word count
        tes = s.query(TextElement).filter_by(document_id=doc.id).all()
        n_words = sum(len((t.text_content or "").split()) for t in tes)
        return PaperMeta(
            pmcid=paper.pmcid, title=doc.title,
            n_te=paper.n_te, n_words=n_words, n_chars=paper.n_chars,
            n_figures=figs, n_tables=tabs, n_distinct_paths=n_paths,
        )


def pick_percentile(papers: list[PaperMeta], p: float) -> tuple[int, PaperMeta]:
    n = len(papers)
    idx = math.ceil(p * n) - 1
    return idx, papers[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageCost:
    stage: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    input_price_per_1m: float | None
    output_price_per_1m: float | None
    cost_usd: float | None

    batch_cost_usd: float | None = None

    @property
    def batch_cost(self) -> float | None:
        return self.batch_cost_usd


def _price_or_none(book: PriceBook, model: str) -> tuple[float | None, float | None]:
    p = book.get(model)
    if p is None:
        return None, None
    return p.input_per_1m, p.output_per_1m


def _cost(book: PriceBook, model: str, in_tok: int, out_tok: int) -> float | None:
    return book.cost(model, in_tok, out_tok, batch=False)


def _batch_cost(book: PriceBook, model: str, in_tok: int, out_tok: int) -> float | None:
    return book.cost(model, in_tok, out_tok, batch=True)


def estimate_map(
    paper: PaperMeta,
    book: PriceBook,
    profile: dict,
    l2_rate: float,
    l3_rate: float,
) -> list[StageCost]:
    """Per-tier cost rows for a given cascade profile.

    Number of chunks reaching each tier follows the rates. L1 voters fire on
    every chunk. L2 voters fire on `n_chunks × l2_rate`. The single L3 model
    fires on `n_chunks × l3_rate`.
    """
    n_chunks = paper.est_chunks
    rows: list[StageCost] = []

    for m in profile["l1"]:
        in_tok = n_chunks * OBS_INPUT_TOK_PER_CHUNK
        out_tok = n_chunks * OBS_OUTPUT_TOK_PER_CHUNK
        ip, op = _price_or_none(book, m)
        rows.append(StageCost(
            stage="MAP/L1", model=m, calls=n_chunks,
            input_tokens=in_tok, output_tokens=out_tok,
            input_price_per_1m=ip, output_price_per_1m=op,
            cost_usd=_cost(book, m, in_tok, out_tok),
            batch_cost_usd=_batch_cost(book, m, in_tok, out_tok),
        ))

    l2_chunks = int(round(n_chunks * l2_rate))
    for m in profile["l2"]:
        in_tok = l2_chunks * OBS_INPUT_TOK_PER_CHUNK
        out_tok = l2_chunks * OBS_OUTPUT_TOK_PER_CHUNK
        ip, op = _price_or_none(book, m)
        rows.append(StageCost(
            stage="MAP/L2", model=m, calls=l2_chunks,
            input_tokens=in_tok, output_tokens=out_tok,
            input_price_per_1m=ip, output_price_per_1m=op,
            cost_usd=_cost(book, m, in_tok, out_tok),
            batch_cost_usd=_batch_cost(book, m, in_tok, out_tok),
        ))

    l3_chunks = int(round(n_chunks * l3_rate))
    l3_model = profile["l3"]
    in_tok = l3_chunks * OBS_INPUT_TOK_PER_CHUNK
    out_tok = l3_chunks * OBS_OUTPUT_TOK_PER_CHUNK
    ip, op = _price_or_none(book, l3_model)
    rows.append(StageCost(
        stage="MAP/L3", model=l3_model, calls=l3_chunks,
        input_tokens=in_tok, output_tokens=out_tok,
        input_price_per_1m=ip, output_price_per_1m=op,
        cost_usd=_cost(book, l3_model, in_tok, out_tok),
        batch_cost_usd=_batch_cost(book, l3_model, in_tok, out_tok),
    ))
    return rows


def estimate_non_llm_stages(paper: PaperMeta) -> list[StageCost]:
    """GROUNDING / NORMALIZE / GROUP / CANONICALIZE / RELATE / RESOLVE → $0 LLM."""
    zeros = []
    for stage in ("GROUNDING", "NORMALIZE", "GROUP", "CANONICALIZE", "RELATE", "RESOLVE"):
        zeros.append(StageCost(
            stage=stage, model="(local NLI / deterministic)",
            calls=0, input_tokens=0, output_tokens=0,
            input_price_per_1m=None, output_price_per_1m=None,
            cost_usd=0.0, batch_cost_usd=0.0,
        ))
    return zeros


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────

def _money(v: float | None) -> str:
    return "n/a" if v is None else f"${v:,.4f}"


def _ppm(v: float | None) -> str:
    return "n/a" if v is None else f"${v:.2f}"


def _sum_cost(rows: Iterable[StageCost]) -> float | None:
    total = 0.0
    seen_unknown = False
    for r in rows:
        if r.cost_usd is None:
            seen_unknown = True
        else:
            total += r.cost_usd
    return None if seen_unknown else total


def _sum_batch_cost(rows: Iterable[StageCost]) -> float | None:
    total = 0.0
    seen_unknown = False
    for r in rows:
        if r.batch_cost is None:
            seen_unknown = True
        else:
            total += r.batch_cost
    return None if seen_unknown else total


def render_paper_table(label: str, paper: PaperMeta, rows: list[StageCost]) -> list[str]:
    lines = [f"### {label}: {paper.pmcid}", ""]
    lines.append(f"- est. sentences: **{paper.est_sentences}** "
                 f"(from {paper.n_words} words × {OBS_SENT_PER_WORD:.4f} sent/word)")
    lines.append(f"- est. MAP chunks: **{paper.est_chunks}** "
                 f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, stride={STRIDE})")
    lines.append("")
    lines.append("| stage | model | calls | input tok | output tok | "
                 "in $/1M | out $/1M | normal | batch (×0.5) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.stage} | `{r.model}` | {r.calls} | "
            f"{r.input_tokens:,} | {r.output_tokens:,} | "
            f"{_ppm(r.input_price_per_1m)} | {_ppm(r.output_price_per_1m)} | "
            f"{_money(r.cost_usd)} | {_money(r.batch_cost)} |"
        )
    total = _sum_cost(rows)
    batch_total = _sum_batch_cost(rows)
    lines.append(f"| **TOTAL** | | | | | | | **{_money(total)}** | **{_money(batch_total)}** |")
    lines.append("")
    return lines


def main() -> None:
    book = PriceBook.load()
    papers = rank_papers()
    n = len(papers)

    # Median, P80, P90 nearest-rank
    selections: dict[str, tuple[int, PaperMeta]] = {
        "P50": pick_percentile(papers, 0.50),
        "P80": pick_percentile(papers, 0.80),
        "P90": pick_percentile(papers, 0.90),
    }
    enriched = {k: (idx, enrich(p)) for k, (idx, p) in selections.items()}

    lines: list[str] = []
    lines += ["# Pipeline Cost Estimate for P80 / P90 Papers", ""]
    lines += [
        "## Dataset",
        "",
        f"- papers with ≥1 text element: **{n}**",
        f"- ordering: number of text elements (ascending)",
        f"- percentile method: nearest-rank — `idx = ceil(p × n) − 1`",
        f"- price book: `configs/model_prices.json` ({len(book.known_models())} models)",
        f"- batch_discount_multiplier: **{book.batch_discount_multiplier}**",
        "",
        "## Selected papers",
        "",
        "| pct | rank | PMCID | title | text elements | words | figs | tables | sections | est. sentences | est. MAP chunks |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, (idx, paper) in enriched.items():
        title = paper.title or "—"
        lines.append(
            f"| {label} | {idx + 1}/{n} | {paper.pmcid} | {title} | "
            f"{paper.n_te} | {paper.n_words:,} | {paper.n_figures} | {paper.n_tables} | "
            f"{paper.n_distinct_paths} | {paper.est_sentences} | {paper.est_chunks} |"
        )
    lines.append("")

    # Per-profile stage tables at the expected cascade rates.
    expected = SCENARIOS["expected"]
    lines += [
        "## Stage-level cost — expected cascade rates, per profile",
        "",
        f"Expected cascade rates: L2 escalation = {expected['l2_rate']:.0%}, "
        f"L3 escalation = {expected['l3_rate']:.0%} of chunks "
        "(calibrated to the one observed cached run on PMC10047158).",
        "",
        "Profiles imported live from "
        "`pipeline/stages/summarization/batch/voter_configs.py`:",
        "",
    ]
    for pname in list_profiles():
        prof = get_profile(pname)
        l1 = ", ".join(v.model for v in prof.l1_voters)
        l2 = ", ".join(v.model for v in prof.l2_voters)
        l3 = prof.l3_voter.model
        lines.append(f"- **{pname}** — L1: {l1}; L2: {l2}; L3: {l3}")
    lines += [
        "",
        "All non-MAP stages spend **zero LLM API tokens**: GROUNDING and RELATE "
        "run a local NLI model (`cross-encoder/nli-deberta-v3-large`); "
        "NORMALIZE / GROUP / CANONICALIZE / RESOLVE are deterministic UMLS + "
        "embedding stages.",
        "",
    ]
    for label in ("P80", "P90"):
        idx, paper = enriched[label]
        lines.append(f"### {label}: `{paper.pmcid}`")
        lines.append("")
        lines.append(
            f"- est. sentences: **{paper.est_sentences}** "
            f"(from {paper.n_words} words × {OBS_SENT_PER_WORD:.4f} sent/word)"
        )
        lines.append(
            f"- est. MAP chunks: **{paper.est_chunks}** "
            f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, stride={STRIDE})"
        )
        lines.append("")
        for pname, profile in PROFILES.items():
            rows = estimate_map(
                paper, book, profile, expected["l2_rate"], expected["l3_rate"]
            )
            lines.append(f"#### profile = `{pname}`")
            lines.append("")
            lines.append(
                "| stage | model | calls | input tok | output tok | "
                "in $/1M | out $/1M | normal | batch (×0.5) |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
            for r in rows:
                lines.append(
                    f"| {r.stage} | `{r.model}` | {r.calls} | "
                    f"{r.input_tokens:,} | {r.output_tokens:,} | "
                    f"{_ppm(r.input_price_per_1m)} | {_ppm(r.output_price_per_1m)} | "
                    f"{_money(r.cost_usd)} | {_money(r.batch_cost)} |"
                )
            t = _sum_cost(rows)
            bt = _sum_batch_cost(rows)
            lines.append(
                f"| **MAP total** | | | | | | | "
                f"**{_money(t)}** | **{_money(bt)}** |"
            )
            lines.append("")

    # Cascade scenario summary across all profiles.
    lines += [
        "## Cascade scenarios (MAP-only) — all profiles",
        "",
        "| paper | profile | scenario | L2 rate | L3 rate | normal $ | batch $ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for label in ("P80", "P90"):
        idx, paper = enriched[label]
        for pname, profile in PROFILES.items():
            for sname, rates in SCENARIOS.items():
                rows = estimate_map(
                    paper, book, profile, rates["l2_rate"], rates["l3_rate"]
                )
                t = _sum_cost(rows)
                bt = _sum_batch_cost(rows)
                lines.append(
                    f"| {label} | {pname} | {sname} | "
                    f"{rates['l2_rate']:.0%} | {rates['l3_rate']:.0%} | "
                    f"{_money(t)} | {_money(bt)} |"
                )
    lines.append("")

    # Missing-price flag — surface any profile model that the price book lacks.
    missing = sorted({
        m
        for profile in PROFILES.values()
        for m in [*profile["l1"], *profile["l2"], profile["l3"]]
        if not book.has(m)
    })
    if missing:
        lines += [
            f"> ⚠️  **Missing prices** for: " + ", ".join(f"`{m}`" for m in missing) + ". "
            f"Cost cells render as `n/a`. Add entries to `configs/model_prices.json` and re-run.",
            "",
        ]

    # Cold vs warm cache
    lines += [
        "## Cold-run vs cache-warm",
        "",
        "MAP results are cached in `PipelineCache` keyed by chunk text + cascade "
        "metadata. A re-run of the *same paper with unchanged sentences and "
        "unchanged cascade config* hits the cache for every chunk → ~$0 LLM cost. "
        "The numbers in the tables above are **cold-run** estimates. Warm-run "
        "cost ≈ 0 (cache hit ratio observed = 100 % on the only existing trace).",
        "",
    ]

    # Assumptions
    lines += [
        "## Assumptions & uncertainty",
        "",
        "- **Per-chunk tokens** taken from the single observed L3 call "
        f"on PMC10047158: input={OBS_INPUT_TOK_PER_CHUNK}, "
        f"output={OBS_OUTPUT_TOK_PER_CHUNK}. Applied uniformly to every voter "
        "at every tier (same prompt; same structured-output schema). "
        "Real L1/L2 outputs are typically smaller — these estimates are "
        "**conservative / upper-bound** for the cheap voters.",
        f"- **Sentence count** estimated as words × {OBS_SENT_PER_WORD:.4f} "
        "from the PMC10047158 ratio (78 sentences / 1335 words). A 25 % swing "
        "either way is plausible — multiply final cost by 0.75–1.25 for "
        "a sensitivity band.",
        f"- **Chunk count** = `ceil(n_sentences / {STRIDE})` (chunk_size="
        f"{CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).",
        "- **Cascade rates** are scenario inputs, not measurements. Only one "
        "trace exists (PMC10047158: 0 / 8 stopped at L2, 1 / 8 reached L3). "
        "Expected-case values (20 % / 10 %) are best-guess defaults.",
        "  - **TODO — recalibrate after ≥3–5 real runs.** Once `out/summaries/"
        "reports/escalation_report_*.json` covers more papers, replace the "
        "`expected` row with the observed escalation fractions "
        "(`l2_escalated / total_chunks`, `l3_escalated / total_chunks`) "
        "averaged across runs. Edit `SCENARIOS['expected']` in this script.",
        "- **Prices** loaded verbatim from `configs/model_prices.json`. "
        "Models present: " + ", ".join(book.known_models()) + ".",
        "- **Cache** assumed cold for the tables; the warm-run section reports "
        "the trivial-cost case separately.",
        "- **Batch discount** = 0.5× across all providers, taken from the price "
        "book. Provider-specific batch availability is not verified here.",
        "- **Non-MAP stages**: GROUNDING and RELATE use a local NLI model — "
        "no API tokens billed, but GPU / wall-time cost is non-zero (out of "
        "scope). UMLS-driven stages issue no LLM calls.",
        "",
    ]

    out_md = "\n".join(lines) + "\n"
    out_path = Path("out") / "cost_percentile_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_md, encoding="utf-8")
    print(out_md)
    print(f"# wrote {out_path}")


if __name__ == "__main__":
    main()
