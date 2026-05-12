"""Count tokens and project ABC-cascade cost for a paper-selection YAML.

Reads the YAML produced by ``eval.paper_selection.run_select``, loads each
paper's sentences from the database (the exact same path the MAP stage uses
via ``SummarizationRunner.load_paper_from_db``), computes chunk counts at the
production MapConfig defaults, tokenises every sentence with the
``cl100k_base`` tokenizer, and prints:

  - per-paper table:  pmcid | bucket | n_sentences | n_chunks | n_tokens
  - totals
  - projected MAP cost at three cascade-escalation scenarios

Usage
-----
    PYTHONPATH=. python scripts/estimate_selection_cost.py \\
        --from-selection configs/paper_selection/smoke_v1.yaml

Or pass PMCIDs explicitly:

    PYTHONPATH=. python scripts/estimate_selection_cost.py PMC10047158 PMC222

Notes
-----
- Token counts use ``tiktoken`` (cl100k_base). Anthropic's tokenizer differs
  by ~10 %, fine for budget estimates.
- Cost projections cover the MAP stage only. CANONICALIZE/RESOLVE add a
  smaller, more variable amount (typically ~10–20 % of MAP); see the
  printed footer.
- Escalation rates are coarse guesses. Override with ``--l2-rate`` and
  ``--l3-rate`` once you have real data from a canary run.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("estimate_selection_cost")


# ── MAP defaults (mirror pipeline/stages/summarization/config.py:MapConfig) ─

DEFAULT_CHUNK_SIZE = 10
DEFAULT_CHUNK_OVERLAP = 2
# Prompt overhead per MAP call (system + JSON schema + voter instructions).
# Conservative middle estimate; real prompts are 1.2–1.8 K tokens.
PROMPT_OVERHEAD_TOKENS = 1500
# Output JSON per chunk (AuditableSummary). Realistic 500–1200; use 800.
OUTPUT_TOKENS_PER_CHUNK = 800


# ── Cost matrix (per million tokens) ────────────────────────────────────────
# Mirrors the production cascade defined in
# pipeline/stages/summarization/batch/voter_configs.py:make_l{1,2,3}_voters().
# Per-voter pricing summed at each tier (not averaged) so the projection
# matches what ``--profile default`` will actually charge.
#
# Verify against the provider price pages before quoting these figures:
#   Anthropic:  https://www.anthropic.com/pricing
#   OpenAI:     https://openai.com/api/pricing/
#   Google:     https://cloud.google.com/vertex-ai/generative-ai/pricing

@dataclass
class VoterPricing:
    name: str
    input_per_mtok: float
    output_per_mtok: float


@dataclass
class TierPricing:
    voters: list[VoterPricing]

    @property
    def n_voters(self) -> int:
        return len(self.voters)

    @property
    def total_input_per_mtok(self) -> float:
        """Sum of input prices across all voters in this tier."""
        return sum(v.input_per_mtok for v in self.voters)

    @property
    def total_output_per_mtok(self) -> float:
        """Sum of output prices across all voters in this tier."""
        return sum(v.output_per_mtok for v in self.voters)


PRICING = {
    "L1": TierPricing(voters=[
        VoterPricing("gemini-2.5-flash-lite",       0.10,  0.40),
        VoterPricing("gpt-4o-mini",                 0.15,  0.60),
        VoterPricing("claude-haiku-4-5-20251001",   1.00,  5.00),
    ]),
    "L2": TierPricing(voters=[
        VoterPricing("gemini-2.5-flash",            0.30,  2.50),
        VoterPricing("gpt-4.1-mini",                0.40,  1.60),
        VoterPricing("claude-haiku-4-5-20251001",   1.00,  5.00),
    ]),
    "L3": TierPricing(voters=[
        VoterPricing("claude-sonnet-4-6-20251001",  3.00, 15.00),
    ]),
}


# ── YAML loader ─────────────────────────────────────────────────────────────

def load_selection_yaml(path: Path) -> dict[str, str]:
    """Return ``{pmcid: bucket}`` in related → diverse → hard order, deduped."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a top-level mapping with a version key")
    version_key = next(iter(data))
    buckets = data[version_key] or {}
    if not isinstance(buckets, dict):
        raise ValueError(f"{path}: version '{version_key}' is not a mapping")

    out: dict[str, str] = {}
    for bucket in ("related", "diverse", "hard"):
        for pmcid in buckets.get(bucket) or []:
            out.setdefault(pmcid, bucket)
    if not out:
        raise ValueError(f"{path}: no PMCIDs found under related/diverse/hard")
    return out


# ── Chunk + token math ──────────────────────────────────────────────────────

def count_chunks(n_sentences: int, chunk_size: int, chunk_overlap: int) -> int:
    """Mirror MapStage._make_chunks chunk count exactly."""
    if n_sentences <= 0:
        return 0
    stride = chunk_size - chunk_overlap
    return len(list(range(0, n_sentences, stride)))


def make_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(tokenizer, sentences: list[str]) -> int:
    return sum(len(tokenizer.encode(s)) for s in sentences)


def per_chunk_input_tokens(text_tokens: int, n_chunks: int,
                           chunk_size: int, n_sentences: int) -> int:
    """Approximate average input tokens for a single MAP call.

    Each chunk feeds ``chunk_size`` sentences (or fewer for the tail). The
    average sentences-per-chunk is ``n_sentences / n_chunks`` (which exceeds
    ``chunk_size`` slightly because of overlap counting sentences twice).
    """
    if n_chunks == 0 or n_sentences == 0:
        return PROMPT_OVERHEAD_TOKENS
    avg_sentences_per_chunk = min(chunk_size, n_sentences / n_chunks * (1 + 0))
    # Effective chunk-text tokens = total tokens × (avg_sentences / n_sentences)
    # but we want per-chunk, so it's text_tokens * avg / n_sentences:
    chunk_text_tokens = text_tokens * avg_sentences_per_chunk / n_sentences
    return PROMPT_OVERHEAD_TOKENS + int(round(chunk_text_tokens))


# ── Paper loading ───────────────────────────────────────────────────────────

@dataclass
class PaperStats:
    pmcid: str
    bucket: str
    n_text_elements: int
    n_sentences: int
    n_chunks: int
    text_tokens: int            # tokens across all sentence text
    avg_chunk_in_tokens: int    # mean input tokens per MAP call


def load_paper_stats(pmcid: str, bucket: str, *,
                     tokenizer, chunk_size: int, chunk_overlap: int,
                     nlp) -> PaperStats | None:
    """Use the same DB path MAP uses (TextElement → spaCy sentencize)."""
    import time as _time
    from database import get_db_connection, Document, TextElement

    t0 = _time.perf_counter()
    db = get_db_connection()
    with db.session_scope() as session:
        doc = session.query(Document).filter_by(pmcid=pmcid).first()
        if doc is None:
            logger.warning("PMCID %s not in database — skipping", pmcid)
            return None
        rows = (
            session.query(TextElement)
            .filter_by(document_id=doc.id)
            .order_by(TextElement.position_in_section)
            .all()
        )
        n_text_elements = len(rows)
        logger.debug("[%s] DB fetched %d text elements in %.2fs",
                     pmcid, n_text_elements, _time.perf_counter() - t0)

        t_sent = _time.perf_counter()
        sentences: list[str] = []
        texts = [te.text_content for te in rows if te.text_content]
        for doc_obj in nlp.pipe(texts, batch_size=64):
            for sent in doc_obj.sents:
                text = sent.text.strip()
                if text:
                    sentences.append(text)
        logger.debug("[%s] sentencized %d paragraphs → %d sentences in %.2fs",
                     pmcid, len(texts), len(sentences),
                     _time.perf_counter() - t_sent)

    t_tok = _time.perf_counter()
    n_sentences = len(sentences)
    n_chunks = count_chunks(n_sentences, chunk_size, chunk_overlap)
    text_tokens = count_tokens(tokenizer, sentences)
    avg_in = per_chunk_input_tokens(text_tokens, n_chunks, chunk_size, n_sentences)
    logger.debug("[%s] tokenized in %.2fs (%d tokens)",
                 pmcid, _time.perf_counter() - t_tok, text_tokens)

    logger.info("[%s] %s: %d sentences, %d chunks, %d tokens (%.2fs)",
                pmcid, bucket, n_sentences, n_chunks, text_tokens,
                _time.perf_counter() - t0)

    return PaperStats(
        pmcid=pmcid, bucket=bucket,
        n_text_elements=n_text_elements,
        n_sentences=n_sentences, n_chunks=n_chunks,
        text_tokens=text_tokens, avg_chunk_in_tokens=avg_in,
    )


# ── Cost projection ─────────────────────────────────────────────────────────

@dataclass
class Scenario:
    label: str
    l2_rate: float   # fraction of chunks escalated to L2
    l3_rate: float   # fraction of chunks escalated to L3


def project_map_cost(total_chunks: int, avg_in_tokens: int,
                     l2_rate: float, l3_rate: float) -> dict[str, float]:
    """Return per-tier and total MAP cost in USD.

    Each tier sums per-voter costs: one chunk → ``len(tier.voters)`` API
    calls, each charged at that voter's own input/output price. No averaging.
    """
    out_tokens = OUTPUT_TOKENS_PER_CHUNK
    cost = {"L1": 0.0, "L2": 0.0, "L3": 0.0}

    def tier_cost(tier: str, n_chunks: float) -> float:
        p = PRICING[tier]
        # Per chunk: each voter makes one call with the same input/output sizes.
        in_cost  = n_chunks * avg_in_tokens / 1_000_000 * p.total_input_per_mtok
        out_cost = n_chunks * out_tokens     / 1_000_000 * p.total_output_per_mtok
        return in_cost + out_cost

    cost["L1"] = tier_cost("L1", total_chunks)
    cost["L2"] = tier_cost("L2", total_chunks * l2_rate)
    cost["L3"] = tier_cost("L3", total_chunks * l3_rate)
    cost["total"] = cost["L1"] + cost["L2"] + cost["L3"]
    return cost


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pmcid", nargs="*",
                        help="Explicit PMCIDs (alternative to --from-selection)")
    parser.add_argument("--from-selection", metavar="PATH",
                        help="Path to a paper-selection YAML")
    parser.add_argument("--chunk-size",    type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--l2-rate", type=float, action="append", default=None,
                        metavar="FRAC",
                        help="L2 escalation fraction for one scenario row "
                             "(repeatable; pair with --l3-rate). "
                             "If unset, three baked-in scenarios are shown.")
    parser.add_argument("--l3-rate", type=float, action="append", default=None,
                        metavar="FRAC", help="L3 escalation fraction "
                                              "(repeatable; matched to --l2-rate).")
    args = parser.parse_args()

    if args.from_selection and args.pmcid:
        parser.error("pass either positional PMCIDs OR --from-selection, not both")

    if args.from_selection:
        sel_map = load_selection_yaml(Path(args.from_selection))
    elif args.pmcid:
        sel_map = {p: "explicit" for p in args.pmcid}
    else:
        parser.print_help()
        return 1

    if args.l2_rate or args.l3_rate:
        l2 = args.l2_rate or []
        l3 = args.l3_rate or []
        if len(l2) != len(l3):
            parser.error("--l2-rate and --l3-rate must be paired (same count)")
        scenarios = [Scenario(f"custom_{i+1}", l2[i], l3[i]) for i in range(len(l2))]
    else:
        scenarios = [
            Scenario("optimistic", l2_rate=0.10, l3_rate=0.02),
            Scenario("middle",     l2_rate=0.25, l3_rate=0.08),
            Scenario("pessimistic", l2_rate=0.50, l3_rate=0.20),
        ]

    logger.info("Loading spaCy en_core_sci_sm (sentencizer only)…")
    import spacy
    nlp = spacy.load(
        "en_core_sci_sm",
        disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    logger.info("Loading tiktoken cl100k_base…")
    tokenizer = make_tokenizer()

    logger.info("Loading %d papers from DB…", len(sel_map))
    import time as _time
    t_all = _time.perf_counter()
    stats: list[PaperStats] = []
    for i, (pmcid, bucket) in enumerate(sel_map.items(), start=1):
        logger.info("(%d/%d) loading %s [%s]…", i, len(sel_map), pmcid, bucket)
        s = load_paper_stats(
            pmcid, bucket,
            tokenizer=tokenizer,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            nlp=nlp,
        )
        if s is not None:
            stats.append(s)
    logger.info("Loaded %d/%d papers in %.1fs", len(stats), len(sel_map),
                _time.perf_counter() - t_all)

    if not stats:
        logger.error("No papers loaded — aborting")
        return 2

    # ── Per-paper table ────────────────────────────────────────────────────
    print()
    print(f"{'pmcid':<48} {'bucket':<10} {'tx_el':>6} {'sents':>6} "
          f"{'chunks':>7} {'tokens':>9} {'in/chunk':>9}")
    print("-" * 100)
    for s in stats:
        print(f"{s.pmcid:<48} {s.bucket:<10} {s.n_text_elements:>6d} "
              f"{s.n_sentences:>6d} {s.n_chunks:>7d} {s.text_tokens:>9d} "
              f"{s.avg_chunk_in_tokens:>9d}")
    print("-" * 100)

    total_chunks = sum(s.n_chunks for s in stats)
    total_tokens = sum(s.text_tokens for s in stats)
    total_sents  = sum(s.n_sentences for s in stats)
    weighted_avg_in = (
        sum(s.avg_chunk_in_tokens * s.n_chunks for s in stats) / max(total_chunks, 1)
    )
    print(f"{'TOTAL':<48} {'':<10} {'':>6} {total_sents:>6d} "
          f"{total_chunks:>7d} {total_tokens:>9d} "
          f"{int(round(weighted_avg_in)):>9d}")

    # ── MAP cost projection ────────────────────────────────────────────────
    print()
    print(f"MAP cost projection ({total_chunks} chunks, "
          f"avg {int(round(weighted_avg_in))} input tokens, "
          f"{OUTPUT_TOKENS_PER_CHUNK} output tokens/chunk)")
    print()
    print(f"{'scenario':<14} {'L2 esc':>7} {'L3 esc':>7} "
          f"{'L1 $':>8} {'L2 $':>8} {'L3 $':>8} {'TOTAL $':>10}")
    print("-" * 70)
    for sc in scenarios:
        c = project_map_cost(total_chunks, int(round(weighted_avg_in)),
                             sc.l2_rate, sc.l3_rate)
        print(f"{sc.label:<14} {sc.l2_rate:>6.0%} {sc.l3_rate:>7.0%} "
              f"{c['L1']:>8.2f} {c['L2']:>8.2f} {c['L3']:>8.2f} "
              f"{c['total']:>10.2f}")
    print()
    print("MAP only. CANONICALIZE + RESOLVE typically add ~10–20 % "
          "(Sonnet calls on dense subsets of findings).")
    print("Cascade configuration (production profile = "
          "voter_configs.make_l{1,2,3}_voters):")
    for tier, p in PRICING.items():
        print(f"  {tier} ({p.n_voters} voter{'s' if p.n_voters != 1 else ''}):")
        for v in p.voters:
            print(f"    {v.name:<36} ${v.input_per_mtok:>5.2f} in / "
                  f"${v.output_per_mtok:>5.2f} out  per MTok")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
