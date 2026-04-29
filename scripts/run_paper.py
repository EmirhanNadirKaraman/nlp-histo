"""
Process one or more histopathology papers through the summarization pipeline.

Usage
-----
python scripts/run_paper.py PMC1234567
python scripts/run_paper.py PMC1234567 --trace      # enable JSONL traces
python scripts/run_paper.py PMC1234567 --dry-run    # print config and exit
python scripts/run_paper.py PMC1234567 --batch      # async batch mode (50 % discount)

# Omit PMCID to auto-sample from eval/data/source_cases.jsonl:
python scripts/run_paper.py                         # 5 random PMCIDs, seed=42
python scripts/run_paper.py --sample 3 --seed 7    # 3 random PMCIDs, seed=7

Batch mode
----------
Run ``--batch`` repeatedly.  On the first call the L1 jobs are submitted and
the state is saved to ``out/summaries/batch_handles/PMC1234567.batch.json``.
Subsequent calls poll for completion and advance the cascade automatically.
When all MAP batches are done the REDUCE and RULE stages run synchronously and
the final result is saved to ``out/summaries/summaries/PMC1234567.json``.

Batch mode requires ANTHROPIC_API_KEY for the Claude models (Haiku L2, Sonnet
L3).  Azure and Vertex Gemini credentials are the same as sync mode.
Vertex Gemini batch additionally requires VERTEX_BATCH_GCS_BUCKET.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_paper")


# ── Sync runner ────────────────────────────────────────────────────────────────

def build_runner(trace: bool):
    from pipeline.stages.summarization import SummarizationRunner
    from pipeline.stages.summarization.llm_providers import (
        azure_foundry_chat,
        claude_vertex_chat,
        vertex_gemini_chat,
    )

    voter_llms = [                                          # Level 1 — cheapest
        azure_foundry_chat("DeepSeek-V3.2-Speciale",  temperature=0.1, strip_thinking=True),
        vertex_gemini_chat("gemini-2.5-flash-lite",   temperature=0.1, location="global"),
        azure_foundry_chat("Mistral-Large-3",          temperature=0.1),
    ]

    level2_voter_llms = [                                  # Level 2 — mid-tier
        vertex_gemini_chat("gemini-2.5-flash",         temperature=0.1),
        azure_foundry_chat("Kimi-K2.5",                temperature=0.1),
        claude_vertex_chat("claude-haiku-4-5@20251001", temperature=0.1),
    ]

    escalation_llm = claude_vertex_chat(                   # Level 3 — strongest
        "claude-sonnet-4-6@default", temperature=0.0
    )

    return SummarizationRunner(
        voter_llms=voter_llms,
        level2_voter_llms=level2_voter_llms,
        escalation_llm=escalation_llm,
        theta=0.7,
        chunk_size=10,
        output_dir=Path("out/summaries"),
        trace_enabled=trace,
    )


def build_direct_runner(trace: bool):
    """Sync runner using direct Gemini / OpenAI / Anthropic APIs (no Azure/Vertex)."""
    from pipeline.stages.summarization import SummarizationRunner
    from pipeline.stages.summarization.llm_providers import (
        gemini_direct_chat,
        openai_direct_chat,
        anthropic_direct_chat,
    )

    voter_llms = [                                          # Level 1 — cheapest
        gemini_direct_chat("gemini-2.5-flash-lite",                temperature=0.1),
        openai_direct_chat("gpt-4o-mini",                         temperature=0.1),
        openai_direct_chat("gpt-4.1-mini",                        temperature=0.1),
    ]

    level2_voter_llms = [                                  # Level 2 — mid-tier
        gemini_direct_chat("gemini-2.5-flash",            temperature=0.1),
        openai_direct_chat("gpt-4o",                      temperature=0.1),
        anthropic_direct_chat("claude-haiku-4-5-20251001", temperature=0.1),
    ]

    escalation_llm = anthropic_direct_chat(                # Level 3 — strongest
        "claude-sonnet-4-6", temperature=0.0
    )

    return SummarizationRunner(
        voter_llms=voter_llms,
        level2_voter_llms=level2_voter_llms,
        escalation_llm=escalation_llm,
        theta=0.7,
        chunk_size=10,
        output_dir=Path("out/summaries"),
        trace_enabled=trace,
    )


# ── Batch runner ───────────────────────────────────────────────────────────────

def build_batch_runner():
    from pipeline.stages.summarization.batch import BatchSummarizationRunner, VoterBatchConfig
    from pipeline.stages.summarization.llm_providers import claude_vertex_chat

    l1_voters = [
        VoterBatchConfig("DeepSeek-V3.2-Speciale", provider="azure",        strip_thinking=True),
        VoterBatchConfig("gemini-2.5-flash-lite",  provider="vertex_gemini"),
        VoterBatchConfig("Mistral-Large-3",         provider="azure"),
    ]
    l2_voters = [
        VoterBatchConfig("gemini-2.5-flash",       provider="vertex_gemini"),
        VoterBatchConfig("Kimi-K2.5",              provider="azure"),
        VoterBatchConfig("claude-haiku-4-5@20251001", provider="claude"),
    ]
    l3_model = VoterBatchConfig("claude-sonnet-4-6@default", provider="claude")

    # REDUCE + RULES still run synchronously (one call per paper at the end)
    escalation_llm = claude_vertex_chat("claude-sonnet-4-6@default", temperature=0.0)

    return BatchSummarizationRunner(
        l1_voters=l1_voters,
        l2_voters=l2_voters,
        l3_model=l3_model,
        escalation_llm=escalation_llm,
        theta=0.7,
        chunk_size=10,
        output_dir=Path("out/summaries"),
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def _sample_pmcids(n: int, seed: int) -> list[str]:
    """Pick n unique PMCIDs from eval/data/source_cases.jsonl using the given seed."""
    import random
    source = Path(__file__).parent.parent / "eval" / "data" / "source_cases.jsonl"
    if not source.exists():
        logger.error("source_cases.jsonl not found at %s — cannot auto-sample", source)
        sys.exit(1)
    import json
    pmcids: list[str] = []
    seen: set[str] = set()
    with source.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pmcid = json.loads(line).get("pmcid", "")
            if pmcid and pmcid not in seen:
                seen.add(pmcid)
                pmcids.append(pmcid)
    rng = random.Random(seed)
    sample = rng.sample(pmcids, min(n, len(pmcids)))
    logger.info("Auto-sampled %d PMCIDs (seed=%d): %s", len(sample), seed, sample)
    return sample


def main():
    parser = argparse.ArgumentParser(description="Run summarization pipeline on one or more papers.")
    parser.add_argument("pmcid",          nargs="?", default=None,
                        help="PubMed Central ID, e.g. PMC1234567. "
                             "Omit to auto-sample from eval/data/source_cases.jsonl.")
    parser.add_argument("--sample",       type=int, default=5, metavar="N",
                        help="Number of PMCIDs to sample when pmcid is omitted (default: 5)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for PMCID sampling (default: 42)")
    parser.add_argument("--trace",        action="store_true", help="Write JSONL traces (sync mode only)")
    parser.add_argument("--batch",        action="store_true", help="Async batch mode (50 %% discount)")
    parser.add_argument("--direct",       action="store_true", help="Use direct Gemini/OpenAI/Anthropic APIs instead of Azure/Vertex")
    parser.add_argument("--dry-run",      action="store_true", help="Print config and exit without API calls")
    parser.add_argument("--limit-chunks", type=int, default=None, metavar="N",
                        help="Process at most N MAP chunks (useful for cheap iteration)")
    parser.add_argument("--start-chunk",  type=int, default=0, metavar="K",
                        help="Start processing from chunk K (0-based, default 0)")
    args = parser.parse_args()

    if args.pmcid is None:
        pmcids = _sample_pmcids(args.sample, args.seed)
    else:
        pmcid = args.pmcid.strip()
        if not pmcid.startswith("PMC"):
            pmcid = "PMC" + pmcid
        pmcids = [pmcid]

    if args.dry_run:
        mode = "batch" if args.batch else ("direct" if args.direct else "sync")
        print(f"PMCIDs: {pmcids}")
        print(f"Mode:   {mode}")
        print(f"Trace:  {args.trace} (sync only)")
        if args.direct:
            print("Models (direct APIs):")
            print("  L1  gemini-2.5-flash-lite                (Google Gemini API)")
            print("  L1  gpt-4o-mini                           (OpenAI)")
            print("  L1  gpt-4.1-mini                          (OpenAI)")
            print("  L2  gemini-2.5-flash                      (Google Gemini API)")
            print("  L2  gpt-4o                                 (OpenAI)")
            print("  L2  claude-haiku-4-5-20251001             (Anthropic)")
            print("  L3  claude-sonnet-4-6                     (Anthropic)")
            print("\nEnv vars required: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
        else:
            print("Models:")
            print("  L1  DeepSeek-V3.2-Speciale    (Azure) [strip_thinking]")
            print("  L1  gemini-2.5-flash-lite      (Vertex, global)")
            print("  L1  Mistral-Large-3             (Azure)")
            print("  L2  gemini-2.5-flash            (Vertex)")
            print("  L2  Kimi-K2.5                   (Azure)")
            print("  L2  claude-haiku-4-5@20251001   (Vertex Claude / Anthropic batch)")
            print("  L3  claude-sonnet-4-6@default   (Vertex Claude / Anthropic batch)")
            if args.batch:
                print("\nBatch env vars required:")
                print("  ANTHROPIC_API_KEY           (Claude L2/L3 batch)")
                print("  VERTEX_BATCH_GCS_BUCKET     (Gemini L1/L2 batch, optional)")
        return

    for pmcid in pmcids:
        logger.info("─── Processing %s (%d/%d) ───", pmcid, pmcids.index(pmcid) + 1, len(pmcids))
        if args.batch:
            _run_batch(pmcid)
        elif args.direct:
            _run_direct(pmcid, trace=args.trace,
                        start_chunk=args.start_chunk, limit_chunks=args.limit_chunks)
        else:
            _run_sync(pmcid, trace=args.trace,
                      start_chunk=args.start_chunk, limit_chunks=args.limit_chunks)


def _run_sync(pmcid: str, trace: bool,
              start_chunk: int = 0, limit_chunks: int | None = None) -> None:
    from pipeline.stages.summarization import SummarizationRunner

    logger.info("Building sync runner…")
    runner = build_runner(trace=trace)

    logger.info("Loading paper %s from database…", pmcid)
    try:
        file_data = runner.load_paper_from_db(pmcid)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    n = len(file_data["sentences_with_provenance"])
    logger.info("Loaded %d sentences — starting pipeline", n)

    result = runner.process(file_data, start_chunk=start_chunk, limit_chunks=limit_chunks)

    if result["status"] == "error":
        logger.error("Pipeline failed: %s", result["error"])
        sys.exit(1)

    rules = result.get("rules", [])
    logger.info("Done — %d rules extracted", len(rules))
    logger.info("Result saved to out/summaries/summaries/%s.json", pmcid)

    print(f"\n{'─' * 60}")
    print(f"Summary for {pmcid}")
    print(f"{'─' * 60}")
    print(result.get("summary", "(no summary)"))
    print(f"\n{len(rules)} rules extracted.")


def _run_direct(pmcid: str, trace: bool,
                start_chunk: int = 0, limit_chunks: int | None = None) -> None:
    from pipeline.stages.summarization import SummarizationRunner

    logger.info("Building direct-API runner (Gemini / OpenAI / Anthropic)…")
    runner = build_direct_runner(trace=trace)

    logger.info("Loading paper %s from database…", pmcid)
    try:
        file_data = runner.load_paper_from_db(pmcid)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    n = len(file_data["sentences_with_provenance"])
    logger.info("Loaded %d sentences — starting pipeline", n)

    result = runner.process(file_data, start_chunk=start_chunk, limit_chunks=limit_chunks)

    if result["status"] == "error":
        logger.error("Pipeline failed: %s", result["error"])
        sys.exit(1)

    rules = result.get("rules", [])
    logger.info("Done — %d rules extracted", len(rules))
    logger.info("Result saved to out/summaries/summaries/%s.json", pmcid)

    print(f"\n{'─' * 60}")
    print(f"Summary for {pmcid}")
    print(f"{'─' * 60}")
    print(result.get("summary", "(no summary)"))
    print(f"\n{len(rules)} rules extracted.")


def _run_batch(pmcid: str) -> None:
    from pipeline.stages.summarization.batch import BatchPhase
    from pipeline.stages.summarization.runner import SummarizationRunner

    logger.info("Building batch runner…")
    runner = build_batch_runner()

    logger.info("Loading paper %s from database…", pmcid)
    try:
        file_data = SummarizationRunner.load_paper_from_db(pmcid)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    handle = runner.load_or_submit(file_data)

    if handle.phase == BatchPhase.COMPLETE:
        logger.info("[%s] All batches already complete — finalising…", pmcid)
        result = runner.finalize(handle)
        rules = result.get("rules", [])
        print(f"\n{'─' * 60}")
        print(f"Summary for {pmcid}")
        print(f"{'─' * 60}")
        print(result.get("summary", "(no summary)"))
        print(f"\n{len(rules)} rules extracted.")
        return

    # Try to advance
    handle = runner.advance(handle)

    if handle.phase == BatchPhase.COMPLETE:
        logger.info("[%s] All batches complete — finalising…", pmcid)
        result = runner.finalize(handle)
        rules = result.get("rules", [])
        print(f"\n{'─' * 60}")
        print(f"Summary for {pmcid}")
        print(f"{'─' * 60}")
        print(result.get("summary", "(no summary)"))
        print(f"\n{len(rules)} rules extracted.")
    else:
        n_pending = sum(1 for j in handle.jobs if j.status not in ("completed", "failed"))
        print(f"\n[{pmcid}] Phase: {handle.phase.value}  |  {n_pending} job(s) still running.")
        print("Run this command again to check progress and advance when ready.")
        print(f"Handle: {runner.handle_path(pmcid)}")


if __name__ == "__main__":
    main()
