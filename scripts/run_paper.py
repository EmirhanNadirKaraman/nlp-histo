"""
Process one or more histopathology papers through the summarization pipeline.

Usage
-----
python scripts/run_paper.py PMC1234567
python scripts/run_paper.py PMC1234567 --trace      # enable JSONL traces
python scripts/run_paper.py PMC1234567 --dry-run    # print config and exit
python scripts/run_paper.py PMC1234567 --batch      # async batch mode (50 % discount)

# Omit PMCID to auto-sample from eval/data/source_cases.jsonl:
python scripts/run_paper.py                         # 2 random PMCIDs, seed=42
python scripts/run_paper.py --sample 3 --seed 7    # 3 random PMCIDs, seed=7
python scripts/run_paper.py --all                   # all PMCIDs in source_cases.jsonl
python scripts/run_paper.py --all --batch           # all papers, batch mode

Requires: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    from pipeline.stages.summarization.agreement import GeminiEmbedder
    from pipeline.stages.summarization.config import SummarizationConfig
    from pipeline.stages.summarization.llm_providers import (
        anthropic_direct_chat,
        gemini_direct_chat,
    )

    voter_llms = [                                          # Level 1 — cheapest
        gemini_direct_chat("gemini-2.5-flash-lite",        temperature=0.1),
        gemini_direct_chat("gemini-2.5-flash-lite",        temperature=0.3),
        anthropic_direct_chat("claude-haiku-4-5-20251001", temperature=0.1),
    ]

    level2_voter_llms = [                                  # Level 2 — mid-tier
        gemini_direct_chat("gemini-2.5-flash",             temperature=0.1),
        gemini_direct_chat("gemini-2.5-flash",             temperature=0.3),
        anthropic_direct_chat("claude-haiku-4-5-20251001", temperature=0.3),
    ]

    escalation_llm = anthropic_direct_chat(                # Level 3 — strongest
        "claude-sonnet-4-6", temperature=0.0,
    )

    return SummarizationRunner(
        voter_llms=voter_llms,
        level2_voter_llms=level2_voter_llms,
        escalation_llm=escalation_llm,
        embed_fn=GeminiEmbedder(),
        config=SummarizationConfig(),
        output_dir=Path("out/summaries"),
        trace_enabled=trace,
    )


# ── Batch runner ───────────────────────────────────────────────────────────────

def build_batch_runner():
    from pipeline.stages.summarization.batch import BatchSummarizationRunner, VoterBatchConfig
    from pipeline.stages.summarization.config import SummarizationConfig
    from pipeline.stages.summarization.llm_providers import anthropic_direct_chat

    l1_voters = [
        VoterBatchConfig("gemini-2.5-flash-lite",      provider="gemini", temperature=0.1),
        VoterBatchConfig("gemini-2.5-flash-lite",      provider="gemini", temperature=0.3),
        # VoterBatchConfig("gpt-4o-mini",              provider="openai"),  # insufficient quota
        VoterBatchConfig("claude-haiku-4-5-20251001",  provider="claude", temperature=0.1),
    ]
    l2_voters = [
        VoterBatchConfig("gemini-2.5-flash",           provider="gemini", temperature=0.1),
        VoterBatchConfig("gemini-2.5-flash",           provider="gemini", temperature=0.3),
        # VoterBatchConfig("gpt-4.1-mini",             provider="openai"),  # insufficient quota
        VoterBatchConfig("claude-haiku-4-5-20251001",  provider="claude", temperature=0.3),
    ]
    l3_model = VoterBatchConfig("claude-sonnet-4-6", provider="claude")

    # REDUCE + RULES still run synchronously (one call per paper at the end)
    escalation_llm = anthropic_direct_chat("claude-sonnet-4-6", temperature=0.0)

    return BatchSummarizationRunner(
        l1_voters=l1_voters,
        l2_voters=l2_voters,
        l3_model=l3_model,
        escalation_llm=escalation_llm,
        config=SummarizationConfig(),
        output_dir=Path("out/summaries"),
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def _all_pmcids() -> list[str]:
    """Return all unique PMCIDs from eval/data/source_cases.jsonl in file order."""
    import json
    source = Path(__file__).parent.parent / "eval" / "data" / "source_cases.jsonl"
    if not source.exists():
        logger.error("source_cases.jsonl not found at %s", source)
        sys.exit(1)
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
    logger.info("Found %d unique PMCIDs in source_cases.jsonl", len(pmcids))
    return pmcids


def _sample_pmcids(n: int, seed: int) -> list[str]:
    """Pick n unique PMCIDs from eval/data/source_cases.jsonl using the given seed."""
    import random
    pmcids = _all_pmcids()
    rng = random.Random(seed)
    sample = rng.sample(pmcids, min(n, len(pmcids)))
    logger.info("Auto-sampled %d PMCIDs (seed=%d): %s", len(sample), seed, sample)
    return sample


def main():
    parser = argparse.ArgumentParser(description="Run summarization pipeline on one or more papers.")
    parser.add_argument("pmcid",          nargs="?", default=None,
                        help="PubMed Central ID, e.g. PMC1234567. "
                             "Omit to auto-sample from eval/data/source_cases.jsonl.")
    parser.add_argument("--all",           action="store_true",
                        help="Run all PMCIDs from eval/data/source_cases.jsonl")
    parser.add_argument("--sample",       type=int, default=2, metavar="N",
                        help="Number of PMCIDs to sample when pmcid is omitted (default: 2)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for PMCID sampling (default: 42)")
    parser.add_argument("--trace",        action="store_true", help="Write JSONL traces (sync mode only)")
    parser.add_argument("--batch",        action="store_true", help="Async batch mode (50 %% discount)")
    parser.add_argument("--dry-run",      action="store_true", help="Print config and exit without API calls")
    parser.add_argument("--limit-chunks", type=int, default=None, metavar="N",
                        help="Process at most N MAP chunks (useful for cheap iteration)")
    parser.add_argument("--start-chunk",  type=int, default=0, metavar="K",
                        help="Start processing from chunk K (0-based, default 0)")
    args = parser.parse_args()

    if args.all:
        pmcids = _all_pmcids()
    elif args.pmcid is None:
        pmcids = _sample_pmcids(args.sample, args.seed)
    else:
        pmcid = args.pmcid.strip()
        if not pmcid.startswith("PMC"):
            pmcid = "PMC" + pmcid
        pmcids = [pmcid]

    if args.dry_run:
        mode = "batch" if args.batch else "sync"
        print(f"PMCIDs: {pmcids}")
        print(f"Mode:   {mode}")
        print(f"Trace:  {args.trace} (sync only)")
        print("Models:")
        print("  L1  gemini-2.5-flash-lite (t=0.1)   (Google)")
        print("  L1  gemini-2.5-flash-lite (t=0.3)   (Google)")
        print("  L1  claude-haiku-4-5-20251001        (Anthropic)")
        print("  L2  gemini-2.5-flash (t=0.1)         (Google)")
        print("  L2  gemini-2.5-flash (t=0.3)         (Google)")
        print("  L2  claude-haiku-4-5-20251001        (Anthropic)")
        print("  L3  claude-sonnet-4-6                (Anthropic)")
        print("\nEnv vars required: GOOGLE_API_KEY, ANTHROPIC_API_KEY")
        return

    if args.batch and len(pmcids) > 1:
        _run_all_batch(pmcids)
    else:
        for pmcid in pmcids:
            logger.info("─── Processing %s (%d/%d) ───", pmcid, pmcids.index(pmcid) + 1, len(pmcids))
            if args.batch:
                _run_batch(pmcid)
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


def _run_all_batch(pmcids: list[str], poll_interval: int = 120) -> None:
    """Submit all papers simultaneously, poll together, finalize all."""
    from pipeline.stages.summarization.batch import BatchPhase
    from pipeline.stages.summarization.runner import SummarizationRunner

    runner = build_batch_runner()

    # ── Phase 1: submit all papers in parallel ────────────────────────────────
    def _submit(pmcid: str):
        try:
            file_data = SummarizationRunner.load_paper_from_db(pmcid)
        except ValueError as exc:
            logger.error("[%s] Load failed: %s", pmcid, exc)
            return pmcid, None
        handle = runner.load_or_submit(file_data)
        return pmcid, handle

    handles: dict[str, object] = {}
    logger.info("Submitting %d papers in parallel…", len(pmcids))
    with ThreadPoolExecutor(max_workers=len(pmcids)) as ex:
        futures = {ex.submit(_submit, p): p for p in pmcids}
        for fut in as_completed(futures):
            pmcid, handle = fut.result()
            if handle is not None:
                handles[pmcid] = handle
                logger.info("[%s] Submitted (phase=%s)", pmcid, handle.phase.value)

    if not handles:
        logger.error("No papers submitted successfully.")
        return

    # ── Phase 2: poll all handles until every one is COMPLETE ─────────────────
    while True:
        pending = [p for p, h in handles.items() if h.phase != BatchPhase.COMPLETE]
        if not pending:
            break
        logger.info("%d / %d papers still pending — sleeping %ds…",
                    len(pending), len(handles), poll_interval)
        time.sleep(poll_interval)
        for pmcid in pending:
            handles[pmcid] = runner.advance(handles[pmcid])
            logger.info("[%s] phase=%s", pmcid, handles[pmcid].phase.value)

    # ── Phase 3: finalize all ─────────────────────────────────────────────────
    logger.info("All papers complete — finalising…")
    for pmcid, handle in handles.items():
        result = runner.finalize(handle)
        rules = result.get("rules", [])
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
