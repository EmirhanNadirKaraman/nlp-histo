"""
Process a single histopathology paper through the summarization pipeline.

Usage
-----
python scripts/run_paper.py PMC1234567
python scripts/run_paper.py PMC1234567 --trace      # enable JSONL traces
python scripts/run_paper.py PMC1234567 --dry-run    # print config and exit
python scripts/run_paper.py PMC1234567 --batch      # async batch mode (50 % discount)

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

def main():
    parser = argparse.ArgumentParser(description="Run summarization pipeline on one paper.")
    parser.add_argument("pmcid", help="PubMed Central ID, e.g. PMC1234567")
    parser.add_argument("--trace",   action="store_true", help="Write JSONL traces (sync mode only)")
    parser.add_argument("--batch",   action="store_true", help="Async batch mode (50 %% discount)")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without API calls")
    args = parser.parse_args()

    pmcid = args.pmcid.strip()
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid

    if args.dry_run:
        mode = "batch" if args.batch else "sync"
        print(f"PMCID:  {pmcid}")
        print(f"Mode:   {mode}")
        print(f"Trace:  {args.trace} (sync only)")
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

    if args.batch:
        _run_batch(pmcid)
    else:
        _run_sync(pmcid, trace=args.trace)


def _run_sync(pmcid: str, trace: bool) -> None:
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

    result = runner.process(file_data)

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
