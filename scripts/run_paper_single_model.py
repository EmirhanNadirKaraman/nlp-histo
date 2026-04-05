"""
Run the full summarization pipeline on one paper using a single LLM.

Bypasses ABC multi-model cascading by wiring the same model into all three
voter slots and setting theta=0.0, so Level-1 always "agrees" and escalation
never fires.  Useful for debugging Phases 1–6 without needing Azure or Claude
credentials.

Usage
-----
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158 --trace
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158 --skip-nli
    PYTHONPATH=. python scripts/run_paper_single_model.py PMC10047158 --chunks 3
    PYTHONPATH=. python scripts/run_paper_single_model.py --list-only

Options
-------
--trace       Write JSONL traces to out/summaries/traces/.
--skip-nli    Disable NLI grounding filter and RELATE stage (faster, no GPU needed).
--chunks N    Limit to the first N×10 sentences (default: all).
--no-canon    Skip CANONICALIZE LLM call; use deterministic (highest score) fallback.
--list-only   Print available PMCIDs from the database and exit.
"""
from __future__ import annotations

import argparse
import json
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
logger = logging.getLogger("run_single_model")


# ── LLM factory ────────────────────────────────────────────────────────────────

def build_llm():
    """Single Gemini 2.5 Flash Lite via ChatVertexAI (same as inspect_phase123_pipeline.py)."""
    import os
    from langchain_google_vertexai import ChatVertexAI

    return ChatVertexAI(
        model="gemini-2.5-flash-lite",
        project=os.environ["VERTEX_PROJECT"],
        location="global",
        temperature=0.1,
        request_timeout=20,   # fail fast: normal responses are 5-10s
    )


# ── Runner factory ─────────────────────────────────────────────────────────────

def build_runner(
    *,
    trace: bool,
    skip_nli: bool,
    no_canon: bool,
) -> "SummarizationRunner":
    from pipeline.stages.summarization import SummarizationRunner

    llm = build_llm()

    # theta=0.0 → first voter always "agrees" with itself; no cascade fires.
    # All three LLM slots receive the same model instance.
    return SummarizationRunner(
        voter_llms=[llm],
        level2_voter_llms=[llm],
        escalation_llm=llm,
        theta=0.0,
        chunk_size=10,
        grounding_threshold=0.3 if not skip_nli else None,
        contradiction_similarity_threshold=None,   # skip pairwise contradiction detection
        canonicalize_with_llm=not no_canon,
        nli_entailment_threshold=0.50,
        nli_contradiction_threshold=0.50,
        output_dir=Path("out/summaries"),
        trace_enabled=trace,
    )


# ── DB helpers ─────────────────────────────────────────────────────────────────

def list_pmcids() -> None:
    from database import get_db_connection
    from database.models import Document

    db = get_db_connection()
    with db.session_scope() as session:
        pmcids = [
            row.pmcid
            for row in session.query(Document.pmcid).order_by(Document.pmcid).all()
        ]
    if not pmcids:
        print("No documents found in database.")
    else:
        print(f"{len(pmcids)} document(s) available:")
        for p in pmcids:
            print(f"  {p}")


# ── Result printer ─────────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    pmcid = result["pmcid"]
    rules = result.get("rules", [])
    final_rules = result.get("final_rules", [])
    canonical_rules = result.get("canonical_rules", [])
    relations = result.get("relations", [])

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  PMCID          : {pmcid}")
    print(f"  Rules (RULE)   : {len(rules)}")
    print(f"  Canonical rules: {len(canonical_rules)}")
    print(f"  Relations      : {len(relations)}")
    print(f"  Final rules    : {len(final_rules)}")
    print(sep)

    if result.get("summary"):
        print("\n── Narrative summary ──\n")
        print(result["summary"])

    if final_rules:
        print(f"\n── Top final rules (by score) ──\n")
        for fr in final_rules[:10]:
            flag = " ⚠ CONTRADICTED" if fr.get("is_contradicted") else ""
            print(
                f"  [{fr['final_score']:.3f}]{flag}  "
                f"{fr['subject_entity']} → {fr['outcome_entity']} "
                f"[{fr['relation_type']}] [{fr.get('direction') or '?'}]\n"
                f"    \"{fr['predicate_text'][:100]}\"\n"
                f"    scope={fr['canonical_scope']}  "
                f"pmcids={fr['supporting_pmcids']}\n"
            )

    if relations:
        print(f"\n── Relations ──\n")
        for rel in relations:
            print(
                f"  {rel['relation_type']:<15}  "
                f"{rel['rule_id_a']}  ↔  {rel['rule_id_b']}  "
                f"(A→B={rel['nli_score_a_to_b']:.2f}, B→A={rel['nli_score_b_to_a']:.2f})"
            )

    out_path = Path("out/summaries/summaries") / f"{pmcid}.json"
    print(f"\n  Full result → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline with a single Gemini 2.5 Flash Lite model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pmcid", nargs="?", help="PubMed Central ID, e.g. PMC10047158")
    parser.add_argument("--trace",     action="store_true", help="Write JSONL traces")
    parser.add_argument("--skip-nli",  action="store_true", help="Disable NLI (faster)")
    parser.add_argument("--no-canon",  action="store_true", help="Skip CANONICALIZE LLM call")
    parser.add_argument("--chunks",    type=int, default=None,
                        help="Limit to first N chunks (each chunk = 10 sentences)")
    parser.add_argument("--list-only", action="store_true", help="Print available PMCIDs and exit")
    args = parser.parse_args()

    if args.list_only:
        list_pmcids()
        return

    if not args.pmcid:
        parser.print_help()
        sys.exit(1)

    pmcid = args.pmcid.strip()
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid

    logger.info("Building single-model runner (Gemini 2.5 Flash Lite)…")
    runner = build_runner(
        trace=args.trace,
        skip_nli=args.skip_nli,
        no_canon=args.no_canon,
    )

    logger.info("Loading %s from database…", pmcid)
    try:
        file_data = runner.load_paper_from_db(pmcid)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    sentences = file_data["sentences_with_provenance"]

    # Optional sentence cap
    if args.chunks is not None:
        cap = args.chunks * 10
        sentences = sentences[:cap]
        file_data = {**file_data, "sentences_with_provenance": sentences}
        logger.info("Capped to first %d sentences (%d chunks)", cap, args.chunks)

    logger.info("Starting pipeline on %d sentences…", len(sentences))
    result = runner.process(file_data)

    if result["status"] == "error":
        logger.error("Pipeline failed: %s", result["error"])
        sys.exit(1)

    _print_result(result)


if __name__ == "__main__":
    main()
