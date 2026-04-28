"""
Match pipeline findings against silver findings, compute metrics, write report.

Usage:
  python -m eval.silver.evaluate
  python -m eval.silver.evaluate --silver eval/data/silver_findings.jsonl \\
                                  --pipeline eval/data/pipeline_findings.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

from eval.silver.jsonl_utils import read_jsonl, write_jsonl
from eval.silver.matcher import compute_metrics, match_case
from eval.silver.schemas import (
    EvalMetrics,
    MatchResult,
    PipelineCaseOutput,
    SilverCaseResult,
)

SILVER_PATH   = Path("eval/data/silver_findings.jsonl")
PIPELINE_PATH = Path("eval/data/pipeline_findings.jsonl")
REPORTS_DIR   = Path("eval/reports")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline vs silver findings")
    parser.add_argument("--silver",   default=str(SILVER_PATH))
    parser.add_argument("--pipeline", default=str(PIPELINE_PATH))
    parser.add_argument("--reports",  default=str(REPORTS_DIR))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Similarity threshold override (default: 0.55)")
    args = parser.parse_args()

    silver_path   = Path(args.silver)
    pipeline_path = Path(args.pipeline)
    reports_dir   = Path(args.reports)

    for p in (silver_path, pipeline_path):
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)

    if args.threshold is not None:
        import eval.silver.matcher as _m
        _m.SIMILARITY_THRESHOLD = args.threshold

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set — required for embedding-based matching", file=sys.stderr)
        sys.exit(1)

    # Load data; index by case_id
    silver_by_case: dict[str, SilverCaseResult] = {}
    for rec in read_jsonl(silver_path, SilverCaseResult):
        # Latest per case_id wins (if re-run with new prompt)
        silver_by_case[rec.case_id] = rec

    pipeline_by_case: dict[str, PipelineCaseOutput] = {}
    for rec in read_jsonl(pipeline_path, PipelineCaseOutput):
        pipeline_by_case[rec.case_id] = rec

    common = set(silver_by_case) & set(pipeline_by_case)
    only_silver = set(silver_by_case) - set(pipeline_by_case)
    only_pipeline = set(pipeline_by_case) - set(silver_by_case)

    if only_silver:
        logger.warning("%d case(s) have silver but no pipeline output: %s",
                       len(only_silver), sorted(only_silver)[:5])
    if only_pipeline:
        logger.warning("%d case(s) have pipeline output but no silver: %s",
                       len(only_pipeline), sorted(only_pipeline)[:5])

    logger.info("Matching %d common case(s)…", len(common))
    match_results: list[MatchResult] = []

    for case_id in sorted(common):
        silver = silver_by_case[case_id]
        pipeline = pipeline_by_case[case_id]
        result = match_case(silver, pipeline, api_key)
        match_results.append(result)
        logger.info("  %s — matched %d / %d silver (pipeline has %d)",
                    case_id, len(result.matched),
                    len(silver.findings), len(pipeline.findings))

    silver_list = [silver_by_case[c] for c in sorted(common)]
    pipeline_list = [pipeline_by_case[c] for c in sorted(common)]

    prompt_version = silver_list[0].prompt_version if silver_list else "unknown"
    model = silver_list[0].model if silver_list else "unknown"

    metrics = compute_metrics(match_results, silver_list, pipeline_list, prompt_version, model)

    # Write outputs
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    reports_dir.mkdir(parents=True, exist_ok=True)

    match_path = reports_dir / f"match_results_{timestamp}.jsonl"
    write_jsonl(match_path, match_results)

    metrics_path = reports_dir / f"metrics_{timestamp}.json"
    metrics_path.write_text(metrics.model_dump_json(indent=2))

    print(f"\n{'─' * 50}")
    print(f"Evaluation results  ({timestamp})")
    print(f"{'─' * 50}")
    print(f"Cases evaluated:    {metrics.n_cases}")
    print(f"Silver findings:    {metrics.n_silver_findings}")
    print(f"Pipeline findings:  {metrics.n_pipeline_findings}")
    print(f"Matched pairs:      {metrics.n_matched}")
    print(f"Precision:          {metrics.precision:.3f}")
    print(f"Recall:             {metrics.recall:.3f}")
    print(f"F1:                 {metrics.f1:.3f}")
    print(f"Avg similarity:     {metrics.avg_similarity:.3f}")
    print(f"\nMatch results → {match_path}")
    print(f"Metrics      → {metrics_path}")


if __name__ == "__main__":
    main()
