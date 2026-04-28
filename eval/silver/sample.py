"""
Sample source cases from the TextElement table.

Usage:
  python -m eval.silver.sample                       # 50 cases, seed 42
  python -m eval.silver.sample --n 100 --seed 7
  python -m eval.silver.sample --pmcids PMC1 PMC2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

from eval.silver.sampler import sample_source_cases
from eval.silver.schemas import SourceCase
from eval.silver.jsonl_utils import write_jsonl

OUTPUT_PATH = Path("eval/data/source_cases.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Sample source cases for silver-label generation")
    parser.add_argument("--n",       type=int, default=50, help="Number of cases to sample")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--pmcids",  nargs="*", help="Restrict to specific PMCIDs")
    parser.add_argument("--output",  default=str(OUTPUT_PATH))
    args = parser.parse_args()

    output = Path(args.output)

    cases_raw = sample_source_cases(n=args.n, seed=args.seed, pmcids=args.pmcids)
    cases = [SourceCase(**c) for c in cases_raw]

    write_jsonl(output, cases)
    print(f"Wrote {len(cases)} source cases → {output}")


if __name__ == "__main__":
    main()
