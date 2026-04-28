"""
Generate silver findings from source cases using Opus.

Reads eval/data/source_cases.jsonl, writes eval/data/silver_findings.jsonl.
Skips cases already in the output file (cache by case_id + prompt_version + model).

Usage:
  python -m eval.silver.generate
  python -m eval.silver.generate --model claude-opus-4-5
  python -m eval.silver.generate --source eval/data/source_cases.jsonl
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

from eval.silver.generator import SilverGenerator, DEFAULT_MODEL

SOURCE_PATH = Path("eval/data/source_cases.jsonl")
OUTPUT_PATH = Path("eval/data/silver_findings.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Generate silver labels via Opus")
    parser.add_argument("--source",  default=str(SOURCE_PATH))
    parser.add_argument("--output",  default=str(OUTPUT_PATH))
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Source file not found: {source}", file=sys.stderr)
        print("Run `python -m eval.silver.sample` first.", file=sys.stderr)
        sys.exit(1)

    gen = SilverGenerator(
        source_cases_path=source,
        output_path=Path(args.output),
        model=args.model,
    )
    gen.run()


if __name__ == "__main__":
    main()
