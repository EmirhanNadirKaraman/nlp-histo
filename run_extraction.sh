#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PDF_DIR="${1:-files/organized_pdfs}"
WORKERS="${2:-4}"
MAX_DOCS="${3:-}"

MAX_DOCS_ARG=""
if [[ -n "$MAX_DOCS" ]]; then
    MAX_DOCS_ARG="max_docs=$MAX_DOCS,"
fi

echo "PDF dir : $PDF_DIR"
echo "Workers : $WORKERS"
[[ -n "$MAX_DOCS" ]] && echo "Max docs: $MAX_DOCS"
echo ""

python - <<EOF
import sys
from pathlib import Path
from pipeline.stages.pdf_text_extraction.batch import ParallelBatchRunner
from pipeline.stages.pdf_text_extraction.config import PipelineConfig

cfg = PipelineConfig()
runner = ParallelBatchRunner(cfg, max_workers=$WORKERS)
stats = runner.run(Path("$PDF_DIR"), ${MAX_DOCS_ARG})
print(f"\nDone — processed={stats['processed']} failed={stats['failed']} skipped={stats['skipped']}")
EOF
