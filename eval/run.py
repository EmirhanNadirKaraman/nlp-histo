"""
eval/run.py — Pipeline run for evaluation (precision / recall benchmarking).

All outputs are written to eval/out/ so they don't pollute the main out/
directory. Visualization is enabled by default so detections can be audited
alongside the extracted text.

Usage (from project root):
    python eval/run.py
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.batch import ParallelBatchRunner  # noqa: E402
from pipeline.config import (  # noqa: E402
    DatabaseConfig,
    PathConfig,
    PipelineConfig,
    RuntimeConfig,
    TableDetectorType,
    TextAssemblyConfig,
    VisualizationConfig,
)

HERE = Path(__file__).parent
OUT  = HERE / "out"

PDF_DIR        = ROOT / "files/organized_pdfs"
N_SAMPLES      = 30
SEED           = 42
MAX_FILE_MB    = 5.0   # skip PDFs larger than this (proxy for page count)


def make_config() -> PipelineConfig:
    cfg = PipelineConfig()

    # ── Redirect all outputs into eval/out/ ───────────────────────────────────
    cfg.paths = PathConfig(
        output_root        = OUT,
        files_root         = ROOT / "files",
        masked_pdf_dir     = OUT / "masked_pdfs",
        docling_full_dir   = OUT / "docling_full",
        docling_masked_dir = OUT / "docling_masked",
        text_dir           = OUT / "text",
        text_raw_dir       = OUT / "text_raw",
        json_dir           = OUT / "json",
        vis_dir            = OUT / "visualization",
        figures_dir        = OUT / "figures",
        tables_dir         = OUT / "tables",
        blacklist_file     = OUT / "blacklist.json",
        run_metadata_dir   = OUT / "run_metadata",
    )

    # ── Visualization: annotated PDFs for detection auditing ──────────────────
    cfg.visualization = VisualizationConfig(
        enabled                     = True,
        save_tatr_visualization     = True,
        save_combined_visualization = True,
        max_pages                   = None,
    )

    # ── Text: also dump raw pre-assembly elements for debugging ───────────────
    cfg.text = TextAssemblyConfig(
        write_raw_text = True,
    )

    # ── No DB for eval runs ───────────────────────────────────────────────────
    cfg.database = DatabaseConfig(enabled=False)

    # ── Table detector: hybrid (Docling + TATR) ───────────────────────────────
    cfg.table_detector = TableDetectorType.HYBRID

    # ── Runtime: expose seed explicitly ──────────────────────────────────────
    cfg.runtime = RuntimeConfig(seed=SEED)

    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = make_config()
    cfg.prepare()

    # ── Sample N_SAMPLES PDFs reproducibly ───────────────────────────────────
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    max_bytes = MAX_FILE_MB * 1024 * 1024
    eligible  = [p for p in all_pdfs if p.stat().st_size <= max_bytes]
    logging.getLogger(__name__).info(
        "Eligible PDFs: %d / %d (≤ %.1f MB)", len(eligible), len(all_pdfs), MAX_FILE_MB
    )
    rng = random.Random(cfg.runtime.seed)
    sample = rng.sample(eligible, min(N_SAMPLES, len(eligible)))
    sample.sort()  # stable order within the sample for readable logs

    logging.getLogger(__name__).info(
        "Eval batch: %d / %d PDFs (seed=%d)", len(sample), len(all_pdfs), cfg.runtime.seed
    )

    ParallelBatchRunner(cfg, max_workers=4).run_paths(sample)


if __name__ == "__main__":
    main()
