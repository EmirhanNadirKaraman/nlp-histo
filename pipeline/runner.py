"""
PipelineRunner

Orchestrates the full PDF-processing pipeline for a single document:

  1. Extract layout from original PDF (Docling)
  2. Detect tables (Docling | TATR | Hybrid — configurable)
  3. Mask detected regions with white rectangles
  4. Re-extract layout from masked PDF (Docling)
  5. Filter layout artifacts
  6. Assemble hierarchical text
  7. Crop figure / table images
  8. Write outputs (text file, database, visualizations)

Usage::

    from pathlib import Path
    from pipeline import PipelineConfig, PipelineRunner

    cfg    = PipelineConfig()
    cfg.database.enabled = True
    cfg.database.db_url  = "postgresql://user:pw@localhost/nlp_histo"

    runner = PipelineRunner(cfg)
    runner.run_document(Path("files/organized_pdfs/PMC123.pdf"), pmcid="PMC123")

    # Batch processing
    runner.run_batch(pdf_dir=Path("files/organized_pdfs"))
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path as _Path

# Allow running as `python pipeline/runner.py` from the project root
sys.path.insert(0, str(_Path(__file__).parent.parent))
import traceback
from pathlib import Path
from typing import List, Optional

from pipeline.blacklist import BlacklistManager
from pipeline.config import PipelineConfig, TableDetectorType
from pipeline.models.dto import LayoutResult

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Orchestrates the pipeline for one or many documents.

    Parameters
    ----------
    config:
        Master PipelineConfig.  Call ``config.prepare()`` before first use to
        validate settings and create output directories.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self._cfg = config or PipelineConfig()
        self._blacklist = BlacklistManager(self._cfg.paths.blacklist_file)

        # Lazy stage instances — created on first use
        self._layout_extractor = None
        self._region_masker    = None
        self._text_assembler   = None
        self._artifact_filter  = None
        self._media_cropper    = None
        self._table_detector   = None
        self._visualizer       = None
        self._nlp              = None
        self._outputs: list    = []

    # ── Stage factory helpers ─────────────────────────────────────────────────

    def _get_layout_extractor(self):
        if self._layout_extractor is None:
            from pipeline.stages.layout_extractor import DoclingLayoutExtractor
            self._layout_extractor = DoclingLayoutExtractor(
                config=self._cfg.docling,
                cache_dir=self._cfg.paths.docling_full_dir if self._cfg.docling.export_intermediate_json else None,
            )
        return self._layout_extractor

    def _get_table_detector(self):
        if self._table_detector is None:
            dtype = self._cfg.table_detector
            if dtype == TableDetectorType.DOCLING:
                from pipeline.stages.table_detectors import DoclingTableDetector
                self._table_detector = DoclingTableDetector()
            elif dtype == TableDetectorType.TATR:
                from pipeline.stages.table_detectors import TATRTableDetector
                self._table_detector = TATRTableDetector(self._cfg.tatr)
            else:  # HYBRID or VLM (fallback to hybrid)
                from pipeline.stages.table_detectors import HybridTableDetector
                self._table_detector = HybridTableDetector(tatr_config=self._cfg.tatr)
        return self._table_detector

    def _get_region_masker(self):
        if self._region_masker is None:
            from pipeline.stages.region_masker import PyMuPDFRegionMasker
            self._region_masker = PyMuPDFRegionMasker(
                config=self._cfg.masking,
                output_dir=self._cfg.paths.masked_pdf_dir,
            )
        return self._region_masker

    def _get_masked_extractor(self):
        """Second layout extractor for the masked PDF (separate cache dir)."""
        from pipeline.stages.layout_extractor import DoclingLayoutExtractor
        return DoclingLayoutExtractor(
            config=self._cfg.docling,
            cache_dir=self._cfg.paths.docling_masked_dir if self._cfg.docling.export_intermediate_json else None,
        )

    def _get_artifact_filter(self):
        if self._artifact_filter is None:
            from pipeline.stages.artifact_filter import ArtifactFilter
            self._artifact_filter = ArtifactFilter(config=self._cfg.filtering)
        return self._artifact_filter

    def _get_text_assembler(self):
        if self._text_assembler is None:
            from pipeline.stages.text_assembler import HierarchicalTextAssembler
            self._text_assembler = HierarchicalTextAssembler(
                config=self._cfg.text,
                skip_references_section=True,
            )
        return self._text_assembler

    def _get_media_cropper(self):
        if self._media_cropper is None:
            from pipeline.stages.media_cropper import PyMuPDFMediaCropper
            self._media_cropper = PyMuPDFMediaCropper(
                config=self._cfg.cropping,
                figures_dir=self._cfg.paths.figures_dir,
                tables_dir=self._cfg.paths.tables_dir,
            )
        return self._media_cropper

    def _get_outputs(self) -> list:
        if not self._outputs:
            from pipeline.outputs.writer import TextFileWriter
            from pipeline.outputs.media_json_writer import MediaJsonWriter
            self._outputs.append(TextFileWriter(
                output_dir=self._cfg.paths.text_dir,
            ))
            self._outputs.append(MediaJsonWriter(
                output_dir=self._cfg.paths.json_dir,
            ))
            if self._cfg.database.enabled:
                from pipeline.outputs.db_ingester import PostgresDatabaseIngester
                self._outputs.append(PostgresDatabaseIngester(db_url=self._cfg.database.db_url))
        return self._outputs

    def _get_nlp(self):
        if self._nlp is None and self._cfg.masking.mask_header_footer_sidebar:
            try:
                import spacy  # type: ignore
                self._nlp = spacy.load("en_core_sci_sm")
                logger.info("scispaCy model loaded for NER-based header/footer detection.")
            except (ImportError, OSError) as exc:
                logger.warning("scispaCy not available — NER fallback disabled: %s", exc)
                self._nlp = False  # falsy sentinel so we don't retry
        return self._nlp or None

    def _get_visualizer(self):
        if self._visualizer is None and self._cfg.visualization.enabled:
            from pipeline.stages.visualizer import DetectionVisualizer
            self._visualizer = DetectionVisualizer(
                config=self._cfg.visualization,
                output_dir=self._cfg.paths.vis_dir,
            )
        return self._visualizer

    # ── Single document ───────────────────────────────────────────────────────

    def run_document(self, pdf_path: Path, pmcid: str) -> bool:
        """
        Process a single PDF document end-to-end.

        Args:
            pdf_path: Path to the source PDF.
            pmcid:    PubMed Central ID for the document.

        Returns:
            True on success, False on failure.
        """
        if self._cfg.runtime.skip_blacklisted and self._blacklist.contains(pmcid):
            logger.info("⚡ %s — skipped (blacklisted)", pmcid)
            return False

        try:
            result = self._process(pdf_path, pmcid)
            logger.info("✅ %s — done (%d rows)", pmcid, len(result))
            return True
        except Exception as exc:
            logger.error("❌ %s — failed: %s", pmcid, exc)
            if self._cfg.runtime.save_error_traces:
                logger.debug(traceback.format_exc())
            if self._cfg.runtime.update_blacklist_on_failure:
                self._blacklist.add(pmcid, reason=str(exc))
            if self._cfg.runtime.fail_fast:
                raise
            return False

    def _process(self, pdf_path: Path, pmcid: str):
        # ── Step 1: Layout extraction ──────────────────────────────────────────
        logger.info("[%s] Step 1 — layout extraction", pmcid)
        layout: LayoutResult = self._get_layout_extractor().extract(pdf_path)

        # ── Step 2: Table detection ────────────────────────────────────────────
        logger.info("[%s] Step 2 — table detection (%s)", pmcid, self._cfg.table_detector)
        detector = self._get_table_detector()

        from pipeline.stages.table_detectors.hybrid_detector import HybridTableDetector
        if isinstance(detector, HybridTableDetector):
            detection = detector.detect_with_layout(layout, pdf_path)
        else:
            from pipeline.stages.table_detectors.docling_detector import DoclingTableDetector
            if isinstance(detector, DoclingTableDetector):
                detection = detector.detect_from_layout(layout)
            else:
                detection = detector.detect(pdf_path)

        # ── Step 2b: Visualization ────────────────────────────────────────────
        if vis := self._get_visualizer():
            vis.visualize_layout(layout, pmcid)
            vis.visualize_detections(detection, layout, pmcid)

        # ── Step 3: Region masking ─────────────────────────────────────────────
        masker = self._get_region_masker()
        regions_to_mask = masker.collect_regions(detection, layout, nlp=self._get_nlp()) if self._cfg.masking.enabled else []
        if regions_to_mask:
            logger.info("[%s] Step 3 — masking %d regions", pmcid, len(regions_to_mask))
            masked_path = masker.mask(pdf_path, regions_to_mask)
        else:
            masked_path = pdf_path

        # ── Step 4: Re-extract from masked PDF ────────────────────────────────
        logger.info("[%s] Step 4 — re-extraction from masked PDF", pmcid)
        masked_layout: LayoutResult = self._get_masked_extractor().extract(masked_path)

        # ── Step 5: Artifact filtering ────────────────────────────────────────
        if self._cfg.filtering.enabled:
            logger.info("[%s] Step 5 — artifact filtering", pmcid)
            masked_layout.elements = self._get_artifact_filter().filter_elements(
                masked_layout.elements
            )

        # ── Step 6: Text assembly ─────────────────────────────────────────────
        logger.info("[%s] Step 6 — text assembly", pmcid)
        rows = self._get_text_assembler().assemble(masked_layout)

        # ── Step 7: Media cropping ────────────────────────────────────────────
        logger.info("[%s] Step 7 — media cropping", pmcid)
        figures, tables = self._get_media_cropper().crop(pdf_path, layout, detection=detection)

        # ── Step 8: Outputs ───────────────────────────────────────────────────
        logger.info("[%s] Step 8 — writing outputs", pmcid)
        for output in self._get_outputs():
            output.write(pmcid, rows, figures, tables, pdf_path=pdf_path)

        return rows

    # ── Batch processing ──────────────────────────────────────────────────────

    def run_batch(
        self,
        pdf_dir: Path,
        glob: str = "*.pdf",
        pmcid_fn=None,
        max_docs: Optional[int] = None,
    ) -> dict:
        """
        Process all PDFs in ``pdf_dir``.

        Args:
            pdf_dir:  Directory containing PDF files.
            glob:     Filename glob pattern (default ``*.pdf``).
            pmcid_fn: Optional callable that maps a Path to a PMCID string.
                      Defaults to using the file stem (e.g. ``PMC123.pdf`` → ``PMC123``).

        Returns:
            Dict with keys ``processed``, ``failed``, ``skipped``.
        """
        self._cfg.prepare()

        pdfs: List[Path] = sorted(pdf_dir.glob(glob))
        if max_docs is not None:
            pdfs = pdfs[:max_docs]
        logger.info("Batch: processing %d PDFs in %s", len(pdfs), pdf_dir)

        stats = {"processed": 0, "failed": 0, "skipped": 0}

        for pdf_path in pdfs:
            pmcid = pmcid_fn(pdf_path) if pmcid_fn else pdf_path.stem
            ok = self.run_document(pdf_path, pmcid)
            if ok:
                stats["processed"] += 1
            elif self._blacklist.contains(pmcid):
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

        logger.info(
            "Batch complete: %d processed, %d failed, %d skipped",
            stats["processed"], stats["failed"], stats["skipped"],
        )
        return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from pipeline.config import PipelineConfig

    # ── Configuration ──────────────────────────────────────────────────────────
    # Edit pipeline/config.py to change defaults (detector, visualization, etc.)
    cfg = PipelineConfig()
    cfg.database.enabled = True  # set to True to ingest; db_url auto-loaded from .env
    cfg.prepare()

    # ── Single document ────────────────────────────────────────────────────────
    # PipelineRunner(cfg).run_document(
    #     pdf_path=Path("files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf"),
    #     pmcid="PMC10047158",
    # )

    # ── Sequential batch ───────────────────────────────────────────────────────
    # PipelineRunner(cfg).run_batch(pdf_dir=Path("files/organized_pdfs"), max_docs=5)

    # ── Parallel batch (use pipeline/batch.py instead) ────────────────────────
    from pipeline.batch import ParallelBatchRunner
    ParallelBatchRunner(cfg, max_workers=4).run(
        pdf_dir=Path("files/organized_pdfs"),
        max_docs=5,
    )


if __name__ == "__main__":
    main()
