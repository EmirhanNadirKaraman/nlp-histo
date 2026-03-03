"""
PyMuPDFRegionMasker

Draws opaque white rectangles over detected table/figure regions, producing
a clean masked PDF suitable for a second Docling extraction pass.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from pipeline.config import MaskingConfig
from pipeline.models.dto import BoundingBox, LayoutResult, TableDetectionResult

logger = logging.getLogger(__name__)


class PyMuPDFRegionMasker:
    """
    Masks a list of bounding boxes in a PDF using PyMuPDF (fitz).

    Parameters
    ----------
    config:
        MaskingConfig controlling whether to expand boxes, merge overlaps, etc.
    output_dir:
        Directory where masked PDFs are written.  Defaults to the same
        directory as the input PDF if not provided.
    """

    def __init__(
        self,
        config: Optional[MaskingConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._config = config or MaskingConfig()
        self._output_dir = output_dir

    _FIGURE_TYPES = frozenset({"PICTURE", "FIGURE"})

    def collect_regions(
        self,
        detection: TableDetectionResult,
        layout: LayoutResult,
    ) -> List[BoundingBox]:
        """
        Build the list of bounding boxes to mask, driven by config flags.

        Args:
            detection: Table detection result (used when mask_tables=True).
            layout:    Full layout result (used when mask_figures=True).

        Returns:
            Combined list of BoundingBoxes to paint white.
        """
        regions: List[BoundingBox] = []
        if self._config.mask_tables:
            regions.extend(r.bbox for r in detection.regions)
        if self._config.mask_figures:
            regions.extend(
                el.bbox for el in layout.elements if el.type in self._FIGURE_TYPES
            )
        return regions

    def mask(self, pdf_path: Path, regions: List[BoundingBox]) -> Path:
        """
        Write a new PDF with all ``regions`` painted white.

        Args:
            pdf_path: Source PDF path.
            regions:  Regions to mask (Docling PDF coordinates).

        Returns:
            Path to the masked PDF (written next to the source or into output_dir).
        """
        import fitz  # type: ignore
        from parsers.layout_utils import merge_rects

        out_dir = self._output_dir or pdf_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pdf_path.stem}_masked.pdf"

        doc = fitz.open(str(pdf_path))

        # Group regions by page
        by_page: dict = {}
        for bbox in regions:
            by_page.setdefault(bbox.page, []).append(bbox)

        exp = self._config.expand_box_px

        for page_num in range(len(doc)):
            page_no = page_num + 1
            if page_no not in by_page:
                continue

            page = doc[page_num]
            page_h = page.rect.height

            rects = []
            for b in by_page[page_no]:
                r = b.to_fitz_rect(page_h)
                if exp:
                    r = fitz.Rect(r.x0 - exp, r.y0 - exp, r.x1 + exp, r.y1 + exp)
                rects.append(r)

            if self._config.merge_overlapping_boxes:
                rects = merge_rects(rects)

            for rect in rects:
                if rect.is_empty:
                    continue
                page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

        doc.save(str(out_path))
        doc.close()
        logger.info("Masked PDF written to %s (%d pages affected)",
                    out_path, len(by_page))
        return out_path
