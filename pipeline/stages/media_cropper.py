"""
PyMuPDFMediaCropper

Crops figure and table regions from a PDF using PyMuPDF, saves each crop as a
PNG image, and returns CroppedMedia metadata for both categories.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline.config import CroppingConfig
from pipeline.models.dto import BoundingBox, CroppedMedia, LayoutResult
from parsers.layout_utils import (
    FIG_NUM_RE,
    TAB_NUM_RE,
    nearest_caption,
    parse_caption_num,
)

logger = logging.getLogger(__name__)

_PICTURE_TYPES = frozenset({"PICTURE"})
_TABLE_TYPES   = frozenset({"TABLE", "RECONSTRUCTED_TABLE"})


class PyMuPDFMediaCropper:
    """
    Crops figure and table regions from PDFs using PyMuPDF.

    Parameters
    ----------
    config:
        CroppingConfig controlling DPI, output format, etc.
    figures_dir:
        Directory to save figure crops.
    tables_dir:
        Directory to save table crops.
    """

    def __init__(
        self,
        config: Optional[CroppingConfig] = None,
        figures_dir: Optional[Path] = None,
        tables_dir:  Optional[Path] = None,
    ) -> None:
        self._config      = config or CroppingConfig()
        self._figures_dir = figures_dir or Path("out/figures")
        self._tables_dir  = tables_dir  or Path("out/tables")

    def crop(
        self,
        pdf_path: Path,
        layout: LayoutResult,
    ) -> Tuple[List[CroppedMedia], List[CroppedMedia]]:
        """
        Crop and save figure and table regions.

        Args:
            pdf_path: Path to the original (unmasked) PDF.
            layout:   Layout result containing PICTURE / TABLE element positions.

        Returns:
            ``(figures, tables)`` — two lists of CroppedMedia.
        """
        import fitz  # type: ignore

        element_dicts = layout.to_element_dicts()
        captions = [e for e in element_dicts if e.get("type") == "CAPTION"]

        figures: List[CroppedMedia] = []
        tables:  List[CroppedMedia] = []

        doc = fitz.open(str(pdf_path))
        scale = self._config.dpi / 72.0
        mat   = fitz.Matrix(scale, scale)

        fig_idx = tab_idx = 0

        for el in layout.elements:
            if el.type in _PICTURE_TYPES and self._config.save_figure_crops:
                cap  = nearest_caption(el.to_dict(), captions)
                cap_text = cap["text"] if cap else None
                num  = parse_caption_num(cap_text or "", FIG_NUM_RE)
                if num is None:
                    fig_idx += 1
                label = f"Figure {num}" if num else f"Figure_p{el.page}_{fig_idx}"
                media = self._crop_element(
                    doc, el.bbox, layout.page_dims, mat, label,
                    num, cap_text, "figure", pdf_path.stem, self._figures_dir,
                )
                if media:
                    figures.append(media)

            elif el.type in _TABLE_TYPES and self._config.save_table_crops:
                cap  = nearest_caption(el.to_dict(), captions)
                cap_text = cap["text"] if cap else None
                num  = parse_caption_num(cap_text or "", TAB_NUM_RE)
                if num is None:
                    tab_idx += 1
                label = f"Table {num}" if num else f"Table_p{el.page}_{tab_idx}"
                media = self._crop_element(
                    doc, el.bbox, layout.page_dims, mat, label,
                    num, cap_text, "table", pdf_path.stem, self._tables_dir,
                )
                if media:
                    tables.append(media)

        doc.close()
        logger.info(
            "MediaCropper: %d figures, %d tables cropped from %s",
            len(figures), len(tables), pdf_path.name,
        )
        return figures, tables

    # ── Internal ──────────────────────────────────────────────────────────────

    def _crop_element(
        self,
        doc,
        bbox: BoundingBox,
        page_dims: dict,
        mat,
        label: str,
        number: Optional[int],
        caption: Optional[str],
        media_type: str,
        stem: str,
        out_dir: Path,
    ) -> Optional[CroppedMedia]:
        page_no = bbox.page
        page_h  = page_dims.get(page_no, {}).get("height", 792.0)
        rect    = bbox.to_fitz_rect(page_h)
        if rect.is_empty:
            return None

        out_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "-")
        filename   = f"{stem}_{safe_label}.{self._config.image_format}"
        out_path   = out_dir / filename

        page = doc[page_no - 1]
        pix  = page.get_pixmap(matrix=mat, clip=rect)
        pix.save(str(out_path))

        return CroppedMedia(
            media_type=media_type,
            label=label,
            number=number,
            caption=caption,
            image_path=out_path,
            bbox=bbox,
            page=page_no,
        )
