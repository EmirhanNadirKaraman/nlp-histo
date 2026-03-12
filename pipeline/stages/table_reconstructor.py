"""
TableReconstructor

Injects RECONSTRUCTED_TABLE elements into a LayoutResult by grouping TEXT /
LIST_ITEM elements that follow a table CAPTION into a single synthetic element.

This handles PDFs where Docling fails to detect a table as a TABLE element and
instead emits it as a sequence of list/text rows.

Algorithm (ported from scripts/visualize_docling_full.py):
  1. Scan elements for a CAPTION whose text contains "table".
  2. Consume following TEXT / LIST_ITEM elements while the vertical gap between
     consecutive elements stays within the adaptive threshold.
  3. The threshold is initialised at 20 pts; after the first row is captured it
     is refined to ``first_inter_row_gap * threshold_multiplier``.
  4. Union the bounding boxes of all consumed elements into a single
     RECONSTRUCTED_TABLE element (caption text carried as ``text``).
"""
from __future__ import annotations

import logging
from typing import List

from pipeline.models.dto import BoundingBox, LayoutElement, LayoutResult

logger = logging.getLogger(__name__)


def reconstruct_tables_from_lists(
    layout: LayoutResult,
    threshold_multiplier: float = 1.2,
) -> LayoutResult:
    """
    Return a new LayoutResult with RECONSTRUCTED_TABLE elements injected.

    Args:
        layout:               Input LayoutResult (not modified in place).
        threshold_multiplier: Multiplier applied to the first inter-row gap to
                              set the maximum allowed vertical gap between rows.

    Returns:
        New LayoutResult with RECONSTRUCTED_TABLE elements spliced in after
        each qualifying table caption.
    """
    elements = layout.elements
    new_elements: List[LayoutElement] = []
    i = 0

    while i < len(elements):
        el = elements[i]

        if el.type == "CAPTION" and "table" in (el.text or "").lower():
            new_elements.append(el)

            sub_elements: List[LayoutElement] = []
            max_allowed_gap = 20.0
            # Docling coords: y2 is the bottom edge (smaller value)
            last_y2 = el.bbox.y2
            i += 1

            while i < len(elements):
                next_el = elements[i]
                if next_el.type not in ("TEXT", "LIST_ITEM"):
                    break

                # In Docling coords y1 > y2; gap between bottom of last element
                # and top of next element = next.y1 - last.y2
                vertical_gap = abs(next_el.bbox.y1 - last_y2)

                # After the first row, refine threshold from actual row spacing
                if len(sub_elements) == 1:
                    true_gutter = abs(next_el.bbox.y1 - sub_elements[0].bbox.y2)
                    max_allowed_gap = true_gutter * threshold_multiplier

                if vertical_gap < max_allowed_gap:
                    sub_elements.append(next_el)
                    last_y2 = next_el.bbox.y2
                    i += 1
                else:
                    break

            if sub_elements:
                page = sub_elements[0].bbox.page
                x1 = min(e.bbox.x1 for e in sub_elements)
                y1 = max(e.bbox.y1 for e in sub_elements)  # topmost (largest y)
                x2 = max(e.bbox.x2 for e in sub_elements)
                y2 = min(e.bbox.y2 for e in sub_elements)  # bottommost (smallest y)
                new_elements.append(LayoutElement(
                    type="RECONSTRUCTED_TABLE",
                    page=page,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, page=page),
                    text=el.text,
                    level=0,
                ))
                logger.debug(
                    "Reconstructed table from %d sub-elements on page %d (caption: %s)",
                    len(sub_elements), page, el.text,
                )
        else:
            new_elements.append(el)
            i += 1

    n_recon = sum(1 for e in new_elements if e.type == "RECONSTRUCTED_TABLE")
    if n_recon:
        logger.info("Table reconstruction: injected %d RECONSTRUCTED_TABLE element(s)", n_recon)

    return LayoutResult(
        elements=new_elements,
        page_dims=layout.page_dims,
        pdf_path=layout.pdf_path,
        source=layout.source,
    )
