from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, runtime_checkable

from pipeline.stages.pdf_text_extraction.models.dto import BoundingBox


@runtime_checkable
class RegionMasker(Protocol):
    """
    Contract for PDF region-masking components.

    An implementation draws opaque white rectangles over the supplied bounding
    boxes and writes the result to a new PDF file, returning its path.
    """

    def mask(self, pdf_path: Path, regions: List[BoundingBox]) -> Path:
        """
        Mask regions in a PDF with white rectangles.

        Args:
            pdf_path: Path to the input PDF.
            regions:  Bounding boxes (Docling coords) of regions to mask.

        Returns:
            Path to the newly written masked PDF.
        """
        ...
