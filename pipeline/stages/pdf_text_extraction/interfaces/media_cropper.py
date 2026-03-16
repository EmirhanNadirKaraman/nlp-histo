from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, Tuple, runtime_checkable

from pipeline.stages.pdf_text_extraction.models.dto import CroppedMedia, LayoutResult


@runtime_checkable
class MediaCropper(Protocol):
    """
    Contract for figure/table image-cropping components.

    An implementation renders each figure and table region from the PDF at a
    specified DPI and saves the crops as image files, returning metadata for
    each cropped item.
    """

    def crop(
        self,
        pdf_path: Path,
        layout: LayoutResult,
    ) -> Tuple[List[CroppedMedia], List[CroppedMedia]]:
        """
        Crop and save figure and table regions from a PDF.

        Args:
            pdf_path: Path to the original (unmasked) PDF.
            layout:   Layout result containing PICTURE / TABLE element positions.

        Returns:
            (figures, tables) — two lists of CroppedMedia metadata.
        """
        ...
