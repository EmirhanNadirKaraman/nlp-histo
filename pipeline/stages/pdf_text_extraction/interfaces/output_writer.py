from __future__ import annotations

from pathlib import Path
from typing import List, Protocol, runtime_checkable

from pipeline.stages.pdf_text_extraction.models.dto import CroppedMedia, HierarchicalRow


@runtime_checkable
class OutputWriter(Protocol):
    """
    Contract for pipeline output-writing components.

    An implementation persists the assembled document data (text rows, figures,
    tables) to whatever backing store it targets (file system, database, …).
    """

    def write(
        self,
        pmcid: str,
        rows: List[HierarchicalRow],
        figures: List[CroppedMedia],
        tables: List[CroppedMedia],
        pdf_path: Path | None = None,
    ) -> None:
        """
        Persist pipeline results for a single document.

        Args:
            pmcid:    PubMed Central document identifier.
            rows:     Hierarchical text rows assembled from the layout.
            figures:  Cropped figure metadata.
            tables:   Cropped table metadata.
            pdf_path: Path to the source PDF (used by some writers, e.g. DB ingester).
        """
        ...
