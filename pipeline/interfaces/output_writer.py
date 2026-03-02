from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from pipeline.models.dto import CroppedMedia, HierarchicalRow


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
    ) -> None:
        """
        Persist pipeline results for a single document.

        Args:
            pmcid:   PubMed Central document identifier.
            rows:    Hierarchical text rows assembled from the layout.
            figures: Cropped figure metadata.
            tables:  Cropped table metadata.
        """
        ...
