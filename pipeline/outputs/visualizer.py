"""
DetectionVisualizer

Renders bounding-box visualizations for layout elements and detected table
regions onto PDF page images using matplotlib.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pipeline.config import VisualizationConfig
from pipeline.models.dto import LayoutResult, TableDetectionResult

logger = logging.getLogger(__name__)


class DetectionVisualizer:
    """
    Saves per-page visualizations of detected regions.

    Parameters
    ----------
    config:
        VisualizationConfig controlling which outputs to generate.
    output_dir:
        Directory where visualization images are written.
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._config = config or VisualizationConfig()
        self._output_dir = output_dir or Path("out/visualization")

    def visualize_layout(self, layout: LayoutResult, pmcid: str) -> None:
        """
        Save page images annotated with layout element bounding boxes.

        Args:
            layout: Layout result to visualize.
            pmcid:  Document identifier used in output filenames.
        """
        if not self._config.enabled or not self._config.save_combined_visualization:
            return
        self._render(layout.pdf_path, layout=layout, detection=None, pmcid=pmcid)

    def visualize_detections(
        self,
        detection: TableDetectionResult,
        layout: Optional[LayoutResult],
        pmcid: str,
    ) -> None:
        """
        Save page images annotated with table detection results.

        Args:
            detection: TableDetectionResult to visualize.
            layout:    Optional base layout for background rendering.
            pmcid:     Document identifier.
        """
        if not self._config.enabled or not self._config.save_tatr_visualization:
            return
        self._render(detection.pdf_path, layout=layout, detection=detection, pmcid=pmcid)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _render(
        self,
        pdf_path: Path,
        layout: Optional[LayoutResult],
        detection: Optional[TableDetectionResult],
        pmcid: str,
    ) -> None:
        try:
            import fitz          # type: ignore
            import matplotlib.pyplot as plt       # type: ignore
            import matplotlib.patches as patches  # type: ignore
        except ImportError:
            logger.warning("Visualization requires fitz and matplotlib — skipping.")
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(str(pdf_path))
        max_pages = self._config.max_pages or len(doc)

        for page_num in range(min(len(doc), max_pages)):
            page   = doc[page_num]
            page_h = page.rect.height
            pix    = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            scale  = 1.5

            import numpy as np  # type: ignore
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            fig, ax = plt.subplots(1, figsize=(10, 14))
            ax.imshow(img)
            ax.axis("off")

            if layout:
                for el in layout.elements:
                    if el.page != page_num + 1:
                        continue
                    b   = el.bbox
                    top = (page_h - max(b.y1, b.y2)) * scale
                    h   = abs(b.y1 - b.y2) * scale
                    rect = patches.Rectangle(
                        (b.x1 * scale, top), (b.x2 - b.x1) * scale, h,
                        linewidth=0.5, edgecolor="blue", facecolor="none", alpha=0.4,
                    )
                    ax.add_patch(rect)

            if detection:
                for region in detection.regions:
                    if region.bbox.page != page_num + 1:
                        continue
                    b   = region.bbox
                    top = (page_h - max(b.y1, b.y2)) * scale
                    h   = abs(b.y1 - b.y2) * scale
                    rect = patches.Rectangle(
                        (b.x1 * scale, top), (b.x2 - b.x1) * scale, h,
                        linewidth=2, edgecolor="red", facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        b.x1 * scale, top - 4,
                        f"{region.source}:{region.score:.2f}",
                        fontsize=6, color="red",
                    )

            out = self._output_dir / f"{pmcid}_p{page_num + 1:03d}_vis.png"
            plt.tight_layout(pad=0)
            plt.savefig(str(out), dpi=120, bbox_inches="tight")
            plt.close(fig)

        doc.close()
        logger.info("Visualizer: saved pages 1–%d for %s", max_pages, pmcid)
