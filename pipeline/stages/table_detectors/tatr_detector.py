"""
TATRTableDetector

Uses the Table Transformer (TATR) model from Microsoft to detect table
bounding boxes in each page of a PDF.  Model is loaded lazily on first use.

Reference model: microsoft/table-transformer-detection
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pipeline.config import TATRConfig
from pipeline.models.dto import BoundingBox, DetectedRegion, TableDetectionResult

logger = logging.getLogger(__name__)

# Points-per-inch for PDF coordinates, and the DPI we render pages at for TATR
_PDF_PPI = 72
_RENDER_DPI = 150
_SCALE = _PDF_PPI / _RENDER_DPI   # pixel → PDF points


class TATRTableDetector:
    """
    Table detector backed by the TATR object-detection model.

    The model and processor are loaded once and cached on the instance.
    """

    def __init__(self, config: Optional[TATRConfig] = None) -> None:
        self._config = config or TATRConfig()
        self._processor = None
        self._model = None

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModelForObjectDetection  # type: ignore

        logger.info("Loading TATR model (%s)…", self._config.model_name)
        self._processor = AutoImageProcessor.from_pretrained(self._config.model_name)
        self._model = AutoModelForObjectDetection.from_pretrained(self._config.model_name)
        self._model.eval()
        logger.info("TATR model loaded.")

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, pdf_path: Path) -> TableDetectionResult:
        """
        Run TATR table detection on every page of ``pdf_path``.

        Args:
            pdf_path: Path to the PDF to analyse.

        Returns:
            TableDetectionResult with one DetectedRegion per detected table.
        """
        import fitz          # type: ignore  (PyMuPDF)
        import torch         # type: ignore
        from PIL import Image as PILImage  # type: ignore

        self._load_model()

        doc = fitz.open(str(pdf_path))
        regions = []
        page_dims: dict = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_no = page_num + 1
            page_dims[page_no] = {"width": page.rect.width, "height": page.rect.height}

            mat = fitz.Matrix(_RENDER_DPI / _PDF_PPI, _RENDER_DPI / _PDF_PPI)
            pix = page.get_pixmap(matrix=mat)
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

            inputs = self._processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_object_detection(
                outputs,
                threshold=self._config.threshold,
                target_sizes=[(img.height, img.width)],
            )[0]

            page_h = page.rect.height
            for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                x1, y1_screen, x2, y2_screen = [v * _SCALE for v in box.tolist()]
                # Convert from screen coords to Docling PDF coords
                bbox = BoundingBox(
                    x1=x1,
                    y1=page_h - y1_screen,
                    x2=x2,
                    y2=page_h - y2_screen,
                    page=page_no,
                )
                label = self._model.config.id2label[label_id.item()]
                regions.append(
                    DetectedRegion(
                        bbox=bbox,
                        score=round(score.item(), 3),
                        source="tatr",
                        label=label,
                    )
                )

        doc.close()
        return TableDetectionResult(
            regions=regions,
            source="tatr",
            pdf_path=pdf_path,
            page_dims=page_dims,
        )
