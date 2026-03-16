"""
ModelRegistry

Centralises lazy loading of heavyweight ML models (Docling converter, TATR,
scispaCy) so they are instantiated at most once per process.

Usage::

    from pipeline.stages.pdf_text_extraction.resources import ModelRegistry
    registry = ModelRegistry()
    converter = registry.docling_converter        # loaded on first access
    proc, model = registry.tatr                   # loaded on first access
    nlp = registry.spacy_nlp                      # loaded on first access
"""
from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

from pipeline.stages.pdf_text_extraction.config import DoclingConfig, TATRConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Thread-safe singleton-style registry for ML models.

    Parameters
    ----------
    docling_config:
        DoclingConfig for the Docling converter.
    tatr_config:
        TATRConfig for the TATR model.
    spacy_model:
        Name of the scispaCy model (default: ``en_core_sci_sm``).
    """

    def __init__(
        self,
        docling_config: Optional[DoclingConfig] = None,
        tatr_config: Optional[TATRConfig] = None,
        spacy_model: str = "en_core_sci_sm",
    ) -> None:
        self._docling_config = docling_config or DoclingConfig()
        self._tatr_config    = tatr_config    or TATRConfig()
        self._spacy_model    = spacy_model

        self._lock = threading.Lock()

        self._converter = None
        self._tatr_proc = None
        self._tatr_model = None
        self._nlp = None

    # ── Docling ───────────────────────────────────────────────────────────────

    @property
    def docling_converter(self):
        """Lazy-loaded Docling DocumentConverter."""
        with self._lock:
            if self._converter is None:
                from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore
                from docling.datamodel.pipeline_options import PdfPipelineOptions          # type: ignore
                from docling.datamodel.base_models import InputFormat                       # type: ignore

                opts = PdfPipelineOptions()
                opts.do_table_structure = self._docling_config.do_table_structure
                opts.do_ocr             = self._docling_config.do_ocr
                opts.images_scale       = 2.0

                self._converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
                )
                logger.info("Docling converter loaded.")
            return self._converter

    # ── TATR ──────────────────────────────────────────────────────────────────

    @property
    def tatr(self) -> Tuple:
        """Lazy-loaded (processor, model) pair for TATR."""
        with self._lock:
            if self._tatr_model is None:
                from transformers import AutoImageProcessor, AutoModelForObjectDetection  # type: ignore

                logger.info("Loading TATR model (%s)…", self._tatr_config.model_name)
                self._tatr_proc  = AutoImageProcessor.from_pretrained(
                    self._tatr_config.model_name
                )
                self._tatr_model = AutoModelForObjectDetection.from_pretrained(
                    self._tatr_config.model_name
                )
                self._tatr_model.eval()
                logger.info("TATR model loaded.")
            return self._tatr_proc, self._tatr_model

    # ── scispaCy ──────────────────────────────────────────────────────────────

    @property
    def spacy_nlp(self):
        """Lazy-loaded scispaCy NLP pipeline."""
        with self._lock:
            if self._nlp is None:
                import spacy  # type: ignore

                logger.info("Loading scispaCy model (%s)…", self._spacy_model)
                self._nlp = spacy.load(self._spacy_model)
                logger.info("scispaCy model loaded.")
            return self._nlp
