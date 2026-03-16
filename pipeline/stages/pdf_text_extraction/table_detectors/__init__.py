"""Table detector implementations."""
from pipeline.stages.pdf_text_extraction.table_detectors.docling_detector import DoclingTableDetector
from pipeline.stages.pdf_text_extraction.table_detectors.tatr_detector import TATRTableDetector
from pipeline.stages.pdf_text_extraction.table_detectors.hybrid_detector import HybridTableDetector

__all__ = ["DoclingTableDetector", "TATRTableDetector", "HybridTableDetector"]
