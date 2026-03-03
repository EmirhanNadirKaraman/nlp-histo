"""Concrete pipeline stage implementations."""
from pipeline.stages.layout_extractor import DoclingLayoutExtractor
from pipeline.stages.region_masker import PyMuPDFRegionMasker
from pipeline.stages.text_assembler import HierarchicalTextAssembler
from pipeline.stages.artifact_filter import ArtifactFilter
from pipeline.stages.media_cropper import PyMuPDFMediaCropper
from pipeline.stages.visualizer import DetectionVisualizer
from pipeline.stages.table_detectors import (
    DoclingTableDetector,
    TATRTableDetector,
    HybridTableDetector,
)

__all__ = [
    "DoclingLayoutExtractor",
    "PyMuPDFRegionMasker",
    "HierarchicalTextAssembler",
    "ArtifactFilter",
    "PyMuPDFMediaCropper",
    "DetectionVisualizer",
    "DoclingTableDetector",
    "TATRTableDetector",
    "HybridTableDetector",
]
