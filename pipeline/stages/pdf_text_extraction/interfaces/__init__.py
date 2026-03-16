"""Pipeline stage Protocol interfaces."""
from pipeline.stages.pdf_text_extraction.interfaces.table_detector import TableDetector
from pipeline.stages.pdf_text_extraction.interfaces.layout_extractor import LayoutExtractor
from pipeline.stages.pdf_text_extraction.interfaces.region_masker import RegionMasker
from pipeline.stages.pdf_text_extraction.interfaces.text_assembler import TextAssembler
from pipeline.stages.pdf_text_extraction.interfaces.artifact_filter import ArtifactFilter
from pipeline.stages.pdf_text_extraction.interfaces.media_cropper import MediaCropper
from pipeline.stages.pdf_text_extraction.interfaces.output_writer import OutputWriter

__all__ = [
    "TableDetector",
    "LayoutExtractor",
    "RegionMasker",
    "TextAssembler",
    "ArtifactFilter",
    "MediaCropper",
    "OutputWriter",
]
