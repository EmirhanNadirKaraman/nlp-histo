"""Pipeline stage Protocol interfaces."""
from pipeline.interfaces.table_detector import TableDetector
from pipeline.interfaces.layout_extractor import LayoutExtractor
from pipeline.interfaces.region_masker import RegionMasker
from pipeline.interfaces.text_assembler import TextAssembler
from pipeline.interfaces.artifact_filter import ArtifactFilter
from pipeline.interfaces.media_cropper import MediaCropper
from pipeline.interfaces.output_writer import OutputWriter

__all__ = [
    "TableDetector",
    "LayoutExtractor",
    "RegionMasker",
    "TextAssembler",
    "ArtifactFilter",
    "MediaCropper",
    "OutputWriter",
]
