"""
Pluggable extraction components for figures and tables.

This package provides a modular architecture for PDF extraction with
multiple interchangeable backends.
"""

from .base import (
    BoundingBox,
    Figure,
    Table,
    ExtractionResult,
    BaseExtractor,
    FigureExtractor,
    TableExtractor,
    CombinedExtractor
)

from .registry import ExtractorRegistry, get_extractor

__all__ = [
    'BoundingBox',
    'Figure',
    'Table',
    'ExtractionResult',
    'BaseExtractor',
    'FigureExtractor',
    'TableExtractor',
    'CombinedExtractor',
    'ExtractorRegistry',
    'get_extractor',
]
