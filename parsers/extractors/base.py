"""
Base classes for pluggable extraction components.

This module defines abstract interfaces that all extractors must implement,
enabling a pluggable architecture for figure and table extraction.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BoundingBox:
    """Normalized bounding box representation."""
    x1: float
    y1: float
    x2: float
    y2: float
    page: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'page': self.page
        }


@dataclass
class Figure:
    """Normalized figure representation."""
    bbox: BoundingBox
    figure_id: Optional[str] = None
    caption: Optional[str] = None
    confidence: Optional[float] = None
    image_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'bbox': self.bbox.to_dict(),
            'figure_id': self.figure_id,
            'caption': self.caption,
            'confidence': self.confidence,
            'image_path': self.image_path,
            'metadata': self.metadata or {}
        }


@dataclass
class Table:
    """Normalized table representation."""
    bbox: BoundingBox
    table_id: Optional[str] = None
    caption: Optional[str] = None
    confidence: Optional[float] = None
    data: Optional[List[List[str]]] = None  # Extracted table data
    html: Optional[str] = None  # HTML representation
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'bbox': self.bbox.to_dict(),
            'table_id': self.table_id,
            'caption': self.caption,
            'confidence': self.confidence,
            'data': self.data,
            'html': self.html,
            'metadata': self.metadata or {}
        }


@dataclass
class ExtractionResult:
    """Combined extraction results."""
    figures: List[Figure]
    tables: List[Table]
    extractor_name: str
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'figures': [f.to_dict() for f in self.figures],
            'tables': [t.to_dict() for t in self.tables],
            'extractor_name': self.extractor_name,
            'success': self.success,
            'error': self.error,
            'metadata': self.metadata or {}
        }


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    def __init__(self, name: str):
        """
        Initialize extractor.

        Args:
            name: Human-readable name for this extractor
        """
        self.name = name

    @abstractmethod
    def extract(self, pdf_path: Path) -> ExtractionResult:
        """
        Extract figures and tables from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractionResult with figures and tables
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this extractor is available (dependencies installed, etc.).

        Returns:
            True if extractor can be used
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Configuration dictionary
        """
        return {}


class FigureExtractor(BaseExtractor):
    """Abstract base class for figure extractors."""

    @abstractmethod
    def extract_figures(self, pdf_path: Path) -> List[Figure]:
        """
        Extract only figures from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of extracted figures
        """
        pass

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Extract figures (tables will be empty)."""
        try:
            figures = self.extract_figures(pdf_path)
            return ExtractionResult(
                figures=figures,
                tables=[],
                extractor_name=self.name,
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                figures=[],
                tables=[],
                extractor_name=self.name,
                success=False,
                error=str(e)
            )


class TableExtractor(BaseExtractor):
    """Abstract base class for table extractors."""

    @abstractmethod
    def extract_tables(self, pdf_path: Path) -> List[Table]:
        """
        Extract only tables from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of extracted tables
        """
        pass

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Extract tables (figures will be empty)."""
        try:
            tables = self.extract_tables(pdf_path)
            return ExtractionResult(
                figures=[],
                tables=tables,
                extractor_name=self.name,
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                figures=[],
                tables=[],
                extractor_name=self.name,
                success=False,
                error=str(e)
            )


class CombinedExtractor(BaseExtractor):
    """Abstract base class for extractors that handle both figures and tables."""

    @abstractmethod
    def extract_both(self, pdf_path: Path) -> tuple[List[Figure], List[Table]]:
        """
        Extract both figures and tables from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (figures, tables)
        """
        pass

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Extract both figures and tables."""
        try:
            figures, tables = self.extract_both(pdf_path)
            return ExtractionResult(
                figures=figures,
                tables=tables,
                extractor_name=self.name,
                success=True
            )
        except Exception as e:
            return ExtractionResult(
                figures=[],
                tables=[],
                extractor_name=self.name,
                success=False,
                error=str(e)
            )
