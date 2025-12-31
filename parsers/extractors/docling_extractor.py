"""
Docling extractor implementation.
"""

from pathlib import Path
from typing import List, Tuple

from .base import CombinedExtractor, Figure, Table, BoundingBox
from .registry import ExtractorRegistry


@ExtractorRegistry.register('docling')
class DoclingExtractor(CombinedExtractor):
    """
    Extractor using Docling AI models.

    Uses deep learning models for figure and table detection.
    """

    def __init__(self, do_ocr: bool = False, do_table_structure: bool = True):
        """
        Initialize Docling extractor.

        Args:
            do_ocr: Enable OCR for text extraction
            do_table_structure: Enable table structure extraction
        """
        super().__init__('docling')
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure

    def is_available(self) -> bool:
        """Check if Docling is installed."""
        try:
            from docling.document_converter import DocumentConverter
            return True
        except ImportError:
            return False

    def extract_both(self, pdf_path: Path) -> Tuple[List[Figure], List[Table]]:
        """
        Extract both figures and tables using Docling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (figures, tables)
        """
        from docling.document_converter import DocumentConverter

        # Create converter
        converter = DocumentConverter()

        # Convert document
        result = converter.convert(str(pdf_path))
        doc = result.document

        figures = []
        tables = []

        # Iterate through document elements
        for element, level in doc.iterate_items():
            element_type = type(element).__name__

            # Extract figures
            if element_type == "PictureItem":
                prov = element.prov[0] if element.prov else None
                if prov and hasattr(prov, 'bbox'):
                    bbox_obj = prov.bbox
                    bbox = BoundingBox(
                        x1=bbox_obj.l,
                        y1=bbox_obj.t,
                        x2=bbox_obj.r,
                        y2=bbox_obj.b,
                        page=prov.page_no if hasattr(prov, 'page_no') else 1
                    )

                    figures.append(Figure(
                        bbox=bbox,
                        caption=element.caption.text if hasattr(element, 'caption') and element.caption else None,
                        metadata={
                            'text': element.text if hasattr(element, 'text') else None
                        }
                    ))

            # Extract tables
            elif element_type == "TableItem":
                prov = element.prov[0] if element.prov else None
                if prov and hasattr(prov, 'bbox'):
                    bbox_obj = prov.bbox
                    bbox = BoundingBox(
                        x1=bbox_obj.l,
                        y1=bbox_obj.t,
                        x2=bbox_obj.r,
                        y2=bbox_obj.b,
                        page=prov.page_no if hasattr(prov, 'page_no') else 1
                    )

                    tables.append(Table(
                        bbox=bbox,
                        caption=element.caption.text if hasattr(element, 'caption') and element.caption else None,
                        metadata={
                            'num_rows': element.num_rows if hasattr(element, 'num_rows') else None,
                            'num_cols': element.num_cols if hasattr(element, 'num_cols') else None,
                            'text': element.text if hasattr(element, 'text') else None
                        }
                    ))

        return figures, tables

    def get_config(self):
        """Get current configuration."""
        return {
            'do_ocr': self.do_ocr,
            'do_table_structure': self.do_table_structure
        }
