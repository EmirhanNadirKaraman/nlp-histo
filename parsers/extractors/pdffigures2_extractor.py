"""
PDFfigures2 extractor implementation.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Tuple

from .base import CombinedExtractor, Figure, Table, BoundingBox
from .registry import ExtractorRegistry


@ExtractorRegistry.register('pdffigures2')
class PDFFigures2Extractor(CombinedExtractor):
    """
    Extractor using PDFfigures2 (Java-based tool).

    Extracts both figures and tables with bounding boxes and captions.
    """

    def __init__(self, jar_path: str = 'pdffigures2.jar', dpi: int = 150):
        """
        Initialize PDFfigures2 extractor.

        Args:
            jar_path: Path to pdffigures2.jar file
            dpi: DPI for figure extraction
        """
        super().__init__('pdffigures2')
        self.jar_path = jar_path
        self.dpi = dpi

    def is_available(self) -> bool:
        """Check if Java and pdffigures2.jar are available."""
        try:
            # Check Java
            subprocess.run(
                ['java', '-version'],
                capture_output=True,
                check=True
            )
            # Check JAR file
            return Path(self.jar_path).exists()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def extract_both(self, pdf_path: Path) -> Tuple[List[Figure], List[Table]]:
        """
        Extract both figures and tables using PDFfigures2.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (figures, tables)
        """
        import tempfile

        # Create temp directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / 'output.json'

            # Run pdffigures2
            cmd = [
                'java', '-jar', self.jar_path,
                str(pdf_path),
                '-m', str(output_json),
                '-d', str(tmpdir),
                '-i', str(self.dpi)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"PDFfigures2 failed: {result.stderr}")

            # Parse results
            if not output_json.exists():
                return [], []

            with open(output_json, 'r') as f:
                data = json.load(f)

            figures = []
            tables = []

            for item in data:
                fig_type = item.get('figType')
                bbox_data = item.get('regionBoundary', {})

                bbox = BoundingBox(
                    x1=bbox_data.get('x1', 0),
                    y1=bbox_data.get('y1', 0),
                    x2=bbox_data.get('x2', 0),
                    y2=bbox_data.get('y2', 0),
                    page=item.get('page', 1)
                )

                if fig_type == 'Figure':
                    figures.append(Figure(
                        bbox=bbox,
                        figure_id=item.get('name'),
                        caption=item.get('caption'),
                        image_path=item.get('renderURL'),
                        metadata={
                            'caption_boundary': item.get('captionBoundary'),
                            'render_dpi': item.get('renderDpi')
                        }
                    ))
                elif fig_type == 'Table':
                    tables.append(Table(
                        bbox=bbox,
                        table_id=item.get('name'),
                        caption=item.get('caption'),
                        metadata={
                            'caption_boundary': item.get('captionBoundary'),
                            'image_text': item.get('imageText', [])
                        }
                    ))

            return figures, tables

    def get_config(self):
        """Get current configuration."""
        return {
            'jar_path': self.jar_path,
            'dpi': self.dpi
        }
