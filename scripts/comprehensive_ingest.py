#!/usr/bin/env python3
"""
Comprehensive Document Ingestion with Masking and Reference Tracking

This script combines PDF parsing, table/figure masking, reference detection,
and database ingestion into a single pipeline.

Features:
- Extracts layout from PDF using Docling
- Identifies and masks tables/figures using bbox intersection
- Saves masked tables/figures for later parsing
- Detects figure/table references in text
- Extracts clean text excluding tables/figures
- Ingests to database with full relationships

Usage:
    python scripts/comprehensive_ingest.py --pdf path/to/file.pdf --pmcid PMC1234567
    python scripts/comprehensive_ingest.py --pdf-dir files/organized_pdfs
"""

import sys
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging first (will be reconfigured based on args)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(debug=False):
    """Configure logging level based on debug flag."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Reconfigure logging
    )
    logger.setLevel(level)

# Import database
from database import get_db_connection, Document, TextElement, Figure, Table
from database.models import TextElementFigureReference, TextElementTableReference

# Import text processing utilities
from parsers.text_processing import remove_citations, ContextAwareStitcher

# Import mask_tables for table reconstruction and masking
try:
    sys.path.insert(0, str(Path(__file__).parent / "docling"))
    from mask_tables import process_pdf_with_masking
    MASK_TABLES_AVAILABLE = True
except ImportError:
    MASK_TABLES_AVAILABLE = False
    logger.warning("mask_tables.py not available")

# Import Docling for automatic extraction
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not available. Install with: pip install docling")


class ComprehensiveDocumentProcessor:
    """
    Processes documents with table/figure masking and database ingestion.
    """

    def __init__(
        self,
        output_dir: str = "out/comprehensive",
        figures_dir: str = "files/figures",
        tables_dir: str = "files/tables"
    ):
        """
        Initialize the processor.

        Args:
            output_dir: Directory for intermediate outputs
            figures_dir: Directory to store extracted figures
            tables_dir: Directory to store extracted tables
        """
        self.db = get_db_connection()
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.tables_dir = Path(tables_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'total_text_elements': 0,
            'total_figures': 0,
            'total_tables': 0,
            'masked_elements': 0
        }

    def _extract_with_docling(self, pdf_path: Path, output_path: Path) -> bool:
        """
        Extract full layout using Docling and save to JSON.

        Args:
            pdf_path: Path to PDF file
            output_path: Path to save JSON output

        Returns:
            True if successful
        """
        if not DOCLING_AVAILABLE:
            logger.error("Docling is not available. Install with: pip install docling")
            return False

        try:
            from datetime import datetime

            # Configure Docling pipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = False  # Coordinates only
            pipeline_options.do_ocr = True
            pipeline_options.images_scale = 2.0

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            # Convert document
            result = converter.convert(str(pdf_path))
            doc = result.document

            # Collect all elements
            all_elements = []
            for element, level in doc.iterate_items():
                # Get element label/type
                label_str = str(getattr(element, "label", "UNKNOWN")).split('.')[-1].upper()

                # Skip elements without spatial data
                if not (hasattr(element, 'prov') and element.prov):
                    continue

                prov = element.prov[0]
                bbox = prov.bbox

                # Get text content
                text_content = ""
                if hasattr(element, 'text'):
                    text_content = element.text
                elif hasattr(element, 'caption') and element.caption:
                    text_content = element.caption.text

                # Create element data
                item_data = {
                    "type": label_str,
                    "page": prov.page_no,
                    "level": level,
                    "bbox": {
                        "x1": bbox.l,
                        "y1": bbox.t,
                        "x2": bbox.r,
                        "y2": bbox.b
                    },
                    "text": text_content.strip() if text_content else None
                }

                all_elements.append(item_data)

            # Get page dimensions
            page_dimensions = {
                no: {"width": p.size.width, "height": p.size.height}
                for no, p in doc.pages.items()
            }

            # Prepare output data
            output_data = {
                "metadata": {
                    "pdf_path": str(pdf_path),
                    "tool": "Docling_v2_Full_Export",
                    "extraction_date": datetime.now().isoformat(),
                    "total_elements_found": len(all_elements)
                },
                "page_dimensions": page_dimensions,
                "elements": all_elements
            }

            # Save to JSON
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"Extracted {len(all_elements)} layout elements")
            return True

        except Exception as e:
            logger.error(f"Docling extraction failed: {e}", exc_info=True)
            return False

    def extract_docling_layout(self, pdf_path: Path) -> Optional[List[Dict]]:
        """
        Extract full layout from PDF using Docling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of elements with type, bbox, text, page
        """
        try:
            # Check if Docling JSON already exists
            json_path = Path("out/docling_full") / f"{pdf_path.stem}_full_layout.json"

            if not json_path.exists():
                logger.info(f"Extracting layout with Docling: {pdf_path.name}")
                # Extract layout using Docling
                if not self._extract_with_docling(pdf_path, json_path):
                    logger.error(f"Failed to extract layout from {pdf_path.name}")
                    return None

            with open(json_path, 'r') as f:
                data = json.load(f)

            elements = data.get('elements', [])
            logger.info(f"Loaded {len(elements)} elements from Docling")
            return elements

        except Exception as e:
            logger.error(f"Failed to extract Docling layout: {e}")
            return None

    def bbox_intersects(self, bbox1: Dict, bbox2: Dict) -> bool:
        """Check if two bounding boxes intersect."""
        if not bbox1 or not bbox2:
            return False

        x1_min = min(bbox1.get('x1', 0), bbox1.get('x2', 0))
        x1_max = max(bbox1.get('x1', 0), bbox1.get('x2', 0))
        y1_min = min(bbox1.get('y1', 0), bbox1.get('y2', 0))
        y1_max = max(bbox1.get('y1', 0), bbox1.get('y2', 0))

        x2_min = min(bbox2.get('x1', 0), bbox2.get('x2', 0))
        x2_max = max(bbox2.get('x1', 0), bbox2.get('x2', 0))
        y2_min = min(bbox2.get('y1', 0), bbox2.get('y2', 0))
        y2_max = max(bbox2.get('y1', 0), bbox2.get('y2', 0))

        no_overlap = (x1_max < x2_min or x2_max < x1_min or
                      y1_max < y2_min or y2_max < y1_min)

        return not no_overlap

    def collect_table_and_figure_regions(
        self,
        elements: List[Dict],
        threshold_multiplier: float = 1.2
    ) -> Tuple[List[Tuple], List[Tuple], List[Dict], List[Dict]]:
        """
        Collect all table and figure bounding boxes, plus their data.
        Ensures captions are always included in the cropped region.

        Returns:
            Tuple of (table_bboxes, figure_bboxes, table_data, figure_data)
            - table_bboxes: List of (page, bbox, caption) tuples
            - figure_bboxes: List of (page, bbox, caption) tuples
            - table_data: List of table metadata dicts
            - figure_data: List of figure metadata dicts
        """
        table_bboxes = []
        figure_bboxes = []
        table_data = []
        figure_data = []

        i = 0
        table_counter = 1
        figure_counter = 1

        while i < len(elements):
            el = elements[i]
            elem_type = el.get("type")
            page = el.get("page")

            # Native TABLE elements - look for adjacent caption
            if elem_type == "TABLE":
                bbox = el.get("bbox")
                caption = el.get("text", f"Table {table_counter}")
                caption_bbox = None

                # Check if next element is a caption for this table
                if i + 1 < len(elements):
                    next_el = elements[i + 1]
                    if (next_el.get("type") == "CAPTION" and
                        "table" in (next_el.get("text") or "").lower() and
                        next_el.get("page") == page):
                        caption = next_el.get("text", caption)
                        caption_bbox = next_el.get("bbox")

                # Check if previous element is a caption
                if i > 0:
                    prev_el = elements[i - 1]
                    if (prev_el.get("type") == "CAPTION" and
                        "table" in (prev_el.get("text") or "").lower() and
                        prev_el.get("page") == page):
                        caption = prev_el.get("text", caption)
                        caption_bbox = prev_el.get("bbox")

                # Combine bbox with caption if found
                if bbox and page:
                    if caption_bbox:
                        # Create combined bbox including caption
                        combined_bbox = {
                            "x1": min(bbox["x1"], caption_bbox["x1"]),
                            "y1": max(bbox["y1"], caption_bbox["y1"]),
                            "x2": max(bbox["x2"], caption_bbox["x2"]),
                            "y2": min(bbox["y2"], caption_bbox["y2"])
                        }
                        table_bboxes.append((page, combined_bbox, caption))
                        table_data.append({
                            'table_id': str(table_counter),
                            'caption': caption,
                            'page': page,
                            'bbox': combined_bbox,
                            'type': 'native_with_caption'
                        })
                    else:
                        table_bboxes.append((page, bbox, caption))
                        table_data.append({
                            'table_id': str(table_counter),
                            'caption': caption,
                            'page': page,
                            'bbox': bbox,
                            'type': 'native'
                        })
                    table_counter += 1
                i += 1
                continue

            # Native FIGURE/PICTURE elements - look for adjacent caption
            if elem_type in ["FIGURE", "PICTURE"]:
                bbox = el.get("bbox")
                caption = el.get("text", f"Figure {figure_counter}")
                caption_bbox = None

                # Check if next element is a caption for this figure
                if i + 1 < len(elements):
                    next_el = elements[i + 1]
                    if (next_el.get("type") == "CAPTION" and
                        ("figure" in (next_el.get("text") or "").lower() or
                         "fig" in (next_el.get("text") or "").lower()) and
                        next_el.get("page") == page):
                        caption = next_el.get("text", caption)
                        caption_bbox = next_el.get("bbox")

                # Check if previous element is a caption
                if i > 0:
                    prev_el = elements[i - 1]
                    if (prev_el.get("type") == "CAPTION" and
                        ("figure" in (prev_el.get("text") or "").lower() or
                         "fig" in (prev_el.get("text") or "").lower()) and
                        prev_el.get("page") == page):
                        caption = prev_el.get("text", caption)
                        caption_bbox = prev_el.get("bbox")

                # Combine bbox with caption if found
                if bbox and page:
                    if caption_bbox:
                        # Create combined bbox including caption
                        combined_bbox = {
                            "x1": min(bbox["x1"], caption_bbox["x1"]),
                            "y1": max(bbox["y1"], caption_bbox["y1"]),
                            "x2": max(bbox["x2"], caption_bbox["x2"]),
                            "y2": min(bbox["y2"], caption_bbox["y2"])
                        }
                        figure_bboxes.append((page, combined_bbox, caption))
                        figure_data.append({
                            'figure_id': str(figure_counter),
                            'caption': caption,
                            'page': page,
                            'bbox': combined_bbox,
                            'type': 'native_with_caption'
                        })
                    else:
                        figure_bboxes.append((page, bbox, caption))
                        figure_data.append({
                            'figure_id': str(figure_counter),
                            'caption': caption,
                            'page': page,
                            'bbox': bbox,
                            'type': 'native'
                        })
                    figure_counter += 1
                i += 1
                continue

            # Reconstructed tables from captions
            if elem_type == "CAPTION":
                text_lower = (el.get("text") or "").lower()
                caption_bbox = el.get("bbox")

                if "table" in text_lower:
                    sub_elements = []
                    max_allowed_gap = 20

                    # First, check if table content comes AFTER the caption
                    last_y2 = caption_bbox["y2"]
                    j = i + 1
                    content_after = []

                    while j < len(elements):
                        next_el = elements[j]
                        if next_el.get("type") not in ["TEXT", "LIST_ITEM"]:
                            break

                        current_y1 = next_el["bbox"]["y1"]
                        vertical_gap = abs(current_y1 - last_y2)

                        if len(content_after) == 1:
                            first_item_y2 = content_after[0]["bbox"]["y2"]
                            true_gutter = abs(current_y1 - first_item_y2)
                            max_allowed_gap = true_gutter * threshold_multiplier

                        if vertical_gap < max_allowed_gap:
                            content_after.append(next_el)
                            last_y2 = next_el["bbox"]["y2"]
                            j += 1
                        else:
                            break

                    # Second, check if table content comes BEFORE the caption
                    # (look backwards for LIST_ITEM/TEXT elements)
                    content_before = []
                    if i > 0:
                        k = i - 1
                        first_y1 = caption_bbox["y1"]

                        while k >= 0:
                            prev_el = elements[k]
                            if prev_el.get("type") not in ["TEXT", "LIST_ITEM"]:
                                break

                            prev_y2 = prev_el["bbox"]["y2"]
                            vertical_gap = abs(first_y1 - prev_y2)

                            # Use similar threshold
                            if vertical_gap < max_allowed_gap or len(content_before) < 3:
                                content_before.insert(0, prev_el)
                                first_y1 = prev_el["bbox"]["y1"]
                                k -= 1
                            else:
                                break

                    # Combine content before + caption + content after
                    all_content = content_before + content_after

                    if all_content:
                        all_bboxes = [caption_bbox] + [e["bbox"] for e in all_content]
                        combined_bbox = {
                            "x1": min(b["x1"] for b in all_bboxes),
                            "y1": max(b["y1"] for b in all_bboxes),
                            "x2": max(b["x2"] for b in all_bboxes),
                            "y2": min(b["y2"] for b in all_bboxes)
                        }
                        table_bboxes.append((page, combined_bbox, el.get("text", "")))
                        table_data.append({
                            'table_id': str(table_counter),
                            'caption': el.get("text", ""),
                            'page': page,
                            'bbox': combined_bbox,
                            'type': 'reconstructed',
                            'sub_elements': all_content,
                            'content_before': len(content_before),
                            'content_after': len(content_after)
                        })
                        table_counter += 1
                        # Skip the elements we consumed
                        i = j
                    else:
                        # Caption only, no content
                        table_bboxes.append((page, caption_bbox, el.get("text", "")))
                        table_data.append({
                            'table_id': str(table_counter),
                            'caption': el.get("text", ""),
                            'page': page,
                            'bbox': caption_bbox,
                            'type': 'caption_only'
                        })
                        table_counter += 1
                        i += 1

                elif "figure" in text_lower or "fig" in text_lower:
                    figure_bboxes.append((page, caption_bbox, el.get("text", "")))
                    figure_data.append({
                        'figure_id': str(figure_counter),
                        'caption': el.get("text", ""),
                        'page': page,
                        'bbox': caption_bbox,
                        'type': 'caption'
                    })
                    figure_counter += 1
                    i += 1
                    continue
                else:
                    i += 1
                    continue

            i += 1

        return table_bboxes, figure_bboxes, table_data, figure_data

    def detect_references(self, text: str) -> Dict[str, List[str]]:
        """
        Detect references to figures and tables in text.

        Args:
            text: Text string to search

        Returns:
            Dict with 'figures' and 'tables' lists containing reference IDs
        """
        references = {
            'figures': [],
            'tables': []
        }

        # Figure patterns
        fig_patterns = [
            r'\b(?:Figure|Fig\.?|FIG\.?)\s+(\d+[A-Za-z]?)',
            r'\((?:Figure|Fig\.?|FIG\.?)\s+(\d+[A-Za-z]?)\)',
        ]

        for pattern in fig_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                fig_id = match.group(1)
                if fig_id not in references['figures']:
                    references['figures'].append(fig_id)

        # Table patterns
        table_patterns = [
            r'\b(?:Table|TABLE)\s+(\d+[A-Za-z]?)',
            r'\((?:Table|TABLE)\s+(\d+[A-Za-z]?)\)',
        ]

        for pattern in table_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                table_id = match.group(1)
                if table_id not in references['tables']:
                    references['tables'].append(table_id)

        return references

    def extract_clean_text(
        self,
        elements: List[Dict],
        table_bboxes: List[Tuple],
        figure_bboxes: List[Tuple]
    ) -> List[Dict]:
        """
        Extract clean text elements, excluding those that intersect with tables/figures.

        Returns:
            List of text element dicts with references detected
        """
        keep_types = {'TEXT', 'PARAGRAPH', 'SECTION_HEADER', 'TITLE', 'LIST', 'LIST_ITEM'}
        text_elements = []
        masked_count = 0

        for element in elements:
            elem_type = element.get('type', 'UNKNOWN')
            elem_page = element.get('page')
            elem_bbox = element.get('bbox')
            elem_text = element.get('text') or ''
            elem_text = elem_text.strip()

            if not elem_text or elem_type not in keep_types:
                continue

            # Check if intersects with any table or figure
            intersects = False
            if elem_bbox and elem_page:
                for page, bbox, _ in table_bboxes + figure_bboxes:
                    if page == elem_page and self.bbox_intersects(elem_bbox, bbox):
                        intersects = True
                        masked_count += 1
                        break

            if not intersects:
                # Detect references in text
                refs = self.detect_references(elem_text)

                text_elements.append({
                    'type': elem_type,
                    'text': elem_text,
                    'page': elem_page,
                    'bbox': elem_bbox,
                    'references': refs if (refs['figures'] or refs['tables']) else {}
                })

        logger.info(f"Extracted {len(text_elements)} clean text elements")
        logger.info(f"Masked {masked_count} elements intersecting with tables/figures")
        self.stats['masked_elements'] = masked_count

        return text_elements

    def crop_and_save_regions(
        self,
        pdf_path: Path,
        pmcid: str,
        table_data: List[Dict],
        figure_data: List[Dict]
    ):
        """
        Crop and save table/figure regions from PDF as images.

        Args:
            pdf_path: Path to source PDF
            pmcid: PMC ID
            table_data: List of table metadata
            figure_data: List of figure metadata
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed - skipping image cropping")
            logger.warning("Install with: pip install PyMuPDF")
            return

        try:
            doc = fitz.open(str(pdf_path))

            # Crop tables
            for table in table_data:
                page_num = table['page']
                bbox = table['bbox']
                table_id = table['table_id']

                if page_num and bbox:
                    page = doc[page_num - 1]  # 0-indexed
                    page_height = page.rect.height

                    # Convert bbox to PyMuPDF rect (handle PDF coordinate system)
                    y1 = page_height - max(bbox['y1'], bbox['y2'])
                    y2 = page_height - min(bbox['y1'], bbox['y2'])
                    rect = fitz.Rect(bbox['x1'], y1, bbox['x2'], y2)

                    # Render region to pixmap
                    pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))  # 2x zoom for quality

                    # Save
                    output_path = self.tables_dir / f"{pmcid}_table_{table_id}.png"
                    pix.save(str(output_path))
                    table['image_path'] = str(output_path)

            logger.info(f"Cropped {len(table_data)} table regions")

            # Crop figures
            for figure in figure_data:
                page_num = figure['page']
                bbox = figure['bbox']
                fig_id = figure['figure_id']

                if page_num and bbox:
                    page = doc[page_num - 1]
                    page_height = page.rect.height

                    y1 = page_height - max(bbox['y1'], bbox['y2'])
                    y2 = page_height - min(bbox['y1'], bbox['y2'])
                    rect = fitz.Rect(bbox['x1'], y1, bbox['x2'], y2)

                    pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))

                    output_path = self.figures_dir / f"{pmcid}_figure_{fig_id}.png"
                    pix.save(str(output_path))
                    figure['image_path'] = str(output_path)

            logger.info(f"Cropped {len(figure_data)} figure regions")

            doc.close()

        except Exception as e:
            logger.error(f"Failed to crop regions: {e}")

    def save_masked_regions(
        self,
        pmcid: str,
        table_data: List[Dict],
        figure_data: List[Dict]
    ):
        """
        Save metadata about masked tables and figures.
        """
        # Save table metadata
        if table_data:
            tables_file = self.tables_dir / f"{pmcid}_tables.json"
            with open(tables_file, 'w') as f:
                json.dump(table_data, f, indent=2)
            logger.info(f"Saved {len(table_data)} table metadata to {tables_file}")

        # Save figure metadata
        if figure_data:
            figures_file = self.figures_dir / f"{pmcid}_figures.json"
            with open(figures_file, 'w') as f:
                json.dump(figure_data, f, indent=2)
            logger.info(f"Saved {len(figure_data)} figure metadata to {figures_file}")

    def create_hierarchical_structure(
        self,
        text_elements: List[Dict]
    ) -> List[Dict]:
        """
        Convert flat text elements into hierarchical structure using SECTION_HEADER elements.
        Skips sections that contain "References" in their header.

        Returns:
            List of dicts with: path_list, path_string, depth, text, references
        """
        hierarchical = []
        current_path = []
        current_depth = 0
        in_references = False  # Track if we're in a References section
        references_depth = None  # Track depth of References section

        for elem in text_elements:
            elem_type = elem['type']
            elem_text = elem['text']

            if elem_type == 'SECTION_HEADER':
                # New section - update path
                # Simple heuristic: if it looks like a subsection (starts with number), it's deeper
                # Otherwise, it's a new top-level section
                if re.match(r'^\d+\.', elem_text):  # e.g., "2.1 Methods"
                    # Count dots to determine depth
                    dot_count = elem_text.split()[0].count('.')
                    depth = dot_count + 1

                    # Truncate path to appropriate depth
                    current_path = current_path[:depth-1] + [elem_text]
                    current_depth = depth
                else:
                    # New top-level section
                    current_path = [elem_text]
                    current_depth = 1

                # Check if this is a References section or if we're exiting one
                if 'reference' in elem_text.lower():
                    in_references = True
                    references_depth = current_depth
                    continue  # Skip adding the References header itself
                elif in_references and references_depth is not None and current_depth <= references_depth:
                    # We've exited the References section (new section at same or shallower depth)
                    in_references = False
                    references_depth = None

                # Skip if we're in References section
                if in_references:
                    continue

                path_list = current_path.copy()
                path_string = ' > '.join(path_list)
                depth = current_depth

            else:
                # Skip if we're in References section
                if in_references:
                    continue

                # Regular text - use current section path
                path_list = current_path.copy()
                path_string = ' > '.join(path_list) if path_list else ''
                depth = current_depth

            hierarchical.append({
                'path_list': path_list,
                'path_string': path_string,
                'depth': depth,
                'text': elem_text,
                'page': elem.get('page'),
                'references': elem.get('references', {})
            })

        return hierarchical

    def ingest_to_database(
        self,
        pmcid: str,
        pdf_path: Path,
        text_elements: List[Dict],
        table_data: List[Dict],
        figure_data: List[Dict],
        title: Optional[str] = None,
        journal: Optional[str] = None,
        publication_year: Optional[int] = None,
        force: bool = False
    ) -> bool:
        """
        Ingest document, text, tables, and figures to database.

        Args:
            pmcid: PMC ID
            text_elements: List of text element dicts with hierarchical structure
            table_data: List of table metadata dicts
            figure_data: List of figure metadata dicts
            title: Document title
            journal: Journal name
            publication_year: Publication year
            force: If True, re-ingest even if document exists

        Returns:
            True if successful
        """
        try:
            with self.db.session_scope() as session:
                # Check if document exists
                existing = session.query(Document).filter_by(pmcid=pmcid).first()
                if existing and not force:
                    logger.info(f"Document {pmcid} already exists (use --force to re-ingest)")
                    self.stats['skipped'] += 1
                    return False

                # Delete existing if force
                if existing and force:
                    logger.info(f"Deleting existing document {pmcid}")
                    session.delete(existing)
                    session.flush()

                # Create document
                doc = Document(
                    pmcid=pmcid,
                    filename=pdf_path.name,
                    file_path=str(pdf_path.absolute()),
                    title=title or f"Document {pmcid}",
                    journal=journal,
                    publication_year=publication_year,
                    text_source='pdf'
                )
                session.add(doc)
                session.flush()

                logger.info(f"Created document: {pmcid}")

                # Add text elements
                path_counts = defaultdict(int)
                for elem in text_elements:
                    path_string = elem['path_string']
                    position = path_counts[path_string]
                    path_counts[path_string] += 1

                    unique_path = f"{pmcid}/{path_string}/{position}" if path_string else f"{pmcid}/(Root)/{position}"

                    text_elem = TextElement(
                        unique_path=unique_path,
                        document_id=doc.id,
                        path_list=elem['path_list'],
                        path_string=path_string,
                        depth=elem['depth'],
                        text_content=elem['text'],
                        position_in_section=position,
                        references=elem.get('references', {})
                    )
                    session.add(text_elem)

                session.flush()
                logger.info(f"Added {len(text_elements)} text elements")

                # Add figures
                for fig in figure_data:
                    # Extract filename if image_path exists
                    image_filename = None
                    image_path = fig.get('image_path')
                    if image_path:
                        image_filename = Path(image_path).name

                    figure = Figure(
                        document_id=doc.id,
                        figure_id=fig['figure_id'],
                        figure_label=f"Figure {fig['figure_id']}",
                        figure_number=fig['figure_id'],
                        caption_text=fig.get('caption'),
                        image_filename=image_filename,
                        image_path=image_path
                    )
                    session.add(figure)

                session.flush()
                logger.info(f"Added {len(figure_data)} figures")

                # Add tables
                for tbl in table_data:
                    # Extract filename if image_path exists
                    image_filename = None
                    image_path = tbl.get('image_path')
                    if image_path:
                        image_filename = Path(image_path).name

                    table = Table(
                        document_id=doc.id,
                        table_id=tbl['table_id'],
                        table_label=f"Table {tbl['table_id']}",
                        table_number=tbl['table_id'],
                        caption_text=tbl.get('caption'),
                        image_filename=image_filename,
                        image_path=image_path
                    )
                    session.add(table)

                session.flush()
                logger.info(f"Added {len(table_data)} tables")

                # Create reference relationships via junction tables
                text_elements_with_refs = session.query(TextElement).filter_by(document_id=doc.id).all()

                figure_refs_created = 0
                table_refs_created = 0

                for text_elem in text_elements_with_refs:
                    if not text_elem.references:
                        continue

                    # Figure references
                    for fig_id in text_elem.references.get('figures', []):
                        # Find matching figure
                        figure = session.query(Figure).filter_by(
                            document_id=doc.id,
                            figure_id=fig_id
                        ).first()

                        if figure:
                            ref = TextElementFigureReference(
                                text_element_id=text_elem.id,
                                figure_id=figure.id
                            )
                            session.add(ref)
                            figure_refs_created += 1

                    # Table references
                    for table_id in text_elem.references.get('tables', []):
                        # Find matching table
                        table = session.query(Table).filter_by(
                            document_id=doc.id,
                            table_id=table_id
                        ).first()

                        if table:
                            ref = TextElementTableReference(
                                text_element_id=text_elem.id,
                                table_id=table.id
                            )
                            session.add(ref)
                            table_refs_created += 1

                session.flush()
                logger.info(f"Created {figure_refs_created} figure references, {table_refs_created} table references")

                self.stats['processed'] += 1
                self.stats['total_text_elements'] += len(text_elements)
                self.stats['total_figures'] += len(figure_data)
                self.stats['total_tables'] += len(table_data)

                return True

        except Exception as e:
            logger.error(f"Failed to ingest {pmcid}: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False

    def create_masked_pdf(
        self,
        pdf_path: Path,
        pmcid: str,
        table_bboxes: List[Tuple[int, Dict, str]],
        figure_bboxes: List[Tuple[int, Dict, str]]
    ) -> Path:
        """
        Create a PDF with white rectangles masking tables and figures.

        Args:
            pdf_path: Path to original PDF
            pmcid: PMC ID for naming the output file
            table_bboxes: List of (page, bbox, caption) tuples for tables
            figure_bboxes: List of (page, bbox, caption) tuples for figures

        Returns:
            Path to masked PDF
        """
        import fitz  # PyMuPDF

        # Create output directory
        output_dir = Path("out/masked_pdfs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create masked PDF path
        masked_path = output_dir / f"{pmcid}_masked.pdf"

        try:
            # Open PDF
            doc = fitz.open(str(pdf_path))

            # Combine all bboxes by page
            masks_by_page = defaultdict(list)
            for page_num, bbox, caption in table_bboxes + figure_bboxes:
                masks_by_page[page_num].append(bbox)

            # Redact (properly mask) tables/figures
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_height = page.rect.height

                # Docling uses 1-indexed pages, PyMuPDF uses 0-indexed
                docling_page_num = page_num + 1

                if docling_page_num in masks_by_page:
                    for bbox in masks_by_page[docling_page_num]:
                        # Convert bbox to PyMuPDF coordinates (PDF uses bottom-up)
                        x1 = bbox['x1']
                        x2 = bbox['x2']
                        y1 = page_height - max(bbox['y1'], bbox['y2'])
                        y2 = page_height - min(bbox['y1'], bbox['y2'])

                        # Create rectangle
                        rect = fitz.Rect(x1, y1, x2, y2)

                        # Add redaction annotation (proper removal of content)
                        page.add_redact_annot(rect, fill=(1, 1, 1))

                    # Apply all redactions on this page
                    page.apply_redactions()

            # Save masked PDF
            doc.save(str(masked_path))
            doc.close()

            logger.info(f"Created masked PDF: {masked_path}")
            return masked_path

        except Exception as e:
            logger.error(f"Failed to create masked PDF: {e}", exc_info=True)
            # Clean up on error
            if masked_path.exists():
                masked_path.unlink()
            raise

    def save_text_to_file(self, pmcid: str, text_elements: List[Dict], stitch: bool = True) -> Path:
        """
        Save extracted text elements to a text file.
        Combines all elements with the same path together and optionally stitches split paragraphs.

        Args:
            pmcid: Document PMC ID
            text_elements: List of text elements with hierarchical structure
            stitch: If True, use ContextAwareStitcher to join split paragraphs

        Returns:
            Path to saved text file
        """
        output_dir = Path("out/text")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{pmcid}_text.txt"

        # Group elements by path_string while preserving order
        path_groups = defaultdict(list)
        path_order = []  # Track order of first appearance

        for elem in text_elements:
            path_key = elem['path_string'] if elem['path_string'] else '(Root)'
            if path_key not in path_groups:
                path_order.append(path_key)  # Record first appearance
            path_groups[path_key].append(elem)

        # Initialize stitcher if needed
        stitcher = ContextAwareStitcher() if stitch else None

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Document: {pmcid}\n")
            f.write(f"{'='*80}\n\n")

            for path_string in path_order:  # Use original order instead of sorted
                elements = path_groups[path_string]

                # Write section path once
                f.write(f"\n[{path_string}]\n")
                f.write(f"{'-'*len(path_string)}\n\n")

                # Prepare paragraphs for this section
                paragraphs = []
                for elem in elements:
                    # Remove citations first
                    cleaned_text = remove_citations(elem['text'])
                    paragraphs.append(cleaned_text)

                # Stitch paragraphs if enabled
                if stitch and stitcher:
                    paragraphs = stitcher.reconstruct_paragraphs(paragraphs)

                # Write stitched paragraphs
                for para in paragraphs:
                    if para.strip():  # Only write non-empty paragraphs
                        f.write(f"{para}\n\n")

                # Collect all references from elements in this section
                all_figure_refs = []
                all_table_refs = []
                for elem in elements:
                    if elem.get('references'):
                        refs = elem['references']
                        all_figure_refs.extend(refs.get('figures', []))
                        all_table_refs.extend(refs.get('tables', []))

                # Remove duplicates and write references for the section
                if all_figure_refs or all_table_refs:
                    f.write(f"\n  [Section References: ")
                    ref_parts = []
                    if all_figure_refs:
                        unique_figs = sorted(set(all_figure_refs))
                        ref_parts.append(f"Figures {', '.join(unique_figs)}")
                    if all_table_refs:
                        unique_tables = sorted(set(all_table_refs))
                        ref_parts.append(f"Tables {', '.join(unique_tables)}")
                    f.write('; '.join(ref_parts))
                    f.write("]\n")

        logger.info(f"Saved text to: {output_path}")
        return output_path

    def process_document_with_reconstruction(
        self,
        pdf_path: Path,
        pmcid: str,
        title: Optional[str] = None,
        journal: Optional[str] = None,
        publication_year: Optional[int] = None,
        force: bool = False,
        text_only: bool = True,
        stitch_paragraphs: bool = True
    ) -> bool:
        """
        Process document using mask_tables.py reconstruction logic.
        This uses the reconstruct_tables_from_lists approach.

        Args:
            pdf_path: Path to PDF file
            pmcid: PMC ID
            title: Document title
            journal: Journal name
            publication_year: Publication year
            force: If True, re-ingest even if exists
            text_only: If True, only extract text without database ingestion

        Returns:
            True if successful
        """
        if not MASK_TABLES_AVAILABLE:
            logger.error("mask_tables.py not available - cannot use reconstruction mode")
            return False

        logger.info(f"{'='*80}")
        logger.info(f"Processing with table reconstruction: {pmcid} ({pdf_path.name})")
        logger.info(f"{'='*80}")

        try:
            # 1. Extract layout from PDF
            elements = self.extract_docling_layout(pdf_path)
            if not elements:
                logger.error("Failed to extract layout")
                self.stats['errors'] += 1
                return False

            # 2. Get JSON path for mask_tables processing
            json_path = Path("out/docling_full") / f"{pdf_path.stem}_full_layout.json"
            if not json_path.exists():
                logger.error(f"Docling JSON not found: {json_path}")
                return False

            # 3. Process with mask_tables (includes table reconstruction)
            logger.info("Processing with table reconstruction and masking...")
            masked_pdf_path, text_elements, masked_elements = process_pdf_with_masking(
                pdf_path, json_path, Path("out/masked_pdfs")
            )

            logger.info(f"Extracted {len(text_elements)} text elements")
            logger.info(f"Masked {len(masked_elements)} table/figure elements")

            # 4. Create hierarchical structure from text elements
            logger.info("Creating hierarchical structure...")
            hierarchical_elements = self.create_hierarchical_structure(text_elements)

            # 5. Extract table/figure metadata from masked elements
            table_data = []
            figure_data = []
            table_counter = 1
            figure_counter = 1

            for elem in masked_elements:
                elem_type = elem.get('type')
                if elem_type in ['TABLE', 'RECONSTRUCTED_TABLE']:
                    table_data.append({
                        'table_id': str(table_counter),
                        'caption': elem.get('text', f'Table {table_counter}'),
                        'page': elem.get('page'),
                        'bbox': elem.get('bbox'),
                        'type': elem_type.lower()
                    })
                    table_counter += 1
                elif elem_type in ['FIGURE', 'PICTURE']:
                    figure_data.append({
                        'figure_id': str(figure_counter),
                        'caption': elem.get('text', f'Figure {figure_counter}'),
                        'page': elem.get('page'),
                        'bbox': elem.get('bbox'),
                        'type': elem_type.lower()
                    })
                    figure_counter += 1

            logger.info(f"Identified {len(table_data)} tables, {len(figure_data)} figures")

            # 6. Crop and save regions as images
            logger.info("Cropping table/figure regions...")
            self.crop_and_save_regions(pdf_path, pmcid, table_data, figure_data)

            # 7. Save metadata
            self.save_masked_regions(pmcid, table_data, figure_data)

            # 8. Save text to file or ingest to database
            if text_only:
                logger.info("Saving text to file...")
                if stitch_paragraphs:
                    logger.info("Using ContextAwareStitcher to join split paragraphs...")
                self.save_text_to_file(pmcid, hierarchical_elements, stitch=stitch_paragraphs)
                logger.info(f"✅ Successfully extracted text for {pmcid}")
                self.stats['processed'] += 1
                return True
            else:
                logger.info("Ingesting to database...")
                success = self.ingest_to_database(
                    pmcid=pmcid,
                    pdf_path=pdf_path,
                    text_elements=hierarchical_elements,
                    table_data=table_data,
                    figure_data=figure_data,
                    title=title,
                    journal=journal,
                    publication_year=publication_year,
                    force=force
                )

                if success:
                    logger.info(f"✅ Successfully processed {pmcid}")
                else:
                    logger.warning(f"⚠️  Skipped {pmcid}")

                return success

        except Exception as e:
            logger.error(f"❌ Failed to process {pmcid}: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False

    def process_document(
        self,
        pdf_path: Path,
        pmcid: str,
        title: Optional[str] = None,
        journal: Optional[str] = None,
        publication_year: Optional[int] = None,
        force: bool = False,
        text_only: bool = True,
        stitch_paragraphs: bool = True
    ) -> bool:
        """
        Process a single document through the full pipeline.

        Args:
            pdf_path: Path to PDF file
            pmcid: PMC ID
            title: Document title
            journal: Journal name
            publication_year: Publication year
            force: If True, re-ingest even if exists

        Returns:
            True if successful
        """
        logger.info(f"{'='*80}")
        logger.info(f"Processing: {pmcid} ({pdf_path.name})")
        logger.info(f"{'='*80}")

        try:
            # 1. Extract layout from PDF
            elements = self.extract_docling_layout(pdf_path)
            if not elements:
                logger.error("Failed to extract layout")
                self.stats['errors'] += 1
                return False

            # 2. Collect table and figure regions
            logger.info("Identifying tables and figures...")
            table_bboxes, figure_bboxes, table_data, figure_data = \
                self.collect_table_and_figure_regions(elements)

            logger.info(f"Found {len(table_data)} tables, {len(figure_data)} figures")

            # 3. Crop and save regions as images
            logger.info("Cropping table/figure regions...")
            self.crop_and_save_regions(pdf_path, pmcid, table_data, figure_data)

            # 4. Save metadata JSON
            self.save_masked_regions(pmcid, table_data, figure_data)

            # 5. Create masked PDF with white rectangles over tables/figures
            logger.info("Creating masked PDF...")
            masked_pdf_path = self.create_masked_pdf(pdf_path, pmcid, table_bboxes, figure_bboxes)

            # 6. Extract text from masked PDF
            logger.info("Extracting text from masked PDF...")
            # Extract layout from masked PDF
            masked_elements = self.extract_docling_layout(masked_pdf_path)

            # Filter to only text elements
            text_elements = []
            for elem in masked_elements:
                elem_type = elem['type']
                # Keep text-like elements, skip tables/figures
                if elem_type in ['TEXT', 'PARAGRAPH', 'SECTION_HEADER', 'LIST_ITEM']:
                    text_elements.append(elem)

            logger.info(f"Extracted {len(text_elements)} clean text elements from masked PDF")
            logger.info(f"Masked PDF saved at: {masked_pdf_path}")

            # 7. Create hierarchical structure
            logger.info("Creating hierarchical structure...")
            hierarchical_elements = self.create_hierarchical_structure(text_elements)

            # 8. Save text to file or ingest to database
            if text_only:
                logger.info("Saving text to file...")
                if stitch_paragraphs:
                    logger.info("Using ContextAwareStitcher to join split paragraphs...")
                self.save_text_to_file(pmcid, hierarchical_elements, stitch=stitch_paragraphs)
                logger.info(f"✅ Successfully extracted text for {pmcid}")
                self.stats['processed'] += 1
                return True
            else:
                logger.info("Ingesting to database...")
                success = self.ingest_to_database(
                    pmcid=pmcid,
                    pdf_path=pdf_path,
                    text_elements=hierarchical_elements,
                    table_data=table_data,
                    figure_data=figure_data,
                    title=title,
                    journal=journal,
                    publication_year=publication_year,
                    force=force
                )

                if success:
                    logger.info(f"✅ Successfully processed {pmcid}")
                else:
                    logger.warning(f"⚠️  Skipped {pmcid}")

                return success

        except Exception as e:
            logger.error(f"❌ Failed to process {pmcid}: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False

    def print_stats(self):
        """Print processing statistics."""
        logger.info(f"\n{'='*80}")
        logger.info("Processing Statistics")
        logger.info(f"{'='*80}")
        logger.info(f"Processed:        {self.stats['processed']}")
        logger.info(f"Skipped:          {self.stats['skipped']}")
        logger.info(f"Errors:           {self.stats['errors']}")
        logger.info(f"Text elements:    {self.stats['total_text_elements']}")
        logger.info(f"Figures:          {self.stats['total_figures']}")
        logger.info(f"Tables:           {self.stats['total_tables']}")
        logger.info(f"Masked elements:  {self.stats['masked_elements']}")
        logger.info(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive document processing with masking and ingestion"
    )
    parser.add_argument(
        "--pdf",
        default="files/organized_pdfs/PMC1448691_his_2369.pdf",
        help="Path to single PDF file (default: files/organized_pdfs/PMC1448691_his_2369.pdf)"
    )
    parser.add_argument(
        "--pmcid",
        help="PMC ID for the document"
    )
    parser.add_argument(
        "--pdf-dir",
        help="Directory containing PDF files (batch mode)"
    )
    parser.add_argument(
        "--title",
        help="Document title (optional)"
    )
    parser.add_argument(
        "--journal",
        help="Journal name (optional)"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Publication year (optional)"
    )
    parser.add_argument(
        "--force",
        action='store_true',
        help="Re-ingest even if document exists"
    )
    parser.add_argument(
        "--output-dir",
        default="out/comprehensive",
        help="Output directory for intermediate files"
    )
    parser.add_argument(
        "--text-only",
        action='store_true',
        help="Extract text only without database ingestion"
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Enable debug logging"
    )
    parser.add_argument(
        "--use-reconstruction",
        action='store_true',
        help="Use mask_tables.py reconstruction logic (caption-based table grouping)"
    )
    parser.add_argument(
        "--stitch-paragraphs",
        action='store_true',
        default=True,
        help="Use ContextAwareStitcher to join split paragraphs (default: True)"
    )
    parser.add_argument(
        "--no-stitch",
        action='store_false',
        dest='stitch_paragraphs',
        help="Disable paragraph stitching"
    )

    args = parser.parse_args()

    # Setup logging based on debug flag
    setup_logging(args.debug)

    # Create processor
    processor = ComprehensiveDocumentProcessor(output_dir=args.output_dir)

    # Batch mode takes priority
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        for pdf_file in pdf_files:
            # Extract PMCID from filename
            stem = pdf_file.stem
            if stem.startswith("PMC"):
                pmcid = stem.split('_')[0]
            else:
                logger.warning(f"Skipping {pdf_file.name} - cannot extract PMCID")
                continue

            # Use reconstruction mode if requested
            if args.use_reconstruction:
                processor.process_document_with_reconstruction(
                    pdf_path=pdf_file,
                    pmcid=pmcid,
                    force=args.force,
                    text_only=args.text_only,
                    stitch_paragraphs=args.stitch_paragraphs
                )
            else:
                processor.process_document(
                    pdf_path=pdf_file,
                    pmcid=pmcid,
                    force=args.force,
                    text_only=args.text_only,
                    stitch_paragraphs=args.stitch_paragraphs
                )

        processor.print_stats()

    # Single file mode (default or explicitly specified)
    else:
        if not args.pmcid:
            # Try to extract PMCID from filename
            stem = Path(args.pdf).stem
            if stem.startswith("PMC"):
                args.pmcid = stem.split('_')[0]
            else:
                logger.error("--pmcid is required when filename doesn't start with PMC")
                sys.exit(1)

        # Use reconstruction mode if requested
        if args.use_reconstruction:
            processor.process_document_with_reconstruction(
                pdf_path=Path(args.pdf),
                pmcid=args.pmcid,
                title=args.title,
                journal=args.journal,
                publication_year=args.year,
                force=args.force,
                text_only=args.text_only,
                stitch_paragraphs=args.stitch_paragraphs
            )
        else:
            processor.process_document(
                pdf_path=Path(args.pdf),
                pmcid=args.pmcid,
                title=args.title,
                journal=args.journal,
                publication_year=args.year,
                force=args.force,
                text_only=args.text_only,
                stitch_paragraphs=args.stitch_paragraphs
            )


if __name__ == "__main__":
    main()
