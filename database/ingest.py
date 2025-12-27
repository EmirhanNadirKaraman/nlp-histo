#!/usr/bin/env python3
"""
Unified Database Ingestion Script

This script provides a complete solution for ingesting documents (XML/PDF) into the database.
It handles:
- Text elements with hierarchical structure
- Figures and tables from PDFFigures2 JSON
- Figure/table references in text
- Junction table relationships
- Both single-file and batch processing

Usage:
    # Single file ingestion
    python database/ingest.py --pdf path/to/file.pdf --pmcid PMC1234567

    # Batch ingestion from XML directory
    python database/ingest.py --xml-dir files/organized_xmls

    # Use as a library
    from database.ingest import ingest_document
    ingest_document(pdf_path="...", pmcid="...", ...)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import parsers
from parsers.xml_parsers.hierarchical_parser import extract_hierarchy as extract_hierarchy_from_xml
from parsers.xml_parsers.figure_parser import extract_figures as extract_figures_from_xml
from lxml import etree
import shutil
import re

# Import database
from database import get_db_connection, Document, TextElement, Figure, Table
from database.models import TextElementFigureReference, TextElementTableReference

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles ingestion of documents into the database."""

    def __init__(self, images_dir: str = "files/figures"):
        """
        Initialize the ingester.

        Args:
            images_dir: Directory to store extracted images
        """
        self.db = get_db_connection()
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'total_text_elements': 0,
            'total_figures': 0,
            'total_tables': 0,
            'from_xml': 0,
            'from_pdf': 0
        }

    def extract_metadata_from_xml(self, xml_path: Path) -> Dict[str, Optional[str]]:
        """
        Extract metadata (title, journal, year) from XML file.

        Args:
            xml_path: Path to XML file

        Returns:
            Dictionary with metadata
        """
        metadata = {
            'title': None,
            'journal': None,
            'publication_year': None
        }

        try:
            tree = etree.parse(str(xml_path))
            root = tree.getroot()

            # Extract title
            title_elem = root.find('.//article-title')
            if title_elem is not None:
                metadata['title'] = ''.join(title_elem.itertext()).strip()

            # Extract journal
            journal_elem = root.find('.//journal-title')
            if journal_elem is not None:
                metadata['journal'] = ''.join(journal_elem.itertext()).strip()

            # Extract publication year
            pub_year_elem = root.find('.//pub-date[@pub-type="epub"]/year')
            if pub_year_elem is None:
                pub_year_elem = root.find('.//pub-date/year')
            if pub_year_elem is not None and pub_year_elem.text:
                try:
                    metadata['publication_year'] = int(pub_year_elem.text)
                except ValueError:
                    pass

        except Exception as e:
            logger.warning(f"Could not extract all metadata: {e}")

        return metadata

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

        # Figure patterns: "Figure 6", "Fig. 6", "figure 6", "FIG 6"
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

        # Table patterns: "Table 2", "table 2", "TABLE 2"
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

    def add_references_to_elements(self, text_elements: List[Dict]) -> List[Dict]:
        """
        Add figure/table references to each text element.

        Args:
            text_elements: List of text element dicts

        Returns:
            Enhanced text elements with 'references' field
        """
        enhanced = []

        for elem in text_elements:
            elem_copy = elem.copy()
            refs = self.detect_references(elem['text'])

            # Only add references field if there are any
            if refs['figures'] or refs['tables']:
                elem_copy['references'] = refs

            enhanced.append(elem_copy)

        return enhanced

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text elements from PDF using the ensemble parser.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of text element dicts with references
        """
        from parsers.pdf_parsers.ensemble_parser import EnsemblePDFParser

        parser = EnsemblePDFParser(mode='routing')
        result = parser.extract_hierarchy(str(pdf_path))

        if not result['success']:
            raise Exception(f"PDF extraction failed: {result.get('error', 'Unknown error')}")

        text_elements = result['text_elements']

        # Add references
        text_elements = self.add_references_to_elements(text_elements)

        return text_elements

    def find_pdffigures_json(self, source_path: Path, pmcid: str) -> Optional[Path]:
        """
        Find PDFFigures2 JSON output for a document.

        Args:
            source_path: Path to source file (PDF or XML)
            pmcid: PMC ID

        Returns:
            Path to JSON file if found, None otherwise
        """
        # Common locations
        search_paths = [
            Path('out/data') / f"{source_path.stem}.json",
            Path('test_figures') / f"{source_path.stem}.json",
            Path('output') / pmcid / 'pdffigures2.json',
            source_path.parent / f"{source_path.stem}.json",
        ]

        for json_path in search_paths:
            if json_path.exists():
                return json_path

        return None

    def find_and_copy_image(self, pmcid: str, graphic_ref: str, source_path: Path) -> Dict[str, Optional[str]]:
        """
        Find and copy an image file to the figures directory.

        Args:
            pmcid: PubMed Central ID
            graphic_ref: Original graphic reference
            source_path: Path to source file

        Returns:
            Dict with image_filename, image_path, and image_format
        """
        result = {
            'image_filename': None,
            'image_path': None,
            'image_format': None
        }

        if not graphic_ref:
            return result

        # Look for the image file
        source_dir = source_path.parent

        # Try to find the image file (may have different extensions)
        possible_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif']
        graphic_base = Path(graphic_ref).stem

        source_image = None
        for ext in possible_extensions:
            candidate = source_dir / f"{graphic_base}{ext}"
            if candidate.exists():
                source_image = candidate
                break

        # Also try the exact filename
        if not source_image:
            candidate = source_dir / graphic_ref
            if candidate.exists():
                source_image = candidate

        if source_image:
            # Create unique filename: PMCID_originalname.ext
            dest_filename = f"{pmcid}_{source_image.name}"
            dest_path = self.images_dir / dest_filename

            # Copy the file
            try:
                shutil.copy2(source_image, dest_path)
                result['image_filename'] = dest_filename
                result['image_path'] = str(dest_path.absolute())
                result['image_format'] = source_image.suffix.lstrip('.')
            except Exception as e:
                logger.warning(f"Failed to copy image {source_image.name}: {e}")

        return result

    def ingest_document(
        self,
        pmcid: str,
        source_path: Optional[Path] = None,
        pdf_path: Optional[Path] = None,
        xml_path: Optional[Path] = None,
        text_elements: Optional[List[Dict]] = None,
        pdffigures_json_path: Optional[Path] = None,
        title: Optional[str] = None,
        journal: Optional[str] = None,
        publication_year: Optional[int] = None,
        force: bool = False
    ) -> bool:
        """
        Ingest a document with all its data into the database.

        Args:
            pmcid: PMC ID of the document
            source_path: Path to source file (if not using pdf_path or xml_path)
            pdf_path: Path to PDF file (overrides source_path)
            xml_path: Path to XML file (overrides source_path)
            text_elements: Pre-parsed text elements (optional)
            pdffigures_json_path: Path to PDFFigures2 JSON (optional, will auto-detect)
            title: Document title (optional, will extract from XML if available)
            journal: Journal name (optional, will extract from XML if available)
            publication_year: Publication year (optional, will extract from XML if available)
            force: If True, re-ingest even if document exists

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"{'='*80}")
            logger.info(f"Ingesting: {pmcid}")
            logger.info(f"{'='*80}")

            # Determine source path
            if pdf_path:
                source_path = pdf_path
                source_type = 'pdf'
            elif xml_path:
                source_path = xml_path
                source_type = 'xml'
            elif source_path:
                source_type = 'pdf' if source_path.suffix.lower() == '.pdf' else 'xml'
            else:
                raise ValueError("Must provide source_path, pdf_path, or xml_path")

            # Check if already processed
            with self.db.session_scope() as session:
                existing = session.query(Document).filter_by(pmcid=pmcid).first()
                if existing and not force:
                    logger.info(f"Document {pmcid} already exists. Use force=True to re-ingest.")
                    self.stats['skipped'] += 1
                    return True
                elif existing and force:
                    logger.info(f"Deleting existing document {pmcid}...")
                    session.delete(existing)
                    session.commit()

            # Extract text elements if not provided
            if text_elements is None:
                if source_type == 'xml':
                    logger.info("Extracting text from XML...")
                    text_elements = extract_hierarchy_from_xml(str(source_path))
                    text_elements = self.add_references_to_elements(text_elements)

                    # Extract metadata from XML
                    if not title or not journal:
                        metadata = self.extract_metadata_from_xml(source_path)
                        title = title or metadata['title']
                        journal = journal or metadata['journal']
                        publication_year = publication_year or metadata['publication_year']
                else:
                    logger.info("Extracting text from PDF...")
                    text_elements = self.extract_text_from_pdf(source_path)

            logger.info(f"  ✓ Extracted {len(text_elements)} text elements")

            # Find PDFFigures2 JSON if not provided
            if pdffigures_json_path is None:
                pdffigures_json_path = self.find_pdffigures_json(source_path, pmcid)

            # Start database transaction
            with self.db.session_scope() as session:
                # 1. Create document
                logger.info("Creating document record...")
                doc = Document(
                    pmcid=pmcid,
                    filename=source_path.name,
                    file_path=str(source_path.absolute()),
                    title=title,
                    journal=journal,
                    publication_year=publication_year,
                    text_source=source_type
                )
                session.add(doc)
                session.flush()  # Get document ID
                logger.info(f"  ✓ Document ID: {doc.id}")

                # 2. Store text elements
                logger.info(f"Storing {len(text_elements)} text elements...")
                position_counter = defaultdict(int)
                text_elem_map = {}  # unique_path -> TextElement object

                for elem in text_elements:
                    path_string = elem.get('path_string', '(Root)')

                    # Create unique path
                    position = position_counter[path_string]
                    unique_path = f"{pmcid}/{path_string}/{position}"
                    position_counter[path_string] += 1

                    # Create text element
                    text_elem = TextElement(
                        document_id=doc.id,
                        unique_path=unique_path,
                        path_list=elem.get('path_list', []),
                        path_string=path_string,
                        depth=elem.get('depth', 0),
                        position_in_section=position,
                        text_content=elem.get('text', ''),
                        references=elem.get('references')  # Store as JSON
                    )

                    text_elem.calculate_metadata()
                    session.add(text_elem)
                    text_elem_map[unique_path] = text_elem

                session.flush()
                logger.info(f"  ✓ Stored {len(text_elements)} text elements")

                # 3. Process PDFFigures2 data (if available)
                figures_data = []
                tables_data = []

                if pdffigures_json_path and pdffigures_json_path.exists():
                    logger.info(f"Processing PDFFigures2 data: {pdffigures_json_path}")

                    with open(pdffigures_json_path, 'r', encoding='utf-8') as f:
                        pdffigures_data = json.load(f)

                    # Separate figures and tables
                    figures_data = [item for item in pdffigures_data if item.get('figType') == 'Figure']
                    tables_data = [item for item in pdffigures_data if item.get('figType') == 'Table']

                    logger.info(f"  Found {len(figures_data)} figures, {len(tables_data)} tables")
                elif source_type == 'xml':
                    # Try extracting figures from XML
                    logger.info("Extracting figures from XML...")
                    figures_data = extract_figures_from_xml(str(source_path))
                    logger.info(f"  Found {len(figures_data)} figures in XML")

                # 4. Store figures
                figure_map = {}  # figure_number -> Figure object

                if figures_data:
                    logger.info(f"Storing {len(figures_data)} figures...")

                    for fig in figures_data:
                        # Extract figure number
                        fig_id = fig.get('name', '') or fig.get('figureID', '') or fig.get('figure_id', '')

                        # Try to extract number (e.g., "Figure 1" -> "1", "Fig 2A" -> "2A")
                        caption = fig.get('caption', '') or fig.get('caption_text', '')
                        match = re.search(r'(\d+[A-Za-z]?)', caption or fig_id)
                        fig_number = match.group(1) if match else None

                        # Handle image paths
                        image_filename = None
                        image_path = None
                        image_format = None

                        # Check if PDFFigures2 format
                        if 'renderURL' in fig:
                            pdf_base = source_path.stem
                            image_filename = f"{pdf_base}_figure_{fig_id}.png" if fig_id else None
                            image_path = f"out/figures/{image_filename}" if image_filename else None
                        # Check if XML format
                        elif 'graphic_ref' in fig:
                            img_info = self.find_and_copy_image(pmcid, fig['graphic_ref'], source_path)
                            image_filename = img_info['image_filename']
                            image_path = img_info['image_path']
                            image_format = img_info['image_format']

                        figure = Figure(
                            document_id=doc.id,
                            figure_id=fig_id,
                            figure_label=fig.get('name', '') or fig.get('figure_label', ''),
                            figure_number=fig_number,
                            caption_text=caption,
                            graphic_ref=fig.get('renderURL', '') or fig.get('graphic_ref', ''),
                            image_filename=image_filename,
                            image_path=image_path,
                            image_format=image_format,
                            section_context=fig.get('section_context')
                        )

                        session.add(figure)
                        if fig_number:
                            figure_map[fig_number] = figure

                    session.flush()
                    logger.info(f"  ✓ Stored {len(figures_data)} figures")

                # 5. Store tables
                table_map = {}  # table_number -> Table object

                if tables_data:
                    logger.info(f"Storing {len(tables_data)} tables...")

                    pdf_base = source_path.stem

                    for tbl in tables_data:
                        # Extract table info
                        tbl_id = tbl.get('name', '') or tbl.get('tableID', '')

                        # Extract table number
                        match = re.search(r'(\d+|[IVX]+)', tbl.get('caption', ''))
                        tbl_number = match.group(1) if match else None

                        # Get bounding box
                        bbox = tbl.get('regionBoundary', {})

                        # Build image paths
                        image_filename = f"{pdf_base}_table_{tbl_id}.png" if tbl_id else None
                        image_path = f"out/tables/{image_filename}" if image_filename else None

                        table = Table(
                            document_id=doc.id,
                            table_id=tbl_id,
                            table_label=tbl.get('name', ''),
                            table_number=tbl_number,
                            caption_text=tbl.get('caption', ''),
                            image_filename=image_filename,
                            image_path=image_path,
                            page_number=tbl.get('page'),
                            bbox_x1=bbox.get('x1'),
                            bbox_y1=bbox.get('y1'),
                            bbox_x2=bbox.get('x2'),
                            bbox_y2=bbox.get('y2')
                        )

                        session.add(table)
                        if tbl_number:
                            table_map[tbl_number] = table

                    session.flush()
                    logger.info(f"  ✓ Stored {len(tables_data)} tables")

                # 6. Create junction table entries for references
                logger.info("Creating reference links...")

                figure_ref_count = 0
                table_ref_count = 0

                for unique_path, text_elem in text_elem_map.items():
                    refs = text_elem.references
                    if not refs:
                        continue

                    # Link to figures
                    figure_refs = refs.get('figures', [])
                    for fig_num in figure_refs:
                        if fig_num in figure_map:
                            ref_link = TextElementFigureReference(
                                text_element_id=text_elem.id,
                                figure_id=figure_map[fig_num].id
                            )
                            session.add(ref_link)
                            figure_ref_count += 1

                    # Link to tables
                    table_refs = refs.get('tables', [])
                    for tbl_num in table_refs:
                        if tbl_num in table_map:
                            ref_link = TextElementTableReference(
                                text_element_id=text_elem.id,
                                table_id=table_map[tbl_num].id
                            )
                            session.add(ref_link)
                            table_ref_count += 1

                logger.info(f"  ✓ Created {figure_ref_count} figure links, {table_ref_count} table links")

                # Print summary
                logger.info(f"{'='*80}")
                logger.info("Ingestion Summary")
                logger.info(f"{'='*80}")
                logger.info(f"Document:        {pmcid}")
                logger.info(f"Text elements:   {len(text_elements)}")
                logger.info(f"Figures:         {len(figures_data)}")
                logger.info(f"Tables:          {len(tables_data)}")
                logger.info(f"Figure links:    {figure_ref_count}")
                logger.info(f"Table links:     {table_ref_count}")
                logger.info(f"{'='*80}")

                # Update stats
                self.stats['processed'] += 1
                self.stats['total_text_elements'] += len(text_elements)
                self.stats['total_figures'] += len(figures_data)
                self.stats['total_tables'] += len(tables_data)

                if source_type == 'pdf':
                    self.stats['from_pdf'] += 1
                else:
                    self.stats['from_xml'] += 1

                return True

        except Exception as e:
            logger.error(f"Error ingesting {pmcid}: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False

    def ingest_directory(
        self,
        directory: Path,
        pattern: str = "*.nxml",
        force: bool = False
    ):
        """
        Ingest all documents from a directory.

        Args:
            directory: Directory containing files
            pattern: Glob pattern for files (default: *.nxml)
            force: If True, re-ingest existing documents
        """
        # Find all files
        files = list(directory.glob(pattern))

        # Also check for .xml and .pdf files if using default pattern
        if pattern == "*.nxml":
            files.extend(list(directory.glob("*.xml")))
            files.extend(list(directory.glob("*.pdf")))

        if not files:
            logger.warning(f"No files found matching {pattern} in {directory}")
            return

        logger.info(f"Found {len(files)} files. Starting batch ingestion...\n")

        # Process each file
        for file_path in sorted(files):
            # Extract PMCID from filename
            pmcid = file_path.stem.split('_')[0]  # Handle files like "PMC1234567_his_2369.pdf"

            self.ingest_document(
                pmcid=pmcid,
                source_path=file_path,
                force=force
            )

    def print_summary(self):
        """Print processing summary statistics."""
        separator = "=" * 80
        logger.info(f"\n{separator}")
        logger.info("Batch Processing Complete!")
        logger.info(separator)
        logger.info(f"Successfully processed: {self.stats['processed']}")
        logger.info(f"  - From XML: {self.stats['from_xml']}")
        logger.info(f"  - From PDF: {self.stats['from_pdf']}")
        logger.info(f"Already in database: {self.stats['skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Total text elements: {self.stats['total_text_elements']}")
        logger.info(f"Total figures: {self.stats['total_figures']}")
        logger.info(f"Total tables: {self.stats['total_tables']}")
        if self.stats['processed'] > 0:
            avg_text = self.stats['total_text_elements'] / self.stats['processed']
            avg_figs = self.stats['total_figures'] / self.stats['processed']
            avg_tables = self.stats['total_tables'] / self.stats['processed']
            logger.info(f"Average text elements per document: {avg_text:.1f}")
            logger.info(f"Average figures per document: {avg_figs:.1f}")
            logger.info(f"Average tables per document: {avg_tables:.1f}")
        logger.info(separator)


# Convenience function for single document ingestion
def ingest_document(
    pmcid: str,
    source_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    xml_path: Optional[str] = None,
    text_elements: Optional[List[Dict]] = None,
    pdffigures_json_path: Optional[str] = None,
    title: Optional[str] = None,
    journal: Optional[str] = None,
    publication_year: Optional[int] = None,
    force: bool = False
) -> bool:
    """
    Convenience function to ingest a single document.

    Args:
        pmcid: PMC ID of the document
        source_path: Path to source file (PDF or XML)
        pdf_path: Path to PDF file (overrides source_path)
        xml_path: Path to XML file (overrides source_path)
        text_elements: Pre-parsed text elements (optional)
        pdffigures_json_path: Path to PDFFigures2 JSON (optional)
        title: Document title (optional)
        journal: Journal name (optional)
        publication_year: Publication year (optional)
        force: If True, re-ingest even if document exists

    Returns:
        True if successful, False otherwise
    """
    ingester = DocumentIngester()

    # Convert string paths to Path objects
    if source_path:
        source_path = Path(source_path)
    if pdf_path:
        pdf_path = Path(pdf_path)
    if xml_path:
        xml_path = Path(xml_path)
    if pdffigures_json_path:
        pdffigures_json_path = Path(pdffigures_json_path)

    return ingester.ingest_document(
        pmcid=pmcid,
        source_path=source_path,
        pdf_path=pdf_path,
        xml_path=xml_path,
        text_elements=text_elements,
        pdffigures_json_path=pdffigures_json_path,
        title=title,
        journal=journal,
        publication_year=publication_year,
        force=force
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified database ingestion for XML/PDF documents'
    )

    # Single file ingestion
    parser.add_argument('--pmcid', help='PMC ID of the document')
    parser.add_argument('--pdf', help='Path to PDF file')
    parser.add_argument('--xml', help='Path to XML file')
    parser.add_argument('--pdffigures-json', help='Path to PDFFigures2 JSON output')
    parser.add_argument('--title', help='Document title')
    parser.add_argument('--journal', help='Journal name')
    parser.add_argument('--year', type=int, help='Publication year')

    # Batch ingestion
    parser.add_argument('--xml-dir', help='Directory containing XML files')
    parser.add_argument('--pdf-dir', help='Directory containing PDF files')
    parser.add_argument('--pattern', default='*.nxml', help='File pattern (default: *.nxml)')

    # Options
    parser.add_argument('--force', action='store_true', help='Re-ingest existing documents')
    parser.add_argument('--images-dir', default='files/figures', help='Directory for extracted images')

    args = parser.parse_args()

    # Create ingester
    ingester = DocumentIngester(images_dir=args.images_dir)

    # Single file mode
    if args.pmcid and (args.pdf or args.xml):
        success = ingester.ingest_document(
            pmcid=args.pmcid,
            pdf_path=Path(args.pdf) if args.pdf else None,
            xml_path=Path(args.xml) if args.xml else None,
            pdffigures_json_path=Path(args.pdffigures_json) if args.pdffigures_json else None,
            title=args.title,
            journal=args.journal,
            publication_year=args.year,
            force=args.force
        )
        sys.exit(0 if success else 1)

    # Batch mode
    elif args.xml_dir or args.pdf_dir:
        directory = Path(args.xml_dir or args.pdf_dir)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            sys.exit(1)

        ingester.ingest_directory(
            directory=directory,
            pattern=args.pattern,
            force=args.force
        )
        ingester.print_summary()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
