"""
XML to Database Importer

This script walks through XML files, extracts hierarchical structure using the
existing parser, and stores them in PostgreSQL with unique paths.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

# Load .env file BEFORE importing database module
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, will use defaults

# Add parent directory to path to import parsers
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.xml_parsers.hierarchical_parser import extract_hierarchy
from parsers.xml_parsers.figure_parser import extract_figures
from parsers.pdf_parser import extract_text_from_pdf
from database import (
    get_db_connection,
    close_db_connection,
    Document,
    TextElement,
    Figure
)
from lxml import etree
import shutil


class XMLDatabaseImporter:
    """
    Imports XML files into PostgreSQL database with hierarchical structure.
    """

    def __init__(self, database_url=None, echo=False, images_dir=None):
        """
        Initialize the importer.

        Args:
            database_url: PostgreSQL connection URL
            echo: If True, SQL statements will be logged
            images_dir: Directory to store extracted images (default: files/figures/)
        """
        self.db = get_db_connection(database_url=database_url, echo=echo)
        self.images_dir = Path(images_dir) if images_dir else Path("files/figures")
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'total_text_elements': 0,
            'total_figures': 0,
            'from_xml': 0,
            'from_pdf': 0
        }

    def extract_metadata(self, xml_path: Path) -> Dict[str, Optional[str]]:
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
            print(f"  Warning: Could not extract all metadata: {e}")

        return metadata

    def generate_unique_path(self, pmcid: str, path_string: str, position: int) -> str:
        """
        Generate a unique path for a text element.

        Format: {PMCID}/{path_string}/{position}
        Example: PMC1448691/Methods > 2.1 Staining/0

        Args:
            pmcid: PubMed Central ID
            path_string: Hierarchical path string
            position: Position within the section

        Returns:
            Unique path string
        """
        # If path_string is empty (root level), use "Root"
        path_part = path_string if path_string else "Root"
        return f"{pmcid}/{path_part}/{position}"

    def find_and_copy_image(self, pmcid: str, graphic_ref: str, xml_path: Path) -> Dict[str, Optional[str]]:
        """
        Find and copy an image file to the figures directory.

        Args:
            pmcid: PubMed Central ID
            graphic_ref: Original graphic reference from XML
            xml_path: Path to the XML file

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

        # Look for the image file in the same directory as XML
        xml_dir = xml_path.parent

        # Try to find the image file (may have different extensions)
        possible_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif']
        graphic_base = Path(graphic_ref).stem

        source_image = None
        for ext in possible_extensions:
            candidate = xml_dir / f"{graphic_base}{ext}"
            if candidate.exists():
                source_image = candidate
                break

        # Also try the exact filename
        if not source_image:
            candidate = xml_dir / graphic_ref
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
                print(f"    ⚠️  Failed to copy image {source_image.name}: {e}")

        return result

    def find_corresponding_pdf(self, pmcid: str, xml_path: Path) -> Optional[Path]:
        """
        Find the corresponding PDF file for an XML file.

        Args:
            pmcid: PubMed Central ID
            xml_path: Path to the XML file

        Returns:
            Path to PDF file if found, None otherwise
        """
        # Common locations to look for PDFs
        search_paths = [
            xml_path.parent,  # Same directory as XML
            xml_path.parent.parent / 'organized_pdfs',  # Organized PDFs directory
            xml_path.parent.parent / 'processed_corpus' / pmcid,  # Original corpus
            Path('files/organized_pdfs'),  # Default organized PDFs location
        ]

        for search_dir in search_paths:
            if not search_dir.exists():
                continue

            # Look for PDF files starting with PMCID
            pdf_files = list(search_dir.glob(f"{pmcid}*.pdf"))
            if pdf_files:
                return pdf_files[0]  # Return first match

        return None

    def process_xml_file(self, xml_path: Path) -> bool:
        """
        Process a single XML file and store in database.

        Args:
            xml_path: Path to XML file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract PMCID from filename (e.g., PMC1448691.nxml -> PMC1448691)
            pmcid = xml_path.stem
            print(f"Processing {pmcid}...", end=" ")

            # Check if already processed
            with self.db.session_scope() as session:
                existing = session.query(Document).filter_by(pmcid=pmcid).first()
                if existing:
                    print("⚠️  Already in database, skipping")
                    self.stats['skipped'] += 1
                    return True

            # Extract hierarchical structure from XML
            parsed_data = extract_hierarchy(str(xml_path))
            text_source = 'xml'

            # If XML parsing returned no or minimal content, try PDF fallback
            if not parsed_data or (len(parsed_data) <= 2 and all('Abstract' in item.get('path_string', '') for item in parsed_data)):
                print(f"⚠️  Minimal XML content ({len(parsed_data)} elements), trying PDF fallback...", end=" ")

                # Find corresponding PDF
                pdf_path = self.find_corresponding_pdf(pmcid, xml_path)

                if pdf_path:
                    try:
                        parsed_data = extract_text_from_pdf(str(pdf_path))
                        text_source = 'pdf'
                        print(f"✓ PDF found, extracted {len(parsed_data)} elements")
                    except Exception as e:
                        print(f"❌ PDF parsing failed: {e}")
                        if not parsed_data:  # If still no data, skip
                            self.stats['skipped'] += 1
                            return True
                else:
                    print("❌ No PDF found")
                    if not parsed_data:  # If still no data, skip
                        self.stats['skipped'] += 1
                        return True

            # Extract metadata
            metadata = self.extract_metadata(xml_path)

            # Store in database
            with self.db.session_scope() as session:
                # Create document
                document = Document(
                    pmcid=pmcid,
                    filename=xml_path.name,
                    file_path=str(xml_path.absolute()),
                    title=metadata['title'],
                    journal=metadata['journal'],
                    publication_year=metadata['publication_year'],
                    text_source=text_source
                )
                session.add(document)
                session.flush()  # Get the document ID

                # Count occurrences of each path to assign positions
                path_counters = defaultdict(int)

                # Create text elements
                text_elements = []
                for item in parsed_data:
                    path_string = item['path_string']
                    position = path_counters[path_string]
                    path_counters[path_string] += 1

                    # Generate unique path
                    unique_path = self.generate_unique_path(
                        pmcid, path_string, position
                    )

                    # Create text element
                    text_element = TextElement(
                        document_id=document.id,
                        unique_path=unique_path,
                        path_list=item['path_list'],
                        path_string=path_string,
                        depth=item['depth'],
                        position_in_section=position,
                        text_content=item['text']
                    )

                    # Calculate metadata
                    text_element.calculate_metadata()

                    text_elements.append(text_element)

                # Bulk insert text elements
                session.bulk_save_objects(text_elements)

                # Extract and store figures
                figures_data = extract_figures(str(xml_path))
                figure_objects = []

                for fig_data in figures_data:
                    # Find and copy image file
                    image_info = self.find_and_copy_image(
                        pmcid,
                        fig_data.get('graphic_ref'),
                        xml_path
                    )

                    # Create figure object
                    figure = Figure(
                        document_id=document.id,
                        figure_id=fig_data.get('figure_id'),
                        figure_label=fig_data.get('figure_label'),
                        caption_text=fig_data.get('caption_text'),
                        graphic_ref=fig_data.get('graphic_ref'),
                        image_filename=image_info['image_filename'],
                        image_path=image_info['image_path'],
                        image_format=image_info['image_format'],
                        section_context=fig_data.get('section_context')
                    )
                    figure_objects.append(figure)

                # Bulk insert figures
                if figure_objects:
                    session.bulk_save_objects(figure_objects)

                source_label = "from PDF" if text_source == 'pdf' else "from XML"
                print(f"✅ Saved {len(text_elements)} text elements, {len(figure_objects)} figures ({source_label})")
                self.stats['processed'] += 1
                self.stats['total_text_elements'] += len(text_elements)
                self.stats['total_figures'] += len(figure_objects)

                # Track source
                if text_source == 'pdf':
                    self.stats['from_pdf'] += 1
                else:
                    self.stats['from_xml'] += 1

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            self.stats['errors'] += 1
            return False

    def process_directory(self, xml_dir: Path, pattern: str = "*.nxml"):
        """
        Process all XML files in a directory.

        Args:
            xml_dir: Directory containing XML files
            pattern: Glob pattern for XML files (default: *.nxml)
        """
        # Find all XML files
        xml_files = list(xml_dir.glob(pattern))

        # Also check for .xml files
        if pattern == "*.nxml":
            xml_files.extend(list(xml_dir.glob("*.xml")))

        if not xml_files:
            print(f"No XML files found in {xml_dir}")
            return

        print(f"Found {len(xml_files)} XML files. Starting processing...\n")

        # Process each file
        for xml_file in sorted(xml_files):
            self.process_xml_file(xml_file)

    def print_summary(self):
        """Print processing summary statistics."""
        separator = "=" * 60
        print(f"\n{separator}")
        print("Processing Complete!")
        print(separator)
        print(f"Successfully processed: {self.stats['processed']}")
        print(f"  - From XML: {self.stats['from_xml']}")
        print(f"  - From PDF fallback: {self.stats['from_pdf']}")
        print(f"Already in database: {self.stats['skipped']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Total text elements: {self.stats['total_text_elements']}")
        print(f"Total figures: {self.stats['total_figures']}")
        if self.stats['processed'] > 0:
            avg_text = self.stats['total_text_elements'] / self.stats['processed']
            avg_figs = self.stats['total_figures'] / self.stats['processed']
            print(f"Average elements per document: {avg_text:.1f}")
            print(f"Average figures per document: {avg_figs:.1f}")
        print(separator)

    def close(self):
        """Close database connection."""
        close_db_connection()


def main():
    """Main execution function."""
    # Configuration
    XML_DIR = Path("files/organized_xmls")
    DATABASE_URL = None  # Will use environment variables or defaults

    # Validate input directory
    if not XML_DIR.exists():
        print(f"Error: XML directory '{XML_DIR}' not found.")
        print("Make sure you have XML files organized in the correct location.")
        return

    print(f"XML Directory: {XML_DIR.absolute()}")
    print("Database: Using environment variables or defaults\n")

    try:
        # Create importer
        importer = XMLDatabaseImporter(database_url=DATABASE_URL, echo=False)

        # Create tables if they don't exist
        print("Ensuring database tables exist...")
        importer.db.create_tables()
        print()

        # Process all XML files
        importer.process_directory(XML_DIR)

        # Print summary
        importer.print_summary()

        # Close connection
        importer.close()

    except Exception as e:
        print(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
