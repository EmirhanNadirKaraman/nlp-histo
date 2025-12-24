"""
PDF to Database Importer

This script processes PDF files, extracts hierarchical structure using the
ensemble PDF parser (Marker/Nougat/PDFFigures), and stores them in PostgreSQL.

Parallel to xml_to_db.py but for PDF extraction as PRIMARY method.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict
import argparse

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

from parsers.pdf_parsers.ensemble_parser import EnsemblePDFParser
from database import (
    get_db_connection,
    close_db_connection,
    Document,
    TextElement,
    Figure
)


class PDFDatabaseImporter:
    """
    Imports PDF files into PostgreSQL database with hierarchical structure.
    Uses ensemble parser (Marker/Nougat/PDFFigures) for extraction.
    """

    def __init__(self, database_url=None, echo=False, prefer_nougat=False):
        """
        Initialize the importer.

        Args:
            database_url: PostgreSQL connection URL
            echo: If True, SQL statements will be logged
            prefer_nougat: If True, prefer Nougat over Marker (slower but better for equations)
        """
        self.db = get_db_connection(database_url=database_url, echo=echo)
        self.parser = EnsemblePDFParser(prefer_nougat=prefer_nougat)

        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'marker_count': 0,
            'nougat_count': 0,
            'total_text_elements': 0,
            'total_figures': 0
        }

        # Checkpoint file for resume capability
        self.checkpoint_file = Path(__file__).parent.parent / 'pdf_import_checkpoint.json'

    def _extract_pmcid(self, pdf_path: Path) -> str:
        """
        Extract PMCID from PDF filename.

        Format: PMC10047158_dermatopathology-10-00017.pdf → PMC10047158

        Args:
            pdf_path: Path to PDF file

        Returns:
            PMCID string
        """
        filename = pdf_path.stem  # Remove .pdf extension
        # Split on underscore and take first part
        pmcid = filename.split('_')[0]
        return pmcid

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

    def process_pdf_file(self, pdf_path: Path, pmcid: Optional[str] = None) -> bool:
        """
        Process a single PDF file and store in database.

        Args:
            pdf_path: Path to PDF file
            pmcid: Optional PMCID (extracted from filename if not provided)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract PMCID from filename if not provided
            if pmcid is None:
                pmcid = self._extract_pmcid(pdf_path)

            print(f"Processing {pmcid}...", end=" ")

            # Check if already processed
            with self.db.session_scope() as session:
                existing = session.query(Document).filter_by(pmcid=pmcid).first()
                if existing:
                    print("⚠️  Already in database, skipping")
                    self.stats['skipped'] += 1
                    return True

            # Extract content using ensemble parser
            extraction_result = self.parser.extract_hierarchy(
                str(pdf_path),
                extract_figures=True
            )

            if not extraction_result['success']:
                print("❌ Extraction failed (no parser succeeded)")
                self.stats['errors'] += 1
                return False

            parsed_data = extraction_result['text_elements']
            figures_data = extraction_result['figures']
            extraction_method = extraction_result['extraction_method']

            # Track which parser was used
            if extraction_method == 'marker':
                self.stats['marker_count'] += 1
            elif extraction_method == 'nougat':
                self.stats['nougat_count'] += 1

            # Store in database
            with self.db.session_scope() as session:
                # Create document
                document = Document(
                    pmcid=pmcid,
                    filename=pdf_path.name,
                    file_path=str(pdf_path.absolute()),
                    text_source='pdf',  # Mark as PDF source
                    title=None,  # PDFs don't have easily extractable metadata
                    journal=None,
                    publication_year=None
                )
                session.add(document)
                session.flush()  # Get the document ID

                # Count occurrences of each path to assign positions
                path_counters = defaultdict(int)

                # Create text elements (same logic as xml_to_db.py)
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

                # Store figures (if any)
                figure_objects = []
                for fig_data in figures_data:
                    figure = Figure(
                        document_id=document.id,
                        figure_id=fig_data.get('figure_id'),
                        figure_label=fig_data.get('figure_type'),
                        caption_text=fig_data.get('caption'),
                        image_path=fig_data.get('image_path'),
                        image_format='png' if fig_data.get('image_path') else None,
                        section_context=None  # PDFs don't have section context
                    )
                    figure_objects.append(figure)

                if figure_objects:
                    session.bulk_save_objects(figure_objects)

                print(f"✅ Saved {len(text_elements)} elements, "
                      f"{len(figure_objects)} figures (via {extraction_method})")

                self.stats['processed'] += 1
                self.stats['total_text_elements'] += len(text_elements)
                self.stats['total_figures'] += len(figure_objects)

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            self.stats['errors'] += 1
            return False

    def process_directory(
        self,
        pdf_dir: Path,
        limit: Optional[int] = None,
        resume: bool = True
    ):
        """
        Process all PDF files in a directory.

        Args:
            pdf_dir: Directory containing PDF files
            limit: Maximum number of files to process (None = all)
            resume: If True, load checkpoint and resume from last position
        """
        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()

        # Find all PDF files
        pdf_files = sorted(list(pdf_dir.glob("*.pdf")))

        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return

        # Apply limit if specified
        if limit:
            pdf_files = pdf_files[:limit]

        print(f"Found {len(pdf_files)} PDF files. Starting processing...\n")

        # Process each file
        for i, pdf_file in enumerate(pdf_files, 1):
            # Process file
            self.process_pdf_file(pdf_file)

            # Save checkpoint every 10 files
            if i % 10 == 0:
                self.save_checkpoint()
                print(f"\n[Checkpoint saved at {i}/{len(pdf_files)} files]\n")

        # Save final checkpoint
        self.save_checkpoint()

    def save_checkpoint(self):
        """Save progress to checkpoint file."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")

    def load_checkpoint(self):
        """Load progress from checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.stats = json.load(f)
                print(f"Resuming from checkpoint: {self.stats['processed']} files processed")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")

    def print_summary(self):
        """Print processing summary statistics."""
        separator = "=" * 60
        print(f"\n{separator}")
        print("PDF Processing Complete!")
        print(separator)
        print(f"Successfully processed: {self.stats['processed']}")
        print(f"  - Via Marker: {self.stats['marker_count']}")
        print(f"  - Via Nougat: {self.stats['nougat_count']}")
        print(f"Already in database: {self.stats['skipped']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Total text elements: {self.stats['total_text_elements']}")
        print(f"Total figures: {self.stats['total_figures']}")

        if self.stats['processed'] > 0:
            avg_text = self.stats['total_text_elements'] / self.stats['processed']
            avg_figs = self.stats['total_figures'] / self.stats['processed']
            print(f"Average elements per PDF: {avg_text:.1f}")
            print(f"Average figures per PDF: {avg_figs:.1f}")

        print(separator)

    def close(self):
        """Close database connection."""
        close_db_connection()


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Import PDF files into PostgreSQL database'
    )
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='files/organized_pdfs',
        help='Directory containing PDF files (default: files/organized_pdfs)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of PDFs to process (default: all)'
    )
    parser.add_argument(
        '--prefer-nougat',
        action='store_true',
        help='Prefer Nougat over Marker (slower but better for equations)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoint'
    )

    args = parser.parse_args()

    PDF_DIR = Path(args.pdf_dir)

    # Validate input directory
    if not PDF_DIR.exists():
        print(f"Error: PDF directory '{PDF_DIR}' not found.")
        print("Make sure you have PDF files organized in the correct location.")
        return 1

    print(f"PDF Directory: {PDF_DIR.absolute()}")
    print("Database: Using environment variables or defaults\n")

    try:
        # Create importer
        importer = PDFDatabaseImporter(
            echo=False,
            prefer_nougat=args.prefer_nougat
        )

        # Check parser availability
        print("Checking parser availability...")
        importer.parser.print_availability()

        availability = importer.parser.get_available_parsers()
        if not (availability['marker'] or availability['nougat']):
            print("\nError: No text extraction parser available!")
            print("\nInstallation instructions:")
            print("  Marker (recommended): pip install marker-pdf")
            print("  Nougat (fallback): pip install transformers torch")
            return 1

        # Create tables if they don't exist
        print("Ensuring database tables exist...")
        importer.db.create_tables()
        print()

        # Process all PDF files
        importer.process_directory(
            PDF_DIR,
            limit=args.limit,
            resume=not args.no_resume
        )

        # Print summary
        importer.print_summary()

        # Close connection
        importer.close()

        return 0

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        print("Progress has been saved. Run again with same arguments to resume.")
        return 1

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
