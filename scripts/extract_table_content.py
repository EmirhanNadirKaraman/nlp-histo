#!/usr/bin/env python3
"""
Extract structured table content from PDFs using PDFFigures2 bounding boxes.

Uses pdfplumber to extract tables from specific regions in PDFs.
"""

import json
import sys
from pathlib import Path
import pdfplumber

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# IMPORTANT: Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from database import get_db_connection, Document, Table


def extract_table_from_bbox(pdf_path: Path, page_num: int, bbox: dict):
    """
    Extract table content from a specific region of a PDF page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-based)
        bbox: Bounding box dict with x1, y1, x2, y2

    Returns:
        Structured table data (list of lists) or None if extraction fails
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return None

            page = pdf.pages[page_num]

            # pdfplumber uses (x0, top, x1, bottom) format
            # PDFFigures2 uses (x1, y1, x2, y2) where y is from top
            crop_bbox = (
                bbox.get('x1', 0),
                bbox.get('y1', 0),
                bbox.get('x2', 0),
                bbox.get('y2', 0)
            )

            # Crop page to table region
            cropped = page.within_bbox(crop_bbox)

            # Extract table
            tables = cropped.extract_tables()

            if tables and len(tables) > 0:
                # Return the first table found
                return tables[0]

            # If extract_tables doesn't work, try extract_table
            table = cropped.extract_table()
            return table

    except Exception as e:
        print(f"    Error extracting table: {e}")
        return None


def format_table_as_markdown(table_data):
    """
    Convert table data (list of lists) to Markdown format.

    Args:
        table_data: List of lists representing table rows

    Returns:
        Markdown string
    """
    if not table_data or len(table_data) == 0:
        return None

    lines = []

    # Header row
    header = table_data[0]
    if header:
        lines.append('| ' + ' | '.join(str(cell or '') for cell in header) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')

    # Data rows
    for row in table_data[1:]:
        if row:
            lines.append('| ' + ' | '.join(str(cell or '') for cell in row) + ' |')

    return '\n'.join(lines)


def format_table_as_csv(table_data):
    """
    Convert table data to CSV format.

    Args:
        table_data: List of lists representing table rows

    Returns:
        CSV string
    """
    if not table_data:
        return None

    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    for row in table_data:
        # Clean cells
        cleaned_row = [str(cell or '').strip() for cell in row]
        writer.writerow(cleaned_row)

    return output.getvalue()


def extract_and_save_table_content(pmcid: str = None, format: str = 'markdown'):
    """
    Extract table content for documents and save to database.

    Args:
        pmcid: If provided, only process this document
        format: Output format - 'markdown', 'csv', or 'json'
    """

    db = get_db_connection()

    with db.session_scope() as session:
        # Get documents to process
        if pmcid:
            docs = session.query(Document).filter_by(pmcid=pmcid).all()
        else:
            docs = session.query(Document).all()

        if not docs:
            print("No documents found")
            return

        print(f"Extracting table content for {len(docs)} documents...")
        print(f"Format: {format}\n")

        total_tables = 0
        successful_extractions = 0

        for doc in docs:
            print(f"Processing: {doc.pmcid}")

            # Get PDF path
            pdf_path = Path(doc.file_path)
            if not pdf_path.exists():
                print(f"  ⚠ PDF not found: {pdf_path}")
                continue

            # Get tables for this document
            tables = session.query(Table).filter_by(document_id=doc.id).all()

            if not tables:
                print(f"  No tables found")
                continue

            print(f"  Found {len(tables)} tables")

            for table in tables:
                total_tables += 1

                # Check if we have bounding box
                if not all([table.bbox_x1, table.bbox_y1, table.bbox_x2, table.bbox_y2]):
                    print(f"    ⚠ Table {table.table_label}: No bounding box")
                    continue

                # Extract table content
                bbox = {
                    'x1': table.bbox_x1,
                    'y1': table.bbox_y1,
                    'x2': table.bbox_x2,
                    'y2': table.bbox_y2
                }

                print(f"    Extracting {table.table_label}...", end=' ')

                table_data = extract_table_from_bbox(
                    pdf_path=pdf_path,
                    page_num=table.page_number,
                    bbox=bbox
                )

                if table_data:
                    # Format table content
                    if format == 'markdown':
                        content = format_table_as_markdown(table_data)
                    elif format == 'csv':
                        content = format_table_as_csv(table_data)
                    else:  # json
                        content = json.dumps(table_data, ensure_ascii=False)

                    # Save to database
                    table.table_content = content
                    successful_extractions += 1
                    print(f"✓ ({len(table_data)} rows)")
                else:
                    print("✗ Failed")

            print()

        print(f"{'='*80}")
        print(f"Summary")
        print(f"{'='*80}")
        print(f"Total tables:       {total_tables}")
        print(f"Successful:         {successful_extractions}")
        print(f"Failed:             {total_tables - successful_extractions}")
        print(f"Success rate:       {successful_extractions/total_tables*100:.1f}%")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract table content from PDFs')
    parser.add_argument(
        '--pmcid',
        help='PMC ID to process (default: process all)'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'csv', 'json'],
        default='markdown',
        help='Output format (default: markdown)'
    )

    args = parser.parse_args()

    extract_and_save_table_content(args.pmcid, args.format)
