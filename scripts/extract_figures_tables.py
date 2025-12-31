#!/usr/bin/env python3
"""
Extract figures and tables from PDFs using PDFFigures2 JSON output.

Reads JSON files from out/data/, uses bounding boxes to extract regions,
and saves tables to out/tables/ and figures to out/figures/.
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_region_from_pdf(pdf_path: Path, page_num: int, bbox: dict, output_path: Path, dpi: int = 150):
    """
    Extract a specific region from a PDF page and save as image.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (0-based)
        bbox: Bounding box dict with keys: x1, y1, x2, y2
        output_path: Where to save the extracted image
        dpi: Resolution for rendering (default 150)
    """
    try:
        doc = fitz.open(str(pdf_path))

        if page_num >= len(doc):
            logger.warning(f"  Page {page_num} not found in PDF")
            return False

        page = doc[page_num]

        # Create rectangle from bounding box
        rect = fitz.Rect(
            bbox.get('x1', 0),
            bbox.get('y1', 0),
            bbox.get('x2', 0),
            bbox.get('y2', 0)
        )

        # Calculate zoom for desired DPI
        # PDF default is 72 DPI, so zoom = dpi/72
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        # Render the page region as pixmap
        pix = page.get_pixmap(matrix=mat, clip=rect)

        # Save as PNG
        pix.save(str(output_path))

        doc.close()
        return True

    except Exception as e:
        logger.error(f"  Failed to extract region: {e}")
        return False


def process_pdffigures_json(json_path: Path, pdf_path: Path, output_dir: Path):
    """
    Process a PDFFigures2 JSON file and extract all figures/tables.

    Args:
        json_path: Path to PDFFigures2 JSON output
        pdf_path: Path to the original PDF
        output_dir: Base output directory (will create 'tables' and 'figures' subdirs)
    """
    logger.info(f"Processing: {json_path.name}")

    # Load JSON
    if not json_path.exists():
        logger.warning(f"  JSON not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    if not items:
        logger.info(f"  No figures/tables found")
        return

    # Check if PDF exists
    if not pdf_path.exists():
        logger.warning(f"  PDF not found: {pdf_path}")
        return

    # Create output directories
    tables_dir = output_dir / 'tables'
    figures_dir = output_dir / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Extract base name for output files
    pdf_base = pdf_path.stem

    # Process each figure/table
    table_count = 0
    figure_count = 0

    for item in items:
        fig_type = item.get('figType', 'Unknown')
        name = item.get('name', 'unknown')
        page_num = item.get('page', 0)
        region_boundary = item.get('regionBoundary')

        if not region_boundary:
            logger.warning(f"  Skipping {fig_type} {name}: no region boundary")
            continue

        # Determine output directory and filename
        if fig_type == 'Table':
            output_subdir = tables_dir
            output_filename = f"{pdf_base}_table_{name}.png"
            table_count += 1
        elif fig_type == 'Figure':
            output_subdir = figures_dir
            output_filename = f"{pdf_base}_figure_{name}.png"
            figure_count += 1
        else:
            logger.warning(f"  Unknown type: {fig_type}")
            continue

        output_path = output_subdir / output_filename

        # Extract region
        logger.info(f"  Extracting {fig_type} {name} from page {page_num + 1}")
        success = extract_region_from_pdf(
            pdf_path=pdf_path,
            page_num=page_num,
            bbox=region_boundary,
            output_path=output_path,
            dpi=150
        )

        if success:
            logger.info(f"    ✓ Saved: {output_path}")

    logger.info(f"  Summary: {figure_count} figures, {table_count} tables\n")
    return {'figures': figure_count, 'tables': table_count}


def process_all_json_files(data_dir: Path, pdf_dir: Path, output_dir: Path):
    """
    Process all JSON files in the data directory.

    Args:
        data_dir: Directory containing PDFFigures2 JSON files
        pdf_dir: Directory containing original PDFs
        output_dir: Base output directory
    """
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Find all JSON files
    json_files = list(data_dir.glob('*.json'))

    if not json_files:
        logger.warning(f"No JSON files found in {data_dir}")
        return

    logger.info(f"Found {len(json_files)} JSON files\n")
    logger.info(f"{'='*80}\n")

    total_figures = 0
    total_tables = 0
    processed_count = 0

    for json_path in json_files:
        # Determine corresponding PDF path
        # Assume PDF has same base name as JSON
        pdf_name = json_path.stem + '.pdf'
        pdf_path = pdf_dir / pdf_name

        # Try without extension if not found
        if not pdf_path.exists():
            # Try to find PDF with similar name
            possible_pdfs = list(pdf_dir.glob(f"{json_path.stem}*.pdf"))
            if possible_pdfs:
                pdf_path = possible_pdfs[0]
            else:
                logger.warning(f"No PDF found for {json_path.name}, skipping\n")
                continue

        result = process_pdffigures_json(json_path, pdf_path, output_dir)

        if result:
            total_figures += result['figures']
            total_tables += result['tables']
            processed_count += 1

    # Final summary
    logger.info(f"{'='*80}")
    logger.info(f"Extraction Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Processed: {processed_count} PDFs")
    logger.info(f"Extracted: {total_figures} figures, {total_tables} tables")
    logger.info(f"Output:")
    logger.info(f"  Figures: {output_dir}/figures/")
    logger.info(f"  Tables:  {output_dir}/tables/")
    logger.info(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract figures and tables from PDFs using PDFFigures2 JSON'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('out/data'),
        help='Directory containing PDFFigures2 JSON files (default: out/data)'
    )
    parser.add_argument(
        '--pdf-dir',
        type=Path,
        default=Path('files/organized_pdfs'),
        help='Directory containing original PDFs (default: files/organized_pdfs)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('out'),
        help='Output directory (will create tables/ and figures/ subdirs, default: out)'
    )

    args = parser.parse_args()

    process_all_json_files(
        data_dir=args.data_dir,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
