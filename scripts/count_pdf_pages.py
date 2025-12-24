#!/usr/bin/env python3
"""
Count total number of pages across all PDF files in the project.
"""

import fitz  # PyMuPDF
from pathlib import Path
import sys


def count_pdf_pages(pdf_path):
    """
    Count the number of pages in a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Number of pages, or 0 if error occurs
    """
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}", file=sys.stderr)
        return 0


def main():
    # Base directory - adjust this path if needed
    base_dir = Path(__file__).parent
    pdf_dir = base_dir / "files" / "organized_pdfs"

    # Check if directory exists
    if not pdf_dir.exists():
        print(f"Directory not found: {pdf_dir}")
        print("Please ensure the organized_pdfs directory exists.")
        return

    # Find all PDF files in organized_pdfs
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in organized_pdfs directory.")
        return

    print(f"Found {len(pdf_files)} PDF files in organized_pdfs/. Counting pages...\n")

    total_pages = 0
    pdf_details = []

    for pdf_path in pdf_files:
        pages = count_pdf_pages(pdf_path)
        total_pages += pages

        # Store just the filename for cleaner output
        pdf_details.append((pdf_path.name, pages))

    # Sort by number of pages (descending)
    pdf_details.sort(key=lambda x: x[1], reverse=True)

    # Print individual file details
    print("PDF Files by Page Count:")
    print("-" * 80)
    for filename, pages in pdf_details:
        print(f"{pages:4d} pages - {filename}")

    print("-" * 80)
    print(f"\nTotal PDFs: {len(pdf_files)}")
    print(f"Total Pages: {total_pages}")
    print(f"Average Pages per PDF: {total_pages / len(pdf_files):.1f}")


if __name__ == "__main__":
    main()
