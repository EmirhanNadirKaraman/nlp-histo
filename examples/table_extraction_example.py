"""
Example: Extract text and tables separately for better quality

This example shows how to:
1. Detect tables in a PDF
2. Mask them so Docling skips them
3. Extract tables separately with OCR
4. Combine the results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.pdf_parsers.table_aware_extractor import TableAwareExtractor, process_tables_with_ocr
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def main():
    # Input PDF
    pdf_path = "files/organized_pdfs/PMC1448691_his_2369.pdf"

    print("="*70)
    print("Table-Aware PDF Extraction Demo")
    print("="*70)
    print(f"\nProcessing: {pdf_path}\n")

    # Create extractor
    extractor = TableAwareExtractor()

    # Extract with table masking
    result = extractor.extract(
        pdf_path,
        mask_tables=True,         # Mask tables before Docling sees them
        extract_table_images=True # Save table images for OCR
    )

    # Show results
    print(f"\n{'='*70}")
    print("Extraction Results")
    print(f"{'='*70}\n")

    print(f"✓ Text elements extracted: {len(result['text_elements'])}")
    print(f"✓ Tables detected: {len(result['tables'])}")
    print(f"✓ Masked PDF saved: {result['masked_pdf']}")
    print(f"✓ Output directory: {result['output_dir']}\n")

    # List detected tables
    if result['tables']:
        print(f"{'='*70}")
        print("Detected Tables")
        print(f"{'='*70}\n")

        for table in result['tables']:
            print(f"  {table['table_id']}:")
            print(f"    Page: {table['page']}")
            print(f"    Caption: {table['caption'][:60]}...")
            print(f"    Bbox: {table['bbox']}")
            print(f"    Image: {table.get('image_path', 'N/A')}")
            print()

    # Now process tables with OCR
    print(f"{'='*70}")
    print("Processing Tables with OCR")
    print(f"{'='*70}\n")

    # Choose OCR method:
    # - 'tesseract': Fast, free, good for simple tables
    # - 'claude': Most accurate, best for complex medical tables
    # - 'easyocr': Good middle ground

    ocr_method = 'tesseract'  # Change to 'claude' for better quality

    tables_with_text = process_tables_with_ocr(
        result['tables'],
        method=ocr_method
    )

    # Show extracted table text
    for table in tables_with_text:
        if table.get('extracted_text'):
            print(f"\n{table['table_id']} (extracted with {table.get('ocr_method')}):")
            print("-" * 70)
            print(table['extracted_text'][:300])
            if len(table['extracted_text']) > 300:
                print("...(truncated)")
            print()

    # Save combined output
    output_file = Path(result['output_dir']) / 'combined_output.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write text elements
        f.write("="*70 + "\n")
        f.write("TEXT CONTENT (from Docling, tables excluded)\n")
        f.write("="*70 + "\n\n")

        for elem in result['text_elements']:
            f.write(elem.get('text', '') + "\n\n")

        # Write tables
        f.write("\n" + "="*70 + "\n")
        f.write("TABLES (extracted separately)\n")
        f.write("="*70 + "\n\n")

        for table in tables_with_text:
            f.write(f"\n{table['table_id']}\n")
            f.write(f"Page {table['page']}: {table['caption']}\n")
            f.write("-"*70 + "\n")
            if table.get('extracted_text'):
                f.write(table['extracted_text'] + "\n\n")
            else:
                f.write("[Table extraction pending]\n\n")

    print(f"\n✓ Combined output saved to: {output_file}")


if __name__ == '__main__':
    main()
