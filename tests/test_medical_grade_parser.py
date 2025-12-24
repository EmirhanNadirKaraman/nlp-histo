#!/usr/bin/env python3
"""
Test script for Medical-Grade Ensemble Parser

Tests both routing and fallback modes with sample PDFs.
"""

import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import fitz
    except ImportError:
        missing.append("PyMuPDF (pip install PyMuPDF)")

    try:
        import pymupdf4llm
    except ImportError:
        missing.append("pymupdf4llm (pip install pymupdf4llm)")

    return missing


def test_routing_mode(pdf_path: str):
    """Test routing mode with page-level detection."""
    print("\n" + "="*80)
    print("TEST 1: ROUTING MODE (Medical-Grade)")
    print("="*80 + "\n")

    from parsers.pdf_parsers.ensemble_parser import EnsemblePDFParser

    # Create parser in routing mode
    parser = EnsemblePDFParser(mode="routing")

    # Check availability
    print("Parser Availability:")
    parser.print_availability()

    # Extract
    print(f"\nExtracting from: {Path(pdf_path).name}")
    print("Using intelligent page-level routing...\n")

    try:
        result = parser.extract_hierarchy(pdf_path)

        if result['success']:
            print("\n" + "-"*80)
            print("RESULTS")
            print("-"*80)

            # Routing information
            if 'page_routing' in result:
                routing = result['page_routing']
                print(f"\nPage Routing:")
                print(f"  Total pages: {routing['total_pages']}")
                print(f"  Summary: {routing['summary']}")

                # Show per-page routing
                print(f"\n  Per-page routing:")
                for route_info in routing['routes'][:10]:  # Show first 10
                    print(f"    Page {route_info['page']}: {route_info['route']} ({route_info['reason']})")
                if len(routing['routes']) > 10:
                    print(f"    ... and {len(routing['routes']) - 10} more pages")

            # Extraction results
            print(f"\nExtraction Results:")
            print(f"  Text elements: {len(result['text_elements'])}")
            print(f"  Figures: {len(result['figures'])}")
            print(f"  Parsers used: {result['extraction_method']}")

            # Show sample text
            if result['text_elements']:
                print(f"\n  Sample text elements:")
                for i, elem in enumerate(result['text_elements'][:3], 1):
                    print(f"    {i}. Path: {elem['path_string']}")
                    print(f"       Text: {elem['text'][:100]}...")

            print("\n✓ Routing mode test PASSED\n")
            return True

        else:
            print("\n✗ Extraction failed")
            return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_mode(pdf_path: str):
    """Test fallback mode (legacy sequential)."""
    print("\n" + "="*80)
    print("TEST 2: FALLBACK MODE (Legacy)")
    print("="*80 + "\n")

    from parsers.pdf_parsers.ensemble_parser import EnsemblePDFParser

    # Create parser in fallback mode
    parser = EnsemblePDFParser(mode="fallback")

    print(f"Extracting from: {Path(pdf_path).name}")
    print("Using sequential fallback strategy...\n")

    try:
        result = parser.extract_hierarchy(pdf_path)

        if result['success']:
            print("\n" + "-"*80)
            print("RESULTS")
            print("-"*80)

            print(f"\nExtraction Results:")
            print(f"  Text elements: {len(result['text_elements'])}")
            print(f"  Figures: {len(result['figures'])}")
            print(f"  Parser used: {result['extraction_method']}")

            # Show sample text
            if result['text_elements']:
                print(f"\n  Sample text elements:")
                for i, elem in enumerate(result['text_elements'][:3], 1):
                    print(f"    {i}. Path: {elem['path_string']}")
                    print(f"       Text: {elem['text'][:100]}...")

            print("\n✓ Fallback mode test PASSED\n")
            return True

        else:
            print("\n✗ Extraction failed")
            return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MEDICAL-GRADE ENSEMBLE PARSER TEST SUITE")
    print("="*80 + "\n")

    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()

    if missing:
        print("\n⚠ Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies:")
        print("  pip install PyMuPDF pymupdf4llm")
        print("\nNote: Routing mode requires PyMuPDF and pymupdf4llm")
        print("      Fallback mode can work with other parsers (Marker, Docling, etc.)")
        sys.exit(1)

    print("✓ All dependencies available\n")

    # Find test PDF
    test_pdf = "files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf"

    if not Path(test_pdf).exists():
        print(f"Test PDF not found: {test_pdf}")
        print("Looking for alternative PDFs...")

        # Try to find any PDF
        pdf_dir = Path("files/organized_pdfs")
        if pdf_dir.exists():
            pdfs = list(pdf_dir.glob("*.pdf"))
            if pdfs:
                test_pdf = str(pdfs[0])
                print(f"Using: {test_pdf}")
            else:
                print("No PDFs found in files/organized_pdfs/")
                sys.exit(1)
        else:
            print("files/organized_pdfs/ directory not found")
            sys.exit(1)

    # Run tests
    results = []

    # Test 1: Routing mode
    try:
        results.append(("Routing Mode", test_routing_mode(test_pdf)))
    except Exception as e:
        print(f"\n✗ Routing mode test failed: {e}")
        results.append(("Routing Mode", False))

    # Test 2: Fallback mode
    try:
        results.append(("Fallback Mode", test_fallback_mode(test_pdf)))
    except Exception as e:
        print(f"\n✗ Fallback mode test failed: {e}")
        results.append(("Fallback Mode", False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✓ All tests PASSED\n")
        sys.exit(0)
    else:
        print("\n✗ Some tests FAILED\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
