#!/usr/bin/env python3
"""
Template for Florence-2 table and figure extraction.
Based on: https://github.com/wjbmattingly/youtube-florence-table

To use this:
1. Install Florence-2 dependencies:
   pip install torch transformers pillow pdf2image

2. Implement the extract_from_pdf() function below

3. Import this in extract_and_visualize.py:
   from scripts.florence_extractor_template import extract_from_pdf
"""

from pathlib import Path
from typing import List, Dict
import json

# TODO: Add your Florence-2 imports here
# Example (you'll need to adjust based on the actual Florence-2 implementation):
# from transformers import AutoProcessor, AutoModelForCausalLM
# from pdf2image import convert_from_path
# import torch


def extract_from_pdf(pdf_path: str) -> Dict:
    """
    Extract figures and tables from PDF using Florence-2 model.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict with structure:
        {
            "figures": [
                {"page": int, "bbox": {"x1": float, "y1": float, "x2": float, "y2": float}, "caption": str or None}
            ],
            "tables": [
                {"page": int, "bbox": {"x1": float, "y1": float, "x2": float, "y2": float}, "caption": str or None}
            ]
        }

        Note: bbox coordinates should use PDF bottom-left origin (like pdffigures2)
        where y1 is the bottom edge and y2 is the top edge.
    """

    figures = []
    tables = []

    # TODO: Implement Florence-2 extraction
    # Example workflow:
    # 1. Convert PDF to images
    # images = convert_from_path(pdf_path)
    #
    # 2. Load Florence-2 model
    # model = AutoModelForCausalLM.from_pretrained("microsoft/florence-2-base")
    # processor = AutoProcessor.from_pretrained("microsoft/florence-2-base")
    #
    # 3. For each page, detect tables and figures
    # for page_num, image in enumerate(images, 1):
    #     # Run detection
    #     inputs = processor(images=image, return_tensors="pt")
    #     outputs = model.generate(**inputs, max_new_tokens=1024)
    #     results = processor.batch_decode(outputs, skip_special_tokens=True)
    #
    #     # Parse results and extract bounding boxes
    #     # Add to figures or tables lists with proper format
    #
    # 4. Return standardized format

    print("WARNING: Florence-2 extraction not implemented yet!")
    print("Edit scripts/florence_extractor_template.py to add the implementation.")

    return {
        "figures": figures,
        "tables": tables
    }


def main():
    """Test the Florence-2 extractor standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract tables/figures with Florence-2")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output JSON path")
    args = parser.parse_args()

    # Run extraction
    results = extract_from_pdf(args.pdf_path)

    # Add metadata
    output_data = {
        "pdf_path": args.pdf_path,
        "extraction_tool": "Florence-2",
        "total_figures": len(results["figures"]),
        "total_tables": len(results["tables"]),
        "figures": results["figures"],
        "tables": results["tables"]
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        pdf_stem = Path(args.pdf_path).stem
        output_path = Path("out/florence") / f"{pdf_stem}_florence.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nFlorence-2 Extraction Results:")
    print(f"  Figures: {output_data['total_figures']}")
    print(f"  Tables:  {output_data['total_tables']}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
