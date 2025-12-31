#!/usr/bin/env python3
"""
Visualize Florence-2 detections from HistoPipeline output.
Works with the JSON format from florence_extractor.py
"""

import json
import sys
from pathlib import Path
import fitz  # PyMuPDF

# Color scheme
COLORS = {
    'table': (0, 0.8, 0),      # Green
    'figure': (1, 0, 0.5),     # Pink/Magenta
    'micrograph': (0.5, 0, 1), # Purple
    'chart': (1, 0.5, 0),      # Orange
    'diagram': (0, 0.5, 1),    # Blue
    'default': (0.7, 0.7, 0.7) # Gray
}


def visualize_florence_detections(pdf_path: str, json_path: str, output_path: str = None):
    """
    Draw bounding boxes from Florence-2 detections on PDF.

    Args:
        pdf_path: Path to original PDF
        json_path: Path to Florence JSON output
        output_path: Where to save visualization (optional)
    """
    pdf_file = Path(pdf_path)
    json_file = Path(json_path)

    if not pdf_file.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return

    if not json_file.exists():
        print(f"❌ JSON not found: {json_path}")
        return

    # Load detection results
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Open PDF
    doc = fitz.open(str(pdf_file))

    print(f"\n{'='*80}")
    print(f"Visualizing Florence-2 Detections")
    print(f"{'='*80}\n")
    print(f"PDF: {pdf_file.name}")
    print(f"Pages: {len(doc)}\n")

    total_detections = 0

    # Process each page
    for page_data in data.get('pages', []):
        page_num = page_data['page_num']
        detected_objects = page_data.get('detected_objects', [])

        if page_num > len(doc):
            print(f"  Warning: Page {page_num} not found in PDF")
            continue

        page = doc[page_num - 1]  # Convert to 0-indexed

        print(f"  Page {page_num}: {len(detected_objects)} detections")

        # Draw each detection
        for idx, obj in enumerate(detected_objects, 1):
            obj_type = obj.get('type', 'unknown').lower()
            bbox = obj.get('bbox', [])

            if len(bbox) != 4:
                continue

            # Create rectangle
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

            # Get color based on type
            color = COLORS.get(obj_type, COLORS['default'])

            # Draw bounding box
            page.draw_rect(rect, color=color, width=2, dashes="[3] 0")

            # Add label
            label = f"{obj_type.upper()[:3]}-{idx}"
            page.insert_text(
                (rect.x0 + 2, rect.y0 + 12),
                label,
                fontsize=9,
                color=color
            )

            total_detections += 1

    # Add legend on first page
    if len(doc) > 0:
        first_page = doc[0]
        lx, ly = 20, 20

        # Get unique types from all detections
        unique_types = set()
        for page_data in data.get('pages', []):
            for obj in page_data.get('detected_objects', []):
                unique_types.add(obj.get('type', 'unknown').lower())

        # Calculate legend height
        legend_height = 30 + len(unique_types) * 12

        # Background for legend
        first_page.draw_rect(
            fitz.Rect(lx-5, ly-5, lx+150, ly+legend_height),
            color=(0,0,0),
            fill=(1,1,1),
            width=0.5
        )

        # Legend title
        first_page.insert_text((lx, ly+10), "Florence-2 Detections:",
                              fontsize=10, color=(0,0,0))

        # Legend items
        for i, obj_type in enumerate(sorted(unique_types)):
            y_pos = ly + 25 + (i * 12)
            color = COLORS.get(obj_type, COLORS['default'])
            first_page.draw_line((lx, y_pos-3), (lx+15, y_pos-3),
                                color=color, width=2)
            first_page.insert_text((lx+20, y_pos),
                                  obj_type.capitalize(),
                                  fontsize=9, color=(0,0,0))

    # Save annotated PDF
    if output_path:
        output_file = Path(output_path)
    else:
        output_dir = Path("out/comparisons")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{pdf_file.stem}_florence_vis.pdf"

    doc.save(str(output_file))
    doc.close()

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}\n")
    print(f"Total detections: {total_detections}")
    print(f"✅ Visualization saved to: {output_file}\n")

    return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Florence-2 detections from HistoPipeline"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to original PDF file"
    )
    parser.add_argument(
        "--json",
        help="Path to Florence JSON output (default: auto-detect from out/florence/)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output PDF path (default: out/comparisons/{filename}_florence_vis.pdf)"
    )

    args = parser.parse_args()

    pdf_file = Path(args.pdf_path)

    # Auto-detect JSON if not specified
    if args.json:
        json_path = args.json
    else:
        json_path = Path("out/florence") / f"{pdf_file.stem}_clean.json"
        if not json_path.exists():
            print(f"❌ Auto-detect failed. JSON not found at: {json_path}")
            print(f"   Specify JSON path with --json")
            sys.exit(1)
        print(f"Auto-detected JSON: {json_path}")

    visualize_florence_detections(args.pdf_path, json_path, args.output)


if __name__ == "__main__":
    main()
