#!/usr/bin/env python3
"""
Visualize and compare figure/table extractions from pdffigures2 and Docling.
Draws bounding boxes on PDFs to show what each tool detected.
"""

import sys
import json
from pathlib import Path
import fitz  # PyMuPDF

# Color scheme (RGB 0.0 to 1.0)
COLORS = {
    'pdffigures_figure': (0, 0.5, 1),      # Blue
    'pdffigures_table': (0, 0.8, 0),       # Green
    'docling_figure': (1, 0, 0.5),         # Pink/Magenta
    'docling_table': (0.8, 0.4, 0),        # Orange
}

def load_pdffigures_results(json_path: str):
    if not Path(json_path).exists():
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        figures = [item for item in data if item.get('figType') == 'Figure']
        tables = [item for item in data if item.get('figType') == 'Table']
        
        return {'figures': figures, 'tables': tables}
    except Exception as e:
        print(f"Error loading PDFfigures2 JSON: {e}")
        return None

def load_docling_results(json_path: str):
    if not Path(json_path).exists():
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return {
            'figures': data.get('figures', []),
            'tables': data.get('tables', [])
        }
    except Exception as e:
        print(f"Error loading Docling JSON: {e}")
        return None

def get_rect(bbox, page_height, debug=False):
    try:
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']

        # Determine which Y is the 'top' and which is the 'bottom' 
        # based on the PDF standard (bottom-up)
        pdf_y_top = max(y1, y2)    # The higher value (e.g., 693)
        pdf_y_bottom = min(y1, y2) # The lower value (e.g., 302)

        # Convert to PyMuPDF (top-down)
        # y0 (top) = height - highest Y
        # y1 (bottom) = height - lowest Y
        rect = fitz.Rect(x1, page_height - pdf_y_top, x2, page_height - pdf_y_bottom)

        if debug:
            print(f"[DEBUG] Docling bbox: {y1} to {y2} -> Rect Ys: {rect.y0:.2f} to {rect.y1:.2f}")

        return rect
    except KeyError:
        return None

def draw_boxes(pdf_path: str, pdffigures_data, docling_data, output_dir: str):
    pdf_file = Path(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_file))
    
    print(f"\nProcessing: {pdf_file.name} ({len(doc)} pages)")

    # Define the tool mapping for the loop
    # (data_source, key_in_data, color_key, label_prefix, bbox_key_name)
    tools_to_draw = [
        (pdffigures_data, 'figures', 'pdffigures_figure', 'PDF2-F', 'regionBoundary'),
        (pdffigures_data, 'tables', 'pdffigures_table', 'PDF2-T', 'regionBoundary'),
        (docling_data, 'figures', 'docling_figure', 'DOC-F', 'bbox'),
        (docling_data, 'tables', 'docling_table', 'DOC-T', 'bbox'),
    ]

    for page_num in range(len(doc)):
        page = doc[page_num]
        h = page.rect.height
        page_no = page_num + 1

        for source_data, data_key, color_key, prefix, bbox_key in tools_to_draw:
            if not source_data:
                continue

            items = source_data.get(data_key, [])
            for i, item in enumerate(items, 1):
                if item.get('page') == page_no:
                    bbox = item.get(bbox_key)
                    if not bbox:
                        continue

                    rect = get_rect(bbox, h)
                    if not rect:
                        continue

                    # Determine style: Docling uses dashed lines, PDFfigures2 uses solid
                    is_docling = "docling" in color_key
                    stroke_width = 2 if is_docling else 3
                    dash_pattern = "[3] 0" if is_docling else None

                    # Draw the box
                    page.draw_rect(rect, color=COLORS[color_key], 
                                  width=stroke_width, dashes=dash_pattern)
                    
                    # Add text label
                    label = f"{prefix}{item.get('name', i)}"
                    page.insert_text((rect.x0 + 2, rect.y0 + 12), label,
                                   fontsize=9, color=COLORS[color_key])

    # Add Legend to the first page
    if len(doc) > 0:
        first_page = doc[0]
        lx, ly = 20, 20
        # Draw background for legend
        first_page.draw_rect(fitz.Rect(lx-5, ly-5, lx+180, ly+75), 
                             color=(0,0,0), fill=(1,1,1), width=0.5)
        
        legend_items = [
            ("PDFfigures2 Figures", COLORS['pdffigures_figure'], False),
            ("PDFfigures2 Tables", COLORS['pdffigures_table'], False),
            ("Docling Figures", COLORS['docling_figure'], True),
            ("Docling Tables", COLORS['docling_table'], True)
        ]
        
        first_page.insert_text((lx, ly+5), "Legend:", fontsize=10, color=(0,0,0))
        for i, (text, color, is_dashed) in enumerate(legend_items):
            y_pos = ly + 20 + (i * 12)
            line_symbol = "---" if is_dashed else "—"
            first_page.insert_text((lx, y_pos), f"{line_symbol} {text}", 
                                   fontsize=9, color=color)

    output_file = output_path / f"{pdf_file.stem}_comparison.pdf"
    doc.save(str(output_file))
    doc.close()
    
    print(f"✓ Saved to: {output_file}")
    return output_file

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare PDF figure extraction tools")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--pdffigures", help="Path to pdffigures2 JSON")
    parser.add_argument("--docling", help="Path to Docling JSON")
    parser.add_argument("--output", "-o", default="out/comparisons", help="Output directory")

    args = parser.parse_args()
    pdf_file = Path(args.pdf_path)
    
    if not pdf_file.exists():
        print(f"Error: PDF not found: {args.pdf_path}")
        return

    # Auto-detection logic
    pdf2_json = args.pdffigures or f"out/data/{pdf_file.stem}.json"
    doc_json = args.docling or f"out/docling/{pdf_file.stem}_docling.json"

    pdff_data = load_pdffigures_results(pdf2_json)
    docl_data = load_docling_results(doc_json)

    if not pdff_data and not docl_data:
        print("No JSON results found for either tool. Check your paths.")
        return

    draw_boxes(args.pdf_path, pdff_data, docl_data, args.output)

if __name__ == "__main__":
    main()
