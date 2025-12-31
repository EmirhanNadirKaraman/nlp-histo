#!/usr/bin/env python3
"""
Extract figures and tables from PDFs using both pdffigures2 and Docling,
then create a combined visualization showing all detections.
"""

import json
import subprocess
from pathlib import Path
import fitz  # PyMuPDF

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("Warning: Docling not installed. Install with: pip install docling")
    DocumentConverter = None

# Color scheme for visualization (RGB 0.0 to 1.0)
COLORS = {
    'pdffigures_figure': (0, 0.5, 1),      # Blue
    'pdffigures_table': (0, 0.8, 0),       # Green
    'docling_figure': (1, 0, 0.5),         # Pink/Magenta
    'docling_table': (0.8, 0.4, 0),        # Orange
    'florence_figure': (0.5, 0, 1),        # Purple
    'florence_table': (1, 0.5, 0),         # Bright Orange
}

def run_pdffigures2(pdf_path: Path, output_json: Path, force: bool = False) -> bool:
    """Run pdffigures2 extraction using Java."""
    if output_json.exists() and not force:
        print(f"✓ Skipping pdffigures2 (already exists): {output_json}")
        return False

    print("\nRunning pdffigures2 extraction...")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    figures_dir = Path("out/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    jar_path = Path("pdffigures2.jar")
    if not jar_path.exists():
        for p in [Path("parsers/extractors/pdffigures2.jar"), Path("lib/pdffigures2.jar")]:
            if p.exists():
                jar_path = p
                break
        else:
            print("Warning: pdffigures2.jar not found in current directory.")
            return False

    try:
        result = subprocess.run(
            ['java', '-cp', str(jar_path), 'org.allenai.pdffigures2.FigureExtractor',
             str(pdf_path), '-m', str(output_json), '-d', str(figures_dir)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print(f"✓ pdffigures2 complete: {output_json}")
            return True
        print(f"Warning: pdffigures2 failed: {result.stderr}")
        return False
    except Exception as e:
        print(f"Warning: pdffigures2 execution error: {e}")
        return False

def run_docling_extraction(pdf_path: Path, output_json: Path, force: bool = False) -> bool:
    """Run Docling extraction and save standardized JSON with Bottom-Left coords."""
    if output_json.exists() and not force:
        print(f"✓ Skipping Docling (already exists): {output_json}")
        return False

    if DocumentConverter is None:
        return False

    print("\nRunning Docling extraction...")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        figures, tables = [], []

        for element, _level in doc.iterate_items():
            element_type = type(element).__name__
            prov = element.prov[0] if element.prov else None
            if not (prov and hasattr(prov, 'bbox')):
                continue

            bbox = prov.bbox
            # Docling coordinates in your environment use Bottom-Left origin
            data = {
                "page": prov.page_no,
                "bbox": {
                    "x1": bbox.l, 
                    "y1": bbox.b,  # Bottom
                    "x2": bbox.r, 
                    "y2": bbox.t   # Top
                },
                "caption": getattr(element, 'caption', None).text if getattr(element, 'caption', None) else None
            }

            if element_type == "PictureItem":
                figures.append(data)
            elif element_type == "TableItem":
                tables.append(data)

        output_data = {
            "pdf_path": str(pdf_path),
            "total_figures": len(figures),
            "total_tables": len(tables),
            "figures": figures,
            "tables": tables
        }

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Docling complete: {output_json}")
        return True
    except Exception as e:
        print(f"Warning: Docling failed: {e}")
        return False

def run_florence_extraction(pdf_path: Path, output_json: Path, force: bool = False) -> bool:
    """
    Run Florence-2 extraction and save standardized JSON.

    To implement: Edit scripts/florence_extractor_template.py
    """
    if output_json.exists() and not force:
        print(f"✓ Skipping Florence-2 (already exists): {output_json}")
        return False

    print("\nRunning Florence-2 extraction...")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try to import the Florence-2 implementation
        try:
            from scripts.florence_extractor import extract_from_pdf
            results = extract_from_pdf(str(pdf_path))
            figures = results.get("figures", [])
            tables = results.get("tables", [])
        except ImportError as e:
            print(f"Warning: Florence-2 dependencies not available: {e}")
            print("Install with: brew install poppler && pip install torch transformers pillow pdf2image")
            figures, tables = [], []

        output_data = {
            "pdf_path": str(pdf_path),
            "extraction_tool": "Florence-2",
            "total_figures": len(figures),
            "total_tables": len(tables),
            "figures": figures,
            "tables": tables
        }

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Florence-2 complete: {output_json} ({len(figures)} figures, {len(tables)} tables)")
        return True
    except Exception as e:
        print(f"Warning: Florence-2 extraction failed: {e}")
        return False


def get_rect(bbox, page_height, origin="bottom-left"):
    """
    Standardizes bounding box coordinates for PyMuPDF.
    Ensures min/max logic to prevent flipped rectangles.
    """
    try:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        if origin == "bottom-left":
            # Map Bottom-Left (PDF) to Top-Left (PyMuPDF)
            # Use max(y1, y2) for the top edge because y increases upwards in PDF
            return fitz.Rect(x1, page_height - max(y1, y2), x2, page_height - min(y1, y2))
        else:
            # Top-left origin: no flip needed
            return fitz.Rect(x1, min(y1, y2), x2, max(y1, y2))
    except (KeyError, TypeError):
        return None

def create_visualization(pdf_path: Path, pdffigures_json: Path, docling_json: Path, output_pdf: Path, force: bool = False):
    """Draws boxes from both tools onto a new PDF using unified Bottom-Left logic."""
    if output_pdf.exists() and not force:
        print(f"✓ Skipping visualization: {output_pdf}")
        return

    print("\nCreating visualization...")
    
    pf_data = None
    if pdffigures_json.exists():
        with open(pdffigures_json, 'r') as f:
            raw = json.load(f)
            pf_data = {
                'figures': [i for i in raw if i.get('figType') == 'Figure'],
                'tables': [i for i in raw if i.get('figType') == 'Table']
            }
    
    dl_data = None
    if docling_json.exists():
        with open(docling_json, 'r') as f:
            dl_data = json.load(f)

    # Load Florence-2 results (add florence_json parameter to function signature)
    florence_json = Path("out/florence") / f"{pdf_path.stem}_florence.json"
    florence_data = None
    if florence_json.exists():
        with open(florence_json, 'r') as f:
            florence_data = json.load(f)

    doc = fitz.open(str(pdf_path))

    # Updated: All three tools with their respective coordinate systems
    configs = [
        (pf_data, 'figures', 'pdffigures_figure', 'PDF2-F', 'regionBoundary', 'bottom-left'),
        (pf_data, 'tables', 'pdffigures_table', 'PDF2-T', 'regionBoundary', 'bottom-left'),
        (dl_data, 'figures', 'docling_figure', 'DOC-F', 'bbox', 'bottom-left'),
        (dl_data, 'tables', 'docling_table', 'DOC-T', 'bbox', 'bottom-left'),
        (florence_data, 'figures', 'florence_figure', 'FLO-F', 'bbox', 'bottom-left'),
        (florence_data, 'tables', 'florence_table', 'FLO-T', 'bbox', 'bottom-left'),
    ]

    for page_num in range(len(doc)):
        page = doc[page_num]
        h = page.rect.height
        p_no = page_num + 1

        for data, key, c_key, pref, b_key, origin in configs:
            if not data: continue
            items = data.get(key, [])
            for i, item in enumerate(items, 1):
                if item.get('page') == p_no:
                    rect = get_rect(item.get(b_key), h, origin=origin)
                    if rect:
                        is_dl = "docling" in c_key
                        page.draw_rect(rect, color=COLORS[c_key], width=2 if is_dl else 3, 
                                      dashes="[3] 0" if is_dl else None)
                        # Offset label slightly to avoid overlap
                        page.insert_text((rect.x0, rect.y0 - 2), f"{pref}{i}", 
                                        fontsize=8, color=COLORS[c_key])

    # Legend (Page 1) - Extended for 3 tools
    if len(doc) > 0:
        page = doc[0]
        page.draw_rect(fitz.Rect(15, 15, 180, 105), color=(0,0,0), fill=(1,1,1), width=0.5)
        page.insert_text((20, 30), "LEGEND:", fontsize=9, color=(0,0,0))
        legend_items = [
            ("PDFfigures2 Fig", 'pdffigures_figure'),
            ("PDFfigures2 Tbl", 'pdffigures_table'),
            ("Docling Fig", 'docling_figure'),
            ("Docling Tbl", 'docling_table'),
            ("Florence-2 Fig", 'florence_figure'),
            ("Florence-2 Tbl", 'florence_table'),
        ]
        for i, (label, color_key) in enumerate(legend_items):
            y = 45 + (i * 10)
            page.draw_line((20, y-3), (35, y-3), color=COLORS[color_key], width=2)
            page.insert_text((40, y), label, fontsize=8, color=(0,0,0))

    doc.save(str(output_pdf))
    doc.close()
    print(f"✓ Visualization saved: {output_pdf}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-tool PDF visualizer")
    parser.add_argument("pdf_path", help="Input PDF")
    parser.add_argument("--force", action="store_true", help="Re-run everything")
    parser.add_argument("--skip-florence", action="store_true", help="Skip Florence-2 extraction")
    args = parser.parse_args()

    pdf_file = Path(args.pdf_path)
    pf_json = Path("out/data") / f"{pdf_file.stem}.json"
    dl_json = Path("out/docling") / f"{pdf_file.stem}_docling.json"
    florence_json = Path("out/florence") / f"{pdf_file.stem}_florence.json"
    viz_pdf = Path("out/comparisons") / f"{pdf_file.stem}_comparison.pdf"

    run_pdffigures2(pdf_file, pf_json, args.force)
    run_docling_extraction(pdf_file, dl_json, args.force)

    if not args.skip_florence:
        run_florence_extraction(pdf_file, florence_json, args.force)
    else:
        print("Skipping Florence-2 (--skip-florence)")

    create_visualization(pdf_file, pf_json, dl_json, viz_pdf, args.force)

if __name__ == "__main__":
    main()