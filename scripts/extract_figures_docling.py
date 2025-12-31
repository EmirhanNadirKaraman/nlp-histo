# # #!/usr/bin/env python3
# # """
# # Extract figure and table coordinates using Docling.

# # Usage:
# #     python extract_figures_docling.py path/to/file.pdf
# #     python extract_figures_docling.py path/to/file.pdf --output results/docling_output.json
# # """

# # import sys
# # import json
# # from pathlib import Path
# # from datetime import datetime

# # try:
# #     from docling.document_converter import DocumentConverter
# # except ImportError:
# #     print("Error: Docling not installed. Run: pip install docling")
# #     sys.exit(1)


# # def extract_with_docling(pdf_path: str, output_path: str = None):
# #     """
# #     Extract figures and tables from PDF using Docling.

# #     Args:
# #         pdf_path: Path to PDF file
# #         output_path: Optional output JSON path

# #     Returns:
# #         Dict with figures and tables data
# #     """
# #     pdf_file = Path(pdf_path)
# #     if not pdf_file.exists():
# #         print(f"Error: PDF not found: {pdf_path}")
# #         sys.exit(1)

# #     print(f"\n{'='*80}")
# #     print(f"Extracting figures and tables with Docling")
# #     print(f"{'='*80}\n")
# #     print(f"Input:  {pdf_path}\n")

# #     # Create converter with default settings
# #     converter = DocumentConverter()

# #     print("Processing PDF...\n")

# #     # Convert document
# #     result = converter.convert(str(pdf_file))
# #     doc = result.document

# #     # Extract figures and tables with coordinates
# #     figures = []
# #     tables = []

# #     # Iterate through all document elements
# #     for element, level in doc.iterate_items():
# #         element_type = type(element).__name__

# #         # Extract figures
# #         if element_type == "PictureItem":
# #             prov = element.prov[0] if element.prov else None
# #             if prov and hasattr(prov, 'bbox'):
# #                 bbox = prov.bbox
# #                 figure_data = {
# #                     "type": "Figure",
# #                     "page": prov.page_no if hasattr(prov, 'page_no') else None,
# #                     "bbox": {
# #                         "x1": bbox.l,  # left
# #                         "y1": bbox.t,  # top
# #                         "x2": bbox.r,  # right
# #                         "y2": bbox.b   # bottom
# #                     },
# #                     "caption": element.caption.text if hasattr(element, 'caption') and element.caption else None,
# #                     "text": element.text if hasattr(element, 'text') else None
# #                 }
# #                 figures.append(figure_data)

# #         # Extract tables
# #         elif element_type == "TableItem":
# #             prov = element.prov[0] if element.prov else None
# #             if prov and hasattr(prov, 'bbox'):
# #                 bbox = prov.bbox
# #                 table_data = {
# #                     "type": "Table",
# #                     "page": prov.page_no if hasattr(prov, 'page_no') else None,
# #                     "bbox": {
# #                         "x1": bbox.l,
# #                         "y1": bbox.t,
# #                         "x2": bbox.r,
# #                         "y2": bbox.b
# #                     },
# #                     "caption": element.caption.text if hasattr(element, 'caption') and element.caption else None,
# #                     "text": element.text if hasattr(element, 'text') else None,
# #                     # Table-specific data
# #                     "num_rows": element.num_rows if hasattr(element, 'num_rows') else None,
# #                     "num_cols": element.num_cols if hasattr(element, 'num_cols') else None
# #                 }
# #                 tables.append(table_data)

# #     # Get page dimensions for normalization
# #     page_dimensions = {}
# #     if hasattr(doc, 'pages'):
# #         for page in doc.pages:
# #             if hasattr(page, 'page_no') and hasattr(page, 'size'):
# #                 page_dimensions[page.page_no] = {
# #                     "width": page.size.width,
# #                     "height": page.size.height
# #                 }

# #     # Prepare output
# #     output_data = {
# #         "pdf_path": str(pdf_file),
# #         "extraction_tool": "Docling",
# #         "extraction_date": datetime.now().isoformat(),
# #         "total_figures": len(figures),
# #         "total_tables": len(tables),
# #         "page_dimensions": page_dimensions,
# #         "figures": figures,
# #         "tables": tables
# #     }

# #     # Print summary
# #     print(f"{'='*80}")
# #     print("Extraction Summary")
# #     print(f"{'='*80}\n")
# #     print(f"Figures found: {len(figures)}")
# #     print(f"Tables found:  {len(tables)}")
# #     print(f"Total pages:   {len(page_dimensions)}")
# #     print()

# #     # Display details
# #     if figures:
# #         print("Figures:")
# #         for i, fig in enumerate(figures, 1):
# #             page = fig.get('page', 'N/A')
# #             caption = fig.get('caption', 'No caption')
# #             caption_preview = caption[:60] + "..." if caption and len(caption) > 60 else caption
# #             print(f"  {i}. Page {page}: {caption_preview}")

# #     if tables:
# #         print("\nTables:")
# #         for i, tbl in enumerate(tables, 1):
# #             page = tbl.get('page', 'N/A')
# #             caption = tbl.get('caption', 'No caption')
# #             caption_preview = caption[:60] + "..." if caption and len(caption) > 60 else caption
# #             dims = f"{tbl.get('num_rows', '?')}x{tbl.get('num_cols', '?')}"
# #             print(f"  {i}. Page {page} ({dims}): {caption_preview}")

# #     # Save to file
# #     if output_path:
# #         output_file = Path(output_path)
# #     else:
# #         output_dir = Path("out/docling")
# #         output_dir.mkdir(parents=True, exist_ok=True)
# #         output_file = output_dir / f"{pdf_file.stem}_docling.json"

# #     output_file.parent.mkdir(parents=True, exist_ok=True)

# #     with open(output_file, 'w', encoding='utf-8') as f:
# #         json.dump(output_data, f, indent=2, ensure_ascii=False)

# #     print(f"\n✓ Results saved to: {output_file}")
# #     print()

# #     return output_data


# # def main():
# #     import argparse

# #     parser = argparse.ArgumentParser(
# #         description="Extract figure and table coordinates using Docling"
# #     )
# #     parser.add_argument(
# #         "pdf_path",
# #         help="Path to PDF file"
# #     )
# #     parser.add_argument(
# #         "--output",
# #         "-o",
# #         help="Output JSON path (default: out/docling/{filename}_docling.json)"
# #     )

# #     args = parser.parse_args()
# #     extract_with_docling(args.pdf_path, args.output)


# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# """
# High-Precision Table & Figure Coordinator for Docling (Updated for v2.x API).
# Optimized for detection coordinates even when structural parsing fails.
# """

# import sys
# import json
# from pathlib import Path
# from datetime import datetime

# try:
#     from docling.document_converter import DocumentConverter, PdfFormatOption
#     from docling.datamodel.pipeline_options import PdfPipelineOptions
#     from docling.datamodel.base_models import InputFormat
#     from docling_core.types.doc import DocItemLabel
# except ImportError:
#     print("Error: Missing dependencies. Run: pip install docling docling-core")
#     sys.exit(1)

# def extract_with_docling(pdf_path: str, output_path: str = None):
#     pdf_file = Path(pdf_path)
#     if not pdf_file.exists():
#         print(f"Error: PDF not found: {pdf_path}")
#         sys.exit(1)

#     print(f"\n{'='*80}")
#     print(f"Docling Robust Coordinator Extraction (v2.x API)")
#     print(f"{'='*80}\n")

#     # 1. CONFIGURE FOR MAXIMUM DETECTION
#     # We disable table_structure because we only need the outer bounding box.
#     pipeline_options = PdfPipelineOptions()
#     pipeline_options.do_table_structure = False 
#     pipeline_options.do_ocr = True  # Catches tables embedded in images
    
#     # Optional: Increase image scale for better small-table detection
#     pipeline_options.images_scale = 2.0 

#     # In Docling v2.x, options must be passed via format_options
#     converter = DocumentConverter(
#         format_options={
#             InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
#         }
#     )

#     print(f"Processing: {pdf_file.name}...")
#     result = converter.convert(str(pdf_file))
#     doc = result.document

#     figures = []
#     tables = []

#     # 2. ITERATE WITH HEURISTICS
#     for element, level in doc.iterate_items():
#         # Check label using the official DocItemLabel enum
#         label = getattr(element, "label", None)
        
#         # Ensure element has coordinate data (provenance)
#         if not (hasattr(element, 'prov') and element.prov):
#             continue
            
#         prov = element.prov[0]
#         bbox = prov.bbox # Docling's bbox object (l, t, r, b)
        
#         # Prepare standard coordinate object for your visualization script
#         coords = {
#             "x1": bbox.l,
#             "y1": bbox.t,
#             "x2": bbox.r,
#             "y2": bbox.b
#         }

#         caption_text = ""
#         if hasattr(element, 'caption') and element.caption:
#             caption_text = element.caption.text or ""

#         # BASE DATA STRUCTURE
#         item_data = {
#             "page": prov.page_no,
#             "bbox": coords,
#             "caption": caption_text,
#             "label_used": str(label)
#         }

#         # HEURISTIC 1: Standard Tables
#         if label == DocItemLabel.TABLE:
#             item_data["type"] = "Table"
#             tables.append(item_data)

#         # HEURISTIC 2: Pictures that might be Tables (Caption Check)
#         elif label == DocItemLabel.PICTURE:
#             if "table" in caption_text.lower():
#                 item_data["type"] = "Table"
#                 item_data["note"] = "Reclassified from Picture via caption"
#                 tables.append(item_data)
#             else:
#                 item_data["type"] = "Figure"
#                 figures.append(item_data)

#         # HEURISTIC 3: Lists that are likely borderless tables
#         elif label == DocItemLabel.LIST:
#             item_data["type"] = "Table"
#             item_data["note"] = "Detected as List (Potential borderless table)"
#             tables.append(item_data)

#     # 3. PAGE DIMENSIONS
#     page_dimensions = {}
#     for page_no, page in doc.pages.items():
#         page_dimensions[page_no] = {
#             "width": page.size.width,
#             "height": page.size.height
#         }

#     # PREPARE FINAL OUTPUT
#     output_data = {
#         "pdf_path": str(pdf_file),
#         "extraction_tool": "Docling_Robust_v2",
#         "total_figures": len(figures),
#         "total_tables": len(tables),
#         "page_dimensions": page_dimensions,
#         "figures": figures,
#         "tables": tables
#     }

#     # SAVE LOGIC
#     if not output_path:
#         output_dir = Path("out/docling")
#         output_dir.mkdir(parents=True, exist_ok=True)
#         output_path = output_dir / f"{pdf_file.stem}_docling.json"

#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(output_data, f, indent=2, ensure_ascii=False)

#     print(f"\nSummary for {pdf_file.name}:")
#     print(f"  - Figures: {len(figures)}")
#     print(f"  - Tables:  {len(tables)}")
#     print(f"✓ Saved coordinates to: {output_path}\n")

#     return output_data

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Extract coordinates using robust Docling v2 heuristics")
#     parser.add_argument(
#         "pdf_path",
#         nargs='?',  # Make it optional
#         default="files/organized_pdfs/PMC1448691_his_2369.pdf",
#         help="Path to the PDF file (default: PMC1448691_his_2369.pdf)"
#     )
#     parser.add_argument("--output", "-o", help="Custom output JSON path")

#     args = parser.parse_args()
#     extract_with_docling(args.pdf_path, args.output)

#!/usr/bin/env python3
"""
Docling Full-Layout Extractor.
Saves every detected element (text, titles, lists, tables, figures) with coordinates.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
except ImportError:
    print("Error: Missing dependencies. Run: pip install docling docling-core")
    sys.exit(1)

def extract_all_from_docling(pdf_path: str, output_path: str = None):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"\n{'='*80}\nFull Document Layout Extraction\n{'='*80}\n")

    # 1. CONFIGURE FOR TOTAL VISIBILITY
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = False # Coordinates only
    pipeline_options.do_ocr = True 
    pipeline_options.images_scale = 2.0 

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"Analyzing everything in: {pdf_file.name}...")
    result = converter.convert(str(pdf_file))
    doc = result.document

    # We will store everything in one list called 'elements'
    all_elements = []

    # 2. CAPTURE EVERY SINGLE ITEM
    for element, level in doc.iterate_items():
        # Clean up the label name (e.g., 'DocItemLabel.PARAGRAPH' -> 'PARAGRAPH')
        label_str = str(getattr(element, "label", "UNKNOWN")).split('.')[-1].upper()
        
        # Ensure the element has spatial data
        if not (hasattr(element, 'prov') and element.prov):
            continue
            
        prov = element.prov[0]
        bbox = prov.bbox
        
        # Capture text content safely
        text_content = ""
        if hasattr(element, 'text'):
            text_content = element.text
        elif hasattr(element, 'caption') and element.caption:
            text_content = element.caption.text

        # Create the comprehensive data object
        item_data = {
            "type": label_str,
            "page": prov.page_no,
            "level": level, # Hierarchical depth in the document
            "bbox": {
                "x1": bbox.l,
                "y1": bbox.t,
                "x2": bbox.r,
                "y2": bbox.b
            },
            "text": text_content.strip() if text_content else None
        }
        
        all_elements.append(item_data)

    # 3. PAGE METADATA
    page_dimensions = {
        no: {"width": p.size.width, "height": p.size.height} 
        for no, p in doc.pages.items()
    }

    # 4. PREPARE JSON
    output_data = {
        "metadata": {
            "pdf_path": str(pdf_file),
            "tool": "Docling_v2_Full_Export",
            "extraction_date": datetime.now().isoformat(),
            "total_elements_found": len(all_elements)
        },
        "page_dimensions": page_dimensions,
        "elements": all_elements
    }

    # SAVE
    if not output_path:
        out_dir = Path("out/docling_full")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{pdf_file.stem}_full_layout.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Extraction Complete!")
    print(f"Detected {len(all_elements)} total layout objects.")
    print(f"Full layout JSON saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract coordinates using robust Docling v2 heuristics")
    parser.add_argument(
        "pdf_path",
        nargs='?',  # Make it optional
        default="files/organized_pdfs/PMC1448691_his_2369.pdf",
        help="Path to the PDF file (default: PMC1448691_his_2369.pdf)"
    )
    parser.add_argument("--output", "-o", help="Custom output JSON path")

    args = parser.parse_args()
    extract_all_from_docling(args.pdf_path, args.output)