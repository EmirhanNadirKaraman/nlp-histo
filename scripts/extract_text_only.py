# #!/usr/bin/env python3
# """
# Extract main text content from Docling JSON, filtering out tables, figures, captions, etc.

# Usage:
#     python extract_text_only.py
#     python extract_text_only.py --json out/docling_full/custom_layout.json
#     python extract_text_only.py --output extracted_text.txt
# """

# import sys
# import json
# from pathlib import Path

# # Element types to KEEP (main narrative text)
# KEEP_TYPES = {
#     'TEXT',
#     'PARAGRAPH',
#     'SECTION_HEADER',
#     'TITLE',
#     'LIST',
#     'LIST_ITEM',
# }

# # Element types to IGNORE
# IGNORE_TYPES = {
#     'TABLE',
#     'RECONSTRUCTED_TABLE',
#     'FIGURE',
#     'PICTURE',
#     'CAPTION',
#     'FOOTNOTE',
#     'PAGE_HEADER',
#     'PAGE_FOOTER',
#     'FORMULA',
#     'CODE',
# }


# def reconstruct_tables_from_elements(elements, threshold_multiplier=1.2):
#     """
#     Reconstruct tables from elements to identify which elements belong to tables.
#     Also marks native TABLE elements for exclusion.
#     Returns tuple of (reconstructed_elements, table_sub_element_indices).
#     """
#     reconstructed = []
#     table_indices = set()  # Indices of elements that belong to tables

#     i = 0
#     while i < len(elements):
#         el = elements[i]
#         elem_type = el.get("type")

#         # Mark native TABLE elements for exclusion
#         if elem_type == "TABLE":
#             table_indices.add(i)
#             reconstructed.append(el)
#             i += 1
#             continue

#         # Identify the start of a table via its caption
#         if elem_type == "CAPTION" and "table" in (el.get("text") or "").lower():
#             # Caption itself should be excluded
#             table_indices.add(i)
#             reconstructed.append(el)

#             table_group = {
#                 "type": "RECONSTRUCTED_TABLE",
#                 "caption": el.get("text"),
#                 "page": el.get("page"),
#                 "sub_elements": [],
#                 "sub_element_indices": []
#             }

#             # Initialize with a safe default, then refine based on actual spacing
#             max_allowed_gap = 20
#             last_y2 = el["bbox"]["y2"]
#             i += 1

#             while i < len(elements):
#                 next_el = elements[i]
#                 if next_el.get("type") not in ["TEXT", "LIST_ITEM"]:
#                     break

#                 current_y1 = next_el["bbox"]["y1"]
#                 vertical_gap = abs(current_y1 - last_y2)

#                 # REFINEMENT: After capturing the first item, use the gap between
#                 # item 1 and item 2 as the true baseline gutter spacing
#                 if len(table_group["sub_elements"]) == 1:
#                     first_item_y2 = table_group["sub_elements"][0]["bbox"]["y2"]
#                     true_gutter = abs(current_y1 - first_item_y2)
#                     max_allowed_gap = true_gutter * threshold_multiplier

#                 if vertical_gap < max_allowed_gap:
#                     table_group["sub_elements"].append(next_el)
#                     table_group["sub_element_indices"].append(i)
#                     table_indices.add(i)  # Mark this element as part of a table
#                     last_y2 = next_el["bbox"]["y2"]
#                     i += 1
#                 else:
#                     break

#             if table_group["sub_elements"]:
#                 reconstructed.append(table_group)
#         else:
#             reconstructed.append(el)
#             i += 1

#     return reconstructed, table_indices


# def load_docling_json(json_path: Path):
#     """Load Docling full layout JSON."""
#     if not json_path.exists():
#         print(f"❌ JSON not found: {json_path}")
#         return None

#     try:
#         with open(json_path, 'r') as f:
#             data = json.load(f)

#         elements = data.get('elements', [])
#         print(f"Loaded: {len(elements)} total elements")
#         return elements
#     except Exception as e:
#         print(f"❌ Error loading JSON: {e}")
#         return None


# def filter_text_elements(elements, table_indices):
#     """Filter to keep only main text elements, excluding table sub-elements."""
#     filtered = []
#     type_counts = {}
#     excluded_count = 0

#     for i, element in enumerate(elements):
#         elem_type = element.get('type', 'UNKNOWN')

#         # Skip elements that are part of reconstructed tables
#         if i in table_indices:
#             excluded_count += 1
#             continue

#         # Keep only text-based elements
#         if elem_type in KEEP_TYPES:
#             filtered.append(element)
#             type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

#     print(f"\nExcluded {excluded_count} elements that are part of tables")
#     print(f"Filtered to {len(filtered)} text elements:")
#     for elem_type, count in sorted(type_counts.items()):
#         print(f"  {elem_type:20s}: {count:4d}")

#     return filtered


# def extract_text_content(elements):
#     """Extract text from elements in reading order."""
#     lines = []
#     current_page = None

#     for element in elements:
#         page = element.get('page')
#         elem_type = element.get('type', 'UNKNOWN')
#         text = element.get('text', '').strip()

#         if not text:
#             continue

#         # Add page separator
#         if page != current_page:
#             if current_page is not None:
#                 lines.append('')  # Blank line between pages
#                 lines.append('='*80)
#                 lines.append(f'PAGE {page}')
#                 lines.append('='*80)
#                 lines.append('')
#             current_page = page

#         # Format based on element type
#         if elem_type in ['SECTION_HEADER', 'TITLE']:
#             lines.append('')
#             lines.append(text.upper())
#             lines.append('-' * len(text))
#         elif elem_type in ['LIST', 'LIST_ITEM']:
#             lines.append(f"• {text}")
#         else:
#             lines.append(text)

#     return '\n'.join(lines)


# def extract_text_to_file(json_path: str, output_path: str = None):
#     """
#     Extract main text content from Docling JSON and save to text file.

#     Args:
#         json_path: Path to Docling full layout JSON
#         output_path: Where to save text file (default: auto-generate)
#     """
#     json_file = Path(json_path)

#     # Load elements
#     elements = load_docling_json(json_file)
#     if not elements:
#         return

#     # Reconstruct tables to identify which elements belong to tables
#     print("\nIdentifying table elements...")
#     _, table_indices = reconstruct_tables_from_elements(elements)

#     # Count native tables vs reconstructed tables
#     native_tables = sum(1 for i in range(len(elements))
#                        if elements[i].get("type") == "TABLE")
#     table_captions = sum(1 for i in range(len(elements))
#                         if elements[i].get("type") == "CAPTION"
#                         and "table" in (elements[i].get("text") or "").lower())

#     print(f"Found {native_tables} native TABLE elements")
#     print(f"Found {table_captions} table captions with reconstructed content")
#     print(f"Total elements to exclude: {len(table_indices)} (tables + captions + table rows/cells)")

#     # Filter to text elements only, excluding table sub-elements
#     text_elements = filter_text_elements(elements, table_indices)
#     if not text_elements:
#         print("❌ No text elements found")
#         return

#     # Extract text content
#     print("Extracting text content...")
#     text_content = extract_text_content(text_elements)

#     # Determine output path
#     if not output_path:
#         output_dir = Path("out/extracted_text")
#         output_dir.mkdir(parents=True, exist_ok=True)
#         output_path = output_dir / f"{json_file.stem}_text.txt"
#     else:
#         output_path = Path(output_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)

#     # Write to file
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(text_content)

#     print(f"\n✅ Text extracted to: {output_path}")
#     print(f"   Total characters: {len(text_content):,}")
#     print(f"   Total lines: {len(text_content.splitlines()):,}")

#     return output_path


# def main():
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Extract main text from Docling JSON (no tables/figures/captions)"
#     )
#     parser.add_argument(
#         "pdf_path",
#         nargs='?',
#         default="files/organized_pdfs/PMC1448691_his_2369.pdf",
#         help="Path to PDF file (for auto-detecting JSON)"
#     )
#     parser.add_argument(
#         "--json",
#         help="Path to Docling full layout JSON (default: auto-detect)"
#     )
#     parser.add_argument(
#         "--output", "-o",
#         help="Output text file path (default: out/extracted_text/<name>_text.txt)"
#     )

#     args = parser.parse_args()

#     # Auto-detect JSON if not specified
#     if args.json:
#         json_path = args.json
#     else:
#         pdf_file = Path(args.pdf_path)
#         json_path = Path("out/docling_full") / f"{pdf_file.stem}_full_layout.json"
#         if not json_path.exists():
#             print(f"❌ Auto-detect failed. JSON not found at: {json_path}")
#             print("   Run: python scripts/extract_figures_docling.py")
#             print("   Or specify JSON with: --json /path/to/file.json")
#             sys.exit(1)
#         print(f"Auto-detected JSON: {json_path}")

#     extract_text_to_file(json_path, args.output)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Extract main text content from Docling JSON, filtering out tables, figures, captions, etc.

Usage:
    python extract_text_only.py
    python extract_text_only.py --json out/docling_full/custom_layout.json
    python extract_text_only.py --output extracted_text.txt
"""

import sys
import json
from pathlib import Path

# Element types to KEEP (main narrative text)
KEEP_TYPES = {
    'TEXT',
    'PARAGRAPH',
    'SECTION_HEADER',
    'TITLE',
    'LIST',
    'LIST_ITEM',
}

# Element types to IGNORE
IGNORE_TYPES = {
    'TABLE',
    'RECONSTRUCTED_TABLE',
    'FIGURE',
    'PICTURE',
    'CAPTION',
    'FOOTNOTE',
    'PAGE_HEADER',
    'PAGE_FOOTER',
    'FORMULA',
    'CODE',
}


def bbox_intersects(bbox1, bbox2):
    """
    Check if two bounding boxes intersect.
    Bboxes are dicts with keys: x1, y1, x2, y2
    """
    if not bbox1 or not bbox2:
        return False

    # Get coordinates, handling potential min/max swaps
    x1_min = min(bbox1.get('x1', 0), bbox1.get('x2', 0))
    x1_max = max(bbox1.get('x1', 0), bbox1.get('x2', 0))
    y1_min = min(bbox1.get('y1', 0), bbox1.get('y2', 0))
    y1_max = max(bbox1.get('y1', 0), bbox1.get('y2', 0))

    x2_min = min(bbox2.get('x1', 0), bbox2.get('x2', 0))
    x2_max = max(bbox2.get('x1', 0), bbox2.get('x2', 0))
    y2_min = min(bbox2.get('y1', 0), bbox2.get('y2', 0))
    y2_max = max(bbox2.get('y1', 0), bbox2.get('y2', 0))

    # Check if boxes don't overlap (then negate)
    # Boxes don't overlap if one is completely to the left, right, above, or below the other
    no_overlap = (x1_max < x2_min or  # bbox1 is to the left of bbox2
                  x2_max < x1_min or  # bbox2 is to the left of bbox1
                  y1_max < y2_min or  # bbox1 is above bbox2
                  y2_max < y1_min)    # bbox2 is above bbox1

    return not no_overlap


def collect_table_bboxes(elements, threshold_multiplier=1.2):
    """
    Collect all table bounding boxes from native TABLEs and reconstructed tables.
    Returns list of (page, bbox) tuples for all tables.
    """
    table_bboxes = []

    i = 0
    while i < len(elements):
        el = elements[i]
        elem_type = el.get("type")
        page = el.get("page")

        # Native TABLE elements
        if elem_type == "TABLE":
            bbox = el.get("bbox")
            if bbox and page:
                table_bboxes.append((page, bbox))
            i += 1
            continue

        # Reconstructed tables from captions
        if elem_type == "CAPTION" and "table" in (el.get("text") or "").lower():
            caption_bbox = el.get("bbox")
            sub_elements = []

            # Use same proximity logic to group table elements
            max_allowed_gap = 20
            last_y2 = caption_bbox["y2"]
            i += 1

            while i < len(elements):
                next_el = elements[i]
                if next_el.get("type") not in ["TEXT", "LIST_ITEM"]:
                    break

                current_y1 = next_el["bbox"]["y1"]
                vertical_gap = abs(current_y1 - last_y2)

                if len(sub_elements) == 1:
                    first_item_y2 = sub_elements[0]["bbox"]["y2"]
                    true_gutter = abs(current_y1 - first_item_y2)
                    max_allowed_gap = true_gutter * threshold_multiplier

                if vertical_gap < max_allowed_gap:
                    sub_elements.append(next_el)
                    last_y2 = next_el["bbox"]["y2"]
                    i += 1
                else:
                    break

            # Create combined bbox for caption + all sub-elements
            if sub_elements:
                all_bboxes = [caption_bbox] + [e["bbox"] for e in sub_elements]
                combined_bbox = {
                    "x1": min(b["x1"] for b in all_bboxes),
                    "y1": max(b["y1"] for b in all_bboxes),
                    "x2": max(b["x2"] for b in all_bboxes),
                    "y2": min(b["y2"] for b in all_bboxes)
                }
                table_bboxes.append((page, combined_bbox))
            else:
                # Just caption, no sub-elements
                table_bboxes.append((page, caption_bbox))
        else:
            i += 1

    return table_bboxes


def load_docling_json(json_path: Path):
    """Load Docling full layout JSON."""
    if not json_path.exists():
        print(f"❌ JSON not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        elements = data.get('elements', [])
        print(f"Loaded: {len(elements)} total elements")
        return elements
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return None


def filter_text_elements(elements, table_bboxes):
    """
    Filter to keep only main text elements, excluding elements that intersect with tables.

    Args:
        elements: List of all elements
        table_bboxes: List of (page, bbox) tuples for all tables
    """
    filtered = []
    type_counts = {}
    excluded_count = 0

    for element in elements:
        elem_type = element.get('type', 'UNKNOWN')
        elem_page = element.get('page')
        elem_bbox = element.get('bbox')

        # Skip elements that intersect with any table bbox
        intersects_table = False
        if elem_bbox and elem_page:
            for table_page, table_bbox in table_bboxes:
                # Only check intersection if on same page
                if table_page == elem_page:
                    if bbox_intersects(elem_bbox, table_bbox):
                        intersects_table = True
                        break

        if intersects_table:
            excluded_count += 1
            continue

        # Keep only text-based elements
        if elem_type in KEEP_TYPES:
            filtered.append(element)
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

    print(f"\nExcluded {excluded_count} elements that intersect with tables")
    print(f"Filtered to {len(filtered)} text elements:")
    for elem_type, count in sorted(type_counts.items()):
        print(f"  {elem_type:20s}: {count:4d}")

    return filtered


def extract_text_content(elements):
    """Extract text from elements in reading order."""
    lines = []
    current_page = None

    for element in elements:
        page = element.get('page')
        elem_type = element.get('type', 'UNKNOWN')
        text = element.get('text', '').strip()

        if not text:
            continue

        # Add page separator
        if page != current_page:
            if current_page is not None:
                lines.append('')  # Blank line between pages
                lines.append('='*80)
                lines.append(f'PAGE {page}')
                lines.append('='*80)
                lines.append('')
            current_page = page

        # Format based on element type
        if elem_type in ['SECTION_HEADER', 'TITLE']:
            lines.append('')
            lines.append(text.upper())
            lines.append('-' * len(text))
        elif elem_type in ['LIST', 'LIST_ITEM']:
            lines.append(f"• {text}")
        else:
            lines.append(text)

    return '\n'.join(lines)


def extract_text_to_file(json_path: str, output_path: str = None):
    """
    Extract main text content from Docling JSON and save to text file.

    Args:
        json_path: Path to Docling full layout JSON
        output_path: Where to save text file (default: auto-generate)
    """
    json_file = Path(json_path)

    # Load elements
    elements = load_docling_json(json_file)
    if not elements:
        return

    # Collect all table bounding boxes
    print("\nIdentifying table bounding boxes...")
    table_bboxes = collect_table_bboxes(elements)

    # Count tables
    native_tables = sum(1 for e in elements if e.get("type") == "TABLE")
    table_captions = sum(1 for e in elements
                        if e.get("type") == "CAPTION"
                        and "table" in (e.get("text") or "").lower())

    print(f"Found {native_tables} native TABLE elements")
    print(f"Found {table_captions} table captions with reconstructed content")
    print(f"Total table regions: {len(table_bboxes)}")

    # Filter to text elements only, excluding elements that intersect with tables
    text_elements = filter_text_elements(elements, table_bboxes)
    if not text_elements:
        print("❌ No text elements found")
        return

    # Extract text content
    print("Extracting text content...")
    text_content = extract_text_content(text_elements)

    # Determine output path
    if not output_path:
        output_dir = Path("out/extracted_text")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{json_file.stem}_text.txt"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text_content)

    print(f"\n✅ Text extracted to: {output_path}")
    print(f"   Total characters: {len(text_content):,}")
    print(f"   Total lines: {len(text_content.splitlines()):,}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract main text from Docling JSON (no tables/figures/captions)"
    )
    parser.add_argument(
        "pdf_path",
        nargs='?',
        default="files/organized_pdfs/PMC1448691_his_2369.pdf",
        help="Path to PDF file (for auto-detecting JSON)"
    )
    parser.add_argument(
        "--json",
        help="Path to Docling full layout JSON (default: auto-detect)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output text file path (default: out/extracted_text/<name>_text.txt)"
    )

    args = parser.parse_args()

    # Auto-detect JSON if not specified
    if args.json:
        json_path = args.json
    else:
        pdf_file = Path(args.pdf_path)
        json_path = Path("out/docling_full") / f"{pdf_file.stem}_full_layout.json"
        if not json_path.exists():
            print(f"❌ Auto-detect failed. JSON not found at: {json_path}")
            print("   Run: python scripts/extract_figures_docling.py")
            print("   Or specify JSON with: --json /path/to/file.json")
            sys.exit(1)
        print(f"Auto-detected JSON: {json_path}")

    extract_text_to_file(json_path, args.output)


if __name__ == "__main__":
    main()
