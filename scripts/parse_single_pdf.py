#!/usr/bin/env python3
"""
Parse a single PDF and save results to files for manual inspection.

Usage:
    python parse_single_pdf.py path/to/file.pdf
    python parse_single_pdf.py path/to/file.pdf --mode fallback
    python parse_single_pdf.py path/to/file.pdf --output-dir results/
"""

import sys
import json
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def detect_references(text):
    """
    Detect references to figures and tables in text.

    Args:
        text: Text string to search

    Returns:
        Dict with 'figures' and 'tables' lists containing reference IDs
    """
    import re

    references = {
        'figures': [],
        'tables': []
    }

    # Figure patterns: "Figure 6", "Fig. 6", "figure 6", "FIG 6"
    fig_patterns = [
        r'\b(?:Figure|Fig\.?|FIG\.?)\s+(\d+[A-Za-z]?)',
        r'\((?:Figure|Fig\.?|FIG\.?)\s+(\d+[A-Za-z]?)\)',
    ]

    for pattern in fig_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            fig_id = match.group(1)
            if fig_id not in references['figures']:
                references['figures'].append(fig_id)

    # Table patterns: "Table 2", "table 2", "TABLE 2"
    table_patterns = [
        r'\b(?:Table|TABLE)\s+(\d+[A-Za-z]?)',
        r'\((?:Table|TABLE)\s+(\d+[A-Za-z]?)\)',
    ]

    for pattern in table_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            table_id = match.group(1)
            if table_id not in references['tables']:
                references['tables'].append(table_id)

    return references


def add_references_to_elements(text_elements):
    """
    Add figure/table references to each text element.

    Args:
        text_elements: List of text element dicts

    Returns:
        Enhanced text elements with 'references' field
    """
    enhanced = []

    for elem in text_elements:
        elem_copy = elem.copy()
        refs = detect_references(elem['text'])

        # Only add references field if there are any
        if refs['figures'] or refs['tables']:
            elem_copy['references'] = refs

        enhanced.append(elem_copy)

    return enhanced


def create_reference_index(text_elements, figures):
    """
    Create an index of all figures and tables with their references.

    Args:
        text_elements: List of text elements with references
        figures: List of extracted figures from PDFFigures

    Returns:
        Dict with figures and tables, each containing:
        - id: Figure/table identifier
        - mentioned_in: List of paths where it's mentioned
        - extracted: Whether we have the actual figure/table data
    """
    index = {
        'figures': {},
        'tables': {}
    }

    # Build index from text references
    for elem in text_elements:
        if 'references' not in elem:
            continue

        path = elem['path_string'] or '(Root)'

        # Index figure references
        for fig_id in elem['references']['figures']:
            if fig_id not in index['figures']:
                index['figures'][fig_id] = {
                    'id': fig_id,
                    'mentioned_in': [],
                    'extracted': False,
                    'figure_data': None
                }
            if path not in index['figures'][fig_id]['mentioned_in']:
                index['figures'][fig_id]['mentioned_in'].append(path)

        # Index table references
        for table_id in elem['references']['tables']:
            if table_id not in index['tables']:
                index['tables'][table_id] = {
                    'id': table_id,
                    'mentioned_in': [],
                    'extracted': False,
                    'table_data': None
                }
            if path not in index['tables'][table_id]['mentioned_in']:
                index['tables'][table_id]['mentioned_in'].append(path)

    # Match with extracted figures from PDFFigures
    for fig in figures:
        fig_id = fig.get('figure_id', '').replace('Figure', '').replace('Fig', '').strip()
        if fig_id in index['figures']:
            index['figures'][fig_id]['extracted'] = True
            index['figures'][fig_id]['figure_data'] = {
                'page': fig.get('page'),
                'caption': fig.get('caption'),
                'image_path': fig.get('image_path'),
                'bbox': fig.get('bbox')
            }

    return index


class ContextAwareStitcher:
    """
    Stitches narrative text while preserving tables and figures.
    If a paragraph is interrupted by a table/figure, it 'looks over'
    the object to find the continuation of the sentence.
    """

    def reconstruct_paragraphs(self, paragraphs):
        """
        Reconstruct paragraphs by stitching split narratives.

        Args:
            paragraphs: List of paragraph strings or dicts

        Returns:
            List of reconstructed paragraphs with tables/figures preserved
        """
        if not paragraphs:
            return []

        # Convert to uniform format with type tags
        elements = []
        for para in paragraphs:
            if isinstance(para, dict):
                text = para['text']
            else:
                text = para

            # Classify paragraph type
            text_start = text.strip()[:50].lower()
            # Check if it's a table (starts with "Table" or is bullet points after a table)
            if text_start.startswith('table '):
                elem_type = 'table'
            # Check if it's a figure
            elif text_start.startswith('figure '):
                elem_type = 'figure'
            # Check if it's table content (bullet points, pipes, etc.)
            elif (text_start.startswith('-') or text_start.startswith('|') or
                  text_start.startswith('*') or text_start.startswith('•')):
                # Likely table content, treat as table to skip over it
                elem_type = 'table'
            else:
                elem_type = 'narrative'

            elements.append({
                'type': elem_type,
                'text': text,
                'consumed': False
            })

        final_flow = []
        i = 0

        while i < len(elements):
            curr = elements[i]

            # If current block is a table or figure, preserve it
            if curr['type'] in ['table', 'figure']:
                final_flow.append(curr['text'])
                i += 1
                continue

            # If current is narrative, check if it's cut off
            if self._is_cut_off(curr['text']):
                # Look ahead for the next NARRATIVE block, skipping tables/figs
                next_text_idx = self._find_next_narrative(elements, i + 1)

                if next_text_idx is not None:
                    next_text_elem = elements[next_text_idx]

                    # Stitch them together
                    curr['text'] = self._merge_text(curr['text'], next_text_elem['text'])

                    # Mark the consumed element
                    elements[next_text_idx]['consumed'] = True

            # Add the (potentially stitched) block if not consumed
            if not curr.get('consumed'):
                final_flow.append(curr['text'])
            i += 1

        return final_flow

    def _is_cut_off(self, text):
        """Check if text appears to be cut off mid-sentence."""
        t = text.strip()
        if not t:
            return False

        # Ends with hyphen (word break)
        if t.endswith('-'):
            return True

        # Ends with comma (sentence continues)
        if t.endswith(','):
            return True

        # Ends with lowercase letter or connector words
        last_word = t.split()[-1] if t.split() else ''
        connectors = ['and', 'or', 'but', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for']
        if last_word.lower() in connectors:
            return True

        # Ends with lowercase (likely mid-sentence)
        if t[-1].islower():
            return True

        return False

    def _find_next_narrative(self, elements, start_idx):
        """Find the next narrative block, skipping tables/figures."""
        for j in range(start_idx, len(elements)):
            if elements[j]['type'] == 'narrative' and not elements[j].get('consumed'):
                return j
        return None

    def _merge_text(self, t1, t2):
        """Merge two text segments intelligently."""
        t1 = t1.rstrip()
        t2 = t2.lstrip()

        # Handle hyphenated word breaks
        if t1.endswith('-'):
            return t1[:-1] + t2

        # Normal merge with space
        return t1 + " " + t2


def remove_citations(text):
    """
    Remove citation numbers from text.

    Citations appear as numbers after periods, like:
    - "text. 1 More text"
    - "text. 19,20 More"
    - "text. 1-3 More"

    Args:
        text: Text with citations

    Returns:
        Text with citations removed
    """
    import re

    # Pattern: period/comma followed by space(s) and number(s) (possibly comma/dash separated)
    # Examples: ". 1 ", ". 19,20 ", ". 1-3 ", ", 5 "
    patterns = [
        r'\.\s+\d+(?:[,–-]\d+)*\s+',  # After period: ". 1 ", ". 19,20 "
        r',\s+\d+(?:[,–-]\d+)*\s+',   # After comma: ", 5 "
        r'\s+\d+(?:[,–-]\d+)*\s+',    # Standalone: " 1 ", " 19,20 "
    ]

    cleaned = text
    for pattern in patterns:
        # Replace citation with just the punctuation and space
        cleaned = re.sub(pattern, lambda m: m.group()[0] + ' ', cleaned)

    return cleaned


def group_by_path(text_elements):
    """
    Group text elements by their hierarchical path.

    Args:
        text_elements: List of text element dicts

    Returns:
        Dict with paths as keys, each containing:
        - paragraphs: List of text strings (or dicts with text + references)
        - depth: Hierarchical depth
        - path_list: Path as list
    """
    grouped = {}

    for elem in text_elements:
        path_string = elem['path_string']

        # Use empty string for root-level elements
        if not path_string:
            path_string = "(Root)"

        # Initialize path group if needed
        if path_string not in grouped:
            grouped[path_string] = {
                'depth': elem['depth'],
                'path_list': elem['path_list'],
                'paragraphs': []
            }

        # Add paragraph with references if present
        if 'references' in elem:
            para_data = {
                'text': elem['text'],
                'references': elem['references']
            }
        else:
            para_data = elem['text']

        grouped[path_string]['paragraphs'].append(para_data)

    return grouped


def parse_and_save(pdf_path: str, mode: str = "routing", output_dir: str = "output"):
    """
    Parse a PDF and save results to files.

    Args:
        pdf_path: Path to PDF file
        mode: "routing" or "fallback"
        output_dir: Directory to save results
    """
    from parsers.pdf_parsers.ensemble_parser import EnsemblePDFParser

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    # Create output directory structure: output/{paper_name}/{timestamp}/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paper_folder = Path(output_dir) / pdf_file.stem / timestamp
    paper_folder.mkdir(parents=True, exist_ok=True)

    # Output path is now the paper-specific timestamped folder
    output_path = paper_folder

    # Simplified base name (mode only, no paper name or timestamp)
    base_name = mode

    print(f"\n{'='*80}")
    print(f"Parsing PDF with Medical-Grade Ensemble Parser")
    print(f"{'='*80}\n")
    print(f"Input:  {pdf_path}")
    print(f"Mode:   {mode}")
    print(f"Output: {output_path}/\n")

    # Create parser
    parser = EnsemblePDFParser(mode=mode)

    # Check availability
    print("Parser Availability:")
    availability = parser.get_available_parsers()
    for parser_name, available in availability.items():
        status = "✓" if available else "✗"
        print(f"  {status} {parser_name}")
    print()

    # Extract
    print("Extracting...\n")
    result = parser.extract_hierarchy(str(pdf_file))

    if not result['success']:
        print("Error: Extraction failed")
        sys.exit(1)

    # Add figure/table references to text elements
    print("Detecting figure and table references...\n")
    result['text_elements'] = add_references_to_elements(result['text_elements'])

    # Save results
    print(f"{'='*80}")
    print("Saving Results")
    print(f"{'='*80}\n")

    # 1. Save full JSON
    json_file = output_path / f"{base_name}_full.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✓ Full JSON:        {json_file}")

    # 2. Save text elements only (easier to read)
    text_json = output_path / f"{base_name}_text_elements.json"
    with open(text_json, 'w', encoding='utf-8') as f:
        json.dump(result['text_elements'], f, indent=2, ensure_ascii=False)
    print(f"✓ Text elements:    {text_json}")

    # 2b. Save grouped by path (array of paragraphs per section)
    grouped_json = output_path / f"{base_name}_grouped.json"
    grouped_data = group_by_path(result['text_elements'])
    with open(grouped_json, 'w', encoding='utf-8') as f:
        json.dump(grouped_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Grouped by path:  {grouped_json}")

    # 2c. Create figure/table index with references
    ref_index = create_reference_index(result['text_elements'], result['figures'])
    if ref_index['figures'] or ref_index['tables']:
        refs_json = output_path / f"{base_name}_references.json"
        with open(refs_json, 'w', encoding='utf-8') as f:
            json.dump(ref_index, f, indent=2, ensure_ascii=False)
        print(f"✓ Reference index:  {refs_json}")

    # 2d. Save grouped readable format (paragraphs separated by empty lines)
    grouped_readable = output_path / f"{base_name}_grouped_readable.txt"
    with open(grouped_readable, 'w', encoding='utf-8') as f:
        f.write(f"PDF Extraction - Grouped by Section\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"File: {pdf_file.name}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Sections: {len(grouped_data)}\n\n")
        f.write(f"{'='*80}\n\n")

        # Create stitcher for reconstructing split paragraphs
        stitcher = ContextAwareStitcher()

        for path, section in grouped_data.items():
            # Section header
            f.write(f"\n{path}\n")
            f.write(f"{'-'*len(path)}\n\n")

            # Apply context-aware stitching to reconstruct split paragraphs
            reconstructed_paras = stitcher.reconstruct_paragraphs(section['paragraphs'])

            # Separate regular paragraphs from table/figure paragraphs
            regular_paras = []
            table_paras = []
            figure_paras = []

            for text in reconstructed_paras:
                # Check if this paragraph IS a table or figure (not just mentions one)
                text_start = text.strip()[:50].lower()

                # Table: starts with "Table" or bullet points/table markers
                if (text_start.startswith('table ') or
                    text_start.startswith('-') or
                    text_start.startswith('|') or
                    text_start.startswith('*') or
                    text_start.startswith('•')):
                    table_paras.append(text)
                # Figure: starts with "Figure"
                elif text_start.startswith('figure '):
                    figure_paras.append(text)
                # Everything else is narrative
                else:
                    regular_paras.append(text)

            # Write regular paragraphs first (with citations removed)
            for para in regular_paras:
                cleaned = remove_citations(para)
                f.write(f"{cleaned}\n\n")

            # Write tables at the end (with citations removed)
            if table_paras:
                f.write(f"\n{'-'*60}\n\n")
                for table in table_paras:
                    cleaned = remove_citations(table)
                    f.write(f"{cleaned}\n\n")

            # Write figures at the end (with citations removed)
            if figure_paras:
                if not table_paras:
                    f.write(f"\n{'-'*60}\n\n")
                for figure in figure_paras:
                    cleaned = remove_citations(figure)
                    f.write(f"{cleaned}\n\n")

            f.write("\n")  # Extra line between sections

    print(f"✓ Grouped readable: {grouped_readable}")

    # 3. Save markdown version (human-readable)
    md_file = output_path / f"{base_name}_readable.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# PDF Extraction Results\n\n")
        f.write(f"**File:** {pdf_file.name}\n\n")
        f.write(f"**Mode:** {mode}\n\n")
        f.write(f"**Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Routing info (if available)
        if 'page_routing' in result:
            f.write(f"## Page Routing\n\n")
            routing = result['page_routing']
            f.write(f"- **Total pages:** {routing['total_pages']}\n")
            f.write(f"- **Summary:** {routing['summary']}\n\n")

            f.write(f"### Per-Page Routing\n\n")
            f.write(f"| Page | Route | Reason | Tables | Images |\n")
            f.write(f"|------|-------|--------|--------|--------|\n")
            for route in routing['routes']:
                f.write(f"| {route['page']} | {route['route']} | {route['reason']} | "
                       f"{'Yes' if route['has_tables'] else 'No'} | "
                       f"{'Yes' if route['has_images'] else 'No'} |\n")
            f.write("\n")

        # Extraction stats
        f.write(f"## Extraction Statistics\n\n")
        f.write(f"- **Text elements:** {len(result['text_elements'])}\n")
        f.write(f"- **Figures:** {len(result['figures'])}\n")
        f.write(f"- **Parser(s) used:** {result['extraction_method']}\n\n")

        # Text hierarchy
        f.write(f"## Hierarchical Text\n\n")

        for i, elem in enumerate(result['text_elements'], 1):
            # Show hierarchy with indentation
            indent = "  " * elem['depth']

            if elem['path_string']:
                f.write(f"{indent}### [{i}] {elem['path_string']}\n\n")
            else:
                f.write(f"{indent}### [{i}] (Root level)\n\n")

            f.write(f"{indent}{elem['text']}\n\n")
            f.write(f"{indent}*Depth: {elem['depth']}*\n\n")
            f.write("---\n\n")

        # Figures (if any)
        if result['figures']:
            f.write(f"## Figures\n\n")
            for i, fig in enumerate(result['figures'], 1):
                f.write(f"### Figure {i}\n\n")
                f.write(f"- **ID:** {fig.get('figure_id', 'N/A')}\n")
                f.write(f"- **Type:** {fig.get('figure_type', 'N/A')}\n")
                f.write(f"- **Page:** {fig.get('page', 'N/A')}\n")
                f.write(f"- **Caption:** {fig.get('caption', 'N/A')}\n\n")

    print(f"✓ Readable markdown: {md_file}")

    # 4. Save routing details (if routing mode)
    if 'page_routing' in result:
        routing_file = output_path / f"{base_name}_routing.json"
        with open(routing_file, 'w', encoding='utf-8') as f:
            json.dump(result['page_routing'], f, indent=2)
        print(f"✓ Routing details:  {routing_file}")

    # 5. Save summary
    summary_file = output_path / f"{base_name}_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"PDF Extraction Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Input file:     {pdf_file.name}\n")
        f.write(f"Mode:           {mode}\n")
        f.write(f"Text elements:  {len(result['text_elements'])}\n")
        f.write(f"Figures:        {len(result['figures'])}\n")
        f.write(f"Parser(s):      {result['extraction_method']}\n")

        if 'page_routing' in result:
            f.write(f"\nPage Routing:\n")
            f.write(f"  Total pages: {result['page_routing']['total_pages']}\n")
            f.write(f"  {result['page_routing']['summary']}\n")

    print(f"✓ Summary:          {summary_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("Extraction Summary")
    print(f"{'='*80}\n")
    print(f"Text elements:  {len(result['text_elements'])}")
    print(f"Figures:        {len(result['figures'])}")
    print(f"Parser(s):      {result['extraction_method']}")

    if 'page_routing' in result:
        print(f"\nPage Routing:")
        print(f"  {result['page_routing']['summary']}")

    print(f"\n✓ All results saved to: {output_path}/")
    print(f"\nRecommended files to inspect:")
    print(f"  1. {md_file.name} (human-readable)")
    print(f"  2. {summary_file.name} (quick overview)")
    print(f"  3. {json_file.name} (complete data)")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse a single PDF and save results for manual inspection"
    )
    parser.add_argument(
        "pdf_path",
        nargs='?',
        default="files/organized_pdfs/PMC1448691_his_2369.pdf",
        help="Path to PDF file"
    )
    parser.add_argument(
        "--mode",
        choices=["routing", "fallback"],
        default="routing",
        help="Parsing mode (default: routing)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output/)"
    )

    args = parser.parse_args()

    parse_and_save(args.pdf_path, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
