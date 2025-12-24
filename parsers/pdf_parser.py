"""
PDF Text Parser

Extracts text from PDF files with basic structure preservation.
Used as a fallback when XML parsing fails or returns no content.
"""

from pathlib import Path
from typing import List, Dict, Optional
import re


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from a PDF file with basic hierarchical structure.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries with text elements and their structure:
        {
            'path_list': list,      # ['PDF', 'Page 1'] or ['PDF', 'Section Title']
            'path_string': str,     # "PDF > Page 1"
            'depth': int,           # 1 or 2
            'text': str             # The text content
        }
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. Install with: pip install PyMuPDF"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_label = f"Page {page_num + 1}"

        # Extract text from page
        text = page.get_text()

        if text.strip():
            # Try to detect sections by looking for common headers
            sections = split_into_sections(text)

            if sections:
                # Found sections - create hierarchical structure
                for section_title, section_text in sections:
                    if section_text.strip():
                        path_list = ['PDF', page_label, section_title]
                        data.append({
                            'path_list': path_list,
                            'path_string': ' > '.join(path_list),
                            'depth': len(path_list),
                            'text': section_text.strip()
                        })
            else:
                # No sections detected - store entire page as one block
                path_list = ['PDF', page_label]
                data.append({
                    'path_list': path_list,
                    'path_string': ' > '.join(path_list),
                    'depth': len(path_list),
                    'text': text.strip()
                })

    doc.close()
    return data


def split_into_sections(text: str) -> List[tuple]:
    """
    Attempt to split text into sections based on common section headers.

    Args:
        text: The text to split

    Returns:
        List of (section_title, section_text) tuples
    """
    # Common section headers in scientific papers
    section_patterns = [
        r'\n\s*(Abstract|ABSTRACT)\s*\n',
        r'\n\s*(Introduction|INTRODUCTION)\s*\n',
        r'\n\s*(Methods|METHODS|Materials and Methods|MATERIALS AND METHODS)\s*\n',
        r'\n\s*(Results|RESULTS)\s*\n',
        r'\n\s*(Discussion|DISCUSSION)\s*\n',
        r'\n\s*(Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\s*\n',
        r'\n\s*(References|REFERENCES)\s*\n',
        r'\n\s*(\d+\.?\s+[A-Z][a-z]+.*?)\s*\n',  # Numbered sections like "1. Introduction"
    ]

    # Try to find section boundaries
    sections = []
    combined_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)

    matches = list(re.finditer(combined_pattern, text))

    if matches:
        for i, match in enumerate(matches):
            section_title = match.group(0).strip()

            # Get text between this match and the next one
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start_pos:end_pos]

            sections.append((section_title, section_text))

    return sections


def extract_text_blocks_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from PDF using block detection (more sophisticated).

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of text blocks with structure
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. Install with: pip install PyMuPDF"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_label = f"Page {page_num + 1}"

        # Get text blocks with position information
        blocks = page.get_text("blocks")

        for block_num, block in enumerate(blocks):
            # block format: (x0, y0, x1, y1, "text", block_no, block_type)
            if len(block) >= 5:
                text = block[4]

                if text.strip():
                    # Detect if this might be a header (larger font, position, etc.)
                    is_header = is_likely_header(text)

                    if is_header:
                        section_title = text.strip()
                        path_list = ['PDF', page_label, section_title]
                    else:
                        path_list = ['PDF', page_label, f'Block {block_num + 1}']

                    data.append({
                        'path_list': path_list,
                        'path_string': ' > '.join(path_list),
                        'depth': len(path_list),
                        'text': text.strip()
                    })

    doc.close()
    return data


def is_likely_header(text: str) -> bool:
    """
    Heuristic to determine if text is likely a section header.

    Args:
        text: The text to check

    Returns:
        True if likely a header
    """
    text = text.strip()

    # Headers are usually:
    # - Short (less than 100 chars)
    # - All caps or title case
    # - May start with a number
    # - Don't end with punctuation (except :)

    if len(text) > 100:
        return False

    # Check for numbered sections
    if re.match(r'^\d+\.?\s+[A-Z]', text):
        return True

    # Check for all caps
    if text.isupper() and len(text) > 3:
        return True

    # Check for title case common headers
    common_headers = [
        'abstract', 'introduction', 'methods', 'results', 'discussion',
        'conclusion', 'references', 'background', 'materials', 'findings'
    ]
    if any(header in text.lower() for header in common_headers) and len(text) < 50:
        return True

    return False


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = "files/organized_pdfs/PMC7147438_978-0-387-72430-0_Chapter_4.pdf"

    if Path(pdf_file).exists():
        print(f"Parsing PDF: {pdf_file}\n")

        # Try simple extraction first
        data = extract_text_from_pdf(pdf_file)
        print(f"Extracted {len(data)} text elements\n")

        for i, elem in enumerate(data[:5], 1):
            print(f"Element {i}:")
            print(f"  Path: {elem['path_string']}")
            print(f"  Depth: {elem['depth']}")
            print(f"  Text: {elem['text'][:100]}...")
            print()

        if len(data) > 5:
            print(f"... and {len(data) - 5} more elements")
    else:
        print(f"File not found: {pdf_file}")
        print("Please provide a valid PDF file path")
