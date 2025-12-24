"""
Figure Parser for JATS XML Files

Extracts figure information including captions, IDs, labels, and graphic references.
"""

from lxml import etree
from pathlib import Path
from typing import List, Dict, Optional


def extract_figures(xml_path: str) -> List[Dict]:
    """
    Extract all figures from a JATS XML file.

    Args:
        xml_path: Path to the XML file

    Returns:
        List of dictionaries containing figure information:
        {
            'figure_id': str,       # e.g., "fig1", "fig2a"
            'figure_label': str,    # e.g., "Figure 1", "Fig 2A"
            'caption_text': str,    # Full caption text
            'graphic_ref': str,     # Original image filename from xlink:href
            'section_context': str  # Section where figure appears
        }
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    figures = []

    # Find all <fig> elements in the document
    fig_elements = root.findall('.//fig')

    for fig in fig_elements:
        figure_data = {}

        # Extract figure ID (from id attribute)
        fig_id = fig.get('id')
        figure_data['figure_id'] = fig_id if fig_id else None

        # Extract label (e.g., "Figure 1", "Fig 2A")
        label_elem = fig.find('./label')
        if label_elem is not None and label_elem.text:
            figure_data['figure_label'] = label_elem.text.strip()
        else:
            figure_data['figure_label'] = None

        # Extract caption text
        caption_elem = fig.find('./caption')
        if caption_elem is not None:
            # Join all text from caption (including nested elements)
            caption_text = ''.join(caption_elem.itertext()).strip()
            figure_data['caption_text'] = caption_text
        else:
            figure_data['caption_text'] = None

        # Extract graphic reference (image filename)
        # JATS uses <graphic> element with xlink:href attribute
        graphic_elem = fig.find('./graphic')
        if graphic_elem is not None:
            # Handle namespaced attribute
            xlink_href = graphic_elem.get('{http://www.w3.org/1999/xlink}href')
            if not xlink_href:
                # Try without namespace
                xlink_href = graphic_elem.get('href')
            figure_data['graphic_ref'] = xlink_href
        else:
            figure_data['graphic_ref'] = None

        # Try to determine section context
        # Walk up the tree to find containing section
        section_context = get_section_context(fig)
        figure_data['section_context'] = section_context

        figures.append(figure_data)

    return figures


def get_section_context(element) -> Optional[str]:
    """
    Get the section context for an element by walking up the tree.

    Args:
        element: XML element

    Returns:
        Section title if found, None otherwise
    """
    current = element.getparent()

    while current is not None:
        if current.tag == 'sec':
            # Try to get section title
            title_elem = current.find('./title')
            if title_elem is not None and title_elem.text:
                return title_elem.text.strip()

            # Try to get label
            label_elem = current.find('./label')
            if label_elem is not None and label_elem.text:
                return label_elem.text.strip()

        # Check if we've hit the body or article root
        if current.tag in ['body', 'article']:
            break

        current = current.getparent()

    return None


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        xml_file = "../../files/organized_xmls/PMC1448691.nxml"

    if Path(xml_file).exists():
        figures = extract_figures(xml_file)
        print(f"Found {len(figures)} figures:\n")

        for i, fig in enumerate(figures, 1):
            print(f"Figure {i}:")
            print(f"  ID: {fig['figure_id']}")
            print(f"  Label: {fig['figure_label']}")
            print(f"  Graphic: {fig['graphic_ref']}")
            print(f"  Section: {fig['section_context']}")
            print(f"  Caption: {fig['caption_text'][:100] if fig['caption_text'] else 'None'}...")
            print()
    else:
        print(f"File not found: {xml_file}")
