from lxml import etree
import pandas as pd
import os

def get_section_header(sec_element):
    """
    Helper to extract 'Label + Title' from a section.
    Example: converts <label>2.1</label><title>Methods</title> -> "2.1 Methods"
    """
    parts = []
    
    # Get the number (e.g., "2.1")
    label = sec_element.find('./label')
    if label is not None and label.text:
        parts.append(label.text.strip())
        
    # Get the name (e.g., "Staining Procedures")
    title = sec_element.find('./title')
    if title is not None and title.text:
        parts.append(title.text.strip())
        
    return " ".join(parts) if parts else "Unnamed Section"

def parse_recursive(element, current_path, collected_data):
    """
    Recursively walks through XML tags.
    If it finds a <sec>, it adds to the path.
    If it finds a <p>, it records the text with the current path.
    Skips sections that contain "References" in their header.
    """
    # 1. If we are at a Section, update the path
    if element.tag == 'sec':
        header = get_section_header(element)

        # Skip References sections entirely
        if 'reference' in header.lower():
            return

        # Extend the path (e.g., ["Methods"] -> ["Methods", "2.1 Staining"])
        current_path = current_path + [header]

    # 2. If we find a Paragraph, save it with the current path
    if element.tag == 'p':
        # Join the text (handling mixed content like bold/italic tags inside p)
        text_content = "".join(element.itertext()).strip()
        
        if text_content: # Skip empty paragraphs
            collected_data.append({
                "path_list": current_path,            # List format: ['Methods', '2.1 Staining']
                "path_string": " > ".join(current_path), # String format: "Methods > 2.1 Staining"
                "depth": len(current_path),           # How deep in the tree (1, 2, 3...)
                "text": text_content
            })

    # 3. Recursively process all children (subsections, paragraphs, etc.)
    for child in element:
        # We only recurse into 'sec' or stay in current scope for 'p'
        # But we need to traverse everything to find nested 'p' or 'sec'
        if isinstance(child.tag, str): # Skip comments/processing instructions
            parse_recursive(child, current_path, collected_data)

def extract_hierarchy(nxml_path):
    """
    Main function to load file and start recursion.
    Handles both regular articles and book chapters.
    """
    tree = etree.parse(nxml_path)
    root = tree.getroot()

    data = []

    # Try to find the body tag (standard articles)
    body = root.find('./body')

    if body is not None:
        # Standard article with <body> tag
        parse_recursive(body, [], data)
    else:
        # No body tag - might be a book chapter or different structure
        # Try to find sections directly under article root
        article_sections = root.findall('./sec')

        if article_sections:
            # Book chapter with sections directly under <article>
            for sec in article_sections:
                parse_recursive(sec, [], data)
        else:
            # No direct sections - try to find in back matter
            back = root.find('./back')
            if back is not None:
                back_sections = back.findall('.//sec')
                for sec in back_sections:
                    parse_recursive(sec, [], data)

        # Also extract from abstract if available and no body content found
        if not data:
            abstract = root.find('.//abstract')
            if abstract is not None:
                # Mark abstract paragraphs with a special path
                parse_recursive(abstract, ['Abstract'], data)

    return data

# --- Batch Processing for All XML Files ---

def process_all_xml_files(input_dir, output_dir):
    """
    Process all XML files in a directory and save each to a separate CSV.

    Args:
        input_dir: Directory containing .xml and .nxml files
        output_dir: Directory where CSV files will be saved
    """
    from pathlib import Path

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all XML files (both .xml and .nxml)
    xml_files = list(input_path.glob("*.xml")) + list(input_path.glob("*.nxml"))

    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return

    print(f"Found {len(xml_files)} XML files. Starting processing...\n")

    success_count = 0
    error_count = 0

    for xml_file in sorted(xml_files):
        try:
            # Extract the PMCID from filename (e.g., PMC1448691.nxml -> PMC1448691)
            pmcid = xml_file.stem

            print(f"Processing {pmcid}...", end=" ")

            # Parse the XML file
            structured_text = extract_hierarchy(str(xml_file))

            if not structured_text:
                print(f"⚠️  No data extracted (empty body or no paragraphs)")
                continue

            # Create DataFrame
            df = pd.DataFrame(structured_text)

            # Save to CSV with PMCID as filename
            output_file = output_path / f"{pmcid}_parsed.csv"
            df.to_csv(output_file, index=False)

            print(f"✅ Saved {len(df)} rows to {output_file.name}")
            success_count += 1

        except Exception as e:
            print(f"❌ Error: {e}")
            error_count += 1

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"{'='*60}")


# --- Main Execution ---

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "files/organized_xmls"      # Directory with XML files
    OUTPUT_DIR = "files/output/hierarchical_parser/parsed_xml_outputs"       # Directory for CSV outputs

    # Process all XML files
    process_all_xml_files(INPUT_DIR, OUTPUT_DIR)