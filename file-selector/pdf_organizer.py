import shutil
from pathlib import Path
from typing import Tuple


# =================CONFIGURATION =================
SOURCE_DIR = "processed_corpus"         # Folder with extracted papers (one subfolder per PMCID)
PDF_OUTPUT_DIR = "organized_pdfs"       # Folder for all PDF files
XML_OUTPUT_DIR = "organized_xmls"       # Folder for all XML files
# ================================================


def organize_files(source_dir: Path, pdf_dir: Path, xml_dir: Path) -> Tuple[int, int]:
    """
    Organize PDF and XML files from paper directories into separate folders.

    Args:
        source_dir: Directory containing paper subdirectories
        pdf_dir: Destination directory for PDF files
        xml_dir: Destination directory for XML files

    Returns:
        Tuple of (pdf_count, xml_count) - number of files organized
    """
    # Create output directories
    pdf_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    pdf_count = 0
    xml_count = 0

    # Find all paper directories (subdirectories in source_dir)
    paper_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

    if not paper_dirs:
        print(f"No paper directories found in {source_dir}")
        return 0, 0

    print(f"Found {len(paper_dirs)} paper directories. Starting organization...")

    for paper_dir in sorted(paper_dirs):
        pmcid = paper_dir.name

        # Find all PDF files in this paper directory
        pdf_files = list(paper_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            # Create a unique name: PMCID_originalname.pdf
            # If the file is already named with PMCID, just use original name
            if pdf_file.stem.startswith(pmcid):
                new_name = pdf_file.name
            else:
                new_name = f"{pmcid}_{pdf_file.name}"

            dest_path = pdf_dir / new_name

            # Handle potential filename conflicts
            counter = 1
            while dest_path.exists():
                stem = dest_path.stem
                new_name = f"{stem}_{counter}.pdf"
                dest_path = pdf_dir / new_name
                counter += 1

            # Copy the file
            shutil.copy2(pdf_file, dest_path)
            pdf_count += 1

        # Find all XML files in this paper directory
        xml_files = list(paper_dir.glob("*.xml")) + list(paper_dir.glob("*.nxml"))
        for xml_file in xml_files:
            # Create a unique name: PMCID.xml or PMCID_2.xml
            if xml_file.stem.startswith(pmcid):
                new_name = xml_file.name
            else:
                # Use .nxml extension if original was .nxml, otherwise .xml
                ext = xml_file.suffix
                new_name = f"{pmcid}{ext}"

            dest_path = xml_dir / new_name

            # Handle potential filename conflicts
            counter = 1
            while dest_path.exists():
                ext = dest_path.suffix
                stem = dest_path.stem.rstrip('_0123456789')  # Remove existing counter
                new_name = f"{stem}_{counter}{ext}"
                dest_path = xml_dir / new_name
                counter += 1

            # Copy the file
            shutil.copy2(xml_file, dest_path)
            xml_count += 1

        if pdf_files or xml_files:
            print(f"Organized {pmcid}: {len(pdf_files)} PDFs, {len(xml_files)} XMLs")
        else:
            print(f"Skipped {pmcid}: No PDF or XML files found")

    return pdf_count, xml_count


def main():
    """Main organization process."""
    source_dir = Path(SOURCE_DIR)
    pdf_dir = Path(PDF_OUTPUT_DIR)
    xml_dir = Path(XML_OUTPUT_DIR)

    # Validate source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        print("Make sure you've run tarball_extractor.py first.")
        return

    print(f"Organizing files from: {source_dir}")
    print(f"PDF output: {pdf_dir}")
    print(f"XML output: {xml_dir}")
    print()

    try:
        pdf_count, xml_count = organize_files(source_dir, pdf_dir, xml_dir)

        print(f"\n{'='*50}")
        print("Organization complete!")
        print(f"Total PDFs organized: {pdf_count}")
        print(f"Total XMLs organized: {xml_count}")
        print(f"{'='*50}")

    except Exception as e:
        print(f"Error during organization: {e}")
        raise


if __name__ == "__main__":
    main()
