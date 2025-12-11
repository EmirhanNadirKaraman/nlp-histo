import tarfile
import shutil
from pathlib import Path


# =================CONFIGURATION =================
SOURCE_DIR = "histopathology_papers"   # Folder where you downloaded the .tar.gz files
DEST_DIR = "processed_corpus"          # Folder where you want the extracted files
# ================================================


def extract_tarball(tar_path: Path, dest_dir: Path) -> bool:
    """
    Extract XML and PDF files from a tarball.

    Args:
        tar_path: Path to the .tar.gz file
        dest_dir: Destination directory for extraction

    Returns:
        True if extraction was successful, False otherwise
    """
    # Get the PMCID from the filename (e.g., 'PMC10047158')
    pmcid = tar_path.stem.replace(".tar", "")

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Look for XML and PDF files inside the tarball
            # We filter the list to avoid extracting images (jpg/png) or other junk
            targets = [
                m for m in tar.getmembers()
                if m.name.endswith(".nxml") or m.name.endswith(".pdf")
            ]

            if not targets:
                print(f"⚠️  Skipping {pmcid}: No XML or PDF found inside.")
                return False

            # Create a folder for this paper to avoid filename collisions
            # (Many papers name their file 'article.pdf' or 'main.pdf')
            paper_dir = dest_dir / pmcid
            paper_dir.mkdir(parents=True, exist_ok=True)

            # Extract to a temporary directory first to handle nested structures
            temp_extract_dir = paper_dir / "_temp_extract"
            temp_extract_dir.mkdir(exist_ok=True)

            # Extract only the target files
            tar.extractall(path=temp_extract_dir, members=targets)

            # Move files to the root of the paper directory and rename
            xml_count = 0
            pdf_count = 0

            for member in targets:
                original_path = temp_extract_dir / member.name

                if not original_path.exists():
                    continue

                if member.name.endswith(".nxml"):
                    xml_count += 1
                    # Rename to pmcid.nxml for the first XML, or pmcid_2.nxml for additional ones
                    if xml_count == 1:
                        new_name = f"{pmcid}.nxml"
                    else:
                        new_name = f"{pmcid}_{xml_count}.nxml"
                    new_path = paper_dir / new_name
                    shutil.move(str(original_path), str(new_path))

                elif member.name.endswith(".pdf"):
                    pdf_count += 1
                    # Keep original PDF name but move to root
                    pdf_name = Path(member.name).name
                    new_path = paper_dir / pdf_name
                    shutil.move(str(original_path), str(new_path))

            # Clean up temporary extraction directory
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

            print(f"✅ Extracted: {pmcid} ({xml_count} XML, {pdf_count} PDF)")
            return True

    except Exception as e:
        print(f"❌ Error processing {pmcid}: {e}")
        return False


def main():
    """Main extraction process."""
    source_dir = Path(SOURCE_DIR)
    dest_dir = Path(DEST_DIR)

    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Find all tarballs
    tar_files = sorted(source_dir.glob("*.tar.gz"))

    if not tar_files:
        print(f"No .tar.gz files found in {SOURCE_DIR}")
        return

    print(f"Found {len(tar_files)} tarballs. Starting extraction...")

    success_count = 0
    for tar_path in tar_files:
        if extract_tarball(tar_path, dest_dir):
            success_count += 1

    print(f"\nExtraction finished: {success_count}/{len(tar_files)} successful.")


if __name__ == "__main__":
    main()