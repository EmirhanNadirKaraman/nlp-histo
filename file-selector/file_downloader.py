import requests
import xml.etree.ElementTree as ET
import os
import time
from pathlib import Path
from typing import List


def load_pmc_ids(file_path: str = "target_pmc_ids.txt") -> List[str]:
    """
    Load PMC IDs from a text file.

    Args:
        file_path: Path to the file containing PMC IDs (one per line)

    Returns:
        List of PMC IDs

    Raises:
        FileNotFoundError: If the input file doesn't exist
    """
    input_file = Path(file_path)

    if not input_file.exists():
        raise FileNotFoundError(f"PMC ID file not found: {file_path}")

    with open(input_file, 'r') as f:
        # Read lines, strip whitespace, and filter out empty lines
        pmc_ids = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(pmc_ids)} PMC IDs from {file_path}")
    return pmc_ids


# 1. Load Target PMCIDs from file
pmc_ids = load_pmc_ids("target_pmc_ids.txt")

# 2. Setup
base_api_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
output_dir = "histopathology_papers"
os.makedirs(output_dir, exist_ok=True)

def get_download_link(pmcid):
    """Asks NCBI for the FTP link to the tarball."""
    try:
        # Query the API
        response = requests.get(f"{base_api_url}?id={pmcid}")
        
        # Parse the XML response
        root = ET.fromstring(response.content)
        
        # Look for the 'link' tag with format='tgz' (The Article Package)
        for link in root.findall(".//link"):
            if link.attrib.get('format') == 'tgz':
                return link.attrib.get('href')
                
        return None  # No OA package found
    except Exception as e:
        print(f"Error querying {pmcid}: {e}")
        return None

def download_file(url, filename):
    """Downloads the file from the FTP link."""
    try:
        # Stream download to handle large files
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

# 3. Main Loop
print(f"Processing {len(pmc_ids)} papers...")

for pmcid in pmc_ids:
    print(f"Looking up {pmcid}...")
    
    ftp_link = get_download_link(pmcid)
    
    if ftp_link:
        # The link usually looks like: ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/.../PMCxxxx.tar.gz
        # We need to change 'ftp://' to 'https://' for requests to handle it easily, 
        # or just use urllib. standard requests doesn't support FTP by default, 
        # but NCBI links often redirect to HTTPS. Let's fix the scheme for stability.
        if ftp_link.startswith("ftp://"):
            ftp_link = ftp_link.replace("ftp://", "https://")
            
        file_name = os.path.join(output_dir, f"{pmcid}.tar.gz")
        download_file(ftp_link, file_name)
    else:
        print(f"⚠️ {pmcid} is not in the Open Access Subset (No Tarball available).")
    
    # Be polite to the server
    time.sleep(0.5)

print("\nDone! Check the folder.")