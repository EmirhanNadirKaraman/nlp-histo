import os
import sys
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db_connection, Entity
from sqlalchemy import or_

# 1. Configuration
SOURCE_DIR = "umls_entities/"  # Folder containing all JSONs
DEST_DIR = "relevant_texts/" # Where you want the filtered files to go

# The relevant TUIs for Diseases, Neoplasms, Mental Disorders, and Symptoms
# Mapping of TUI codes to their human-readable UMLS Semantic Type names
UMLS_DISEASE_TYPES = {
    'T047': 'Disease or Syndrome',
    'T191': 'Neoplastic Process',
    'T048': 'Mental or Behavioral Dysfunction',
    'T037': 'Injury or Poisoning',
    'T046': 'Pathologic Function',
    'T020': 'Congenital Abnormality',
    'T184': 'Sign or Symptom',
    'T033': 'Finding'
}

def copy_disease_files():
    # Create destination directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created directory: {DEST_DIR}")

    db = get_db_connection()
    RELEVANT_TUIS = list(UMLS_DISEASE_TYPES.keys())

    with db.session_scope() as session:
        # 2. Query the DB for unique CUIs that have the relevant semantic types
        # We use a filter that checks if the semantic_types column contains our target TUIs
        query_filters = [Entity.semantic_types.contains(tui) for tui in RELEVANT_TUIS]
        
        disease_entities = session.query(Entity.umls_cui).filter(
            or_(*query_filters)
        ).distinct().all()

        disease_cuis = {res[0] for res in disease_entities if res[0]}
        print(f"Found {len(disease_cuis)} unique disease CUIs in the database.")

        # 3. Iterate through the files and copy if the CUI matches
        copied_count = 0
        all_files = os.listdir(SOURCE_DIR)
        
        for filename in all_files:
            if filename.endswith(".txt"):
                # Extract CUI from filename (e.g., 'C0000737_Abdominal_Pain.json')
                cui = filename.split('_')[0]
                
                if cui in disease_cuis:
                    src_path = os.path.join(SOURCE_DIR, filename)
                    dst_path = os.path.join(DEST_DIR, filename)
                    
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"Error copying {filename}: {e}")

        print("\n--- Process Complete ---")
        print(f"Files scanned: {len(all_files)}")
        print(f"Files copied to '{DEST_DIR}': {copied_count}")

if __name__ == "__main__":
    copy_disease_files()