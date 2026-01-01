import sys
import argparse
from pathlib import Path
from datetime import datetime
import spacy
import scispacy
from scispacy.linking import EntityLinker

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db_connection, Document
from ner import run_ner_on_db

def batch_process_all_documents(min_chars: int = 50, force: bool = False):
    db = get_db_connection()

    # 1. PRE-LOAD MODEL (Optimization: Load once for the whole batch)
    print("Loading spaCy model and UMLS Linker into RAM (one-time cost)...")
    nlp = spacy.load("en_core_sci_sm", disable=["parser", "attribute_ruler", "lemmatizer"])
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, 
        "linker_name": "umls", 
        "threshold": 0.7
    })

    with db.session_scope() as session:
        documents = session.query(Document.pmcid).all()
        pmcids = [doc.pmcid for doc in documents]

    if not pmcids:
        print("No documents found in database.")
        return

    print("="*80)
    print(f"Batch NER Processing Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    processed, skipped, errors = 0, 0, 0

    for i, pmcid in enumerate(pmcids, 1):
        print(f"\n[{i}/{len(pmcids)}] Processing {pmcid}...")
        
        try:
            # 2. PASS NLP OBJECT (Ensures ner.py uses the existing model)
            results = run_ner_on_db(
                pmcid, 
                min_chars, 
                save_to_db=True, 
                force=force, 
                nlp=nlp
            )

            if results:
                processed += 1
            else:
                skipped += 1

        except Exception as e:
            errors += 1
            print(f"✗ Error processing {pmcid}: {e}")

    print("\n" + "="*80)
    print(f"Summary: {processed} Processed | {skipped} Skipped | {errors} Errors")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process NER for all documents')
    parser.add_argument('--min-chars', type=int, default=50)
    parser.add_argument('--force', '-f', action='store_true')
    args = parser.parse_args()

    batch_process_all_documents(min_chars=args.min_chars, force=args.force)