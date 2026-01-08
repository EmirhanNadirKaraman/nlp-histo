import sys
import argparse
from pathlib import Path
from datetime import datetime
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy
import scispacy
from scispacy.linking import EntityLinker

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db_connection, Document
from ner import run_ner_on_db

def process_document_worker(args):
    """
    Worker function for processing a single document's NER in a thread.

    Args:
        args: Tuple of (pmcid, index, total_count, nlp, min_chars, force, stats, stats_lock)

    Returns:
        Dict with processing result and metadata
    """
    pmcid, index, total_count, nlp, min_chars, force, stats, stats_lock = args
    thread_name = threading.current_thread().name

    print(f"[{index}/{total_count}] [{thread_name}] Processing {pmcid}...")

    try:
        # Call the existing run_ner_on_db function (already thread-safe)
        results = run_ner_on_db(
            pmcid,
            min_chars,
            save_to_db=True,
            force=force,
            nlp=nlp  # Shared model across threads (safe for inference)
        )

        if results:
            with stats_lock:
                stats['processed'] += 1
            return {
                'status': 'processed',
                'pmcid': pmcid,
                'entities': len(results)
            }
        else:
            with stats_lock:
                stats['skipped'] += 1
            return {
                'status': 'skipped',
                'pmcid': pmcid,
                'reason': 'no_entities_or_already_exists'
            }

    except Exception as e:
        with stats_lock:
            stats['errors'] += 1
        print(f"[{thread_name}] ✗ Error processing {pmcid}: {e}")
        return {
            'status': 'error',
            'pmcid': pmcid,
            'error': str(e)
        }

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

    # Thread-safe stats tracking
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    stats_lock = threading.Lock()

    # Auto-detect worker count: CPU cores / 2, minimum 1
    cpu_count = os.cpu_count() or 1
    max_workers = max(1, cpu_count // 2)

    print(f"\nUsing {max_workers} worker threads (CPU count: {cpu_count})")
    print("="*80)

    # Prepare work items
    work_items = [
        (pmcid, idx + 1, len(pmcids), nlp, min_chars, force, stats, stats_lock)
        for idx, pmcid in enumerate(pmcids)
    ]

    # Process with ThreadPoolExecutor
    results = []

    try:
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="NERWorker"
        ) as executor:

            # Submit all tasks
            future_to_pmcid = {
                executor.submit(process_document_worker, item): item[0]  # item[0] is pmcid
                for item in work_items
            }

            print(f"Submitted {len(future_to_pmcid)} NER tasks to thread pool\n")

            # Collect results as they complete
            for future in as_completed(future_to_pmcid):
                pmcid = future_to_pmcid[future]

                try:
                    result = future.result()
                    results.append(result)

                    # Status emoji
                    status_emoji = {
                        'processed': '✓',
                        'skipped': '⊘',
                        'error': '✗'
                    }.get(result['status'], '?')

                    # Progress logging
                    print(f"{status_emoji} {result['pmcid']:15s} - {result['status']:12s} ({len(results):3d}/{len(pmcids):3d} completed)")

                except Exception as e:
                    print(f"✗ Future exception for {pmcid}: {e}")
                    results.append({'status': 'error', 'pmcid': pmcid, 'error': str(e)})

    except KeyboardInterrupt:
        print("\n⚠️  Interrupt received. Waiting for active threads to finish...")
        print("Press Ctrl+C again to force quit")

    # Print detailed result breakdown
    if results:
        print("\n" + "="*80)
        print("Processing Results by Status")
        print("="*80)

        status_counts = {}
        for result in results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        for status in sorted(status_counts.keys()):
            count = status_counts[status]
            print(f"  {status:15s}: {count:4d}")

    print("\n" + "="*80)
    print(f"Summary: {stats['processed']} Processed | {stats['skipped']} Skipped | {stats['errors']} Errors")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process NER for all documents')
    parser.add_argument('--min-chars', type=int, default=50)
    parser.add_argument('--force', '-f', action='store_true')
    args = parser.parse_args()

    batch_process_all_documents(min_chars=args.min_chars, force=args.force)