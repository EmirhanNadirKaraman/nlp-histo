# query the database to get all texts in a document, filter out the ones with few character counts, and run NER from scispacy on it
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path so we can import database module
sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
from sqlalchemy import func
from database import get_db_connection, TextElement, Document, Entity
import scispacy
from scispacy.linking import EntityLinker

def run_ner_on_db(pmcid: str, min_chars: int = 50, save_to_db: bool = False, force: bool = False, nlp=None):
    """
    1. Queries DB for text elements belonging to a PMCID.
    2. Filters out short snippets (noise).
    3. Runs NER in optimized batches.
    4. Optionally saves entities to database.

    Args:
        pmcid: PubMed Central ID to process
        min_chars: Minimum character count for text filtering
        save_to_db: If True, saves entities to database. If False, just returns results.
        force: If True, reprocess even if entities already exist (deletes old entities)

    Returns:
        List of dicts with section and entities information
    """

    # Load spaCy model (try scispacy first, fall back to standard English)
    # Speed Optimization: Disable components you don't need
    # 'parser' and 'lemmatizer' take up ~60% of processing time

    linker = None
    linker_name = None

    # Step 0: Load or Reuse the NLP Model
    if nlp is None:
        try:
            nlp = spacy.load("en_core_sci_sm", disable=["parser", "attribute_ruler", "lemmatizer"])
            nlp.add_pipe("scispacy_linker", config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "threshold": 0.7 
            })
            print("Using scispacy model: en_core_sci_sm with UMLS Linker")

        except Exception as e:
            # Fallback logic as you had it...
            print(e)

    # Retrieve the linker if it exists in the pipeline
    if "scispacy_linker" in nlp.pipe_names:
        linker = nlp.get_pipe("scispacy_linker")
        linker_name = "umls"
        print("✓ UMLS Linker initialized successfully.")
        

    db = get_db_connection()

    with db.session_scope() as session:
        # Check if entities already exist for this document
        if save_to_db and not force:
            existing_count = (
                session.query(func.count(Entity.id))
                .join(TextElement)
                .join(Document)
                .filter(Document.pmcid == pmcid)
                .scalar()
            )

            if existing_count > 0:
                print(f"⚠ Document {pmcid} already has {existing_count} entities in database.")
                print(f"Skipping processing. Use --force to reprocess and replace.\n")
                return []

        # Step 1: SQL-level filtering (much faster than Python filtering)
        results = (
            session.query(
                TextElement.id,
                TextElement.text_content,
                TextElement.path_string
            )
            .join(Document)
            .filter(Document.pmcid == pmcid)
            .filter(func.length(TextElement.text_content) >= min_chars)
            .all()
        )

        if not results:
            print(f"No text found for {pmcid} over {min_chars} chars.")
            return []

        text_element_ids = [r[0] for r in results]
        texts = [r[1] for r in results]
        metadata = [r[2] for r in results]

        print(f"Processing {len(texts)} text blocks for {pmcid}...")

        # Get model name for metadata
        model_name = nlp.meta.get('name', 'unknown')

        # Step 2: High-speed batch processing
        # nlp.pipe is significantly faster than calling nlp(text) in a loop
        extracted_data = []
        entities_to_save = []

        # Using smaller batch_size (10) because the Linker is RAM-intensive
        batch_size = 10 if linker else 20

        for i, doc in tqdm(enumerate(nlp.pipe(texts, batch_size=batch_size)), 
                           total=len(texts), 
                           desc="🔍 Running NER"):
            if doc.ents:
                section_info = {"section": metadata[i], "entities": []}

                for ent in doc.ents:
                    umls_cui = None
                    umls_score = None
                    canonical = None
                    semantic_types = None

                    # Extract UMLS Linking Info if available
                    if linker and hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                        # Get top match: (CUI, Score)

                        for umls_cui, umls_score in ent._.kb_ents:
                            # Look up the human-readable concept
                            try:
                                concept = linker.kb.cui_to_entity.get(umls_cui)
                                if concept:
                                    canonical = concept.canonical_name
                                    semantic_types = concept.types
                            except Exception as e:
                                # Skip canonical name if lookup fails
                                print(e)

                            # Prepare entities for database if save_to_db is True
                            if save_to_db:
                                entities_to_save.append(Entity(
                                    text_element_id=text_element_ids[i],
                                    entity_text=ent.text,
                                    entity_label=ent.label_,
                                    umls_cui=umls_cui,
                                    umls_score=umls_score,
                                    canonical_name=canonical,
                                    start_char=ent.start_char,
                                    end_char=ent.end_char,
                                    model_name=model_name,
                                    linker_name=linker_name,
                                    semantic_types=semantic_types
                                ))
                        # Collect for display
                        section_info["entities"].append((ent.text, ent.label_, umls_cui, canonical, semantic_types))

                extracted_data.append(section_info)

        # Step 3: Save to database if requested
        if save_to_db and entities_to_save:
            try: 
                if force:
                    print(f"Queueing deletion of old entities for {pmcid}...")
                    session.query(Entity).filter(
                        Entity.text_element_id.in_(text_element_ids)
                    ).delete(synchronize_session=False)

                print(f"Saving {len(entities_to_save)} entities to database...")
                # session.add_all(entities_to_save)
                session.bulk_save_objects(entities_to_save)
                session.commit()
                print(f"✓ Saved {len(entities_to_save)} entities successfully!")
                
            except Exception as e:
                session.rollback()
                print(f"❌ Error during DB transaction: {e}")
                raise e            

        return extracted_data

if __name__ == "__main__":
    # Parse command line arguments
    pmcid = sys.argv[1] if len(sys.argv) > 1 else "PMC7149534"
    min_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    save_to_db = "--save" in sys.argv or "-s" in sys.argv
    force = "--force" in sys.argv or "-f" in sys.argv

    print(f"Running NER on {pmcid} with min_chars={min_chars}...")
    if save_to_db:
        print("Will save results to database.\n")
    else:
        print("Running in preview mode (use --save to store in database).\n")

    ner_results = run_ner_on_db(pmcid, min_chars, save_to_db=save_to_db, force=force)

    if ner_results:
        # Display results
        print(f"\nFound {len(ner_results)} text blocks with entities\n")
        for i, item in enumerate(ner_results[:5], 1):
            print(f"{i}. Section: {item['section']}")
            print(f"   Entities:")
            for ent_data in item['entities'][:10]:  # Show first 10 entities per section
                text, label, cui, canonical, semantic_types = ent_data
                if cui and canonical:
                    print(f"     - {text} ({label}) -> {canonical} [CUI: {cui}]: {semantic_types}")
                else:
                    print(f"     - {text} ({label})")
            if len(item['entities']) > 10:
                print(f"     ... and {len(item['entities']) - 10} more entities")
            print()

        if len(ner_results) > 5:
            print(f"... and {len(ner_results) - 5} more sections with entities")

        print("\n" + "="*80)
        if save_to_db:
            print("✓ NER processing complete! Results saved to database.")
        else:
            print("Preview complete. To save to database, run with --save flag:")
            print(f"  python named-entity-recognition/ner.py {pmcid} {min_chars} --save")
        print("="*80)