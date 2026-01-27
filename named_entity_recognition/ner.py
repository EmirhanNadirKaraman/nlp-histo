# query the database to get all texts in a document, filter out the ones with few character counts, and run NER from scispacy on it
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path so we can import database module
sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy
import scispacy  # noqa: F401 - needed to register scispacy components
from scispacy.linking import EntityLinker  # noqa: F401 - registers scispacy_linker factory
from sqlalchemy import func
from database import get_db_connection, TextElement, Document, Entity


def load_ner_model():
    """Load fast NER model without linker."""
    nlp = spacy.load("en_core_sci_lg", disable=["parser", "attribute_ruler", "lemmatizer"])
    return nlp


def load_linker_model():
    """Load model with UMLS linker for entity linking."""
    nlp = spacy.load("en_core_sci_lg", disable=["parser", "attribute_ruler", "lemmatizer"])
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "threshold": 0.7
    })
    return nlp


def link_entities_batch(entity_texts: list, linker_nlp, linker, entity_cache: dict) -> dict:
    """
    Link a batch of unique entity texts to UMLS concepts.

    Args:
        entity_texts: List of unique entity text strings to link
        linker_nlp: spaCy model with linker
        linker: The linker component for KB lookups
        entity_cache: Existing cache to check/update

    Returns:
        dict mapping entity_text -> list of (cui, score, canonical_name, semantic_types)
    """
    new_mappings = {}
    texts_to_link = [t for t in entity_texts if t not in entity_cache]

    if not texts_to_link:
        return new_mappings

    # Process uncached entities through linker
    for doc in linker_nlp.pipe(texts_to_link, batch_size=50):
        text = doc.text
        mappings = []

        # The doc itself is the entity since we're passing individual entity texts
        if doc.ents:
            for ent in doc.ents:
                if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                    for cui, score in ent._.kb_ents:
                        concept = linker.kb.cui_to_entity.get(cui)
                        if concept:
                            mappings.append({
                                "cui": cui,
                                "score": float(score),
                                "canonical_name": concept.canonical_name,
                                "semantic_types": list(concept.types) if concept.types else None
                            })

        # If no entities found, try treating whole text as entity
        if not mappings and hasattr(doc._, 'kb_ents') and doc._.kb_ents:
            for cui, score in doc._.kb_ents:
                concept = linker.kb.cui_to_entity.get(cui)
                if concept:
                    mappings.append({
                        "cui": cui,
                        "score": float(score),
                        "canonical_name": concept.canonical_name,
                        "semantic_types": list(concept.types) if concept.types else None
                    })

        new_mappings[text] = mappings

    return new_mappings


def run_ner_on_db(pmcid: str, min_chars: int = 50, save_to_db: bool = False, force: bool = False,
                  nlp=None, linker_nlp=None, entity_cache: dict = None):
    """
    Optimized NER with entity linking cache.

    1. Queries DB for text elements belonging to a PMCID.
    2. Filters out short snippets (noise).
    3. Runs FAST NER (no linker) to extract entities.
    4. Checks cache for already-linked entity texts.
    5. Only links NEW entity texts through UMLS linker.
    6. Updates cache with new mappings.
    7. Optionally saves entities to database.

    Args:
        pmcid: PubMed Central ID to process
        min_chars: Minimum character count for text filtering
        save_to_db: If True, saves entities to database. If False, just returns results.
        force: If True, reprocess even if entities already exist (deletes old entities)
        nlp: Fast NER model (no linker) - will load if None
        linker_nlp: Model with UMLS linker - will load if None
        entity_cache: Dict mapping entity_text -> UMLS mappings (shared across docs)

    Returns:
        List of dicts with section and entities information
    """
    # Initialize cache if not provided
    if entity_cache is None:
        entity_cache = {}

    # Step 0: Load models if not provided
    if nlp is None:
        print("Loading fast NER model...")
        nlp = load_ner_model()
        print("✓ Fast NER model loaded")

    if linker_nlp is None:
        print("Loading UMLS linker model...")
        linker_nlp = load_linker_model()
        print("✓ UMLS linker model loaded")

    # Get linker component for KB lookups
    linker = None
    linker_name = None
    if "scispacy_linker" in linker_nlp.pipe_names:
        linker = linker_nlp.get_pipe("scispacy_linker")
        linker_name = "umls"

    db = get_db_connection()

    with db.session_scope() as session:
        # Get model name early for skip check
        model_name = nlp.meta.get('name', 'unknown')

        # Check if entities from THIS MODEL already exist for this document
        if save_to_db and not force:
            existing_count = (
                session.query(func.count(Entity.id))
                .join(TextElement)
                .join(Document)
                .filter(Document.pmcid == pmcid)
                .filter(Entity.model_name == model_name)
                .scalar()
            )

            if existing_count > 0:
                print(f"⚠ Document {pmcid} already has {existing_count} entities from model '{model_name}'.")
                print("Skipping processing. Use --force to reprocess and replace.\n")
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

        # Step 2: FAST NER pass (no linker) - collect all entities
        all_entities = []  # List of (doc_idx, ent_text, ent_label, start_char, end_char)
        unique_entity_texts = set()

        print("Pass 1: Fast NER extraction...")
        for i, doc in tqdm(enumerate(nlp.pipe(texts, batch_size=50)),
                          total=len(texts),
                          desc="🔍 NER"):
            for ent in doc.ents:
                all_entities.append((i, ent.text, ent.label_, ent.start_char, ent.end_char))
                unique_entity_texts.add(ent.text)

        print(f"Found {len(all_entities)} entity mentions, {len(unique_entity_texts)} unique texts")

        # Step 3: Check cache and link only uncached entities
        uncached_texts = [t for t in unique_entity_texts if t not in entity_cache]
        cached_count = len(unique_entity_texts) - len(uncached_texts)

        print(f"Cache status: {cached_count} cached, {len(uncached_texts)} need linking")

        if uncached_texts and linker:
            print(f"Pass 2: Linking {len(uncached_texts)} new entity texts...")
            new_mappings = link_entities_batch(uncached_texts, linker_nlp, linker, entity_cache)
            entity_cache.update(new_mappings)
            print(f"✓ Linked {len(new_mappings)} new entities")

        # Step 4: Build results using cache
        extracted_data = []
        entities_to_save = []

        # Group entities by document index
        from collections import defaultdict
        doc_entities = defaultdict(list)
        for doc_idx, ent_text, ent_label, start_char, end_char in all_entities:
            doc_entities[doc_idx].append((ent_text, ent_label, start_char, end_char))

        for doc_idx, ents in doc_entities.items():
            section_info = {"section": metadata[doc_idx], "entities": []}

            for ent_text, ent_label, start_char, end_char in ents:
                # Get UMLS mappings from cache
                umls_mappings = entity_cache.get(ent_text, [])

                if umls_mappings:
                    # Create an entity record for each UMLS mapping
                    for mapping in umls_mappings:
                        if save_to_db:
                            entity_dict = {
                                "text_element_id": text_element_ids[doc_idx],
                                "entity_text": ent_text,
                                "entity_label": ent_label,
                                "umls_cui": mapping["cui"],
                                "umls_score": mapping["score"],
                                "canonical_name": mapping["canonical_name"],
                                "start_char": start_char,
                                "end_char": end_char,
                                "model_name": model_name,
                                "linker_name": linker_name,
                                "semantic_types": mapping["semantic_types"]
                            }
                            entities_to_save.append(entity_dict)

                        section_info["entities"].append((
                            ent_text, ent_label, mapping["cui"],
                            mapping["canonical_name"], mapping["semantic_types"]
                        ))
                else:
                    # No UMLS mapping found - still save the entity
                    if save_to_db:
                        entity_dict = {
                            "text_element_id": text_element_ids[doc_idx],
                            "entity_text": ent_text,
                            "entity_label": ent_label,
                            "umls_cui": None,
                            "umls_score": None,
                            "canonical_name": None,
                            "start_char": start_char,
                            "end_char": end_char,
                            "model_name": model_name,
                            "linker_name": linker_name,
                            "semantic_types": None
                        }
                        entities_to_save.append(entity_dict)

                    section_info["entities"].append((ent_text, ent_label, None, None, None))

            if section_info["entities"]:
                extracted_data.append(section_info)

        # Step 5: Save to database if requested
        if save_to_db and entities_to_save:
            try:
                if force:
                    print(f"Queueing deletion of old entities from model '{model_name}' for {pmcid}...")
                    session.query(Entity).filter(
                        Entity.text_element_id.in_(text_element_ids),
                        Entity.model_name == model_name
                    ).delete(synchronize_session=False)

                print(f"Saving {len(entities_to_save)} entities to database...")
                session.bulk_insert_mappings(Entity, entities_to_save)
                session.commit()

                # Clear session to free up memory for the next PMCID
                session.expunge_all()

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
            print("   Entities:")
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
