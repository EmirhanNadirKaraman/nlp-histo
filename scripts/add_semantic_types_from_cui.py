import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db_connection, Entity
from scispacy.linking import EntityLinker

def update_entities_with_semantic_types():
    # 1. Load the UMLS Linker to access the knowledge base
    print("Loading UMLS Knowledge Base...")
    linker = EntityLinker(linker_name="umls")
    kb = linker.kb 

    db = get_db_connection()

    with db.session_scope() as session:
        # 2. Fetch all entities that have a CUI but no semantic types yet
        entities = session.query(Entity).filter(
            Entity.umls_cui.isnot(None),
            Entity.semantic_types.is_(None)
        ).all()

        print(f"Found {len(entities)} rows to update.")

        count = 0
        for ent in entities:
            # 3. Look up the CUI in the Scispacy KB
            concept = kb.cui_to_entity.get(ent.umls_cui)
            
            if concept and concept.types:
                # Store semantic types as a comma-separated string (e.g., 'T047, T191')
                ent.semantic_types = ", ".join(concept.types)
                count += 1
            
            # Progress marker
            if count % 500 == 0 and count > 0:
                print(f"Updated {count} entities...")

        # 4. Commit changes to the database
        session.commit()
        print(f"✓ Successfully updated {count} entities with semantic types.")

if __name__ == "__main__":
    update_entities_with_semantic_types()