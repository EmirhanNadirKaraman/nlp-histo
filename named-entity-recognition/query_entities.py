"""
Query script for analyzing NER entities stored in the database.

Usage:
    python named-entity-recognition/query_entities.py
    python named-entity-recognition/query_entities.py --pmcid PMC7149534
    python named-entity-recognition/query_entities.py --label PERSON
    python named-entity-recognition/query_entities.py --entity "COVID-19"
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import database module
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db_connection, Entity, TextElement, Document
from sqlalchemy import func


def query_entities(pmcid=None, label=None, entity_text=None):
    """
    Query entities from the database with optional filters.

    Args:
        pmcid: Filter by document PMCID
        label: Filter by entity label (e.g., PERSON, ORG, GPE)
        entity_text: Filter by entity text (partial match)
    """
    db = get_db_connection()

    with db.session_scope() as session:
        # Build query
        query = session.query(
            Entity.entity_text,
            Entity.entity_label,
            TextElement.path_string,
            Document.pmcid
        ).join(TextElement).join(Document)

        # Apply filters
        if pmcid:
            query = query.filter(Document.pmcid == pmcid)
        if label:
            query = query.filter(Entity.entity_label == label)
        if entity_text:
            query = query.filter(Entity.entity_text.ilike(f'%{entity_text}%'))

        # Execute query
        results = query.all()

        if not results:
            print("No entities found matching the criteria.")
            return

        print(f"\nFound {len(results)} entities:\n")
        for text, lbl, section, doc_pmcid in results[:50]:
            print(f"[{lbl}] {text} - {doc_pmcid}/{section}")

        if len(results) > 50:
            print(f"\n... and {len(results) - 50} more entities")


def entity_statistics(pmcid=None):
    """Print statistics about entities in the database."""
    db = get_db_connection()

    with db.session_scope() as session:
        # Build base query
        query = session.query(Entity)

        if pmcid:
            query = query.join(TextElement).join(Document).filter(Document.pmcid == pmcid)
            print(f"\n{'='*80}")
            print(f"Entity Statistics for {pmcid}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"Global Entity Statistics")
            print(f"{'='*80}\n")

        # Total count
        total = query.count()
        print(f"Total entities: {total}")

        # Count by label
        if pmcid:
            labels = (
                session.query(
                    Entity.entity_label,
                    func.count(Entity.id)
                )
                .join(TextElement)
                .join(Document)
                .filter(Document.pmcid == pmcid)
                .group_by(Entity.entity_label)
                .all()
            )
        else:
            labels = (
                session.query(
                    Entity.entity_label,
                    func.count(Entity.id)
                )
                .group_by(Entity.entity_label)
                .all()
            )

        print("\nEntities by label:")
        for label, count in sorted(labels, key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")

        # Most common entities
        if pmcid:
            common = (
                session.query(
                    Entity.entity_text,
                    Entity.entity_label,
                    func.count(Entity.id)
                )
                .join(TextElement)
                .join(Document)
                .filter(Document.pmcid == pmcid)
                .group_by(Entity.entity_text, Entity.entity_label)
                .order_by(func.count(Entity.id).desc())
                .limit(20)
                .all()
            )
        else:
            common = (
                session.query(
                    Entity.entity_text,
                    Entity.entity_label,
                    func.count(Entity.id)
                )
                .group_by(Entity.entity_text, Entity.entity_label)
                .order_by(func.count(Entity.id).desc())
                .limit(20)
                .all()
            )

        print("\nMost common entities:")
        for text, label, count in common:
            print(f"  [{label}] {text}: {count} occurrences")

        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Query NER entities from database')
    parser.add_argument('--pmcid', help='Filter by PMCID')
    parser.add_argument('--label', help='Filter by entity label')
    parser.add_argument('--entity', help='Search for entity text (partial match)')
    parser.add_argument('--stats', action='store_true', help='Show statistics')

    args = parser.parse_args()

    if args.stats:
        entity_statistics(pmcid=args.pmcid)
    else:
        query_entities(pmcid=args.pmcid, label=args.label, entity_text=args.entity)
