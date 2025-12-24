"""
Query Examples for NLP Histopathology Database

This script demonstrates various ways to query the database.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db_connection, close_db_connection, Document, TextElement
from sqlalchemy import func


def example_1_get_all_documents():
    """Example 1: Get all documents in the database."""
    print("\n" + "="*60)
    print("Example 1: Get All Documents")
    print("="*60)

    db = get_db_connection()

    with db.session_scope() as session:
        documents = session.query(Document).all()

        print(f"\nFound {len(documents)} documents:\n")

        for doc in documents[:10]:  # Show first 10
            print(f"PMCID: {doc.pmcid}")
            print(f"  Title: {doc.title[:80] if doc.title else 'N/A'}...")
            print(f"  Journal: {doc.journal}")
            print(f"  Year: {doc.publication_year}")
            print()


def example_2_get_document_by_pmcid(pmcid="PMC1448691"):
    """Example 2: Get a specific document by PMCID."""
    print("\n" + "="*60)
    print(f"Example 2: Get Document by PMCID ({pmcid})")
    print("="*60)

    db = get_db_connection()

    with db.session_scope() as session:
        doc = session.query(Document).filter_by(pmcid=pmcid).first()

        if doc:
            print(f"\nFound document:")
            print(f"  PMCID: {doc.pmcid}")
            print(f"  Title: {doc.title}")
            print(f"  Journal: {doc.journal}")
            print(f"  Year: {doc.publication_year}")
            print(f"  File: {doc.filename}")

            # Count text elements
            count = session.query(TextElement)\
                .filter_by(document_id=doc.id)\
                .count()
            print(f"  Text Elements: {count}")
        else:
            print(f"\nDocument {pmcid} not found in database.")


def example_3_get_text_elements(pmcid="PMC1448691", limit=5):
    """Example 3: Get text elements from a specific document."""
    print("\n" + "="*60)
    print(f"Example 3: Get Text Elements from {pmcid}")
    print("="*60)

    db = get_db_connection()

    with db.session_scope() as session:
        doc = session.query(Document).filter_by(pmcid=pmcid).first()

        if not doc:
            print(f"\nDocument {pmcid} not found.")
            return

        elements = session.query(TextElement)\
            .filter_by(document_id=doc.id)\
            .limit(limit)\
            .all()

        print(f"\nShowing {len(elements)} text elements:\n")

        for elem in elements:
            print(f"Path: {elem.path_string}")
            print(f"  Unique Path: {elem.unique_path}")
            print(f"  Depth: {elem.depth}")
            print(f"  Position: {elem.position_in_section}")
            print(f"  Text: {elem.text_content[:100]}...")
            print(f"  Words: {elem.word_count}, Chars: {elem.char_count}")
            print()


def example_4_search_by_path(search_term="Methods"):
    """Example 4: Search for text elements in specific sections."""
    print("\n" + "="*60)
    print(f"Example 4: Search for '{search_term}' in Path")
    print("="*60)

    db = get_db_connection()

    with db.session_scope() as session:
        elements = session.query(TextElement)\
            .filter(TextElement.path_string.like(f'%{search_term}%'))\
            .limit(10)\
            .all()

        print(f"\nFound {len(elements)} text elements with '{search_term}' in path:\n")

        for elem in elements:
            print(f"Path: {elem.path_string}")
            print(f"  Document: {elem.document.pmcid}")
            print(f"  Text: {elem.text_content[:80]}...")
            print()


def example_5_get_by_depth(depth=2):
    """Example 5: Get all text elements at a specific depth."""
    print("\n" + "="*60)
    print(f"Example 5: Get Text Elements at Depth {depth}")
    print("="*60)

    db = get_db_connection()

    with db.session_scope() as session:
        elements = session.query(TextElement)\
            .filter_by(depth=depth)\
            .limit(10)\
            .all()

        print(f"\nFound text elements at depth {depth}:\n")

        for elem in elements:
            print(f"Path: {elem.path_string}")
            print(f"  Document: {elem.document.pmcid}")
            print(f"  Text: {elem.text_content[:80]}...")
            print()


def example_6_statistics():
    """Example 6: Get database statistics."""
    print("\n" + "="*60)
    print("Example 6: Database Statistics")
    print("="*60)

    db = get_db_connection()

    with db.session_scope() as session:
        # Total documents
        doc_count = session.query(Document).count()

        # Total text elements
        elem_count = session.query(TextElement).count()

        # Average elements per document
        avg_elements = elem_count / doc_count if doc_count > 0 else 0

        # Total words
        total_words = session.query(func.sum(TextElement.word_count)).scalar() or 0

        # Documents with most elements
        top_docs = session.query(
            Document.pmcid,
            Document.title,
            func.count(TextElement.id).label('element_count')
        )\
            .join(TextElement)\
            .group_by(Document.id)\
            .order_by(func.count(TextElement.id).desc())\
            .limit(5)\
            .all()

        print(f"\nDatabase Statistics:")
        print(f"  Total Documents: {doc_count}")
        print(f"  Total Text Elements: {elem_count}")
        print(f"  Average Elements per Document: {avg_elements:.1f}")
        print(f"  Total Words: {total_words:,}")

        print(f"\n  Top 5 Documents by Element Count:")
        for pmcid, title, count in top_docs:
            print(f"    {pmcid}: {count} elements")
            print(f"      {title[:70]}...")


def example_7_search_text_content(keyword="staining"):
    """Example 7: Search for text content containing specific keywords."""
    print("\n" + "="*60)
    print(f"Example 7: Search Text Content for '{keyword}'")
    print("="*60)

    db = get_db_connection()

    with db.session_scope() as session:
        elements = session.query(TextElement)\
            .filter(TextElement.text_content.ilike(f'%{keyword}%'))\
            .limit(10)\
            .all()

        print(f"\nFound {len(elements)} text elements containing '{keyword}':\n")

        for elem in elements:
            # Highlight the keyword in text
            text = elem.text_content
            if len(text) > 150:
                # Find keyword position and show context
                pos = text.lower().find(keyword.lower())
                if pos != -1:
                    start = max(0, pos - 50)
                    end = min(len(text), pos + len(keyword) + 50)
                    text = f"...{text[start:end]}..."

            print(f"Document: {elem.document.pmcid}")
            print(f"  Path: {elem.path_string}")
            print(f"  Text: {text}")
            print()


def example_8_hierarchical_query():
    """Example 8: Hierarchical query using path_list array."""
    print("\n" + "="*60)
    print("Example 8: Hierarchical Query (Find all in 'Methods' sections)")
    print("="*60)

    db = get_db_connection()

    with db.session_scope() as session:
        # Using PostgreSQL array contains operator
        # Find all text elements where 'Methods' appears anywhere in the path
        elements = session.query(TextElement)\
            .filter(TextElement.path_list.contains(['Methods']))\
            .limit(10)\
            .all()

        print(f"\nFound {len(elements)} text elements in Methods sections:\n")

        for elem in elements:
            print(f"Path: {elem.path_string}")
            print(f"  Document: {elem.document.pmcid}")
            print(f"  Hierarchy: {' > '.join(elem.path_list)}")
            print(f"  Text: {elem.text_content[:80]}...")
            print()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("NLP Histopathology Database - Query Examples")
    print("="*60)

    try:
        # Load environment variables
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            pass

        # Run examples
        example_1_get_all_documents()
        example_2_get_document_by_pmcid()
        example_3_get_text_elements()
        example_4_search_by_path()
        example_5_get_by_depth()
        example_6_statistics()
        example_7_search_text_content()
        example_8_hierarchical_query()

        print("\n" + "="*60)
        print("All Examples Complete!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. Database has been set up (run: python database/setup_db.py)")
        print("  3. XML files have been imported (run: python database/xml_to_db.py)")

    finally:
        close_db_connection()


if __name__ == "__main__":
    main()
