"""
Database Setup Script

This script initializes the database and creates all tables.
For fresh installations, this is the ONLY script you need to run.

For existing databases created before the schema updates, use the migration scripts:
  - migrate_add_tables_and_references.py
  - migrate_add_image_paths.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import database module (.env is automatically loaded by db_connection.py)
from database import get_db_connection, close_db_connection


def check_existing_tables(db):
    """
    Check which tables already exist in the database.

    Returns:
        dict: Table names and whether they exist
    """
    from sqlalchemy import inspect

    inspector = inspect(db.engine)
    existing_tables = inspector.get_table_names()

    expected_tables = [
        'documents',
        'text_elements',
        'figures',
        'tables',
        'text_element_figure_references',
        'text_element_table_references',
        'entities'
    ]

    return {table: table in existing_tables for table in expected_tables}


def setup_database(drop_existing=False, check_only=False):
    """
    Set up the database by creating all tables.

    Args:
        drop_existing: If True, drops existing tables before creating new ones.
                      WARNING: This will delete all data!
        check_only: If True, only checks existing tables without creating
    """
    print("\n" + "="*80)
    print("Database Setup - NLP Histopathology Project")
    print("="*80 + "\n")

    try:
        # Connect to database
        print("Connecting to database...")
        db = get_db_connection(echo=False)
        print("✓ Connected successfully!\n")

        # Check existing tables
        print("Checking existing tables...")
        table_status = check_existing_tables(db)

        tables_exist = any(table_status.values())
        all_tables_exist = all(table_status.values())

        if tables_exist:
            print("\nExisting tables found:")
            for table, exists in table_status.items():
                status = "✓" if exists else "✗"
                print(f"  {status} {table}")
            print()
        else:
            print("✓ No existing tables found (fresh installation)\n")

        # Check-only mode
        if check_only:
            print("="*80)
            print("Check Complete!")
            print("="*80)
            if all_tables_exist:
                print("\n✓ All tables exist. Database is ready to use.")
                print("\nTo reset the database:")
                print("  python database/setup_db.py --drop")
            elif tables_exist:
                print("\n⚠ Some tables are missing!")
                print("\nOptions:")
                print("  1. Drop and recreate: python database/setup_db.py --drop")
                print("  2. Run migrations: python database/migrate_add_tables_and_references.py")
            else:
                print("\n→ Run setup to create tables:")
                print("  python database/setup_db.py")
            print("="*80 + "\n")
            close_db_connection()
            return

        # Handle existing tables
        if tables_exist and not drop_existing:
            if all_tables_exist:
                print("="*80)
                print("All tables already exist!")
                print("="*80)
                print("\nDatabase is ready to use. No action needed.")
                print("\nTo reset the database (WARNING: deletes all data):")
                print("  python database/setup_db.py --drop")
                print("="*80 + "\n")
                close_db_connection()
                return
            else:
                print("Some tables are missing. Creating missing tables...\n")

        # Drop existing tables if requested
        if drop_existing and tables_exist:
            print("⚠️  WARNING: About to drop existing tables!")
            print("This will DELETE ALL DATA in the following tables:")
            for table, exists in table_status.items():
                if exists:
                    print(f"  • {table}")
            print()
            response = input("Are you sure? Type 'yes' to confirm: ")
            if response.lower() == 'yes':
                print("\nDropping tables...")
                db.drop_tables()
                print("✓ Tables dropped\n")
            else:
                print("\n✗ Cancelled. Keeping existing tables.")
                print("="*80 + "\n")
                close_db_connection()
                return

        # Create tables
        print("Creating database tables...")
        print()
        db.create_tables()

        # Verify creation
        table_status_after = check_existing_tables(db)
        all_created = all(table_status_after.values())

        if all_created:
            print("✓ All tables created successfully!\n")
        else:
            print("⚠ Some tables may not have been created:")
            for table, exists in table_status_after.items():
                status = "✓" if exists else "✗"
                print(f"  {status} {table}")
            print()

        # Show summary
        print("="*80)
        print("Setup Complete!")
        print("="*80)
        print("\nDatabase Schema Created:")
        print()
        print("  Core Tables:")
        print("    • documents              - Document metadata (PMCID, title, journal, etc.)")
        print("    • text_elements          - Hierarchical text with paths and references")
        print()
        print("  Figure/Table Tables:")
        print("    • figures                - Figure metadata, captions, and images")
        print("    • tables                 - Table metadata, captions, and bounding boxes")
        print()
        print("  Reference Junction Tables:")
        print("    • text_element_figure_references - Links text to figures")
        print("    • text_element_table_references  - Links text to tables")
        print()
        print("  Entity Tables:")
        print("    • entities               - Named entities with UMLS mapping and semantic types")
        print()
        print("  Key Features:")
        print("    ✓ Hierarchical path system (path_list, path_string)")
        print("    ✓ Automatic reference detection (JSON in text_elements.references)")
        print("    ✓ Many-to-many relationships (text ↔ figures/tables)")
        print("    ✓ Cascade delete (remove document → removes all related data)")
        print("    ✓ Full-text search ready (GIN indexes on text_content)")
        print()
        print("="*80)
        print("Next Steps:")
        print("="*80)
        print()
        print("  1. Verify tables were created:")
        print("     psql -U postgres -d nlp_histo -c '\\dt'")
        print()
        print("  2. Start ingesting documents:")
        print()
        print("     # Single file:")
        print("     python database/ingest.py --pmcid PMC1234567 --pdf file.pdf")
        print()
        print("     # Batch mode:")
        print("     python database/ingest.py --pdf-dir files/organized_pdfs")
        print()
        print("  3. Query the data:")
        print("     python database/query_examples.py")
        print()
        print("="*80 + "\n")

        # Close connection
        close_db_connection()

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        print("="*80)
        print("Troubleshooting:")
        print("="*80)
        print()
        print("  1. Check if PostgreSQL is running:")
        print("     pg_isready")
        print()
        print("  2. List available databases:")
        print("     psql -U postgres -l")
        print()
        print("  3. Create the database if it doesn't exist:")
        print("     createdb -U postgres nlp_histo")
        print()
        print("  4. Verify .env file has correct credentials:")
        print("     cat .env")
        print()
        print("  5. Test connection manually:")
        print("     psql -U postgres -d nlp_histo")
        print()
        print("="*80 + "\n")
        sys.exit(1)


def main():
    """Main setup function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up the NLP Histopathology database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh installation (create all tables)
  python database/setup_db.py

  # Check existing tables without creating
  python database/setup_db.py --check

  # Drop and recreate all tables (WARNING: deletes data!)
  python database/setup_db.py --drop

Notes:
  - For fresh installations, this is the ONLY script you need
  - For existing databases, use migration scripts instead
  - Make sure PostgreSQL is running and database exists
        """
    )
    parser.add_argument(
        '--drop',
        action='store_true',
        help='Drop existing tables before creating (WARNING: deletes all data!)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check existing tables without creating new ones'
    )

    args = parser.parse_args()

    setup_database(drop_existing=args.drop, check_only=args.check)


if __name__ == "__main__":
    main()
