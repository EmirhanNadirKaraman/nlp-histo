"""
Migration: Add semantic_types column to entities table

This migration adds a semantic_types column to the entities table
to store UMLS semantic type information (TUI codes or names).

Usage:
    python database/migrations/add_semantic_types.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database import get_db_connection, close_db_connection
from sqlalchemy import text


def check_column_exists(engine, table_name, column_name):
    """
    Check if a column exists in a table.

    Args:
        engine: SQLAlchemy engine
        table_name: Name of the table
        column_name: Name of the column

    Returns:
        bool: True if column exists, False otherwise
    """
    query = text("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = :table_name
            AND column_name = :column_name
        );
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"table_name": table_name, "column_name": column_name})
        return result.scalar()


def add_semantic_types_column(db):
    """
    Add semantic_types column to entities table.

    Args:
        db: DatabaseConnection instance
    """
    print("\n" + "="*80)
    print("Migration: Add semantic_types column to entities table")
    print("="*80 + "\n")

    # Check if column already exists
    print("Checking if semantic_types column already exists...")
    column_exists = check_column_exists(db.engine, 'entities', 'semantic_types')

    if column_exists:
        print("✓ Column 'semantic_types' already exists in 'entities' table.")
        print("  No migration needed.\n")
        print("="*80)
        return

    print("✗ Column 'semantic_types' does not exist.")
    print("  Proceeding with migration...\n")

    # Add the column
    print("Adding semantic_types column...")
    alter_query = text("""
        ALTER TABLE entities
        ADD COLUMN semantic_types VARCHAR NULL;
    """)

    try:
        with db.engine.connect() as conn:
            conn.execute(alter_query)
            conn.commit()

        print("✓ Successfully added semantic_types column to entities table.\n")

        # Verify the column was added
        if check_column_exists(db.engine, 'entities', 'semantic_types'):
            print("✓ Verification: Column exists in database.\n")
        else:
            print("⚠ Warning: Column may not have been added correctly.\n")

        print("="*80)
        print("Migration Complete!")
        print("="*80)
        print("\nThe entities table now has a semantic_types column to store")
        print("UMLS semantic type information (TUI codes or names).\n")
        print("="*80 + "\n")

    except Exception as e:
        print(f"❌ Error adding column: {e}\n")
        print("="*80)
        raise


def main():
    """Run the migration."""
    try:
        # Connect to database
        print("Connecting to database...")
        db = get_db_connection(echo=False)
        print("✓ Connected successfully!\n")

        # Run migration
        add_semantic_types_column(db)

        # Close connection
        close_db_connection()

    except Exception as e:
        print(f"\n❌ Migration failed: {e}\n")
        print("="*80)
        print("Troubleshooting:")
        print("="*80)
        print()
        print("  1. Check if PostgreSQL is running:")
        print("     pg_isready")
        print()
        print("  2. Verify database exists:")
        print("     psql -U postgres -l | grep nlp_histo")
        print()
        print("  3. Check if entities table exists:")
        print("     psql -U postgres -d nlp_histo -c '\\d entities'")
        print()
        print("="*80 + "\n")
        close_db_connection()
        sys.exit(1)


if __name__ == "__main__":
    main()
