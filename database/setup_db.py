"""
Database Setup Script

This script initializes the database, creates tables, and optionally
loads environment variables from a .env file.
"""

import sys
from pathlib import Path

# IMPORTANT: Load .env file BEFORE importing database module
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print("No .env file found. Using default configuration.")
        print("To configure, copy .env.example to .env and update values.")
except ImportError:
    print("python-dotenv not installed. Using default configuration.")
    print("Install with: pip install python-dotenv")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# NOW import database module (after .env is loaded)
from database import get_db_connection, close_db_connection


def setup_database(drop_existing=False):
    """
    Set up the database by creating all tables.

    Args:
        drop_existing: If True, drops existing tables before creating new ones.
                      WARNING: This will delete all data!
    """
    print("\n" + "="*60)
    print("Database Setup")
    print("="*60 + "\n")

    try:
        # Connect to database
        print("Connecting to database...")
        db = get_db_connection(echo=True)
        print("✅ Connected successfully!\n")

        if drop_existing:
            print("⚠️  WARNING: Dropping existing tables...")
            response = input("Are you sure? This will delete all data! (yes/no): ")
            if response.lower() == 'yes':
                db.drop_tables()
                print("✅ Tables dropped\n")
            else:
                print("Cancelled. Keeping existing tables.\n")

        # Create tables
        print("Creating database tables...")
        db.create_tables()
        print("✅ Tables created successfully!\n")

        # Show summary
        print("="*60)
        print("Setup Complete!")
        print("="*60)
        print("\nTables created:")
        print("  - documents: Stores XML file metadata")
        print("  - text_elements: Stores hierarchical text data")
        print("\nNext steps:")
        print("  1. Run: python database/xml_to_db.py")
        print("  2. This will import XML files into the database")
        print("="*60 + "\n")

        # Close connection
        close_db_connection()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure PostgreSQL is running")
        print("  2. Check your .env file has correct database credentials")
        print("  3. Verify the database exists: psql -U postgres -l")
        print("  4. Create database if needed: createdb -U postgres nlp_histo")
        sys.exit(1)


def main():
    """Main setup function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up the NLP Histopathology database"
    )
    parser.add_argument(
        '--drop',
        action='store_true',
        help='Drop existing tables before creating (WARNING: deletes all data!)'
    )

    args = parser.parse_args()

    setup_database(drop_existing=args.drop)


if __name__ == "__main__":
    main()
