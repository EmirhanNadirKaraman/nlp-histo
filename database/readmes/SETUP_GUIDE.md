# Database Setup Guide

## Fresh Installation (Simple!)

For new installations, you only need **one command**:

```bash
python database/setup_db.py
```

This creates all 6 tables:
- ✓ `documents` - Document metadata
- ✓ `text_elements` - Hierarchical text with references
- ✓ `figures` - Figure metadata and images
- ✓ `tables` - Table metadata and bounding boxes
- ✓ `text_element_figure_references` - Text ↔ Figure links
- ✓ `text_element_table_references` - Text ↔ Table links

**That's it!** You don't need to run migration scripts for fresh installations.

---

## Useful Commands

### Check Existing Tables

```bash
# Check what tables exist without modifying anything
python database/setup_db.py --check
```

### Reset Database

```bash
# Drop and recreate all tables (WARNING: deletes all data!)
python database/setup_db.py --drop
```

### Verify with psql

```bash
# List all tables
psql -U postgres -d nlp_histo -c '\dt'

# Show table schema
psql -U postgres -d nlp_histo -c '\d documents'
psql -U postgres -d nlp_histo -c '\d text_elements'
```

---

## Existing Database (Migration)

**Only use migration scripts if:**
- ✓ You have an existing database with old schema
- ✓ You want to add new tables/columns without losing data

**Migration scripts:**
```bash
# Add new tables and columns
python database/migrate_add_tables_and_references.py

# Add image path columns
python database/migrate_add_image_paths.py
```

**For fresh installs:** Don't use migration scripts! Just use `setup_db.py`.

---

## Troubleshooting

### Database doesn't exist

```bash
# Create the database
createdb -U postgres nlp_histo
```

### Connection failed

```bash
# Check if PostgreSQL is running
pg_isready

# Or manually
brew services list | grep postgresql
```

### Wrong credentials

```bash
# Check your .env file
cat .env

# Should have:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nlp_histo
DB_USER=postgres
DB_PASSWORD=your_password
```

### Tables already exist

```bash
# Check which tables exist
python database/setup_db.py --check

# If you want to start fresh (deletes data!)
python database/setup_db.py --drop
```

---

## Quick Start Workflow

```bash
# 1. Create database (if needed)
createdb -U postgres nlp_histo

# 2. Configure .env file
cp .env.example .env
# Edit .env with your credentials

# 3. Setup database
python database/setup_db.py

# 4. Verify
psql -U postgres -d nlp_histo -c '\dt'

# 5. Start ingesting!
python database/ingest.py --pdf-dir files/organized_pdfs
```

---

## Summary

### ✅ DO THIS (Fresh Install)
```bash
python database/setup_db.py
```

### ❌ DON'T DO THIS (Fresh Install)
```bash
# These are only for existing databases!
python database/migrate_add_tables_and_references.py  # ❌ Not needed
python database/migrate_add_image_paths.py            # ❌ Not needed
```

### 🔄 DO THIS (Existing Database)
```bash
# Only if you have old database and want to preserve data
python database/migrate_add_tables_and_references.py  # ✅
python database/migrate_add_image_paths.py            # ✅
```

---

## Need Help?

Run with `--help` flag:
```bash
python database/setup_db.py --help
```

Check the logs:
```bash
# Setup creates detailed output showing what it did
python database/setup_db.py
```
