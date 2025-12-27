# Database Ingestion Guide

This guide explains how to use the unified ingestion script (`database/ingest.py`) to load documents into the PostgreSQL database.

## Quick Start

### Prerequisites

**Database setup** (run once):
```bash
python database/setup_db.py
```

This creates all tables you need. The migration scripts are only for existing databases created before the schema updates.

### Single File Ingestion

```bash
# Ingest a PDF file
python database/ingest.py --pmcid PMC1448691 --pdf files/organized_pdfs/PMC1448691_his_2369.pdf

# Ingest an XML file
python database/ingest.py --pmcid PMC1448691 --xml files/organized_xmls/PMC1448691.nxml

# With PDFFigures2 JSON (auto-detected if in common locations)
python database/ingest.py --pmcid PMC1448691 --pdf file.pdf --pdffigures-json out/data/file.json

# With metadata
python database/ingest.py \
    --pmcid PMC1448691 \
    --pdf file.pdf \
    --title "My Paper Title" \
    --journal "Nature Medicine" \
    --year 2023
```

### Batch Ingestion

```bash
# Ingest all XML files
python database/ingest.py --xml-dir files/organized_xmls

# Ingest all PDF files
python database/ingest.py --pdf-dir files/organized_pdfs

# Custom pattern
python database/ingest.py --xml-dir files/organized_xmls --pattern "PMC*.nxml"

# Re-ingest existing documents
python database/ingest.py --xml-dir files/organized_xmls --force
```

## What Gets Ingested

The unified script automatically handles:

### 1. Text Elements
- Hierarchical structure from XML or PDF
- Automatic reference detection for figures/tables
- Path-based organization (e.g., `PMC1234567/Methods > 2.1 Staining/0`)

### 2. Figures
- From PDFFigures2 JSON (if available)
- From XML `<fig>` tags
- Automatic number extraction from captions
- Image file management

### 3. Tables
- From PDFFigures2 JSON
- Bounding box information
- Automatic number extraction

### 4. References
- Automatic detection of "Figure 1", "Table 2", etc. in text
- Junction table entries linking text to figures/tables
- Stored both as JSON and relational links

## Using as a Library

### Basic Usage

```python
from database.ingest import ingest_document

# Simple ingestion
success = ingest_document(
    pmcid='PMC1234567',
    pdf_path='files/organized_pdfs/PMC1234567.pdf'
)

if success:
    print("Document ingested successfully!")
```

### Advanced Usage

```python
from database.ingest import ingest_document

# Full control
success = ingest_document(
    pmcid='PMC1234567',
    pdf_path='files/organized_pdfs/PMC1234567.pdf',
    pdffigures_json_path='out/data/PMC1234567.json',
    title='Paper Title',
    journal='Nature Medicine',
    publication_year=2023,
    force=True  # Re-ingest if already exists
)
```

### With Pre-Parsed Data

```python
from database.ingest import ingest_document

# If you already have parsed text elements
text_elements = [
    {
        'path_list': ['Methods', '2.1 Staining'],
        'path_string': 'Methods > 2.1 Staining',
        'depth': 2,
        'text': 'Tissue sections were stained...',
        'references': {
            'figures': ['1', '2'],
            'tables': []
        }
    },
    # ... more elements
]

success = ingest_document(
    pmcid='PMC1234567',
    source_path='files/organized_pdfs/PMC1234567.pdf',
    text_elements=text_elements
)
```

### Batch Processing

```python
from database.ingest import DocumentIngester
from pathlib import Path

ingester = DocumentIngester()

# Process directory
ingester.ingest_directory(
    directory=Path('files/organized_xmls'),
    pattern='*.nxml',
    force=False
)

# Print statistics
ingester.print_summary()
```

## Reference Detection

The script automatically detects figure and table mentions in text:

**Detected patterns:**
- `Figure 1`, `Fig. 2A`, `figure 3`
- `Table 1`, `TABLE 2`
- `(Figure 1)`, `(Table 2)`

**Stored as:**
```json
{
  "figures": ["1", "2A"],
  "tables": ["1"]
}
```

## File Path Auto-Detection

The script automatically searches for PDFFigures2 JSON files in common locations:
- `out/data/{filename}.json`
- `test_figures/{filename}.json`
- `output/{pmcid}/pdffigures2.json`
- Same directory as source file

You can override by providing `--pdffigures-json` explicitly.

## Error Handling

The ingestion script is designed to be robust:

- **Duplicates**: Skips existing documents (use `--force` to override)
- **Missing files**: Logs warning and continues with batch processing
- **Extraction errors**: Logs error and continues to next document
- **Transaction safety**: Database changes are rolled back on error

## Verifying Ingestion

After ingestion, verify the data:

```bash
# Check documents table
psql -U postgres -d nlp_histo -c "SELECT pmcid, title, text_source FROM documents;"

# Check counts
psql -U postgres -d nlp_histo -c "
  SELECT
    d.pmcid,
    COUNT(DISTINCT t.id) as text_elements,
    COUNT(DISTINCT f.id) as figures,
    COUNT(DISTINCT tb.id) as tables
  FROM documents d
  LEFT JOIN text_elements t ON d.id = t.document_id
  LEFT JOIN figures f ON d.id = f.document_id
  LEFT JOIN tables tb ON d.id = tb.document_id
  GROUP BY d.pmcid;
"

# Query with Python
from database.ingest_with_references import query_reference_examples
query_reference_examples('PMC1234567')
```

## Troubleshooting

### Document already exists

```bash
# Use --force to re-ingest
python database/ingest.py --pmcid PMC1234567 --pdf file.pdf --force
```

### PDFFigures2 JSON not found

The script will continue without figures/tables. To fix:
1. Run PDFFigures2 extraction first
2. Provide explicit path: `--pdffigures-json path/to/file.json`

### Missing references column

Run the migration:
```bash
python database/migrate_add_tables_and_references.py
```

### PDF extraction fails

The script will log the error and continue. Check:
1. PDF file is valid and readable
2. Required parsers are installed (see `requirements.txt`)
3. Check logs for specific error messages

## Performance Tips

### Batch Processing
- Use batch mode for multiple files (faster than individual ingestion)
- The script uses database sessions efficiently
- Progress is logged for each document

### Large Datasets
- Consider processing in chunks
- Monitor database disk space
- Use `VACUUM ANALYZE` after large ingestions

```bash
psql -U postgres -d nlp_histo -c "VACUUM ANALYZE;"
```

## Migration from Old Scripts

If you're using the old scripts, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions.

Quick summary:
- `database/ingest_with_references.py` → `database/ingest.py`
- `database/example_store_with_references.py` → `database/ingest.py`
- `scripts/test_ingest_single_paper.py` → `database/ingest.py --pmcid ... --pdf ...`
- `database/xml_to_db.py` → `database/ingest.py --xml-dir ...`

## Complete Example Workflow

```bash
# 1. Setup database
python database/setup_db.py

# 2. Run migrations
python database/migrate_add_tables_and_references.py

# 3. Download and organize files (if needed)
cd file-selector
python file_downloader.py
python tarball_extractor.py
python pdf_organizer.py
cd ..

# 4. Run PDFFigures2 extraction (if using PDFs)
# ... your PDFFigures2 extraction commands ...

# 5. Ingest into database
python database/ingest.py --pdf-dir files/organized_pdfs

# 6. Verify
psql -U postgres -d nlp_histo -c "SELECT COUNT(*) FROM documents;"
```

## Getting Help

```bash
# See all options
python database/ingest.py --help

# Check database schema
psql -U postgres -d nlp_histo -c "\dt"

# View Python docstrings
python -c "from database.ingest import ingest_document; help(ingest_document)"
```
