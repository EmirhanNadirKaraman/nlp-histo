# Migration Guide: Using the Unified Ingestion Script

The database ingestion scripts have been consolidated into a single unified script: `database/ingest.py`

## What Changed

Previously, there were multiple scripts for ingesting data:
- `database/xml_to_db.py` - For XML files
- `database/ingest_with_references.py` - For PDFs with references
- `database/example_store_with_references.py` - Simple example
- `scripts/test_ingest_single_paper.py` - Test script

Now, **all functionality is unified** in `database/ingest.py`.

## Quick Start

### Single File Ingestion

```bash
# Ingest a single PDF
python database/ingest.py --pmcid PMC1234567 --pdf files/organized_pdfs/PMC1234567_his_2369.pdf

# Ingest a single XML
python database/ingest.py --pmcid PMC1234567 --xml files/organized_xmls/PMC1234567.nxml

# With PDFFigures2 JSON
python database/ingest.py --pmcid PMC1234567 --pdf file.pdf --pdffigures-json out/data/file.json

# With metadata
python database/ingest.py --pmcid PMC1234567 --pdf file.pdf --title "Paper Title" --year 2023
```

### Batch Ingestion

```bash
# Ingest all XML files in a directory
python database/ingest.py --xml-dir files/organized_xmls

# Ingest all PDF files
python database/ingest.py --pdf-dir files/organized_pdfs --pattern "*.pdf"

# Re-ingest existing documents (force update)
python database/ingest.py --xml-dir files/organized_xmls --force
```

## Using as a Library

The new script can also be imported and used in your Python code:

```python
from database.ingest import ingest_document

# Simple usage
success = ingest_document(
    pmcid='PMC1234567',
    pdf_path='files/organized_pdfs/PMC1234567_his_2369.pdf'
)

# With all options
success = ingest_document(
    pmcid='PMC1234567',
    pdf_path='files/organized_pdfs/PMC1234567.pdf',
    pdffigures_json_path='out/data/PMC1234567.json',
    title='My Paper Title',
    journal='Nature Medicine',
    publication_year=2023,
    force=True  # Re-ingest if exists
)

# With pre-parsed text elements
text_elements = [...]  # Your parsed data
success = ingest_document(
    pmcid='PMC1234567',
    source_path='file.pdf',
    text_elements=text_elements
)
```

## What Gets Ingested

The unified script handles:

1. **Text Elements**: Hierarchical structure with paths
2. **Figures**: From PDFFigures2 JSON or XML
3. **Tables**: From PDFFigures2 JSON
4. **References**: Automatic detection of figure/table mentions in text
5. **Junction Tables**: Relationships between text and figures/tables
6. **Metadata**: Title, journal, publication year

## Migration Examples

### Before (Old Scripts)

```python
# Old way - using ingest_with_references.py
from database.ingest_with_references import ingest_document_with_references

ingest_document_with_references(
    text_elements=text_elements,
    pdffigures_json_path='test_figures/PMC1448691.json',
    pmcid='PMC1448691',
    pdf_path='files/organized_pdfs/PMC1448691.pdf',
    title='Some Title'
)
```

### After (New Unified Script)

```python
# New way - using ingest.py
from database.ingest import ingest_document

ingest_document(
    pmcid='PMC1448691',
    pdf_path='files/organized_pdfs/PMC1448691.pdf',
    pdffigures_json_path='test_figures/PMC1448691.json',
    text_elements=text_elements,  # Optional - will auto-extract if not provided
    title='Some Title'
)
```

### Before (Batch XML Import)

```bash
# Old way
python database/xml_to_db.py
```

### After (Batch Import)

```bash
# New way - same functionality, more features
python database/ingest.py --xml-dir files/organized_xmls
```

## Key Improvements

1. **Unified Interface**: One script for all ingestion tasks
2. **Auto-Detection**: Automatically finds PDFFigures2 JSON files
3. **Reference Detection**: Automatically detects figure/table mentions
4. **Better Logging**: Clear progress and error messages
5. **Force Re-Ingestion**: `--force` flag to update existing documents
6. **Flexible Input**: Accepts pre-parsed data or extracts automatically

## Backwards Compatibility

The old scripts are still available with deprecation warnings. They will continue to work but are no longer actively developed. New features will only be added to `database/ingest.py`.

## Need Help?

Check the help message for all options:

```bash
python database/ingest.py --help
```

Or see examples in the docstring:

```python
from database import ingest
help(ingest.ingest_document)
```
