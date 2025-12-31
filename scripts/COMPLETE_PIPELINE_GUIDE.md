# Complete Pipeline Guide

The `complete_pipeline.py` script combines all features from both `comprehensive_ingest.py` and the Jupyter notebook workflow into a single unified pipeline.

## Features

### From comprehensive_ingest.py
- ✅ Reference detection (figures/tables mentioned in text)
- ✅ References section skipping
- ✅ Reference junction table creation (many-to-many relationships)
- ✅ Sophisticated hierarchical structure handling
- ✅ Database ingestion with full relationships

### From Jupyter Notebook
- ✅ Step-by-step processing workflow
- ✅ Hierarchical path-based text grouping
- ✅ Paragraph stitching within sections
- ✅ Masked PDF re-extraction for clean text
- ✅ Table reconstruction from captions

### Additional Features
- ✅ Text-only mode (no database required)
- ✅ Batch processing of multiple PDFs
- ✅ Comprehensive logging and statistics
- ✅ Error handling and recovery

## Installation

```bash
# Install required dependencies
pip install docling PyMuPDF

# For database ingestion (optional)
pip install psycopg2-binary sqlalchemy
```

## Usage

### Text-Only Mode (No Database)

Process a single PDF and extract text organized by hierarchical sections:

```bash
python scripts/complete_pipeline.py \
    --pdf files/organized_pdfs/PMC1448691_his_2369.pdf \
    --pmcid PMC1448691
```

### Database Ingestion Mode

Process and ingest to PostgreSQL database:

```bash
python scripts/complete_pipeline.py \
    --pdf files/organized_pdfs/PMC1448691_his_2369.pdf \
    --pmcid PMC1448691 \
    --db-ingest \
    --title "Update on extranodal lymphomas" \
    --journal "Histopathology" \
    --year 2006
```

### Batch Processing

Process all PDFs in a directory:

```bash
python scripts/complete_pipeline.py \
    --pdf-dir files/organized_pdfs \
    --db-ingest
```

### Advanced Options

```bash
python scripts/complete_pipeline.py \
    --pdf files/organized_pdfs/PMC1448691_his_2369.pdf \
    --pmcid PMC1448691 \
    --db-ingest \
    --force \                    # Re-ingest even if exists
    --include-references \       # Don't skip References sections
    --debug                      # Enable debug logging
```

## Output Files

The pipeline creates the following outputs:

### Text Files
- `out/text/{PMCID}_text.txt` - Clean text organized by hierarchical paths with section references

### JSON Files
- `out/complete_pipeline/{PMCID}_layout.json` - Original PDF layout
- `files/tables/{PMCID}_tables.json` - Table metadata
- `files/figures/{PMCID}_figures.json` - Figure metadata

### Images
- `files/tables/{PMCID}_table_{N}.png` - Cropped table images
- `files/figures/{PMCID}_figure_{N}.png` - Cropped figure images

### Masked PDF
- `out/masked_pdfs/{PMCID}_masked.pdf` - PDF with tables/figures masked

### Database (if --db-ingest used)
- `documents` table - Document metadata
- `text_elements` table - Text with hierarchical paths
- `figures` table - Figure metadata and image paths
- `tables` table - Table metadata and image paths
- `text_element_figure_references` - Links text to figures
- `text_element_table_references` - Links text to tables

## Pipeline Steps

The pipeline processes documents in the following order:

1. **Extract Layout** - Use Docling to extract all elements from original PDF
2. **Table Reconstruction** - Group table captions with content rows
3. **Create Masked PDF** - Replace tables/figures with white rectangles
4. **Re-extract Layout** - Extract clean text from masked PDF
5. **Build Hierarchy** - Create hierarchical structure from section headers
6. **Detect References** - Find figure/table mentions in text
7. **Stitch Paragraphs** - Join split paragraphs within each section
8. **Crop Media** - Extract table/figure images
9. **Save Text File** - Export organized text with section references
10. **Database Ingestion** (optional) - Store everything with relationships

## Key Differences from Other Scripts

### vs. comprehensive_ingest.py
- ✅ Uses masked PDF re-extraction for cleaner text
- ✅ Stitches paragraphs within sections (joins split text)
- ✅ Groups text by hierarchical paths before stitching
- ✅ Supports text-only mode (no database required)

### vs. Jupyter Notebook
- ✅ Includes reference detection
- ✅ Skips References sections automatically
- ✅ Creates reference junction tables
- ✅ Command-line interface for automation
- ✅ Batch processing support

## Example Output Structure

### Text File Format
```
Document: PMC1448691
================================================================================

[Introduction]
--------------------------------------------------------------------------------

This paper presents an update on extranodal lymphomas based on discussions
from the Workshop held by the European Association for Haematopathology...

  [Section References: Figures 1, 2; Tables 1]


[Methods > 2.1 Staining]
--------------------------------------------------------------------------------

Immunohistochemical staining was performed using the avidin-biotin complex
method with the following antibodies...

  [Section References: Table 2]
```

### Database Schema
```
documents (id, pmcid, title, journal, year, ...)
├── text_elements (id, document_id, unique_path, path_list, path_string, text_content, references, ...)
│   ├── text_element_figure_references (text_element_id, figure_id)
│   └── text_element_table_references (text_element_id, table_id)
├── figures (id, document_id, figure_id, caption_text, image_path, ...)
└── tables (id, document_id, table_id, caption_text, image_path, ...)
```

## Troubleshooting

### "Docling not available"
```bash
pip install docling
```

### "PyMuPDF not available"
```bash
pip install PyMuPDF
```

### "Database not available"
Ensure you have:
1. Created the database: `createdb -U postgres nlp_histo`
2. Configured `.env` file with database credentials
3. Initialized schema: `python database/setup_db.py`
4. Installed dependencies: `pip install psycopg2-binary sqlalchemy`

### "mask_tables.py not available"
Ensure `scripts/docling/mask_tables.py` exists in your project.

### References section still appearing
Use `--skip-references` flag (this is the default behavior).

## Performance Tips

1. **Batch Processing** - Use `--pdf-dir` for multiple files
2. **Text-Only Mode** - Skip `--db-ingest` if you only need text extraction
3. **Parallel Processing** - Run multiple instances with different PDF directories
4. **Debug Mode** - Use `--debug` only when troubleshooting

## Integration with Existing Scripts

### Query the Database
```bash
python database/query_examples.py
```

### Use with Existing Workflows
```python
from scripts.complete_pipeline import CompletePipelineProcessor

processor = CompletePipelineProcessor()
processor.process_document(
    pdf_path=Path("path/to/file.pdf"),
    pmcid="PMC1234567",
    db_ingest=True
)
```

## Future Enhancements

Potential additions:
- [ ] Multi-threaded batch processing
- [ ] Progress bars for long operations
- [ ] Resume capability for interrupted processing
- [ ] Custom reference patterns
- [ ] Configurable output directories
- [ ] PDF merge for supplementary materials
