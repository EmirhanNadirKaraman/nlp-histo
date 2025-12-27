# Figure and Table References Guide

This guide explains how to store and query figure/table references in the database.

## Overview

The database schema supports tracking which text elements mention which figures and tables. This is implemented using:

1. **JSON column** (`text_elements.references`) - Fast lookup of what a text mentions
2. **Junction tables** - Proper relational links for complex queries
3. **Figure/Table models** - Store metadata about figures and tables

## Database Schema

```
documents
    ├── text_elements (paragraphs)
    │   └── references (JSON): {"figures": ["1", "2"], "tables": ["3"]}
    ├── figures
    │   └── figure_number: "1", "2A", etc.
    └── tables
        └── table_number: "1", "II", etc.

Junction tables (many-to-many):
    • text_element_figure_references
    • text_element_table_references
```

## Setup

### 1. Create Database Schema

For new databases, use the setup script:

```bash
python database/setup_db.py
```

This creates all tables including the new reference tracking tables.

### 2. Migrate Existing Database

If you already have a database, run the migration script:

```bash
python database/migrate_add_tables_and_references.py
```

This adds:
- `tables` table
- `text_element_figure_references` junction table
- `text_element_table_references` junction table
- `figure_number` column to `figures`
- `references` JSON column to `text_elements`

## Data Ingestion

### Complete Workflow

1. **Extract text with references** using `parse_single_pdf.py`:
   ```bash
   python scripts/parse_single_pdf.py \
       files/masked_pdfs/PMC1448691_his_2369_masked.pdf
   ```

   This automatically creates text elements with a `references` field:
   ```python
   {
       "text": "As shown in Figure 1 and Table 2...",
       "references": {
           "figures": ["1"],
           "tables": ["2"]
       },
       "path_string": "Results > Analysis",
       ...
   }
   ```

2. **Extract figures/tables** using PDFFigures2:
   ```bash
   sbt "runMain org.allenai.pdffigures2.FigureExtractor \
       input.pdf -m output.json -d output_figs/"
   ```

3. **Ingest everything** using `ingest_with_references.py`:
   ```python
   import json
   from database.ingest_with_references import ingest_document_with_references

   # Load text elements (output from parse_single_pdf.py)
   with open('output/PMC1448691_his_2369_masked_text_elements.json', 'r') as f:
       text_elements = json.load(f)

   # Ingest
   ingest_document_with_references(
       text_elements=text_elements,
       pdffigures_json_path='test_figures/PMC1448691_his_2369.json',
       pmcid='PMC1448691',
       pdf_path='files/organized_pdfs/PMC1448691_his_2369.pdf',
       title='Paper Title'
   )
   ```

## Querying References

### Example Queries

```python
from database import get_db_connection, Document, TextElement, Figure, Table
from database.models import TextElementFigureReference, TextElementTableReference

db = get_db_connection()

with db.session_scope() as session:
    # 1. Find all text elements that mention Figure 1
    texts = session.query(TextElement).join(
        TextElementFigureReference
    ).join(
        Figure
    ).filter(
        Figure.figure_number == '1'
    ).all()

    # 2. Get all figures mentioned in a specific section
    texts_in_results = session.query(TextElement).filter(
        TextElement.path_string.like('Results%')
    ).all()

    for text in texts_in_results:
        # Access via relationship
        for fig_ref in text.figure_references:
            print(f"Figure: {fig_ref.figure.figure_label}")

    # 3. Find all tables mentioned in text elements
    texts_with_tables = session.query(TextElement).filter(
        TextElement.references['tables'].isnot(None)
    ).all()

    # 4. Get a figure and find all texts that mention it
    figure = session.query(Figure).filter_by(figure_number='1').first()

    mentioning_texts = session.query(TextElement).join(
        TextElementFigureReference
    ).filter(
        TextElementFigureReference.figure_id == figure.id
    ).all()

    # 5. Complex query: Find Results section texts mentioning tables
    from sqlalchemy import and_

    results_texts_with_tables = session.query(TextElement).join(
        TextElementTableReference
    ).filter(
        and_(
            TextElement.path_string.like('Results%'),
            TextElementTableReference.table_id.isnot(None)
        )
    ).all()
```

### Using the Relationship API

The ORM provides convenient relationship navigation:

```python
# Get a text element
text = session.query(TextElement).first()

# Access figures it mentions (via relationship)
for fig_ref in text.figure_references:
    figure = fig_ref.figure
    print(f"Mentions: {figure.figure_label}")
    print(f"Caption: {figure.caption_text}")

# Access tables it mentions
for tbl_ref in text.table_references:
    table = tbl_ref.table
    print(f"Mentions: {table.table_label}")
    print(f"Bounding box: ({table.bbox_x1}, {table.bbox_y1})")

# Go the other direction - from figure to texts
figure = session.query(Figure).first()

for text_ref in figure.text_references:
    text_element = text_ref.text_element
    print(f"Mentioned in: {text_element.path_string}")
    print(f"Text: {text_element.text_content[:100]}...")
```

### Query Examples Script

Run the built-in query examples:

```bash
python database/ingest_with_references.py --query --pmcid PMC1448691
```

## Data Structure

### TextElement.references (JSON)

Quick lookup format:
```json
{
  "figures": ["1", "2A", "3"],
  "tables": ["1", "II"]
}
```

Query examples:
```python
# Find texts mentioning any figure
session.query(TextElement).filter(
    TextElement.references.has_key('figures')
)

# Find texts mentioning Figure 1 (substring match)
session.query(TextElement).filter(
    TextElement.references['figures'].astext.contains('1')
)
```

### Figure Model

```python
Figure(
    document_id=1,
    figure_id='fig1',           # From PDFFigures2
    figure_label='Figure 1',     # Display name
    figure_number='1',           # For matching references
    caption_text='This shows...',
    image_filename='fig1.png',
    section_context='Results'
)
```

### Table Model

```python
Table(
    document_id=1,
    table_id='table1',
    table_label='Table 1',
    table_number='1',
    caption_text='Summary of...',
    page_number=5,
    bbox_x1=100, bbox_y1=200,   # Bounding box
    bbox_x2=500, bbox_y2=600
)
```

## Benefits of This Design

1. **Fast lookups** - JSON column allows quick checks
2. **Relational integrity** - Foreign keys ensure data consistency
3. **Flexible queries** - Can query from any direction
4. **Cascade deletion** - Deleting a document removes all related data
5. **Extensible** - Easy to add more metadata or relationships

## Migration Path

If you have existing data without references:

1. Run migration to add schema
2. Re-extract PDFs with reference detection
3. Populate junction tables using the ingestion script
4. Old data still works (references will be NULL)

## See Also

- `database/models.py` - Full model definitions
- `database/ingest_with_references.py` - Complete ingestion example
- `scripts/parse_single_pdf.py` - Reference detection logic
- `database/example_store_with_references.py` - Simple storage example
