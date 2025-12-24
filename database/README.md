# Database Module

This module handles storing and managing hierarchical text data extracted from XML files in PostgreSQL.

## Overview

The database stores:
- **Documents**: Metadata about each XML file (PMCID, title, journal, year)
- **Text Elements**: Individual paragraphs with their hierarchical paths

Each text element has a **unique path** in the format: `{PMCID}/{path_string}/{position}`

For example:
- `PMC1448691/Methods > 2.1 Staining/0`
- `PMC1448691/Methods > 2.1 Staining/1`
- `PMC1448691/Results/0`

## Setup

### 1. Install PostgreSQL

Make sure you have PostgreSQL installed on your system:

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Windows:**
Download and install from [postgresql.org](https://www.postgresql.org/download/windows/)

### 2. Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE nlp_histo;

# Exit psql
\q
```

### 3. Configure Environment Variables

Copy the example environment file and update with your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your database settings:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nlp_histo
DB_USER=postgres
DB_PASSWORD=your_password
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Initialize Database Schema

You can initialize the database schema in two ways:

**Option A: Using SQL directly**
```bash
psql -U postgres -d nlp_histo -f database/schema.sql
```

**Option B: Using Python (recommended)**
```bash
python database/setup_db.py
```

## Usage

### Import XML Files to Database

Run the main import script:

```bash
python database/xml_to_db.py
```

This will:
1. Scan the `files/organized_xmls/` directory for XML files
2. Parse each file using the hierarchical parser
3. Extract metadata (title, journal, year)
4. Store documents and text elements in PostgreSQL
5. Skip files that are already in the database

### Query the Database

You can query the database using any PostgreSQL client or Python:

**Example: Get all text elements from a specific document**
```python
from database import get_db_connection, Document, TextElement

db = get_db_connection()

with db.session_scope() as session:
    # Get document
    doc = session.query(Document).filter_by(pmcid="PMC1448691").first()

    # Get all text elements
    elements = session.query(TextElement).filter_by(document_id=doc.id).all()

    for elem in elements:
        print(f"{elem.unique_path}: {elem.text_content[:50]}...")
```

**Example: Search for text in a specific section**
```python
with db.session_scope() as session:
    # Find all text elements in "Methods" sections
    elements = session.query(TextElement)\
        .filter(TextElement.path_string.like('%Methods%'))\
        .all()

    for elem in elements:
        print(f"{elem.unique_path}")
```

**Example: Get text at a specific depth**
```python
with db.session_scope() as session:
    # Get all level-2 sections
    elements = session.query(TextElement)\
        .filter_by(depth=2)\
        .all()
```

## Database Schema

### documents table
- `id`: Primary key
- `pmcid`: PubMed Central ID (unique)
- `filename`: Original filename
- `file_path`: Absolute path to XML file
- `title`: Paper title
- `journal`: Journal name
- `publication_year`: Year of publication
- `processed_at`: When the file was processed
- `created_at`: Record creation timestamp
- `updated_at`: Record update timestamp

### text_elements table
- `id`: Primary key
- `document_id`: Foreign key to documents
- `unique_path`: Unique identifier (format: `{PMCID}/{path_string}/{position}`)
- `path_list`: Array of section names (PostgreSQL array)
- `path_string`: Human-readable path (e.g., "Methods > 2.1 Staining")
- `depth`: Nesting level (1, 2, 3, ...)
- `position_in_section`: Position within the same path
- `text_content`: The actual paragraph text
- `word_count`: Number of words
- `char_count`: Number of characters
- `created_at`: Record creation timestamp

## Advanced Queries

### Full-Text Search
```sql
-- Search for specific terms in text content
SELECT unique_path, text_content
FROM text_elements
WHERE to_tsvector('english', text_content) @@ to_tsquery('english', 'staining & procedure');
```

### Hierarchical Path Queries
```sql
-- Find all text in Methods sections and subsections
SELECT unique_path, text_content
FROM text_elements
WHERE 'Methods' = ANY(path_list);
```

### Statistics
```sql
-- Count text elements per document
SELECT d.pmcid, d.title, COUNT(te.id) as element_count
FROM documents d
LEFT JOIN text_elements te ON d.id = te.document_id
GROUP BY d.id
ORDER BY element_count DESC;
```

## Files

- `schema.sql`: Database schema definition
- `models.py`: SQLAlchemy models (Document, TextElement)
- `db_connection.py`: Database connection utilities
- `xml_to_db.py`: Main import script
- `setup_db.py`: Database initialization script
- `README.md`: This file

## Troubleshooting

**Connection Error:**
- Make sure PostgreSQL is running
- Check your `.env` file has correct credentials
- Test connection: `psql -U postgres -d nlp_histo`

**Import Errors:**
- Ensure XML files exist in `files/organized_xmls/`
- Check XML files are valid JATS format
- Look for specific error messages in output

**Duplicate Key Errors:**
- The script automatically skips documents already in the database
- To re-import, delete the document first or drop/recreate tables
