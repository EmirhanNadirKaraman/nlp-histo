# NLP Histopathology

A comprehensive system for parsing, extracting, and storing hierarchical text data from histopathology papers (PubMed Central XML files) into a PostgreSQL database.

## Features

- **XML Parsing**: Extract hierarchical structure (sections, paragraphs) from JATS XML files
- **Database Storage**: Store text data in PostgreSQL with unique paths for each text element
- **Hierarchical Paths**: Each text element has a unique path like `{PMCID}/{section_path}/{position}`
- **Metadata Extraction**: Extract title, journal, publication year from XML files
- **Query Tools**: Example scripts for querying and analyzing the stored data

## Project Structure

```
nlp-histo/
├── database/                    # Database module
│   ├── models.py               # SQLAlchemy models
│   ├── db_connection.py        # Database connection utilities
│   ├── schema.sql              # PostgreSQL schema
│   ├── setup_db.py             # Database setup script
│   ├── xml_to_db.py            # XML import script
│   ├── query_examples.py       # Query examples
│   └── README.md               # Database documentation
├── parsers/                     # XML parsing modules
│   └── xml_parsers/
│       ├── hierarchical_parser.py  # Main hierarchical parser
│       └── parser_script.py        # Alternative parser
├── file-selector/               # File organization scripts
│   ├── pdf_organizer.py
│   ├── tarball_extractor.py
│   └── file_downloader.py
├── files/                       # Data directory
│   ├── organized_xmls/          # XML files
│   └── organized_pdfs/          # PDF files
├── requirements.txt             # Python dependencies
├── .env.example                 # Database configuration template
└── README.md                    # This file
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install PostgreSQL (if not already installed)
# macOS:
brew install postgresql
brew services start postgresql

# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### 2. Set Up Database

```bash
# Create database
createdb -U postgres nlp_histo

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your database credentials

# Initialize database schema
python database/setup_db.py
```

### 3. Import XML Files

```bash
# Make sure XML files are in files/organized_xmls/
# Then run the import script
python database/xml_to_db.py
```

This will:
- Parse all XML files in `files/organized_xmls/`
- Extract hierarchical structure (sections, paragraphs)
- Store documents and text elements in PostgreSQL
- Assign unique paths to each text element

### 4. Query the Database

```bash
# Run example queries
python database/query_examples.py
```

Or use Python to query programmatically:

```python
from database import get_db_connection, Document, TextElement

db = get_db_connection()

with db.session_scope() as session:
    # Get all documents
    documents = session.query(Document).all()

    # Search for text in Methods sections
    elements = session.query(TextElement)\
        .filter(TextElement.path_string.like('%Methods%'))\
        .all()
```

## Database Schema

### Unique Path Format

Each text element has a unique path in the format:
```
{PMCID}/{section_hierarchy}/{position_in_section}
```

Examples:
- `PMC1448691/Methods > 2.1 Staining/0`
- `PMC1448691/Methods > 2.1 Staining/1`
- `PMC1448691/Results > Clinical Findings/0`

### Tables

**documents**
- Stores metadata about each XML file
- Fields: `pmcid`, `title`, `journal`, `publication_year`, `filename`, `file_path`

**text_elements**
- Stores individual text elements with hierarchical paths
- Fields: `unique_path`, `path_list`, `path_string`, `depth`, `position_in_section`, `text_content`, `word_count`, `char_count`
- Each element is linked to a document via `document_id`

## Example Queries

### Get all text from a specific document

```python
with db.session_scope() as session:
    doc = session.query(Document).filter_by(pmcid="PMC1448691").first()
    elements = session.query(TextElement).filter_by(document_id=doc.id).all()
```

### Search for text containing specific keywords

```python
with db.session_scope() as session:
    elements = session.query(TextElement)\
        .filter(TextElement.text_content.ilike('%staining%'))\
        .all()
```

### Get all text at a specific hierarchical depth

```python
with db.session_scope() as session:
    # Get all level-2 sections
    elements = session.query(TextElement).filter_by(depth=2).all()
```

### Find all text in Methods sections

```python
with db.session_scope() as session:
    elements = session.query(TextElement)\
        .filter(TextElement.path_list.contains(['Methods']))\
        .all()
```

## Documentation

- [Database Module README](database/README.md) - Detailed database documentation
- [Query Examples](database/query_examples.py) - Example queries and usage patterns

## Configuration

Database connection is configured via environment variables in `.env`:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nlp_histo
DB_USER=your_username
DB_PASSWORD=your_password
```

## License

See LICENSE file for details.
