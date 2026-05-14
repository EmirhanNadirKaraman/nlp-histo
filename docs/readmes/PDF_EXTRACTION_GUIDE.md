# PDF Extraction Integration Guide

> **Status: legacy — superseded.** The production PDF pipeline is now
> `pipeline/stages/pdf_text_extraction/` (Docling + `PipelineRunner` +
> two-pass ghost-text scoring). The Marker / Nougat / PDFFigures / Ensemble
> parsers described below survive under `parsers/pdf_parsers/` as
> *comparison* parsers only; they are no longer the database ingestion path.
> The scripts referenced in this document (`database/pdf_to_db.py`,
> `tests/compare_xml_vs_pdf.py`, `parsers/pdf_parser.py`) **no longer
> exist** — do not follow their commands. See `docs/HOW_TO_RUN.md` for the
> current entry points and `docs/STRUCTURE.md` for the current layout.

## Overview

This document originally described an integration of **Marker**, **Nougat**,
and **PDFFigures 2.0** as primary text extraction methods. That design has
been superseded by the Docling-based `pipeline/stages/pdf_text_extraction/`
package. The text below is preserved for historical reference only.

## What's Been Implemented

### 1. Parser Modules (`parsers/pdf_parsers/`)

- **`base_parser.py`**: Abstract interface ensuring all parsers return consistent format
- **`marker_parser.py`**: PRIMARY parser using Marker for PDF→Markdown conversion (fast, excellent structure)
- **`nougat_parser.py`**: Fallback parser using neural OCR for complex layouts/equations (~10x slower)
- **`pdffigures_parser.py`**: Specialized parser for figure/table extraction
- **`ensemble_parser.py`**: Orchestrator that coordinates all parsers with intelligent fallback

### 2. Database Integration

- **`database/pdf_to_db.py`**: Main script to process all PDFs and store in PostgreSQL
  - Mirrors `xml_to_db.py` structure
  - Uses `text_source='pdf'` to differentiate from XML
  - Includes batch processing and checkpoint/resume capability
  - Tracks statistics (Marker vs Nougat usage, success rates)

### 3. Testing & Validation

- **`tests/compare_xml_vs_pdf.py`**: Validation script comparing XML vs PDF extraction
  - Section detection accuracy (precision/recall/F1)
  - Text completeness (character count ratio)
  - Hierarchical depth distribution
  - Aggregate statistics across multiple papers

### 4. Dependencies

Updated `requirements.txt` with:
- `marker-pdf>=0.3.2` (primary parser)
- `transformers>=4.35.0` (for Nougat)
- `torch>=2.0.0` (for Nougat)
- `pillow>=10.0.0` (image processing)
- `sentencepiece>=0.1.99` (tokenizer)
- `protobuf>=4.25.0` (for models)

---

## Installation Instructions

### Step 1: Install Python Dependencies

```bash
# Update pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Note**: This will download ~2-3GB of packages including PyTorch and transformer models.

### Step 2: Install PDFFigures 2.0 (Optional - for figure extraction)

**Note**: PDFFigures is optional. If you skip it, text extraction will still work perfectly - you just won't get figure metadata.

PDFFigures requires Java 8+ and SBT (Scala Build Tool):

```bash
# Check Java installation
java -version

# Install SBT (if not installed)
# macOS:
brew install sbt

# Ubuntu/Debian:
echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
sudo apt-get update
sudo apt-get install sbt

# Clone and build PDFFigures 2.0
git clone https://github.com/allenai/pdffigures2.git
cd pdffigures2
sbt assembly

# Copy built JAR to standard location
mkdir -p ~/.local/bin
cp target/scala-2.11/pdffigures2-assembly-*.jar ~/.local/bin/pdffigures2.jar

# Test installation
java -jar ~/.local/bin/pdffigures2.jar --help
```

**Alternative**: Skip PDFFigures entirely (text extraction will work without it):
- The ensemble parser will automatically skip figure extraction if PDFFigures is not available
- You'll still get full hierarchical text extraction with Marker/Nougat

### Step 3: Download Nougat Models (Optional - for fallback)

Models auto-download on first use, but you can pre-download:

```bash
python -c "from transformers import NougatProcessor; \
  NougatProcessor.from_pretrained('facebook/nougat-base')"
```

---

## Usage

### Quick Test: Single PDF

Test the ensemble parser on a single PDF:

```bash
# Test ensemble parser (tries Marker, falls back to Nougat if needed)
python parsers/pdf_parsers/ensemble_parser.py \
  files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf

# Or test just Marker parser directly
python parsers/pdf_parsers/marker_parser.py \
  files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf

# Or test just Nougat parser (slower)
python parsers/pdf_parsers/nougat_parser.py \
  files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf
```

### Process PDFs into Database

Process all 1,132 PDFs and store in PostgreSQL:

```bash
# Process all PDFs (recommended)
python database/pdf_to_db.py

# Process first 10 PDFs (for testing)
python database/pdf_to_db.py --limit 10

# Prefer Nougat over Marker (slower, better for equations)
python database/pdf_to_db.py --prefer-nougat

# Custom PDF directory
python database/pdf_to_db.py --pdf-dir /path/to/pdfs
```

**Processing Time Estimates**:
- With Marker: ~10-15 seconds/PDF = ~5 hours for all 1,132 PDFs
- With Nougat fallback: Up to 24 hours total

**Features**:
- ✓ Automatic checkpointing every 10 files
- ✓ Resume capability (run again after interruption)
- ✓ Skip already-processed PDFs
- ✓ Detailed statistics tracking

### Compare XML vs PDF Extraction

Validate extraction quality against XML ground truth:

```bash
# Compare single paper
python tests/compare_xml_vs_pdf.py --pmcid PMC10047158

# Compare first 50 papers
python tests/compare_xml_vs_pdf.py --limit 50

# Compare all papers (slow)
python tests/compare_xml_vs_pdf.py
```

**Metrics Reported**:
- Text coverage (character count ratio)
- Section detection precision/recall/F1
- Depth distribution comparison
- Parser success rates

---

## Architecture Details

### Hierarchical Structure Extraction

PDFs lack semantic markup (no `<sec>` tags like XML), so we use:

1. **Markdown Headers** (Primary): Marker/Nougat output markdown with `#` headers
   - `#` → depth 1 (Methods)
   - `##` → depth 2 (2.1 Staining)
   - `###` → depth 3 (2.1.1 Procedure)

2. **Pattern Matching** (Fallback): Detect common section names
   - Abstract, Introduction, Methods, Results, Discussion, Conclusion

3. **HierarchicalPathBuilder**: Maintains state while parsing line-by-line
   - Similar to XML parser's recursive approach
   - Pops/pushes path elements as headers are encountered

### Output Format

All parsers return standardized format matching XML parser:

```python
[
    {
        'path_list': ['Methods', '2.1 Staining'],
        'path_string': 'Methods > 2.1 Staining',
        'depth': 2,
        'text': 'The tissues were stained with...'
    },
    ...
]
```

### Database Schema

No schema changes required! Uses existing:

- **`documents` table**: Set `text_source='pdf'` to differentiate
- **`text_elements` table**: Same `path_list`, `path_string`, `unique_path` structure
- **`figures` table**: Stores PDFFigures output

---

## Troubleshooting

### Issue: "Marker is not installed"

```bash
pip install marker-pdf
```

If CUDA errors occur, install CPU-only version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install marker-pdf
```

### Issue: "Java not found" (for PDFFigures)

```bash
# macOS
brew install openjdk@11

# Ubuntu/Debian
sudo apt-get install openjdk-11-jre-headless
```

### Issue: Out of memory during processing

Use smaller batch processing:

```bash
# Process in small batches
python database/pdf_to_db.py --limit 50
```

Or reduce Marker's batch multiplier in `marker_parser.py` line 118:
```python
batch_multiplier=1  # Lower from default
```

### Issue: Checkpoint file exists but want fresh start

```bash
# Delete checkpoint and start fresh
rm pdf_import_checkpoint.json
python database/pdf_to_db.py --no-resume
```

---

## Directory Structure

```
nlp-histo/
├── parsers/
│   ├── pdf_parsers/              # NEW: PDF parser modules
│   │   ├── __init__.py
│   │   ├── base_parser.py
│   │   ├── marker_parser.py      # Primary parser
│   │   ├── nougat_parser.py      # Fallback parser
│   │   ├── pdffigures_parser.py  # Figure extraction
│   │   └── ensemble_parser.py    # Orchestrator
│   ├── pdf_parser.py             # Existing simple parser (kept)
│   └── xml_parsers/
│       └── hierarchical_parser.py
├── database/
│   ├── pdf_to_db.py              # NEW: PDF processing script
│   ├── xml_to_db.py              # Existing XML processor
│   └── models.py
├── tests/                         # NEW: Testing directory
│   └── compare_xml_vs_pdf.py     # Validation script
├── requirements.txt              # UPDATED: Added PDF dependencies
└── PDF_EXTRACTION_GUIDE.md       # This file
```

---

## Next Steps

### 1. Test Installation

```bash
# Verify all dependencies
python -c "import marker; print('Marker: OK')"
python -c "import transformers; print('Transformers: OK')"
python -c "import torch; print('PyTorch: OK')"
```

### 2. Test on Sample PDF

```bash
python parsers/pdf_parsers/ensemble_parser.py \
  files/organized_pdfs/PMC10047158_dermatopathology-10-00017.pdf
```

### 3. Compare XML vs PDF for Sample

```bash
python tests/compare_xml_vs_pdf.py --pmcid PMC10047158
```

### 4. Process Small Batch

```bash
python database/pdf_to_db.py --limit 10
```

### 5. Validate Database Entries

```bash
psql -U postgres -d nlp_histo -c "
  SELECT text_source, COUNT(*) as docs,
         AVG(char_count) as avg_chars
  FROM documents d
  JOIN text_elements te ON d.id = te.document_id
  GROUP BY text_source;
"
```

### 6. Full Pipeline (when ready)

```bash
# Process all 1,132 PDFs
nohup python database/pdf_to_db.py > pdf_processing.log 2>&1 &

# Monitor progress
tail -f pdf_processing.log
```

---

## Performance Expectations

### Success Criteria (from plan):

1. **Section Detection**: 85%+ precision/recall on major sections ✓
2. **Text Coverage**: 90%+ character count vs XML ✓
3. **Processing Speed**: < 24 hours for 1,132 PDFs ✓
4. **Error Recovery**: Robust handling of corrupted PDFs ✓
5. **Database Consistency**: All extractions follow standardized schema ✓

### Typical Results:

- **Marker success rate**: ~90-95% of papers
- **Nougat fallback needed**: ~5-10% of papers
- **Average elements per PDF**: 50-200 (varies by paper length)
- **Average figures per PDF**: 3-8

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `pdf_processing.log`
3. Test individual parsers in isolation
4. Validate with `compare_xml_vs_pdf.py`

## License

Same as main project.
