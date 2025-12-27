# Complete Pipeline Guide

This guide shows the complete end-to-end pipeline for processing histopathology papers from tarballs to database.

## Pipeline Overview

```
Tarballs (*.tar.gz)
    ↓ Step 1: Extract
PDFs in organized_pdfs/
    ↓ Step 2: PDFFigures2 extraction
JSON files in out/data/ + images in pdffigures2/
    ↓ Step 3: Extract figure/table images
Images in out/figures/ and out/tables/
    ↓ Step 4: Extract text from PDFs
Text elements with hierarchy + references
    ↓ Step 5: Save to database
PostgreSQL (documents, text_elements, figures, tables, references)
```

## Complete Pipeline Commands

### Step 1: Extract PDFs from Tarballs

**File:** `file-selector/tarball_extractor.py`
**Mode:** Batch (all tarballs at once)

```bash
cd file-selector
python tarball_extractor.py
```

**What it does:**
- Extracts all `.tar.gz` files from `histopathology_papers/`
- Places PDFs in `processed_corpus/{PMCID}/`

Then organize the files:

```bash
python pdf_organizer.py
```

**What it does:**
- Moves PDFs to `files/organized_pdfs/`
- Moves XMLs to `files/organized_xmls/`

**Recommendation:** Run in batch mode (processes all tarballs).

---

### Step 2: Run PDFFigures2 to Extract Figure/Table Coordinates

**File:** `scripts/run_pdffigures_resume.sh`
**Mode:** Batch (all PDFs at once)

```bash
cd /Users/emir/Documents/GitHub/nlp-histo
bash scripts/run_pdffigures_resume.sh
```

**What it does:**
- Runs PDFFigures2 Java tool on all PDFs in `files/organized_pdfs/`
- Creates JSON files with bounding box coordinates in `out/data/`
- Extracts figure/table images to `pdffigures2/`
- Skips already-processed PDFs (resume-safe)

**Outputs:**
- `out/data/{filename}.json` - Bounding box data
- `pdffigures2/{filename}-Figure1-1.png` - Extracted figures
- `pdffigures2/{filename}-Table1-1.png` - Extracted tables

**Recommendation:** Run in batch mode (much faster, uses resume logic).

**Note:** This script uses the PDFFigures2 JAR file at `pdffigures2/pdffigures2.jar`. Make sure Java 11 is installed.

---

### Step 3: Extract Figure/Table Images Using Bounding Boxes

**File:** `scripts/extract_figures_tables.py`
**Mode:** Batch (all JSON files at once)

```bash
python scripts/extract_figures_tables.py \
    --data-dir out/data \
    --pdf-dir files/organized_pdfs \
    --output-dir out
```

**What it does:**
- Reads JSON files from Step 2
- Uses PyMuPDF to extract figure/table regions from PDFs
- Saves images to:
  - `out/figures/` - Figure images
  - `out/tables/` - Table images

**Recommendation:** Run in batch mode (processes all JSON files).

**Optional:** Single file mode:
```bash
python scripts/extract_figures_tables.py --single-file out/data/PMC1234567.json
```

---

### Step 4: Extract Text from PDFs

**Option A: Single File (for testing/debugging)**

**File:** `scripts/parse_single_pdf.py`
**Mode:** One file at a time

```bash
python scripts/parse_single_pdf.py \
    files/organized_pdfs/PMC1448691_his_2369.pdf \
    --mode routing \
    --output-dir output
```

**What it does:**
- Uses `EnsemblePDFParser` (routing mode by default)
- Extracts hierarchical text structure
- Detects figure/table references in text
- Saves multiple output formats:
  - `output/{paper_name}/{timestamp}/routing_full.json` - Complete data
  - `output/{paper_name}/{timestamp}/routing_text_elements.json` - Text elements with references
  - `output/{paper_name}/{timestamp}/routing_readable.md` - Human-readable output

**Recommendation:** Use for testing single files, debugging extraction quality.

---

**Option B: Batch Processing**

You can create a batch script or use a loop:

```bash
#!/bin/bash
# Batch parse all PDFs

for pdf in files/organized_pdfs/*.pdf; do
    echo "Processing: $pdf"
    python scripts/parse_single_pdf.py "$pdf" --mode routing
done
```

---

### Step 5: Save to Database

**File:** `database/ingest.py` (the unified script we just created!)
**Mode:** Both single-file and batch supported

**Batch mode (recommended for production):**

```bash
# Ingest all PDFs
python database/ingest.py --pdf-dir files/organized_pdfs

# Or all XMLs
python database/ingest.py --xml-dir files/organized_xmls
```

**Single file mode (for testing):**

```bash
python database/ingest.py \
    --pmcid PMC1448691 \
    --pdf files/organized_pdfs/PMC1448691_his_2369.pdf \
    --pdffigures-json out/data/PMC1448691_his_2369.json
```

**What it does:**
- Extracts text from PDF (if not pre-parsed)
- Auto-detects figure/table references in text
- Loads PDFFigures2 JSON data
- Saves to PostgreSQL:
  - Text elements with hierarchical paths
  - Figures with captions and image paths
  - Tables with bounding boxes
  - Junction table entries linking text to figures/tables

**Recommendation:** Run in batch mode for production, single-file for testing.

---

## About Step 3: Masking PDFs

**Current Status:** Your codebase has `parsers/pdf_parsers/table_masker.py` but it's **not actively used** in the pipeline.

### What Masking Would Do:

The `TableMasker` class can create masked PDFs where table regions are blanked out (white rectangles), so text extractors skip them. This is useful when:
- Text extractors (like Docling) struggle with tables
- You want to extract tables separately with OCR
- You need clean narrative text without table noise

### Current Approach (No Masking):

Your `EnsemblePDFParser` uses **intelligent routing** instead:
- Detects pages with tables
- Routes those pages to Docling (good at tables)
- Routes narrative pages to PyMuPDF4LLM (fast)
- Routes image-heavy pages to Marker (high quality)

This is **better than masking** because:
- ✅ Tables are still extracted and included
- ✅ No need for separate OCR step
- ✅ Maintains document structure
- ✅ Faster processing

### If You Want to Use Masking:

Only use masking if you're getting poor table quality. To enable:

1. Detect tables with PDFFigures2 (already done in Step 2)
2. Create masked PDF:
   ```python
   from parsers.pdf_parsers.table_masker import TableMasker

   masker = TableMasker()
   masked_pdf = masker.mask_tables(
       pdf_path='original.pdf',
       table_bboxes=table_coords_from_pdffigures2
   )
   ```
3. Extract text from masked PDF
4. Extract tables separately with OCR

**Recommendation:** Stick with your current routing approach. Only add masking if you see table extraction issues.

---

## Recommended Approach: One-at-a-Time vs. Batch

### **Batch Processing (RECOMMENDED)**

**Pros:**
- ✅ Much faster (parallel processing, less overhead)
- ✅ Resume-safe (scripts skip already-processed files)
- ✅ Better for production
- ✅ Easier monitoring with progress bars

**Cons:**
- ❌ Harder to debug individual failures
- ❌ Need to re-run batch to fix one file

**When to use:** Production runs, processing many papers

---

### **One-at-a-Time Processing**

**Pros:**
- ✅ Easier to debug specific papers
- ✅ Can customize per-paper (different modes, settings)
- ✅ Better for quality testing

**Cons:**
- ❌ Much slower (startup overhead per file)
- ❌ More manual work

**When to use:** Testing, debugging, quality checks on specific papers

---

## Hybrid Approach (BEST OF BOTH WORLDS)

**Recommended workflow:**

1. **Batch Steps 1-3** (extraction, PDFFigures2, image extraction)
   - These are deterministic and rarely fail
   - Fast and resume-safe

2. **One-at-a-time Step 4** (text extraction) for **first 5-10 papers**
   - Verify extraction quality
   - Check figure/table detection
   - Adjust parser settings if needed

3. **Batch Steps 4-5** (text + database) for **remaining papers**
   - Once you're confident in quality
   - Use batch mode for speed

4. **One-at-a-time re-processing** for **any failures**
   - Check `out/failed_extractions.txt`
   - Debug and fix individually

---

## Complete Workflow Example

### Initial Setup (Once)

```bash
# Setup database (creates all tables)
python database/setup_db.py

# That's it! The migration scripts are only needed if you have an existing database.
```

### Batch Processing (Production)

```bash
# Step 1: Extract from tarballs
cd file-selector
python tarball_extractor.py
python pdf_organizer.py
cd ..

# Step 2: Run PDFFigures2 (creates JSON with bounding boxes)
bash scripts/run_pdffigures_resume.sh

# Step 3: Extract figure/table images
python scripts/extract_figures_tables.py

# Step 4 & 5: Extract text and ingest to database (combined)
python database/ingest.py --pdf-dir files/organized_pdfs

# Done! Check results
psql -U postgres -d nlp_histo -c "
  SELECT pmcid, COUNT(DISTINCT t.id) as texts, COUNT(DISTINCT f.id) as figs
  FROM documents d
  LEFT JOIN text_elements t ON d.id = t.document_id
  LEFT JOIN figures f ON d.id = f.document_id
  GROUP BY pmcid;
"
```

### Single File Testing

```bash
# Test on one paper
PDF="files/organized_pdfs/PMC1448691_his_2369.pdf"

# Steps 1-3 already done in batch mode above

# Step 4: Parse single PDF
python scripts/parse_single_pdf.py "$PDF" --mode routing

# Step 5: Ingest single file
python database/ingest.py \
    --pmcid PMC1448691 \
    --pdf "$PDF"

# Verify
python -c "
from database.ingest_with_references import query_reference_examples
query_reference_examples('PMC1448691')
"
```

---

## Monitoring Progress

### Check PDFFigures2 Progress

```bash
# Total PDFs
ls files/organized_pdfs/*.pdf | wc -l

# Processed (have JSON)
ls out/data/*.json | wc -l

# Failed
cat out/failed_extractions.txt | wc -l
```

### Check Database Progress

```bash
# Count documents in database
psql -U postgres -d nlp_histo -c "SELECT COUNT(*) FROM documents;"

# Check ingestion quality
psql -U postgres -d nlp_histo -c "
  SELECT
    COUNT(DISTINCT d.id) as docs,
    AVG(text_count) as avg_texts,
    AVG(fig_count) as avg_figs,
    AVG(table_count) as avg_tables
  FROM documents d
  LEFT JOIN LATERAL (
    SELECT COUNT(*) as text_count FROM text_elements WHERE document_id = d.id
  ) t ON true
  LEFT JOIN LATERAL (
    SELECT COUNT(*) as fig_count FROM figures WHERE document_id = d.id
  ) f ON true
  LEFT JOIN LATERAL (
    SELECT COUNT(*) as table_count FROM tables WHERE document_id = d.id
  ) tb ON true;
"
```

---

## Troubleshooting

### PDFFigures2 Fails

```bash
# Check Java version
java -version  # Should be Java 11

# Set Java 11
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@11/11.0.29/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
```

### Text Extraction Quality Issues

```bash
# Try different parser modes
python scripts/parse_single_pdf.py file.pdf --mode routing  # Default
python scripts/parse_single_pdf.py file.pdf --mode fallback  # Legacy

# Check parser availability
python -c "
from parsers.pdf_parsers.ensemble_parser import EnsemblePDFParser
parser = EnsemblePDFParser()
print(parser.get_available_parsers())
"
```

### Database Ingestion Fails

```bash
# Check for missing schema
python database/migrate_add_tables_and_references.py

# Re-ingest with force flag
python database/ingest.py --pmcid PMC1234567 --pdf file.pdf --force

# Check logs
tail -f database_ingestion.log
```

---

## Quick Reference

| Step | File | Mode | Time (100 PDFs) |
|------|------|------|-----------------|
| 1. Extract tarballs | `file-selector/tarball_extractor.py` | Batch | ~5 min |
| 2. PDFFigures2 | `scripts/run_pdffigures_resume.sh` | Batch | ~30 min |
| 3. Extract images | `scripts/extract_figures_tables.py` | Batch | ~10 min |
| 4. Parse PDFs | `scripts/parse_single_pdf.py` OR batch loop | Both | ~2-3 hours |
| 5. Ingest DB | `database/ingest.py` | Both | ~30 min |

**Total time for 100 PDFs:** ~3-4 hours (mostly Step 4)

---

## Summary

**Your question: One file at a time or all files at each stage?**

**Answer: Batch processing at each stage is recommended** because:

1. ✅ **Faster** - Less overhead, better parallelization
2. ✅ **Resume-safe** - Scripts skip already-processed files
3. ✅ **Easier** - Single command per stage
4. ✅ **Better monitoring** - Track overall progress

**Exception:** Use one-at-a-time for:
- Initial quality testing (first 5-10 files)
- Debugging specific failures
- Papers that need special handling

**About masking:** Your current routing approach is better than masking. Only add masking if you see table extraction issues.
