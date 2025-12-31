# Caption Assignment Fix - Complete Summary

## Problem Statement

The original pipeline had two major issues with caption assignment:

1. **Caption Mixing**: Figure captions were being assigned to tables and vice versa due to weak string matching
2. **Missing TEXT Captions**: TEXT elements containing table/figure captions (like "Table 6. Anaplastic large-cell lymphoma") were being ignored

## Root Causes

### Issue 1: Weak String Matching
The original code checked if words like "table" or "figure" appeared **anywhere** in a caption:
```python
if 'table' in (caption.get('text') or '').lower():
    # Assign to table
```

This caused problems like:
- "Figure 3 shows the same results as Table 1" → assigned to Table 1 ❌
- Any caption mentioning both figures and tables could be assigned incorrectly

### Issue 2: Ignoring TEXT Elements
Docling sometimes extracts table/figure captions as TEXT elements instead of CAPTION elements. The original code only looked at CAPTION elements adjacent to TABLE/PICTURE elements, missing standalone captions.

## The Complete Fix

### Changes Made to `scripts/complete_pipeline.py`

#### 1. Robust Caption Number Extraction
```python
def extract_number_from_caption(text, element_type):
    """Extract number from caption that STARTS with Table/Figure."""
    if element_type == 'table':
        pattern = r'^\s*Table\s+(\d+[A-Za-z]?)'  # Must start with "Table"
    else:
        pattern = r'^\s*(?:Figure|Fig\.?)\s+(\d+[A-Za-z]?)'

    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None
```

**Key improvement**: Only matches captions that **start** with the keyword, preventing false matches.

#### 2. Strict Caption Validation
```python
def is_valid_caption_for_element(caption_text, expected_number, element_type):
    """Verify caption number matches expected ID."""
    extracted_number = extract_number_from_caption(caption_text, element_type)

    if extracted_number:
        return extracted_number == expected_number  # Must match!

    return False  # No number = reject
```

**Key improvement**: Verifies the extracted number matches the expected table/figure ID.

#### 3. Duplicate Prevention
```python
used_caption_indices = set()  # Track used captions by index
seen_table_ids = set()        # Track seen table IDs
seen_figure_ids = set()       # Track seen figure IDs
```

**Key improvement**: Ensures each caption is only used once, even if it matches multiple patterns.

#### 4. TEXT Element Caption Extraction (NEW!)
```python
# Process TEXT/CAPTION elements that contain table/figure captions
if elem_type in ['TEXT', 'CAPTION'] and i not in used_caption_indices:
    text = elem.get('text', '')

    # Check for table caption
    table_num = extract_number_from_caption(text, 'table')
    if table_num and table_num not in seen_table_ids:
        table_data.append({
            'table_id': table_num,
            'caption': text,
            'page': page,
            'bbox': elem.get('bbox'),
            'type': 'text_caption'  # No physical table detected
        })
        seen_table_ids.add(table_num)
        used_caption_indices.add(i)
```

**Key improvement**: Extracts table/figure captions from TEXT elements even when no TABLE/PICTURE element is detected.

#### 5. Smart ID Assignment
Tables and figures now get the next available ID, skipping any already claimed by TEXT captions:
```python
# Find next available table ID
table_id = None
test_counter = 1
while True:
    test_id = str(test_counter)
    if test_id not in seen_table_ids:
        table_id = test_id
        break
    test_counter += 1
```

**Key improvement**: Handles cases where "Table 6" appears as TEXT but no TABLE element was detected.

## Test Results

All test suites pass ✅:

### Test 1: Caption Number Extraction (8/8)
- ✓ "Table 1: Patient demographics" → extracts "1"
- ✓ "Figure 3 shows Table 1" → extracts "3" (not "1"!)
- ✓ "See Table 3" → extracts None (doesn't start with "Table")

### Test 2: Caption Validation (8/8)
- ✓ "Table 1: Data" matches table_id="1" ✅
- ✓ "Table 2: Data" does NOT match table_id="1" ❌
- ✓ "Figure 3 shows Table 1" matches figure_id="3", NOT table_id="1"

### Test 3: Duplicate Prevention
- ✓ Caption at index 1 can only be used once
- ✓ Subsequent tables/figures cannot reuse it

### Test 4: TEXT Element Extraction (NEW)
- ✓ Extracts "Table 1" from TEXT element
- ✓ Extracts "Table 6" from TEXT element
- ✓ Extracts "Figure 2" from TEXT element
- ✓ Correctly identifies table IDs: {1, 6}

## How to Verify the Fix

### 1. Run Tests
```bash
python scripts/test_caption_fix.py
```

Expected output: `✅ ALL TEST SUITES PASSED`

### 2. Process a Sample PDF
```bash
python scripts/complete_pipeline.py \
  --pdf files/organized_pdfs/PMC1448691_his_2369.pdf \
  --pmcid PMC1448691
```

### 3. Check Output Files

#### Tables
```bash
cat files/tables/PMC1448691_tables.json
```

Expected: Each table has correct caption, no figure captions mixed in.

Example:
```json
[
  {
    "table_id": "1",
    "caption": "Table 1. B-cell cutaneous lymphoma. Learning from the Workshop",
    "page": 2,
    "type": "caption"
  },
  {
    "table_id": "6",
    "caption": "Table 6. Anaplastic large-cell lymphoma and Its differential diagnosis",
    "page": 20,
    "type": "text_caption"  # Note: This was extracted from TEXT, not CAPTION element!
  }
]
```

#### Figures
```bash
cat files/figures/PMC1448691_figures.json
```

Expected: Each figure has correct caption, no table captions mixed in.

### 4. Database Verification (if using --db-ingest)
```python
from database import get_db_connection, Figure, Table

db = get_db_connection()
with db.session_scope() as session:
    # Check tables
    tables = session.query(Table).filter_by(document_id=<doc_id>).all()
    for t in tables:
        print(f"Table {t.table_id}: {t.caption_text[:60]}")

    # Check figures
    figures = session.query(Figure).filter_by(document_id=<doc_id>).all()
    for f in figures:
        print(f"Figure {f.figure_id}: {f.caption_text[:60]}")
```

Expected: No table captions in figures list, no figure captions in tables list.

## Edge Cases Handled

1. **Interleaved tables/figures**: "Table 1", "Figure 1", "Table 2", "Figure 2" → All correctly assigned
2. **Non-sequential IDs**: "Table 1", "Table 6" (missing 2-5) → Both extracted with correct IDs
3. **Alphanumeric IDs**: "Table 2A", "Figure 3B" → Correctly extracted
4. **TEXT vs CAPTION**: Works with both element types
5. **Caption-only tables**: "Table 6" as TEXT with no TABLE element → Creates table entry with type="text_caption"
6. **Cross-references in captions**: "Figure 3 shows Table 1" → Assigned only to Figure 3

## Benefits

1. **No more caption mixing**: Tables get table captions, figures get figure captions
2. **Complete extraction**: TEXT element captions no longer ignored
3. **Duplicate prevention**: Each caption used exactly once
4. **Robust matching**: Only matches captions that start with the keyword
5. **Flexible ordering**: Handles tables and figures in any order
6. **Better metadata**: `type` field indicates whether caption came from TEXT or CAPTION element

## Migration Notes

If you have existing data with mixed captions:

1. Re-run the pipeline with `--force` flag to re-ingest:
   ```bash
   python scripts/complete_pipeline.py \
     --pdf-dir files/organized_pdfs \
     --db-ingest \
     --force
   ```

2. Or for single files:
   ```bash
   python scripts/complete_pipeline.py \
     --pdf files/organized_pdfs/PMC1234567.pdf \
     --pmcid PMC1234567 \
     --force \
     --db-ingest
   ```

The new logic will correctly assign all captions without mixing.
