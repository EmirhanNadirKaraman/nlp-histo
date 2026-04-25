# Manual Known Issues — Summarization Pipeline

Issues logged manually. Read before editing any pipeline stage.  
Each issue has a **severity**, the **file:location** to touch, a description, and a suggested fix.

---

## Bugs (code is wrong today)

---

## High-Impact Accuracy Risks

---

## Medium-Impact Accuracy Risks

### ~~ACC-4 — RELATE pair truncation discards high-value pairs~~ ✓ Fixed
**File:** `relate_stage.py`  
Truncation now sorts eligible pairs by combined `mean_grounding_score` descending before applying the cap.

### ~~BUG-2 — `filter_findings` does not write `grounding_score` back onto findings~~ ✓ Fixed
**File:** `grounding_filter.py`  
`filter_findings` now delegates to `filter_findings_with_scores` and discards the dropped list.

### ~~BUG-3 — `_NLI_PIPE_CACHE` key ignores `batch_size` and `device`~~ ✓ Fixed
**File:** `grounding_filter.py`  
Cache now keyed on `(model_name, device, batch_size)`.

### ~~DES-1 — Intra-paper corpus relations not recomputed when a paper is reprocessed~~ ✓ Fixed
**File:** `corpus_relate.py` (`relate_incremental`)  
`relate_incremental` now runs a second pass over `new_rules` only (using `_should_compare_cross_paper`)
to produce intra-paper pairs, then combines them with cross-paper pairs before `_replace_for_pmcid`.

---

## Lower-Impact / Design Choices

### DES-4 — `best_cui` returns only one CUI — loses valid secondary matches
**Severity:** Low  
**File:** `umls_utils.py` (`best_cui`), `normalize_stage.py` (`_umls_canonical`)  
**Symptom:** `best_cui` picks a single top non-junk CUI and discards all remaining valid
candidates. An entity like "CD30" may have 3–5 valid UMLS hits above threshold (e.g. the
protein, the gene, the receptor ligand). Only the highest-scoring one reaches downstream
stages; the others are silently dropped.  
**Fix:** Return the top-N (e.g. 3 or 5) non-junk CUIs sorted by score descending instead of
a single CUI. Downstream impact: `_umls_canonical` in `normalize_stage.py` currently returns
`str | None`; changing to `list[str]` requires updating `NormalFinding.cui` (models.py),
GROUP grouping key, CANONICALIZE predicate selection, and any DB column that stores a single
CUI. Do not attempt without a schema migration.

### ~~DES-2 — `compute_scope()` returns a single enum value — cannot express conflicted + multi_study~~ ✓ Fixed
**File:** `canonicalize_stage.py`, `models.py`, `database/models.py` + Alembic `0009`  
`CanonicalScopeEnum` removed. `CanonicalRule`/`FinalRule` now carry `is_conflicted: bool` and
`study_coverage: Literal["single_study", "multi_study", "unknown"]` independently. DB columns
`canonical_scope` → `is_conflicted` + `study_coverage` (same for corpus relations a/b).
Migration `0009` data-migrates existing rows.

### DES-3 — RESOLVE scoring weights are hand-tuned, not empirically validated
**Severity:** Low  
**File:** `resolve_stage.py:48–63` (constants)  
**Symptom:** Weights were chosen to produce intuitively reasonable distributions, not fit
to a gold-labeled dataset.  
**Fix options:**
- Proxy labels from citation count of supporting papers.
- LLM-as-judge confidence rating.
- Agreement across papers as weak supervision proxy.  
See iPad notes for more details.

---

## Summary Table

| ID | Severity | Stage | One-line description | Status |
|----|----------|-------|----------------------|--------|
| BUG-2 | Medium | GROUNDING | `filter_findings` never writes `grounding_score` — downstream sees `None` | ✓ Fixed |
| BUG-3 | Low | GROUNDING | `_NLI_PIPE_CACHE` key ignores `batch_size` and `device` | ✓ Fixed |
| ACC-4 | Medium | RELATE | Pair truncation is index-ordered, drops high-value Discussion/Conclusion rules | ✓ Fixed |
| DES-1 | Medium | CORPUS RELATE | Intra-paper corpus relations not recomputed on reprocess | ✓ Fixed |
| DES-2 | Low | CANONICALIZE | `compute_scope()` single enum loses conflicted + multi_study combination | ✓ Fixed |
| DES-3 | Low | RESOLVE | Scoring weights hand-tuned — no gold label dataset to validate against | Open |
| DES-4 | Low | NORMALIZE | `best_cui` returns one CUI — top-N (3–5) valid matches would improve coverage | Open |
