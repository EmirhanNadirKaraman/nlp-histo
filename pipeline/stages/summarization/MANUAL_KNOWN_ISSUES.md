# Manual Known Issues — Summarization Pipeline

Issues logged manually. Read before editing any pipeline stage.  
Each issue has a **severity**, the **file:location** to touch, a description, and a suggested fix.

---

## Bugs (code is wrong today)

---

## High-Impact Accuracy Risks

---

## Medium-Impact Accuracy Risks

### ACC-3 — `select_predicate()` sends all candidates to LLM unnecessarily
**Severity:** Medium  
**File:** `canonicalize_stage.py` (`_select_predicate`)  
**Symptom:** All findings in a direction bin are sent to an LLM to pick a representative
predicate, adding latency and cost. The LLM selection is not demonstrably better than
deterministic selection.  
**Fix:** Just select the finding with the highest grounding score. Remove the LLM call.

### ACC-4 — RELATE pair truncation discards high-value pairs
**Severity:** Medium  
**File:** `relate_stage.py:294–299`  
**Symptom:** When eligible pairs exceed `MAX_PAIRS`, the first 500 by index order are kept.
Rules from later chunks (Discussion/Conclusion) are systematically dropped.  
**Fix:** Remove the truncation cap, or if a cap is needed, sort by combined grounding score
descending before truncating.

### BUG-2 — `filter_findings` does not write `grounding_score` back onto findings
**Severity:** Medium  
**File:** `grounding_filter.py` (`filter_findings`)  
**Symptom:** `filter_findings` runs NLI and drops low-scoring findings but never writes
`grounding_score` onto the kept ones. Any downstream code that reads `grounding_score` will
see `None`. `filter_findings_with_scores` is the correct replacement and the runner comment
says to use it, but `filter_findings` is still present and callable.  
**Fix:** Remove `filter_findings` or have it delegate to `filter_findings_with_scores` and
discard the dropped list.

### BUG-3 — `_NLI_PIPE_CACHE` key ignores `batch_size` and `device`
**Severity:** Low  
**File:** `grounding_filter.py` (`GroundingFilter._pipe`)  
**Symptom:** The cache is keyed only on `model_name`. Two `GroundingFilter` instances with
different `batch_size` or `device` settings will share the first instance's pipeline silently.  
**Fix:** Key the cache on `(model_name, device, batch_size)`.

### DES-1 — Intra-paper corpus relations not recomputed when a paper is reprocessed
**Severity:** Medium  
**File:** `corpus_relate.py` (`_incremental_gate`)  
**Symptom:** The incremental gate rejects pairs where both rules come from the same paper.
When a paper is reprocessed, its intra-paper corpus relations are never updated in the DB.  
**Fix:** After computing cross-paper pairs, run a second pass over `new_rules` without the
XOR restriction to produce intra-paper pairs, then include them in `_replace_for_pmcid`.

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

### DES-2 — `compute_scope()` returns a single enum value — cannot express conflicted + multi_study
**Severity:** Low  
**File:** `canonicalize_stage.py` (`_compute_scope`), `models.py` (`CanonicalScopeEnum`)  
**Symptom:** A rule that is both `conflicted` and `multi_study` can only hold one value.
`conflicted` wins by priority, discarding study-coverage information.  
**Fix:** Return two separate values: `is_conflicted: bool` and
`study_coverage: Literal["single_study", "multi_study", "unknown"]`.
Requires a schema migration.

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

| ID | Severity | Stage | One-line description |
|----|----------|-------|----------------------|
| BUG-2 | Medium | GROUNDING | `filter_findings` never writes `grounding_score` — downstream sees `None` |
| BUG-3 | Low | GROUNDING | `_NLI_PIPE_CACHE` key ignores `batch_size` and `device` |
| ACC-3 | Medium | CANONICALIZE | `select_predicate()` uses LLM — deterministic highest-grounding is sufficient |
| ACC-4 | Medium | RELATE | Pair truncation is index-ordered, drops high-value Discussion/Conclusion rules |
| DES-1 | Medium | CORPUS RELATE | Intra-paper corpus relations not recomputed on reprocess |
| DES-2 | Low | CANONICALIZE | `compute_scope()` single enum loses conflicted + multi_study combination |
| DES-3 | Low | RESOLVE | Scoring weights hand-tuned — no gold label dataset to validate against |
| DES-4 | Low | NORMALIZE | `best_cui` returns one CUI — top-N (3–5) valid matches would improve coverage |
