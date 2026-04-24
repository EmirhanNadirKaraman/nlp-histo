# Manual Known Issues — Summarization Pipeline

Issues logged manually. Read before editing any pipeline stage.  
Each issue has a **severity**, the **file:location** to touch, a description, and a suggested fix.

---

## Bugs (code is wrong today)

### BUG-1 — UMLS hit selection checks quality after picking the top scorer
**Severity:** High  
**File:** `umls_utils.py` (`best_cui`)  
**Symptom:** When multiple CUI candidates score above `UMLS_THRESHOLD = 0.85`, the code picks
the single highest-scoring one first, then checks whether it is junk. If it is junk, it is
discarded — but the next-best candidate (which may be valid) is never tried.  
**Fix:** Filter out junk candidates before ranking. Select the highest-scoring non-junk CUI,
not the highest-scoring candidate regardless of quality.

---

## High-Impact Accuracy Risks

---

## Medium-Impact Accuracy Risks

### ~~ACC-2~~ — RESOLVED
`_split_windows()` now uses spaCy sentencizer + greedy packing (≤1800 chars ≈ 450 tokens)
with 1-sentence overlap. Oversized sentences split further by `;` then `,`.

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

### DES-1 — Intra-paper corpus relations not recomputed when a paper is reprocessed
**Severity:** Medium  
**File:** `corpus_relate.py` (`_incremental_gate`)  
**Symptom:** The incremental gate rejects pairs where both rules come from the same paper.
When a paper is reprocessed, its intra-paper corpus relations are never updated in the DB.  
**Fix:** After computing cross-paper pairs, run a second pass over `new_rules` without the
XOR restriction to produce intra-paper pairs, then include them in `_replace_for_pmcid`.

---

## Lower-Impact / Design Choices

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
| BUG-1 | High | NORMALIZE | UMLS junk check runs after top-scorer selection, not before |
| ~~ACC-2~~ | ~~Medium~~ | GROUNDING | RESOLVED — sentence-aware windowing with ; , fallback |
| ACC-3 | Medium | CANONICALIZE | `select_predicate()` uses LLM — deterministic highest-grounding is sufficient |
| ACC-4 | Medium | RELATE | Pair truncation is index-ordered, drops high-value Discussion/Conclusion rules |
| DES-1 | Medium | CORPUS RELATE | Intra-paper corpus relations not recomputed on reprocess |
| DES-2 | Low | CANONICALIZE | `compute_scope()` single enum loses conflicted + multi_study combination |
| DES-3 | Low | RESOLVE | Scoring weights hand-tuned — no gold label dataset to validate against |
