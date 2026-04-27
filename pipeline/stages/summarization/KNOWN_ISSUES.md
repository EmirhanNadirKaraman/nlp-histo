# Known Issues — Summarization Pipeline

Compiled 2026-04-15. Read before editing any pipeline stage.  
Each issue has a **severity**, the **file:location** to touch, a description, and a suggested fix.

---

## Bugs (code is wrong today)

---

## High-Impact Accuracy Risks

### ACC-2 — `relation_type` LLM variance splits the same fact into different groups
**Severity:** High  
**File:** `map_stage.py` / `prompts.py` (extraction), `group_stage.py:55–56` (grouping key)  
**Symptom:** "CD30 was expressed" can legitimately be tagged `expression` or `has_feature`
depending on which model or chunk sees it. Same physical fact → different `relation_type` →
different `FindingGroup` → never deduplicated or compared in RELATE.  
**Fix options:**
- Add an explicit normalization step in NORMALIZE that collapses `has_feature` → `expression`
  for findings whose `outcome_entity` is a known biomarker (check against `_SYNONYMS` keys).
- Or relax the RELATE gate to compare across `has_feature` / `expression` when subject and
  outcome match.
- Minimum: count how often the same `(subject, outcome)` pair appears in both `has_feature`
  and `expression` groups in a typical paper output.


### ACC-6 — `unclear` direction findings inflated into the largest direction bin in CANONICALIZE
**Severity:** High  
**File:** `canonicalize_stage.py` (`_split_by_direction`)  
**Symptom:** When a group has e.g. 3 positive + 1 negative + 2 unclear findings, all unclear go
into the positive bin. This inflates the positive count, can make the negative bin appear
negligible, and masks real directional ambiguity in the canonical rule.  
**Fix options:**
- Create a separate `unclear` CanonicalRule for unclear-direction findings (analogous to how
  positive and negative get separate rules).
- Or: exclude unclear findings from direction bins entirely and set `is_conflicted=True` on
  the group when unclear count is significant relative to total.
- Minimum: store `direction_counts` on `CanonicalRule` so downstream stages can see the raw
  split, not just the bin they were assigned to.

### DES-1 — RESOLVE ignores cross-paper relations when scoring FinalRules
**Severity:** High  
**File:** `runner.py:381–388` (RESOLVE call), `resolve_stage.py`  
**Symptom:** `corpus_relate_incremental` runs before per-paper RELATE and writes cross-paper
relations to DB, but RESOLVE only receives per-paper `Relation[]` (line 382). A rule
contradicted by 5 other papers scores identically to one with no cross-paper contradictions.
Cross-paper support is also ignored — a rule backed by 10 papers gets no score boost over
one backed by 1.  
**Fix:** Pass corpus relations into RESOLVE alongside per-paper relations. Cross-paper
CONTRADICTs should penalise the rule score; cross-paper SUPPORTs should boost it.
Requires resolving DES-7 first (score scale mismatch between single-paper and multi-paper runs).

---

## Medium-Impact Accuracy Risks

### ACC-10 — Cross-paper gate does not check outcome for non-expression rules
**Severity:** Medium  
**File:** `corpus_relate.py:93–131` (`_should_compare_cross_paper`)  
**Symptom:** For non-expression rules: if either rule lacks a CUI, subject gating is skipped
entirely. For outcome: no gating at all (only done for expression rules). As a result,
"MGA has feature X" and "DLBCL has feature Y" with no CUIs and the same category + relation_type
pass the gate. NLI then runs on unrelated predicate texts and may fire spurious SUPPORT or CONTRADICT.  
**Fix:** For non-expression rules, also gate on outcome_entity when both rules have a CUI; and
add a fallback `_norm_outcome()` string comparison for outcome when CUIs are absent.

### ACC-11 — `infer_direction()` keyword heuristic incorrect for complex clinical negation
**Severity:** Medium  
**File:** `normalize_stage.py:275–291` (`infer_direction`)  
**Symptom:** The heuristic uses prefix/substring matching without syntax awareness.
Incorrect cases: "not uncommon" → negative (wrong; means positive), "no significant difference"
→ negative (direction-neutral claim about an association), "non-specific" → negative (wrong).  
**Fix options:**
- Add a list of known false-negative trigger phrases (e.g. "not uncommon", "not rare",
  "not infrequent") that should return `unclear` or `positive` instead.
- Longer term: replace with a lightweight NLI-based direction classifier or use the LLM's
  direction field without heuristic overriding.
- The heuristic should only fire on `unclear` findings (it does today) — do not expand its scope.

---

## Lower-Impact / Design Choices

### ACC-12 — `_nli_scores()` in RELATE does not use the sliding window
**Severity:** Low (predicate text from CANONICALIZE is almost always short)  
**File:** `relate_stage.py:62–73` (`_nli_scores`)  
**Symptom:** Unlike `grounding_filter._score_pairs()`, the RELATE NLI function does not split
long premises into overlapping 400-char windows. Predicate text from CANONICALIZE is typically
short, so this rarely fires — but if a rule has a long `predicate_text` (e.g. a complex
multi-clause clinical statement), it will be silently truncated at 512 tokens.  
**Fix:** Apply the same `_split_windows` approach from `grounding_filter.py`. Since the function
is shared, consider moving `_split_windows` and `_score_pairs` to a shared utilities module and
importing from both.

### DES-7 — RESOLVE two-mode scoring produces non-comparable scales
**Severity:** Low  
**File:** `resolve_stage.py`  
**Symptom:** Single-paper runs always use the absent-relations formula (grounding weight 0.80);
multi-paper corpus runs use the relations-present formula (grounding weight 0.60). Final scores
from single-paper and multi-paper runs are not on the same scale. Any downstream threshold
applied uniformly across both types is miscalibrated.  
**Fix:** Document the two modes clearly in output JSON and add a `scoring_mode` field to
`FinalRule`. Consider normalizing scores to a common range post-hoc when merging outputs.

### DES-8 — RESOLVE scoring weights are hand-tuned, not empirically validated
**Severity:** Low  
**File:** `resolve_stage.py:48–63` (constants)  
**Symptom:** Weights (`_GROUNDING_WEIGHT=0.60`, `_SUPPORT_BOOST_PER_REL=0.08`,
`_CONTRADICT_PEN_PER_REL=0.15`, etc.) were chosen to produce intuitively reasonable score
distributions, not derived from a gold-labeled dataset. There is no evaluation harness
verifying these weights produce better rankings than alternatives.  
**What would be needed:** ~200–500 rules labeled by a pathologist with confidence tiers
(high / medium / low), covering rules with varying support_count, contradict_count,
finding_count, and study_coverage. Weights could then be fit to minimise ranking error.  
**Without expert labelers, options are:**
- Proxy labels from citation count of supporting papers
- LLM-as-judge confidence rating (noisy but cheap)
- Weak supervision: treat `multi_study` + high `finding_count` as pseudo-high-confidence,
  `single_study` + low grounding as pseudo-low — at minimum verifies formula direction is correct  
**Fix:** Build a small gold set using one of the above proxies and run a grid search or
linear regression over the weight constants.

### DES-9 — `_best_scope()` uses scope from highest-grounding finding only
**Severity:** Low  
**File:** `normalize_stage.py:358–364` (`_best_scope`)  
**Symptom:** When merging a cluster of findings, the representative scope is taken from the
single highest-grounding finding. A well-grounded but scope-sparse finding (no fields extracted)
wins over a lower-grounding finding that has full scope populated.  
**Fix:** Aggregate scope fields across all findings in the cluster: for each scope field,
take the first non-None value across the cluster (or the majority value).

---

## Summary Table

| ID | Severity | Stage | One-line description |
|----|----------|-------|----------------------|
| ACC-2 | High | MAP/GROUP | `relation_type` variance splits same fact into different groups |
| ACC-6 | High | CANONICALIZE | `unclear` direction findings inflated into largest bin |
| DES-1 | High | RESOLVE | Cross-paper relations ignored in FinalRule scoring |
| ACC-10 | Medium | CORPUS RELATE | Cross-paper gate skips outcome gating for non-expression rules |
| ACC-11 | Medium | NORMALIZE | `infer_direction()` wrong on complex clinical negation |
| ACC-12 | Low | RELATE | `_nli_scores()` lacks sliding window — rarely fires in practice |
| DES-7 | Low | RESOLVE | Two scoring modes produce non-comparable final score scales |
| DES-8 | Low | RESOLVE | Scoring weights hand-tuned, not empirically validated |
| DES-9 | Low | NORMALIZE | `_best_scope()` ignores scope from lower-grounding findings |
