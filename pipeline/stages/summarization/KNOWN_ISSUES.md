# Known Issues — Summarization Pipeline

Compiled 2026-04-15. Read before editing any pipeline stage.  
Each issue has a **severity**, the **file:location** to touch, a description, and a suggested fix.

---

## Bugs (code is wrong today)

### BUG-1 — `_compute_scope()` reads group-level `direction_counts` after bin split
**Severity:** High  
**File:** `canonicalize_stage.py:83–110` (`_compute_scope`)  
**Symptom:** After `_split_by_direction` produces a positive-only bin, `_compute_scope` still
reads `group.direction_counts` — the group-level tally that includes negative/unclear findings.
If the original group had mixed directions, `conflicted` fires for every bin, even bins that
are internally direction-consistent. A positive-only bin from a positive+negative group is
always marked `conflicted`.  
**Fix:** Pass bin-level `member_nfs` direction counts into `_compute_scope` instead of relying
on `group.direction_counts`. Compute conflicted from the bin, not the group.

---

## High-Impact Accuracy Risks

### ACC-1 — Chunk boundary finding loss
**Severity:** High  
**File:** `map_stage.py:583–587` (`_make_chunks`)  
**Symptom:** A finding that spans the junction of two 10-sentence chunks is either extracted as
a fragment by both, or missed by both. No cross-chunk context is available to the LLM.
Affects complex multi-sentence causal or prognostic claims most.  
**Fix options:**
- Add a configurable sentence overlap (e.g. 2 sentences shared between adjacent chunks).
- Or use a sliding window with stride < chunk_size.
- Minimum: log how many MAP findings cite sentence IDs from only the last or first position
  of a chunk (proxy for boundary fragments).

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

### ACC-3 — RELATE subject exact-match rejects normalization-near-misses
**Severity:** High  
**File:** `relate_stage.py:168` (`_should_compare`), line `191` for the subject comparison  
**Symptom:** After normalization, surface differences that survive (e.g. "ALK-positive ALCL"
vs "ALK+ ALCL") cause `subject_mismatch` rejection and the pair never reaches NLI.
`_norm_outcome()` was applied to outcome comparison but not to subject comparison.  
**Fix:** Apply `_norm_outcome()` (lowercase + strip) to subject comparison too:
```python
if a.subject_entity.strip().lower() != b.subject_entity.strip().lower():
    return False, "subject_mismatch"
```
Longer term: apply the same synonym dict normalization used in `NormalizeStage._norm()`.

### ACC-4 — `verbatim_support` is LLM-generated, often a paraphrase
**Severity:** High  
**File:** `prompts.py` (MAP prompt), `grounding_filter.py:200–272`  
**Symptom:** The MAP prompt asks for an "exact quote" but LLMs paraphrase. The grounding filter
runs NLI on `(verbatim_support, claim)`. A paraphrased verbatim scores lower than the actual
source sentence would, causing real findings to be incorrectly dropped or scored low.
This depresses `mean_grounding_score` for genuinely grounded findings.  
**Fix options:**
- After MAP, search for the actual sentence in the source chunk that has the highest NLI
  entailment with the claim, and use that as the verified verbatim (replacing the LLM-supplied
  one). This makes grounding scores reflect real evidence, not LLM quoting quality.
- Or: run grounding against all sentences in the source chunk (not just the LLM quote) and
  take the max score.

### ACC-5 — CUI enrichment runs post-canonicalize, cannot fix grouping errors
**Severity:** High  
**File:** `entity_linker.py:62–97` (`enrich_rules_with_cuis`), `runner.py` (call site)  
**Symptom:** `enrich_rules_with_cuis` assigns `subject_cui` / `outcome_cui` to `CanonicalRule`
objects after canonicalize has already run. By that point, grouping is done — two `NormalFinding`s
with different surface strings (e.g. "Ki-67" vs "Ki67") that both failed UMLS lookup during
NORMALIZE will have landed in different `FindingGroup`s and produced separate `CanonicalRule`s
for the same real-world concept.  
**Fix:** Resolve CUIs during NORMALIZE, store them on `NormalFinding`, and key GROUP on CUI
when available (falling back to normalized string only when CUI is None). This requires
`normalize_entity()` to return `(canonical_name, cui)` instead of just `canonical_name`.

### ACC-6 — `unclear` direction findings inflated into the largest direction bin in CANONICALIZE
**Severity:** High  
**File:** `canonicalize_stage.py` (`_split_by_direction`)  
**Symptom:** When a group has e.g. 3 positive + 1 negative + 2 unclear findings, all unclear go
into the positive bin. This inflates the positive count, can make the negative bin appear
negligible, and masks real directional ambiguity in the canonical rule.  
**Fix options:**
- Create a separate `unclear` CanonicalRule for unclear-direction findings (analogous to how
  positive and negative get separate rules).
- Or: exclude unclear findings from direction bins entirely and flag the group as
  `canonical_scope=conflicted` when unclear count is significant relative to total.
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

### ACC-7 — Unbounded candidate list sent to LLM in CANONICALIZE
**Severity:** Medium  
**File:** `canonicalize_stage.py:289–293` (`_select_predicate`)  
**Symptom:** All findings in a direction bin are sent as candidates to the LLM — no cap. On
per-paper runs groups are typically small, but with cross-paper pooling a bin could have dozens
of candidates, causing unbounded token usage, latency, and cost per group.  
**Fix:** Cap at a configurable `max_candidates` (e.g. top-10 by grounding score). Findings
are already sorted descending by score (line 230–234), so truncation is safe.

### ACC-8 — Position bias in LLM predicate selection
**Severity:** Medium  
**File:** `canonicalize_stage.py:289–293` (`_select_predicate`)  
**Symptom:** Candidates are sorted descending by grounding score, so candidate #1 always has
the highest score. LLMs have known primacy bias and systematically favour items listed first.
The LLM will be nudged toward the highest-grounding candidate regardless of clinical
informativeness — partially defeating the purpose of using an LLM over the deterministic
fallback.  
**Fix options:**
- Randomize candidate order before sending to LLM (makes selection independent of score rank).
- Or: remove score prefix from the prompt so the LLM cannot use rank as a signal.
- Minimum: log how often the LLM picks candidate #1 vs others to measure actual bias.

### ACC-9 — RELATE pair truncation is index-ordered, not importance-ordered
**Severity:** Medium  
**File:** `relate_stage.py:294–299`  
**Symptom:** When `len(eligible) > MAX_PAIRS (500)`, the first 500 by `itertools.combinations`
index order are kept. Rules from later chunks (end of paper — often Discussion/Conclusion,
which carry high-confidence summary statements) are systematically dropped.  
**Fix:** Sort eligible pairs by `(rules[i].mean_grounding_score + rules[j].mean_grounding_score)`
descending before truncating to `max_pairs`.

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

### DES-2 — Incremental corpus relate never computes intra-paper relations for the new paper
**Severity:** Medium  
**File:** `corpus_relate.py:333–340` (`_incremental_gate`)  
**Symptom:** The XOR gate rejects any pair where both rules come from the same paper. So when
a paper is processed incrementally, its own intra-paper corpus relations are never written to
`SumCorpusRelation`. Only a full `relate_from_dir` run produces them. If incremental is the
only mode used, intra-paper corpus relations are permanently absent for all papers.  
**Fix:** After computing cross-paper pairs, run a second pass over `new_rules` with the
standard `_should_compare_cross_paper` gate (no XOR restriction) to produce intra-paper
pairs, then include them in `_replace_for_pmcid`.

### DES-3 — NLI model `cross-encoder/nli-deberta-v3-base` is wrong for rule-to-rule comparison
**Severity:** Medium (no fix landed yet)  
**File:** `relate_stage.py:44` (`_NLI_MODEL`)  
**Symptom:** The model was trained on NLI sentence pairs (e.g. MNLI), not on clinical predicate
pairs. It may fire CONTRADICT on two statements that are clinically co-occurring but lexically
distinct. Wrong CONTRADICT labels feed directly into RESOLVE and silently lower final rule scores.  
**Better model:** `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` or a model fine-tuned on
biomedical entailment. Would need threshold recalibration.

### DES-4 — NLI model in grounding filter is general-domain, not biomedical
**Severity:** Medium  
**File:** `grounding_filter.py:23` (`_DEFAULT_MODEL`)  
**Symptom:** The grounding filter uses the same `cross-encoder/nli-deberta-v3-base` (trained on
MNLI/general text) to score `(verbatim_support, claim)` pairs. The use case here is different
from RELATE's rule-to-rule comparison — it is claim grounding verification against a clinical
source sentence. General-domain entailment may underfire on biomedical phrasing, causing
genuinely grounded findings to receive low scores and be dropped.  
**Fix:** Replace with a biomedical NLI model (e.g. `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
or a BioBERT-based cross-encoder). Threshold will need recalibration after swap. Fix is
independent of DES-3 (different call site, different pair type).

### DES-5 — `_split_windows()` cuts at fixed character boundaries, not sentence boundaries
**Severity:** Medium  
**File:** `grounding_filter.py:185–197` (`_split_windows`)  
**Symptom:** The current 400-char / 200-char-step sliding window cuts mid-sentence, so a
premise window may begin or end with an incomplete sentence fragment. This can produce
misleading NLI scores when the entailment-supporting clause is split across two windows.  
**Fix:** Replace character-based windowing with sentence-aware packing:
1. Split premise into sentences (e.g. spaCy sentencizer or simple regex).
2. Pack sentences into windows under a token budget (≤450 tokens).
3. Add 1–2 sentence overlap between adjacent windows (analogous to current step overlap).
This ensures every window starts and ends on a sentence boundary and the NLI model always
sees complete clauses.

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

### DES-6 — `CanonicalScopeEnum` is a single value — cannot express conflicted + multi_study
**Severity:** Low  
**File:** `canonicalize_stage.py:83–110` (`_compute_scope`), `models.py:265–270` (`CanonicalScopeEnum`)  
**Symptom:** A rule supported by mixed-direction evidence across multiple papers is both
`conflicted` and `multi_study`, but the enum forces one value. `conflicted` wins by priority,
discarding the PMCID coverage information entirely.  
**Fix:** Replace the single-value enum with two separate fields on `CanonicalRule`:
`is_conflicted: bool` and `study_coverage: Literal["single_study", "multi_study", "unknown"]`.
This is a schema change requiring a migration.

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
finding_count, and canonical_scope. Weights could then be fit to minimise ranking error.  
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
| BUG-1 | High | CANONICALIZE | `_compute_scope` reads group-level direction_counts on a direction-split bin |
| ACC-1 | High | MAP | Chunk boundary findings lost — no overlap |
| ACC-2 | High | MAP/GROUP | `relation_type` variance splits same fact into different groups |
| ACC-3 | High | RELATE | Subject exact-match drops normalization-near-miss pairs |
| ACC-4 | High | MAP/GROUNDING | `verbatim_support` is LLM paraphrase, depresses real grounding scores |
| ACC-5 | High | CANONICALIZE/GROUP | CUI enrichment post-canonicalize cannot fix grouping errors |
| ACC-6 | High | CANONICALIZE | `unclear` direction findings inflated into largest bin |
| DES-1 | High | RESOLVE | Cross-paper relations ignored in FinalRule scoring — defeats cross-paper feature |
| ACC-7 | Medium | CANONICALIZE | Unbounded candidate list sent to LLM — expensive with cross-paper pooling |
| ACC-8 | Medium | CANONICALIZE | Position bias from grounding-score ordering in LLM predicate selection |
| ACC-9 | Medium | RELATE | Pair truncation is index-ordered, not importance-ordered |
| ACC-10 | Medium | CORPUS RELATE | Cross-paper gate skips outcome gating for non-expression rules |
| ACC-11 | Medium | NORMALIZE | `infer_direction()` wrong on complex clinical negation |
| DES-2 | Medium | CORPUS RELATE | Incremental mode never computes intra-paper corpus relations |
| DES-3 | Medium | RELATE | NLI model general-domain, wrong CONTRADICT feeds into RESOLVE scoring |
| DES-4 | Medium | GROUNDING | NLI model general-domain — first quality gate, biomedical underscoring risk |
| DES-5 | Medium | GROUNDING | `_split_windows()` char-boundary cuts depress grounding scores at first gate |
| ACC-12 | Low | RELATE | `_nli_scores()` lacks sliding window — rarely fires in practice |
| DES-6 | Low | CANONICALIZE | `CanonicalScopeEnum` single value loses conflicted + multi_study combination |
| DES-7 | Low | RESOLVE | Two scoring modes produce non-comparable final score scales |
| DES-8 | Low | RESOLVE | Scoring weights hand-tuned, not empirically validated — no gold dataset |
| DES-9 | Low | NORMALIZE | `_best_scope()` ignores scope from lower-grounding findings |
