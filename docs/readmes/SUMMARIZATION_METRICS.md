# Summarization pipeline — numerical metrics & formulas

Reference for every numerical signal in `pipeline/stages/summarization/`:
grounding, agreement scorers, similarity strategies, RELATE NLI gates,
RESOLVE final-score, UMLS threshold.

---

## 1. GROUNDING (NLI entailment)

`helpers/grounding_filter.py`. Cross-encoder NLI scores
`(premise=verbatim_support, hypothesis=claim)`.

Per-pair raw score = model's `entailment` probability over
`{entailment, contradiction, neutral}`.

**Long-premise windowing** (`_split_windows`): premise sentenced via spaCy,
packed greedily into windows fitting

```
budget = 512 − len(hyp_tokens) − 3_special_tokens   (floor 64)
```

Adjacent windows share their last chunk (overlap) so a boundary sentence
appears with context in at least one window. Per pair:

```
g(p, h) = max over w ∈ windows(p) of P_ent(w, h)
```

Empty premise → `0.0`. Stored as `Finding.grounding_score`.

**Filter decision** (`GroundingFilter`):
keep finding if `g ≥ threshold` (default `0.5`, set via
`MapConfig.grounding.threshold` / active NLI spec).

**Rule grounding** (`filter_rules`): hypothesis = `condition + " " + action`.
Drop evidence items with `g < threshold`; drop entire rule when no evidence
survives.

**Voter pass fraction** (`VoterContext.grounding_pass_fraction`):

```
pf_v = |{ f ∈ findings_v : g_f ≥ θ_g }| / |findings_v|
```

Fallback when no validator context available
(`semantic_scorer._pass_frac`): fraction of findings with non-empty
evidence list — structural proxy, not validator-checked.

---

## 2. EMBEDDING AGREEMENT (soft-alignment cosine)

`agreement/embedding.py` + `embedding_similarity.py`.
Per voter pair (A, B), claim embeddings `e_a`, `e_b` (L2-normalised
OpenAI `text-embedding-3-small`).

**Similarity matrix:**
```
S_ij = clip(e_a_i · e_b_j, 0, 1)            shape (n_a, n_b)
```

**Weak-match cutoff** (`tau = 0.15`):
```
S^τ_ij = S_ij if S_ij ≥ τ else 0
```

**Bidirectional soft coverage:**
```
cov_A→B = (1/n_a) Σ_i max_j S^τ_ij
cov_B→A = (1/n_b) Σ_j max_i S^τ_ij
base    = (cov_A→B + cov_B→A) / 2
```

**Count-mismatch factor** (`count_alpha = 0.25`):
```
cf = (min(n_a, n_b) / max(n_a, n_b)) ** 0.25
```
4:1 split → ≈0.71×.

**Reuse-concentration factor** (`reuse_weight = 0.15`), per direction
using `argmax` indices `best`:
```
max_reuse_frac = max(bincount(best)) / n_source
excess         = max(0, max_reuse_frac − 1/n_target)
rf             = 1 − w * excess / (1 − 1/n_target)
```
Mean of both directions. Floor = `1 − w` = 0.85.

**Contradiction factor** (`contradiction_weight = 0.20`):

- *Polarity ratio*: among strong best-matches (`S^τ ≥ τ`),
  fraction where polarities of A's claim and B's claim oppose.
  Lexicon `_POSITIVE` / `_NEGATIVE` ± 3-token negation window;
  `non-` prefix flips polarity.
- *Numeric ratio*: same strong pairs, both claims contain exactly one
  number, `max/min ≥ 2.0`.

```
r            = max(polarity_ratio, numeric_ratio)
contra_factor = 1 − 0.20 * r
```

**Pre-grounding score:**
```
pre = base * cf * rf * contra_factor
```

**Grounding penalty** (per pair, `grounding_floor = 0.50`):
```
gf       = 0.50 + 0.50 * min(pf_i, pf_j)
score_ij = pre * gf
```

`EmbeddingScorer.embedding_agreement` = mean of all
`C(n, 2)` pair scores. `EmbeddingSimilarityStrategy.compute_matrix`
builds the full N×N.

---

## 3. NER ENTITY OVERLAP (Jaccard)

`agreement/ner_scorer.py`. scispaCy `en_core_sci_lg`,
linker disabled. Per voter:
```
E_v = { lower(ent.text) for ent ∈ scispacy(" ".join(claims)).ents }
```
Pair score:
```
J(E_a, E_b) = |E_a ∩ E_b| / |E_a ∪ E_b|
```
Both empty → `1.0`; exactly one empty → `0.0`.
Bundle: mean over all pairs.

---

## 4. CATEGORY JACCARD

`agreement/category_jaccard.py`. Same Jaccard on
`frozenset(f.category for f in findings)`. Fast, no API.

---

## 5. LEXICAL JACCARD (debug baseline)

`agreement/lexical_similarity.py`. Jaccard on lowercased word sets
extracted from all claims joined. Debug only — not for thesis eval.

---

## 6. EVIDENCE JACCARD

`agreement/hybrid_structured.py::_evidence_jaccard`. Parse
`pmcid|path|te_id` evidence strings; Jaccard on `te_id` sets.

---

## 7. HYBRID STRUCTURED (v1, weighted)

`HybridStructuredSimilarity`. Weights sum to 1.0:
```
sim = 0.25 * J_category
    + 0.40 * embedding_align
    + 0.25 * J_entity
    + 0.10 * J_evidence
```
`summary_text` similarity is reserved for future versions (weight 0).

---

## 8. SEMANTIC AGREEMENT (deferral score)

`agreement/semantic_scorer.py`. Excludes empty voters first; build
`m × m` similarity matrix `M` via active strategy.

**Average similarity per candidate:**
```
avg_sim_i = (1 / (m − 1)) * Σ_{j ≠ i} M_ij
```

**Deferral score:**
```
confidence = max_i avg_sim_i
```

**Best candidate:** `argmax avg_sim_i`, ties broken by
`(grounding_pass_fraction, mean_evidence_length, len(findings))`.

---

## 9. AGREEMENT DECISION (theta cascade)

`AgreementChecker` — fallback when scorer doesn't set decision
(default `theta = 0.7`, `reject_theta = 0.2`):

```
primary = confidence or embedding_agreement or 0.0
primary ≥ θ        → KEEP
primary ≤ θ_reject → REJECT
otherwise          → ESCALATE
```

`MapConfig` defaults: `theta = 0.8`, `reject_theta = 0.2`.
Paper-test config overrides to `theta = 0.65`, `reject_theta = 0.15`.

---

## 10. CASCADED COMPOSITE (LP-tuned 2D threshold)

`agreement/composite.py`. Decisions on `(emb, ner)`:

```
KEEP    if emb ≥ keep_emb_thr (0.80) AND ner ≥ keep_ner_thr (0.50)
        confidence = (emb + ner) / 2
REJECT  if emb ≤ reject_thr (0.20)
        confidence = emb
else    ESCALATE
        confidence = (emb + ner) / 2
```

Thresholds loadable from `OptimizedThresholds.json`
(LP optimiser in `agreement/calibration/`).

---

## 11. LLM JUDGE

`agreement/llm_judge.py`. One LLM call rates `[0, 1]` overall agreement
across all voter claim lists. Structured output
`_JudgeScore(score, rationale)`. Stored in
`ScoreBundle.judge_agreement`.

---

## 12. RELATE (NLI pairwise on canonical rules)

`current_stages/relate_stage.py`. Both directions
`(p_a → p_b)` and `(p_b → p_a)` scored by NLI windowing
(max-pooled over windows).

**Defaults:** `entailment_threshold = 0.55`,
`contradiction_threshold = 0.65` (config default `0.50`).

**Eligibility gate** `_should_compare` — pair reaches NLI only when
ALL hold:
1. `category` matches exactly
2. `relation_type` matches exactly
3. `subject_entity` matches exactly (synonym-normalised, lowercased)
4. `outcome_entity` matches exactly
   (for `expression`: marker name after stripping
   `expression / positivity / staining / immunoreactivity` suffixes)

**Decision:**
```
CONTRADICT   if con_ab ≥ τ_con AND con_ba ≥ τ_con
             AND polarities differ
             ({positive, partial} vs {negative, absent})
SUPPORT      if ent_ab ≥ τ_ent AND ent_ba ≥ τ_ent
else         UNRELATED
```

---

## 13. UMLS LINK SCORE

`umls_utils.py::UMLS_THRESHOLD = 0.85`. scispaCy UMLS linker hits
below this threshold are discarded during entity normalisation.

---

## 14. RESOLVE `final_score`

`current_stages/resolve_stage.py`. Two formulas depending on whether
RELATE produced any relational signal.

**relations_present** (any rule has relations):
```
g                 = rule.mean_grounding_score      (default 0.50 if None)
base              = g * 0.60                       (0–0.60)
finding_bonus     = min(N_f / 5, 1) * 0.10         (0–0.10)
support_bonus     = min(N_s * 0.08, 0.20)          (0–0.20)
single_study_pen  = 0.10 if single_study else 0
contradict_pen    = min(N_c * 0.15, 0.30)          (0–0.30)
```

**relations_absent** (single paper / NLI skipped):
```
g                 = rule.mean_grounding_score      (default 0.40 if None)
base              = g * 0.80                       (0–0.80, wider spread)
finding_bonus     = min(N_f / 5, 1) * 0.15         (0–0.15)
support_bonus     = 0
contradict_pen    = 0
single_study_pen  = 0.05 if single_study else 0    (halved)
```

**Final:**
```
final_score = clip(
    base + finding_bonus + support_bonus
       − single_study_pen − contradict_pen,
    0.0, 1.0
)
```

Sorted desc; `is_contradicted = (N_c > 0)`.

Worked examples (relations_absent mode):
```
g=0.9, N_f=3, single_study:  0.72 + 0.09 − 0.05 = 0.76
g=0.3, N_f=1, single_study:  0.24 + 0.03 − 0.05 = 0.22
```

---

## 15. CANONICALIZE rule grounding

`mean_grounding_score` per `CanonicalRule` =
mean of member `NormalFinding.mean_grounding_score`s
(None when no member has a score).

Per `NormalFinding.mean_grounding_score` =
mean of member `Finding.grounding_score`s.

---

## Threshold / default summary

| Knob | Default | Source |
|---|---|---|
| Grounding NLI threshold | 0.5 | active NLI spec |
| Emb tau (weak-match) | 0.15 | `EmbeddingScorer` |
| count_alpha | 0.25 | `EmbeddingScorer` |
| reuse_weight | 0.15 | `EmbeddingScorer` |
| contradiction_weight | 0.20 | `EmbeddingScorer` |
| grounding_floor | 0.50 | `EmbeddingScorer` |
| Numeric ratio threshold | 2.0 | `_NUMERIC_RATIO_THRESHOLD` |
| Theta (KEEP) | 0.8 (paper-test 0.65) | `MapConfig` |
| reject_theta | 0.2 (paper-test 0.15) | `MapConfig` |
| RELATE entailment_thr | 0.50 (code default 0.55) | `RelateConfig` |
| RELATE contradiction_thr | 0.50 (code default 0.65) | `RelateConfig` |
| UMLS link score | 0.85 | `umls_utils.UMLS_THRESHOLD` |
| Composite keep_emb / keep_ner / reject_emb | 0.80 / 0.50 / 0.20 | `CascadedCompositeScorer` |
| Resolve grounding_weight (rel / no-rel) | 0.60 / 0.80 | `ResolveConfig` |
| Resolve finding_bonus_max (rel / no-rel) | 0.10 / 0.15 | `ResolveConfig` |
| Resolve single_study_pen (rel / no-rel) | 0.10 / 0.05 | `ResolveConfig` |
| Resolve support_boost cap / per-rel | 0.20 / 0.08 | `ResolveConfig` |
| Resolve contradict_pen cap / per-rel | 0.30 / 0.15 | `ResolveConfig` |
