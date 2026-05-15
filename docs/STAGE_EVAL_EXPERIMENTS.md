# Stage Evaluation Experiments — Model-Agnostic Edition

A complementary eval battery to the Opus-silver harness described in
[`../eval/llm_judge/STAGE_EVAL_DESIGN.md`](../eval/llm_judge/STAGE_EVAL_DESIGN.md).
Every experiment in this document avoids any LLM-generated reference label
so swapping the MAP voter cascade (or the L3 escalator) does not invalidate
the metric.

---

## 1. Motivation

The Opus-silver harness gives the strongest signal on "is this the right
finding", but it has two structural problems for thesis-time experimentation:

1. Every change to the LLM cascade — different L1 voters, different L3 model,
   different prompt — can in principle shift the *content* of MAP findings,
   so the silver corpus must be regenerated to compare apples-to-apples.
2. Silver generation is expensive and pins the eval to a single judge model.

Conclusion: we also need experiments whose **references are not LLM output**.
A "reference" here is one of:

- PMC source text already in the DB or in `verbatim_support` fields,
- a pinned cross-encoder NLI model (see
  [`pipeline/stages/summarization/nli_config.py::get_active_spec`](../pipeline/stages/summarization/nli_config.py)),
- UMLS at a pinned scispaCy version (see
  [`pipeline/stages/summarization/umls_resources.py`](../pipeline/stages/summarization/umls_resources.py)),
- structural / set-equality replays of deterministic stage logic,
- cross-run differentials (run cascade A and cascade B on the same papers
  and compare; both runs are unlabelled — neither is "truth").

Scope: stage-level transformations only. The question is *did each stage do
the right thing with the input it was given*, not *was the LLM's claim
correct in the absolute sense*. The latter belongs in the Opus harness.

Cascade context for the experiments below lives in
[`ABC_IMPLEMENTATION_COMPARISON.md`](ABC_IMPLEMENTATION_COMPARISON.md).

---

## 2. Identifier stability

Every cross-run experiment needs to know which IDs are deterministic.

| Identifier | Stable across cascades? | Source | Use it for |
|---|---|---|---|
| `finding_id` | No — hashes the claim string (and chunk position) | [`models.py::compute_finding_id`](../pipeline/stages/summarization/models.py) | Within-run dedup audits only. |
| `normal_id` | Mostly — stable iff UMLS resolves the same CUI for the same input string | `normalize_stage.py` | Within-run joins; X-series MAP↔NORMALIZE alignment. |
| `group_id` | **Yes** — sha8 of `(pmcid, subject_cui, outcome, relation_type, category)` | `group_stage.py` | Primary join key for all cross-run experiments from GROUP onward. |
| `canonical_id` | **Yes** — sha8 of `(group_id, direction)` | `canonicalize_stage.py` | RELATE and RESOLVE cross-run joins. |
| `final_id` | **Yes** — derived from `canonical_id` | `resolve_stage.py` | RESOLVE cross-run ranking experiments. |

Implication: **MAP-level cross-run experiments need fuzzy alignment**
(embedding cosine via `eval/silver/matcher.py::compute_sim_matrix`). From
GROUP onward, direct set / join operations on the stable IDs are correct.

---

## 3. Per-stage experiments

Every experiment below is specified with eight fields: **ID · Stage ·
Reference · P/R/F1 · Artifacts · Helper · Model-agnostic? · Cost · Gotcha**.
Experiments are ranked cheapest-first within each stage.

### 3.1 MAP

#### M1 — `verbatim_support` source-text containment

- **Stage:** MAP
- **Reference:** the raw chunk text (`map/{pmcid}/chunks.jsonl::text`),
  which is the input the MAP voter was given.
- **P/R/F1:**
  - tp = finding where every span in `verbatim_support` is a substring of
    the chunk text after NFKC + whitespace normalisation.
  - fp = finding that fails the substring check.
  - Recall is N/A — no enumerated truth set.
  - **Precision = TP / (TP + FP).**
- **Artifacts:** `map/{pmcid}/findings.jsonl`, `map/{pmcid}/chunks.jsonl`.
- **Helper:** reuse the normalisation helper in
  [`eval/silver/matcher.py`](../eval/silver/matcher.py) (`_normalize`).
- **Model-agnostic?** Yes — pure string containment.
- **Cost:** cheap.
- **Gotcha:** Unicode-fold once (NFKC) and collapse whitespace before
  comparison; otherwise smart-quotes and non-breaking spaces will look
  like fabricated verbatim.

#### M2 — chunk-yield distribution stability

- **Stage:** MAP
- **Reference:** the run's own distribution.
- **P/R/F1:** not P/R/F1; report **coverage stability = % of chunks whose
  `n_findings / n_sentences` falls within the run-global IQR**. Flag outliers
  (very low yield or very high) for inspection.
- **Artifacts:** `map/{pmcid}/findings.jsonl` grouped by `chunk_id`,
  `map/{pmcid}/chunks.jsonl`.
- **Helper:** any percentile routine; no project-specific helper required.
- **Model-agnostic?** Yes — distribution-only, no external reference.
- **Cost:** cheap.
- **Gotcha:** sample size — for short papers, IQR collapses; report n.

#### M3 — NLI entailment of `claim` by chunk text

- **Stage:** MAP
- **Reference:** the pinned cross-encoder NLI model from
  [`nli_config.py`](../pipeline/stages/summarization/nli_config.py).
- **P/R/F1:**
  - tp = finding where NLI(premise = chunk text, hypothesis = `claim`) =
    `ENTAILMENT`.
  - fp = `NEUTRAL` or `CONTRADICTION`.
  - Recall is N/A (no enumerated truth set).
  - **Precision = TP / (TP + FP).**
- **Artifacts:** `map/{pmcid}/findings.jsonl`, `map/{pmcid}/chunks.jsonl`.
- **Helper:**
  [`pipeline/stages/summarization/helpers/grounding_filter.py::GroundingFilter`](../pipeline/stages/summarization/helpers/grounding_filter.py)
  invoked offline (NLI inference only; do not write back to the run).
- **Model-agnostic?** Yes — the NLI model is pinned in `nli_config.py` and
  the LLM under test is not used as judge.
- **Cost:** medium — one NLI call per finding; expect O(thousands) per run.
- **Gotcha:** record `NLIModelSpec.hf_id` and `batch_size` in the report
  header for reproducibility.

### 3.2 GROUNDING

#### G1 — `rejected_findings` recall against NLI labels

This is the core grounding experiment.

- **Stage:** GROUNDING (the filter that drops findings before NORMALIZE).
- **Reference:** the same pinned NLI used in M3.
- **P/R/F1:**
  - Treat NLI label as ground truth (`ENTAIL` = correct claim, otherwise =
    incorrect).
  - kept-in-`findings.jsonl` ∧ NLI=ENTAIL → tp
  - kept-in-`findings.jsonl` ∧ NLI≠ENTAIL → fp
  - in-`rejected_findings.jsonl` ∧ NLI=ENTAIL → fn
  - in-`rejected_findings.jsonl` ∧ NLI≠ENTAIL → tn
  - **Precision / Recall / F1 standard from those counts.**
- **Artifacts:** `map/{pmcid}/findings.jsonl`,
  `map/{pmcid}/rejected_findings.jsonl`.
- **Helper:**
  [`eval/precision_recall.py::metrics`](../eval/precision_recall.py).
- **Model-agnostic?** Yes — NLI model is pinned; LLM under test is not.
- **Cost:** cheap if M3's NLI scores were cached; otherwise medium.
- **Gotcha:** `GroundingFilter`'s threshold at run time and the offline NLI
  pass must use the **same** `nli_config.py` spec. Pin once per eval run
  and stamp into the report header.

#### G2 — rejection-reason attribution

- **Stage:** GROUNDING
- **Reference:** NLI labels (as in G1).
- **P/R/F1:** per-bucket precision = fraction of `rejected_findings` carrying
  reason `r` whose NLI label confirms rejection was warranted (i.e. NLI ≠
  ENTAIL). Surfaces miscalibrated rules.
- **Artifacts:** `map/{pmcid}/rejected_findings.jsonl::rejection_reason`.
- **Helper:** group-by-then-`metrics()`.
- **Model-agnostic?** Yes.
- **Cost:** cheap (reuses G1 NLI scores).
- **Gotcha:** requires `rejection_reason` to be populated; some legacy runs
  store only the threshold without the reason — verify on the target run.

#### G3 — span-offset precision

- **Stage:** GROUNDING
- **Reference:** chunk text itself.
- **P/R/F1:**
  - For kept findings, take the stored `evidence[].span` offsets, slice the
    chunk text, and compare to `verbatim_support`.
  - tp = exact match, fp = mismatch.
  - **Precision = TP / (TP + FP).**
- **Artifacts:** `map/{pmcid}/findings.jsonl`, `map/{pmcid}/chunks.jsonl`.
- **Helper:** string slicing; no helper.
- **Model-agnostic?** Yes — pure structural.
- **Cost:** cheap.
- **Gotcha:** catches off-by-one tokenisation regressions, especially after
  whitespace/Unicode-normalisation changes.

### 3.3 NORMALIZE

#### N1 — CUI assignment determinism

- **Stage:** NORMALIZE
- **Reference:** the offline UMLS scispaCy linker at the pinned version.
- **P/R/F1:**
  - For each `NormalFinding`, re-run the linker on
    `subject_entity` / `outcome_entity`.
  - tp = top-1 CUI matches the stored `subject_cui` / `outcome_cui`.
  - fp = top-1 disagrees with stored CUI.
  - fn = stored CUI present, recomputed top-1 null.
  - **Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 standard.**
- **Artifacts:** `normalize/{pmcid}/normal_findings.jsonl`,
  `normalize/{pmcid}/entity_links.jsonl`.
- **Helper:**
  [`pipeline/stages/summarization/umls_resources.py::get_nlp`,
  `get_linker`](../pipeline/stages/summarization/umls_resources.py).
- **Model-agnostic?** Yes — UMLS version is pinned, linker is deterministic
  given the input string and threshold.
- **Cost:** cheap (single-threaded UMLS calls; thousands per run).
- **Gotcha:** record UMLS / scispaCy version hash in the report header.

#### N2 — dedup cohesion via embedding cosine

- **Stage:** NORMALIZE
- **Reference:** embeddings of the source claims.
- **P/R/F1:** precision-only. For every `NormalFinding` with
  `len(source_finding_ids) > 1`, compute pairwise cosine between the
  source claims. tp = cluster where every pair ≥ pinned θ_cohesion
  (proposed 0.85); fp = at least one pair below θ.
  **Precision = TP / (TP + FP).**
- **Artifacts:** `normalize/{pmcid}/normal_findings.jsonl`,
  `normalize/{pmcid}/dedup_trace.jsonl`, `map/{pmcid}/findings.jsonl`.
- **Helper:**
  [`eval/silver/matcher.py::compute_sim_matrix`,
  `EmbeddingCache`](../eval/silver/matcher.py).
- **Model-agnostic?** Yes — embedding model is pinned in the report header.
- **Cost:** cheap (cached embeddings).
- **Gotcha:** θ_cohesion must be pinned and reported; tune once across a
  small sample before locking.

#### N3 — `entity_links` roundtrip

- **Stage:** NORMALIZE
- **Reference:** N1's recomputed CUIs.
- **P/R/F1:** trivially equivalent to N1 (precision-only). Keep as a
  separate row only if `entity_links.jsonl` ever stores extra context not
  present on `NormalFinding`.
- **Artifacts:** `normalize/{pmcid}/entity_links.jsonl`.
- **Helper:** same as N1.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** consider folding into N1 unless there's a divergence in
  practice.

### 3.4 GROUP

#### GR1 — `group_id` determinism replay

- **Stage:** GROUP
- **Reference:** recomputed `group_id` from the same hash inputs.
- **P/R/F1:** precision-only. tp = stored `group_id` matches recomputed.
  fp = mismatch.
- **Artifacts:** `group/{pmcid}/groups.jsonl`,
  `normalize/{pmcid}/normal_findings.jsonl`.
- **Helper:** import the sha8 builder directly from
  `pipeline/stages/summarization/current_stages/group_stage.py`.
- **Model-agnostic?** Yes — pure structural replay.
- **Cost:** cheap.
- **Gotcha:** targets serialisation / persistence bugs; expected to be 100%.

#### GR2 — outcome-text embedding cohesion

- **Stage:** GROUP
- **Reference:** embeddings.
- **P/R/F1:** precision-only. For each group, compute pairwise cosine
  between member `outcome_entity` strings. tp = group where every pair ≥
  θ_cohesion (proposed 0.60, looser than N2 because outcome wording
  varies more). fp = at least one pair below θ. Surfaces normaliser
  leakage.
- **Artifacts:** `group/{pmcid}/groups.jsonl`,
  `normalize/{pmcid}/normal_findings.jsonl`.
- **Helper:** same as N2.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** θ_cohesion is per-stage — don't reuse N2's value verbatim.

#### GR3 — `non_groupable.jsonl` audit

- **Stage:** GROUP
- **Reference:** the rest of `normal_findings.jsonl`.
- **P/R/F1:** precision-only. For each row in `non_groupable.jsonl`,
  verify there is *no* other NormalFinding with the same
  `(subject_cui, outcome, relation_type)` triple. tp = correctly
  excluded; fp = should have been groupable.
- **Artifacts:** `group/{pmcid}/non_groupable.jsonl`,
  `normalize/{pmcid}/normal_findings.jsonl`.
- **Helper:** none — set comparison.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** make sure the comparison uses the same normalisation
  function `_group_id` used internally (case, trim).

### 3.5 CANONICALIZE

#### C1 — `canonical_id` determinism replay

- **Stage:** CANONICALIZE
- **Reference:** recomputed `canonical_id` from `(group_id, direction)`.
- **P/R/F1:** precision-only. tp = stored matches recomputed.
- **Artifacts:** `canonicalize/{pmcid}/canonical_rules.jsonl`.
- **Helper:** import sha8 helper from
  `pipeline/stages/summarization/current_stages/canonicalize_stage.py`.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** mirrors GR1 in spirit; expected to be 100%.

#### C2 — direction consistency via NLI

- **Stage:** CANONICALIZE
- **Reference:** pinned NLI on a templated sentence.
- **P/R/F1:** For each `CanonicalRule`, build a templated hypothesis
  (`"{subject_entity} {direction_verb} {outcome_entity}"`). Premise is the
  concatenation of the `verbatim_support` strings from all member findings.
  tp = NLI=ENTAIL; fp = NEUTRAL or CONTRADICT.
  **Precision = TP / (TP + FP).**
- **Artifacts:** `canonicalize/{pmcid}/canonical_rules.jsonl`,
  `normalize/{pmcid}/normal_findings.jsonl`,
  `map/{pmcid}/findings.jsonl` (for `verbatim_support` backfill).
- **Helper:** `GroundingFilter` as in M3 / G1.
- **Model-agnostic?** Yes.
- **Cost:** medium (one NLI call per rule).
- **Gotcha:** template wording matters; commit one phrasing per direction
  enum and stamp it into the report.

#### C3 — `member_normal_ids` closure

- **Stage:** CANONICALIZE
- **Reference:** the parent group's `member_ids`.
- **P/R/F1:** precision-only. tp = every `CanonicalRule.member_normal_ids`
  is a subset of its parent `FindingGroup.member_ids`; fp = otherwise.
- **Artifacts:** `canonicalize/{pmcid}/canonical_rules.jsonl`,
  `group/{pmcid}/groups.jsonl`.
- **Helper:** none — set comparison.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** catches join breakage from persistence layer changes.

### 3.6 RELATE

> **Precondition:** `relate/{pmcid}/raw_pairs.jsonl` must be populated.
> [`../eval/llm_judge/STAGE_EVAL_DESIGN.md`](../eval/llm_judge/STAGE_EVAL_DESIGN.md)
> flags this file as empty in some recorded runs — R1, R4 cannot run
> without it. Re-run RELATE with the persistence path enabled before
> attempting those.

#### R1 — NLI replay determinism

- **Stage:** RELATE
- **Reference:** pinned NLI re-run on the same `(text_a, text_b)` predicate
  pairs that produced the stored relation labels.
- **P/R/F1:** treat stored label as prediction, NLI replay as reference.
  tp = same label; fp = mismatch (per-class one-vs-rest).
  **Precision per class plus macro F1.**
- **Artifacts:** `relate/{pmcid}/relations.jsonl`,
  `relate/{pmcid}/raw_pairs.jsonl`.
- **Helper:** instantiate the same NLI engine RELATE used (see
  [`pipeline/stages/summarization/current_stages/relate_stage.py`](../pipeline/stages/summarization/current_stages/relate_stage.py)).
- **Model-agnostic?** Yes — both sides use the same pinned NLI.
- **Cost:** medium.
- **Gotcha:** detects threshold drift and NLI cache poisoning. Any
  divergence > 1% indicates a config drift; run is invalidated.

#### R2 — pre-NLI gate audit

- **Stage:** RELATE
- **Reference:** the gate logic in `relate_stage.py`.
- **P/R/F1:** precision-only. For each row in `skipped_pairs.jsonl`, replay
  the gate predicates on the referenced pair. tp = same `skip_reason`;
  fp = newly admissible (or different reason).
- **Artifacts:** `relate/{pmcid}/skipped_pairs.jsonl`,
  `canonicalize/{pmcid}/canonical_rules.jsonl`.
- **Helper:** import gate predicates directly from `relate_stage.py`.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** the gate is deterministic — fp > 0 is a regression.

#### R3 — symmetry checks

- **Stage:** RELATE
- **Reference:** the relation set itself.
- **P/R/F1:** precision-only. For every relation (a→b), check that the
  reverse (b→a) is either absent or carries the dual label (SUPPORT ↔
  SUPPORT, CONTRADICT ↔ CONTRADICT, SCOPE_QUALIFY is asymmetric and
  expected only one-way).
- **Artifacts:** `relate/{pmcid}/relations.jsonl`.
- **Helper:** none — set comparison.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** SCOPE_QUALIFY is intentionally directional; do not penalise
  asymmetry there.

#### R4 — UNRELATED-rate vs. embedding distance calibration

- **Stage:** RELATE
- **Reference:** embeddings of predicate text.
- **P/R/F1:** not P/R/F1 — a calibration curve. For all pairs in
  `raw_pairs.jsonl`, bin by `cosine(embed(text_a), embed(text_b))` and
  report `P(label ≠ UNRELATED | bin)`. Inverts to "at what similarity does
  the model start to recognise a relationship?".
- **Artifacts:** `relate/{pmcid}/raw_pairs.jsonl`.
- **Helper:**
  [`eval/silver/matcher.py::compute_sim_matrix`](../eval/silver/matcher.py).
- **Model-agnostic?** Yes — embedding model and binning are pinned.
- **Cost:** medium (one embedding call per unique predicate).
- **Gotcha:** report bin counts; sparse bins (n < 20) are noise.

### 3.7 RESOLVE

#### Rs1 — score formula replay

- **Stage:** RESOLVE
- **Reference:** the deterministic scoring formula in `resolve_stage.py`.
- **P/R/F1:** Spearman ρ and Kendall τ between stored `score` and replayed
  formula output; top-k overlap at k ∈ {5, 10, 20}.
- **Artifacts:** `resolve/{pmcid}/final_rules.jsonl`,
  `resolve/{pmcid}/score_trace.jsonl`,
  `canonicalize/{pmcid}/canonical_rules.jsonl`,
  `relate/{pmcid}/relations.jsonl`.
- **Helper:** import scoring helpers directly from
  `pipeline/stages/summarization/current_stages/resolve_stage.py`.
- **Model-agnostic?** Yes — pure replay.
- **Cost:** cheap.
- **Gotcha:** `score_trace.jsonl` currently duplicates `final_rules.jsonl`
  without per-component breakdown; Rs1 still works (it replays the full
  formula), but a future improvement is to persist component scores
  (`base`, `support_bonus`, `pen_*`) so ablation becomes per-component.

#### Rs2 — top-k stability under bootstrap

- **Stage:** RESOLVE
- **Reference:** own bootstrap resamples.
- **P/R/F1:** not P/R/F1. Drop 10% of member relations at random (seeded
  PRNG), re-score offline. Report Jaccard(top-k_full, top-k_bootstrap)
  at k ∈ {5, 10, 20} averaged over n=50 bootstrap rounds. Stability metric.
- **Artifacts:** as Rs1.
- **Helper:** same scoring helpers.
- **Model-agnostic?** Yes.
- **Cost:** medium (n × replay).
- **Gotcha:** pin the seed; report it.

#### Rs3 — `final_rule` ↔ `canonical_id` closure

- **Stage:** RESOLVE
- **Reference:** `canonical_rules.jsonl`.
- **P/R/F1:** precision-only. Every `FinalRule.canonical_id` must exist
  in `canonical_rules.jsonl`. Trivially structural.
- **Artifacts:** `resolve/{pmcid}/final_rules.jsonl`,
  `canonicalize/{pmcid}/canonical_rules.jsonl`.
- **Helper:** none.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** catches join breakage; expected 100%.

---

## 4. Differential cross-run experiments

Run pipeline twice on the **same paper set** with two different cascade
profiles (e.g. `smoke_haiku` vs the default ABC `real` profile). Treat one
as reference, one as prediction (or compute symmetric agreement). Both runs
are unlabelled — neither needs a silver corpus.

> **Precondition:** invoke both runs with the same `--from-selection
> <selection_yaml>` so the PMCID set is identical. Cascade profile is
> selected via `--profile {smoke_haiku|real}` on
> `scripts/run_paper.py`.

#### X1 — MAP cross-run finding alignment

- **Stage:** MAP
- **Reference:** cascade A's `findings.jsonl` (or B's — symmetrise).
- **P/R/F1:** greedy 1:1 match on `(claim, joined evidence text)`
  embeddings (cosine ≥ θ, proposed 0.80). tp = matched; fp = unmatched in
  prediction; fn = unmatched in reference. **Standard P/R/F1.** Use
  [`eval/silver/matcher.py::compute_sim_matrix`,
  `match_from_matrix`,
  `compute_metrics`](../eval/silver/matcher.py).
  Field-mismatch breakdown via `_field_mismatches`.
- **Artifacts:** `map/{pmcid}/findings.jsonl` from both runs.
- **Model-agnostic?** Yes — neither run is labelled; the metric is symmetric
  agreement.
- **Cost:** medium (embeddings cached).
- **Gotcha:** `finding_id` is **not** stable across cascades (it hashes the
  claim string). Do not attempt an id-join — use the embedding alignment.

#### X2 — NORMALIZE triple-set agreement

- **Stage:** NORMALIZE
- **Reference:** the other run's NORMALIZE output.
- **P/R/F1:** set of `(subject_cui, outcome_normalised, relation_type)`
  triples per run, per paper. tp / fp / fn from set difference. Report
  Jaccard and macro P/R/F1.
- **Artifacts:** `normalize/{pmcid}/normal_findings.jsonl` from both runs.
- **Helper:** stdlib `set`.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** triples ignore `direction` and `scope` deliberately — that's
  what GROUP does. If you need direction in the comparison, switch to X3.

#### X3 — GROUP / CANONICALIZE set agreement

This is the cleanest model-agnostic signal in the whole battery — one set
difference, no fuzzy matching, no NLI, no embeddings.

- **Stage:** GROUP and CANONICALIZE
- **Reference:** the other run's GROUP / CANONICALIZE output.
- **P/R/F1:** exact set comparison of `group_id` (and separately
  `canonical_id`), per paper. Both IDs are **fully deterministic** across
  cascades (§2), so any disagreement is a real disagreement, not
  identifier noise.
- **Artifacts:** `group/{pmcid}/groups.jsonl`,
  `canonicalize/{pmcid}/canonical_rules.jsonl` from both runs.
- **Helper:** stdlib `set`.
- **Model-agnostic?** Yes — exact ID join.
- **Cost:** cheap.
- **Gotcha:** if `group_id` agreement is low, the source of divergence is
  upstream in MAP / NORMALIZE — diagnose with X1 / X2 before chasing GROUP.

#### X4 — RELATE label agreement

- **Stage:** RELATE
- **Reference:** the other run's relations.
- **P/R/F1:** join `relations.jsonl` from A and B on `(rule_id_a,
  rule_id_b)` (both canonical_id-stable). Compute Cohen's κ across the
  4-label scheme (SUPPORT / CONTRADICT / SCOPE_QUALIFY / UNRELATED) plus
  per-label one-vs-rest P/R/F1.
- **Artifacts:** `relate/{pmcid}/relations.jsonl` from both runs.
- **Helper:** stdlib + sklearn's `cohen_kappa_score` or hand-rolled κ.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** restrict the join to pairs present in **both** runs — pairs
  unique to one run aren't comparable.

#### X5 — RESOLVE ranking agreement

- **Stage:** RESOLVE
- **Reference:** the other run's `FinalRule` ranking.
- **P/R/F1:** join `final_rules.jsonl` from A and B on `canonical_id`,
  compute Kendall τ on `score`, plus top-k overlap (Jaccard) at k ∈
  {5, 10, 20}.
- **Artifacts:** `resolve/{pmcid}/final_rules.jsonl` from both runs.
- **Helper:** stdlib + scipy's `kendalltau`.
- **Model-agnostic?** Yes.
- **Cost:** cheap.
- **Gotcha:** rank-only comparison is robust to absolute-score drift; tau
  near 1 with low top-k overlap means small permutations near the cut-off
  — that's expected and fine.

---

## 5. Implementation roadmap

Sequence the implementation cheapest-first so the user gets coverage on
day one and adds the slower experiments incrementally.

**Phase 1 — structural replays (no model calls).** Implementable in one
afternoon, zero API cost.
- GR1, C1, C3, R2, R3, Rs3 — pure replays.
- X2, X3, X4, X5 — set / join differentials (X3 in particular).
- M1, G3 — string containment / span checks.

**Phase 2 — UMLS + embedding replays.**
- N1, N2, N3, GR2, GR3, M2, R4.

**Phase 3 — NLI replays.**
- M3, G1, G2, C2, R1.

**Phase 4 — cross-run alignment.**
- X1 (needs cascade A and cascade B runs on overlapping PMCIDs).

Preconditions across phases:

- `raw_pairs.jsonl` populated for the target run (R1, R4).
- `rejection_reason` populated on rejected findings (G2).
- Two completed runs with different cascades on a shared PMCID set
  (entire X-series).
- One frozen reference configuration recorded per eval pass (NLI model id,
  UMLS / scispaCy version, embedding model, similarity thresholds).

Suggested deliverable: `python -m eval.stage_experiments --run <RUN_ID>
[--baseline <RUN_ID_B>]` emitting per-paper JSON and a run-level Markdown
table (one row per experiment, columns: id, stage, n, precision, recall,
F1 or stability metric, cost USD).

---

## 6. Reusable helpers reference

Frozen reference configuration (record in the report header for every run):

- **NLI model:**
  [`pipeline/stages/summarization/nli_config.py::get_active_spec`](../pipeline/stages/summarization/nli_config.py)
  → `NLIModelSpec(hf_id, batch_size, …)`. Default key from
  `configs/nli_models.yaml`.
- **UMLS:**
  [`pipeline/stages/summarization/umls_resources.py::get_nlp`,
  `get_linker`](../pipeline/stages/summarization/umls_resources.py) (process-wide singleton).
- **Embedding:** see [`eval/silver/embedders.py`](../eval/silver/embedders.py); default
  OpenAI `text-embedding-3-small`, Gemini optional.
- **Thresholds:** M3 / G1 NLI threshold = `cfg.grounding.threshold` of the
  source run; N2 cohesion θ = 0.85 (proposed); GR2 cohesion θ = 0.60
  (proposed); X1 match θ = 0.80 (proposed). Stamp every threshold in the
  report header.

Helper table:

| Need | Helper | Path |
|---|---|---|
| P/R/F1 from tp/fp/fn | `metrics(tp, fp, fn)` | [`eval/precision_recall.py`](../eval/precision_recall.py) |
| Greedy 1:1 matching | `compute_sim_matrix`, `match_from_matrix` | [`eval/silver/matcher.py`](../eval/silver/matcher.py) |
| Aggregate matched P/R/F1 | `compute_metrics` | [`eval/silver/matcher.py`](../eval/silver/matcher.py) |
| Field-level error attribution | `_field_mismatches` | [`eval/silver/matcher.py`](../eval/silver/matcher.py) |
| Embedding cache | `EmbeddingCache` | [`eval/silver/matcher.py`](../eval/silver/matcher.py) |
| NLI inference (pinned) | `GroundingFilter` | [`pipeline/stages/summarization/helpers/grounding_filter.py`](../pipeline/stages/summarization/helpers/grounding_filter.py) |
| Same NLI engine RELATE used | direct import | [`pipeline/stages/summarization/current_stages/relate_stage.py`](../pipeline/stages/summarization/current_stages/relate_stage.py) |
| UMLS linker | `get_nlp`, `get_linker` | [`pipeline/stages/summarization/umls_resources.py`](../pipeline/stages/summarization/umls_resources.py) |
| Finding ID helper | `compute_finding_id` | [`pipeline/stages/summarization/models.py`](../pipeline/stages/summarization/models.py) |
| NLI model pin | `get_active_spec`, `NLIModelSpec` | [`pipeline/stages/summarization/nli_config.py`](../pipeline/stages/summarization/nli_config.py) |
| Scoring formula replay | scoring helpers | [`pipeline/stages/summarization/current_stages/resolve_stage.py`](../pipeline/stages/summarization/current_stages/resolve_stage.py) |

Artifact paths consumed by this battery (placeholder `{pmcid}` per run
under `out/summaries/runs/<run_id>/`):

```
manifest.json
map/{pmcid}/findings.jsonl
map/{pmcid}/chunks.jsonl
map/{pmcid}/rejected_findings.jsonl
normalize/{pmcid}/normal_findings.jsonl
normalize/{pmcid}/entity_links.jsonl
normalize/{pmcid}/dedup_trace.jsonl
group/{pmcid}/groups.jsonl
group/{pmcid}/non_groupable.jsonl
canonicalize/{pmcid}/canonical_rules.jsonl
relate/{pmcid}/relations.jsonl
relate/{pmcid}/raw_pairs.jsonl
relate/{pmcid}/skipped_pairs.jsonl
resolve/{pmcid}/final_rules.jsonl
resolve/{pmcid}/score_trace.jsonl
```

Plus `out/summaries/cascade_decisions/{pmcid}.jsonl` for cross-referencing
specific cascade decisions when a per-chunk drilldown is useful.

---

## See also

- [`ABC_IMPLEMENTATION_COMPARISON.md`](ABC_IMPLEMENTATION_COMPARISON.md) — the cascade audit that motivated this eval design.
- [`../eval/llm_judge/STAGE_EVAL_DESIGN.md`](../eval/llm_judge/STAGE_EVAL_DESIGN.md) — the Opus-silver companion battery covered by this doc's §1 motivation.
- [`STRUCTURE.md`](STRUCTURE.md) — pipeline changelog (this doc is recorded there).
- [`THESIS.md`](THESIS.md) — TODOs tracking individual experiment IDs.
