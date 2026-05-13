# TODO — summarization pipeline

Captured 2026-05-13 during Run A smoke session.

## Evaluation harness — NLI model comparison

Goal: run the full pipeline across the candidates in `configs/nli_models.yaml`
and produce a side-by-side precision / recall / F1 report.

- Define the eval target(s). Options:
  - **GROUNDING decisions** — per-finding entailment pass/fail vs labels in
    `eval/data/silver_findings.jsonl` or `pipeline_findings.jsonl`. Cheapest
    signal, isolates NLI quality directly.
  - **Final rules vs silver** — compare RESOLVE output to silver. Captures
    end-to-end NLI effect; mixes in MAP/NORMALIZE/GROUP confounds.
  - **RELATE relations vs gold pairs** — needs labeled pairs (probably hand-
    annotate ~50).
  - Recommendation: run all three, report side-by-side.
- Run strategy: reuse cached MAP outputs (`out/summaries/pipeline_cache.json`)
  and re-run only GROUNDING → ... → RESOLVE per candidate. Fast, no API cost.
  Sweep with `NLP_HISTO_NLI_MODEL=<key>` and a pinned `--artifact-run-id` per
  candidate (e.g. `runEval_<nli_key>`).
- Output: `out/eval/nli_comparison_<ts>.{json,csv}` — per-candidate metrics +
  agreement-distribution histograms.

## Threshold recalibration for the active NLI model

Defaults were tuned for DeBERTa-large logits and are unlikely to be optimal
for PubMedBERT-MNLI-MedNLI.

- Plot grounding-score distribution from a Run A artifact:
  ```
  python -c "import json,statistics; \
    scores=[json.loads(l)['grounding_score'] for l in open(...findings.jsonl) \
      if json.loads(l).get('grounding_score') is not None]; \
    print(statistics.quantiles(scores, n=10))"
  ```
- Sweep `entailment_threshold` ∈ {0.3, 0.4, 0.5, 0.6, 0.7} and
  `contradiction_threshold` ∈ {0.5, 0.6, 0.7, 0.8}; report ROC/F1 per choice.

## UMLS RAM reductions

Biggest single RAM hog. Three levers in order of effort:

1. **DB-backed UMLS lookup** (`NLP_HISTO_UMLS_SOURCE=db`):
   - Build `{lower(entity_text): (cui, canonical_name)}` from the
     `entities` table at NORMALIZE start.
   - Override `_umls_canonical_with_cui` to hit the dict; never call
     `get_nlp()` when source=db.
   - Misses fall through to dict-only / synonyms.yaml.
   - Saves ~7 GB. Quality drops on LLM paraphrases that don't match any
     NER-tagged span. Measure coverage before committing.
2. **Persistent UMLS cache + lazy KB load**:
   - Persist `_UMLS_CACHE` to `out/summaries/umls_cache.json` between runs.
   - Defer `get_nlp()` call until first cache miss → for stable corpora the
     KB never loads after warmup.
3. **Unify spaCy model** (small):
   - Replace `spacy.load("en_core_sci_sm")` at `runner.py:801` with
     `umls_resources.get_nlp()` so the sentence splitter shares the
     singleton. Free 50 MB; mostly a tidiness win. Guard against
     `--disable-umls` returning None.

## Pipeline resume / checkpointing

Persistence is observational only (`persistence.py:3`); downstream stages
re-run on every finalize. Concrete win: read stage JSONL back as inputs.

- Per-stage skip predicate: if `runs/<run_id>/<stage>/<pmcid>/<artifact>.jsonl`
  exists and matches a fingerprint, load it instead of recomputing.
- Required to make pinned `--artifact-run-id` reruns actually cheap.
- Particular value for NLI sweeps (skip MAP entirely, run from GROUNDING).

## NLI redundancy / caching

Model loads once per process via `_NLI_PIPE_CACHE`, but inference re-runs
every time. Concrete wins:

- **Persist NLI scores per pair** so reruns don't recompute. RelateStage
  already writes `relate/<pmcid>/raw_pairs.jsonl` with entailment /
  contradiction scores; treat it as a read-back cache keyed by
  `(rule_id_a, rule_id_b, model_name)`. Threshold sweeps then become free.
- **Dedup corpus-relate intra-paper pass.**
  `helpers/corpus_relate.py:360-364` runs a second RelateStage over a
  paper's own rules (`new_rules`) to produce the intra-paper neighborhood
  — those pairs were already scored by the per-paper RELATE during
  `SummarizationRunner` finalize. Reuse the per-paper `raw_pairs.jsonl`
  (or the in-memory `self._relate_raw_pairs[pmcid]`) instead of running
  NLI again.
- **Cache GROUNDING scores per (verbatim_support, claim).** Many findings
  share verbatim spans across reruns; persistable similarly to NLI raw
  pairs.

## Corpus-relation export

`_corpus_relate_incremental` writes only to `sum_corpus_relations`. The HTML
inspector wants a `corpus_relations.json` file. Workaround today: rerun
`python -m pipeline.stages.summarization.helpers.corpus_relate ...`
(re-runs NLI). Better:

- Add an exporter: read `sum_corpus_relations` for the latest run per pmcid
  → write `out/summaries/corpus_relations.json` in the schema the inspector
  expects.

## Known cleanup

- `run_paper.py --profile` help text lists stale names (smoke_haiku,
  dev_sonnet, final_opus, default). Only `cheap` and `real` resolve in
  `voter_configs.py`. Update help string + dry-run "Env vars required" line
  (currently always says ANTHROPIC_API_KEY even for cheap profile).
- `--from-selection` always loads all PMCIDs from the YAML. Add `--limit N`
  or `--bucket related|diverse|hard` so we don't need throwaway YAMLs to
  run a 2-paper subset.

## Open follow-ups from corpus-relate quality review

The fixes below addressed gate weakness in corpus-relate. Remaining issues
(NORMALIZE over-collapsing, predicate-level prefiltering) still produce
nonsense pairs even with a tighter gate.

- **NORMALIZE: stop collapsing distinct subjects to generic UMLS umbrellas.**
  Observed: separate biomedical concepts being mapped to "Body tissue",
  "Techniques", "Tissue Fixation", etc. — washing out specificity so the
  cross-paper gate still pairs unrelated rules. Add a specificity guard:
  if the chosen UMLS canonical name is significantly more generic (shorter
  / higher in the hierarchy) than the original entity, keep the original.
- **Predicate-level prefilter before NLI.** Add a TF-IDF / embedding
  similarity gate on `predicate_text` so pairs with low predicate overlap
  never reach NLI. Cheap (~ms per pair), reduces NLI noise dramatically.

## Done in this session

- Switched default NLI model to `pritamdeka/PubMedBERT-MNLI-MedNLI`
  (`configs/nli_models.yaml` + `pipeline/stages/summarization/nli_config.py`).
- GROUNDING and RELATE now share one NLI pipeline instance (saves ~1.5 GB
  vs prior dual-DeBERTa).
- `configs/paper_selection/runA.yaml` — 2-paper smoke selection.
- Archived stale summaries under
  `out/summaries/summaries/_archive_pre_runA/`.
- **Legacy REDUCE / RULES output suppressed in per-paper JSON unless
  `run_reduce=True`.** Previously, every run wrote `summary: null`,
  `rules: []`, `contradiction_report: null`, plus
  `audit_trail.master_summary: null` and `audit_trail.rules_provenance: null`
  even when the legacy block wasn't run, which produced the misleading
  "0 rules extracted" / "(no summary)" lines.
  - Files touched: `pipeline/stages/summarization/batch/runner.py`,
    `pipeline/stages/summarization/runner.py` — result dict no longer
    contains those keys by default; they are added only when
    `self._run_reduce` is True.
  - `scripts/run_paper.py` — print loops in `_run_sync`, `_run_batch`,
    `_run_all_batch` now print `canonical_rules / relations / final_rules`
    counts instead of legacy `summary` / `rules`.
  - **To reverse:** move the conditional `if self._run_reduce: result[...] = ...`
    block back into the result dict literal in both runners; restore the
    pre-edit print loops in `run_paper.py` (use `git diff` to recover the
    exact lines).

- **Corpus-relate gate tightened.** Two fixes in
  `pipeline/stages/summarization/helpers/corpus_relate.py`:
  1. `_should_compare_cross_paper` no longer skips the subject gate when
     either rule lacks a CUI — it now falls back to normalized subject
     string match. Outcome gate now applies to **all** relation_types
     (previously only `expression`), with `_norm_outcome_expression` for
     expression and `_norm_outcome` otherwise.
  2. The intra-paper pass inside `relate_incremental` (line ~363) now
     uses the strict `_should_compare` gate instead of
     `_should_compare_cross_paper`, restoring parity with per-paper
     RelateStage. Previously the CLI corpus_relate produced 260
     intra-paper relations that the per-paper stage had already rejected.
  - **To reverse:**
    Revert the `_should_compare_cross_paper` body to the prior CUI-only
    logic (`if a.subject_cui and b.subject_cui:` without the else branch;
    outcome gate inside `if a.relation_type == RelationTypeEnum.expression:`).
    In `relate_incremental` change `gate=_should_compare` back to
    `gate=_should_compare_cross_paper`. Imports at the top of the file
    can be reduced back to `RelateStage, _norm_outcome_expression`.

- **RELATE: dropped SCOPE_QUALIFY emit branch (Path A).** Asymmetric
  entailment now classifies as UNRELATED (None) instead of SCOPE_QUALIFY,
  so the output label set aligns with 3-class NLI gold
  (SUPPORT ≈ entailment, CONTRADICT ≈ contradiction, UNRELATED ≈ neutral).
  - Files touched: `pipeline/stages/summarization/current_stages/relate_stage.py`
    (deleted the 3-line `if ent_ab >= ... or ent_ba >= ...` branch and
    updated the `_classify_pair` docstring).
  - Schema, enum, `FinalRule.scope_qualify_count`, DB column, HTML
    inspector column, eval/silver counts, and `eval/llm_judge/prompts.py`
    label list are **unchanged**. `scope_qualify_count` will be 0 on all
    new runs; old rows preserved.
  - **To reverse:** re-add the deleted branch in `_classify_pair`:
    ```python
    # Asymmetric entailment → scope qualification
    if ent_ab >= entailment_threshold or ent_ba >= entailment_threshold:
        return RelationTypeLabel.SCOPE_QUALIFY
    ```
    Place it immediately before `return None  # UNRELATED` and restore
    the corresponding docstring line. No other file needs changing.
  - **Path B (full removal of the label/enum/column)** is documented
    above under "Done cleanup" — not applied yet; do only after
    confirming SCOPE_QUALIFY-touching consumers (silver sweep, LLM judge
    prompts, HTML templates) are updated.
