# Thesis Notebook — Issues found, decisions made

A running ledger of substantive issues discovered in the `nlp-histo` pipeline,
the diagnosis trail for each, the fix shipped, and the artifacts that
demonstrate the before/after behaviour. Reproduction commands live in
[`HOW_TO_RUN.md`](HOW_TO_RUN.md); code layout lives in
[`STRUCTURE.md`](STRUCTURE.md); the bug catalogue and per-bug write-ups live
in [`BUGS.md`](BUGS.md).

> **How to use this file:**
> * [Bugs](BUGS.md) — every substantive defect, catalogue + per-bug detail.
> * [TODOs](#todos) — work to carry forward into the thesis or follow-up commits.
> * [Decisions log](#decisions-log) — non-obvious calls made along the way.
>
> Add a new entry the same day the issue is discovered. New bug → row in
> [`BUGS.md`](BUGS.md) catalogue + detail section. New TODO → bullet in the
> TODO list below (with the bug or section it belongs to). Never delete
> entries — flip `Status` to `Fixed` / `Won't fix` / `Superseded` so the
> history survives.

---

## TODOs

Carry-forward items. Tick `- [x]` when shipped, then move the entry to the
"Decisions log" if it represents a permanent decision, or to the matching Bug
detail section in [`BUGS.md`](BUGS.md) if it closes a bug.

* [ ] **B-001 follow-up:** regenerate `out/summaries/runs/*/` artifacts for papers that participated in a corpus-relate run *before* the `pmcid`-in-`group_id` fix — their `canonical_id`s are still colliding values on disk.
* [ ] **B-001 defence-in-depth:** in [`pipeline/stages/summarization/helpers/corpus_relate.py`](../pipeline/stages/summarization/helpers/corpus_relate.py), guard against the case where two rules in `all_rules` happen to share a `canonical_id` (impossible after B-001 fix, but cheap to assert).
* [ ] **B-002 audit:** for every paper in production, count how often `ContextAwareStitcher` had to absorb a phantom element pre-fix vs. post-fix. If the post-fix count is materially lower it strengthens the result in the thesis.
* [ ] **B-003 refactor:** reconsider the blanket `NodeScorer._ALWAYS_KEEP` exemption for `SECTION_HEADER`. R1 and R3 are safe to apply to headers; only R-color needs the exemption, and R-color is now disabled by default. Per-rule exemptions would be cleaner.
* [ ] **Summarisation throughput:** the cheap-tier MAP cascade hits the L3 escalation more than expected on multi-claim sentences. Worth profiling once cost-percentile sweep ([`scripts/estimate_pipeline_cost_percentiles.py`](../scripts/estimate_pipeline_cost_percentiles.py)) lands.
* [ ] **Thesis figure polish:** the `out/thesis_demo/ghost_text/*.png` crops are 110-dpi; bump to 200-dpi when finalising thesis figures so banner type stays sharp at print size.
* [ ] **B-005 dedup:** the persistence + verbatim-from-DB helpers are line-for-line copies between `SummarizationRunner` and `BatchSummarizationRunner`. Lift into `pipeline/stages/summarization/persistence.py` as module-level functions taking `db` as a parameter, then point both runners at the shared impl. Markers: `TODO: deduplicate with SummarizationRunner` in `batch/runner.py`.
* [ ] **B-005 end-to-end verification:** run `scripts/run_paper.py PMC<x> --sync=false` against a real DB with `db=get_db_connection()` wired in, confirm `sum_map_findings`/`sum_normal_findings`/…/`sum_rejection_summaries` get rows for `pipeline_run_id`, and spot-check `sum_map_findings.verbatim_support == text_element.text_content` for at least one finding. `scripts/run_paper.py:build_batch_runner` does **not** wire `db` by default — only `scripts/run_paper_single_model.py` does.
* [x] **B-006:** decide policy on `SCOPE_QUALIFY` — emit it from `_classify_pair` or tear the plumbing out. — **Stripped 2026-05-14.** Enum value removed, RESOLVE filter / RELATE log column dropped; `FinalRule.scope_qualify_count` + DB column retained as hard-zero for back-compat.
* [x] **B-007:** stamp + check `pipeline_config_hash` in `SummarizationRunner._save_result` / `_load_result`. Strict parity with batch runner. — **Shipped 2026-05-14.** Added `_pipeline_config_hash()` on sync runner; hash now includes `enable_router` state on both runners; manifest builder reuses the same helper.
* [ ] **B-008:** tag cache-hit return value with `status="skipped"` so `process_batch`'s skip counter is non-zero.
* [ ] **B-009:** stop keeping per-paper state on `self._*` in `SummarizationRunner`. Either pop at end of `process()` or localise.
* [ ] **B-010:** rewrite `filter_artifacts` to return indices (not filtered dicts) so `artifact_filter.py:59` rebuilds by identity, not dict equality.
* [ ] **B-011:** either delete `ModelRegistry.docling_converter` or bring it up to parity with `DoclingLayoutExtractor._get_converter`.
* [ ] **B-012:** rewrite `two_pass_extractor` header/footer strip in pure fitz coords, convert once with `BoundingBox.from_fitz_rect`.
* [ ] **B-015 propagation:** surface `raw_{relation_type,direction,category}` on `NormalFinding` / `FindingGroup` / `CanonicalRule` (and their DB tables) so audit trails survive past MAP. Currently only `sum_map_findings` has the raw columns.
* [ ] **B-015 audit query:** add a thesis-table query that buckets `(raw_relation_type, relation_type)` pairs across a corpus to expose systematic out-of-enum tendencies — useful evidence for "the enum is too narrow" or "X model prefers label Y".
* [ ] **B-015 confidence:** `Finding.confidence` is a strict `Literal` with no coercion; out-of-vocab values cause the whole finding to be dropped (`bad_findings.jsonl`). No raw column added because successful rows would always equal `confidence`. Revisit if we ever case-fold / coerce confidence.
* [ ] **B-016 DB backfill:** existing rows in `sum_map_findings.category` from pre-2026-05-14 runs still hold `"demographics"`. Either re-run MAP (default plan) or `UPDATE sum_map_findings SET category='demographic' WHERE category='demographics'`.
* [ ] **B-016 silver eval:** `eval/silver/prompts.py` still emits `"demographics"` (own `PROMPT_VERSION = v2`). Main pipeline's alias-repair handles it, but for cleanliness align + bump silver prompt to `v3` when next regenerating silver labels.
* [ ] **ABC P1 — structured agreement:** extend agreement scoring beyond `Finding.claim` embedding similarity to a weighted combination of entity match, relation type, polarity, scope, evidence overlap, and semantic/NLI score. `HybridStructuredSimilarity` (in `pipeline/stages/summarization/agreement/hybrid_structured.py`) already covers category + claim-emb + entity + evidence; add explicit polarity + scope components and promote to default scorer via `SemanticAgreementScorer(strategy=HybridStructuredSimilarity(...))`. Detail and proposed weights in [`ABC_IMPLEMENTATION_COMPARISON.md` §7](ABC_IMPLEMENTATION_COMPARISON.md#7-proposed-agreement-score).
* [ ] **ABC P1 — lightweight comparison-normalisation in MAP:** before computing agreement, apply a cheap normaliser (case-fold + dictionary-snap polarity + category alias snap) so two voters saying the same thing in different surface forms don't escalate spuriously. Must not be persisted — full `NORMALIZE` still runs after MAP. Sketch in [`ABC_IMPLEMENTATION_COMPARISON.md` §6](ABC_IMPLEMENTATION_COMPARISON.md#6-recommended-target-design-for-our-pipeline).
* [ ] **ABC P1 — hard-fail rules:** when a router-eligible voter pair shows opposite polarity on the same entity, force escalation regardless of weighted score. Likewise for evidence-disjoint pairs. Today `EmbeddingScorer._polarity` is a soft 20% multiplicative penalty; a polarity contradiction can still pass `theta=0.8`. Wire into `AgreementChecker` or a wrapping scorer. See [`ABC_IMPLEMENTATION_COMPARISON.md` §7](ABC_IMPLEMENTATION_COMPARISON.md#7-proposed-agreement-score) hard-fail block.
* [ ] **ABC P1 — joint theta sweep:** extend `eval/silver/map_theta_sweep.py` to (i) sweep `(theta, reject_theta)` jointly, (ii) sweep alternate scorers (`SemanticAgreementScorer` over `EmbeddingSimilarityStrategy` vs. `HybridStructuredSimilarity`), and (iii) add a silver-judge precision column per row so we can pick a safe-deferral threshold instead of guessing `MapConfig.theta=0.8`. See [`ABC_IMPLEMENTATION_COMPARISON.md` §8](ABC_IMPLEMENTATION_COMPARISON.md#8-evaluation-plan).
* [ ] **ABC P1 — accepted-vs-L3 silver eval:** quantify the "is cascade safe?" question by replaying cached voter outputs through the silver judge for chunks where L1/L2 KEPT vs. chunks that escalated to L3. Reuse `eval/silver/` machinery. The cascade decision JSONL (`out/summaries/cascade_decisions/{pmcid}.jsonl`) gives the per-chunk acceptance level needed to bucket.
* [ ] **ABC P1 — pair-breakdown in reports:** surface `EmbeddingScorer.score_details.pairwise_upper` (already populated in `AgreementTrace`) into the HTML batch index and the cost report so disagreement cases can be audited without re-running MAP. Touches `scripts/templates/pipeline_batch_index.html.jinja2` and the trace/report glue in `pipeline/stages/summarization/observability/`.

---

## Decisions log

Permanent design calls — keep terse, link to the discussion in the detail section.

| Date | Decision | Rationale | Detail |
|------|----------|-----------|--------|
| 2026-05-13 | `group_id` includes `pmcid` in its hash. | Prevents cross-paper `canonical_id` collisions; cross-paper matching uses the dedicated CUI / normalised-string gate in `corpus_relate.py`, not id equality. | [Bug 1](BUGS.md#bug-1--duplicate-intra-paper-relations-produced-by-canonical_id-collisions) |
| 2026-05-13 | `TwoPassConfig.enabled = True` by default. | R1 (pixel render) reliably catches Docling phantom elements and PDF render-mode-3 / fill-opacity-0 ghost text. The stitcher no longer has to be the only line of defence. | [Topic — ghost-text detection](BUGS.md#topic--ghost-text-detection-empirical-verification-and-policy-fix) §2.3 |
| 2026-05-13 | `TwoPassConfig.max_white_char_fraction = 1.0` (R-color disabled). | Empirically produces false positives on legitimate inverted headers; R1 is the source of truth. Re-enable per-corpus only if R1 is unavailable. | [Bug 3](BUGS.md#bug-3--r-color-white-text-false-positive-latent) |
| 2026-05-13 | Don't add a pdfminer.six render-mode pass. | R1 already catches Tr=3 and opacity-0 as "no ink in bbox" — verified synthetically (`scripts/verify_ghost_text_detection.py`). A second detector would be duplicated work. | [Topic — ghost-text detection](BUGS.md#topic--ghost-text-detection-empirical-verification-and-policy-fix) §2.1 |
| 2026-05-14 | Batch runner brought to parity with sync runner. | Six features (verbatim-from-DB, stable `finding_id`, DB persistence, `corpus_relate_incremental`, `rejection_summary`, NER) were sync-only; production batch runs were grounding against paraphrased text. Copied the methods directly into `batch/runner.py` rather than refactoring sync to keep blast radius small. | [Bug 5](BUGS.md#bug-5--batch-runner-missing-sync-parity-features) |
| 2026-05-14 | Batch result caching uses `pipeline_config_hash` invalidation. | Stamping the result JSON with the current config hash + checking it at `submit()` is the cheapest correct cache key. Any threshold/model/schema/prompt change auto-invalidates without manual cleanup. Pre-fix JSONs in `out/summaries/summaries/` were deleted to force regeneration. | [Bug 5](BUGS.md#bug-5--batch-runner-missing-sync-parity-features) |
| 2026-05-14 | Default MAP scorer switched to `SemanticAgreementScorer(EmbeddingSimilarityStrategy)`. | Centrality-based best-output selection (Soiffer 2025). Previous default `EmbeddingScorer` left `bundle.best_index` unset, so `AgreementChecker.best()` fell back to a `(mean_evidence_length, n_findings)` heuristic — which can be gamed by a voter that copies the whole paragraph as evidence. | [`ABC_IMPLEMENTATION_COMPARISON.md` §5 Gap 1](ABC_IMPLEMENTATION_COMPARISON.md#gap-1-no-centrality-based-output-selection-on-default-path) |
| 2026-05-14 | `MapOutputRouter` wired into both runners by default. | Schema + provenance validation runs before agreement; unusable voters drop out before scoring instead of inflating the agreement matrix with fabricated content. Router path escalates L1 → L3 directly because L2 voters are the same risk class as L1. Opt-out via `enable_router=False`. | [`ABC_IMPLEMENTATION_COMPARISON.md` §5 Gap 2 + Gap 8](ABC_IMPLEMENTATION_COMPARISON.md#gap-2-grounding-does-not-gate-the-cascade) |
| 2026-05-14 | Per-level cascade decision factored into `agreement/decision.py::evaluate_chunk`. | Sync (`MapStage._cascade`) and batch (`BatchSummarizationRunner._process_level`) previously re-implemented the KEEP/escalate decision inline — a recurring source of drift. Both runners now call the same function, so a paper processed in sync vs batch takes the same cascade path by construction. | [`ABC_IMPLEMENTATION_COMPARISON.md` §5 Gap 3](ABC_IMPLEMENTATION_COMPARISON.md#gap-3-sync-and-batch-run-different-cascade-code) |
| 2026-05-14 | Per-chunk cascade decision log: one JSONL row per L1/L2/L3 decision. | `CascadeDecisionLog` writes to `output_dir/cascade_decisions/{pmcid}.jsonl` independently of `trace_enabled`. Enables offline aggregation of acceptance rate, gate origin, reason codes, and selected provider/model per chunk — needed before any theta sweep can claim "safe deferral". | [`ABC_IMPLEMENTATION_COMPARISON.md` §9 P0 task 4](ABC_IMPLEMENTATION_COMPARISON.md#p0--must-have-before-expensive-experiments) |
