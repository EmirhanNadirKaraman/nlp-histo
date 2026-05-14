# ABC Implementation Comparison

Audit of `nlp-histo`'s Agreement-Based Cascading (ABC) implementation against the
recommendations in:

1. **Kolawole et al. 2025** — "Agreement-Based Cascading for Efficient Inference"
   (classic ABC, structured short-output tasks).
2. **Soiffer et al. 2025** — "Semantic Agreement Enables Efficient Open-Ended LLM
   Cascades" (semantic ABC, mean-pairwise / centrality, max-consensus).

This is an implementation audit, not a literature review. All claims about the
current codebase are backed by file references at HEAD of `eval-speedrun`.

---

## 1. Executive Summary

- **What we have:** A **partial hybrid** between classic ABC and semantic ABC.
  We have a real 3-tier cascade, embedding-based pairwise similarity over claim
  lists, an explicit deferral score, and a working `theta` knob. We have one
  scorer (`SemanticAgreementScorer`) that already implements Soiffer-style
  max-consensus + centrality selection — but **it is not the one wired into the
  production runners**. The sync (`MapStage`) and batch (`BatchSummarizationRunner`)
  defaults use `EmbeddingScorer`, which is mean-pairwise without centrality
  selection; "best" falls back to a hand-rolled `_quality_key` heuristic in
  `agreement/checker.py::_quality_key`.
- **Already aligned:** tiered cascade (L1/L2/L3), heterogeneous L1 voters across
  providers, deferral score in `[0, 1]`, `theta` + `reject_theta` thresholds,
  cache keyed on `cascade_signature`, NLI grounding via cross-encoder DeBERTa,
  schema + provenance validation in `routing/`, theta sweep harness
  (`eval/silver/map_theta_sweep.py`).
- **Biggest missing pieces:**
  1. The production cascade does **not** select the most-central output per
     Soiffer — it picks by `(mean_evidence_chain_length, n_findings)`.
  2. The grounding-first router (`MapOutputRouter`) is implemented but **not
     wired** in either the sync runner (`SummarizationRunner.__init__` never
     passes `router=`) or the batch runner (`BatchSummarizationRunner._process_level`
     constructs its own bare `AgreementChecker`). Schema / provenance validation
     is therefore dead code on the default path.
  3. Sync and batch paths run **different agreement code**. Sync goes through
     `MapStage._cascade` (which can route, retry voters, do in-run dedup, etc).
     Batch implements its own minimal `_process_level` and does **not** run the
     router, does **not** apply grounding pre-MAP, and does **not** carry an
     `AgreementContext`.
  4. Agreement is computed over `Finding.claim` text only. There is no
     structured agreement over (entity, relation, polarity, scope) inside MAP.
     The polarity heuristic in `agreement/embedding.py::_polarity` is a
     keyword-list contradiction penalty, not a true polarity-agreement signal.
  5. Grounding is applied **after MAP, not before accepting cheap-tier output**.
     A high-agreement-but-hallucinated L1 result is accepted, then trimmed
     post-hoc by `GroundingFilter` — which is too late for the cascade
     decision itself.
- **Fix first:** wire the existing `MapOutputRouter` (or at least its
  `ProvenanceValidator`) into both runners and adopt
  `SemanticAgreementScorer` as the default scorer so centrality selection
  matches what we already cite. Everything else (structured agreement, theta
  sweep instrumentation, evaluator) builds on top of that.

---

## 2. Paper Recommendations in Implementation Terms

### Kolawole et al. (classic ABC)

| # | Engineering requirement |
|---|---|
| K1 | A tiered cascade of increasingly expensive models. |
| K2 | Ensemble agreement is the deferral rule (not self-reported confidence / logprob). |
| K3 | Configurable agreement threshold `theta`; tunable per task. |
| K4 | Accept cheap tier only when ensemble agreement exceeds `theta`; otherwise defer. |
| K5 | Calibrate `theta` so accepted cheap-tier outputs have low expected error ("safe deferral"). |
| K6 | Cost / accuracy tradeoff must be measurable end-to-end. |
| K7 | Works for short, structured outputs where exact / vote agreement is meaningful. |

### Soiffer et al. (semantic ABC)

| # | Engineering requirement |
|---|---|
| S1 | Open-ended outputs: exact agreement is useless; compare meaning. |
| S2 | Pairwise semantic similarity over voter outputs. |
| S3 | Mean pairwise similarity / centrality as the deferral score. |
| S4 | Select the **most central** output (highest mean similarity to peers) when not deferring. |
| S5 | Defer when mean pairwise similarity is below threshold. |
| S6 | Work with black-box APIs: no logprobs needed. |
| S7 | Richer semantic metrics for long outputs; lexical only when outputs are short. |

### Translation for our biomedical extraction use case

Voter output is `AuditableSummary(findings=List[Finding])`, not free text. So
agreement is over a **list of structured records**. Each `Finding` carries
`claim`, `category`, `verbatim_support`, `evidence` (list of `[PMCID|te_id]`
citations), and (post-NORMALIZE) entity / relation fields. Therefore:

- B1 — Agreement should be at finding-level, after a lightweight alignment,
  not over the raw JSON blob.
- B2 — Structured fields to compare: entity (subject/biomarker/morphology),
  relation type (`category`), polarity / direction, scope (subtype / tissue /
  setting), evidence span (`text_element_id` overlap).
- B3 — A grounding check against the cited source sentence must run before
  trusting cheap-tier output. Two models agreeing on a hallucination is the
  classic ABC failure mode the papers warn about.
- B4 — Hard contradiction (same entity, opposite polarity) must force escalation
  regardless of textual similarity.
- B5 — Unparseable / schema-invalid voters must drop out of the agreement set
  before scoring.

---

## 3. Current Implementation Walkthrough

### 3.1 Cascade orchestration (MAP, sync)

- `pipeline/stages/summarization/current_stages/map_stage.py::MapStage`
  is the active sync orchestrator. Three-level cascade is hardcoded in
  `MapStage._cascade()`:
  - L1: every chain in `self._voter_chains` runs in parallel via
    `_run_voters()`.
  - If `self._router is None` (the default path used by both runners), it
    calls `self._agreement.compute(voters, source_text=...)` and accepts on
    `ChunkDecision.KEEP`, else escalates to L2.
  - L2 escalation re-runs `_run_voters()` against `self._level2_voter_chains`,
    then re-computes agreement.
  - L3 is a single call to `self._escalation_chain`.
- `MapStage` has an in-process voter cache keyed on
  `(provider, model, round(temp, 3), sha256(text))` so e.g. Haiku at L1 and
  Haiku at L2 on the same chunk only hit the API once. See
  `MapStage._voter_cache_key`.
- Per-paper escalation counts are exposed via `MapStage.last_escalation_counts`
  and consumed by `scripts/run_paper.py::_escalation_stats_sync`.

### 3.2 Cascade orchestration (MAP, batch)

- `pipeline/stages/summarization/batch/runner.py::BatchSummarizationRunner._process_level`
  is the active batch orchestrator. It does **not** call `MapStage`. It
  re-implements the cascade decision inline:
  - Parses voter outputs from `BatchResult`s.
  - Pre-embeds every unique claim string in one call (good).
  - Constructs **its own** `AgreementChecker(scorer=EmbeddingScorer(_cached_embed),
    theta=..., reject_theta=...)` and ignores the scorer the rest of the
    pipeline may have been configured with.
  - On `ChunkDecision.KEEP` calls `agreement.best(voters, bundle=bundle)` and
    finalises; otherwise escalates to the next level.
  - There is no router, no schema validator, no provenance validator on this
    path.
- Grounding (`GroundingFilter`) is applied **once at finalize time** in
  `BatchSummarizationRunner.finalize` (`batch/runner.py:430-438`), after the
  cascade has already decided to keep or escalate.

### 3.3 Agreement scoring

Multiple scorers exist; only one is wired by default.

| File | Class | Role |
|---|---|---|
| `agreement/checker.py` | `AgreementChecker` | Thin shim: runs `scorer.compute()`, applies `theta`/`reject_theta` when the scorer hasn't set `decision`, picks `best()` from `bundle.best_index` or falls back to `_quality_key`. |
| `agreement/embedding.py` | `EmbeddingScorer` | **Default scorer.** Mean of pairwise soft-aligned cosine similarity over `Finding.claim` embeddings, with `tau` weak-match threshold, count-mismatch penalty, reuse-concentration penalty, and a polarity/numeric contradiction penalty. Does NOT set `bundle.best_index`. |
| `agreement/semantic_scorer.py` | `SemanticAgreementScorer` | **Implemented but unused.** Soiffer-style: build N×N similarity matrix via a pluggable `SimilarityStrategy`, mean off-diagonal per voter, deferral = `max(avg_sim)`, **best = argmax(avg_sim) tie-broken by grounding**. Sets `score_details.pairwise_upper`, `eligible_voter_indices`, `avg_sim`. |
| `agreement/composite.py` | `CascadedCompositeScorer` | LP-optimised thresholds on `(emb, ner)`. Sets its own `KEEP/REJECT/ESCALATE` so `theta` is bypassed. |
| `agreement/hybrid_scorer.py` / `hybrid_structured.py` | `HybridScorer`, `HybridStructuredSimilarity` | Multi-signal: category-Jaccard + claim-embedding + entity-Jaccard + evidence-Jaccard. Plumbed to `EmbeddingSimilarityStrategy.compute_matrix` for batching. |
| `agreement/ner_scorer.py` | `NERScorer` | Pairwise biomedical-entity Jaccard via scispaCy. |
| `agreement/llm_judge.py` | `LLMJudgeScorer` | Uses an LLM as judge between voter outputs. |
| `agreement/embedding_similarity.py` | `EmbeddingSimilarityStrategy` | Implements `SimilarityStrategy` for `SemanticAgreementScorer`. |
| `agreement/calibration/threshold_optimizer.py` | `ThresholdOptimizer` | LP fit of `(keep_emb, keep_ner, reject_emb)` from labelled disagreement data. |

How the wiring lands in production:

- `pipeline/stages/summarization/runner.py:185-189`: when no scorer is passed,
  builds `EmbeddingScorer(embed_fn=embed_fn)`. None of the
  `Hybrid*`/`SemanticAgreementScorer`/`CascadedCompositeScorer` paths are
  selected by `scripts/run_paper.py`.
- `pipeline/stages/summarization/batch/runner.py:130`: hard-codes
  `EmbeddingScorer(self._embed_fn)`. The user-supplied scorer (if any) is
  ignored on the batch path.

`AgreementChecker.best()` is the production selection function. When the
scorer does not set `bundle.best_index` (the case for `EmbeddingScorer`),
`best()` falls back to:

```python
# checker.py::_quality_key
mean_ev = sum(len(f.evidence) for f in o.findings) / len(o.findings)
return (mean_ev, len(o.findings))
```

That is: **prefer the voter with longest mean evidence chain, tie-break on
number of findings**. This is not centrality and is not from any paper.

### 3.4 Grounding / NLI

- `pipeline/stages/summarization/helpers/grounding_filter.py::GroundingFilter`
  — DeBERTa-v3 NLI cross-encoder, runs on `(verbatim_support, claim)` pairs.
  Applied via `filter_findings_with_scores` to every `AuditableSummary` AFTER
  the cascade decision in both runners. Threshold from
  `SummarizationConfig.grounding.threshold` (default `None` → filter disabled
  unless caller sets it).
- `pipeline/stages/summarization/helpers/contradiction_detector.py::ContradictionDetector`
  — runs at RESOLVE stage on `CanonicalRule` pairs; does not feed MAP escalation.
- `pipeline/stages/summarization/routing/provenance_validator.py::ProvenanceValidator`
  — produces `FindingValidation` records with reason codes like
  `FABRICATED_VERBATIM_SUPPORT`, `NONEXISTENT_SOURCE`,
  `CROSS_DOCUMENT_SOURCE_ERROR`, `WEAK_GROUNDING`,
  `PARTIAL_SUPPORT`, `AMBIGUOUS_SUPPORT`, `UNSUPPORTED_CLAIM`. **Only invoked
  when `MapOutputRouter.route()` is called** — i.e. only when `MapStage` is
  constructed with `router=<MapOutputRouter>`.

### 3.5 Routing layer (currently optional)

- `pipeline/stages/summarization/routing/router.py::MapOutputRouter` —
  grounding-first decision flow:
  1. `_classify_voters()` tiers voters as ELIGIBLE / WEAKLY_GROUNDED / UNUSABLE
     using `SchemaValidator` (`routing/schema_validator.py`) and
     `ProvenanceValidator`.
  2. `_chunk_decision_from_classifications()` returns REJECT / ESCALATE early
     when `N_eligible < 2`.
  3. `_agreement_gate()` calls `AgreementChecker.compute` with an
     `AgreementContext` carrying per-voter grounding metadata, then maps
     `ChunkDecision` → `RoutingDecision`.
- `MapStage._cascade` has a router-aware branch (`if self._router is not None`)
  that skips L2 and goes straight L1 → L3 when escalating. Unused on the
  default path.
- `routing/routing_dataset.py::RoutingDataset` collects router decisions for
  offline analysis.

### 3.6 Theta, calibration, sweeps

- `pipeline/stages/summarization/config.py::MapConfig` defaults:
  `theta=0.8`, `reject_theta=0.2`, `chunk_size=10`, `chunk_overlap=2`,
  `chunk_workers=5`.
- `eval/silver/map_theta_sweep.py` — submits all L1+L2+L3 batches up-front so
  any theta value can be replayed offline. Grid:
  `[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]`. Uses `EmbeddingScorer` for
  scoring (so the sweep does **not** sweep `SemanticAgreementScorer` or any
  structured scorer).
- `pipeline/stages/summarization/agreement/calibration/threshold_optimizer.py`
  — LP optimiser that fits `(keep_emb, keep_ner, reject_emb)` thresholds for
  `CascadedCompositeScorer`, but those thresholds are not loaded by the
  production builders.

### 3.7 Cost / latency tracking

- `pipeline/stages/summarization/costing/` (`UsageCollector`, pricing) wires
  callbacks via `LangChainCallback`-style attachment in
  `MapStage._run_voters`. Records cache hits.
- `scripts/run_paper.py::_escalation_stats_sync` / `_escalation_stats` /
  `_save_escalation_report` produce the per-paper JSON + CSV report and a
  "baseline-vs-cascade" cost delta. Baseline cost extrapolates by dividing L1
  totals by `len(make_l1_voters())` — a rough proxy.
- Per-chunk latency is captured in `_run_voters` (`timings[i]`) and
  forwarded to `VoterTrace.latency_ms` when traces are enabled.

### 3.8 Cache

- `pipeline/stages/summarization/cache.py::PipelineCache` keys MAP entries on
  `schema_version | prompt_version | stage_name | cascade_signature | input_hash`
  where `input_hash = sha256(sorted(text_element_ids))`. Cascade reshuffles
  and schema/prompt bumps invalidate naturally.
- `MapStage._process_inner` rebuilds `chunk_id` after a cache hit
  (`hit.model_copy(update={"chunk_id": f"C{abs_idx + 1}"})`) so the chunk_id
  field cannot drift across runs.
- Risk: the `input_hash` ignores sentence **text** — only `text_element_id`.
  If the same `te_id`s ever get re-parsed (e.g. PDF re-extraction with
  different sentence boundaries), cached MAP outputs will still hit. Out of
  scope for the MAP cascade per se, but worth flagging because it would mask
  cascade behaviour during evals.

### 3.9 Sync / batch parity

- Cascade decision and escalation: both use `AgreementChecker` +
  `EmbeddingScorer` + `theta` / `reject_theta`. Same default knobs.
- **Diverges:** sync goes through `MapStage` (router-aware, chunk-level
  parallelism, voter retries, in-run voter dedup); batch goes through
  `BatchSummarizationRunner._process_level` (no router, no per-voter retry,
  no dedup until escalation, different embed-cache strategy).
- **Diverges:** grounding ordering. Sync runs grounding after MAP in the
  runner (`runner.py` orchestration). Batch runs grounding at `finalize()`.
  In both cases, grounding does not influence the cascade decision.
- **Same:** no router, no semantic-best selection, no structured agreement.

---

## 4. Comparison Table

Status legend: ✅ Implemented · 🟡 Partially · ❌ Missing · ⚠️ Implemented but risky/unclear.

| Recommendation | Paper | Current Implementation | Status | Gap / Risk | Suggested Fix |
|---|---|---|---|---|---|
| Multiple cheap models at L1 | K1 | `voter_configs.make_l1_voters()` = Gemini-Flash-Lite + GPT-4o-mini + GPT-4.1-nano | ✅ | None | — |
| Tiered escalation to stronger models | K1 | L1 → L2 → L3 in `MapStage._cascade` and `BatchSummarizationRunner._process_level` | ✅ | Router path skips L2 | Document |
| Agreement threshold `theta` | K3 | `MapConfig.theta=0.8`, `AgreementChecker.theta` | ✅ | Not calibrated to safe-deferral target | Theta sweep with quality target |
| Exact/vote agreement | K2 | None at chunk level; not appropriate for open-ended JSON | ❌ | — | Skip; semantic is right tool here |
| Semantic agreement | S1, S2 | `EmbeddingScorer` (mean pairwise soft-alignment) | 🟡 | Only over `Finding.claim` text | Add structured signals (see §6) |
| Pairwise output similarity | S2 | `EmbeddingScorer._align_precomputed_with_breakdown` | ✅ | Records breakdown but no aggregate matrix | Persist N×N for audit |
| Mean pairwise centrality | S3 | `SemanticAgreementScorer` computes `avg_sim`, but not wired | 🟡 | Default uses mean-of-pairs without per-voter `avg_sim` | Make `SemanticAgreementScorer` the default |
| Highest-centrality output selection | S4 | `SemanticAgreementScorer._select_best` does argmax(avg_sim) — unused. Production uses `_quality_key` (mean evidence length, n findings) | ❌ | Selection is not centrality | Adopt `SemanticAgreementScorer` |
| Structured-field agreement | B1, B2 | `HybridStructuredSimilarity` exists (category + claim-emb + entity + evidence); not default | 🟡 | No relation / polarity / scope signal yet | Promote + extend (see §7) |
| Entity normalization before agreement | B2 | Full NORMALIZE only runs after MAP. NER scorer (`agreement/ner_scorer.py`) extracts raw spans but doesn't UMLS-link them | 🟡 | Surface-form mismatches inflate disagreement | Lightweight comparison-only normalisation inside MAP |
| Relation / polarity agreement | B2 | `EmbeddingScorer._polarity` is a contradiction *penalty*, not a relation-type agreement signal. `Finding.category` is part of `HybridStructuredSimilarity` (Jaccard). | 🟡 | Penalty applies only to embedding-pair score; not a first-class component | Add explicit polarity & relation-type scores |
| Scope agreement (subtype, tissue, setting) | B2 | None | ❌ | — | Add scope extractor (regex / dictionary) inside MAP |
| Evidence-span overlap | B2 | `_evidence_jaccard` in `HybridStructuredSimilarity`; not in default scorer | 🟡 | Easy win | Move into default agreement formula |
| NLI / embedding agreement | S6, S7 | NLI in `GroundingFilter` only (post-MAP). Pairwise NLI between voters not done | 🟡 | NLI never used to compare voters | Optional NLI-on-disagreement layer |
| Grounding check before accepting cheap output | B3, K5 | Grounding runs AFTER cascade decision in both runners | ⚠️ | Two voters agreeing on a hallucination get accepted | Wire `MapOutputRouter` (grounding-first) into both runners |
| Hard contradiction escalation | B4 | `EmbeddingScorer` polarity penalty (max 20% score reduction); no hard-fail | ❌ | Strong contradictions can still score above theta | Hard-fail rule overriding weighted score |
| Cost tracking | K6 | `costing/UsageCollector` + `scripts/run_paper.py` reports | ✅ | Baseline cost is a rough estimate | Track per-decision cost |
| Latency tracking | K6 | `VoterTrace.latency_ms` in `MapStage._run_voters`; not in batch | 🟡 | Batch path has no per-voter latency | Add latency capture to batch path |
| Sync / batch behavioural parity | K6 | Two implementations; only the bare `EmbeddingScorer` + theta path is shared | 🟡 | Same theta, different decision surface | Route batch through `MapStage` or factor common decision function |
| Cache invalidation safety | K6 | Cascade signature + schema/prompt versions in key | ✅ | `input_hash` ignores actual sentence text | Add text hash for paranoia (low priority) |
| Theta sweep / calibration | K3, K5 | `eval/silver/map_theta_sweep.py` (sweeps θ on `EmbeddingScorer`) | 🟡 | Doesn't sweep alternate scorers or `reject_theta`; uses replay-only `EmbeddingScorer` | Extend sweep to scorers, joint (theta, reject_theta) |
| Evaluation of accepted cheap vs escalated | K5 | `eval/silver/` has gold/silver judges, but no per-paper "accepted-cheap vs L3-baseline" comparison auto-emitted | 🟡 | Have parts, no report | One-shot silver-comparison report |

---

## 5. Specific Gaps in Our Current Design

### Gap 1: No centrality-based output selection on default path

- **Current behavior:** `AgreementChecker.best()` (`agreement/checker.py:85-106`)
  returns the voter with highest `(mean_evidence_chain_length, n_findings)`
  when `bundle.best_index` is None. `EmbeddingScorer.compute()` never sets
  `best_index`. So the production selection key is "longest evidence" — not
  the voter most central to the consensus.
- **Why this is a problem:** Soiffer (S4) shows centrality picks the
  semantically representative output. We instead pick whichever voter
  produced the most verbose evidence chain, which has known failure modes
  (one model copies the whole paragraph as evidence → wins regardless of
  whether its claim agrees with peers).
- **Paper-recommended behavior:** Best = `argmax_i mean_{j≠i} sim(i, j)`, with
  a stable tie-break.
- **Proposed implementation:** Adopt `SemanticAgreementScorer` (already in
  the repo, already does this exactly) as the default in
  `SummarizationRunner.__init__` and `BatchSummarizationRunner.__init__`.
  Keep `EmbeddingSimilarityStrategy` as its strategy so we don't lose the
  contradiction / count / reuse penalties.
- **Files likely affected:** `pipeline/stages/summarization/runner.py`,
  `pipeline/stages/summarization/batch/runner.py`,
  `pipeline/stages/summarization/agreement/__init__.py` (re-exports).
- **Priority:** P0.

### Gap 2: Grounding does not gate the cascade

- **Current behavior:** Grounding runs *after* the cascade has finalised a
  chunk's output. The cascade can accept an L1 result whose claims are not
  entailed by the cited verbatim text, and only `GroundingFilter` trims the
  bad findings later.
- **Why this is a problem:** Both papers' "safe deferral" assumption (K5)
  collapses when agreement is over unsupported content. Two voters can
  hallucinate the same plausible-sounding claim and pass `theta` easily.
- **Paper-recommended behavior:** Grounding (or any other quality gate) must
  influence the cascade decision, not be a post-hoc cleanup.
- **Proposed implementation:** Wire `MapOutputRouter` into both runners. It
  already classifies voters as ELIGIBLE / WEAKLY_GROUNDED / UNUSABLE using
  `SchemaValidator` and `ProvenanceValidator`, drops unusable voters before
  scoring, and threads `AgreementContext` so `SemanticAgreementScorer` can
  use grounding quality for tie-break. The router exists. It just isn't
  passed `router=` from any production builder.
- **Files likely affected:** `scripts/run_paper.py::build_runner` and
  `build_batch_runner`, `pipeline/stages/summarization/runner.py::__init__`,
  `pipeline/stages/summarization/batch/runner.py::__init__` and `_process_level`.
- **Priority:** P0.

### Gap 3: Sync and batch run different cascade code

- **Current behavior:** Sync path = `MapStage._cascade` (router-aware,
  per-voter retry, in-run voter dedup, full trace collector). Batch path =
  `BatchSummarizationRunner._process_level` (none of the above, builds its
  own `AgreementChecker` and ignores the runner-level scorer).
- **Why this is a problem:** Any thesis-relevant cascade change has to be
  implemented twice, with no static guarantee they match. Past hand-edits
  have caused drift (KNOWN_ISSUES.md lists a few). It also breaks fair
  comparison: a paper run in sync vs batch can take different cascade
  paths.
- **Paper-recommended behavior:** Not explicit, but K6's measurability
  requires a single decision surface so cost/accuracy can be attributed to
  the cascade and not to which runner ran.
- **Proposed implementation:** Factor a `cascade_decision(voters, ctx,
  scorer, agreement_checker, router) -> Decision` pure function into a
  shared module (e.g. `agreement/decision.py`) and call it from both
  runners. Until that lands, at minimum make the batch runner consume the
  same `scorer` instance that the sync builder constructs.
- **Files likely affected:** `pipeline/stages/summarization/batch/runner.py`,
  `pipeline/stages/summarization/current_stages/map_stage.py`, new
  `agreement/decision.py`.
- **Priority:** P0.

### Gap 4: No structured biomedical agreement inside MAP

- **Current behavior:** Default agreement is mean of pairwise soft-aligned
  cosine on `Finding.claim` strings. The polarity heuristic is a
  contradiction penalty (max 20% reduction), not a polarity-agreement
  signal. There is no first-class signal for relation type, scope, or
  evidence-span overlap.
- **Why this is a problem:** Two voters saying *"HER2 amplification is
  associated with poor prognosis"* and *"HER2 amplification is associated
  with favorable response to trastuzumab"* will look similar to a claim
  embedder, because the embedding emphasises lexical overlap and topic.
  They disagree on **scope** (prognostic vs. predictive) and **polarity**
  relative to outcome. A purely embedding-based score misses both.
- **Paper-recommended behavior:** Compare structured fields directly when
  available (B2). Soiffer specifically calls out lexical-only metrics as
  too brittle (S7) for non-trivial outputs.
- **Proposed implementation:** See §6 and §7 for the target design.
- **Files likely affected:** `pipeline/stages/summarization/agreement/hybrid_structured.py`
  (extend), `pipeline/stages/summarization/models.py::Finding` (optional
  scope / polarity fields), MAP prompt in `prompts.py` (ask voters to emit
  structured polarity / scope).
- **Priority:** P1.

### Gap 5: No hard-fail rules; everything is weighted

- **Current behavior:** `EmbeddingScorer` applies a max-20% multiplicative
  contradiction penalty (`contradiction_weight=0.20`). Even a 100%
  contradiction ratio leaves 80% of the raw similarity intact, which can
  still exceed `theta=0.8` for high-base-agreement cases.
- **Why this is a problem:** "Same entity, opposite polarity" should never
  KEEP a chunk regardless of how textually similar the rest of the claims
  are. A soft penalty cannot enforce that.
- **Paper-recommended behavior:** B4 — hard-fail rules override the
  weighted score and force escalation.
- **Proposed implementation:** In the agreement gate (or in a wrapping
  scorer), check a small set of hard predicates. If any fires, force
  `ChunkDecision.ESCALATE` regardless of `confidence`. Predicates to start
  with: (a) any voter has `polarity contradiction ratio > 0.5` against
  another eligible voter, (b) any voter is `UNUSABLE`, (c) any aligned
  claim pair has `evidence` pointing to non-overlapping `te_id` sets.
- **Files likely affected:** new `agreement/hard_fail.py`,
  `routing/router.py` (or wrap `EmbeddingScorer`), `agreement/checker.py`.
- **Priority:** P1.

### Gap 6: Theta sweep doesn't cover the structured / semantic path

- **Current behavior:** `eval/silver/map_theta_sweep.py` replays a fixed
  `EmbeddingScorer` against cached voter outputs over
  `THETA_GRID = [0.30..0.90]`. It does not sweep `reject_theta`. It does
  not sweep alternate scorers. It does not measure grounding-failure rate
  per theta.
- **Why this is a problem:** We can't pick a "safe deferral" theta (K5)
  if the sweep doesn't measure precision under it.
- **Paper-recommended behavior:** K3 / K5 / S5 — sweep the threshold, and
  pick the value satisfying a precision constraint.
- **Proposed implementation:** Extend the sweep to: (i) sweep scorers
  (`EmbeddingScorer`, `HybridStructuredSimilarity`,
  `SemanticAgreementScorer(EmbeddingSimilarityStrategy)`), (ii) sweep
  `(theta, reject_theta)` jointly, (iii) include grounding-pass-rate and a
  silver judge precision column per row.
- **Files likely affected:** `eval/silver/map_theta_sweep.py`,
  `eval/silver/pipeline_sweep.py`, `eval/reports/`.
- **Priority:** P1.

### Gap 7: Cost-aware reporting is paper-level, not decision-level

- **Current behavior:** `scripts/run_paper.py::_save_escalation_report`
  reports per-paper actual cost and a counterfactual all-L3 baseline. There
  is no breakdown per cascade decision, no acceptance-rate-by-theta, no
  cost-per-accepted-finding column.
- **Why this is a problem:** Hard to argue "cascade saved X% at no quality
  cost" in the thesis without per-decision attribution.
- **Paper-recommended behavior:** K6 — cost/accuracy tradeoff must be
  measurable. The current report measures cost only.
- **Proposed implementation:** Emit one JSONL row per chunk decision with
  `(pmcid, chunk_id, level_accepted, agreement_score, n_findings_kept,
  voter_cost_l1, voter_cost_l2, voter_cost_l3, accepted_grounding_pass)`.
  Aggregate offline.
- **Files likely affected:** `pipeline/stages/summarization/observability/`,
  `pipeline/stages/summarization/costing/`, `scripts/run_paper.py`.
- **Priority:** P1.

### Gap 8: `MapOutputRouter` is dead code on the default path

- **Current behavior:** `MapStage` accepts `router=...`, but
  `scripts/run_paper.py::build_runner` does not construct or pass a router,
  and `BatchSummarizationRunner` does not accept one. So `SchemaValidator`,
  `ProvenanceValidator`, `VoterClassification`, `RoutingDataset` — all
  implemented — are dead on the default path.
- **Why this is a problem:** Schema-broken or fabricated voters currently
  enter the agreement set. They inflate disagreement (good — they
  escalate) but can also inflate agreement (two equally-fabricated voters
  agree on a hallucinated claim).
- **Paper-recommended behavior:** B5 — unparseable / schema-invalid voters
  must drop out before scoring.
- **Proposed implementation:** Same as Gap 2.
- **Files likely affected:** Same as Gap 2.
- **Priority:** P0.

---

## 6. Recommended Target Design for Our Pipeline

```text
For each MAP chunk:
    Run L1 voters in parallel (sync) or in batched submission (batch).
    For each voter:
        Schema-validate AuditableSummary           [SchemaValidator]
        Provenance-validate evidence citations     [ProvenanceValidator]
        Tier voter as ELIGIBLE / WEAKLY_GROUNDED / UNUSABLE

    If fewer than 2 ELIGIBLE voters:
        ESCALATE to next level (skip the agreement gate)
    Else:
        Lightweight comparison-normalize for agreement only:
            - lowercase + strip punctuation on entity surface form
            - dictionary-snap polarity to {pos, neg, neutral}
            - dictionary-snap relation/category to canonical set
            - keep evidence as (PMCID, te_id) set
            (DO NOT run full NORMALIZE here — too expensive, too downstream-dependent.)

        Align findings across voters by entity + relation match.

        Compute weighted agreement score (see §7) over aligned groups.

        Apply hard-fail rules (see §7).

        If score >= theta AND no hard-fail:
            accept the most-central voter's AuditableSummary
        Else:
            ESCALATE to next level

After cascade finalises a chunk:
    Run GroundingFilter on the kept summary (already happens — fine to keep).
```

### Where each existing stage fits

| Stage | What it should keep doing | What it should NOT do |
|---|---|---|
| MAP | Lightweight comparison-normalisation, agreement scoring, cascade decision, per-voter schema + provenance gating, picking the most-central output. | Don't run full UMLS linking; don't compute group-level canonicalization; don't depend on NORMALIZE output. |
| Grounding (post-MAP) | NLI filter on `(verbatim_support, claim)` of the kept voter's findings. Drop unsupported findings. | Don't influence cascade decision (the router's `ProvenanceValidator` is doing that gating already with a cheap textual check). NLI cross-encoder is too slow per voter pair to put inside MAP. |
| NORMALIZE | Full UMLS-linked entity normalisation, dedup, surface-form canonicalisation. | Don't reach back into MAP. |
| GROUP | Bucket NormalFindings by `(subject, outcome, relation, category)`. Unchanged. | Doesn't need to know about cascade level. |
| CANONICALIZE | Predicate normalisation per group. Unchanged. | — |
| RELATE | NLI-based pairwise relation detection between canonical rules. Unchanged. | — |
| RESOLVE | Weighted scoring of canonical rules. Unchanged. | — |

### Which decisions belong in MAP vs later

| Decision | Stage |
|---|---|
| Did voters agree on this chunk? | MAP |
| Is voter X usable at all? (schema/provenance) | MAP (via router) |
| Are two voters contradicting each other in-chunk? | MAP (hard-fail rules) |
| Is each finding entailed by its cited verbatim? | post-MAP grounding |
| Do two findings across chunks normalise to the same entity? | NORMALIZE |
| Do two canonical rules across papers contradict each other? | RELATE / CONTRADICT detector |

The split keeps MAP cheap and avoids the trap of "MAP needs NORMALIZE which
needs MAP". Comparison-normalisation inside MAP is intentionally lossy
(string-level + dictionary snap) and **must not be persisted** — the real
NORMALIZE re-derives canonical forms from the accepted summary.

### Why a lightweight comparison-normalisation step is enough

Full NORMALIZE in `current_stages/normalize_stage.py` is expensive (UMLS
linking, scispaCy parse, synonyms.yaml lookups, dedup pass). MAP runs on
every chunk on every paper across every cascade level, so an MAP-internal
NORMALIZE would dominate runtime. A comparison-only step that only fixes
the obvious mismatches (case, polarity vocabulary, category names) recovers
most of the agreement signal at near-zero cost. The full pipeline still
runs after MAP, so the persisted output is unchanged.

---

## 7. Proposed Agreement Score

For each chunk, after lightweight comparison-normalisation and finding alignment:

```text
pair_score(A, B) =
      0.25 * entity_match_score(A, B)
    + 0.20 * relation_type_score(A, B)
    + 0.20 * polarity_score(A, B)
    + 0.15 * scope_score(A, B)
    + 0.10 * evidence_overlap_score(A, B)
    + 0.10 * semantic_or_nli_score(A, B)
```

The chunk-level deferral score is `max_i mean_{j != i} pair_score(i, j)`
(Soiffer S3) and `best_index = argmax_i mean_{j != i} pair_score(i, j)`
(Soiffer S4).

Component definitions:

- **`entity_match_score`** — Jaccard over the set of comparison-normalised
  entity surface forms across each voter's findings. Effectively replaces
  what `agreement/ner_scorer.py::_extract_entities` already produces, with
  a lighter normaliser.
- **`relation_type_score`** — Jaccard over `Finding.category` after
  dictionary-snap (e.g. `"expression"`, `"association"`, `"morphology"`,
  `"prognosis"`, `"diagnosis"`, `"treatment"`, `"staging"`). Already
  partly implemented as `_cat_jaccard` in `agreement/hybrid_structured.py`.
- **`polarity_score`** — Per aligned finding pair, 1.0 if both polarities
  are equal (after dictionary-snap to {`pos`, `neg`, `neutral`}), 0.0 if
  opposite, 0.5 if either is `neutral`. Mean across aligned pairs. The
  existing `_polarity` function in `agreement/embedding.py` is the right
  building block; just promote it from "penalty signal" to "first-class
  component".
- **`scope_score`** — Compare a small structured scope tuple
  `(disease_subtype, tissue, stage, treatment_setting)`. For v1, extract
  via regex / dictionary on the claim text and call it a `scope_set`,
  then Jaccard. The MAP prompt can be extended to emit a `scope` field
  later for a real version.
- **`evidence_overlap_score`** — Jaccard of `(PMCID, te_id)` sets across
  the two summaries' findings' `evidence` lists. Already implemented as
  `_evidence_jaccard` in `agreement/hybrid_structured.py`.
- **`semantic_or_nli_score`** — Mean of `EmbeddingScorer`'s soft-aligned
  cosine over `Finding.claim`. Optionally swap for cross-encoder NLI when
  outputs are short (Soiffer S7) — but NLI between every voter pair is
  expensive, so default to embedding.

### Hard-fail rules (override the weighted score)

```text
Escalate immediately if ANY of:
    same entity/relation but opposite polarity         (per aligned pair)
    one voter has a clinically critical finding and another voter omits it
    evidence spans point to incompatible source claims (disjoint te_id sets)
    source grounding fails                              (ProvenanceValidator: NONEXISTENT_SOURCE / FABRICATED_VERBATIM_SUPPORT)
    voter output cannot be parsed                       (schema validator)
    voter output violates the expected schema           (SchemaValidator)
```

These map cleanly onto reason codes that
`pipeline/stages/summarization/routing/models.py::ReasonCode` already defines.
Most are already detectable; the missing one is the "clinically critical
finding omitted" predicate — for v1, define "critical" as `category in
{Diagnostic, Prognostic, Management}` and require coverage. Anything fancier
needs a manual critical-finding list and can wait.

### Output selection

When no hard-fail and `score >= theta`:

```text
best_voter_index = argmax_i mean_{j != i} pair_score(i, j)

# Tie-break (already in SemanticAgreementScorer._select_best):
#   1. grounding_pass_fraction
#   2. mean_evidence_chain_length
#   3. len(findings)
```

If centrality is too expensive to compute over aligned finding groups in
the first pass, the temporary compromise is to use centrality over the raw
`Finding.claim` embedding matrix (i.e. exactly what
`SemanticAgreementScorer` already does) and label it explicitly as
"claim-centrality v0 — pending structured-centrality v1".

---

## 8. Evaluation Plan

### Quality metrics

- Precision of accepted L1 chunks vs. silver judge (`eval/silver/`).
- Grounding-failure rate for accepted chunks (% findings dropped by
  `GroundingFilter` after acceptance — should be low if grounding-first
  routing works).
- Contradiction rate among accepted chunks (`ContradictionDetector`,
  `contradiction_similarity_threshold` in config).
- Schema/parse failure rate per voter / cascade level.
- L1-accepted findings vs. L3 findings on the same chunk (Cohen's κ over
  claims after alignment).
- Stability of NORMALIZE / GROUP / CANONICALIZE output when run on L1 vs.
  L3 inputs (do canonical rules change?).
- Per-paper summary table; cross-paper aggregate.

### Efficiency metrics

- Escalation rate (% chunks reaching L2, L3).
- Acceptance rate at each level.
- Cost per paper, per text element, per accepted finding.
- Latency per paper (sync) and turnaround per paper (batch).
- Number of API calls avoided vs. all-L3 baseline.
- Per-level call distribution.

### Agreement diagnostics

- Distribution of agreement scores (kept vs. escalated).
- Joint sweep over `(theta, reject_theta)` with one column per metric.
- Accepted vs. escalated histogram per category / per entity type / per
  text-element length / per paper.
- For the structured score: distribution per component (entity, relation,
  polarity, scope, evidence, semantic) so we can see which signal moves
  the score in disputed cases.

### What we can do without expert labels

- Silver judge: route a sample of chunks through Opus / Sonnet at L3 and
  treat its output as silver truth. Compare L1-accepted output against
  silver via existing scorers in `eval/silver/`.
- NLI-as-precision: use `GroundingFilter` entailment rate on accepted
  chunks as a proxy for precision (a chunk where 90%+ of accepted findings
  pass NLI is high-precision regardless of judge).
- Compare accepted L1 chunks against L2 / L3 outputs on the same chunk
  (we have all three since the theta sweep already runs all levels
  upfront — reuse those caches).
- Manual inspection of disagreement cases via existing HTML artifacts.
- Downstream-canonical-rule stability test: rerun pipeline with cascade ON
  vs. forced-L3-only on the same papers, diff the `final_rules` set.
- Sample N accepted and N escalated cases per paper for manual review.

### Threshold sweep

```text
For (theta, reject_theta) in candidate grid:
    Replay cached voter outputs through the new scorer.
    For each chunk:
        decision = scorer + theta + reject_theta + hard_fail
    Aggregate:
        acceptance_rate
        escalation_rate
        est_cost (using batch prices)
        grounding_pass_rate on accepted
        contradiction_rate on accepted
        silver_judge_match_rate on accepted
    Pick (theta, reject_theta) satisfying:
        silver_judge_match_rate >= target_precision
        AND grounding_pass_rate >= target_grounding
        AND estimated_cost minimised.
```

For the thesis we should prioritise **safe precision over maximum cost
savings**. Pick the highest theta where precision targets still hold; do
not pick the cost-minimising theta unless it also clears the precision
floor.

---

## 9. Implementation Roadmap

### P0 — Must-have before expensive experiments

| Task | Files | Benefit | Risk if skipped |
|---|---|---|---|
| Wire `MapOutputRouter` into sync and batch runners | `scripts/run_paper.py::build_runner` / `build_batch_runner`, `pipeline/stages/summarization/runner.py::__init__`, `pipeline/stages/summarization/batch/runner.py::__init__` + `_process_level` | Schema + provenance gating becomes live; routing dataset starts collecting; grounding signals are no longer post-hoc | Unsupported / fabricated voters keep inflating agreement and we never see it |
| Adopt `SemanticAgreementScorer(EmbeddingSimilarityStrategy)` as default | `runner.py` (both), `agreement/__init__.py` re-exports, `scripts/run_paper.py` | Centrality-based selection, `score_details.pairwise_upper` written to traces | "Best" voter remains "longest evidence" |
| Factor shared cascade decision into one function called by both runners | new `pipeline/stages/summarization/agreement/decision.py`, `current_stages/map_stage.py`, `batch/runner.py` | Sync/batch parity is enforced by construction | Drift across runners keeps creeping in |
| Add per-chunk decision log (one JSONL row per chunk) with score, level accepted, hard-fail reasons | `pipeline/stages/summarization/observability/` | Cascade behaviour becomes inspectable per chunk | Cannot debug why a chunk did or didn't escalate |
| Per-level cost rollup in the cost report | `scripts/run_paper.py::_save_escalation_report`, `costing/` | Honest cost/savings story for thesis | Reported savings remain rough |
| Cache integrity check: refuse to load MAP cache entries whose `cascade_signature` doesn't match the active configuration (already done) — add a startup log line that prints `cascade_signature` and the cache hit rate for the run | `cache.py`, `runner.py` (both) | Stale cache hits surface immediately | Easy to silently rerun against old voter outputs |

### P1 — Important improvements aligned with the papers

| Task | Files | Benefit | Risk if skipped |
|---|---|---|---|
| Implement structured agreement (entity + relation + polarity + scope + evidence + semantic) per §7 | `agreement/hybrid_structured.py`, new `agreement/structured_scorer.py`, optional new fields on `Finding` in `models.py`, prompt update in `prompts.py` | Disagreement matches our actual error modes | Cascade decisions stay surface-form-driven |
| Add lightweight comparison-normalisation inside MAP (case, polarity vocab, category snap) | new `agreement/normalize_for_agreement.py`, used by the new scorer | Higher real agreement on semantically-identical voters | Inflated disagreement on surface-form variation |
| Hard-fail rules (polarity contradiction, schema/provenance hard codes, evidence-disjoint) | new `agreement/hard_fail.py`, wired into `AgreementChecker` or the router | Catches the failure mode soft penalties miss | Same-entity opposite-polarity cases can still KEEP |
| Extend `map_theta_sweep.py` to (i) joint `(theta, reject_theta)`, (ii) multiple scorers, (iii) silver-judge precision column | `eval/silver/map_theta_sweep.py`, `eval/silver/pipeline_sweep.py` | Picks `theta` by quality, not by guesswork | We pick `theta` blind |
| Accepted-vs-L3 silver evaluation harness | `eval/silver/`, `eval/llm_judge/` | Quantitative answer to "is cascade safe?" | We have to argue safety informally |
| Per-pair score breakdown surfaced to traces and reports | `agreement/embedding.py` (already produces it), `observability/models.py::AgreementTrace` (already supports it), reports | We can show *why* a pair agreed or didn't | Audit remains opaque |

### P2 — Nice-to-have research improvements

| Task | Files | Benefit | Risk if skipped |
|---|---|---|---|
| Learned / optimised agreement weights | `agreement/calibration/`, possibly extend `threshold_optimizer.py` | Better than hand-set weights | Weights remain hand-tuned |
| Pairwise NLI between voters (replace embedding for hard cases) | `agreement/`, `helpers/grounding_filter.py` (reuse pipeline) | Sharper signal on near-paraphrases | None — embedding works for v1 |
| Per-relation-type thresholds | `agreement/`, `routing/policy.py` | Tighter calibration per finding category | Single theta serves all relation types |
| Active-learning / human-review interface for borderline cases | new tool under `eval/` | Cheap path to gold labels | Manual review stays ad-hoc |
| Richer semantic similarity metrics (BERTScore, NLI matrices, LLM judge in `agreement/llm_judge.py`) | `agreement/` | Better signal at higher cost | Embedding is "good enough" |
| Per-paper / per-domain calibration | `agreement/calibration/`, `eval/` | Domain-adaptive theta | Universal theta works as baseline |
| API-provider-specific latency / cost optimisation (batch endpoint preferences, region routing) | `batch/` providers | Cost cuts | Marginal |

---

## 10. Final Recommendation

**Implement now (P0):**

1. Adopt `SemanticAgreementScorer(EmbeddingSimilarityStrategy)` as the
   default in both runners. We already have it. Switch it on.
2. Wire `MapOutputRouter` into both runners so schema + provenance gating
   actually runs. Drop unusable voters before scoring.
3. Factor the cascade decision into one function shared by sync and
   batch. Stop maintaining two copies.
4. Emit a per-chunk decision JSONL so we can answer "why did this chunk
   escalate" without re-running the cascade.

These four changes use code we already have. They take the cascade from
"runs and produces numbers" to "produces numbers we can trust and audit".

**Do not overengineer:**

- Don't build a learned-weight agreement model yet — fixed weights from
  §7 are good enough for the thesis baseline.
- Don't add pairwise NLI between voters until embedding agreement is
  shown to fail concrete cases.
- Don't try to merge `MapStage` and `BatchSummarizationRunner._process_level`
  into one super-runner. A shared `cascade_decision` function is enough.
- Don't add new prompt fields (scope, polarity) until the structured score
  is the bottleneck. Start with what the current MAP output already
  contains (`category`, `evidence`, `claim`).

**Postpone until post-thesis:**

- Learned agreement weights.
- Per-relation-type thresholds.
- LLM-judge in the hot loop (cost / latency too high for routine runs).
- Active learning / annotation UI.

**Order of priorities for the thesis timeline:**

1. **Sync/batch parity + router wiring** (P0). Without this, every
   later result is contaminated.
2. **Centrality-based selection** (P0). Cheapest paper-aligned change we
   have; already coded.
3. **Per-decision logging + cost rollup** (P0). Needed before any
   defensible theta sweep.
4. **Structured agreement + hard-fail rules** (P1). The thesis story
   improves materially with this.
5. **Theta sweep with silver-judge precision column** (P1). Picks the
   operating point we report.
6. **Grounding-as-gate** is folded into step 1 (router wiring) — that's
   what the router does.

The single biggest immediate win is **stop using `_quality_key` to pick
"best" and use centrality instead** (Gap 1). It is one constructor change
in two files and it is the change most directly aligned with Soiffer's
contribution.
