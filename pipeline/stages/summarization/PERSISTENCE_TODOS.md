# Summarization persistence — open TODOs

> **2026-05-10 update**: batch path now writes the full artifact layout via
> `BatchSummarizationRunner.finalize()` and runs the modern post-MAP chain
> (NORMALIZE → GROUP → CANONICALIZE → RELATE → RESOLVE). Original "batch
> persistence missing" item is **CLOSED**. Stage persisters live as free
> functions in `persistence.py` and are shared by both runners. DB
> persistence (`Sum*` tables) for batch mode remains a TODO — file artifacts
> only today.



Tracks gaps in `persistence.py` / runner integration. Each item is "decide
whether to do, then do or close". Source files cited inline.

Status legend: **DECIDE** = needs a yes/no, **DO** = decision is to implement,
**CLOSED** = no longer needed (record reason).

---

## 1. MAP `Finding.finding_id` — DECIDE

**Where**: `pipeline/stages/summarization/models.py` (`Finding`),
`runner.py::_persist_map_artifacts`, `runner.py::_persist_normal_findings`.

**Today**: `Finding` has no stable id. We persist
`(pmcid, chunk_id, position_in_chunk, evidence_refs)` as the MAP coordinate.
`NormalFinding.source_finding_ids` is declared but unpopulated.

**Risk**: lineage chain `FinalRule → CanonicalRule → FindingGroup → NormalFinding → MAP Finding → sentence` has a soft link at the last hop — we can find which sentences were cited but not which **specific MAP finding** generated which NormalFinding when multiple share the same coord.

**Options**:
- (a) Add `finding_id: str` to `Finding` (sha8 of `pmcid|chunk_id|position|claim`) and propagate into `NormalFinding.source_finding_ids` during NORMALIZE. Stable, deterministic, cheap. Touches one model + one merge call site.
- (b) Continue using the coord; document the soft-link risk and accept it.

**Decision needed by**: before any threshold-calibration / evaluation work that depends on per-finding lineage (i.e. before the eval pipeline is wired up).

---

## 2. NORMALIZE `source_finding_ids` population — DEPENDENT on #1

**Where**: `current_stages/normalize_stage.py::_merge`.

**Today**: `runner._persist_normalize_artifacts` writes
`dedup_trace.jsonl` listing evidence-span coordinates and an empty
`source_finding_ids` list per row.

**Action**: when #1 lands, fill `NormalFinding.source_finding_ids` in
`NormalizeStage._merge` and `_wrap_single`. No persistence change required;
the writer already serialises the field.

---

## 3. RELATE skipped/blocking pair trace — DECIDE

**Where**: `current_stages/relate_stage.py::relate` (gate rejections only counted via `rejection_counts`),
`runner.py::_persist_relate_artifacts` (writes empty `skipped_pairs.jsonl`).

**Today**: gate rejections are logged at INFO with reason counts, then the per-pair detail is dropped. Stored counts only end up in stdout/log files, not artifacts.

**Risk for threshold calibration**: we cannot answer "which pairs were filtered out by the gate vs the NLI threshold?" offline.

**Options**:
- (a) Change `RelateStage.relate()` signature to also return `skipped: list[SkippedPair]`. Persist directly. Most useful, smallest data; touches one stage signature + one Pydantic model.
- (b) Add a `skipped_counts` summary field to the manifest (cheap, partial answer).
- (c) Leave as TODO until calibration actually needs it.

**Recommendation**: (a) once we start sweeping `entailment_threshold` / `contradiction_threshold` offline.

---

## 4. CANONICALIZE LLM/predicate-selection trace — DECIDE

**Where**: `current_stages/canonicalize_stage.py::_select_predicate`,
`runner.py::_persist_canonicalize_artifacts` (writes `canonical_rules.jsonl` only).

**Today**: predicate-selection is deterministic (best-grounded fallback). No LLM trace today, so there's nothing to persist beyond the rule itself.

**Action**:
- Close as **CLOSED — N/A** while canonicalize stays deterministic.
- Reopen if/when an LLM is plugged in for predicate canonicalisation. At that point persist `{group_id, candidates[], chosen, fallback_reason, model, prompt_hash}` to `canonicalize_trace.jsonl`.

---

## 5. Corpus-level RELATE artifact — DECIDE

**Where**: `helpers/corpus_relate.py::CorpusRelateStage.relate_incremental`,
`runner.py::_corpus_relate_incremental` (DB-only side effect, no return mirror).

**Today**: incremental cross-paper relations land in Postgres only. The runs/.../relate/corpus/ directory is never created.

**Options**:
- (a) Have `relate_incremental` return the relations it produced; runner mirrors them to `runs/{run_id}/relate/corpus/relations.jsonl` (append-mode, since corpus state is cumulative across papers).
- (b) Provide a separate `runner.export_corpus_relations()` post-run helper that reads from DB.
- (c) Skip — DB is source of truth.

**Recommendation**: (a). Append is fine because corpus relate already de-dupes via DB. Keeps file-only debugging viable.

---

## 6. CSV summaries — DECIDE

**Where**: `persistence.py::write_csv` (helper exists, unused).

**Today**: spec lists CSVs as nice-to-have. v1 omits them entirely so the writer surface stays small.

**Options**:
- (a) Wire `_persist_*_artifacts` to also emit `*_summary.csv` per stage. ~30 lines per stage.
- (b) Provide an offline `runs_to_csv.py` script that reads JSONL and writes CSVs on demand. Decouples format churn.

**Recommendation**: (b). CSV schemas drift; JSONL is the durable record.

---

## 7. NLP_HISTO_LOG_DIR contention — DECIDE (low priority)

**Where**: `runner.py::process` (sets env, restores in `finally`).

**Risk**: two `process()` calls running concurrently in the same Python process (if anyone ever multithreads at this level) would race on the env var. Not a concern today (`process_batch` is serial), but worth noting.

**Action**: only fix if/when concurrent `process()` becomes a use case.
Likely fix: pass `log_dir` directly to the enum logger via a contextvar
instead of an env var.

---

## 8. Manifest `pipeline_config_hash` — DECIDE

**Spec asked for**: `pipeline_config_hash if available`.

**Today**: not computed. The full config dict goes into `manifest.config` so the data is recoverable, but there's no stable short hash for cross-run comparisons.

**Action** (small): compute `sha256(json.dumps(_to_jsonable(config), sort_keys=True))[:16]` in `_make_artifact_writer` and write to `manifest.extra["pipeline_config_hash"]`. ~5 lines.

---

## 9. RESOLVE per-component score breakdown — DEPENDENT on stage exposure

**Where**: `current_stages/resolve_stage.py::resolve` returns only the final
score and counts. `runner.py::_persist_resolve_artifacts::score_trace.jsonl`
just duplicates `final_rules` fields.

**Action**: when `ResolveStage` is updated to return a per-component breakdown
(grounding contribution, support boost, single-study penalty, contradiction
penalty), surface those fields in `score_trace.jsonl`. No writer change needed.

---

## Quick triage table

| # | Item | Effort | Impact | Decision needed |
|---|------|--------|--------|-----------------|
| 1 | MAP `finding_id`             | S | High (lineage) | Before eval pipeline |
| 2 | NF `source_finding_ids` fill | S | High (lineage) | After #1 |
| 3 | RELATE skipped trace         | S | High (calibration) | Before threshold sweep |
| 4 | CANONICALIZE LLM trace       | — | N/A today | Reopen if LLM added |
| 5 | Corpus relate file mirror    | S | Med | Soon |
| 6 | CSV summaries                | M | Low | Defer to offline tool |
| 7 | Log-dir env race             | S | Low | Only if concurrent |
| 8 | `pipeline_config_hash`       | XS | Low | Cheap; just do it |
| 9 | RESOLVE component breakdown  | — | Med | After stage exposes |
