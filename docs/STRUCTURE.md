# STRUCTURE.md

Authoritative map of the pipeline architecture. **Update this file in the same
commit as any change to pipeline stages, runners, configs, DB schema, or
artifact layout.** If a file move, rename, or new stage isn't reflected here
within the commit that introduced it, the change is incomplete.

The detailed per-file project walkthrough lives in
[`.claude/CLAUDE.md`](../.claude/CLAUDE.md); this file is the higher-level
architectural index — what flows where, and which files are load-bearing.

---

## Top-level layout

```
nlp-histo/
├── file-selector/          Stage 1 — download PMC tarballs, organise PDFs/XMLs
├── parsers/                Shared parsing utilities (layout, text, XML)
├── pipeline/               Stage 2 — two sub-pipelines
│   ├── utils/              Cross-pipeline utilities (memory logging, …)
│   └── stages/
│       ├── pdf_text_extraction/   PDF → hierarchical text + media → DB
│       └── summarization/         Text → structured rules via LLM cascade
├── database/               SQLAlchemy ORM, connection pool, schema setup
├── named_entity_recognition/  scispaCy + UMLS entity extraction
├── eval/                   Evaluation harness, annotation tools
├── scripts/                One-off / inspection / thesis-demo scripts
├── tests/                  pytest suite (summarisation-heavy)
├── configs/                YAML run configs, NLI/pricing tables
├── notebooks/              Demo / workflow notebooks
├── out/                    Runtime outputs (cached layouts, summaries, demos)
└── files/                  Input PDFs/XMLs (not in repo)
```

---

## Pipeline A — PDF text extraction

**Orchestrator:** [`pipeline/stages/pdf_text_extraction/runner.py`](../pipeline/stages/pdf_text_extraction/runner.py) — `PipelineRunner.run_document(pdf_path, pmcid)`

**Active 8-step flow:**

```
PDF
  │
  ▼
 (1) Extract full layout (Docling) ─────────── out/docling_full/*.json (cached)
  │
  ▼
 (2) Detect tables (Docling | TATR | Hybrid)  ─ TableDetectionResult
  │
  ▼
 (3) Mask detected regions (PyMuPDF)           ─ out/masked_pdfs/*.pdf
  │
  ▼
 (4) Re-extract layout from masked PDF         ─ out/docling_masked/*.json
  │
  ▼
 (5) Filter artifacts (layout_utils)
  │
  ▼
 (6) Assemble hierarchical text                ─ List[HierarchicalRow]
  │
  ▼
 (7) Crop figure/table images                  ─ out/figures/, out/tables/
  │
  ▼
 (8) Write outputs (text files + Postgres)
```

When `TwoPassConfig.enabled=True` (default since May-2026), steps 1/3/4 are
replaced by a single `TwoPassTextExtractor` call that gathers per-element
evidence and applies `NodeScorer` rules R1/R-color/R2/R3 *before* text reaches
later stages.

### Key files

| Role                              | File                                                                |
|-----------------------------------|---------------------------------------------------------------------|
| Orchestrator                      | `pipeline/stages/pdf_text_extraction/runner.py`                     |
| Config (all sub-configs)          | `pipeline/stages/pdf_text_extraction/config.py`                     |
| Batch driver (ThreadPoolExecutor) | `pipeline/stages/pdf_text_extraction/batch.py`                      |
| Blacklist (skip list)             | `pipeline/stages/pdf_text_extraction/blacklist.py`                  |
| Lazy model registry               | `pipeline/stages/pdf_text_extraction/resources.py`                  |
| Two-pass extractor                | `pipeline/stages/pdf_text_extraction/components/two_pass_extractor.py` |
| Evidence gatherer (PyMuPDF)       | `pipeline/stages/pdf_text_extraction/components/evidence_gatherer.py` |
| NodeScorer (R0–R3 rules)          | `pipeline/stages/pdf_text_extraction/components/node_scorer.py`     |
| Layout extractor (Docling + cache)| `pipeline/stages/pdf_text_extraction/components/layout_extractor.py` |
| Region masker                     | `pipeline/stages/pdf_text_extraction/components/region_masker.py`   |
| Text assembler                    | `pipeline/stages/pdf_text_extraction/components/text_assembler.py`  |
| Artifact filter                   | `pipeline/stages/pdf_text_extraction/components/artifact_filter.py` |
| Media cropper                     | `pipeline/stages/pdf_text_extraction/components/media_cropper.py`   |
| Visualiser                        | `pipeline/stages/pdf_text_extraction/components/visualizer.py`      |
| Table detectors                   | `pipeline/stages/pdf_text_extraction/table_detectors/{docling,tatr,hybrid}_detector.py` |
| Output writers                    | `pipeline/stages/pdf_text_extraction/outputs/{writer,db_ingester,media_json_writer}.py` |
| Shared layout helpers             | `parsers/layout_utils.py`                                           |
| Text post-processing              | `parsers/text_processing.py`                                        |
| XML parser                        | `parsers/xml_parsers/hierarchical_parser.py`                        |

### Config sub-configs (`pipeline/stages/pdf_text_extraction/config.py`)

`PathConfig`, `DoclingConfig`, `TATRConfig`, `MaskingConfig`,
`FilteringConfig`, `CroppingConfig`, `TextAssemblyConfig`,
`VisualizationConfig`, `DatabaseConfig`, `RuntimeConfig`, `TwoPassConfig`.

Enums: `TableDetectorType ∈ {TATR, DOCLING, HYBRID, VLM}`,
`BaselineMode ∈ {MASKED, UNMASKED, BOTH}`.

---

## Pipeline B — Summarisation

**Orchestrator:** [`pipeline/stages/summarization/runner.py`](../pipeline/stages/summarization/runner.py) — `SummarizationRunner.process(file_data) -> dict`

**Stage flow (one paper):**

```
Sentences (per paper)
  │
  ▼ MAP             current_stages/map_stage.py             3-tier ABC cascade
  │                                                          → AuditableSummary[] per chunk
  ▼ GROUNDING       helpers/grounding_filter.py             NLI entailment filter
  ▼ NORMALIZE       current_stages/normalize_stage.py       UMLS-normalised entities + dedup
  ▼ GROUP           current_stages/group_stage.py           NormalFinding[] → FindingGroup[]
  │                                                          (group_id includes pmcid since May-2026)
  ▼ CANONICALIZE    current_stages/canonicalize_stage.py    Pick predicate, split by direction → CanonicalRule[]
  ▼ RELATE          current_stages/relate_stage.py          NLI pairwise → Relation[]
  ▼ RESOLVE         current_stages/resolve_stage.py         Score → FinalRule[]

[Optional, off by default]
  REDUCE            old_stages/reduce_stage.py              Tree-reduce chunk summaries → ConsolidatedSummary
  RULES             old_stages/rule_stage.py                IF-THEN extraction → ExtractedRules
```

Cross-paper relations are produced separately by
`helpers/corpus_relate.py:CorpusRelateStage` over the pooled per-paper
`CanonicalRule[]`.

### Three-tier voter cascade (MAP)

| Tier | Models                                                        | Approx. \$/call |
|------|----------------------------------------------------------------|----------------|
| L1   | DeepSeek, Gemini Flash-Lite, Mistral Large                     | ~\$0.001       |
| L2   | Gemini Flash, Kimi K2.5, Claude Haiku — fires on L1 disagreement | mid-tier       |
| L3   | Claude Sonnet 4.6 — final escalation when L1+L2 disagree       | premium        |

### Key files

| Role                                | File                                                               |
|-------------------------------------|--------------------------------------------------------------------|
| Orchestrator                        | `pipeline/stages/summarization/runner.py`                          |
| Persistence helpers                 | `pipeline/stages/summarization/persistence.py`                     |
| Config                              | `pipeline/stages/summarization/config.py`                          |
| All Pydantic models                 | `pipeline/stages/summarization/models.py`                          |
| LLM prompt builders                 | `pipeline/stages/summarization/prompts.py`                         |
| LLM call cache                      | `pipeline/stages/summarization/cache.py`                           |
| LLM provider wiring                 | `pipeline/stages/summarization/llm_providers.py`                   |
| Error classification (retry / fail) | `pipeline/stages/summarization/llm_errors.py`                      |
| UMLS singleton                      | `pipeline/stages/summarization/umls_resources.py`                  |
| UMLS lookup helpers                 | `pipeline/stages/summarization/umls_utils.py`                      |
| Synonym table                       | `pipeline/stages/summarization/synonyms.yaml`                      |
| Cross-paper relate stage            | `pipeline/stages/summarization/helpers/corpus_relate.py`           |
| Grounding NLI filter                | `pipeline/stages/summarization/helpers/grounding_filter.py`        |
| Async batch dispatch                | `pipeline/stages/summarization/batch/{runner,dispatch,models,voter_configs}.py` |
| Trace collector                     | `pipeline/stages/summarization/observability/`                     |
| Usage / cost accounting             | `pipeline/stages/summarization/costing/`                           |
| Agreement scorers                   | `pipeline/stages/summarization/agreement/`                         |
| MAP router (provenance gates)       | `pipeline/stages/summarization/routing/`                           |
| Artifact writers                    | `pipeline/stages/summarization/output/`                            |

### Critical invariants

* `group_id = GRP_{sha8(pmcid)}_{sha8(subject)}_{sha8(outcome)}_{relation}_{sha8(category)}`
  — includes `pmcid` so identical (subject, outcome, relation, category)
  tuples in two different papers produce distinct group IDs (fix for the
  cross-paper `canonical_id` collision; see [`THESIS.md`](THESIS.md) §1).
* `canonical_id = CR_{sha8(group_id)}_{direction}` — inherits the pmcid
  embedding via `group_id`.
* Cross-paper merging happens via the
  `helpers/corpus_relate.py:_should_compare_cross_paper` gate (CUIs first,
  normalised entity strings as fallback) — never via `canonical_id` equality.

---

## Database

**ORM:** `database/models.py`

| Table                                | Purpose                                              |
|--------------------------------------|------------------------------------------------------|
| `documents`                          | Per-paper metadata (unique `pmcid`)                  |
| `text_elements`                      | One row per paragraph with `unique_path`, `path_list` (TEXT[]), `path_string`, `depth`, `position_in_section` |
| `figures`, `tables`                  | Cropped media with bbox/page metadata                |
| `entities`                           | scispaCy/UMLS named entities                         |
| `text_element_figure_references`     | Many-to-many text ↔ figure                           |
| `text_element_table_references`      | Many-to-many text ↔ table                            |
| `pipeline_runs`                      | Summarisation run manifest                           |
| `sum_map_findings`                   | MAP-stage output (AuditableSummary[] flattened)      |
| `sum_normal_findings`                | NORMALIZE output                                     |
| `sum_normal_finding_spans`           | Provenance spans for NormalFindings                  |
| `sum_finding_groups`                 | GROUP output (`group_id` is the pmcid-namespaced hash)|
| `sum_group_members`                  | NormalFinding ↔ FindingGroup junction                |
| `sum_canonical_rules`                | CANONICALIZE output                                  |
| `sum_relations`                      | RELATE output (intra-paper)                          |
| `sum_corpus_relations`               | CorpusRelateStage output (intra + cross paper)       |
| `sum_final_rules`                    | RESOLVE output                                       |
| `sum_rejected_findings`              | Findings filtered out during NORMALIZE/GROUP         |
| `sum_rejection_summaries`            | Per-paper rejection roll-up                          |
| `llm_judge_cache`                    | LLM judge call cache (semantic agreement scoring)    |

**Connection:** `database/db_connection.py` →
`get_db_connection().session_scope()`.

**Schema setup:** `database/setup_db.py` (`--check`, `--drop`).

---

## Output directories (runtime)

```
out/
├── docling_full/         Full Docling layout JSON (cached)
├── docling_masked/       Docling layout JSON from masked PDFs
├── masked_pdfs/          PDFs with detected regions whited out
├── text/                 Hierarchical text
├── text_raw/             Pre-assembly raw elements
├── figures/, tables/     Cropped media
├── visualization/        Annotated debug PDFs
├── json/                 Per-paper media JSON
├── run_metadata/         Per-paper timing + processing stats
├── summaries/            Summarisation artifacts (runs/<run_id>/...)
├── thesis_demo/          Thesis demo PNG/JSON artifacts
└── failed_pdfs_blacklist.json
```

---

## Scripts that matter

| Script                                          | Purpose                                                   |
|-------------------------------------------------|-----------------------------------------------------------|
| `scripts/run_paper.py`                          | End-to-end driver for one paper                           |
| `scripts/estimate_selection_cost.py`            | Cheap-tier cost estimate before a run                     |
| `scripts/estimate_pipeline_cost_percentiles.py` | Cost-percentile sweep                                     |
| `scripts/inspect_pipeline_output.py`            | HTML inspector for a single paper                         |
| `scripts/inspect_normalize_group.py`            | Walks NORMALIZE → GROUP for one paper                     |
| `scripts/inspect_phase123_pipeline.py`          | Walks MAP → NORMALIZE → GROUP for one paper               |
| `scripts/verify_ghost_text_detection.py`        | Synthetic Tr=3 / opacity=0 PDF → R1 pixel-check passes    |
| `scripts/scan_ghost_text_real_papers.py`        | Corpus-sample ghost-text scan                             |
| `scripts/thesis_demo_ghost_text.py`             | Before/after demos (writes PNGs + JSON for THESIS.md)     |
| `scripts/test_map_schema.py`                    | Standalone MAP-output schema validator                    |

---

## Coordinate system

| System              | Origin             | Used by                                  |
|---------------------|--------------------|------------------------------------------|
| Docling PDF coords  | y=0 at **bottom**; y1=top, y2=bottom | `LayoutElement.bbox`, `BoundingBox` |
| fitz/screen coords  | y=0 at **top**; rect.y0 < rect.y1   | PyMuPDF masking, cropping, rendering |

Convert via `BoundingBox.to_fitz_rect(page_height)` /
`BoundingBox.from_fitz_rect(rect, page_height, page)`.

---

## When to edit this file

* New pipeline stage, file move, or significant rename → update the relevant
  section here in the **same commit** as the code change.
* New table or schema column → update the Database table.
* New top-level output directory → add it to the Output directories list.
* New thesis-relevant script → add a row to "Scripts that matter".
* Config defaults that affect the whole pipeline (e.g. the May-2026
  `TwoPassConfig.enabled` flip) → mention in the affected pipeline section
  *and* link to the THESIS.md entry that motivated the change.
* **Always** append a row to the Pipeline changelog below so the history is
  visible at a glance even without diffing the file.

---

## Pipeline changelog

Append one row per change that affects pipeline architecture, behaviour, or
artifact layout. Date in ISO format. Link to the THESIS.md bug / decision
that motivated the change when applicable.

| Date       | Area                              | Change                                                                                                                       | Motivated by |
|------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------|--------------|
| 2026-05-13 | Summarisation › GROUP             | `_group_id` now hashes `pmcid` alongside `(subject, outcome, relation, category)`. `canonical_id` inherits the namespace.    | THESIS.md [Bug 1](THESIS.md#bug-1--duplicate-intra-paper-relations-produced-by-canonical_id-collisions) |
| 2026-05-13 | PDF extraction › TwoPassConfig    | `enabled` default flipped `False → True`. NodeScorer now runs on every Docling element by default.                           | THESIS.md [Bug 2](THESIS.md#bug-2--docling-phantom-layout-elements) |
| 2026-05-13 | PDF extraction › TwoPassConfig    | `max_white_char_fraction` default flipped `0.5 → 1.0` — Rule R-color disabled by default.                                    | THESIS.md [Bug 3](THESIS.md#bug-3--r-color-white-text-false-positive-latent) |
| 2026-05-13 | Scripts                           | Added `verify_ghost_text_detection.py`, `scan_ghost_text_real_papers.py`, `thesis_demo_ghost_text.py`.                       | THESIS.md [Topic — ghost-text detection](THESIS.md#topic--ghost-text-detection-empirical-verification-and-policy-fix) |
| 2026-05-13 | Docs                              | New `docs/` folder with `THESIS.md`, `HOW_TO_RUN.md`, `STRUCTURE.md`. CLAUDE.md updated to enforce keeping them in sync.      | Thesis-supporting docs request |
| 2026-05-13 | Docs                              | Consolidated pre-existing root MDs into `docs/`: `REPOSITORY_GUIDE.md`, `PIPELINE_BUGS.md`, `TODO.md`. Moved `readmes/` → `docs/readmes/`. `README.md` stays at repo root. Co-located module docs untouched. | Tidy-up |
| 2026-05-14 | Summarisation › `BatchSummarizationRunner` | `finalize()` now mirrors sync: verbatim-from-DB before grounding, stable `finding_id`, DB persistence (`sum_*` tables), `corpus_relate_incremental`, `rejection_summary` build + persist, optional NER. `__init__` accepts `db`/`force_rerun`/`run_ner`. `BatchHandle` carries `pipeline_run_db_id` + `cached_result_only`. | THESIS.md [Bug 5](THESIS.md#bug-5--batch-runner-missing-sync-parity-features) |
| 2026-05-14 | Summarisation › `BatchSummarizationRunner` | Result caching added: `_load_result`/`_save_result` stamp + check `pipeline_config_hash`; valid cache short-circuits at `submit()` so stale results cannot consume L1/L2/L3 batch dollars. Pre-fix `out/summaries/summaries/*.json` deleted to force regeneration. | THESIS.md [Bug 5](THESIS.md#bug-5--batch-runner-missing-sync-parity-features) |
