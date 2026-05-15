# Repository Guide — nlp-histo

A complete file-by-file reference for the nlp-histo codebase. Organized by folder, with three levels of detail:

- **Folder overview** — one paragraph on the folder's purpose and place in the system.
- **File table** — quick-scan one-liners for every file.
- **Detailed entries** — 2–4 sentences per file covering purpose, key classes/functions, inputs/outputs, and notable design decisions.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Root-level files](#root-level-files)
3. [`file-selector/` — data acquisition](#file-selector--data-acquisition)
4. [`database/` — ORM + connection management](#database--orm--connection-management)
5. [`alembic/` — schema migrations](#alembic--schema-migrations)
6. [`parsers/` — PDF / XML / text utilities](#parsers--pdf--xml--text-utilities)
7. [`named_entity_recognition/` — scispaCy + UMLS](#named_entity_recognition--scispacy--umls)
8. [`pipeline/utils/` — shared pipeline utilities](#pipelineutils--shared-pipeline-utilities)
9. [`pipeline/stages/pdf_text_extraction/` — main PDF→DB pipeline](#pipelinestagespdf_text_extraction--main-pdfdb-pipeline)
10. [`pipeline/stages/summarization/` — LLM summarization pipeline](#pipelinestagessummarization--llm-summarization-pipeline)
11. [`eval/` — evaluation harness](#eval--evaluation-harness)
12. [`scripts/` — utilities, runners, inspectors](#scripts--utilities-runners-inspectors)
13. [`notebooks/` — standalone demo / workflow notebooks](#notebooks--standalone-demo--workflow-notebooks)
14. [`tests/` — pytest suite](#tests--pytest-suite)
15. [`langchain-summarization/` — legacy LangChain stack](#langchain-summarization--legacy-langchain-stack)
16. [`misc/` — repo introspection utilities](#misc--repo-introspection-utilities)
17. [`pdffigures2/` — Allen AI evaluation harness (vendored)](#pdffigures2--allen-ai-evaluation-harness-vendored)
18. [`.agents/` — Claude Code skills](#agents--claude-code-skills)
19. [`READMEs/` — specialized documentation](#readmes--specialized-documentation)

---

## System Overview

The repository implements a research pipeline for extracting structured medical knowledge from histopathology papers. Its three logical stages are:

1. **Data acquisition** (`file-selector/`) — pull PDFs/XMLs from PubMed Central and flatten them.
2. **PDF text extraction** (`pipeline/stages/pdf_text_extraction/`) — parse PDFs with Docling, detect tables/figures, mask + re-extract, assemble hierarchical text, and persist into PostgreSQL via the SQLAlchemy ORM in `database/`.
3. **LLM summarization** (`pipeline/stages/summarization/`) — turn paragraph text into auditable, normalized clinical findings (`MAP → GROUNDING → NORMALIZE → GROUP → CANONICALIZE → RELATE → RESOLVE`) using a 3-tier ABC cascade of LLMs.

Around these three sit `eval/` (precision/recall + LLM-judge silver labels), `scripts/` (one-off runners and inspectors), `alembic/` (schema migrations), and `tests/` (pytest unit tests).

---

## Root-level files

Top-level configuration, documentation, and lockfiles for the project.

| File | Role |
|---|---|
| `README.md` | Top-level project documentation (audit-trail architecture, stages, quickstart). |
| `CLAUDE.md` | Project memory file for Claude Code (placeholder, empty). |
| `.env.example` | Template for secrets: DB credentials, Anthropic / OpenAI / Vertex / Azure keys. |
| `pyproject.toml` | Setuptools project metadata + package discovery (`pipeline`, `database`, `parsers`, `named_entity_recognition`). |
| `requirements.txt` | Pinned Python dependencies (Docling, scispaCy, SQLAlchemy, PyMuPDF, marker-pdf, Nougat, LangChain, OpenAI/Anthropic clients, …). |
| `alembic.ini` | Alembic configuration: SQLAlchemy URL, logging, versions path. |
| `skills-lock.json` | Locked versions of Claude Code custom skills used in the project. |
| `REPOSITORY_GUIDE.md` | This file. |

### Detailed entries

- **`README.md`** — Long-form documentation explaining the audit-trail architecture (every clinical rule traces back to its source sentence), the three pipeline stages, the database schema, the LLM-judge evaluation harness, and a quickstart. Refer here first for the project's high-level intent and tech stack.
- **`CLAUDE.md`** — Currently empty. Reserved for project-specific instructions to Claude Code; the active project rules live in `.claude/CLAUDE.md`.
- **`.env.example`** — Documents the variables consumed by `database/db_connection.py` and the LLM provider modules. Copy to `.env` and fill in real credentials.
- **`pyproject.toml`** — Minimal metadata; the heavy dependency list lives in `requirements.txt`.
- **`requirements.txt`** — Full pip dependency list. Notable groups: PDF processing (docling, marker-pdf, pymupdf, pymupdf4llm, nougat-ocr, pdffigures2 wrappers), NLP (spacy, scispacy, transformers), LLM clients (anthropic, openai, langchain-{openai,anthropic,google-vertexai}), DB (sqlalchemy, psycopg2, alembic), web (flask, jinja2), and dev (pytest, ruff).
- **`alembic.ini`** — Standard Alembic config. `script_location = alembic`. Database URL is loaded at runtime from `.env`.
- **`skills-lock.json`** — Plugin manifest pinning Claude Code skills used in this repo (`caveman-commit`, `caveman-review`, `compress`, etc.).

---

## `file-selector/` — data acquisition

Stage 1 of the pipeline. Downloads histopathology paper tarballs from PubMed Central and flattens them into a directory of PDFs and XMLs ready to be parsed.

| File | Role |
|---|---|
| `file_selector.py` | Filter the giant `oa_file_list.csv` for histopathology-related PMC IDs. |
| `file_downloader.py` | Stream-download `.tar.gz` packages from NCBI's OA portal using a list of PMC IDs. |
| `tarball_extractor.py` | Extract `.nxml` and `.pdf` files from the downloaded tarballs into per-paper folders. |
| `pdf_organizer.py` | Flatten per-paper folders into `organized_pdfs/` and `organized_xmls/`. |

### Detailed entries

- **`file_selector.py`** — Reads the OA file list, applies inclusion regexes (histopathology, IHC, digital pathology, etc.) and exclusion regexes, and writes the surviving PMC IDs to `target_pmc_ids.txt`. Run once to seed the downloader.
- **`file_downloader.py`** — Reads `target_pmc_ids.txt`, queries the NCBI OA API for the package URL of each PMC ID, and streams the `.tar.gz` to `histopathology_papers/`. Includes basic retry handling.
- **`tarball_extractor.py`** — Iterates the downloaded tarballs, extracts only `.nxml` (JATS XML) and `.pdf` files into `processed_corpus/{PMCID}/` to avoid filename collisions. Uses temporary directories during extraction so a partial failure cannot corrupt the destination.
- **`pdf_organizer.py`** — Walks `processed_corpus/` and copies/renames the PDFs into a flat `organized_pdfs/` directory and the XMLs into `organized_xmls/`, naming each file `{PMCID}_{original_stem}.{ext}`. The flat layout is what `PipelineRunner.run_batch()` expects.

---

## `database/` — ORM + connection management

SQLAlchemy ORM layer and a small helper for loading credentials from `.env`. Everything that touches PostgreSQL goes through here.

| File | Role |
|---|---|
| `models.py` | All SQLAlchemy ORM models: documents, text elements, figures, tables, entities, plus all summarization-pipeline tables. |
| `db_connection.py` | Connection factory + `session_scope()` context manager; auto-loads `.env`. |
| `__init__.py` | Re-exports models and connection helpers for clean imports. |
| `ENV_LOADING.md` | Explains why `.env` is auto-loaded at module import. |

### Detailed entries

- **`models.py`** — Single source of truth for the schema. Defines:
  - **Core ingestion tables**: `Document` (one per paper, unique `pmcid`), `TextElement` (paragraph with `unique_path`, `path_list`, `path_string`, `depth`, `position_in_section`, full-text `text_content`), `Figure`, `Table`, `Entity`, plus junction tables `TextElementFigureReference` / `TextElementTableReference`.
  - **Pipeline-run table**: `PipelineRun` (one row per summarization invocation; `status`, `config_snapshot`, `started_at`, `finished_at`, `error`).
  - **Summarization tables**: `SumMapFinding` (raw MAP findings), `SumNormalFinding` + `SumNormalFindingSpan`, `SumFindingGroup` + `SumGroupMember`, `SumCanonicalRule`, `SumRelation`, `SumFinalRule`, `SumCorpusRelation` (cross-paper relations), `SumRejectionSummary` + `SumRejectedFinding` (audit trail of dropped findings), and `LlmJudgeCache` (eval result cache).
  - GIN full-text index on `text_content`; cascade delete from `Document`.
- **`db_connection.py`** — Loads `.env` once at module import, exposes `get_db_connection(database_url=None)` factory, and returns a `DatabaseConnection` whose `session_scope()` is the canonical commit/rollback context manager. Pooling is configured via SQLAlchemy defaults.
- **`__init__.py`** — Re-exports `Document`, `TextElement`, `Figure`, `Table`, `Entity`, junction tables, summarization models, and `get_db_connection` so callers can `from database import …` without knowing the file layout.
- **`ENV_LOADING.md`** — Explains the design choice to centralize `dotenv` loading in `db_connection.py` (so individual scripts don't each need their own `load_dotenv()` boilerplate).

---

## `alembic/` — schema migrations

Versioned database migrations. Always pair a `database/models.py` change with a new revision here.

| File | Role |
|---|---|
| `env.py` | Alembic env: imports `Base` from `database.models`, loads `.env`, configures online + offline modes. |
| `versions/0001_add_pipeline_runs.py` | Create `pipeline_runs` table with status tracking. |
| `versions/0002_add_sum_phase2_tables.py` | Add Phase-2 tables (normalized findings, spans). |
| `versions/0003_add_sum_phase3_tables.py` | Add Phase-3 tables (finding groups + members). |
| `versions/0004_add_sum_phase46_tables.py` | Add Phase-4–6 tables (canonical rules, relations, final rules). |
| `versions/0005_add_cui_columns_and_corpus_relations.py` | Add UMLS CUI columns + corpus-level relations table. |
| `versions/0006_add_entity_semantic_types.py` | Add UMLS semantic type metadata to entities. |
| `versions/0007_drop_assertion_status.py` | Drop obsolete `assertion_status` column. |
| `versions/0008_add_rejection_summary_tables.py` | Add audit-trail tables for findings dropped at MAP/GROUP. |
| `versions/0009_split_canonical_scope.py` | Split a single `scope` JSON column into typed scope fields. |
| `versions/0010_add_llm_judge_cache.py` | Add `llm_judge_cache` table for eval-result caching. |

### Detailed entries

- **`env.py`** — Standard Alembic env hooked up to import `database.models.Base`. Reads the live DB URL from `.env`, supports both `--sql` (offline, emit SQL only) and online (live connection) modes.
- **`versions/*.py`** — Each migration is a single atomic schema delta. Read in numerical order to reconstruct the current schema. The migrations roughly mirror the historical evolution of the summarization pipeline: phase-2 tables came first, then phase-3 grouping, then phase-4–6 canonicalization/relations/final rules, then the rejection-summary audit trail, then incremental refinements (CUI columns, semantic types, scope split, judge cache).

---

## `parsers/` — PDF / XML / text utilities

Shared parsing utilities. The PDF pipeline delegates here rather than duplicating logic, and the alternative parsers under `pdf_parsers/` exist primarily for research/comparison.

| File / Subdir | Role |
|---|---|
| `layout_utils.py` | Hub for layout processing: text extraction, artifact filtering, bbox merging, caption matching. |
| `text_processing.py` | `ContextAwareStitcher` (paragraph stitching across tables/figures) + `remove_citations`. |
| `__init__.py` | Empty package marker. |
| `TEXT_PROCESSING_README.md` | Notes on text-processing utilities. |
| `pdf_parsers/` | Alternative PDF parsers (Docling, Marker, PyMuPDF4LLM, Nougat, PDFFigures2, Ensemble). |

### `parsers/` root

- **`layout_utils.py`** — Central hub. Exports `extract_text(layout, pdf_path) → List[HierarchicalRow]`, `filter_artifacts(layout)`, `fix_ligatures`, `merge_rects`, `nearest_caption`, `parse_caption_num`, plus the constants `DOCLING_MASK_TYPES`, `TEXT_ELEMENT_TYPES`, `SKIP_TYPES` and the regexes `FIG_NUM_RE`, `TAB_NUM_RE`. Every PDF-pipeline component delegates here so logic stays single-sourced.
- **`text_processing.py`** — `ContextAwareStitcher.reconstruct_paragraphs()` stitches narrative text that's been split by interleaved tables/figures, tracking source chunks so provenance survives. `remove_citations(text)` strips `[1, 2, 3]`-style markers without breaking inline numerics.

### `parsers/pdf_parsers/` — alternative PDF parsers

| File | Role |
|---|---|
| `base_parser.py` | `BasePDFParser` abstract interface + `HierarchicalPathBuilder` helper. |
| `docling_parser.py` | IBM Docling parser (best for tables and complex layouts). |
| `marker_parser.py` | Fast Marker-based PDF→Markdown parser. |
| `pymupdf4llm_parser.py` | PyMuPDF4LLM clean-markdown parser. |
| `nougat_parser.py` | Nougat neural OCR for equations and complex math. |
| `pdffigures_parser.py` | Allen AI PDFFigures2 (Java) figure/table extractor wrapper. |
| `ensemble_parser.py` | Page-level routing across all parsers. |
| `deduplicator.py` | Strip repeated headers/footers (running text on every page). |
| `__init__.py` | Package marker. |

#### Detailed entries

- **`base_parser.py`** — Defines `BasePDFParser` abstract class with `extract_hierarchy()`, `is_available()`, and validation. Provides `HierarchicalPathBuilder` for assembling section-path tuples from heading levels.
- **`docling_parser.py`** — Wraps Docling's `DocumentConverter` to produce structured Markdown, preserves the section hierarchy, optimized for scientific/technical documents. Slower but most accurate for tables.
- **`marker_parser.py`** — Wraps Marker (`marker-pdf`) for fast PDF→Markdown. Hierarchy is reconstructed from heading levels in the markdown output.
- **`pymupdf4llm_parser.py`** — PyMuPDF4LLM. Fast, clean markdown; good with tables and inline formatting.
- **`nougat_parser.py`** — Transformer OCR (Nougat) trained on arXiv/PMC; ~10× slower than Marker but excellent on equations.
- **`pdffigures_parser.py`** — Subprocess wrapper around the Allen AI PDFFigures2 Java tool. Extracts figure/table bboxes and captions.
- **`ensemble_parser.py`** — Page-level routing: a fast PyMuPDF probe classifies each page (tables / images / narrative) and dispatches to the optimal parser. Outputs are merged. Includes a fallback mode that simply tries parsers in sequence until one succeeds.
- **`deduplicator.py`** — Heuristic header/footer removal for medical papers: lines repeated more than twice in the top/bottom 10% of pages are dropped; lines in the middle 80% are retained as content. Avoids mistakenly removing section headings.

---

## `named_entity_recognition/` — scispaCy + UMLS

Medical entity extraction layered on top of `database.TextElement.text_content`. Uses scispaCy's `en_core_sci_lg` plus the UMLS linker.

| File | Role |
|---|---|
| `ner.py` | Core scispaCy + UMLS linker pipeline. |
| `batch_ner.py` | Threaded NER over many documents with shared cache. |
| `merge_entities_by_umls.py` | Aggregate text elements grouped by UMLS CUI. |
| `export_disease_entities.py` | Filter to disease-relevant TUIs and export grouped JSON/TXT. |
| `count_tokens.py` | Tokenization + LLM cost estimator for downstream summarization. |
| `enums.py` | UMLS TUI → human-readable disease semantic type map. |
| `__init__.py` | Package marker. |

### Detailed entries

- **`ner.py`** — Loads `en_core_sci_lg` plus `scispacy.linking.EntityLinker` configured for UMLS with abbreviation resolution. Filters out junk semantic types (organisms, animals, geographic entities) and acronyms that confuse the linker. The single-document entry point used by `pipeline/stages/summarization/runner.py` is `run_ner_on_db(pmcid, save_to_db=True)`.
- **`batch_ner.py`** — `ThreadPoolExecutor` over many documents. Loads `entity_linking_cache.json` (~66 MB) at startup, gives each worker its own model instance (scispaCy is not thread-safe), updates the shared cache under a lock.
- **`merge_entities_by_umls.py`** — Groups all `TextElement` rows by the CUI of the entities they contain, producing one JSON per concept with the canonical name, all unique surface forms, and every sentence (with `pmcid` + `text_element_id` + position) where it appears.
- **`export_disease_entities.py`** — Filters entities to disease-relevant TUIs (`T047 Disease or Syndrome`, `T191 Neoplastic Process`, `T048`, …) using `enums.py`, then exports JSON/TXT files grouped by CUI. Configurable output dir and minimum-occurrence threshold.
- **`count_tokens.py`** — Tokenization (whitespace and OpenAI tiktoken) plus per-stage cost estimator for the MAP/REDUCE/RULES pipeline. Includes pricing for GPT-4o and GPT-4o-mini.
- **`enums.py`** — Static dict mapping UMLS TUI codes to human-readable semantic type names (`T047 → Disease or Syndrome`, etc.).

---

## `pipeline/utils/` — shared pipeline utilities

Cross-cutting helpers shared by both the PDF-extraction and summarization pipelines.

| File | Role |
|---|---|
| `memory_logging.py` | `MemoryLogger` + `get_default_memory_logger()` — emits grep-friendly `MEMORY stage=… event=… rss_mb=… vms_mb=… elapsed_s=… delta_rss_mb=…` lines at pipeline checkpoints. Uses `psutil`; degrades to stage/event/elapsed only when psutil is unavailable. Used by `summarization/runner.py` (before/after every stage) and `summarization/umls_resources.py` (scispaCy load/enrich events). |

---

## `pipeline/stages/pdf_text_extraction/` — main PDF→DB pipeline

The production pipeline that turns one PDF into rows in the database. Eight steps:

1. extract layout (Docling) → 2. detect tables → 3. mask detected regions → 4. re-extract from masked PDF → 5. filter artifacts → 6. assemble hierarchical text → 7. crop figures/tables → 8. write outputs (text file, media JSON, DB).

| File / Subdir | Role |
|---|---|
| `runner.py` | `PipelineRunner` — orchestrates all 8 steps for one document or a directory. |
| `config.py` | All sub-configs (`PathConfig`, `DoclingConfig`, `TATRConfig`, `MaskingConfig`, …) + `PipelineConfig`. |
| `batch.py` | `ParallelBatchRunner` — `ThreadPoolExecutor` with per-thread `PipelineRunner` instances. |
| `blacklist.py` | `BlacklistManager` — thread-safe JSON-backed skip list. |
| `__init__.py` | Public exports (`PipelineRunner`, `PipelineConfig`, …). |
| `models/` | DTOs: `BoundingBox`, `LayoutElement`, `LayoutResult`, `HierarchicalRow`, `ScoredNode`. |
| `interfaces/` | Protocols for each replaceable stage. |
| `components/` | Concrete stage implementations. |
| `table_detectors/` | Docling, TATR, and Hybrid table detectors. |
| `outputs/` | Output writers: text file, media JSON, Postgres ingester. |

### `pipeline/stages/pdf_text_extraction/` root

- **`runner.py`** — `PipelineRunner` lazily instantiates each stage on first use. `run_document(pdf_path, pmcid)` runs the 8-step pipeline; `run_batch(pdf_dir)` iterates a directory. Important methods include `_steps_1_3_4_standard` (mask → re-extract) and `_steps_1_3_4_two_pass` (use the ghost-text two-pass extractor as an alternative). `_patch_section_header_types` re-promotes `TEXT` elements back to `SECTION_HEADER` if the masked re-extraction lost the type. Skip logic: blacklist, completed-set, already-in-DB, existing media-JSON.
- **`config.py`** — Dataclass-based hierarchical configuration. Sub-configs: `PathConfig` (output dirs), `DoclingConfig` / `TATRConfig` (model knobs), `MaskingConfig` / `FilteringConfig` / `CroppingConfig` / `TextAssemblyConfig` / `VisualizationConfig` / `DatabaseConfig` / `RuntimeConfig`. Enums: `TableDetectorType` (`TATR`, `DOCLING`, `HYBRID`, `VLM`), `BaselineMode` (`MASKED`, `UNMASKED`, `BOTH`), `OcrEngine`. `PipelineConfig.prepare()` validates and creates output directories.
- **`batch.py`** — `ParallelBatchRunner` builds one `PipelineRunner` per worker thread (Docling models are not thread-safe). Manages a shared `BlacklistManager`, locks for stats collection. Default `max_workers = cpu_count // 2`.
- **`blacklist.py`** — `BlacklistManager` is a thread-safe set persisted to JSON. Used for "skip these PDFs" (failed) and "completed" lists.
- **`__init__.py`** — Re-exports `PipelineRunner`, `PipelineConfig`, table-detector enums, etc., for clean imports. Each component lazy-loads its own ML model (Docling converter in `DoclingLayoutExtractor`, TATR in `TATRTableDetector`, scispaCy via `summarization/umls_resources.get_nlp()`); there is no shared model registry.

### `pipeline/stages/pdf_text_extraction/models/`

- **`dto.py`** — Pure data classes shared across stages: `BoundingBox` (Docling coords with `to_fitz_rect()` / `from_fitz_rect()` for PyMuPDF interop), `LayoutElement` (one element with `type`, `page`, `bbox`, `text`, `level`), `LayoutResult` (all elements + `page_dims`), `HierarchicalRow` (assembled paragraph + path), `CroppedMedia`, `TableDetectionResult`.
- **`scored_node.py`** — `ScoredNode` for the two-pass extractor: holds an element plus its R1 (pixel), R2 (word-count), and R3 (caption-pattern) scores.

### `pipeline/stages/pdf_text_extraction/interfaces/`

Protocol classes (PEP 544) defining the contract for each pluggable stage.

| File | Protocol |
|---|---|
| `layout_extractor.py` | `LayoutExtractor.extract(pdf_path) → LayoutResult` |
| `table_detector.py` | `TableDetector.detect(pdf_path) → TableDetectionResult` |
| `region_masker.py` | `RegionMasker.collect_regions(...)`, `mask(pdf_path, regions)` |
| `artifact_filter.py` | `ArtifactFilter.filter_elements(elements)` |
| `text_assembler.py` | `TextAssembler.assemble(layout) → List[HierarchicalRow]` |
| `media_cropper.py` | `MediaCropper.crop(pdf_path, layout, detection)` |
| `output_writer.py` | `OutputWriter.write(pmcid, rows, figures, tables, …)` |

### `pipeline/stages/pdf_text_extraction/components/`

| File | Role |
|---|---|
| `layout_extractor.py` | `DoclingLayoutExtractor` — Docling wrapper with JSON cache + ghost-text dedup. |
| `region_masker.py` | `PyMuPDFRegionMasker` — overlays white rectangles on detected regions. |
| `text_assembler.py` | `HierarchicalTextAssembler` — delegates to `parsers/layout_utils.extract_text`. |
| `artifact_filter.py` | `ArtifactFilter` — delegates to `parsers/layout_utils.filter_artifacts`. |
| `media_cropper.py` | `PyMuPDFMediaCropper` — saves figure/table crop images. |
| `two_pass_extractor.py` | `TwoPassTextExtractor` — pixel-based ghost-text detection alternative to masking. |
| `evidence_gatherer.py` | `PyMuPDFEvidenceGatherer` — gather pixel/text evidence for a node. |
| `node_scorer.py` | `NodeScorer` — R1/R2/R3 scoring used by the two-pass extractor. |
| `table_reconstructor.py` | `reconstruct_tables_from_lists()` — rebuild tables from list-element clusters Docling missed. |
| `visualizer.py` | `DetectionVisualizer` — produce annotated PDFs for auditing detections. |

#### Detailed entries

- **`layout_extractor.py`** — Wraps Docling's `DocumentConverter`. Caches the full layout JSON to `out/docling_full/` so re-runs of the same PDF skip the slow Docling call. Includes ghost-text deduplication for PDFs that ship with stripped, invisible text layers.
- **`region_masker.py`** — `PyMuPDFRegionMasker` collects regions to mask (detected tables/figures, optionally headers/footers/sidebars), opens the PDF with PyMuPDF, and overlays white rectangles. Output PDF is saved to `out/masked_pdfs/{pmcid}_masked.pdf`.
- **`text_assembler.py`** — `HierarchicalTextAssembler.assemble(layout)` delegates almost entirely to `parsers.layout_utils.extract_text`. Optional scispaCy NLP for paragraph-relevance filtering. `skip_references_section=True` by default.
- **`artifact_filter.py`** — Removes layout false positives (very short/long strings, isolated punctuation, page numbers, etc.). Optional NER pass for additional filtering. Delegates to `parsers.layout_utils.filter_artifacts`.
- **`media_cropper.py`** — Uses PyMuPDF to crop bounding boxes into PNG/JPEG images. Supports optional caption-based merging (TATR + Docling detections that share a caption are merged) and footnote expansion (extend table bboxes downward to capture footnotes).
- **`two_pass_extractor.py`** — `TwoPassTextExtractor.process(pdf_path)` extracts the layout twice: pass 1 is full extraction; pass 2 masks elements that scored as "ghost text" (likely OCR artifacts or invisible glyphs) and re-extracts. Uses `NodeScorer` and `PyMuPDFEvidenceGatherer`. R1 (pixel-based, primary) is fallback-stable to R2 (word count, fitz) when numpy is unavailable; R3 is caption-pattern matching.
- **`evidence_gatherer.py`** — Per-node evidence: rendered pixels (numpy) and PyMuPDF word counts inside the bbox.
- **`node_scorer.py`** — Combines the three rules (R1/R2/R3) into a per-node score that decides whether a node is real text or ghost text.
- **`table_reconstructor.py`** — `reconstruct_tables_from_lists(layout)` clusters consecutive `LIST_ITEM` elements that look tabular into synthetic `RECONSTRUCTED_TABLE` regions Docling missed.
- **`visualizer.py`** — `DetectionVisualizer.visualize_layout()` and `visualize_detections()` produce annotated PDFs in `out/visualization/` for human auditing.

### `pipeline/stages/pdf_text_extraction/table_detectors/`

| File | Role |
|---|---|
| `docling_detector.py` | `DoclingTableDetector` — table regions extracted from Docling's existing layout. |
| `tatr_detector.py` | `TATRTableDetector` — Microsoft TATR transformer model. |
| `hybrid_detector.py` | `HybridTableDetector` — runs both, merges via `merge_rects`. |

#### Detailed entries

- **`docling_detector.py`** — `DoclingTableDetector.detect_from_layout(layout) → TableDetectionResult`. Pure layout-based detection; cheap because Docling already ran.
- **`tatr_detector.py`** — `TATRTableDetector(config).detect(pdf_path)`. Loads the TATR transformer; only exposes `detect(pdf_path)` (no `detect_from_layout`).
- **`hybrid_detector.py`** — Runs both detectors and merges overlapping bboxes with `parsers.layout_utils.merge_rects`. Exposes `detect_with_layout(layout, pdf_path)` so the runner can pass in the already-extracted layout to avoid re-running Docling. This is the production default.

### `pipeline/stages/pdf_text_extraction/outputs/`

| File | Role |
|---|---|
| `writer.py` | `TextFileWriter` — write hierarchical text to `out/text/{pmcid}.txt`. |
| `media_json_writer.py` | `MediaJsonWriter` — write `{pmcid}_media.json` with figure/table metadata. |
| `db_ingester.py` | `PostgresDatabaseIngester` — upsert documents, text elements, figures, tables, references. |

#### Detailed entries

- **`writer.py`** — Writes the hierarchical text rows to `out/text/{pmcid}.txt` with section paths inlined.
- **`media_json_writer.py`** — Writes a sidecar JSON with figure/table metadata (label, caption, page, bbox, crop file path) for downstream tools.
- **`db_ingester.py`** — `PostgresDatabaseIngester.write(...)` upserts the `Document` row, then bulk-inserts the `TextElement` rows (computed `unique_path = {pmcid}/{section_path}/{position}`), `Figure` and `Table` rows, and the junction tables linking text elements to the figures/tables they reference. Uses `database.session_scope()`.

---

## `pipeline/stages/summarization/` — LLM summarization pipeline

Turns the paragraph text in `TextElement` rows into auditable, normalized clinical findings, then into canonical rules. Uses a 3-tier ABC cascade of LLM voters with explicit grounding checks at every stage.

Pipeline order:

```
sentences ─► MAP ─► GROUNDING ─► NORMALIZE ─► GROUP ─► CANONICALIZE ─► RELATE ─► RESOLVE
                                                                                  └─► [optional] REDUCE → RULES
```

| File / Subdir | Role |
|---|---|
| `runner.py` | `SummarizationRunner` — orchestrates all stages + DB persistence. |
| `config.py` | `SummarizationConfig` (nested `MapConfig`, `GroundingConfig`, `RelateConfig`, `ResolveConfig`, …). |
| `models.py` | All Pydantic models for findings, groups, rules, relations, scope, etc. |
| `prompts.py` | LangChain prompt templates and chain factories for MAP / REDUCE / RULES / JUDGE. |
| `cache.py` | `PipelineCache` — three-tier (map/reduce/rule) disk-backed JSON cache. |
| `llm_providers.py` | LLM instantiation helpers (Azure / Vertex / Anthropic). |
| `llm_errors.py` | Retryable vs. non-retryable LLM exception classification. |
| `umls_utils.py` | UMLS lookup utilities. |
| `umls_resources.py` | Process-wide singleton loader for scispaCy + UMLS linker (loaded at most once per process). |
| `persistence.py` | DB persistence helpers used by `runner.py` to write per-stage results. |
| `artifact_models.py` | Pydantic schemas for on-disk pipeline artifacts (rejection summaries, audit dumps). |
| `enum_logging.py` | Helper for logging Enum values without their repr. |
| `synonyms.yaml` | Curated subject/outcome surface-form synonyms used by `normalize_stage`. |
| `current_stages/` | Active stage implementations. |
| `old_stages/` | Optional REDUCE + RULES (off by default). |
| `helpers/` | Cross-stage helpers: grounding filter, contradiction detector, corpus relate, entity linker. |
| `agreement/` | Voter-agreement scorers (embedding, NER, hybrid, semantic, LLM judge). |
| `routing/` | MAP-stage router: schema/provenance gates and policy. |
| `batch/` | Async batch submission to Anthropic / Azure / Vertex / Gemini. |
| `observability/` | JSONL trace collection. |
| `interfaces/` | Protocols for scorers, grounding, contradiction, similarity. |

### `pipeline/stages/summarization/` root

- **`runner.py`** — `SummarizationRunner.process(file_data)` runs one paper through every stage, persisting intermediate outputs to the `Sum*` tables under a single `PipelineRun` row. Skips re-processing when a cached result JSON exists (override with `force_rerun=True`). Replaces LLM-paraphrased `verbatim_support` with the actual `TextElement.text_content` from the DB before grounding (so NLI scores real paragraphs, not paraphrases). Optionally runs scispaCy NER + UMLS linking after the pipeline. The optional REDUCE→RULES block is disabled by default. Wraps every stage call in a `MemoryLogger` context manager so each run emits a sequence of grep-friendly `MEMORY stage=X event=before/after/failed …` lines for OOM diagnosis. Static helper `load_paper_from_db(pmcid)` returns a `{pmcid, sentences_with_provenance}` dict ready for `process()`.
- **`config.py`** — `SummarizationConfig` is a single dataclass holding nested configs for every stage. Every numeric/boolean knob (theta, reject_theta, chunk_size, NLI thresholds, grounding threshold, scoring weights, …) lives here. Use `dataclasses.replace()` to override specific fields.
- **`models.py`** — Pydantic schemas for the entire pipeline. Highlights: `DirectionEnum`, `RelationTypeEnum`, `FindingScope` (typed scope with `disease_subtype`, `cohort_n`, `assay_method`, `biomarker_cutoff`, `tissue_site`, `treatment_context`, `endpoint`, `study_design`), `Finding` (raw MAP output with claim/verbatim_support/evidence/scope/confidence), `AuditableSummary` (one chunk's findings + voter trace), `NormalFinding`, `FindingGroup`, `CanonicalRule`, `Relation`, `FinalRule`, `RejectionSummary` + `RejectedFinding`, plus `ConsolidatedSummary` and `ExtractedRules` for the legacy REDUCE/RULES block.
- **`prompts.py`** — LangChain `ChatPromptTemplate`s plus chain factories. MAP prompt instructs voters to extract atomic findings with verbatim source + evidence refs `S{i}|{pmcid}|{te_id}` and a strict typed scope. REDUCE / RULES / JUDGE prompts also live here. All chains use `.with_structured_output()` for reliable JSON parsing.
- **`cache.py`** — `PipelineCache` is a disk-backed JSON cache keyed by deterministic fingerprints (text-element IDs, chunk IDs, model name). Three independent caches (map/reduce/rule). Tracks hits/misses; `stats_str()` produces a one-line summary. Survives restarts so re-runs are cheap.
- **`llm_providers.py`** — Helper factories for instantiating LangChain chat models for Azure OpenAI, Vertex / Gemini, Anthropic, etc. Handles auth / region / endpoint plumbing.
- **`llm_errors.py`** — Exception classification: distinguishes transient (rate-limit, 5xx, network) from permanent (auth, schema validation) errors so retry logic can be smart.
- **`umls_utils.py`** — Helpers for UMLS lookups used by `helpers/entity_linker.py` and `current_stages/normalize_stage.py`.
- **`umls_resources.py`** — `get_nlp()` / `get_linker()` return a process-wide scispaCy + UMLS singleton. Loading the ~5 GB KB twice was a reliable way to OOM-kill the pipeline; every component that needs the linker obtains it through this module. Silent-failing: returns `None` when scispaCy/UMLS is unavailable or has been disabled via `$NLP_HISTO_DISABLE_UMLS`. Emits `MEMORY stage=UMLS event=before_scispaCy_load / after_scispaCy_load` checkpoints through the shared `MemoryLogger`.
- **`persistence.py`** — SQLAlchemy persistence helpers (`_persist_map_findings`, `_persist_normal_findings`, `_persist_finding_groups`, …) that `runner.py` calls between stages to materialize per-stage results into the `Sum*` tables.
- **`artifact_models.py`** — Pydantic schemas for the per-stage artifacts written to disk (rejection summaries, audit dumps) when an artifact-writer root is configured.
- **`enum_logging.py`** — Utility for logging Enum values as `.value` so `RelationTypeEnum.SUPPORTS` shows up as `SUPPORTS` instead of `<RelationTypeEnum.SUPPORTS: 'SUPPORTS'>`.

### `pipeline/stages/summarization/current_stages/`

The actively-used stages. They are deterministic except for `map_stage.py` (which calls LLM voters).

- **`map_stage.py`** — Agreement-Based Cascading (ABC) per chunk. Level-1 voter LLMs run in parallel; if `ScoreBundle` agreement < `theta`, escalates to Level-2; if still disagree, escalates to Level-3 (premium model). Output: `AuditableSummary[]` with per-chunk findings + full voter trace + agreement metadata. The router-mode path can bypass Level-2 for faster escalation; the legacy path is the standard 3-tier cascade.
- **`normalize_stage.py`** — Entity normalization via synonym dictionary + UMLS linker + identity fallback, then conditional dedup on `(text_element_id, subject, outcome, relation_type)`. Also infers `direction` from claim keywords. Splits inputs into groupable vs. non-groupable; only groupable are passed downstream.
- **`group_stage.py`** — Pure grouping by `(subject_entity, outcome_entity, relation_type, category)`. Computes `direction_counts` and a `scope_heterogeneity` flag indicating whether members disagree on scope. Validates inputs come pre-filtered to groupable findings.
- **`canonicalize_stage.py`** — Picks each group's canonical predicate text (highest grounding wins), computes `study_coverage` and `is_conflicted` metadata, optionally splits groups by direction. No LLM; fully deterministic.
- **`relate_stage.py`** — NLI-based pairwise comparison of canonical rules. Four-condition comparability gate (category match, relation_type match, subject and outcome share entity / CUI). Uses bidirectional entailment scores from a DeBERTa cross-encoder to label pairs `SUPPORT` / `CONTRADICT` / `SCOPE_QUALIFY` / `UNRELATED`. Direction-polarity guard prevents same-direction rules from being labelled as contradicting.
- **`resolve_stage.py`** — Weighted scoring of canonical rules using grounding, finding count, support relations, and study coverage. Two formulas: one for rules with relations, one for rules without. Outputs `FinalRule[]` ranked by `final_score` descending.

### `pipeline/stages/summarization/old_stages/`

Optional secondary block. Disabled by default (`run_reduce=False`).

- **`reduce_stage.py`** — `ReduceStage.reduce()` does a tree-reduce over chunk summaries into a single `ConsolidatedSummary` with sections (clinical_significance, histopathological_features, management_outcomes, risk_factors_associations).
- **`rule_stage.py`** — `RuleStage.extract()` extracts `IF-THEN` rules from a `ConsolidatedSummary`.

### `pipeline/stages/summarization/helpers/`

- **`grounding_filter.py`** — `GroundingFilter` runs an NLI cross-encoder (DeBERTa) on `(verbatim_support, claim)` pairs. Threshold-based drop (default 0.5). Used after MAP to drop ungrounded findings, and optionally after RULES to drop ungrounded rules.
- **`contradiction_detector.py`** — Two-step contradiction detection between `Rule[]` outputs: pairwise embedding similarity selects candidate pairs above `similarity_threshold`, then an LLM judge confirms whether each pair genuinely contradicts. Produces `ContradictionReport`.
- **`corpus_relate.py`** — `CorpusRelateStage` pools `CanonicalRule[]` from many papers and runs the same NLI comparison as `RelateStage`. Each output `CorpusRelation` is labelled `intra_paper` or `cross_paper`. `relate_incremental()` does the cross-paper diff for a newly processed paper without re-running the whole corpus.
- **`entity_linker.py`** — `enrich_rules_with_cuis(rules)` fills in `subject_cui` / `outcome_cui` on `CanonicalRule[]` in-place. Delegates model loading to `umls_resources.get_nlp()` so scispaCy + UMLS is loaded at most once per process. Silently no-ops when the linker is unavailable, or when `$NLP_HISTO_SKIP_UMLS_ENRICHMENT` is set. Emits `MEMORY stage=UMLS event=before_UMLS_enrichment / after_UMLS_enrichment` checkpoints.

### `pipeline/stages/summarization/agreement/`

Pluggable voter-agreement scorers used by `MapStage` to decide whether to escalate.

- **`checker.py`** — `AgreementChecker` thin orchestrator wrapping a scorer. `compute()` ensures `ScoreBundle.decision` is always set by applying `theta` thresholds when the scorer doesn't decide. `best()` selects the highest-scoring voter from a bundle.
- **`providers.py`** — `OpenAIEmbedder` and `GeminiEmbedder` callables implementing the `EmbedFn` protocol. Lazy-loaded; injectable into scorers.
- **`embedding.py`** — `EmbeddingScorer`: pairwise cosine similarity over claim embeddings, averaged off-diagonal per voter. Polarity / numeric contradiction heuristics override the score when claims disagree on direction or magnitude.
- **`embedding_similarity.py`** — Reusable similarity utilities (cosine matrix, top-k).
- **`composite.py`** — `CascadedCompositeScorer` applies LP-optimized thresholds (`keep_emb`, `keep_ner`, `reject`) over a hybrid embedding+NER feature vector. Loads thresholds from a calibration JSON. Decision rule: KEEP / REJECT / ESCALATE.
- **`hybrid_scorer.py`** — `HybridScorer` combines `EmbeddingScorer` + `NERScorer` into one `ScoreBundle` (no decision; the composite scorer applies thresholds).
- **`hybrid_structured.py`** — Structured similarity that also considers field-level agreement on subject / outcome / relation_type.
- **`lexical_similarity.py`** — Token / term overlap similarity (no embeddings).
- **`llm_judge.py`** — `LLMJudgeScorer` does a single LLM call presenting all voter outputs together, asking for one overall agreement score. Cheaper than O(n²) pairwise judges for small voter counts.
- **`ner_scorer.py`** — `NERScorer` computes pairwise scispaCy entity Jaccard overlap. Thread-safe lazy spaCy load.
- **`semantic_scorer.py`** — `SemanticAgreementScorer` implements the max-consensus deferral metric from "Semantic Agreement Enables Efficient Open-Ended LLM Cascades": `deferral_score = max(per-voter avg similarity)`, `best_candidate = argmax` with grounding tie-breaking.
- **`category_jaccard.py`** — Jaccard similarity over the per-finding `category` set (morphology, IHC, prognosis, …).

#### `agreement/calibration/`

- **`dataset.py`** — Routing-record dataset used to fit thresholds.
- **`gold_labeler.py`** — Labels routing records with KEEP/ESCALATE ground truth so the optimizer can fit thresholds.
- **`threshold_optimizer.py`** — `OptimizedThresholds` + `ThresholdOptimizer` LP-based fitting of KEEP/REJECT/ESCALATE boundaries on hybrid features. Loads / saves JSON consumed by `CascadedCompositeScorer.from_file()`.

### `pipeline/stages/summarization/routing/`

The MAP-stage routing layer that gates voter outputs on schema and provenance before agreement scoring.

- **`router.py`** — `MapOutputRouter` runs `SchemaValidator` + `ProvenanceValidator`, classifies each voter as `ELIGIBLE` / `WEAKLY_GROUNDED` / `UNUSABLE`, and derives a chunk-level decision (`KEEP` / `ESCALATE` / `REJECT`) from the count of eligible voters.
- **`schema_validator.py`** — Structural checks on `AuditableSummary`: valid categories, non-empty claims/evidence, required fields present. Returns per-finding `FindingValidation` with `ReasonCode`.
- **`provenance_validator.py`** — Validates voter outputs against source text: detects fabricated verbatim, missing sentence IDs, cross-document evidence errors, weakly grounded claims.
- **`policy.py`** — `RoutingPolicySpec` (config), `PolicyEvaluationResult` (precision/recall/etc.), `PolicySelectionResult` (ILP optimizer output with Pareto frontier).
- **`models.py`** — Enums and dataclasses for the routing audit trail (`RoutingDecision`, `ReasonCode`, `FindingValidation`, `GateOrigin`, …).
- **`routing_dataset.py`** — `RoutingDataset` / `RoutingRecord` for collecting and persisting routing decisions with full audit trail (voters, scores, decision, deferral gate, escalation flags).

### `pipeline/stages/summarization/batch/`

Asynchronous batch dispatch for cheap multi-paper runs.

- **`runner.py`** — `BatchSummarizationRunner` orchestrates `submit() → advance() → finalize()`. `submit()` queues L1 voter jobs; `advance()` polls, applies the agreement gate, and escalates to L2/L3 as needed; `finalize()` runs REDUCE+RULES synchronously. Persists `BatchHandle` to disk so a restart resumes where it left off.
- **`dispatch.py`** — Shared utilities: `format_messages()` renders MAP prompts; `build_requests()` produces one `BatchRequest` per `(chunk, voter)` pair; `parse_result()` extracts `AuditableSummary` from provider responses. Defines the `OPENAI_MAP_TOOL` JSON schema and provider instantiation.
- **`models.py`** — `BatchHandle`, `BatchPhase`, `BatchResult`, `ProviderJob`, `VoterBatchConfig` dataclasses.
- **`azure_batch.py` / `claude_batch.py` / `gemini_batch.py` / `vertex_batch.py`** — Provider-specific batch APIs implementing the common interface (create batch, submit, poll, retrieve, parse).

### `pipeline/stages/summarization/observability/`

- **`collector.py`** — `TraceCollector` is a mutable accumulator for one paper's pipeline run. Records `ChunkTrace`, `MapStageTrace`, grounding stats, warnings, config snapshot. Finalized into a `RunTrace` and flushed to JSONL.
- **`models.py`** — Dataclasses for traces (`ChunkTrace`, `MapStageTrace`, `GroundingTrace`, …).
- **`export.py`** — JSONL writer; also has `export_all_csv(trace_dir)` to convert per-paper JSONL traces into a corpus-wide CSV summary.

### `pipeline/stages/summarization/interfaces/`

- **`agreement.py`** — `MapOutputScorer` Protocol (`compute(outputs, source_text, context) → ScoreBundle`). Defines `ChunkDecision` enum, `ScoreBundle` (with `embedding_agreement`, `entity_overlap`, `judge_agreement`, etc.), `AgreementContext` and `VoterContext`.
- **`contradiction.py`** — `ContradictionChecker` protocol.
- **`grounding.py`** — `GroundingChecker` protocol (`filter_findings_with_scores`, `filter_rules`).
- **`scoring.py`** — Scoring-related protocols (separation between scorers and graders).
- **`similarity.py`** — `SimilarityFn` and related protocols.

---

## `eval/` — evaluation harness

Two complementary evaluation tracks:
- **Detection eval** — `run.py` + `precision_recall.py` + `recall.py` + `annotate.py` + `ground_truth.py` measure figure/table detection precision/recall.
- **LLM-judge eval (silver labels)** — `llm_judge/` and `silver/` use Claude Opus to produce silver labels, then evaluate the pipeline against them on Q1 (precision), Q2 (relations), Q3 (recall), Q5 (extraction F1).

| File / Subdir | Role |
|---|---|
| `run.py` | Sample N PDFs and run the PDF pipeline with visualization into `eval/out/`. |
| `annotate.py` | Interactive terminal annotator for figure/table/text correctness. |
| `auto_annotate.py` | Copy annotations between identical detector variants. |
| `ground_truth.py` | Three-pass interactive ground-truth collection (missed figs, missed tables, total tables). |
| `precision_recall.py` | Compute P/R/F1 from annotations vs. ground truth CSV. |
| `recall.py` | Substring-based recall: do annotated correct elements appear in pipeline output? |
| `llm_judge.py` | Backward-compat wrapper delegating to `eval/llm_judge/`. |
| `__init__.py` | Package marker. |
| `llm_judge/` | LLM-as-judge harness (Claude Opus + caching). |
| `silver/` | Silver-label generation, matching, and threshold sweeping. |

### `eval/` root

- **`run.py`** — Build a `PipelineConfig` rerouted to `eval/out/`, sample N PDFs (default 10) reproducibly with a seed, filter to `0.5–10 MB`, and run `ParallelBatchRunner`. Visualization, raw text dump, and multi-source crops are all on so detections can be audited.
- **`annotate.py`** — Terminal UI for labelling extracted items. Modes: `text`, `text_raw`, `docling_full`, `figures`, `tables` (and per-detector variants). Keystrokes: `y/n/s/b/r/q`. Saves progress to `annotations_{mode}.json` after every keypress.
- **`auto_annotate.py`** — Copies annotations between detector variants when the underlying figures/tables match exactly on `(label, caption, page, bbox)`. Cuts re-annotation cost.
- **`ground_truth.py`** — Three sequential passes, prompting the user for missed-figure count, missed-table count, total-table count per PDF. Opens crops in macOS Preview. Saves to `ground_truth.csv` with resume support.
- **`precision_recall.py`** — Computes per-detector and per-label precision/recall/F1. Classifies each annotated row as TP/FP/skip from label prefix and compares against ground-truth counts.
- **`recall.py`** — 40-character substring match: did each "correct" element from `docling_full` survive into the pipeline's `text_raw` output? Per-document and overall recall, with optional per-document log.
- **`llm_judge.py`** — Thin wrapper that ensures the project root is on `sys.path` then dispatches to `eval/llm_judge/__main__.py`.

### `eval/llm_judge/`

| File | Role |
|---|---|
| `__main__.py` | argparse CLI: mode (`sync` / `batch`), paper count, per-test sampling, cache control. |
| `runner.py` | Orchestrator: paper selection, request building, cache filtering, sync/batch execution, result writing. |
| `client.py` | Anthropic API client (sync + Anthropic Batches API). |
| `cache.py` | Postgres-backed eval cache keyed by SHA256 of `(model, prompt_version, schema_version, request)`. |
| `prompts.py` | System + user prompt templates and JSON schemas for Q1/Q2/Q3/Q5. |
| `metrics.py` | Aggregate judge results into per-test summary metrics + calibration curves. |
| `sampling.py` | Deterministic stratified paper / paragraph sampling. |
| `tests/q1_precision.py` | Q1: judge MAP-finding grounding correctness. |
| `tests/q2_relations.py` | Q2: judge RELATE labels (SUPPORT/CONTRADICT/SCOPE_QUALIFY/UNRELATED). |
| `tests/q3_recall.py` | Q3: identify generalizable findings missing from pipeline extraction. |
| `tests/q5_f1.py` | Q5: produce silver findings, align to pipeline, compute extraction F1. |

#### Detailed entries

- **`__main__.py`** — argparse entry point; flags for mode, paper count, per-test sampling caps, cache control, batch resume. Builds a `RunConfig` and calls `runner.run(cfg)`.
- **`runner.py`** — Selects the latest successful `PipelineRun` per PMCID, builds judge requests from the DB, filters out cache hits, executes sync or batch with exponential backoff, parses results, writes per-test JSONL + a metrics summary. Carefully manages `session_scope` to avoid nested-session corruption.
- **`client.py`** — Wraps the Anthropic SDK for both sync calls and the Batches API. Submits batches, polls for completion, retrieves results, and persists batch metadata so a restart can resume.
- **`cache.py`** — `LlmJudgeCache` table CRUD. Cache key is SHA256 of `(model_id, prompt_version, schema_version, request_payload)`. Supports version-aware invalidation and upsert-on-conflict.
- **`prompts.py`** — Per-test system prompts, user templates, and JSON schemas. The schemas use controlled enums that mirror the pipeline `models.py` enums so judge output is comparable. Filter rules distinguish generalizable medical claims from patient-specific narrative.
- **`metrics.py`** — Aggregates per-test results: field accuracy for Q1, confusion matrix for Q2, recall-gap rate for Q3, P/R/F1 for Q5. Includes per-stratum breakdowns and calibration curves binned by judge confidence.
- **`sampling.py`** — Deterministic seeded sampling. Per-paper: latest successful run, word-count filter. Per-paragraph: stratified by `with-extraction` vs. `zero-extraction`, with boilerplate-section filtering on the zero stratum.
- **`tests/q1_precision.py`** — For each MAP finding, Opus is asked: is the claim grounded in the verbatim source? Returns the corrected fields if not. Computes `fields_changed`.
- **`tests/q2_relations.py`** — For each persisted relation, Opus blind-judges the `SUPPORT/CONTRADICT/SCOPE_QUALIFY/UNRELATED` label. (Recall over candidate pairs is deferred until all candidate pairs have NLI labels.)
- **`tests/q3_recall.py`** — Opus reviews paragraphs (mixed strata) and reports any generalizable findings the pipeline missed. Returns `missing_findings` and a `has_gaps` bool.
- **`tests/q5_f1.py`** — Opus produces a complete silver finding set per paragraph. The runner aligns silver→pipeline with `match_type` (match / partial / unmatched), then Python computes TP/FP/FN/P/R/F1.

### `eval/silver/`

The silver-label generation track. Generate Opus labels once, then evaluate the pipeline against them with cheap embedding matching.

| File | Role |
|---|---|
| `sample.py` / `sampler.py` | CLI + implementation for sampling source paragraphs from the DB. |
| `generate.py` / `generator.py` | CLI + implementation for Opus-generated silver findings. |
| `prompts.py` | Opus extraction prompt (v2) with strict telegraphic style and atomicity rules. |
| `export_pipeline.py` / `exporter.py` | CLI + implementation for exporting matching pipeline findings. |
| `matcher.py` | Embedding-based silver↔pipeline matching with greedy assignment. |
| `evaluate.py` | End-to-end: sample → generate → export → match → report. |
| `inspect.py` | Render human-readable Markdown + CSV inspection reports. |
| `sweep.py` | Threshold sweep (0.40–0.80) over similarity threshold. |
| `split.py` | Hash-based deterministic dev/test split (80/20). |
| `schemas.py` | Pydantic models: `SourceCase`, `SilverFinding`, `PipelineFinding`, `MatchResult`, `EvalMetrics`. |
| `jsonl_utils.py` | Read/write/append JSONL with Pydantic validation. |
| `tests/test_matcher.py` | Unit test for `matcher.py`. |

#### Detailed entries

- **`sample.py` / `sampler.py`** — Samples paragraphs from `TextElement` (min 40 words, boilerplate sections excluded), n per PMCID, deterministic seed. Output: `source_cases.jsonl`.
- **`generate.py` / `generator.py`** — Calls Opus on each source paragraph with the v2 prompt, parses `SilverFinding[]` via tool_use, caches by `(case_id, prompt_version, model)`. Output: `silver_findings.jsonl`.
- **`prompts.py`** — Opus extraction prompts (v2). Emphasis on atomicity, zero loss, no contextual drift, telegraphic style; explicit exclusion of patient narratives.
- **`export_pipeline.py` / `exporter.py`** — For each `SourceCase`, queries `sum_map_findings` (latest successful pipeline run per PMCID) for findings whose `evidence_refs` cite the case's `text_element_id`. Output: `pipeline_findings.jsonl`.
- **`matcher.py`** — Builds rich embedding input from `claim + subject + outcome + relation_type + direction + category`, computes pairwise cosine similarity, greedy one-to-one matching above a threshold (default 0.55). Counts field mismatches; computes strict F1 with partial TP support.
- **`evaluate.py`** — Top-level entry point: read `source_cases.jsonl`, run exporter, run matcher, write metrics + inspection reports. Supports filtering by split.
- **`inspect.py`** — Generates human-readable Markdown case reports + CSVs (matched pairs, field mismatches, unmatched silver, unmatched pipeline). Timestamps every artifact.
- **`sweep.py`** — Sweeps the matching threshold from 0.40 to 0.80; embeddings are cached. Outputs metrics-per-threshold CSV + Markdown report. Dev split only to avoid test contamination.
- **`split.py`** — `assign_split(case_id, seed) → "dev" | "test"` via `MD5(seed:case_id)`. `filter_by_split()` filters lists by split.
- **`schemas.py`** — Pydantic models for inputs/outputs/results: `SourceCase`, `SilverFinding`, `SilverCaseResult`, `PipelineFinding`, `PipelineCaseOutput`, `MatchResult`, `MatchedPair`, `FieldMismatch`, `EvalMetrics`.
- **`jsonl_utils.py`** — `read_jsonl` (generator), `write_jsonl` (batch), `append_jsonl` (single row), `exists_in_jsonl` (key lookup). Pydantic-validated.
- **`tests/test_matcher.py`** — Unit tests for the matcher (greedy assignment, threshold behaviour, partial TP).

---

## `scripts/` — utilities, runners, inspectors

A grab-bag of one-off scripts: production runners, debugging inspectors, evaluation helpers, and legacy code. The currently-active modular pipeline lives in `pipeline/`; many scripts here predate it.

| File | Role |
|---|---|
| `latest_ingest.py` | **Legacy** monolithic PDF→DB pipeline (1407 LOC). Superseded by `pipeline/stages/pdf_text_extraction/runner.py`. |
| `combined_pipeline.py` | Earlier Docling+TATR combined pipeline (733 LOC). |
| `merged_pipeline.py` | Larger experimental pipeline combining layout + NER + summarization (982 LOC). |
| `viewer_server.py` | Flask web inspector reading from PostgreSQL. |
| `inspect_pipeline_output.py` | HTML report generator from pipeline output (1297 LOC). |
| `inspect_phase123_pipeline.py` | Inspector for NER + NLI + entity grounding stages. |
| `inspect_map_normalize.py` | Show how MAP findings transform through normalization. |
| `inspect_normalize_group.py` | Visualize normalization → grouping. |
| `summarize_paper.py` | Wrapper to summarize a single paper end-to-end. |
| `run_paper.py` | Single-paper runner using OpenAI models (GPT-4o). |
| `run_paper_single_model.py` | Run summarization Phases 1–6 with a single LLM (Gemini Flash Lite via Vertex). |
| `run_single_doc.py` | Generic single-document runner. |
| `two_pass_extract.py` | Two-pass extraction experiment. |
| `compare_docling_options.py` | Compare Docling configurations on sample PDFs. |
| `compare_policies.py` | Compare routing policies side-by-side. |
| `compare_prefilter.py` | Compare pre-filter strategies for text element selection. |
| `select_policy.py` | Policy selection utility. |
| `eval_policy.py` | Evaluate a policy on a benchmark dataset. |
| `fit_routing_threshold.py` | Fit deferral-score threshold for the MAP-stage router. |
| `label_routing_records.py` | Interactive labeller for routing decisions. |
| `process_pdffigures_results.py` | Post-process PDFFigures2 JSON output. |
| `visualize_docling_full.py` | Visualization helper for table reconstruction. |
| `create_tui_gin_index.py` | Create GIN index for UMLS TUI text search in Postgres. |
| `copy_relevant_files.py` | File-copy utility for sharing/analysis. |
| `test_model_connections.py` | Smoke-test connectivity to Anthropic / OpenAI / Vertex. |
| `docling_files/mask_tables.py` | PDF masking helper used by some legacy scripts. |
| `docling_files/__init__.py` | Package marker. |

### Highlights

- **`latest_ingest.py`** — Legacy unified pipeline. Downloads, extracts with Docling, masks, re-extracts, reconstructs tables from captions, stitches text, detects refs, ingests to DB. Thread-safe via per-worker model instances. Kept around as a reference; **prefer `PipelineRunner`** for new work.
- **`viewer_server.py`** — Flask server exposing `/`, `/paper/<pmcid>`, `/paper/<pmcid>/<run_id>`, `/api/runs`, `/api/paper/<pmcid>/<run_id>`. Renders pipeline state from the DB with Jinja2 templates, including corpus relations and low-grounding-threshold highlighting.
- **`inspect_pipeline_output.py`** — Big (1297 LOC) HTML report generator. Renders findings, relations, detected concepts, field-confidence scores, and a corpus-level relation graph. Supports both DB and JSONL inputs.
- **`fit_routing_threshold.py`** — Reads `routing_records.jsonl` (with `keep_ok` labels), sweeps thresholds, writes `metrics.csv` + `summary.json` (recommended theta), and optionally a curve PNG. Optimizes max-false-accept against a min-recall floor.
- **`run_paper_single_model.py`** — Bypasses the ABC cascade by wiring the same model into all voter slots with `theta=0.0` and no escalation; used for cost-sensitive smoke tests.
- **`docling_files/mask_tables.py`** — PDF masking via PyMuPDF using Docling JSON bboxes. Reconstructs tables from raw elements, overlays white rectangles, categorizes elements as text vs maskable.

---

## `notebooks/` — standalone demo / workflow notebooks

Jupyter notebooks that are independent of the production pipeline runners. Useful for one-off exploration; not loaded by any production code.

| File | Role |
|---|---|
| `PDF_Processing_Pipeline.ipynb` | End-to-end demo: layout extraction → table reconstruction → masking → text extraction → DB ingestion. Imports from `parsers/layout_utils.py`. |

> Legacy LangChain-stack notebooks (`langchain_summarization.ipynb`, `test_pipeline_50_docs.ipynb`) live under [`langchain-summarization/`](#langchain-summarization--legacy-langchain-stack).

---

## `tests/` — pytest suite

Unit tests for the summarization pipeline + a couple of cross-cutting integration tests. Most tests are pure-Python and don't require the DB or a live LLM.

| File | Role |
|---|---|
| `test_corpus_relate.py` | Tests for `pipeline/stages/summarization/helpers/corpus_relate.py`. |
| `test_inspector.py` | Tests for the pipeline inspector script. |
| `summarization/conftest.py` | Pytest fixtures (sample data, fake DB connection). |
| `summarization/test_phase1_schema.py` | Phase-1 schema roundtrip + grounding filter (no LLM). |
| `summarization/test_phase1_nli.py` | NLI-based grounding score computation. |
| `summarization/test_phase2_normalize.py` | NormalizeStage entity normalization + dedup. |
| `summarization/test_phase3_group.py` | GroupStage bucketing and direction counting. |
| `summarization/test_phase_a_gate.py` | Gate / filter logic. |
| `summarization/test_map_chunking.py` | Sentence-to-chunk chunking. |
| `summarization/test_map_chunk_slicing.py` | Chunk-slice helpers (`start_chunk`, `limit_chunks`). |
| `summarization/test_llm_errors.py` | Retryable / non-retryable classification. |
| `summarization/agreement/test_embedding_scorer.py` | `EmbeddingScorer` similarity + polarity heuristics. |
| `summarization/agreement/test_semantic_agreement.py` | `SemanticAgreementScorer` deferral-score computation. |
| `summarization/routing/test_routing.py` | `MapOutputRouter` end-to-end gating. |
| `summarization/routing/test_policy.py` | Policy evaluation metrics. |
| `summarization/routing/test_routing_dataset.py` | RoutingDataset loading and metrics. |

---

## `langchain-summarization/` — legacy LangChain stack

Predecessor to `pipeline/stages/summarization/`. Kept for reference. Production code should not import from here.

| File | Role |
|---|---|
| `evaluator.py` | Resilient hallucination evaluator (DB ID provenance + numeric integrity + UMLS + Negex). |
| `langchain_summarization.ipynb` | Main demo notebook for the LangChain pipeline. |
| `test_pipeline_50_docs.ipynb` | Notebook test on a 50-doc sample. |
| `price-estimator/estimator.py` | LLM cost estimator (tiktoken). |
| `README.md` | LangChain-pipeline documentation. |
| `test_results_50_docs/` | Input data: 50 documents. |
| `summarization_results/` | Output: summaries + extracted rules. |

### Detailed entries

- **`evaluator.py`** — `EvaluationResult(is_valid: bool, medical_score: float, …)` with three resilient checks: (1) DB ID provenance for evidence refs, (2) regex-based numeric integrity, (3) negation-aware UMLS concept matching using scispaCy + Negex. Streams over 45M+ token runs without loading everything into memory. Threshold for valid: `medical_score ≥ 0.7`.
- **`price-estimator/estimator.py`** — Uses `tiktoken` to count tokens, applies per-model pricing tables, and estimates total cost before kicking off a long batch.

---

## `misc/` — repo introspection utilities

Small standalone scripts for understanding the codebase itself.

| File | Role |
|---|---|
| `line_counter.py` | Lines-of-code report by file extension. |
| `pdf_page_counter.py` | Count pages of un-ingested PDFs (excluding blacklist + already in DB). |
| `relevant_file_checker.py` | AST-based Python import dependency analyzer. |

### Detailed entries

- **`line_counter.py`** — Walks the repo, filters `.py` and `.sh` files, ignores `__pycache__` / `.venv` / similar, prints a table of file counts and line totals per extension.
- **`pdf_page_counter.py`** — Queries the DB for processed PMCIDs, loads `failed_pdfs_blacklist.json`, then counts pages of the remaining PDFs in `files/organized_pdfs/` with PyMuPDF. Sorts ascending so short papers go first; optional CSV output with full per-PDF details.
- **`relevant_file_checker.py`** — Parses Python ASTs to extract imports, builds a local-vs-external import graph, prints the import hierarchy, reverse dependencies, and external packages. Useful for understanding coupling before refactoring.

---

## `pdffigures2/` — Allen AI evaluation harness (vendored)

Vendored copy of Allen AI's PDFFigures2 evaluation tools, used to compare our figure/table detection against their published benchmarks. Mostly thin wrappers over the underlying Java tool.

| File | Role |
|---|---|
| `evaluation/extractors.py` | Subprocess wrapper for the PDFFigures2 JAR. |
| `evaluation/pdffigures_utils.py` | `Figure` / `Table` / `Citation` dataclasses + utilities. |
| `evaluation/parse_evaluation.py` | Parse PDFFigures2 evaluation JSON. |
| `evaluation/build_evaluation.py` | Build evaluation dataset from PDFs. |
| `evaluation/build_section_eval.py` | Build section-specific evaluation. |
| `evaluation/compare_evaluation.py` | Compare PDFFigures2 output against ground truth. |
| `evaluation/section_extractors.py` | Extract figure/table refs from section text. |
| `evaluation/time_extractor.py` | Extract temporal expressions. |
| `evaluation/print_dataset_stats.py` | Print stats for the evaluation dataset. |
| `evaluation/download_from_urls.py` | Download PDFs from URLs. |
| `evaluation/datasets/datasets.py` | Dataset loading and management. |
| `evaluation/datasets/build_dataset_images.py` | Extract figure/table images. |
| `evaluation/datasets/test_datasets.py` | Tests for dataset loading. |
| `evaluation/datasets/visualize_annotations.py` | Visualize annotations. |
| `evaluation/__init__.py` | Package marker. |
| `evaluation/datasets/__init__.py` | Package marker. |

---

## `.agents/` — Claude Code skills

Custom Claude Code skills for this project (referenced via `skills-lock.json`). Not part of the production pipeline.

| File | Role |
|---|---|
| `skills/compress/scripts/compress.py` | Core compression logic (natural language → caveman format). |
| `skills/compress/scripts/cli.py` | CLI entry point. |
| `skills/compress/scripts/__main__.py` | Module-as-script entry. |
| `skills/compress/scripts/benchmark.py` | Compression-ratio benchmarks. |
| `skills/compress/scripts/validate.py` | Validate semantic equivalence after compression. |
| `skills/compress/scripts/detect.py` | Detect files eligible for compression. |
| `skills/compress/scripts/__init__.py` | Package marker. |

---

## `READMEs/` — specialized documentation

Topic-focused READMEs that are too large to belong inline.

| File | Role |
|---|---|
| `MEDICAL_GRADE_PARSING.md` | Architecture of the intelligent ensemble parser with page-level routing; M1 optimizations. |
| `PDF_EXTRACTION_GUIDE.md` | Integration guide for Marker / Nougat / PDFFigures2; install instructions. |
| `PYMUPDF_LAYOUT_INTEGRATION.md` | Notes on integrating PyMuPDF layout signals. |

---

## Runtime output directories (not in repo)

Generated at runtime by the pipeline; gitignored.

| Path | Contents |
|---|---|
| `out/docling_full/` | Cached full Docling layout JSON (one per PDF). |
| `out/docling_masked/` | Layout JSON from masked PDFs. |
| `out/masked_pdfs/` | PDFs with detected regions whited out. |
| `out/text/` | Final assembled hierarchical text per paper. |
| `out/text_raw/` | Pre-assembly raw element dumps. |
| `out/figures/`, `out/tables/` | Cropped figure / table images. |
| `out/json/` | Per-paper media metadata JSON. |
| `out/visualization/` | Annotated PDFs for auditing detections. |
| `out/run_metadata/` | Per-run timing and processing stats. |
| `out/failed_pdfs_blacklist.json` | Persisted blacklist (failed / too-large PDFs). |
| `langchain-summarization/summarization_results/` | Per-paper summarization JSON + corpus_relations.json. |
| `eval/out/` | Eval-pipeline outputs (mirror of `out/` but isolated). |

---

## Cross-cutting reference

### Coordinate systems

| System | Origin | Used by |
|---|---|---|
| Docling PDF coords | y=0 at **bottom** of page; `y1=top, y2=bottom` | `LayoutElement.bbox`, `BoundingBox` |
| fitz / screen coords | y=0 at **top** of page; `rect.y0 < rect.y1` | PyMuPDF masking + cropping |

Convert with `BoundingBox.to_fitz_rect(page_height)` and `BoundingBox.from_fitz_rect(rect, page_height)`.

### `unique_path` format

`{PMCID}/{section_hierarchy}/{position_in_section}` — e.g. `PMC1448691/Methods > 2.1 Staining/0`. Used as the `TextElement.unique_path` unique key.

### Hierarchical query example

```python
# All paragraphs anywhere under a "Methods" section
session.query(TextElement).filter(TextElement.path_list.contains(['Methods']))
```

### Provenance chain

For the LLM-summarization pipeline:

```
FinalRule → CanonicalRule → FindingGroup → NormalFinding → Finding (chunk_id) → evidence "S{i}|{pmcid}|{te_id}" → TextElement
```

Every step persists, so any rule can be traced back to the exact source paragraph.

---

*Generated 2026-05-03. Update when adding/renaming significant files; this file is a reference, not a rolling changelog.*
