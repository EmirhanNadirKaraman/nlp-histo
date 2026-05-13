# HOW_TO_RUN.md

End-to-end commands to reproduce the thesis result. Every block is meant to be
copy-paste runnable from the repo root. Update this file whenever a script's
invocation, a config default, or a pipeline stage changes.

---

## 0. Environment

```bash
# Python deps
pip install -r requirements.txt

# Postgres (local default — edit .env to point elsewhere)
createdb -U postgres nlp_histo
cp .env.example .env
# fill in DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# Schema
python database/setup_db.py            # create tables
python database/setup_db.py --check    # inspect, no changes
python database/setup_db.py --drop     # destructive: wipe + recreate
```

Optional kill-switches (useful on low-RAM machines):

```bash
export NLP_HISTO_DISABLE_UMLS=1            # skip scispaCy + UMLS entirely
export NLP_HISTO_SKIP_UMLS_ENRICHMENT=1    # load scispaCy but skip CUI enrichment
```

---

## 1. Acquire papers

```bash
cd file-selector
python file_downloader.py     # needs target_pmc_ids.txt
python tarball_extractor.py
python pdf_organizer.py
cd ..
```

Produces `files/organized_pdfs/` and `files/organized_xmls/`.

---

## 2. PDF → text extraction pipeline

```bash
python -c "
from pipeline.stages.pdf_text_extraction import PipelineRunner, PipelineConfig
cfg = PipelineConfig()       # two_pass.enabled defaults to True after May-2026 fix
runner = PipelineRunner(cfg)
runner.run_batch('files/organized_pdfs')
"
```

Outputs land under `out/`:

| Directory             | Contents                                  |
|-----------------------|-------------------------------------------|
| `out/docling_full/`   | Full Docling layout JSON (cached)         |
| `out/docling_masked/` | Docling layout JSON from masked PDFs      |
| `out/masked_pdfs/`    | PDFs with figures/tables whited out       |
| `out/text/`           | Hierarchical text                         |
| `out/figures/`        | Cropped figure images                     |
| `out/tables/`         | Cropped table images                      |
| `out/visualization/`  | Annotated debug PDFs                      |
| `out/json/`           | Detection metadata                        |
| `out/run_metadata/`   | Per-paper timing/processing stats         |
| `out/failed_pdfs_blacklist.json` | Skip list (thread-safe)        |

---

## 3. Summarisation pipeline

```bash
# Single paper (driven by scripts/run_paper.py)
python scripts/run_paper.py --pmcid PMC7150310_main

# Cheap-tier corpus run (matches configs/paper_selection/runA.yaml)
python scripts/run_paper.py --config configs/paper_selection/runA.yaml
```

Outputs land in `out/summaries/runs/<run_id>/`:

```
canonicalize/<pmcid>/canonical_rules.jsonl
group/<pmcid>/groups.jsonl
map/<pmcid>/findings.jsonl
normalize/<pmcid>/normal_findings.jsonl
relate/<pmcid>/relations.jsonl
resolve/<pmcid>/final_rules.jsonl
cost/cost_report.json
logs/
manifest.json
```

The pooled cross-paper relations live at `out/summaries/corpus_relations.json`.

---

## 4. Cost estimation

```bash
python scripts/estimate_selection_cost.py
python scripts/estimate_pipeline_cost_percentiles.py
```

---

## 5. Tests

```bash
# Full summarisation suite (~3 min)
python -m pytest tests/summarization/ -q

# Focused — group/canonicalize hash invariants
python -m pytest tests/summarization/test_phase3_group.py tests/summarization/test_demographics.py -q
```

---

## 6. Thesis demos (see [`THESIS.md`](THESIS.md))

```bash
# Synthetic Tr=3 / opacity=0 PDF — confirms R1 catches ghost text
python scripts/verify_ghost_text_detection.py

# Real-paper corpus scan (N defaults to 8; pass a positional integer to widen)
python scripts/scan_ghost_text_real_papers.py 12

# Before/after demos that back up THESIS.md (writes PNGs + JSON)
python scripts/thesis_demo_ghost_text.py
```

Artifacts land in `out/thesis_demo/ghost_text/`.

---

## 7. Inspection helpers

```bash
python scripts/inspect_pipeline_output.py --pmcid PMC7150310_main
python scripts/inspect_normalize_group.py --pmcid PMC7150310_main
python scripts/inspect_phase123_pipeline.py --pmcid PMC7150310_main
python scripts/test_map_schema.py
```

---

## 8. Database housekeeping

```bash
# Count text elements for a paper
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "
SELECT COUNT(*) AS total, COUNT(DISTINCT text_content) AS distinct
FROM text_elements
WHERE document_id = (SELECT id FROM documents WHERE pmcid = 'PMC7150310_main');"

# Find a paper's canonical rules
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "
SELECT canonical_id, subject_entity, outcome_entity, relation_type, predicate_text
FROM sum_canonical_rules
WHERE pmcid LIKE 'PMC7150310%';"
```
