# Bug catalogue — nlp-histo

Per-bug write-ups with status, evidence, diagnosis, fix, and verification.
Carry-forward work items live in [`THESIS.md`](THESIS.md#todos); permanent
design calls live in [`THESIS.md`](THESIS.md#decisions-log).

> **How to use this file:**
> * [Bugs (catalogue)](#bugs-catalogue) — every substantive defect, with status.
> * Detailed write-ups follow as `## Bug N — …` and `## Topic — …` sections.
>
> Add a new entry the same day the issue is discovered. New bug → row in the
> catalogue + detail section. Bump the ID monotonically (`B-017`, `B-018`, …).
> Never delete entries — flip `Status` to `Fixed` / `Won't fix` / `Superseded`
> so the history survives.

---

## Bugs (catalogue)

| ID | Status | Severity | Surface | One-line summary | Detail |
|----|--------|----------|---------|------------------|--------|
| B-001 | Fixed (2026-05-13) | High | Summarisation, corpus relate | `canonical_id` collided across papers because `group_id` did not include `pmcid`; same-rule self-pairs appeared as bogus "intra_paper" SUPPORT relations. | [Bug 1](#bug-1--duplicate-intra-paper-relations-produced-by-canonical_id-collisions) |
| B-002 | Mitigated (2026-05-13) | Medium | PDF extraction, Docling layout | Docling emits phantom layout elements — real text content at a bbox that does not render (often in the header zone of the wrong page). `ContextAwareStitcher` had been masking this; now caught upstream by `TwoPassConfig.enabled=True` + Rule R1/R3. | [Bug 2](#bug-2--docling-phantom-layout-elements) |
| B-003 | Mitigated (2026-05-13) | Low (latent) | PDF extraction, color signal | Rule R-color (`max_white_char_fraction < 1.0`) treats *any* near-white span as ghost text, ignoring the rendered background. Would have produced false positives on white-on-coloured headers if applied to TEXT/CAPTION types. Currently dormant because `SECTION_HEADER` is in `NodeScorer._ALWAYS_KEEP`. Disabled by default now (`max_white_char_fraction=1.0`). | [Bug 3](#bug-3--r-color-white-text-false-positive-latent) |
| B-004 | Observed | Low | PDF extraction, Docling glyph fallback | CID-only PDFs surface in Docling as `GLYPH<…>` / `/gid00001` text strings. R1 drops them because their bboxes have no ink (Docling didn't decode the font). No production impact, but worth keeping an eye on for corpora with subset-only fonts. | [Bug 4](#bug-4--cid-glyph-fallback-strings) |
| B-005 | Mitigated (2026-05-14) | High | Summarisation, batch runner | `BatchSummarizationRunner.finalize()` was missing six features the sync runner had: (1) `_replace_verbatim_from_db` — grounding NLI ran against LLM paraphrases instead of source text; (2) stable `compute_finding_id`; (3) DB persistence to `sum_*` tables; (4) `corpus_relate_incremental`; (5) `rejection_summary` build + persist; (6) NER + UMLS linking. Since `scripts/run_paper.py` defaults to batch mode, every batched production result between commit `5c59c3e` (2026-04-27) and the 05-14 backport was grounded against paraphrased text. | [Bug 5](#bug-5--batch-runner-missing-sync-parity-features) |
| B-006 | Fixed (2026-05-14) | Medium | Summarisation, RELATE / RESOLVE | `RelationTypeLabel.SCOPE_QUALIFY` plumbing (the enum, the RESOLVE filter, the RELATE info-log column) was wired end-to-end but no `_classify_pair` branch ever emitted it. Stripped: enum value removed, RESOLVE `scope_qualifies` list-comp dropped, RELATE log no longer prints the column. `FinalRule.scope_qualify_count` and the DB column retained as hard-zero fields so existing readers (HTML inspector, downstream consumers) don't break. | [Bug 6](#bug-6--scope_qualify-plumbing-is-dead) |
| B-007 | Fixed (2026-05-14) | Medium | Summarisation, sync runner result cache | `SummarizationRunner._load_result` returned cached `{pmcid}.json` unconditionally and `_save_result` never stamped a hash. Now mirrors batch: a `_pipeline_config_hash()` helper composes cascade signature + thresholds + model identifiers + schema/prompt versions + `enable_router` state; load compares stored vs current hash and re-runs on mismatch; save stamps the hash. Manifest builder reuses the same helper to avoid drift. | [Bug 7](#bug-7--sync-runner-cached-result-load-ignores-pipeline_config_hash) |
| B-008 | Observed | Low | Summarisation, sync runner batch reporting | `SummarizationRunner.process_batch` reports `n_skip = len(results) - n_ok - n_err` but `_load_result` returns `status="success"` for cached dicts, so cached papers count in `n_ok` and `n_skip` is structurally 0. Cosmetic but the log message is misleading. | [Bug 8](#bug-8--process_batch-skip-counter-is-structurally-zero) |
| B-009 | Observed | Low | Summarisation, sync runner instance state | `SummarizationRunner` keeps per-paper state in instance dicts (`_relate_raw_pairs`, `_relate_skipped_pairs`, `_normal_findings`, `_scored_map_findings`, `_canonical_rules`, `_relations`, `_finding_groups`, `_final_rules`). Inside `process_batch` they accumulate across papers and are never cleared. Memory grows O(papers × avg eligible pairs). | [Bug 9](#bug-9--sync-runner-instance-dicts-leak-across-papers) |
| B-010 | Observed | Medium | PDF extraction, artifact filter | `components/artifact_filter.py:59` rebuilds `List[LayoutElement]` after filtering via `[el for i, el in enumerate(elements) if element_dicts[i] in filtered_dicts]` — list-`__contains__` over dicts. O(N²); and the moment `filter_artifacts` ever mutates a kept dict (e.g. a future ligature normalisation), the post-filter dict no longer `==`'s the pre-filter dict and the corresponding `LayoutElement` is silently dropped. | [Bug 10](#bug-10--artifact_filter-rebuild-uses-dict-equality-instead-of-identity) |
| B-011 | Observed | Low | PDF extraction, `ModelRegistry` | `resources.py` `ModelRegistry.docling_converter` ignores `DoclingConfig.images_scale`, `accelerator_device`, `ocr_engine`, `force_full_page_ocr`; hard-codes `images_scale=2.0` and never builds `AcceleratorOptions`. Currently unused by `PipelineRunner` (each component constructs its own converter), but exported as public API — a caller who flips a non-default `DoclingConfig` and uses `ModelRegistry` silently gets CPU + scale 2.0. | [Bug 11](#bug-11--modelregistrydocling_converter-ignores-doclingconfig) |
| B-012 | Observed | Low | PDF extraction, two-pass extractor | `components/two_pass_extractor.py:382-398` header/footer strip construction mixes Docling y-coords (`docling_y1=page_h`) and fitz coords (`fitz_header_bottom`) on adjacent lines. Today only a `docling_y1 > docling_y2` comparison guards against a sign-flip if those names ever get muddled. Clarity issue today, latent bug surface for the next refactor. | [Bug 12](#bug-12--two_pass_extractor-header-strip-mixes-coordinate-systems) |
| B-013 | Fixed (2026-05-14) | Low | Inspector batch index, sort handler | `scripts/templates/pipeline_batch_index.html.jinja2:276` read `dataset.nilBa` instead of `dataset.nliBa`. `parseFloat(undefined) → NaN → 0`, so clicking the "NLI B→A" column compared zeros and produced no reorder. Fixed by correcting the typo. | [Bug 13](#bug-13--inspector-nli-ba-sort-typo) |
| B-014 | Fixed (2026-05-14) | Low (latent) | Inspector batch index, badge style | `pipeline_batch_index.html.jinja2:194` renders SCOPE_QUALIFY relations with class `badge-blue`, but the stylesheet only defined `badge-green/red/orange/gray/cyan`. Badge rendered unstyled. Currently dormant because B-006 means SCOPE_QUALIFY is never emitted; would surface the moment B-006 is fixed. Added `.badge-blue` rule. | [Bug 14](#bug-14--inspector-badge-blue-class-missing) |
| B-015 | Fixed (2026-05-14) | Medium | Summarisation, MAP enum coercion | Raw LLM-emitted `relation_type` / `direction` / `category` values were coerced (or alias-repaired) to enum members and the originals were dropped from the row — only landed in `logs/enum_observations.jsonl` with no FK back to the finding. Downstream stages saw only `unclear` / coerced values. Fixed by capturing raw values in a `model_validator(mode="wrap")` on `Finding`, persisting them to new `sum_map_findings.raw_{relation_type,direction,category}` columns (Alembic `0011`). | [Bug 15](#bug-15--raw-llm-enum-values-lost-on-coercion) |
| B-016 | Fixed (2026-05-14) | Low | Summarisation, MAP prompt + schema | `category` enum was `"demographics"` (plural) while `relation_type` enum was `"demographic"` (singular) — same concept, two spellings, requiring an alias map and prompt warning. `Rule.confidence` Literal was `"High"|"Medium"|"Low"` while MAP `Finding.confidence` was lowercase. Aligned both to `"demographic"` (singular, consistent with sibling category labels) and lowercase confidence; inverted `_CATEGORY_ALIASES` to repair legacy `"demographics"`; bumped `MAP_PROMPT_VERSION` to `map_prompt_v2_singular_demographic`. | [Bug 16](#bug-16--demographic-spelling-and-confidence-casing-divergence) |

Add new rows here when you discover something. Bump the ID monotonically (`B-017`, `B-018`, …). Put the long write-up in a new `## Bug N — …` section below.

---

## Bug 1 — Duplicate "intra-paper" relations produced by `canonical_id` collisions

### Symptom

The cross-paper relations table contained rows whose `pmcid_a == pmcid_b` and
whose `predicate_a == predicate_b` (NLI scores 1.0 / 1.0). Example for
`PMC7150310_main`:

| Scope      | Type    | PMCID A           | PMCID B           | Predicate A                                              | Predicate B                                              |
|------------|---------|-------------------|-------------------|----------------------------------------------------------|----------------------------------------------------------|
| intra_paper | SUPPORT | PMC7150310_main   | PMC7150310_main   | Rickets → calcitriol (vitamin D) deficiency             | Rickets → calcitriol (vitamin D) deficiency             |
| intra_paper | SUPPORT | PMC7150310_main   | PMC7150310_main   | Beri-beri → thiamine (vitamin B1) deficiency            | Beri-beri → thiamine (vitamin B1) deficiency            |
| intra_paper | SUPPORT | PMC7150310_main   | PMC7150310_main   | Scurvy → ascorbic acid (vitamin C) deficiency           | Scurvy → ascorbic acid (vitamin C) deficiency           |
| intra_paper | SUPPORT | PMC7150310_main   | PMC7150310_main   | Pernicious anemia → cobalamin (vitamin B12) deficiency  | Pernicious anemia → cobalamin (vitamin B12) deficiency  |

The first hypothesis — that the PDF extractor had emitted the same sentence
twice (e.g. a publisher "ghost text" layer duplicating body content) — turned
out to be wrong.

### Diagnosis

1. `text_elements` for `PMC7150310_main` were verified clean:
   * 61 rows, 61 distinct `text_content` values
   * 0 near-duplicates after whitespace/case normalisation
   * The five vitamin-deficiency findings all trace back to **one** sentence
     in **one** text element under `Natural Product Chemistry and the Rise
     of Clinical Laboratories`.
2. The summarisation artifacts on disk were inspected:
   * `out/summaries/runs/runA_cheap_main/canonicalize/PMC7150310_main/canonical_rules.jsonl`
     contained 196 rules with 196 distinct `canonical_id`s.
   * `out/summaries/corpus_relations.json` contained 19 rows where
     `rule_id_a == rule_id_b` for this paper.
3. Cross-paper comparison of canonical IDs:
   * `PMC7150046_main` produced 171 rules; `PMC7150310_main` produced 196.
   * **22 `canonical_id`s collided across the two papers**, including
     `CR_3193d882_positive` ("Rickets → calcitriol").
4. Root cause traced to `pipeline/stages/summarization/current_stages/group_stage.py:_group_id`:

   ```python
   def _group_id(subject, outcome, relation_type, category="",
                 subject_cui=None, outcome_cui=None) -> str:
       subj_key = subject_cui if subject_cui else subject
       out_key  = outcome_cui if outcome_cui else outcome
       return f"GRP_{_sha8(subj_key)}_{_sha8(out_key)}_{relation_type}_{_sha8(category)}"
   ```

   Two different papers that produced the same `(subject, outcome,
   relation_type, category)` got the same `group_id` → same `canonical_id`
   (`canonical_id = CR_{sha8(group_id)}_{direction}`).
5. In `corpus_relate.py`, the rule pool concatenates rules from all papers,
   so the colliding IDs landed at two distinct list indices. `RelateStage`
   used `itertools.combinations(range(len(rules)), 2)`, which guarantees
   `i != j` but *not* `rules[i].canonical_id != rules[j].canonical_id`,
   so the collisions were paired against themselves. The final enrichment
   step mapped the canonical_id back to a PMCID through a single-writer-wins
   dict, so both sides of each pair displayed the same `pmcid`, producing
   the misleading "intra_paper" rows above.

### Fix

[`pipeline/stages/summarization/current_stages/group_stage.py:57`](../pipeline/stages/summarization/current_stages/group_stage.py#L57)
— added `pmcid` to the hash input:

```python
def _group_id(subject, outcome, relation_type, category="",
              subject_cui=None, outcome_cui=None,
              pmcid: str = "") -> str:
    subj_key = subject_cui if subject_cui else subject
    out_key  = outcome_cui if outcome_cui else outcome
    return (
        f"GRP_{_sha8(pmcid)}_{_sha8(subj_key)}_{_sha8(out_key)}"
        f"_{relation_type}_{_sha8(category)}"
    )
```

`canonical_id` inherits the change because it derives from `group_id`:
`CR_{_sha8(group_id)}_{direction}`. Cross-paper matching still works — it is
performed by the
[`_should_compare_cross_paper`](../pipeline/stages/summarization/helpers/corpus_relate.py)
gate (which keys on CUIs or normalised entity strings), not on `canonical_id`
equality.

### Verification

* Updated tests in `tests/summarization/test_phase3_group.py` (added
  `test_group_id_differs_across_pmcids`) and
  `tests/summarization/test_demographics.py` (updated existing `_group_id`
  call sites). 40/40 pass.
* Full summarisation test suite: **437 passed** in 174 s.
* On-disk artifacts under `out/summaries/` still hold the colliding IDs;
  regenerate them by re-running summarisation on the affected papers.

---

## Topic — Ghost-text detection: empirical verification and policy fix

The duplicate-relations investigation surfaced the question of how the
extractor handles "ghost text" — selectable but unrendered text layers
embedded in publisher PDFs (e.g. accessibility duplicates, watermarks, white
text on white background). Two policy fixes followed, plus three bug entries
in the catalogue ([B-002](#bug-2--docling-phantom-layout-elements),
[B-003](#bug-3--r-color-white-text-false-positive-latent),
[B-004](#bug-4--cid-glyph-fallback-strings)).

### Background

The pipeline already implements three ghost-text-detection rules in
`pipeline/stages/pdf_text_extraction/components/node_scorer.py`:

* **R1 (pixel-render)** — render the element's bbox at 150 dpi; if mean
  luminance ≥ 245 *and* dark-pixel fraction ≤ 0.02, mark `visually_blank=True`
  and drop. Operates on the actual ink that reaches the page.
* **R-color** — read `page.get_text("dict")` and tally span colors;
  drop elements whose near-white-character fraction exceeds
  `max_white_char_fraction`.
* **R3 (dense-text)** — drop elements where `len(text) / bbox_height` exceeds
  `max_chars_per_bbox_pt`. Hidden text layers tend to cram many characters
  into a sliver of vertical space.

The two-pass extractor that *invokes* these rules is gated by
`TwoPassConfig.enabled`.

### 2.1 — Synthetic verification (does R1 catch Tr=3 / opacity=0?)

Built a one-page PDF with three text rows:

| Row | Render style                              |
|-----|-------------------------------------------|
| A   | Visible baseline (Tr=0)                   |
| B   | PDF render-mode 3 (`3 Tr`)                |
| C   | ExtGState fill_opacity `ca=0`             |

Output from
[`scripts/verify_ghost_text_detection.py`](../scripts/verify_ghost_text_detection.py):

```
row            visually_blank  brightness  dark_frac  inv_char_frac  render_skipped
visible        False           246.53      0.0351     0.0            False
tr3            True            255.0       0.0        0.0            False
opacity_0      True            255.0       0.0        0.0            False

PASS: pixel-render path catches both Tr=3 and fill_opacity=0 text.
```

Both ghost variants reduce to "no ink in the rendered bbox" — exactly what
R1 measures. No content-stream parser (e.g. `pdfminer.six` render-mode
extraction) was needed; adding one would have been duplicated work.

### 2.2 — Corpus scan (12 papers, 3,131 text-bearing elements)

`scripts/scan_ghost_text_real_papers.py` evaluated every TEXT-typed Docling
element against the rules. Aggregate:

| Signal                                         | Count | Rate  |
|------------------------------------------------|------:|------:|
| `visually_blank = True` (R1 fires)             | 20    | 0.64% |
| `invisible_char_fraction > 0.5` (R-color fires) | 18    | 0.57% |

Two patterns dominated:

**Pattern A — Docling phantom layout elements (`PMC10047158`, 18 hits).**
Docling occasionally emits a layout element with real text content but a
bbox that points to an empty area of a *different* page. Example:

| Field            | Value                                                                                              |
|------------------|----------------------------------------------------------------------------------------------------|
| Reported page    | 2                                                                                                  |
| Reported bbox    | `(168.2, 809.4) – (563.6, 799.3)` — i.e. 10 pt tall, sitting in the page header zone (≈ y=32 from top) |
| Reported text    | `large, centrally-located, occasionally bilobated nuclei with conspicuous nucleoli...`             |
| Pixel evidence   | `pixel_brightness_mean=255.0`, `dark_pixel_fraction=0.0` → `visually_blank=True`                    |
| `fitz.get_text('words', clip=…)` in that rect | `0` words found                                                                          |
| Actual location of the text | Page 1, inside the main body paragraph                                                  |

The downstream `ContextAwareStitcher` happens to merge these phantoms back
into legitimate paragraphs in most cases — verified by querying
`text_elements`:

```
unique_path                                            | preview
PMC10047158_dermatopathology-10-00017/2. Case Report/1 | Histological examination of routinely-stained sections showed
                                                       | a well-circumscribed exo-endophytic de…
```

— only one DB row holds "large, centrally" rather than two. But the stitcher
is incidental cleanup; future Docling layout regressions could surface
extra rows.

**Pattern B — White text on coloured banners (`PMC7158325`, 18 hits).**
"Key Points" SECTION_HEADER elements report `span.color = 0xFFFFFF` (pure
white) → `invisible_char_fraction = 1.0`. But the rendered bbox is full
of ink: `dark_pixel_fraction ≈ 0.73`, `pixel_brightness_mean ≈ 136`.
These are legitimate headers printed in white type on a coloured banner.
R-color would label them ghost text; R1 correctly keeps them.

This established that **R1 is the trustworthy signal**; R-color produces
false positives whenever a publisher inverts header type colours.

### 2.3 — Policy decisions

Two defaults changed in
[`pipeline/stages/pdf_text_extraction/config.py`](../pipeline/stages/pdf_text_extraction/config.py):

| Setting                         | Before | After | Reason                                                                                                        |
|---------------------------------|--------|-------|---------------------------------------------------------------------------------------------------------------|
| `TwoPassConfig.enabled`         | `False` | `True` | R1 catches Docling phantoms upstream; the stitcher is no longer the only line of defence.                  |
| `TwoPassConfig.max_white_char_fraction` | `0.5` | `1.0` | Disables R-color by default. R1 is the source of truth; R-color produced false positives on inverted headers. |

### 2.4 — Before/after demonstrations

[`scripts/thesis_demo_ghost_text.py`](../scripts/thesis_demo_ghost_text.py)
runs both demos and writes JSON + PNG artifacts under
`out/thesis_demo/ghost_text/`.

#### Demo 1 — phantom passes through old policy, dropped by new policy

`PMC10047158_dermatopathology-10-00017`, page 2 phantom element.

Old policy (`TwoPassConfig.enabled=False`): NodeScorer never runs.

| Policy | Total Docling text elements | Kept | Dropped | Phantom kept/dropped |
|--------|-----------------------------:|-----:|---------:|----------------------|
| OLD    | 129                          | 129  | 0        | **kept** (it would survive Docling layout output) |
| NEW    | 129                          | 82   | 47       | **dropped** by R3 ("hidden text layer — 181 chars in 10.2pt bbox = 17.8 chars/pt, in header zone"); R1 would also fire because the bbox is `visually_blank=True`. |

Rendered crop of the phantom region (page 2, header strip outlined in red —
note that the visible text in that strip is the journal banner
"Dermatopathology", not the body sentence Docling reported):

![Phantom bbox on page 2 of PMC10047158](../out/thesis_demo/ghost_text/demo1_phantom_false_negative_p2.png)

#### Demo 2 — legitimate white-on-dark header (latent R-color false positive)

`PMC7158325_main`, 17 occurrences of "Key Points".

| Policy | "Key Points" headers kept | "Key Points" headers dropped |
|--------|---------------------------:|-----------------------------:|
| OLD    | 17                         | 0                            |
| NEW    | 17                         | 0                            |

**Caveat:** SECTION_HEADER appears in `NodeScorer._ALWAYS_KEEP`, so R-color
never actually fires on these headers in production. The demo therefore
documents the *evidence* (span color = `0xFFFFFF`, `invisible_char_fraction
= 1.0` on a bbox whose rendered pixels are 73% dark ink) rather than an
old-vs-new behaviour difference. The false-positive risk it captures is
latent — it would have surfaced if R-color were applied to a TEXT/CAPTION
element printed in the same white-on-coloured style (e.g. a callout box
or a figure caption banner). Disabling R-color by default closes that door
before such an element appears.

Rendered crops of three "Key Points" banners (red outline = bbox; the white
ink reads as ≈ 73 % dark pixels against the coloured background):

![Key Points banner on page 3 of PMC7158325](../out/thesis_demo/ghost_text/demo2_key_points_false_positive_p3.png)

![Key Points banner on page 4 of PMC7158325](../out/thesis_demo/ghost_text/demo2_key_points_false_positive_p4.png)

![Key Points banner on page 27 of PMC7158325](../out/thesis_demo/ghost_text/demo2_key_points_false_positive_p27.png)

### Reproducibility

```bash
python scripts/verify_ghost_text_detection.py       # synthetic Tr=3 / opacity=0 PDF
python scripts/scan_ghost_text_real_papers.py 12    # 12-paper corpus sample
python scripts/thesis_demo_ghost_text.py            # writes demos + PNGs
```

JSON artifacts in `../out/thesis_demo/ghost_text/` carry the full evidence
table for each demo and can be re-rendered into the thesis figures without
re-running the scorers.

---

## Bug 2 — Docling phantom layout elements

**Status:** Mitigated (2026-05-13) · **Severity:** Medium · **Surface:**
PDF extraction, Docling layout.

**Symptom.** Some Docling-emitted layout elements carry legitimate body
text but a bbox that points to an empty region of the wrong page (typically
the page header zone of the page *after* the page where the text actually
lives).

**Evidence.** On `PMC10047158_dermatopathology-10-00017`, 18 such phantom
elements were found across one paper. See the full data table and rendered
crop in [Topic — Ghost-text detection §2.2 Pattern A](#22--corpus-scan-12-papers-3131-text-bearing-elements)
and [§2.4 Demo 1](#24--beforeafter-demonstrations).

**Mitigation.** Flipping `TwoPassConfig.enabled` to `True` (May-2026) makes
NodeScorer R1 (`visually_blank=True`) and R3 (`chars/pt` ratio) reject the
phantoms upstream rather than relying on `ContextAwareStitcher` to absorb
them downstream. Demo confirms 47 / 129 phantom-like elements dropped on
this paper post-fix; one phantom would have entered the DB without the fix
unless the stitcher caught it.

**Why "mitigated" not "fixed".** The root cause is Docling's layout
extraction emitting incorrect bboxes — we cannot fix Docling. We just catch
the symptom downstream. Tracked in TODO list for follow-up audit.

---

## Bug 3 — R-color white-text false positive (latent)

**Status:** Mitigated (2026-05-13) · **Severity:** Low (latent) · **Surface:**
PDF extraction, color signal.

**Symptom.** Rule R-color in `NodeScorer` rejects any element whose text
spans report near-white colour. On `PMC7158325_main`, 18 SECTION_HEADER
elements (`"Key Points"`) report span colour `0xFFFFFF` and trigger the
rule — even though the bbox renders as 73 % dark ink (white type on a
coloured banner).

**Why "latent".** `SECTION_HEADER` is in `NodeScorer._ALWAYS_KEEP`, so the
rule never actually fires in production. But the same publisher style
applied to a TEXT or CAPTION-typed element would have produced data loss.

**Mitigation.** `TwoPassConfig.max_white_char_fraction` default flipped
from `0.5` → `1.0`, which disables R-color. Rule R1 (pixel render) is the
trustworthy signal because it measures actual rendered ink, not span
metadata. Detail and rendered crops in [Topic §2.4 Demo 2](#24--beforeafter-demonstrations).

---

## Bug 4 — CID-glyph fallback strings

**Status:** Observed · **Severity:** Low · **Surface:** PDF extraction,
Docling glyph fallback.

**Symptom.** When a PDF embeds a font subset whose glyphs cannot be mapped
to Unicode, Docling falls back to placeholder strings: `GLYPH<0>GLYPH<20>…`,
`/gid00001`, etc. Found in `PMC11863827_main` and
`PMC7583592_dermatopathology-07-00003` during the 12-paper scan.

**Pipeline behaviour.** R1 correctly drops these because their bboxes contain
no rendered ink (the unmapped glyphs were never decoded into visible glyphs
either). No further action needed at the extraction stage.

**Open question for the thesis.** Worth a brief note that the corpus
contains a small number of font-subset PDFs that lose text content
end-to-end. If the rate grows on a different corpus, an OCR fallback
becomes worth costing out.

---

## Bug 5 — Batch runner missing sync-parity features

**Status:** Mitigated (2026-05-14) · **Severity:** High · **Surface:**
Summarisation, `BatchSummarizationRunner`.

### Symptom

`scripts/run_paper.py` defaults to batch mode. Batched production runs since
late April 2026 were quietly skipping six runner-level features that were
present on `SummarizationRunner.process()`:

1. **`_replace_verbatim_from_db`** — sync replaces LLM-paraphrased
   `verbatim_support` with the actual `TextElement.text_content` from the DB
   *before* grounding so NLI entailment scores against the real source. Batch
   never did this — grounding was scoring paraphrases against paraphrases.
2. **Stable `compute_finding_id`** — sync stamps every finding with a
   deterministic id `(pmcid, chunk_id, position, claim)` before grounding so
   downstream stages share a lineage key. Batch findings had no stable id.
3. **DB persistence** — sync writes `sum_map_findings`,
   `sum_normal_findings`, `sum_finding_groups`, `sum_canonical_rules`,
   `sum_relations`, `sum_final_rules`, `sum_rejection_summaries` via
   `pipeline_run_db_id`. Batch wrote JSON files but never touched the DB.
4. **`corpus_relate_incremental`** — sync runs cross-paper RELATE after
   CANONICALIZE. Batch did not. Corpus-relate tables only saw sync papers.
5. **`rejection_summary`** — sync builds a `RejectionSummary` recording
   grounding + non-groupable drops and persists it. Batch had no equivalent.
6. **NER + UMLS** — sync optionally runs `run_ner_on_db` per paper. Batch
   did not.

### Diagnosis

`git log` shows the gap is purely temporal — sync existed first
(commit `b3eecf3`), batch was bolted on later (`2aa71f6`). Six
sync-only commits (`5c59c3e`, `fb1b9af`, `a64fa9a`, …) added features
that never got backported to `batch/runner.py`. The most recent
multi-file commit `d002eb2` (2026-05-13) touched both runners but only
added voter-dedup, not the parity features. Stage-level fixes
(`group_stage.py`, `map_stage.py`, `relate_stage.py`,
`grounding_filter.py`) *did* propagate, since both runners import them —
so the gap is runner-level only.

### Fix

Copied the missing helpers verbatim from `SummarizationRunner` into
`BatchSummarizationRunner` (`_replace_verbatim_from_db`,
`_create_pipeline_run`, `_finish_pipeline_run`, `_clear_normalized_run_data`,
`_persist_map_findings`, `_persist_normal_findings`, `_persist_finding_groups`,
`_persist_canonical_rules`, `_persist_relations`, `_persist_final_rules`,
`_persist_rejection_summary`, `_corpus_relate_incremental`). Wired into
`finalize()` between the existing filesystem-artifact persistence and
REDUCE/RULES. `__init__` now accepts `db`, `force_rerun`, `run_ner` (defaults
`None`, `False`, `False` to preserve existing call-sites).

Also added result caching with `pipeline_config_hash` invalidation
(`_load_result` + `_save_result`) checked at the top of `submit()` so a
stale cache cannot waste L1/L2/L3 batch dollars; `BatchHandle` now carries
`pipeline_run_db_id` and a `cached_result_only` marker so a resumed run
keeps its DB pointer and a cache-hit short-circuits `finalize()`.

### Verification

`tests/summarization/test_batch_persistence.py` (4 tests) and
`tests/summarization/test_batch_voter_dedup.py` (10 tests) pass on the
modified runner. End-to-end DB integration verification is tracked as a
follow-up TODO until a real-DB run can be done.

### Why not refactor `SummarizationRunner` to share code?

Tempting but high-blast-radius. The sync runner is working code touched by
multiple recent commits; reshuffling its private methods into a shared
module while also changing the batch runner doubles the surface area of
this change. A follow-up TODO captures the dedup once both runners have
been verified to behave identically.

---

## Bug 6 — `SCOPE_QUALIFY` plumbing is dead

**Status:** Fixed (2026-05-14) · **Severity:** Medium · **Surface:**
Summarisation, RELATE → RESOLVE.

**Symptom.** `RelationTypeLabel.SCOPE_QUALIFY` was declared in
`models.py`, `FinalRule.scope_qualify_count` existed, the RESOLVE filter
at `current_stages/resolve_stage.py:120-123, 183` populated it, and the
RELATE log line at `current_stages/relate_stage.py:412-421` included a
SCOPE_QUALIFY count — yet `_classify_pair` had no branch that returned
`SCOPE_QUALIFY`. Mutual entailment → `SUPPORT`; mutual contradiction with
opposite polarity → `CONTRADICT`; everything else fell through to
`None`/`UNRELATED`. Result: `scope_qualify_count` was always 0, the log
misreported class counts, and any downstream consumer expecting an
asymmetric-entailment signal silently got none.

**Diagnosis.** Likely the asymmetric-entailment branch
(`ent_ab >= threshold xor ent_ba >= threshold → SCOPE_QUALIFY`) was
deleted at some point but the scaffolding around it wasn't.

**Fix.** Option 2 chosen — torn out the unused branch.

* `models.py`: removed `SCOPE_QUALIFY` from `RelationTypeLabel`; updated
  `Relation` docstring; defaulted `FinalRule.scope_qualify_count` to 0
  with a comment marking it dormant.
* `current_stages/relate_stage.py`: dropped `SCOPE_QUALIFY` column from
  the info log; updated module + ctor docstrings.
* `current_stages/resolve_stage.py`: removed the `scope_qualifies`
  list-comp; `scope_qualify_count=0` written verbatim.
* `current_stages/group_stage.py`, `PIPELINE.md`, `database/models.py`
  docstrings updated to drop the SCOPE_QUALIFY callouts.
* `final_rules.scope_qualify_count` DB column and the inspector template
  filter option retained so existing rows / pages still render. Drop via
  Alembic only if the asymmetric branch isn't reinstated.

**Verification.** `python -c "from pipeline.stages.summarization.models
import RelationTypeLabel; assert 'SCOPE_QUALIFY' not in [m.value for m in
RelationTypeLabel]"` passes. RELATE log message no longer mentions the
column. RESOLVE writes `scope_qualify_count=0` for every FinalRule.

---

## Bug 7 — Sync runner cached-result load ignores `pipeline_config_hash`

**Status:** Observed · **Severity:** Medium · **Surface:** Summarisation,
`SummarizationRunner`.

**Symptom.** `SummarizationRunner._load_result` at
[`runner.py:1568-1572`](../pipeline/stages/summarization/runner.py)
returns any on-disk `out/summaries/summaries/{pmcid}.json` unconditionally
(only `force_rerun` bypasses it), and `_save_result` never stamps a hash.
Run sync once with cascade profile A, then again with profile B → the
second run silently returns profile A's cached result. The `runs/{run_id}/`
artifact tree (which has its own hash in `manifest.json`) tells the right
story; the user-visible result JSON does not.

**Evidence.** Compare with
[`batch/runner.py:_load_result`](../pipeline/stages/summarization/batch/runner.py)
which (after B-005) computes the current `pipeline_config_hash` via
`compute_pipeline_config_hash(...)` and ignores the cached file on
mismatch. Sync runner has the helper imported (line 1080-1089 in
`_make_artifact_writer`) but doesn't use it for the result cache.

**Diagnosis.** The cache code at the top of `process()` predates
`pipeline_config_hash`; it was never updated when the hash landed in
`fb1b9af` / `a64fa9a`.

**Fix.** Mirror the batch implementation: in `_save_result` stamp
`result["pipeline_config_hash"] = self._pipeline_config_hash()`; in
`_load_result` recompute the current hash and return `None` on mismatch.
Same five lines as the batch helper. Open question: should the sync
runner aggressively delete the stale file or leave it for audit? Batch
leaves it.

---

## Bug 8 — `process_batch` skip counter is structurally zero

**Status:** Observed · **Severity:** Low · **Surface:** Summarisation,
sync runner reporting.

**Symptom.**
[`runner.py:707-723`](../pipeline/stages/summarization/runner.py)
computes `n_ok = sum(1 for r in results if r["status"] == "success")`,
`n_err = sum(... == "error")`, and `n_skip = len(results) - n_ok - n_err`.
But `_load_result` returns the cached dict with `status="success"` —
cached papers count in `n_ok` and `n_skip` is always 0. The summary log
("Batch complete: X ok / 0 skipped (cached) / Y errors") is therefore
always wrong about the cached count.

**Fix.** Cleaner: tag the loaded dict with `status="skipped"` inside
`_load_result` (callers that destructure on status already special-case
both `success` and `error`, and a skipped status is a clearer signal
upstream than success). Alternative: count from inside `process()` itself
where the cache-hit branch is taken.

---

## Bug 9 — Sync runner instance dicts leak across papers

**Status:** Observed · **Severity:** Low · **Surface:** Summarisation,
`SummarizationRunner`.

**Symptom.** The runner stores every paper's intermediate state on
`self`:
[`runner.py:245-259`](../pipeline/stages/summarization/runner.py)
defines `_scored_map_findings`, `_normal_findings`, `_finding_groups`,
`_canonical_rules`, `_relations`, `_relate_raw_pairs`,
`_relate_skipped_pairs`, `_final_rules` — all keyed by `pmcid`. `process()`
writes to them but never deletes. Inside `process_batch` a single runner
processes N papers and these dicts grow without bound. For a 100-paper
sweep with thousands of RELATE pairs per paper, the runner ends up
carrying every raw NLI pair from every paper in memory.

**Diagnosis.** The dicts exist only because `process()` builds its result
dict from them at the end. Local variables would do.

**Fix.** Either (a) `pop` the per-paper entries at the end of `process()`
after the result dict has been materialised, or (b) move the state to
local variables and stop touching `self` (preferred — also makes the
runner thread-safe, currently it isn't). The docstring at line 102 even
acknowledges the shape is per-paper.

---

## Bug 10 — `artifact_filter` rebuild uses dict equality instead of identity

**Status:** Observed · **Severity:** Medium (latent) · **Surface:**
PDF extraction, artifact filter.

**Symptom.**
[`components/artifact_filter.py:59`](../pipeline/stages/pdf_text_extraction/components/artifact_filter.py)
rebuilds the filtered `List[LayoutElement]` via
`[el for i, el in enumerate(elements) if element_dicts[i] in filtered_dicts]`
— a list-`__contains__` membership check on dicts. Two issues. (1) The
scan is O(N²) in element count. (2) Critically, it relies on the dicts
being byte-equal across the call: if `filter_artifacts` ever mutates a
kept dict in place (a future Unicode normalisation, a `fix_ligatures`
pass on `text`, anything), the post-filter dict no longer `==`'s the
pre-filter dict and the corresponding `LayoutElement` is silently dropped
with no log or error.

**Evidence.** Today
[`parsers/layout_utils.filter_artifacts`](../parsers/layout_utils.py)
returns the dicts unchanged so the bug is dormant. But the abstraction
boundary doesn't enforce that — a contributor adding any normalisation
will trigger silent data loss.

**Fix.** Two options:
1. Rebuild by identity: change `filter_artifacts` to return the list of *indices* it kept, and reuse those indices to index into the original `LayoutElement` list. Eliminates both the O(N²) and the mutation hazard.
2. Build a `dict[id(d), LayoutElement]` lookup before calling filter, then look up each kept dict by `id()`. Still depends on the filter not constructing new dicts.

(1) is the cleaner contract.

---

## Bug 11 — `ModelRegistry.docling_converter` ignores `DoclingConfig`

**Status:** Observed · **Severity:** Low · **Surface:** PDF extraction,
`ModelRegistry`.

**Symptom.**
[`pipeline/stages/pdf_text_extraction/resources.py`](../pipeline/stages/pdf_text_extraction/resources.py)
exposes `ModelRegistry.docling_converter` which builds a
`PdfPipelineOptions(do_table_structure=..., do_ocr=...)` from
`DoclingConfig` but hard-codes `images_scale=2.0` and never constructs
`AcceleratorOptions`. So `DoclingConfig.images_scale`,
`accelerator_device`, `ocr_engine`, `force_full_page_ocr` are silently
ignored when a caller goes through `ModelRegistry`.

**Evidence.** `PipelineRunner` does not currently use
`ModelRegistry.docling_converter` — each component
(`DoclingLayoutExtractor._get_converter`, etc.) constructs its own
converter that honours all four fields. But `ModelRegistry` is exported
from `pipeline/stages/pdf_text_extraction/__init__.py` as public API; any
external caller (a notebook, a future tool) reaching for it gets the
broken converter.

**Fix.** Either (a) delete `ModelRegistry.docling_converter` (and the
class if nothing else uses it) since the components already do this work
themselves, or (b) bring the converter construction up to parity with
`DoclingLayoutExtractor._get_converter`.

---

## Bug 12 — `two_pass_extractor` header strip mixes coordinate systems

**Status:** Observed · **Severity:** Low (clarity / latent) · **Surface:**
PDF extraction, two-pass extractor.

**Symptom.**
[`components/two_pass_extractor.py:382-398`](../pipeline/stages/pdf_text_extraction/components/two_pass_extractor.py)
constructs the header strip with `docling_y1 = page_h` (Docling
coordinates, y=0 at bottom) and `docling_y2 = page_h - fitz_header_bottom`
(fitz coordinates, y=0 at top). The two are adjacent lines named with
different prefixes; a single `docling_y1 > docling_y2` comparison is the
only thing preventing a sign-flip if the names get muddled in a future
refactor.

**Fix.** Build the rect in fitz coordinates throughout
(`fitz.Rect(0, 0, page_w, fitz_header_bottom)`) and convert once with
`BoundingBox.from_fitz_rect`. Same numeric result, no mixed-coord lines.

---

## Bug 13 — Inspector "NLI B→A" sort typo

**Status:** Fixed (2026-05-14) · **Severity:** Low · **Surface:** Inspector
batch index template.

**Symptom.** Clicking the **NLI B→A** column header in
`out/inspector/**/index.html` did not reorder rows. Other numeric columns
sorted correctly.

**Diagnosis.**
[`scripts/templates/pipeline_batch_index.html.jinja2:276`](../scripts/templates/pipeline_batch_index.html.jinja2)
read `parseFloat(a.dataset.nilBa)` instead of `dataset.nliBa`. HTML5
`dataset` camel-cases dashed attributes (`data-nli-ba` → `nliBa`), so the
typed key resolved to `undefined` and `parseFloat(undefined) → NaN → 0` —
every row compared as 0, no movement.

**Fix.** One-character correction: `nilBa` → `nliBa`. Regenerate index
HTMLs with `scripts/inspect_pipeline_output.py --batch-dir <dir>`.

---

## Bug 14 — Inspector `badge-blue` class missing

**Status:** Fixed (2026-05-14) · **Severity:** Low (latent) · **Surface:**
Inspector batch index template.

**Symptom.** SCOPE_QUALIFY corpus relations would render with an unstyled
"badge-blue" tag (just text on the default body background).

**Diagnosis.**
[`pipeline_batch_index.html.jinja2:194`](../scripts/templates/pipeline_batch_index.html.jinja2)
applies class `badge-blue` for SCOPE_QUALIFY rows, but the stylesheet
(lines 44-48) only defined `badge-green/red/orange/gray/cyan`. CSS class
selector silently does nothing for an undefined class.

Currently dormant because [B-006](#bug-6--scope_qualify-plumbing-is-dead)
means `_classify_pair` never emits SCOPE_QUALIFY — no row ever takes the
blue branch. The moment B-006 is fixed and SCOPE_QUALIFY can be emitted,
this would have surfaced.

**Fix.** Added `.badge-blue { background: #1e3a8a; color: #93c5fd; }`
matching the neighbouring badge palette.

---

## Bug 15 — Raw LLM enum values lost on coercion

**Status:** Fixed (2026-05-14) · **Severity:** Medium · **Surface:**
Summarisation MAP — `Finding` Pydantic model + `sum_map_findings` table.

### Symptom

When an LLM voter emitted a `relation_type` not in
`RelationTypeEnum` (e.g. `"associates_with"`, `"correlation"`), the
`_coerce_invalid_relation_type` field-validator in
[`pipeline/stages/summarization/models.py`](../pipeline/stages/summarization/models.py)
silently rewrote it to `RelationTypeEnum.unclear`. Same story for
`direction` and (post-B-016) for `category` legacy `"demographics"`. The
row in `sum_map_findings` then carried only the coerced enum value;
downstream NORMALIZE / GROUP / CANONICALIZE / RELATE / RESOLVE stages
only ever saw the post-coercion value.

The raw string *was* logged to `logs/enum_observations.jsonl` via
`enum_logging.log_enum_observation`, but with empty `context` — no
`finding_id`, no `pmcid`, no `chunk_id`. There was no way to SQL-join a
row in `sum_map_findings` to the raw value the LLM had actually produced
for it.

### Diagnosis

Two failure modes:

1. **Hidden coercion.** If a strong model (Sonnet 4.6) systematically
   reaches for an out-of-enum label, the result table shows it "gave up"
   (`unclear`). That's wrong — it had an opinion, the enum was too
   narrow. We were losing the signal that would tell us so.
2. **No join key.** `enum_observations.jsonl` records were
   write-only telemetry: no PK back to the finding row, so even the raw
   strings we *did* capture were unattributable.

### Fix

1. **Pydantic capture.** Added a `@model_validator(mode="wrap")` to
   `Finding` (`models.py`) — `_capture_raw_then_validate` reads
   `relation_type`, `direction`, `category` from the raw input dict
   *before* any field validator runs, stashes them into three new
   `PrivateAttr`s (`_raw_relation_type`, `_raw_direction`,
   `_raw_category`), then calls `handler(data)` to run the existing
   validation. Exposed via `raw_relation_type` / `raw_direction` /
   `raw_category` read-only properties. PrivateAttrs don't appear in the
   OpenAI strict schema, so the prompt schema is unchanged.
2. **DB schema.** Added nullable `Text` columns
   `raw_relation_type`, `raw_direction`, `raw_category` on
   [`SumMapFinding`](../database/models.py) via Alembic
   [`0011_add_raw_llm_columns_to_sum_map_findings.py`](../alembic/versions/0011_add_raw_llm_columns_to_sum_map_findings.py).
3. **Plumbing.** `SummarizationRunner._persist_map_findings`
   ([`runner.py:1127`](../pipeline/stages/summarization/runner.py))
   passes `f.raw_relation_type` / `f.raw_direction` / `f.raw_category`
   into the row dict.

### Verification

`python -c "..."` smoke test exercised five cases — valid values, invalid
`relation_type` ("associates_with"), legacy `"demographics"`,
`direction=None`, invalid `direction="maybe"` — all coerce correctly and
preserve the raw string. `tests/summarization/test_demographics.py`
(12 tests) + `scripts/test_map_schema.py` (18/19, 1 unrelated
pre-existing failure) all still pass.

### Limitations

* Raw values only land on `sum_map_findings`. They do *not* propagate
  to `sum_normal_findings`, `sum_finding_groups`, `sum_canonical_rules`
  yet — see TODO under [B-015 propagation](THESIS.md#todos).
* `confidence` is a strict `Literal`; out-of-vocab values cause the whole
  finding to be dropped before this capture runs (see
  `_drop_invalid_findings` and `bad_findings.jsonl`). So there is no
  `raw_confidence` column — it would always equal `confidence` for
  successful rows.

---

## Bug 16 — `demographic` spelling and confidence casing divergence

**Status:** Fixed (2026-05-14) · **Severity:** Low · **Surface:**
Summarisation MAP prompt + Pydantic models.

### Symptom

Two prompt-schema inconsistencies that were live in production:

1. **`demographic` / `demographics` split.** `category` enum (Literal)
   used `"demographics"` (plural). `relation_type` enum
   (`RelationTypeEnum`) used `"demographic"` (singular). The MAP prompt
   had to carry an explicit warning about the divergence
   (`prompts.py:30, 73-75`) and a `_CATEGORY_ALIASES` map was needed to
   repair `"demographic" → "demographics"` on `category`. Every category
   sibling (`morphology, IHC, molecular_genetics, staging, treatment,
   prognosis`) is singular — `demographics` was the lone plural even
   within its own enum.
2. **`confidence` casing.** `Finding.confidence` =
   `Literal["high","medium","low"]`. `Rule.confidence` =
   `Literal["High","Medium","Low"]`. MAP prompt + RULE prompt
   instructed the LLM in matching cases. Different fields, but
   gratuitously inconsistent.

### Diagnosis

Both inconsistencies were accidents of incremental development, not
semantic. The first one even had a warning in the prompt explaining "do
not confuse them" — the right reaction is to remove the divergence, not
document it.

### Fix

1. Aligned `category` Literal → `"demographic"` across all 5 occurrences
   in `models.py` (`Finding`, `NormalFinding`, `FindingGroup`,
   `AtomicFinding`, `CanonicalRule`, `FinalRule`), plus the alias-repair
   `valid_values` list.
2. Inverted `_CATEGORY_ALIASES` to `{"demographics": "demographic"}` —
   legacy LLM output that still says `"demographics"` is now repaired
   the other way.
3. Rewrote MAP prompt category list (`prompts.py:26`); dropped the
   divergence warning at lines 30 and 73-75; updated the OutputFormat
   exemplar at line 175.
4. Aligned `Rule.confidence` Literal → lowercase (`models.py:368`).
   Updated RULE prompt exemplar at `prompts.py:298`.
5. Updated `routing/schema_validator.py:_VALID_CATEGORIES`.
6. Bumped `MAP_PROMPT_VERSION` → `map_prompt_v2_singular_demographic`
   so MAP caches built against the old prompt are invalidated.
7. Updated tests: `tests/summarization/test_demographics.py` and
   `scripts/test_map_schema.py` — assertions flipped, alias-repair
   direction inverted.

### Verification

12/12 `test_demographics.py` pass; 11/11 enum-related checks in
`scripts/test_map_schema.py` pass.

### Out of scope

* `eval/silver/prompts.py` still uses `"demographics"` — own
  `PROMPT_VERSION`, separate cache lineage. Main pipeline's alias-repair
  handles it. TODO entry filed for next silver regeneration.
* `README.md` example snippets show old casing — cosmetic only.
* Existing DB rows from pre-fix runs hold `"demographics"`. User plan is
  to re-run MAP rather than backfill.
