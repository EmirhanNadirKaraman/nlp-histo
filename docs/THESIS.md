# Thesis Notebook — Issues found, decisions made

A running ledger of substantive issues discovered in the `nlp-histo` pipeline,
the diagnosis trail for each, the fix shipped, and the artifacts that
demonstrate the before/after behaviour. Reproduction commands live in
[`HOW_TO_RUN.md`](HOW_TO_RUN.md); code layout lives in
[`STRUCTURE.md`](STRUCTURE.md).

> **How to use this file:**
> * [Bugs (catalogue)](#bugs-catalogue) — every substantive defect, with status.
> * [TODOs](#todos) — work to carry forward into the thesis or follow-up commits.
> * [Decisions log](#decisions-log) — non-obvious calls made along the way.
> * Detailed write-ups follow as `## Bug N — …` and `## Topic — …` sections.
>
> Add a new entry the same day the issue is discovered. New bug → row in the
> catalogue + detail section. New TODO → bullet in the TODO list (with the bug
> or section it belongs to). Never delete entries — flip `Status` to `Fixed` /
> `Won't fix` / `Superseded` so the history survives.

---

## Bugs (catalogue)

| ID | Status | Severity | Surface | One-line summary | Detail |
|----|--------|----------|---------|------------------|--------|
| B-001 | Fixed (2026-05-13) | High | Summarisation, corpus relate | `canonical_id` collided across papers because `group_id` did not include `pmcid`; same-rule self-pairs appeared as bogus "intra_paper" SUPPORT relations. | [Bug 1](#bug-1--duplicate-intra-paper-relations-produced-by-canonical_id-collisions) |
| B-002 | Mitigated (2026-05-13) | Medium | PDF extraction, Docling layout | Docling emits phantom layout elements — real text content at a bbox that does not render (often in the header zone of the wrong page). `ContextAwareStitcher` had been masking this; now caught upstream by `TwoPassConfig.enabled=True` + Rule R1/R3. | [Bug 2](#bug-2--docling-phantom-layout-elements) |
| B-003 | Mitigated (2026-05-13) | Low (latent) | PDF extraction, color signal | Rule R-color (`max_white_char_fraction < 1.0`) treats *any* near-white span as ghost text, ignoring the rendered background. Would have produced false positives on white-on-coloured headers if applied to TEXT/CAPTION types. Currently dormant because `SECTION_HEADER` is in `NodeScorer._ALWAYS_KEEP`. Disabled by default now (`max_white_char_fraction=1.0`). | [Bug 3](#bug-3--r-color-white-text-false-positive-latent) |
| B-004 | Observed | Low | PDF extraction, Docling glyph fallback | CID-only PDFs surface in Docling as `GLYPH<…>` / `/gid00001` text strings. R1 drops them because their bboxes have no ink (Docling didn't decode the font). No production impact, but worth keeping an eye on for corpora with subset-only fonts. | [Bug 4](#bug-4--cid-glyph-fallback-strings) |
| B-005 | Mitigated (2026-05-14) | High | Summarisation, batch runner | `BatchSummarizationRunner.finalize()` was missing six features the sync runner had: (1) `_replace_verbatim_from_db` — grounding NLI ran against LLM paraphrases instead of source text; (2) stable `compute_finding_id`; (3) DB persistence to `sum_*` tables; (4) `corpus_relate_incremental`; (5) `rejection_summary` build + persist; (6) NER + UMLS linking. Since `scripts/run_paper.py` defaults to batch mode, every batched production result between commit `5c59c3e` (2026-04-27) and the 05-14 backport was grounded against paraphrased text. | [Bug 5](#bug-5--batch-runner-missing-sync-parity-features) |

Add new rows here when you discover something. Bump the ID monotonically (`B-006`, `B-007`, …). Put the long write-up in a new `## Bug N — …` section below.

---

## TODOs

Carry-forward items. Tick `- [x]` when shipped, then move the entry to the
"Decisions log" if it represents a permanent decision, or to the matching Bug
detail section if it closes a bug.

* [ ] **B-001 follow-up:** regenerate `out/summaries/runs/*/` artifacts for papers that participated in a corpus-relate run *before* the `pmcid`-in-`group_id` fix — their `canonical_id`s are still colliding values on disk.
* [ ] **B-001 defence-in-depth:** in [`pipeline/stages/summarization/helpers/corpus_relate.py`](../pipeline/stages/summarization/helpers/corpus_relate.py), guard against the case where two rules in `all_rules` happen to share a `canonical_id` (impossible after B-001 fix, but cheap to assert).
* [ ] **B-002 audit:** for every paper in production, count how often `ContextAwareStitcher` had to absorb a phantom element pre-fix vs. post-fix. If the post-fix count is materially lower it strengthens the result in the thesis.
* [ ] **B-003 refactor:** reconsider the blanket `NodeScorer._ALWAYS_KEEP` exemption for `SECTION_HEADER`. R1 and R3 are safe to apply to headers; only R-color needs the exemption, and R-color is now disabled by default. Per-rule exemptions would be cleaner.
* [ ] **Summarisation throughput:** the cheap-tier MAP cascade hits the L3 escalation more than expected on multi-claim sentences. Worth profiling once cost-percentile sweep ([`scripts/estimate_pipeline_cost_percentiles.py`](../scripts/estimate_pipeline_cost_percentiles.py)) lands.
* [ ] **Thesis figure polish:** the `out/thesis_demo/ghost_text/*.png` crops are 110-dpi; bump to 200-dpi when finalising thesis figures so banner type stays sharp at print size.
* [ ] **B-005 dedup:** the persistence + verbatim-from-DB helpers are line-for-line copies between `SummarizationRunner` and `BatchSummarizationRunner`. Lift into `pipeline/stages/summarization/persistence.py` as module-level functions taking `db` as a parameter, then point both runners at the shared impl. Markers: `TODO: deduplicate with SummarizationRunner` in `batch/runner.py`.
* [ ] **B-005 end-to-end verification:** run `scripts/run_paper.py PMC<x> --sync=false` against a real DB with `db=get_db_connection()` wired in, confirm `sum_map_findings`/`sum_normal_findings`/…/`sum_rejection_summaries` get rows for `pipeline_run_id`, and spot-check `sum_map_findings.verbatim_support == text_element.text_content` for at least one finding. `scripts/run_paper.py:build_batch_runner` does **not** wire `db` by default — only `scripts/run_paper_single_model.py` does.

---

## Decisions log

Permanent design calls — keep terse, link to the discussion in the detail section.

| Date | Decision | Rationale | Detail |
|------|----------|-----------|--------|
| 2026-05-13 | `group_id` includes `pmcid` in its hash. | Prevents cross-paper `canonical_id` collisions; cross-paper matching uses the dedicated CUI / normalised-string gate in `corpus_relate.py`, not id equality. | [Bug 1](#bug-1--duplicate-intra-paper-relations-produced-by-canonical_id-collisions) |
| 2026-05-13 | `TwoPassConfig.enabled = True` by default. | R1 (pixel render) reliably catches Docling phantom elements and PDF render-mode-3 / fill-opacity-0 ghost text. The stitcher no longer has to be the only line of defence. | [Topic — ghost-text detection](#topic--ghost-text-detection-empirical-verification-and-policy-fix) §2.3 |
| 2026-05-13 | `TwoPassConfig.max_white_char_fraction = 1.0` (R-color disabled). | Empirically produces false positives on legitimate inverted headers; R1 is the source of truth. Re-enable per-corpus only if R1 is unavailable. | [Bug 3](#bug-3--r-color-white-text-false-positive-latent) |
| 2026-05-13 | Don't add a pdfminer.six render-mode pass. | R1 already catches Tr=3 and opacity-0 as "no ink in bbox" — verified synthetically (`scripts/verify_ghost_text_detection.py`). A second detector would be duplicated work. | [Topic — ghost-text detection](#topic--ghost-text-detection-empirical-verification-and-policy-fix) §2.1 |
| 2026-05-14 | Batch runner brought to parity with sync runner. | Six features (verbatim-from-DB, stable `finding_id`, DB persistence, `corpus_relate_incremental`, `rejection_summary`, NER) were sync-only; production batch runs were grounding against paraphrased text. Copied the methods directly into `batch/runner.py` rather than refactoring sync to keep blast radius small. | [Bug 5](#bug-5--batch-runner-missing-sync-parity-features) |
| 2026-05-14 | Batch result caching uses `pipeline_config_hash` invalidation. | Stamping the result JSON with the current config hash + checking it at `submit()` is the cheapest correct cache key. Any threshold/model/schema/prompt change auto-invalidates without manual cleanup. Pre-fix JSONs in `out/summaries/summaries/` were deleted to force regeneration. | [Bug 5](#bug-5--batch-runner-missing-sync-parity-features) |

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
