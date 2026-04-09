"""
Corpus-level relation stage: pool CanonicalRules from all per-paper JSONs and
run NLI-based pairwise comparison over the full corpus.

Each output relation carries a comparison_scope label:
  - "intra_paper"  — both rules come from the same paper's processing run
  - "cross_paper"  — rules come from different papers

All pairs go through the same _should_compare gate and NLI pipeline as the
per-paper RELATE stage.  Intra-paper pairs are included so the corpus graph is
complete — a rule's full neighborhood (both within-paper support and cross-paper
contradictions) is visible in one artifact.

This stage is entirely post-hoc.  It does NOT modify any per-paper output, does
NOT affect ResolveStage scoring, and does NOT alter per-paper `relations` keys.
Per-paper `relations` remain the authoritative input for ResolveStage.

Run-selection policy
--------------------
The current pipeline writes exactly one JSON per PMCID (``{pmcid}.json``),
always overwriting it with the latest successful result.  That makes single-file-
per-PMCID the common case.  However, when a corpus directory was assembled from
multiple batch runs (or when ``run_selection="all"`` is requested), the stage
must decide which runs to include.

  "latest_per_pmcid" (default)
      Group all files by PMCID.  Discard files whose status is not "success" or
      whose canonical_rules list is empty.  Among the remaining candidates for
      each PMCID, pick the one with the lexicographically largest run_id
      timestamp suffix (``{pmcid}_{YYYYMMDDTHHmmss}``).  Fall back to file mtime
      when run_id is absent or unparseable.  Emit a WARNING if no usable run is
      found for a PMCID.

  "all"
      Include every file that has status="success" and non-empty canonical_rules.
      If a PMCID has multiple runs all of them are pooled, which means the same
      claim may appear more than once in the corpus graph.  Use only when you
      specifically need to compare runs.

  manifest (list[Path])
      Ignore source_dir.  Load exactly the provided files without any filtering.
      Intended for testing and version-pinned workflows.

Usage
-----
Standalone (no runner required):
    from pathlib import Path
    from pipeline.stages.summarization.corpus_relate import CorpusRelateStage

    CorpusRelateStage().relate_from_dir(
        source_dir=Path("out/summaries/summaries/"),
        output_path=Path("out/summaries/corpus_relations.json"),
    )

Via SummarizationRunner (after process_batch):
    results = runner.process_batch(papers)
    runner.corpus_relate(
        source_dir=runner._summaries_dir,
        output_path=runner._output_dir / "corpus_relations.json",
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .models import CanonicalRule, CorpusRelation
from .relate_stage import RelateStage

logger = logging.getLogger(__name__)

# Matches the timestamp suffix in run_id: "{pmcid}_{YYYYMMDDTHHmmss}"
_TS_RE = re.compile(r"_(\d{8}T\d{6})$")


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _run_id_ts(run_id: str) -> str:
    """Extract the YYYYMMDDTHHmmss suffix from run_id, or "" if absent."""
    m = _TS_RE.search(run_id or "")
    return m.group(1) if m else ""


class CorpusRelateStage:
    """
    Corpus-level NLI relation stage.

    Reuses RelateStage.relate() verbatim — same comparability gate (_should_compare),
    same NLI model singleton, same direction-polarity guard (_classify_pair).
    The additions are:
      - run-selection: one representative run per PMCID by default
      - source_pmcid tracking per canonical_id (built while loading JSONs)
      - post-hoc enrichment of each Relation with corpus-level fields
      - JSON output with comparison_scope and same_paper labels

    Parameters
    ----------
    entailment_threshold:
        Forwarded to RelateStage.  Default 0.50 matches SummarizationRunner default.
    contradiction_threshold:
        Forwarded to RelateStage.  Default 0.50 matches SummarizationRunner default.
    """

    def __init__(
        self,
        entailment_threshold: float = 0.50,
        contradiction_threshold: float = 0.50,
    ) -> None:
        self._relate = RelateStage(
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def relate_from_dir(
        self,
        source_dir: Path | None,
        output_path: Path,
        run_selection: str = "latest_per_pmcid",
        manifest: list[Path] | None = None,
    ) -> list[CorpusRelation]:
        """
        Pool CanonicalRules from per-paper JSONs, run RelateStage, write output.

        Parameters
        ----------
        source_dir:
            Directory containing per-paper JSON files (*.json).  Ignored when
            manifest is provided.  Required when manifest is None.
        output_path:
            Destination for corpus_relations.json.  Parent directories are
            created if absent.
        run_selection:
            "latest_per_pmcid" (default) — one representative run per PMCID.
            "all" — every valid file (status=success, non-empty canonical_rules).
            Ignored when manifest is provided.
        manifest:
            Explicit list of JSON paths to load.  When given, source_dir and
            run_selection are both ignored.  No status/rules filtering is applied
            (the caller is responsible for selecting valid files).

        Returns
        -------
        List of CorpusRelation objects written to output_path.
        """
        # ── Resolve the file list ──────────────────────────────────────────────
        if manifest is not None:
            selected: dict[str, tuple[Path, dict]] = self._load_manifest(manifest)
        else:
            if source_dir is None:
                raise ValueError("source_dir is required when manifest is not provided")
            json_files = sorted(source_dir.glob("*.json"))
            if not json_files:
                logger.warning("CORPUS RELATE: no JSON files found in %s", source_dir)
                return []
            selected = self._select_runs(json_files, run_selection)

        if not selected:
            logger.warning("CORPUS RELATE: no usable runs found — nothing to compare")
            return []

        logger.info(
            "CORPUS RELATE: selected %d papers (run_selection=%r)",
            len(selected),
            "manifest" if manifest is not None else run_selection,
        )

        # ── Build canonical rule pool + lookup indexes ─────────────────────────
        all_rules: list[CanonicalRule] = []
        id_to_pmcid: dict[str, str] = {}     # canonical_id → source PMCID
        rule_meta: dict[str, dict] = {}      # canonical_id → raw rule dict
        pmcids_loaded: list[str] = []

        for pmcid, (_path, data) in selected.items():
            raw_rules = data.get("canonical_rules") or []
            pmcids_loaded.append(pmcid)
            for rd in raw_rules:
                try:
                    rule = CanonicalRule.model_validate(rd)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "CORPUS RELATE: [%s] could not parse canonical rule %r — %s",
                        pmcid, rd.get("canonical_id", "?"), exc,
                    )
                    continue
                all_rules.append(rule)
                id_to_pmcid[rule.canonical_id] = pmcid
                rule_meta[rule.canonical_id] = rd

        logger.info(
            "CORPUS RELATE: loaded %d canonical rules from %d papers",
            len(all_rules), len(pmcids_loaded),
        )

        # ── Run RelateStage on the full pool ───────────────────────────────────
        if len(all_rules) < 2:
            logger.warning(
                "CORPUS RELATE: fewer than 2 rules across the corpus — nothing to compare"
            )
            corpus_relations: list[CorpusRelation] = []
        else:
            raw_relations = self._relate.relate(all_rules, pmcid="corpus")
            logger.info(
                "CORPUS RELATE: %d raw relations from pool of %d rules",
                len(raw_relations), len(all_rules),
            )
            corpus_relations = self._enrich(raw_relations, id_to_pmcid, rule_meta)

        # ── Write output ───────────────────────────────────────────────────────
        intra_count = sum(1 for r in corpus_relations if r.same_paper)
        cross_count = len(corpus_relations) - intra_count

        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source_dir) if source_dir else None,
            "run_selection": "manifest" if manifest is not None else run_selection,
            "pmcids_loaded": pmcids_loaded,
            "canonical_rules_loaded": len(all_rules),
            "total_relations": len(corpus_relations),
            "intra_paper_count": intra_count,
            "cross_paper_count": cross_count,
            "relations": [r.model_dump() for r in corpus_relations],
        }
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "CORPUS RELATE: written %d relations "
            "(%d intra-paper, %d cross-paper) → %s",
            len(corpus_relations), intra_count, cross_count, output_path,
        )
        return corpus_relations

    # ── Run-selection ──────────────────────────────────────────────────────────

    def _select_runs(
        self,
        json_files: list[Path],
        run_selection: str,
    ) -> dict[str, tuple[Path, dict]]:
        """
        Parse all json_files and return a selected subset.

        Returns
        -------
        Dict of pmcid → (path, parsed_data) for the selected runs.
        Files that fail to parse are skipped with a WARNING.
        Files with status != "success" or empty canonical_rules are silently
        discarded (not a warning — this is expected for error-status results).
        """
        # Parse every file once
        parsed: list[tuple[Path, dict]] = []
        for p in json_files:
            try:
                with open(p, encoding="utf-8") as fh:
                    data = json.load(fh)
                parsed.append((p, data))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CORPUS RELATE: skipping %s — could not parse JSON: %s",
                    p.name, exc,
                )

        if run_selection == "all":
            return self._filter_valid(parsed)

        # Default: latest_per_pmcid
        return self._latest_per_pmcid(parsed)

    def _filter_valid(
        self,
        parsed: list[tuple[Path, dict]],
    ) -> dict[str, tuple[Path, dict]]:
        """
        Keep every file that has status="success" and non-empty canonical_rules.
        When a PMCID appears more than once, both entries are kept under distinct
        keys (pmcid and pmcid_{run_id} for duplicates).
        """
        result: dict[str, tuple[Path, dict]] = {}
        seen_counts: dict[str, int] = defaultdict(int)

        for p, data in parsed:
            if data.get("status") != "success":
                continue
            if not data.get("canonical_rules"):
                continue
            pmcid = data.get("pmcid") or p.stem
            seen_counts[pmcid] += 1
            # First occurrence uses plain pmcid key; duplicates get a suffix
            key = pmcid if seen_counts[pmcid] == 1 else f"{pmcid}__{seen_counts[pmcid]}"
            result[key] = (p, data)

        return result

    def _latest_per_pmcid(
        self,
        parsed: list[tuple[Path, dict]],
    ) -> dict[str, tuple[Path, dict]]:
        """
        For each PMCID, select the latest successful run with non-empty
        canonical_rules.

        Sort key priority:
          1. run_id timestamp suffix (YYYYMMDDTHHmmss) — present in all runner-
             generated files since _make_run_id was introduced.
          2. file mtime — fallback when run_id is absent or has no timestamp.

        A WARNING is emitted for any PMCID where no usable run is found (i.e.,
        all candidates were either failed or had empty canonical_rules).
        """
        # Group by PMCID, collecting all (path, data) candidates
        by_pmcid: dict[str, list[tuple[Path, dict]]] = defaultdict(list)

        for p, data in parsed:
            pmcid = data.get("pmcid") or p.stem
            by_pmcid[pmcid].append((p, data))

        result: dict[str, tuple[Path, dict]] = {}

        for pmcid, candidates in by_pmcid.items():
            # Filter to usable runs
            usable = [
                (p, d) for p, d in candidates
                if d.get("status") == "success" and d.get("canonical_rules")
            ]
            if not usable:
                logger.warning(
                    "CORPUS RELATE: [%s] no usable run found "
                    "(all %d candidate(s) failed or had empty canonical_rules)",
                    pmcid, len(candidates),
                )
                continue

            # Sort descending: prefer run_id timestamp, fall back to mtime
            def _sort_key(item: tuple[Path, dict]) -> tuple[str, float]:
                path, data = item
                ts = _run_id_ts(data.get("run_id") or "")
                mtime = path.stat().st_mtime if path.exists() else 0.0
                return (ts, mtime)

            best_path, best_data = max(usable, key=_sort_key)
            result[pmcid] = (best_path, best_data)

        return result

    def _load_manifest(
        self,
        manifest: list[Path],
    ) -> dict[str, tuple[Path, dict]]:
        """
        Load exactly the files in manifest.  No status/rules filtering.
        PMCID key collisions (same pmcid in multiple files) are resolved by
        appending a counter suffix, matching _filter_valid behavior.
        """
        result: dict[str, tuple[Path, dict]] = {}
        seen_counts: dict[str, int] = defaultdict(int)

        for p in manifest:
            try:
                with open(p, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CORPUS RELATE: manifest entry %s could not be parsed — %s",
                    p.name, exc,
                )
                continue
            pmcid = data.get("pmcid") or p.stem
            seen_counts[pmcid] += 1
            key = pmcid if seen_counts[pmcid] == 1 else f"{pmcid}__{seen_counts[pmcid]}"
            result[key] = (p, data)

        return result

    # ── Enrichment ─────────────────────────────────────────────────────────────

    def _enrich(
        self,
        raw_relations: list,
        id_to_pmcid: dict[str, str],
        rule_meta: dict[str, dict],
    ) -> list[CorpusRelation]:
        """
        Convert raw Relation objects → CorpusRelation by adding provenance and
        human-readable fields.

        relation_id is a deterministic 8-char hash of the sorted canonical_id
        pair so it is stable across re-runs (same JSON files → same IDs).
        """
        result: list[CorpusRelation] = []

        for rel in raw_relations:
            cid_a = rel.rule_id_a
            cid_b = rel.rule_id_b
            pmcid_a = id_to_pmcid.get(cid_a, "unknown")
            pmcid_b = id_to_pmcid.get(cid_b, "unknown")
            same_paper = pmcid_a == pmcid_b

            # Sort IDs for a deterministic hash regardless of which rule was A vs B
            sorted_pair = "|".join(sorted([cid_a, cid_b]))
            relation_id = f"COR_{_sha8(sorted_pair)}"

            meta_a = rule_meta.get(cid_a, {})
            meta_b = rule_meta.get(cid_b, {})

            corpus_rel = CorpusRelation(
                # ── inherited Relation fields ──────────────────────────────────
                rule_id_a=cid_a,
                rule_id_b=cid_b,
                relation_type=rel.relation_type,
                nli_score_a_to_b=rel.nli_score_a_to_b,
                nli_score_b_to_a=rel.nli_score_b_to_a,
                # ── corpus-level provenance ────────────────────────────────────
                relation_id=relation_id,
                comparison_scope="intra_paper" if same_paper else "cross_paper",
                same_paper=same_paper,
                pmcid_a=pmcid_a,
                pmcid_b=pmcid_b,
                # ── structural metadata (from rule A; gate ensures A and B share these) ─
                subject_entity=meta_a.get("subject_entity") or "",
                outcome_entity=meta_a.get("outcome_entity") or "",
                category=meta_a.get("category") or "",
                relation_type_structural=meta_a.get("relation_type") or "",
                # ── per-rule fields ────────────────────────────────────────────
                direction_a=meta_a.get("direction"),
                direction_b=meta_b.get("direction"),
                predicate_a=meta_a.get("predicate_text") or "",
                predicate_b=meta_b.get("predicate_text") or "",
                mean_grounding_a=meta_a.get("mean_grounding_score"),
                mean_grounding_b=meta_b.get("mean_grounding_score"),
                finding_count_a=meta_a.get("finding_count") or 0,
                finding_count_b=meta_b.get("finding_count") or 0,
                supporting_pmcids_a=meta_a.get("supporting_pmcids") or [],
                supporting_pmcids_b=meta_b.get("supporting_pmcids") or [],
                canonical_scope_a=meta_a.get("canonical_scope") or "unknown",
                canonical_scope_b=meta_b.get("canonical_scope") or "unknown",
                # ── scope qualifier check (v1: not yet implemented) ────────────
                scope_check_result="scope_unknown",
                scope_note="",
            )
            result.append(corpus_rel)

        return result
