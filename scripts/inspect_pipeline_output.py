"""
inspect_pipeline_output.py
──────────────────────────
Generate a single self-contained HTML inspector from one pipeline output JSON.

Usage
-----
    PYTHONPATH=. python scripts/inspect_pipeline_output.py out/summaries/summaries/PMC10047158.json
    PYTHONPATH=. python scripts/inspect_pipeline_output.py out/summaries/summaries/PMC10047158.json -o out/inspector/PMC10047158.html

The output HTML file can be opened directly in any browser — no server needed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("ERROR: jinja2 is required.  pip install jinja2", file=sys.stderr)
    sys.exit(1)


# ── Suspicious-entity heuristics ──────────────────────────────────────────────

# Biomedical taxonomy strings that often leak into subject_entity due to UMLS
# normalization overshooting (CEAN → Cetacea, etc.)
_TAXONOMY_RE = re.compile(
    r"\b(cetacea|chiroptera|rodentia|mammalia|reptilia|aves|amphibia|"
    r"actinopterygii|insecta|arachnida|bacteria|fungi|archaea|viridae)\b",
    re.I,
)
# Procedure / technique strings that shouldn't be subject entities
_PROCEDURE_RE = re.compile(
    r"\b(electroretinography|immunohistochemistry|pcr|elisa|western blot|"
    r"flow cytometry|sequencing|radiography|tomography|colonoscopy|endoscopy)\b",
    re.I,
)
# Generic / vacuous entity strings
_GENERIC_RE = re.compile(
    r"^\s*(patient|patients|case|cases|subject|subjects|individual|individuals|"
    r"sample|samples|group|groups|cohort|cohorts|cell|cells|tissue|tissues|"
    r"tumor|tumour|lesion|lesions|the patient)\s*$",
    re.I,
)
# Polarity words that imply the assertion_status shouldn't be "uncertain"
_POLARITY_RE = re.compile(
    r"\b(not|no |absent|negative|negated|present|expressed|increased|decreased|"
    r"elevated|reduced|no evidence|lack of|without)\b",
    re.I,
)


def _entity_flags(entity: str | None) -> list[str]:
    """Return warning labels for a suspicious entity string."""
    if not entity:
        return ["empty-entity"]
    flags: list[str] = []
    if _TAXONOMY_RE.search(entity):
        flags.append("taxonomy-leak")
    if _PROCEDURE_RE.search(entity):
        flags.append("procedure-as-entity")
    if _GENERIC_RE.match(entity):
        flags.append("generic-entity")
    return flags


def _compute_flags(obj: dict, low_gs_threshold: float = 0.4) -> list[str]:
    """Return all warning flags for a finding / rule dict."""
    flags: list[str] = []
    flags.extend(_entity_flags(obj.get("subject_entity")))
    outcome = obj.get("outcome_entity") or ""
    if not outcome.strip():
        flags.append("empty-outcome")
    # assertion_status vs. claim polarity mismatch
    status = obj.get("assertion_status", "")
    claim  = obj.get("claim", "") or obj.get("predicate_text", "")
    if status == "uncertain" and _POLARITY_RE.search(claim):
        flags.append("polarity-mismatch")
    # low grounding score
    gs = obj.get("grounding_score") or obj.get("mean_grounding_score")
    if gs is not None and gs < low_gs_threshold:
        flags.append("low-grounding")
    return flags


def _is_low_grounding(obj: dict, threshold: float = 0.4) -> bool:
    gs = obj.get("grounding_score") or obj.get("mean_grounding_score")
    return gs is not None and gs < threshold


# ── Lineage builder ────────────────────────────────────────────────────────────

def _build_lineage_index(data: dict) -> dict[str, Any]:
    """
    Return a lookup dict:  canonical_id → list[map_finding dicts]

    Strategy:
      1. canonical_rule.member_normal_ids  (direct link when available)
         — we don't store NormalFindings in the JSON, so use as string labels only
      2. Fuzzy match: for each canonical rule, scan all MAP findings for those
         with the same (subject_entity, outcome_entity) — useful when normal_ids
         don't appear verbatim in MAP findings.
    """
    # Build a flat list of MAP findings with chunk_id attached
    map_findings_flat: list[dict] = []
    for chunk in (data.get("audit_trail") or {}).get("map_chunks", []):
        chunk_id = chunk.get("chunk_id", "?")
        for f in chunk.get("findings", []):
            map_findings_flat.append({**f, "chunk_id": chunk_id})

    # Index: (subject_entity_lower, outcome_entity_lower) → [findings]
    by_entity: dict[tuple, list[dict]] = {}
    for f in map_findings_flat:
        key = (
            (f.get("subject_entity") or "").lower().strip(),
            (f.get("outcome_entity") or "").lower().strip(),
        )
        by_entity.setdefault(key, []).append(f)

    lineage: dict[str, dict] = {}
    for cr in data.get("canonical_rules", []):
        cid = cr.get("canonical_id", "")
        subj = (cr.get("subject_entity") or "").lower().strip()
        outc = (cr.get("outcome_entity") or "").lower().strip()

        matched: list[dict] = []
        # Exact (subject, outcome) match
        exact = by_entity.get((subj, outc), [])
        matched.extend(exact)

        # Subject-only fallback if exact gives nothing
        if not matched:
            for (s, o), flist in by_entity.items():
                if s == subj:
                    matched.extend(flist)

        lineage[cid] = {"map_findings": matched[:8]}  # cap to avoid bloat

    return lineage


# ── Template context builders ──────────────────────────────────────────────────

def _enrich_final_rule(fr: dict, lineage_index: dict) -> dict:
    cid = fr.get("canonical_id", "")
    lin = lineage_index.get(cid, {})
    return {
        **fr,
        "flags": _compute_flags(fr),
        "low_grounding": _is_low_grounding(fr),
        "lineage": lin,
        "raw_json": json.dumps(fr, indent=2, ensure_ascii=False),
    }


def _enrich_canonical_rule(cr: dict) -> dict:
    return {
        **cr,
        "flags": _compute_flags(cr),
        "low_grounding": _is_low_grounding(cr),
        "raw_json": json.dumps(cr, indent=2, ensure_ascii=False),
    }


def _enrich_finding(f: dict) -> dict:
    return {
        **f,
        "flags": _compute_flags(f),
        "low_grounding": _is_low_grounding(f),
        "raw_json": json.dumps(f, indent=2, ensure_ascii=False),
    }


def _unique_sorted(values: list[str | None]) -> list[str]:
    return sorted({v for v in values if v})


def build_context(data: dict) -> dict:
    pmcid   = data.get("pmcid", "unknown")
    run_id  = data.get("run_id", "")
    status  = data.get("status", "unknown")
    summary = data.get("summary", "")

    lineage_index = _build_lineage_index(data)

    raw_final_rules    = data.get("final_rules", []) or []
    raw_canonical_rules = data.get("canonical_rules", []) or []
    raw_relations       = data.get("relations", []) or []

    audit = data.get("audit_trail") or {}
    raw_chunks = audit.get("map_chunks", []) or []

    # Enrich each layer
    final_rules     = [_enrich_final_rule(fr, lineage_index) for fr in raw_final_rules]
    canonical_rules = [_enrich_canonical_rule(cr) for cr in raw_canonical_rules]

    enriched_chunks = []
    all_map_findings: list[dict] = []
    for chunk in raw_chunks:
        findings = [_enrich_finding(f) for f in (chunk.get("findings") or [])]
        all_map_findings.extend(findings)
        enriched_chunks.append({**chunk, "findings": findings})

    total_map_findings = len(all_map_findings)
    flagged_count      = sum(1 for fr in final_rules if fr["flags"])
    contradicted_count = sum(1 for fr in final_rules if fr.get("is_contradicted"))
    low_grounding_count = sum(1 for fr in final_rules if fr["low_grounding"])

    # Category / relation_type option lists for filter dropdowns
    fr_categories      = _unique_sorted([fr.get("category") for fr in final_rules])
    fr_relation_types  = _unique_sorted([fr.get("relation_type") for fr in final_rules])
    cr_categories      = _unique_sorted([cr.get("category") for cr in canonical_rules])
    cr_relation_types  = _unique_sorted([cr.get("relation_type") for cr in canonical_rules])
    mf_categories      = _unique_sorted([f.get("category") for f in all_map_findings])
    mf_relation_types  = _unique_sorted([f.get("relation_type") for f in all_map_findings])

    return {
        "pmcid":               pmcid,
        "run_id":              run_id,
        "status":              status,
        "summary":             summary,
        "final_rules":         final_rules,
        "canonical_rules":     canonical_rules,
        "relations":           raw_relations,
        "map_chunks":          enriched_chunks,
        "total_map_findings":  total_map_findings,
        "flagged_count":       flagged_count,
        "contradicted_count":  contradicted_count,
        "low_grounding_count": low_grounding_count,
        "fr_categories":       fr_categories,
        "fr_relation_types":   fr_relation_types,
        "cr_categories":       cr_categories,
        "cr_relation_types":   cr_relation_types,
        "mf_categories":       mf_categories,
        "mf_relation_types":   mf_relation_types,
    }


# ── Renderer ───────────────────────────────────────────────────────────────────

def render(json_path: Path, output_path: Path) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    ctx = build_context(data)

    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,  # we control the data; raw JSON is already escaped in <pre>
    )
    # Custom filter: format float with fallback
    env.filters["format_score"] = lambda v, fmt=".3f": format(v, fmt) if v is not None else "—"

    template = env.get_template("pipeline_inspector.html.jinja2")
    html = template.render(**ctx)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Inspector written to: {output_path}")
    print(f"  {ctx['total_map_findings']} MAP findings  |  "
          f"{len(ctx['canonical_rules'])} canonical rules  |  "
          f"{len(ctx['final_rules'])} final rules  |  "
          f"{ctx['flagged_count']} flagged  |  "
          f"{ctx['contradicted_count']} contradicted")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML inspector from a pipeline output JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("json_path", type=Path, help="Path to pipeline output JSON file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output HTML path (default: out/inspector/<pmcid>.html)",
    )
    parser.add_argument(
        "--low-gs-threshold", type=float, default=0.4,
        help="Grounding score below which a finding is flagged as low-grounding (default: 0.4)",
    )
    args = parser.parse_args()

    json_path: Path = args.json_path
    if not json_path.exists():
        print(f"ERROR: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        stem = json_path.stem
        output_path = Path("out/inspector") / f"{stem}.html"

    render(json_path, output_path)


if __name__ == "__main__":
    main()
