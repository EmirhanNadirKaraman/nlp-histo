"""
NORMALIZE stage: Finding[] → NormalFinding[]

Two operations:
1. Entity normalization — resolve common surface-form variants to canonical
   names using a curated biomedical synonym dictionary.  Lowercased,
   whitespace-stripped lookup.  No embedding fallback in Phase 2.

2. Conditional dedup — findings that share the same
   (text_element_id, subject_entity, outcome_entity, relation_type) after
   normalization are merged into one NormalFinding (provenance union).
   Findings that differ on any of those keys stay separate even if they
   came from the same text element.  Findings where subject_entity or
   outcome_entity is None, or relation_type is unclear, are kept as
   individual NormalFindings without attempting dedup.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

from .models import DirectionEnum, Finding, FindingScope, NormalFinding, RelationTypeEnum, SourceSpan

logger = logging.getLogger(__name__)

# ── Synonym dictionary ─────────────────────────────────────────────────────────
# Keys are lowercase surface forms; values are canonical names.
# Extend this dict as new papers are processed.

_SYNONYMS: dict[str, str] = {
    # Overall survival
    "overall survival":          "OS",
    "os":                        "OS",
    "overall survival rate":     "OS",
    # Progression-free survival
    "progression-free survival": "PFS",
    "pfs":                       "PFS",
    "progression free survival": "PFS",
    # Disease-free survival
    "disease-free survival":     "DFS",
    "dfs":                       "DFS",
    # Common biomarkers — surface variants
    "cd30":                      "CD30",
    "ber-h2":                    "CD30",
    "ki-67":                     "Ki-67",
    "ki67":                      "Ki-67",
    "mib-1":                     "Ki-67",
    "bcl-2":                     "BCL2",
    "bcl2":                      "BCL2",
    "bcl-6":                     "BCL6",
    "bcl6":                      "BCL6",
    "myc":                       "MYC",
    "c-myc":                     "MYC",
    "pd-l1":                     "PD-L1",
    "pdl1":                      "PD-L1",
    "cd20":                      "CD20",
    "cd3":                       "CD3",
    "cd8":                       "CD8",
    "cd4":                       "CD4",
    "cd10":                      "CD10",
    "cd138":                     "CD138",
    "cd56":                      "CD56",
    "cd30 expression":           "CD30",
    "cd30 positivity":           "CD30",
    "bcl2 expression":           "BCL2",
    "myc expression":            "MYC",
    # Treatment
    "r-chop":                    "R-CHOP",
    "r chop":                    "R-CHOP",
    "rituximab-chop":            "R-CHOP",
    "chop":                      "CHOP",
    # Disease subtypes — keep as-is (subtype normalization is scope's job)
}


def normalize_entity(name: str | None) -> str | None:
    """Return the canonical form of an entity name, or None if input is None."""
    if name is None:
        return None
    return _SYNONYMS.get(name.strip().lower(), name.strip())


# ── Dedup key ──────────────────────────────────────────────────────────────────

def _dedup_key(
    text_element_id: int,
    subject: str | None,
    outcome: str | None,
    relation_type: RelationTypeEnum,
) -> str | None:
    """
    Return a grouping key for conditional dedup, or None if any field that
    would make the key meaningful is absent (→ finding kept as-is, not merged).
    """
    if subject is None or outcome is None or relation_type is RelationTypeEnum.unclear:
        return None
    return f"{text_element_id}|{subject}|{outcome}|{relation_type.value}"


# ── Source span extraction ─────────────────────────────────────────────────────

def _spans_from_finding(f: Finding) -> list[SourceSpan]:
    """
    Convert a Finding's evidence list to SourceSpan objects.

    Evidence strings have the format "S{i}|{pmcid}|{te_id}".
    Malformed entries are skipped with a debug log.
    """
    spans = []
    for ev in f.evidence:
        parts = ev.split("|")
        if len(parts) != 3:
            logger.debug("Skipping malformed evidence string: %r", ev)
            continue
        sentence_id, pmcid, te_id_str = parts
        try:
            te_id = int(te_id_str)
        except ValueError:
            logger.debug("Non-integer text_element_id in evidence: %r", ev)
            continue
        spans.append(SourceSpan(
            sentence_id=sentence_id,
            pmcid=pmcid,
            text_element_id=te_id,
            verbatim=f.verbatim_support,
        ))
    return spans


def _normal_id(pmcid: str, subject: str | None, outcome: str | None,
               relation_type: RelationTypeEnum, te_ids: list[int]) -> str:
    raw = f"{subject}|{outcome}|{relation_type.value}|{'_'.join(map(str, sorted(te_ids)))}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"NF_{pmcid}_{digest}"


def _best_scope(findings: list[Finding]) -> FindingScope:
    """
    Return the scope from the finding with the highest grounding_score,
    falling back to the first finding if none have scores.
    """
    scored = [(f.grounding_score or 0.0, f) for f in findings]
    return max(scored, key=lambda x: x[0])[1].scope or FindingScope()


def _mean_score(findings: list[Finding]) -> float | None:
    scores = [f.grounding_score for f in findings if f.grounding_score is not None]
    return sum(scores) / len(scores) if scores else None


# ── Public API ─────────────────────────────────────────────────────────────────

class NormalizeStage:
    """
    NORMALIZE stage: entity normalization + conditional dedup.

    Parameters
    ----------
    extra_synonyms:
        Additional synonym mappings to merge with the built-in dictionary.
        Keys must be lowercase surface forms.
    """

    def __init__(self, extra_synonyms: dict[str, str] | None = None) -> None:
        self._synonyms = dict(_SYNONYMS)
        if extra_synonyms:
            self._synonyms.update(extra_synonyms)

    def normalize(self, findings: list[Finding], pmcid: str) -> list[NormalFinding]:
        """
        Normalize and deduplicate findings for one paper.

        Returns one NormalFinding per distinct (te_id, subject, outcome,
        direction) group, plus one per ungroupable finding (missing fields).
        """
        if not findings:
            return []

        # Step 1: normalize entity names in-place on a working copy
        normalized = self._normalize_entities(findings)

        # Step 2: partition into groupable and ungroupable
        groupable: dict[str, list[Finding]] = defaultdict(list)
        ungroupable: list[Finding] = []

        for f, te_id in normalized:
            key = _dedup_key(te_id, f.subject_entity, f.outcome_entity, f.relation_type)
            if key is None:
                ungroupable.append(f)
            else:
                groupable[key].append(f)

        results: list[NormalFinding] = []

        # Step 3: merge each groupable cluster
        for key, group in groupable.items():
            results.append(self._merge(group, pmcid))

        # Step 4: wrap each ungroupable finding individually
        for f in ungroupable:
            results.append(self._wrap_single(f, pmcid))

        logger.info(
            "[%s] NORMALIZE: %d findings → %d NormalFindings "
            "(%d grouped clusters, %d ungroupable)",
            pmcid, len(findings), len(results),
            len(groupable), len(ungroupable),
        )
        return results

    # ── Internals ──────────────────────────────────────────────────────────────

    def _normalize_entities(
        self, findings: list[Finding]
    ) -> list[tuple[Finding, int]]:
        """
        Return (finding_with_normalized_entities, text_element_id) pairs.
        text_element_id is extracted from the first evidence string; 0 if absent.
        """
        result = []
        for f in findings:
            # Extract te_id from first evidence string for grouping key
            te_id = 0
            if f.evidence:
                parts = f.evidence[0].split("|")
                if len(parts) == 3:
                    try:
                        te_id = int(parts[2])
                    except ValueError:
                        pass

            normed = f.model_copy(update={
                "subject_entity": self._norm(f.subject_entity),
                "outcome_entity": self._norm(f.outcome_entity),
            })
            result.append((normed, te_id))
        return result

    def _norm(self, name: str | None) -> str | None:
        if name is None:
            return None
        return self._synonyms.get(name.strip().lower(), name.strip())

    def _merge(self, group: list[Finding], pmcid: str) -> NormalFinding:
        # Representative finding: highest grounding_score
        rep = max(group, key=lambda f: f.grounding_score or 0.0)

        # Deduplicated spans (by sentence_id + te_id)
        seen: set[tuple[str, int]] = set()
        spans: list[SourceSpan] = []
        te_ids: list[int] = []
        for f in group:
            for span in _spans_from_finding(f):
                key = (span.sentence_id, span.text_element_id)
                if key not in seen:
                    seen.add(key)
                    spans.append(span)
                if span.text_element_id not in te_ids:
                    te_ids.append(span.text_element_id)

        unique_pmcids = sorted({s.pmcid for s in spans}) or [pmcid]

        return NormalFinding(
            normal_id=_normal_id(pmcid, rep.subject_entity, rep.outcome_entity,
                                 rep.relation_type, te_ids),
            subject_entity=rep.subject_entity,
            outcome_entity=rep.outcome_entity,
            relation_type=rep.relation_type,
            direction=rep.direction,
            category=rep.category,
            predicate_text=rep.claim,
            scope=rep.scope or FindingScope(),
            source_finding_ids=[],   # populated by Phase 3+ when finding_id exists
            evidence=spans,
            pmcids=unique_pmcids,
            mean_grounding_score=_mean_score(group),
        )

    def _wrap_single(self, f: Finding, pmcid: str) -> NormalFinding:
        spans = _spans_from_finding(f)
        te_ids = [s.text_element_id for s in spans]
        unique_pmcids = sorted({s.pmcid for s in spans}) or [pmcid]
        return NormalFinding(
            normal_id=_normal_id(pmcid, f.subject_entity, f.outcome_entity,
                                 f.relation_type, te_ids),
            subject_entity=f.subject_entity,
            outcome_entity=f.outcome_entity,
            relation_type=f.relation_type,
            direction=f.direction,
            category=f.category,
            predicate_text=f.claim,
            scope=f.scope or FindingScope(),
            source_finding_ids=[],
            evidence=spans,
            pmcids=unique_pmcids,
            mean_grounding_score=f.grounding_score,
        )
