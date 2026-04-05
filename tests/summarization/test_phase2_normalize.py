"""
Phase 2 smoke tests — NormalizeStage: entity normalization + conditional dedup.
No LLM or NLI model required.
"""
import pytest

from pipeline.stages.summarization.models import (
    DirectionEnum,
    Finding,
    FindingScope,
    NormalFinding,
    RelationTypeEnum,
    SourceSpan,
)
from pipeline.stages.summarization.normalize_stage import NormalizeStage, normalize_entity


PMCID = "PMC12345"


def _finding(
    claim: str = "CD30 predicts worse OS",
    subject: str | None = "CD30",
    outcome: str | None = "OS",
    relation_type: RelationTypeEnum = RelationTypeEnum.prognostic,
    direction: DirectionEnum | None = DirectionEnum.negative,
    category: str = "IHC",
    evidence: list[str] | None = None,  # None → default span; [] → explicitly empty
    verbatim: str = "CD30 was associated with worse OS",
    grounding_score: float | None = 0.8,
    scope: FindingScope | None = None,
) -> Finding:
    return Finding(
        category=category,
        claim=claim,
        evidence=[f"S1|{PMCID}|101"] if evidence is None else evidence,
        confidence="high",
        verbatim_support=verbatim,
        subject_entity=subject,
        outcome_entity=outcome,
        relation_type=relation_type,
        direction=direction,
        scope=scope or FindingScope(disease_subtype="DLBCL", scope_parsed=True),
        grounding_score=grounding_score,
    )


# ── normalize_entity ───────────────────────────────────────────────────────────

def test_normalize_entity_known_synonym():
    assert normalize_entity("overall survival") == "OS"
    assert normalize_entity("Overall Survival") == "OS"
    assert normalize_entity("CD30 expression") == "CD30"
    assert normalize_entity("bcl-2") == "BCL2"
    assert normalize_entity("Ki67") == "Ki-67"
    assert normalize_entity("r-chop") == "R-CHOP"


def test_normalize_entity_unknown_passes_through():
    assert normalize_entity("some novel marker") == "some novel marker"


def test_normalize_entity_none_returns_none():
    assert normalize_entity(None) is None


def test_normalize_entity_strips_whitespace():
    assert normalize_entity("  CD30  ") == "CD30"


# ── NormalizeStage: no dedup needed ───────────────────────────────────────────

def test_single_finding_passes_through():
    stage = NormalizeStage()
    f = _finding()
    result = stage.normalize([f], PMCID)
    assert len(result) == 1
    assert result[0].subject_entity == "CD30"
    assert result[0].outcome_entity == "OS"
    assert result[0].direction == DirectionEnum.negative


def test_entity_normalization_applied():
    stage = NormalizeStage()
    f = _finding(subject="CD30 expression", outcome="overall survival")
    result = stage.normalize([f], PMCID)
    assert result[0].subject_entity == "CD30"
    assert result[0].outcome_entity == "OS"


def test_extra_synonyms_respected():
    stage = NormalizeStage(extra_synonyms={"custom marker": "CUSTOM"})
    f = _finding(subject="custom marker")
    result = stage.normalize([f], PMCID)
    assert result[0].subject_entity == "CUSTOM"


# ── Conditional dedup ─────────────────────────────────────────────────────────

def test_same_claim_same_te_merged():
    """Three voters extracting the same claim from the same TE → one NormalFinding."""
    stage = NormalizeStage()
    findings = [
        _finding(evidence=[f"S1|{PMCID}|101"], grounding_score=0.7),
        _finding(evidence=[f"S1|{PMCID}|101"], grounding_score=0.9),
        _finding(evidence=[f"S1|{PMCID}|101"], grounding_score=0.8),
    ]
    result = stage.normalize(findings, PMCID)
    assert len(result) == 1
    # Representative should be the highest-grounded one
    assert result[0].predicate_text == findings[0].claim


def test_same_te_different_claims_not_merged():
    """Two distinct claims from the same TE stay as separate NormalFindings."""
    stage = NormalizeStage()
    f1 = _finding(claim="CD30 predicts worse OS", outcome="OS",
                  direction=DirectionEnum.negative, evidence=[f"S1|{PMCID}|101"])
    f2 = _finding(claim="CD30 predicts no R-CHOP response", outcome="R-CHOP response",
                  direction=DirectionEnum.absent, evidence=[f"S2|{PMCID}|101"])
    result = stage.normalize([f1, f2], PMCID)
    assert len(result) == 2


def test_different_te_same_claim_not_merged():
    """Same claim from two different TEs stays separate — different source spans."""
    stage = NormalizeStage()
    f1 = _finding(evidence=[f"S1|{PMCID}|101"])
    f2 = _finding(evidence=[f"S3|{PMCID}|102"])
    result = stage.normalize([f1, f2], PMCID)
    assert len(result) == 2


def test_merged_evidence_deduped():
    """Multiple findings with the same sentence_id + te_id produce one span."""
    stage = NormalizeStage()
    findings = [_finding(evidence=[f"S1|{PMCID}|101"]) for _ in range(3)]
    result = stage.normalize(findings, PMCID)
    assert len(result[0].evidence) == 1


def test_merged_mean_grounding_score():
    stage = NormalizeStage()
    findings = [
        _finding(evidence=[f"S1|{PMCID}|101"], grounding_score=0.6),
        _finding(evidence=[f"S1|{PMCID}|101"], grounding_score=0.8),
    ]
    result = stage.normalize(findings, PMCID)
    assert result[0].mean_grounding_score == pytest.approx(0.7)


# ── Ungroupable findings (None fields) ────────────────────────────────────────

def test_none_subject_kept_ungrouped():
    stage = NormalizeStage()
    f = _finding(subject=None)
    result = stage.normalize([f], PMCID)
    assert len(result) == 1
    assert result[0].subject_entity is None


def test_unclear_relation_type_kept_ungrouped():
    """Finding with unclear relation_type is not deduped — kept as individual NormalFinding."""
    stage = NormalizeStage()
    f = _finding(relation_type=RelationTypeEnum.unclear)
    result = stage.normalize([f], PMCID)
    assert len(result) == 1
    assert result[0].relation_type == RelationTypeEnum.unclear


def test_none_fields_not_merged_with_each_other():
    """Two ungroupable findings (both None subject) stay as two NormalFindings."""
    stage = NormalizeStage()
    f1 = _finding(subject=None, claim="Finding A", evidence=[f"S1|{PMCID}|101"])
    f2 = _finding(subject=None, claim="Finding B", evidence=[f"S2|{PMCID}|102"])
    result = stage.normalize([f1, f2], PMCID)
    assert len(result) == 2


# ── Source span parsing ───────────────────────────────────────────────────────

def test_source_spans_parsed_correctly():
    stage = NormalizeStage()
    f = _finding(evidence=[f"S3|{PMCID}|999"])
    result = stage.normalize([f], PMCID)
    assert len(result[0].evidence) == 1
    span = result[0].evidence[0]
    assert span.sentence_id == "S3"
    assert span.pmcid == PMCID
    assert span.text_element_id == 999


def test_malformed_evidence_string_skipped():
    stage = NormalizeStage()
    f = _finding(evidence=["BADFORMAT", f"S1|{PMCID}|101"])
    result = stage.normalize([f], PMCID)
    # Malformed entry skipped; valid one kept
    assert len(result[0].evidence) == 1
    assert result[0].evidence[0].sentence_id == "S1"


def test_no_evidence_produces_no_spans():
    stage = NormalizeStage()
    f = _finding(evidence=[], subject=None)  # ungroupable so won't be merged
    result = stage.normalize([f], PMCID)
    assert result[0].evidence == []


# ── Scope and pmcids ──────────────────────────────────────────────────────────

def test_scope_from_best_grounded_finding():
    stage = NormalizeStage()
    f1 = _finding(evidence=[f"S1|{PMCID}|101"], grounding_score=0.3,
                  scope=FindingScope(disease_subtype="DLBCL", scope_parsed=True))
    f2 = _finding(evidence=[f"S1|{PMCID}|101"], grounding_score=0.9,
                  scope=FindingScope(disease_subtype="GCB-DLBCL", scope_parsed=True))
    result = stage.normalize([f1, f2], PMCID)
    assert result[0].scope.disease_subtype == "GCB-DLBCL"


def test_pmcids_populated():
    stage = NormalizeStage()
    f = _finding(evidence=[f"S1|PMC99|101"])
    result = stage.normalize([f], PMCID)
    assert "PMC99" in result[0].pmcids


def test_empty_input_returns_empty():
    stage = NormalizeStage()
    assert stage.normalize([], PMCID) == []


# ── NormalFinding model ───────────────────────────────────────────────────────

def test_normal_finding_roundtrip():
    stage = NormalizeStage()
    f = _finding()
    nf = stage.normalize([f], PMCID)[0]
    data = nf.model_dump()
    nf2 = NormalFinding(**data)
    assert nf2.normal_id == nf.normal_id
    assert nf2.direction == DirectionEnum.negative
