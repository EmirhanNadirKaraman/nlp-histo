"""Tests for 'demographics' category addition to main pipeline models and validator.

No LLM or database required.
"""
from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from pipeline.stages.summarization.models import (
    AuditableSummary,
    AuditMetadata,
    Finding,
    FindingScope,
    NormalFinding,
    RelationTypeEnum,
    SourceSpan,
)
from pipeline.stages.summarization.current_stages.group_stage import _group_id
from pipeline.stages.summarization.routing.schema_validator import (
    SchemaValidator,
    _VALID_CATEGORIES,
)

PMCID = "PMC99999"


# ── Schema acceptance ─────────────────────────────────────────────────────────

def test_finding_demographics_valid():
    """category='demographics' passes Pydantic validation on Finding."""
    f = Finding(
        category="demographics",
        claim="Slight male predominance (37 men/29 women)",
        evidence=["S1|PMC123|456"],
        confidence="high",
        verbatim_support="slight male predominance (37 men/29 women)",
        relation_type=RelationTypeEnum.demographic,
    )
    assert f.category == "demographics"


def test_finding_unknown_category_rejected():
    """Unknown categories are still rejected by Pydantic."""
    with pytest.raises(ValidationError):
        Finding(
            category="bogus_category",
            claim="x",
            evidence=["S1|PMC123|456"],
            confidence="high",
            verbatim_support="x",
        )


def test_normal_finding_demographics_valid():
    """category='demographics' passes Pydantic validation on NormalFinding."""
    nf = NormalFinding(
        normal_id="NF_demo",
        subject_entity="CEAN patients",
        outcome_entity="sex distribution",
        relation_type=RelationTypeEnum.demographic,
        direction=None,
        category="demographics",
        predicate_text="Slight male predominance",
        scope=FindingScope(),
        source_finding_ids=[],
        evidence=[SourceSpan(sentence_id="S1", pmcid=PMCID, text_element_id=1, verbatim="x")],
        pmcids=[PMCID],
        mean_grounding_score=None,
    )
    assert nf.category == "demographics"


# ── SchemaValidator ───────────────────────────────────────────────────────────

def test_valid_categories_includes_demographics():
    """Regression guard: _VALID_CATEGORIES must contain 'demographics'."""
    assert "demographics" in _VALID_CATEGORIES


def test_schema_validator_accepts_demographics():
    """SchemaValidator passes a demographics finding without INVALID_CATEGORY."""
    from pipeline.stages.summarization.routing.models import ReasonCode

    summary = AuditableSummary(
        chunk_id="PMC123|0",
        findings=[
            Finding(
                category="demographics",
                claim="Male predominance in CEAN cohort",
                evidence=["S1|PMC123|0"],
                confidence="high",
                verbatim_support="male predominance",
                relation_type=RelationTypeEnum.demographic,
            )
        ],
        summary_text="",
        audit_metadata=AuditMetadata(
            sentences_analyzed=1,
            sentences_cited=["S1|PMC123|0"],
            pmcids_referenced=["PMC123"],
            uncited_sentences=[],
        ),
    )
    results = SchemaValidator().validate(summary)
    assert len(results) == 1
    assert ReasonCode.INVALID_CATEGORY not in results[0].reason_codes


# ── Grouping ──────────────────────────────────────────────────────────────────

def test_two_demographics_findings_same_group_id():
    """Same subject/outcome/relation_type/category='demographics' → same group_id."""
    id1 = _group_id("patients", "sex", "demographic", "demographics")
    id2 = _group_id("patients", "sex", "demographic", "demographics")
    assert id1 == id2


def test_demographics_vs_morphology_different_group_id():
    """Same subject/outcome/relation_type but different category → different group_id."""
    id_demo = _group_id("patients", "sex", "demographic", "demographics")
    id_morph = _group_id("patients", "sex", "demographic", "morphology")
    assert id_demo != id_morph


def test_demographics_prognosis_different_group_id():
    """demographics and prognosis with the same entities produce distinct buckets."""
    id_demo = _group_id("patients", "sex distribution", "demographic", "demographics")
    id_prog = _group_id("patients", "sex distribution", "demographic", "prognosis")
    assert id_demo != id_prog


# ── Alias repair (field_validator) ────────────────────────────────────────────

def test_finding_alias_repaired_via_constructor(caplog):
    """'demographic' alias is repaired to 'demographics' when constructing Finding directly."""
    with caplog.at_level(logging.WARNING, logger="pipeline.stages.summarization.models"):
        f = Finding(
            category="demographic",
            claim="Male predominance",
            evidence=["S1|PMC123|0"],
            confidence="high",
            verbatim_support="male predominance",
        )
    assert f.category == "demographics"
    assert "demographic" in caplog.text


def test_finding_alias_repaired_via_model_validate(caplog):
    """Alias repair fires on model_validate() — the path used by batch/dispatch.py:100."""
    raw_json = json.dumps({
        "chunk_id": "PMC123|0",
        "findings": [
            {
                "category": "demographic",
                "claim": "Male predominance",
                "evidence": ["S1|PMC123|0"],
                "confidence": "high",
                "verbatim_support": "male predominance",
            }
        ],
        "summary_text": "",
        "audit_metadata": {
            "sentences_analyzed": 1,
            "sentences_cited": ["S1|PMC123|0"],
            "pmcids_referenced": ["PMC123"],
            "uncited_sentences": [],
        },
    })
    with caplog.at_level(logging.WARNING, logger="pipeline.stages.summarization.models"):
        summary = AuditableSummary.model_validate(json.loads(raw_json))
    assert summary.findings[0].category == "demographics"
    assert "demographic" in caplog.text


def test_finding_category_demographics_with_relation_type_demographic():
    """category='demographics' and relation_type=RelationTypeEnum.demographic can co-exist.

    The trailing-s asymmetry is intentional: category uses the plural form, while
    RelationTypeEnum.demographic uses the singular (and is not repaired).
    """
    f = Finding(
        category="demographics",
        claim="Slight male predominance",
        evidence=["S1|PMC123|0"],
        confidence="high",
        verbatim_support="slight male predominance",
        relation_type=RelationTypeEnum.demographic,
    )
    assert f.category == "demographics"
    assert f.relation_type == RelationTypeEnum.demographic


def test_alias_repair_does_not_touch_unknown_categories():
    """Unknown categories are not coerced — they pass through and Pydantic rejects them."""
    with pytest.raises(ValidationError):
        Finding(
            category="bogus_category",
            claim="x",
            evidence=["S1|PMC123|0"],
            confidence="high",
            verbatim_support="x",
        )
