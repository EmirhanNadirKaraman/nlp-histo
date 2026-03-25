"""
Pydantic schemas for the three pipeline stages: MAP → REDUCE → RULES.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


# ── MAP output ─────────────────────────────────────────────────────────────────

class Finding(BaseModel):
    category: Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis"]
    claim: str = Field(description="Atomic, telegraphic medical fact")
    evidence: List[str] = Field(description="Citation IDs, e.g. ['S1|PMC123|456']")
    confidence: Literal["high", "medium", "low"]
    verbatim_support: str = Field(description="Exact quote from source text")


class AuditMetadata(BaseModel):
    sentences_analyzed: int
    sentences_cited: List[str]
    pmcids_referenced: List[str]
    uncited_sentences: List[str]


class AuditableSummary(BaseModel):
    """Output of the MAP stage for a single chunk."""
    chunk_id: str
    findings: List[Finding]
    summary_text: str
    audit_metadata: AuditMetadata


# ── REDUCE output ──────────────────────────────────────────────────────────────

class SourceReference(BaseModel):
    sentences: List[str] = Field(description="Citation IDs, e.g. ['S1|PMC123|456']")
    verbatim: str


class SectionFinding(BaseModel):
    claim: str
    sources: List[SourceReference]
    strength: Literal["strong", "moderate", "weak"]


class Section(BaseModel):
    findings: List[SectionFinding]


class SectionsContainer(BaseModel):
    clinical_significance: Section
    histopathological_features: Section
    management_outcomes: Section
    risk_factors_associations: Section


class EvidenceConflict(BaseModel):
    topic: str
    conflicting_sources: List[str]


class ReduceAuditTrail(BaseModel):
    chunks_processed: int
    total_sentences_cited: int
    unique_pmcids: List[str]
    unique_text_element_ids: List[int]
    evidence_conflicts: List[EvidenceConflict]


class ConsolidatedSummary(BaseModel):
    """Output of the REDUCE stage."""
    pmcid: str
    sections: SectionsContainer
    narrative_summary: str
    audit_trail: ReduceAuditTrail


# ── RULE output ────────────────────────────────────────────────────────────────

class EvidenceChainItem(BaseModel):
    sentence_id: str
    pmcid: str
    text_element_id: int
    verbatim: str


class Rule(BaseModel):
    rule_id: str
    type: Literal["Diagnostic", "Prognostic", "Management"]
    condition: str = Field(description="IF <observation>")
    action: str = Field(description="THEN <conclusion>")
    confidence: Literal["High", "Medium", "Low"]
    evidence_chain: List[EvidenceChainItem]
    contraindications: List[str]


class RuleCounts(BaseModel):
    Diagnostic: int
    Prognostic: int
    Management: int


class RuleAuditSummary(BaseModel):
    total_rules: int
    rules_by_type: RuleCounts
    pmcids_supporting_rules: List[str]
    average_evidence_per_rule: float


class ExtractedRules(BaseModel):
    """Output of the RULE EXTRACTION stage."""
    rules: List[Rule]
    audit_summary: RuleAuditSummary


# ── CONTRADICTION DETECTION output ─────────────────────────────────────────────

class ContradictingPair(BaseModel):
    rule_id_a: str
    rule_id_b: str
    rule_a: Rule
    rule_b: Rule
    explanation: str


class ContradictionReport(BaseModel):
    """Output of the contradiction detection stage."""
    contradicting_pairs: List[ContradictingPair]
    non_contradicting_rules: List[Rule]


# ── LLM judge intermediate ─────────────────────────────────────────────────────

class PairJudgment(BaseModel):
    contradicts: bool
    explanation: str
