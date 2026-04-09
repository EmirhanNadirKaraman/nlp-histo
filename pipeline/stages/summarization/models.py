"""
Pydantic schemas for all pipeline stages:
  MAP → REDUCE → RULES → NORMALIZE → GROUP → CANONICALIZE → RELATE → RESOLVE
"""
from __future__ import annotations

from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field


# ── Phase 1: new enums and scope model ────────────────────────────────────────

class DirectionEnum(str, Enum):
    positive = "positive"
    negative = "negative"
    absent   = "absent"
    partial  = "partial"
    unclear  = "unclear"


class RelationTypeEnum(str, Enum):
    has_feature        = "has_feature"        # entity exhibits / shows / has a histological feature
    expression         = "expression"         # biomarker/protein expression level
    prognostic         = "prognostic"         # entity predicts or associates with a clinical outcome
    comparative        = "comparative"        # relative frequency or likelihood vs another group
    demographic        = "demographic"        # age, sex, prevalence in a population
    treatment_response = "treatment_response" # response to a treatment regimen
    unclear            = "unclear"            # relation type genuinely not inferable


class AssertionStatusEnum(str, Enum):
    positive  = "positive"   # claim asserts presence, expression, or positive association
    negative  = "negative"   # claim asserts absence, non-expression, or negative finding
    uncertain = "uncertain"  # polarity not determinable from surface form


class FindingScope(BaseModel):
    disease_subtype:   str | None = None
    cohort_n:          int | None = None
    assay_method:      str | None = None
    biomarker_cutoff:  str | None = None
    tissue_site:       str | None = None
    treatment_context: str | None = None
    endpoint:          str | None = None
    study_design:      str | None = None
    scope_parsed:      bool       = False


# ── MAP output ─────────────────────────────────────────────────────────────────

class Finding(BaseModel):
    category: Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis"]
    claim: str = Field(description="Atomic, telegraphic medical fact")
    evidence: List[str] = Field(description="Citation IDs, e.g. ['S1|PMC123|456']")
    confidence: Literal["high", "medium", "low"]
    verbatim_support: str = Field(description="Exact quote from source text")
    # ── Phase 1 additions (optional; populated by updated MAP prompt) ──────────
    subject_entity:    str | None               = None
    outcome_entity:    str | None               = None
    relation_type:     RelationTypeEnum         = RelationTypeEnum.unclear
    direction:         DirectionEnum | None     = None
    assertion_status:  AssertionStatusEnum      = AssertionStatusEnum.uncertain
    scope:             FindingScope | None      = None
    grounding_score:   float | None             = None


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


# ── Phase 2: NORMALIZE output ─────────────────────────────────────────────────

class SourceSpan(BaseModel):
    """Provenance pointer to a single sentence in the source document."""
    sentence_id:     str        # e.g. "S3"
    pmcid:           str
    text_element_id: int
    verbatim:        str


class NormalFinding(BaseModel):
    """
    Output of the NORMALIZE stage.

    One NormalFinding represents a single distinct claim (subject, outcome,
    direction) after deduplication across voters and entity normalization.
    Multiple source spans are merged when they share the same claim and source
    sentence; different claims from the same text element remain separate.
    """
    normal_id:            str               # "NF_{pmcid}_{sha8(subject+outcome+relation_type+te_ids)}"
    subject_entity:       str | None        # normalized; None if MAP did not extract
    outcome_entity:       str | None        # normalized; None if MAP did not extract
    relation_type:        RelationTypeEnum  # groupability key; unclear = non-groupable
    direction:            DirectionEnum | None
    assertion_status:     AssertionStatusEnum  # deterministic; inferred from claim text
    category:             Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis"]
    predicate_text:       str               # representative claim text
    scope:                FindingScope      # from the best-grounded source finding
    source_finding_ids:   List[str]         # Finding.finding_id values merged here (reserved for Phase 3+)
    evidence:             List[SourceSpan]  # deduplicated source spans
    pmcids:               List[str]         # unique PMCIDs
    mean_grounding_score: float | None      # mean of grounding_score across source findings


# ── Phase 3: GROUP output ─────────────────────────────────────────────────────

class FindingGroup(BaseModel):
    """
    Output of the GROUP stage.

    Contains only groupable NormalFindings — those where subject_entity,
    outcome_entity are non-None and relation_type is not unclear.
    Non-groupable findings are excluded before GROUP runs and remain in
    the normalized-findings cache.

    All NormalFindings about the same (subject, outcome, relation_type) are
    collected here regardless of direction.  Mixed directions within a group
    are intentional — contradictions surface as relations in Phase 5 RELATE,
    not as separate groups.
    """
    group_id:             str                    # "GRP_{sha8(subject)}_{sha8(outcome)}_{relation_type}_{sha8(category)}"
    subject_entity:       str                    # always non-None (groupability invariant)
    outcome_entity:       str                    # always non-None (groupability invariant)
    relation_type:        RelationTypeEnum       # grouping key
    category:             Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis"]
    member_ids:           List[str]              # NormalFinding.normal_id values
    direction_counts:     dict[str, int]         # direction.value → count (None keys stored as "unclear")
    scope_heterogeneity:  float                  # 0.0 = all scope fields agree; 1.0 = maximum variation


# ── Phase 2 forward-looking model (not wired in Phase 1) ──────────────────────

class AtomicFinding(BaseModel):
    """
    Future Phase 2 runtime object. Defined here for schema stability.
    Not imported or used anywhere in Phase 1 runner/filter code.
    """
    finding_id:       str
    category:         Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis"]
    claim:            str
    subject_entity:   str | None
    outcome_entity:   str | None
    direction:        DirectionEnum | None
    scope:            FindingScope        # required (not Optional) — Phase 2 invariant
    confidence:       Literal["high", "medium", "low"]
    evidence:         List[str]
    verbatim_support: str
    grounding_score:  float | None


# ── Phase 4: CANONICALIZE output ──────────────────────────────────────────────

class CanonicalScopeEnum(str, Enum):
    single_study  = "single_study"   # only one PMCID supports this rule
    multi_study   = "multi_study"    # ≥2 distinct PMCIDs support this rule
    conflicted    = "conflicted"     # ≥2 opposing directions within the group
    unknown       = "unknown"        # no PMCID information available


class CanonicalRule(BaseModel):
    """
    Output of the CANONICALIZE stage for one direction-bin of a FindingGroup.

    One CanonicalRule represents the best-grounded, direction-consistent
    predicate text that can be generalised across member NormalFindings.
    """
    canonical_id:        str                   # "CR_{sha8(group_id)}_{direction}"
    group_id:            str                   # source FindingGroup.group_id
    subject_entity:      str
    outcome_entity:      str
    relation_type:       RelationTypeEnum
    direction:           DirectionEnum | None  # dominant direction for this bin
    predicate_text:      str                   # LLM-selected or best-score fallback
    canonical_scope:     CanonicalScopeEnum
    category:            Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis"]
    supporting_pmcids:   List[str]             # unique PMCIDs across member NFs
    member_normal_ids:   List[str]             # NormalFinding.normal_id values in this bin
    mean_grounding_score: float | None
    finding_count:       int                   # number of NormalFindings in this bin


# ── Phase 5: RELATE output ────────────────────────────────────────────────────

class RelationTypeLabel(str, Enum):
    SUPPORT        = "SUPPORT"
    CONTRADICT     = "CONTRADICT"
    SCOPE_QUALIFY  = "SCOPE_QUALIFY"
    UNRELATED      = "UNRELATED"


class Relation(BaseModel):
    """
    Output of the RELATE stage: a directional NLI-derived relation between
    two CanonicalRules.

    rule_id_a and rule_id_b are symmetric for SUPPORT; for SCOPE_QUALIFY the
    rule with higher entailment score is rule_id_a (broader claim).
    """
    rule_id_a:       str
    rule_id_b:       str
    relation_type:   RelationTypeLabel
    nli_score_a_to_b: float   # entailment score from A→B direction
    nli_score_b_to_a: float   # entailment score from B→A direction


# ── Phase 6: RESOLVE output ───────────────────────────────────────────────────

class FinalRule(BaseModel):
    """
    Output of the RESOLVE stage: a scored, ranked canonical rule.

    Inherits all fields from CanonicalRule and adds scoring metadata.
    """
    final_id:              str                   # "FR_{canonical_id}"
    canonical_id:          str
    group_id:              str
    subject_entity:        str
    outcome_entity:        str
    relation_type:         RelationTypeEnum
    direction:             DirectionEnum | None
    predicate_text:        str
    canonical_scope:       CanonicalScopeEnum
    category:              Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis"]
    supporting_pmcids:     List[str]
    member_normal_ids:     List[str]
    mean_grounding_score:  float | None
    finding_count:         int
    # ── Scoring ──────────────────────────────────────────────────────────────
    final_score:           float                 # 0–1, higher is better
    support_count:         int                   # SUPPORT relations touching this rule
    contradict_count:      int                   # CONTRADICT relations touching this rule
    scope_qualify_count:   int                   # SCOPE_QUALIFY relations touching this rule
    is_contradicted:       bool
    contradicted_by:       List[str]             # canonical_ids that contradict this rule


# ── Corpus-level RELATE output ────────────────────────────────────────────────

class CorpusRelation(Relation):
    """
    Output of CorpusRelateStage: a pairwise relation from the full pooled corpus.

    Extends Relation with provenance (pmcid_a/b, comparison_scope), human-readable
    claim text, and grounding metadata.

    comparison_scope distinguishes intra-paper edges (both rules from the same
    paper's processing run) from cross-paper edges (rules from different papers).
    NLI methodology is identical for both; the label is purely informational.

    Per-paper JSON `relations` remain the authoritative input for ResolveStage
    scoring.  This model is for the analytical corpus graph only.
    """
    relation_id:              str
    comparison_scope:         Literal["intra_paper", "cross_paper"]
    same_paper:               bool
    pmcid_a:                  str
    pmcid_b:                  str
    subject_entity:           str
    outcome_entity:           str
    category:                 str
    relation_type_structural: str                # has_feature / expression / etc.
    direction_a:              str | None
    direction_b:              str | None
    predicate_a:              str
    predicate_b:              str
    mean_grounding_a:         float | None
    mean_grounding_b:         float | None
    finding_count_a:          int
    finding_count_b:          int
    supporting_pmcids_a:      List[str]
    supporting_pmcids_b:      List[str]
    canonical_scope_a:        str
    canonical_scope_b:        str
    # Scope qualifier check — "scope_unknown" until FindingScope is carried on
    # CanonicalRule (v2 work).  Values: scope_unknown | scope_compatible |
    # scope_mismatch | scope_qualified.
    scope_check_result:       str = "scope_unknown"
    scope_note:               str = ""
