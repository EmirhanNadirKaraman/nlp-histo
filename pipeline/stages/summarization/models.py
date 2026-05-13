"""
Pydantic schemas for all pipeline stages:
  MAP → REDUCE → RULES → NORMALIZE → GROUP → CANONICALIZE → RELATE → RESOLVE
"""
from __future__ import annotations

from enum import Enum
from typing import List, Literal

import logging

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from .enum_logging import log_bad_finding, log_enum_observation

_log = logging.getLogger(__name__)


# ── Schema / prompt versioning ────────────────────────────────────────────────
# Bump these whenever the MAP output schema or prompt changes in a
# behaviour-affecting way. They are included in cache keys / run metadata so
# stale outputs don't collide with new ones.
MAP_SCHEMA_VERSION: str = "map_v1_explicit_direction"
MAP_PROMPT_VERSION: str = "map_prompt_v1_explicit_enums"
MAP_STAGE_NAME:     str = "map"


# ── Cascade signature helper ──────────────────────────────────────────────────
def compute_finding_id(
    pmcid: str,
    chunk_id: str,
    position_in_chunk: int,
    claim: str,
) -> str:
    """Stable 12-char hash identifying a MAP Finding.

    Deterministic over (pmcid, chunk_id, position_in_chunk, normalized claim).
    Whitespace in the claim is collapsed and case-folded so cosmetic
    re-serialisation does not change the id.
    """
    import hashlib
    normalized_claim = " ".join((claim or "").lower().split())
    payload = f"{pmcid}|{chunk_id}|{position_in_chunk}|{normalized_claim}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def compute_cascade_signature(voter_specs: list[tuple[str, str]]) -> str:
    """Stable short hash of an ordered (provider, model) sequence.

    Used in MAP cache keys and run-artifact metadata so the same chunk under a
    different cascade configuration is considered a cache miss.
    """
    import hashlib
    import json
    payload = json.dumps([list(t) for t in voter_specs], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class MapRunMetadata(BaseModel):
    """Provenance attached to every MAP cache entry and run artifact.

    The fields are deliberately strings so they can round-trip through JSON
    without any custom (de)serialisers. ``cascade_profile`` is the human-
    readable profile name (``default`` / ``smoke_haiku`` / ``dev_sonnet`` /
    ``custom``); ``cascade_signature`` is the deterministic content hash that
    actually drives cache invalidation.

    ``provider`` and ``model`` identify the producer of the *stored* result —
    in the cascade this is the voter or escalation model whose output was
    selected, not the cascade as a whole.
    """
    schema_version:    str = MAP_SCHEMA_VERSION
    prompt_version:    str = MAP_PROMPT_VERSION
    stage_name:        str = MAP_STAGE_NAME
    provider:          str
    model:             str
    cascade_profile:   str = "custom"
    cascade_signature: str


# ── Phase 1: new enums and scope model ────────────────────────────────────────

class DirectionEnum(str, Enum):
    positive      = "positive"
    negative      = "negative"
    absent        = "absent"
    partial       = "partial"
    unclear       = "unclear"
    no_direction  = "no_direction"


class RelationTypeEnum(str, Enum):
    has_feature        = "has_feature"        # entity exhibits / shows / has a histological feature
    expression         = "expression"         # biomarker/protein expression level
    prognostic         = "prognostic"         # entity predicts or associates with a clinical outcome
    comparative        = "comparative"        # relative frequency or likelihood vs another group
    demographic        = "demographic"        # age, sex, prevalence in a population
    treatment_response = "treatment_response" # response to a treatment regimen
    unclear            = "unclear"            # relation type genuinely not inferable



_SCOPE_FIELDS_DEFAULTS: dict[str, object] = {
    "disease_subtype":   None,
    "cohort_n":          None,
    "assay_method":      None,
    "biomarker_cutoff":  None,
    "tissue_site":       None,
    "treatment_context": None,
    "endpoint":          None,
    "study_design":      None,
    "scope_parsed":      False,
}


class FindingScope(BaseModel):
    # No JSON-schema defaults — every sub-field is required when scope is
    # non-null so the OpenAI strict schema is satisfied. The model is
    # instructed to emit null for sub-fields it cannot determine.
    #
    # Internal Python callers may still construct an empty scope via
    # ``FindingScope.empty()``; legacy cached payloads with missing keys are
    # filled in by ``_fill_legacy_defaults`` below.
    disease_subtype:   str | None
    cohort_n:          int | None
    assay_method:      str | None
    biomarker_cutoff:  str | None
    tissue_site:       str | None
    treatment_context: str | None
    endpoint:          str | None
    study_design:      str | None
    scope_parsed:      bool

    @classmethod
    def empty(cls) -> "FindingScope":
        return cls(**_SCOPE_FIELDS_DEFAULTS)

    @model_validator(mode="before")
    @classmethod
    def _fill_legacy_defaults(cls, data: object) -> object:
        """Tolerate cached payloads / internal callers that omit sub-fields."""
        if isinstance(data, dict):
            for key, default in _SCOPE_FIELDS_DEFAULTS.items():
                if key not in data:
                    data[key] = default
        return data

    @field_validator(
        "disease_subtype", "assay_method", "biomarker_cutoff",
        "tissue_site", "treatment_context", "endpoint", "study_design",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, v: object) -> object:
        if v == "null" or v == "":
            return None
        return v


# ── MAP output ─────────────────────────────────────────────────────────────────

_CATEGORY_ALIASES: dict[str, str] = {
    "demographic": "demographics",  # LLM alias; category uses trailing 's', relation_type does not
}


class Finding(BaseModel):
    # NOTE: relation_type uses "demographic" (no 's'); category uses "demographics" (with 's').
    # Some models emit the alias — _repair_category_alias normalises it before Pydantic validates.
    #
    # Strict-schema invariant: no field has a Pydantic default. Optional fields
    # are typed as `X | None` so Pydantic emits anyOf:[X,null] and the OpenAI
    # strict schema can list every property in `required`.
    category: Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographics"]
    claim: str = Field(description="Atomic, telegraphic medical fact")
    evidence: List[str] = Field(description="Citation IDs, e.g. ['S1|PMC123|456']")
    confidence: Literal["high", "medium", "low"]
    verbatim_support: str = Field(description="Exact quote from source text")
    subject_entity:    str | None
    outcome_entity:    str | None
    relation_type:     RelationTypeEnum
    direction:         DirectionEnum
    scope:             FindingScope | None
    grounding_score:   float | None

    # Pipeline-assigned stable identifier. Kept as a PrivateAttr so it never
    # appears in the OpenAI strict schema generated from this model, and so
    # cached LLM outputs that omit it parse cleanly. The runner fills it via
    # `set_finding_id` after MAP completes; persistence reads it via the
    # `finding_id` property.
    _finding_id: str | None = PrivateAttr(default=None)

    @property
    def finding_id(self) -> str | None:
        return self._finding_id

    def set_finding_id(self, value: str) -> None:
        self._finding_id = value

    @model_validator(mode="before")
    @classmethod
    def _fill_legacy_defaults(cls, data: object) -> object:
        """Tolerate cached payloads that omit Phase-1 optional keys."""
        if isinstance(data, dict):
            data.setdefault("subject_entity", None)
            data.setdefault("outcome_entity", None)
            data.setdefault("scope", None)
            data.setdefault("grounding_score", None)
            data.setdefault("relation_type", "unclear")
            data.setdefault("direction", "no_direction")
        return data

    @field_validator("category", mode="before")
    @classmethod
    def _repair_category_alias(cls, v: object) -> object:
        if isinstance(v, str) and v in _CATEGORY_ALIASES:
            repaired = _CATEGORY_ALIASES[v]
            _log.warning("Repaired category alias %r → %r", v, repaired)
            log_enum_observation(
                field_name="category",
                raw_value=v,
                valid_values=["morphology", "IHC", "molecular_genetics",
                              "staging", "treatment", "prognosis", "demographics"],
                coerced_value=repaired,
                reason="alias_repair",
            )
            return repaired
        return v

    @field_validator("relation_type", mode="before")
    @classmethod
    def _coerce_invalid_relation_type(cls, v: object) -> object:
        valid = [e.value for e in RelationTypeEnum]
        if v is None or v == "null" or v == "":
            _log.warning("Missing relation_type — coercing to 'unclear'")
            log_enum_observation(
                field_name="relation_type", raw_value=v, valid_values=valid,
                coerced_value=RelationTypeEnum.unclear.value, reason="missing",
            )
            return RelationTypeEnum.unclear
        if isinstance(v, str) and v not in set(valid):
            _log.warning("Unknown relation_type %r — coercing to 'unclear'", v)
            log_enum_observation(
                field_name="relation_type", raw_value=v, valid_values=valid,
                coerced_value=RelationTypeEnum.unclear.value, reason="unknown_value",
            )
            return RelationTypeEnum.unclear
        return v

    @field_validator("direction", mode="before")
    @classmethod
    def _coerce_invalid_direction(cls, v: object) -> object:
        valid = [e.value for e in DirectionEnum]
        # Old caches and lenient producers may emit null/""/"null" — those mean
        # "direction does not apply": coerce to no_direction.
        if v is None or v == "null" or v == "":
            log_enum_observation(
                field_name="direction", raw_value=v, valid_values=valid,
                coerced_value=DirectionEnum.no_direction.value,
                reason="null_to_no_direction",
            )
            return DirectionEnum.no_direction
        if isinstance(v, str) and v not in set(valid):
            _log.warning("Unknown direction %r — coercing to 'unclear'", v)
            log_enum_observation(
                field_name="direction", raw_value=v, valid_values=valid,
                coerced_value=DirectionEnum.unclear.value, reason="unknown_value",
            )
            return DirectionEnum.unclear
        return v

    @field_validator("scope", "subject_entity", "outcome_entity",
                     "grounding_score", mode="before")
    @classmethod
    def _null_string_to_none(cls, v: object) -> object:
        return None if v == "null" else v


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

    @field_validator("findings", mode="before")
    @classmethod
    def _drop_invalid_findings(cls, v: object) -> object:
        if not isinstance(v, list):
            return v
        clean = []
        for item in v:
            try:
                Finding.model_validate(item)
                clean.append(item)
            except Exception as exc:
                _log.warning("Dropping malformed finding: %s", exc)
                log_bad_finding(raw=item, error=str(exc))
        return clean


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
    subject_cui:          str | None = None  # UMLS CUI resolved during NORMALIZE
    outcome_cui:          str | None = None  # UMLS CUI resolved during NORMALIZE
    relation_type:        RelationTypeEnum  # groupability key; unclear = non-groupable
    direction:            DirectionEnum | None
    category:             Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographics"]
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
    group_id:             str                    # "GRP_{sha8(pmcid)}_{sha8(subject)}_{sha8(outcome)}_{relation_type}_{sha8(category)}"
    subject_entity:       str                    # always non-None (groupability invariant)
    outcome_entity:       str                    # always non-None (groupability invariant)
    relation_type:        RelationTypeEnum       # grouping key
    category:             Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographics"]
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
    category:         Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographics"]
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
    is_conflicted:       bool                  # True if ≥2 non-unclear opposing directions in bin
    study_coverage:      Literal["single_study", "multi_study", "unknown"]
    category:            Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographics"]
    supporting_pmcids:   List[str]             # unique PMCIDs across member NFs
    member_normal_ids:   List[str]             # NormalFinding.normal_id values in this bin
    mean_grounding_score: float | None
    finding_count:       int                   # number of NormalFindings in this bin
    subject_cui:         str | None = None     # UMLS CUI for subject_entity (set post-canonicalize)
    outcome_cui:         str | None = None     # UMLS CUI for outcome_entity (set post-canonicalize)


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


class RawNLIPair(BaseModel):
    """
    Raw NLI scores for one eligible pair from the RELATE stage, including
    pairs classified as UNRELATED.  Saved alongside Relation[] so that
    entailment/contradiction thresholds can be swept offline without
    re-running the NLI model.
    """
    rule_id_a:   str
    rule_id_b:   str
    ent_a_to_b:  float
    ent_b_to_a:  float
    con_a_to_b:  float
    con_b_to_a:  float
    classified_label: RelationTypeLabel   # UNRELATED when neither threshold fired
    pmcid_a:     str = ""                 # source paper for rule_id_a (corpus-level only)
    pmcid_b:     str = ""                 # source paper for rule_id_b (corpus-level only)


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
    is_conflicted:         bool
    study_coverage:        Literal["single_study", "multi_study", "unknown"]
    category:              Literal["morphology", "IHC", "molecular_genetics", "staging", "treatment", "prognosis", "demographics"]
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
    is_conflicted_a:          bool
    study_coverage_a:         Literal["single_study", "multi_study", "unknown"]
    is_conflicted_b:          bool
    study_coverage_b:         Literal["single_study", "multi_study", "unknown"]
    # Scope qualifier check — "scope_unknown" until FindingScope is carried on
    # CanonicalRule (v2 work).  Values: scope_unknown | scope_compatible |
    # scope_mismatch | scope_qualified.
    scope_check_result:       str = "scope_unknown"
    scope_note:               str = ""


# ── Rejection summary ──────────────────────────────────────────────────────────

class RejectedFinding(BaseModel):
    """A finding dropped or excluded at a pipeline stage, with reason metadata."""
    stage: Literal["grounding_map", "group_non_groupable"]
    reason: str                       # e.g. "grounding_score=0.23 < threshold=0.50"
    claim: str
    category: str
    chunk_id: str | None = None       # set for grounding_map stage
    grounding_score: float | None = None
    subject_entity: str | None = None
    outcome_entity: str | None = None
    relation_type: str | None = None


class RejectionSummary(BaseModel):
    """
    Structured report of all findings rejected or excluded during one paper's run.

    Attached to the pipeline result under the ``rejection_summary`` key.
    Use this to audit filter calibration: high grounding_rejection_rate suggests
    the threshold is too strict; high non_groupable_rate suggests the MAP prompt
    is failing to extract entities.
    """
    pmcid: str
    grounding_threshold: float | None
    # Counts
    map_findings_total: int
    map_grounding_rejected: int
    normal_findings_total: int
    non_groupable_total: int
    # Non-groupable breakdown (can overlap — a finding may fail multiple conditions)
    non_groupable_no_subject: int
    non_groupable_no_outcome: int
    non_groupable_unclear_relation: int
    # Rates
    grounding_rejection_rate: float   # map_grounding_rejected / map_findings_total
    non_groupable_rate: float         # non_groupable_total / normal_findings_total
    # Per-category breakdown for grounding rejections
    grounding_rejected_by_category: dict[str, int]
    # Full item list
    rejected: List[RejectedFinding]
