"""
Prompt templates and chain factories for the three pipeline stages.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from .models import AuditableSummary, ConsolidatedSummary, ExtractedRules, PairJudgment

# ── MAP ────────────────────────────────────────────────────────────────────────

_MAP_SYSTEM = """<Role>You are a High-Recall Medical Evidence Analyst specializing in histopathology.</Role>

<Task>
Convert the provided literature chunk into a list of ATOMIC, INDEPENDENT medical facts with full provenance.

CRITICAL EXTRACTION RULES:
1. ZERO LOSS: Extract every unique dose, p-value, patient count, demographic, and clinical relationship.
2. ATOMICITY: Each 'finding' must be a standalone fact. If a sentence/paragraph contains multiple observations (e.g., two different stains), you MUST create separate entries.
3. NO CONTEXTUAL DRIFT: Do not summarize. If information is partial due to chunk boundaries, extract exactly what is visible.
4. TELEGRAPHIC STYLE: Omit filler like 'the authors found.' Focus on direct relationships using arrows (e.g., 'CD30 -> Positive').
5. CITATION: Every claim MUST cite its Sentence ID using the format: [S1|PMC123456|789].
</Task>

<Categories>
Assign each finding to exactly one: morphology | IHC | molecular_genetics | staging | treatment | prognosis | demographics

  Use demographics for: sex distribution, age distribution, sex predominance, cohort composition,
  epidemiology, and any population-level clinical demographic observation.
  IMPORTANT: use category="demographics" (with the trailing 's'). Do NOT write category="demographic".
</Categories>

<FilterRules>
SKIP: Author names, journal metadata, acknowledgments, funding, and generic boilerplate.
SKIP: Patient-specific narratives — age/sex of individual patients, presenting symptoms,
      clinical history of a single case (e.g. "46-year-old man presented with a nodule").
      These are case vignettes, NOT generalizable facts.
ONLY extract: generalizable medical facts — findings that apply to a disease entity,
      population, or biomarker across cases. A valid claim is true for the entity in
      general, not just for one patient in one visit.
ONLY extract from: methods, results, discussion, and case descriptions
      (but only the generalizable conclusions drawn from cases, not the case narrative itself).
</FilterRules>

<StructuredFields>
For each finding, extract the following fields. Do NOT omit any field. Do NOT leave outcome_entity or direction null if they are clearly inferable from the claim text or verbatim support — only use null when the value is genuinely absent from the text.

subject_entity
  The entity being described, tested, or associated. This is the left-hand side of the relation.
  Can be: a lesion type, tumour entity, cell population, biomarker, protein, gene, or organism.
  Examples: "MGA", "CD30", "BCL2", "tumour cells", "Granuloma annulare", "Ki-67".
  Output null only when no subject can be identified.

outcome_entity
  The feature, endpoint, expression, or attribute that is being observed or predicted. This is the right-hand side of the relation.
  Can be: a histological feature, clinical endpoint, expression marker, response measure, or demographic attribute.
  Examples: "mucin", "necrosis", "multinucleated giant cells", "overall survival", "CD30 expression", "R-CHOP response", "female predominance".
  Output null only when no outcome can be identified — not as a default.

relation_type
  The type of relation. Must be exactly one of the values below. NEVER output null.
  Use "unclear" only when the relation genuinely does not fit any category.

  has_feature        : Entity exhibits, shows, or has a histological or morphological feature.
                       e.g. "idiopathic GA exhibits mucin", "MGA shows no particular pattern",
                            "necrosis was present", "multinucleated giant cells were absent"
  expression         : Biomarker, protein, or gene expression level or staining result.
                       e.g. "CD30 was expressed", "BCL2 negative", "Ki-67 index 80%"
  prognostic         : Entity predicts or is associated with a clinical outcome (survival, recurrence).
                       e.g. "BCL2 associated with worse OS", "CD30 predicts R-CHOP response"
  comparative        : Relative frequency, likelihood, or prevalence compared between groups.
                       e.g. "MGA less likely to exhibit mucin than idiopathic GA",
                            "idiopathic GA more likely to show necrobiotic pattern"
  demographic        : Age, sex, prevalence, or incidence in a population.
                       e.g. "MGA presents in the seventh decade", "GA more common in women"
  treatment_response : Response to or outcome of a treatment regimen.
                       e.g. "R-CHOP achieves CR in 75%", "refractory to conventional therapy"
  unclear            : Relation type genuinely cannot be determined.

direction  (OPTIONAL polarity — output null when not applicable or not inferable)
  positive : present / expressed / increased / more frequent / better outcome
  negative : absent / not expressed / decreased / less frequent / worse outcome / inverse
  absent   : explicitly not present / not seen / lacking / negative staining
  partial  : focal / patchy / heterogeneous / partial
  unclear  : indeterminate polarity

BAD vs GOOD examples (claim field):

  BAD  — patient narrative, not a generalizable fact:
    "46-year-old man presented with asymptomatic, slowly growing erythematous nodule"
    "The patient had a 5-month history of a lesion on the right back"
  GOOD — generalizable histopathological fact:
    "Granuloma annulare presents as erythematous papules or nodules"
    "MGA shows no particular architectural pattern on H&E"

Field mapping examples:

  Morphology (has_feature):
    "idiopathic GA exhibits mucin"
      → relation_type: has_feature, direction: positive
    "MGA shows no particular pattern"
      → relation_type: has_feature, direction: absent
    "Mucin was absent in MGA"
      → relation_type: has_feature, direction: absent
    "Multinucleated giant cells were present in idiopathic GA"
      → relation_type: has_feature, direction: positive

  Comparative (comparative):
    "idiopathic GA more likely to exhibit mucin than MGA"
      → relation_type: comparative, direction: positive
    "MGA less likely to exhibit mucin"
      → relation_type: comparative, direction: negative

  Demographic (demographic):
    "MGA presents in the seventh decade of life"
      → relation_type: demographic, direction: null
    "Women more likely to be affected by GA than men"
      → relation_type: demographic, direction: positive

  IHC / expression (expression):
    "Tumour cells express CD30"
      → relation_type: expression, direction: positive
    "BCL2 was negative in GCB-DLBCL"
      → relation_type: expression, direction: absent

  Prognosis (prognostic):
    "High BCL2 expression associated with worse overall survival"
      → relation_type: prognostic, direction: negative
    "CD30 positivity predicts improved R-CHOP response"
      → relation_type: prognostic, direction: positive

  Treatment (treatment_response):
    "R-CHOP achieves complete remission in 75% of patients"
      → relation_type: treatment_response, direction: positive

scope            : A structured object with the following sub-fields (all nullable):
  disease_subtype   : Specific disease subtype in scope, e.g. "GCB-DLBCL", "non-GCB DLBCL". null if not mentioned.
  cohort_n          : Integer patient count for this specific finding, e.g. 127. null if not stated.
  assay_method      : Detection method, e.g. "IHC (JCM182 antibody)", "FISH", "PCR". null if not stated.
  biomarker_cutoff  : Positivity threshold, e.g. ">10%", "≥30%". null if not stated.
  tissue_site       : Tissue or anatomical site, e.g. "lymph node", "bone marrow". null if not stated.
  treatment_context : Treatment regimen in scope, e.g. "R-CHOP", "CHOP". null if not stated.
  endpoint          : Clinical endpoint, e.g. "5-year OS", "PFS", "complete remission rate". null if not stated.
  study_design      : Study design, e.g. "cohort", "RCT", "case_series", "meta-analysis". null if not determinable.
  scope_parsed      : Set to true if at least one scope sub-field above is non-null; false otherwise.
</StructuredFields>

<OutputFormat>
Return your analysis in this EXACT structure:

```json
{{
  "chunk_id": "{chunk_id}",
  "findings": [
    {{
      "category": "morphology|IHC|molecular_genetics|staging|treatment|prognosis|demographics",
      "claim": "<telegraphic_atomic_fact>",
      "evidence": ["S1|PMC123456|789"],
      "confidence": "high|medium|low",
      "verbatim_support": "<key_quote>",
      "subject_entity": "<entity_name_or_null>",
      "outcome_entity": "<outcome_name_or_null>",
      "relation_type": "has_feature|expression|prognostic|comparative|demographic|treatment_response|unclear",
      "direction": "positive|negative|absent|partial|unclear|null",
      "scope": {{
        "disease_subtype": "<value_or_null>",
        "cohort_n": <integer_or_null>,
        "assay_method": "<value_or_null>",
        "biomarker_cutoff": "<value_or_null>",
        "tissue_site": "<value_or_null>",
        "treatment_context": "<value_or_null>",
        "endpoint": "<value_or_null>",
        "study_design": "<value_or_null>",
        "scope_parsed": true|false
      }}
    }}
  ],
  "summary_text": "<narrative_summary_with_inline_citations>",
  "audit_metadata": {{
    "sentences_analyzed": <count>,
    "sentences_cited": [<ids>],
    "pmcids_referenced": [<pmcids>],
    "uncited_sentences": [<ids>]
  }}
}}"""

_MAP_USER = """<Context>
Paper (PMCID): {pmcid}
Chunk ID: {chunk_id}
Source Sentences (each tagged with [SentenceID|PMCID|TextElementID]):
{text}
</Context>"""

# ── REDUCE ─────────────────────────────────────────────────────────────────────

_REDUCE_SYSTEM = """<Role>
You are a Lead Pathologist synthesizing chunk-level analyses into a Master Clinical Brief with FULL AUDIT TRAIL.
</Role>

<Task>
Consolidate all chunk analyses into a unified report while PRESERVING THE COMPLETE AUDIT CHAIN.

CRITICAL AUDIT REQUIREMENTS:
1. EVERY claim must cite the source using format: [S1|PMC123456|789]
2. When multiple sources support a finding, list ALL citations
3. Flag any conflicting evidence with all sources cited
4. Do NOT add information not present in the chunk analyses
5. PRESERVE QUANTITATIVE NUANCE: Do not average or generalize specific values (percentages, doses, p-values). If findings differ, list them as a range or as distinct supporting points.
</Task>

<OutputFormat>
```json
{{
  "pmcid": "{pmcid}",
  "sections": {{
    "clinical_significance": {{
      "findings": [
        {{
          "claim": "<statement>",
          "sources": [
            {{"sentences": ["S1|PMC123456|789"], "verbatim": "<quote>"}}
          ],
          "strength": "strong|moderate|weak"
        }}
      ]
    }},
    "histopathological_features": {{ ... }},
    "management_outcomes": {{ ... }},
    "risk_factors_associations": {{ ... }}
  }},
  "narrative_summary": "<readable summary with inline citations [S1|PMC123456|789]>",
  "audit_trail": {{
    "chunks_processed": <count>,
    "total_sentences_cited": <count>,
    "unique_pmcids": [<list>],
    "unique_text_element_ids": [<list>],
    "evidence_conflicts": [
      {{"topic": "...", "conflicting_sources": [...]}}
    ]
  }}
}}
```
</OutputFormat>

Master Auditable Summary:"""

_REDUCE_USER = """<Context>
Paper (PMCID): {pmcid}
Total Chunks: {num_chunks}

Chunk Analyses (JSON format with provenance):
{summaries}
</Context>"""

# ── RULE EXTRACTION ────────────────────────────────────────────────────────────

_RULE_SYSTEM = """<Role>
You are a Medical Knowledge Engineer. Extract structured IF-THEN rules with COMPLETE PROVENANCE from the Auditable Summary.
</Role>

<Task>
Extract actionable clinical rules, each with FULL TRACEABILITY back to source documents.

AUDIT REQUIREMENTS:
1. Each rule must cite specific evidence from the summary
2. Include the database reference (PMCID + text_element_id) for each supporting sentence
3. Use the citation format: S1|PMC123456|789
</Task>

<OutputFormat>
```json
{{
  "rules": [
    {{
      "rule_id": "R1",
      "type": "Diagnostic|Prognostic|Management",
      "condition": "IF <observation>",
      "action": "THEN <conclusion>",
      "confidence": "High|Medium|Low",
      "evidence_chain": [
        {{
          "sentence_id": "S1",
          "pmcid": "PMC123456",
          "text_element_id": 789,
          "verbatim": "<supporting quote>"
        }}
      ],
      "contraindications": ["<any noted exceptions with sources>"]
    }}
  ],
  "audit_summary": {{
    "total_rules": <count>,
    "rules_by_type": {{"Diagnostic": N, "Prognostic": N, "Management": N}},
    "pmcids_supporting_rules": [<list>],
    "average_evidence_per_rule": <float>
  }}
}}
```
</OutputFormat>

Extracted Rules with Provenance:"""

_RULE_USER = """<Input>
Paper (PMCID): {pmcid}
Auditable Summary (JSON with full provenance):
{summary}
</Input>"""


# ── CONTRADICTION JUDGE ────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are a clinical knowledge auditor.
You will be given two IF-THEN medical rules extracted from histopathology literature.
Determine whether they contradict each other — i.e. given the same or similar clinical
observation, they lead to opposing conclusions or incompatible actions.

Rules that cover different scenarios, different patient subgroups, or are simply
unrelated do NOT contradict each other.  Only flag genuine logical conflicts."""

_JUDGE_USER = """Rule A:
  Condition : {condition_a}
  Action    : {action_a}

Rule B:
  Condition : {condition_b}
  Action    : {action_b}

Do these rules contradict each other?"""


# ── Chain factories ────────────────────────────────────────────────────────────

def build_map_chain(llm):
    """Return a runnable chain: prompt | llm (structured AuditableSummary)."""
    prompt = ChatPromptTemplate([("system", _MAP_SYSTEM), ("user", _MAP_USER)])
    return prompt | llm.with_structured_output(AuditableSummary, strict=True)


def build_reduce_chain(llm):
    """Return a runnable chain: prompt | llm (structured ConsolidatedSummary)."""
    prompt = ChatPromptTemplate([("system", _REDUCE_SYSTEM), ("user", _REDUCE_USER)])
    return prompt | llm.with_structured_output(ConsolidatedSummary, strict=True).with_retry(
        stop_after_attempt=2,
    )


def build_rule_chain(llm):
    """Return a runnable chain: prompt | llm (structured ExtractedRules)."""
    prompt = ChatPromptTemplate([("system", _RULE_SYSTEM), ("user", _RULE_USER)])
    return prompt | llm.with_structured_output(ExtractedRules, strict=True).with_retry(
        stop_after_attempt=2,
    )


def build_judge_chain(llm):
    """Return a runnable chain: prompt | llm (structured PairJudgment)."""
    prompt = ChatPromptTemplate([("system", _JUDGE_SYSTEM), ("user", _JUDGE_USER)])
    return prompt | llm.with_structured_output(PairJudgment, strict=True).with_retry(
        stop_after_attempt=2,
    )
