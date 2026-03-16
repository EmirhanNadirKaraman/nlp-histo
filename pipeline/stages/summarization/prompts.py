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
Assign each finding to exactly one: morphology | IHC | molecular_genetics | staging | treatment | prognosis
</Categories>

<FilterRules>
SKIP: Author names, journal metadata, acknowledgments, funding, and generic boilerplate.
ONLY extract from: methods, results, discussion, and case descriptions.
</FilterRules>

<OutputFormat>
Return your analysis in this EXACT structure:

```json
{{
  "chunk_id": "{chunk_id}",
  "findings": [
    {{
      "category": "morphology|IHC|molecular_genetics|staging|treatment|prognosis",
      "claim": "<telegraphic_atomic_fact>",
      "evidence": ["S1|PMC123456|789"],
      "confidence": "high|medium|low",
      "verbatim_support": "<key_quote>"
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
Concept: {concept_name}
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
  "concept": "{concept_name}",
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
Concept: {concept_name}
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
Concept: {concept_name}
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
    return prompt | llm.with_structured_output(AuditableSummary, strict=True).with_retry(
        stop_after_attempt=3
    )


def build_reduce_chain(llm):
    """Return a runnable chain: prompt | llm (structured ConsolidatedSummary)."""
    prompt = ChatPromptTemplate([("system", _REDUCE_SYSTEM), ("user", _REDUCE_USER)])
    return prompt | llm.with_structured_output(ConsolidatedSummary, strict=True).with_retry(
        stop_after_attempt=3
    )


def build_rule_chain(llm):
    """Return a runnable chain: prompt | llm (structured ExtractedRules)."""
    prompt = ChatPromptTemplate([("system", _RULE_SYSTEM), ("user", _RULE_USER)])
    return prompt | llm.with_structured_output(ExtractedRules, strict=True).with_retry(
        stop_after_attempt=3
    )


def build_judge_chain(llm):
    """Return a runnable chain: prompt | llm (structured PairJudgment)."""
    prompt = ChatPromptTemplate([("system", _JUDGE_SYSTEM), ("user", _JUDGE_USER)])
    return prompt | llm.with_structured_output(PairJudgment, strict=True).with_retry(
        stop_after_attempt=3
    )
