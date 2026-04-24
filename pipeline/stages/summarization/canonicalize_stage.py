"""
CANONICALIZE stage: FindingGroup[] → CanonicalRule[]

For each FindingGroup:
  1. Pick a canonical predicate text — LLM selects the best-grounded claim from
     member predicate_text values, with a deterministic fallback (highest
     mean_grounding_score) when the LLM is unavailable or fails.
  2. Compute a CanonicalScope (4-state: single_study, multi_study, conflicted,
     unknown) based on direction_counts and cross-paper PMCID coverage.
  3. Groups with mixed directions (e.g. both "positive" and "negative") are
     split into separate CanonicalRules, one per majority direction.

No cross-paper merging is done here — that requires a multi-paper pool fed
into GROUP before CANONICALIZE runs.  With per-paper runs the groups will
typically have size=1; the stage still produces valid output in that case.
"""
from __future__ import annotations

import hashlib
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from .models import (
    CanonicalRule,
    CanonicalScopeEnum,
    DirectionEnum,
    FindingGroup,
    NormalFinding,
)

logger = logging.getLogger(__name__)


# ── Prompt ─────────────────────────────────────────────────────────────────────

_CANONICALIZE_SYSTEM = """You are a medical knowledge editor.
You will be given a group of related histopathology claims that all describe the
same (subject, outcome, relation) triple.  Your task is to select the single
best predicate text from the provided candidates.

Selection criteria (in priority order):
1. Highest clinical informativeness — concrete, specific, quantified if possible.
2. Grammatically correct and complete sentence.
3. Widest scope (covers the finding without over-claiming).
4. Avoid hedging words like "may", "might", "possibly" unless the uncertainty is real.

Return ONLY the chosen predicate text string, verbatim from the candidates list.
Do NOT paraphrase or combine candidates."""

_CANONICALIZE_USER = """Subject entity  : {subject}
Outcome entity  : {outcome}
Relation type   : {relation_type}
Direction       : {direction}

Candidate predicate texts (one per line, prefixed with rank by grounding score):
{candidates}

Which candidate best represents this finding?"""


def _build_canonicalize_chain(llm):
    """Return a plain (str) output chain — no structured output needed here."""
    from langchain_core.output_parsers import StrOutputParser
    prompt = ChatPromptTemplate([
        ("system", _CANONICALIZE_SYSTEM),
        ("user", _CANONICALIZE_USER),
    ])
    return prompt | llm | StrOutputParser()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _canonical_rule_id(group_id: str, direction: str) -> str:
    return f"CR_{_sha8(group_id)}_{direction}"


def _compute_scope(member_nfs: list[NormalFinding]) -> CanonicalScopeEnum:
    """
    Determine canonical scope from bin-level direction counts and PMCID coverage.

    Rules:
      - If bin has ≥2 non-'unclear' directions with count≥1 → conflicted
      - Else if unique PMCIDs across members ≥ 2            → multi_study
      - Else if unique PMCIDs == 1                          → single_study
      - Else                                                → unknown
    """
    # Count distinct non-unclear directions from the bin, not the group
    bin_directions: set[str] = set()
    for nf in member_nfs:
        d = nf.direction.value if nf.direction is not None else "unclear"
        if d not in ("unclear", "None"):
            bin_directions.add(d)
    if len(bin_directions) >= 2:
        return CanonicalScopeEnum.conflicted

    # Count unique PMCIDs across member NormalFindings
    all_pmcids: set[str] = set()
    for nf in member_nfs:
        all_pmcids.update(nf.pmcids)

    if len(all_pmcids) >= 2:
        return CanonicalScopeEnum.multi_study
    elif len(all_pmcids) == 1:
        return CanonicalScopeEnum.single_study
    return CanonicalScopeEnum.unknown


def _pick_best_predicate_deterministic(
    candidates: list[tuple[float, str]]
) -> str:
    """
    Fallback: return the predicate_text with the highest grounding score.
    candidates: list of (mean_grounding_score, predicate_text), possibly with None scores.
    """
    return max(candidates, key=lambda x: x[0] or 0.0)[1]


def _split_by_direction(
    group: FindingGroup,
    member_nfs: list[NormalFinding],
) -> list[tuple[str, list[NormalFinding]]]:
    """
    Split member NormalFindings by direction.

    If the group has only one direction (or all 'unclear'), return one bin.
    If multiple directions exist, return one bin per non-'unclear' direction
    (findings with direction=None/'unclear' are added to the largest bin).
    """
    non_unclear = {
        d: [] for d, c in group.direction_counts.items()
        if d not in ("unclear", "None") and c > 0
    }
    unclear_nfs: list[NormalFinding] = []

    for nf in member_nfs:
        d = nf.direction.value if nf.direction is not None else "unclear"
        if d in non_unclear:
            non_unclear[d].append(nf)
        else:
            unclear_nfs.append(nf)

    if not non_unclear:
        # All findings are 'unclear' — single bin
        return [("unclear", member_nfs)]

    if len(non_unclear) == 1:
        direction = next(iter(non_unclear))
        return [(direction, member_nfs)]  # keep unclear in same bucket

    # Mixed directions — assign unclear nfs to the largest direction bin
    largest_dir = max(non_unclear, key=lambda d: len(non_unclear[d]))
    non_unclear[largest_dir].extend(unclear_nfs)
    return list(non_unclear.items())


# ── Public API ─────────────────────────────────────────────────────────────────

class CanonicalizeStage:
    """
    CANONICALIZE stage: FindingGroup[] → CanonicalRule[].

    Parameters
    ----------
    llm:
        LangChain chat model used to select canonical predicate text.
        Pass None to always use the deterministic fallback.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm
        self._chain = _build_canonicalize_chain(llm) if llm is not None else None

    def canonicalize(
        self,
        groups: list[FindingGroup],
        normal_findings_by_id: dict[str, NormalFinding],
        pmcid: str,
    ) -> list[CanonicalRule]:
        """
        Produce CanonicalRule objects from FindingGroups.

        Parameters
        ----------
        groups:
            Output of GroupStage.group().
        normal_findings_by_id:
            Dict mapping NormalFinding.normal_id → NormalFinding for all
            findings that were passed into GROUP.  Used to retrieve predicate
            texts and PMCID lists.
        pmcid:
            For logging.

        Returns
        -------
        List of CanonicalRule, possibly more than len(groups) when mixed
        directions cause a group to be split.
        """
        if not groups:
            return []

        canonical_rules: list[CanonicalRule] = []

        for group in groups:
            # Resolve member NormalFindings
            member_nfs: list[NormalFinding] = [
                normal_findings_by_id[mid]
                for mid in group.member_ids
                if mid in normal_findings_by_id
            ]
            if not member_nfs:
                logger.warning(
                    "[%s] CANONICALIZE: group %s has no resolvable members — skipping",
                    pmcid, group.group_id,
                )
                continue

            # Split by direction (may yield multiple bins)
            direction_bins = _split_by_direction(group, member_nfs)

            for direction, bin_nfs in direction_bins:
                if not bin_nfs:
                    continue

                # Build candidate list (score, text) sorted by score desc
                candidates: list[tuple[float, str]] = sorted(
                    [(nf.mean_grounding_score or 0.0, nf.predicate_text) for nf in bin_nfs],
                    key=lambda x: x[0],
                    reverse=True,
                )

                # LLM selection
                predicate_text = self._select_predicate(
                    candidates, group, direction, pmcid
                )

                scope = _compute_scope(bin_nfs)

                # Aggregate evidence and PMCID lists
                all_pmcids: list[str] = sorted(
                    {p for nf in bin_nfs for p in nf.pmcids}
                )
                all_member_ids = [nf.normal_id for nf in bin_nfs]

                # mean_grounding_score across the bin
                scores = [nf.mean_grounding_score for nf in bin_nfs if nf.mean_grounding_score is not None]
                mean_score = sum(scores) / len(scores) if scores else None

                rule = CanonicalRule(
                    canonical_id=_canonical_rule_id(group.group_id, direction),
                    group_id=group.group_id,
                    subject_entity=group.subject_entity,
                    outcome_entity=group.outcome_entity,
                    relation_type=group.relation_type,
                    direction=DirectionEnum(direction) if direction in DirectionEnum._value2member_map_ else None,
                    predicate_text=predicate_text,
                    canonical_scope=scope,
                    category=group.category,
                    supporting_pmcids=all_pmcids,
                    member_normal_ids=all_member_ids,
                    mean_grounding_score=mean_score,
                    finding_count=len(bin_nfs),
                )
                canonical_rules.append(rule)

        logger.info(
            "[%s] CANONICALIZE: %d groups → %d CanonicalRules",
            pmcid, len(groups), len(canonical_rules),
        )
        return canonical_rules

    def _select_predicate(
        self,
        candidates: list[tuple[float, str]],
        group: FindingGroup,
        direction: str,
        pmcid: str,
    ) -> str:
        """
        Use LLM to pick the best predicate text, falling back deterministically.
        """
        if self._chain is None or len(candidates) == 1:
            return _pick_best_predicate_deterministic(candidates)

        # Build numbered candidate list for the prompt
        candidate_str = "\n".join(
            f"{i+1}. [{score:.3f}] {text}"
            for i, (score, text) in enumerate(candidates)
        )

        try:
            result: str = self._chain.invoke({
                "subject": group.subject_entity,
                "outcome": group.outcome_entity,
                "relation_type": group.relation_type.value,
                "direction": direction,
                "candidates": candidate_str,
            })
            result = result.strip()

            # Validate: the returned text must be in our candidate set
            valid_texts = {text for _, text in candidates}
            if result in valid_texts:
                return result

            # LLM returned something not in the candidates — log and fall back
            logger.warning(
                "[%s] CANONICALIZE LLM returned text not in candidates for group %s "
                "(got %r). Using deterministic fallback.",
                pmcid, group.group_id, result[:80],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] CANONICALIZE LLM failed for group %s (%s). Falling back.",
                pmcid, group.group_id, exc,
            )

        return _pick_best_predicate_deterministic(candidates)
