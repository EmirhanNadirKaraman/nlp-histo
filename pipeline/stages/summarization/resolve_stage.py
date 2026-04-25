"""
RESOLVE stage: CanonicalRule[] + Relation[] → FinalRule[]

Deterministic weighted confidence scoring.  No LLM required.

Scoring formula
---------------
Two modes depending on whether the RELATE stage produced any relations:

  relations_present (support_count + contradict_count > 0 for at least one rule):
    base          = mean_grounding_score * 0.60   (range 0–0.60; default 0.30 if unknown)
    finding_bonus = min(finding_count / 5, 1.0) * 0.10   (up to +0.10 for ≥5 findings)
    support_bonus = min(support_count * 0.08, 0.20)       (up to +0.20)
    single_study_pen = 0.10 if study_coverage == "single_study" else 0.0  (−0.10)
    contradict_pen   = min(contradict_count * 0.15, 0.30)  (up to −0.30)

  relations_absent (no RELATE output, e.g. single-paper run or NLI skipped):
    Grounding weight is raised to 0.80 so scores spread meaningfully across
    rules rather than collapsing to a narrow band around 0.50.
    base          = mean_grounding_score * 0.80   (range 0–0.80; default 0.40 if unknown)
    finding_bonus = min(finding_count / 5, 1.0) * 0.15   (up to +0.15 for ≥5 findings)
    single_study_pen = 0.05 (halved — less punishing without cross-rule signal)
    support_bonus = 0   (no relations → no support signal)
    contradict_pen = 0

  final_score = clip(base + finding_bonus + support_bonus
                     − single_study_pen − contradict_pen, 0.0, 1.0)

Example scores, relations-absent mode:
  Well-grounded (0.9), 3 findings, single_study:
    0.72 + 0.09 − 0.05 = 0.76
  Low-grounded (0.3), 1 finding, single_study:
    0.24 + 0.03 − 0.05 = 0.22

is_contradicted flag is set when any CONTRADICT relation touches the rule.

FinalRules are returned sorted by final_score descending.
"""
from __future__ import annotations

import logging

from .models import CanonicalRule, FinalRule, Relation, RelationTypeLabel

logger = logging.getLogger(__name__)

# Scoring constants — relations-present mode
_GROUNDING_WEIGHT       = 0.60
_GROUNDING_DEFAULT      = 0.50   # used when mean_grounding_score is None
_FINDING_BONUS_MAX      = 0.10
_FINDING_BONUS_SCALE    = 5
_SUPPORT_BOOST_PER_REL  = 0.08
_SUPPORT_BOOST_CAP      = 0.20
_SINGLE_STUDY_PEN       = 0.10
_CONTRADICT_PEN_PER_REL = 0.15
_CONTRADICT_PEN_CAP     = 0.30

# Scoring constants — relations-absent mode (RELATE skipped or empty)
# Grounding weight raised so scores spread across [0, 1] rather than
# clustering in a narrow band around 0.50.
_NO_REL_GROUNDING_WEIGHT  = 0.80
_NO_REL_FINDING_BONUS_MAX = 0.15
_NO_REL_SINGLE_STUDY_PEN  = 0.05


def _build_adjacency(
    rules: list[CanonicalRule],
    relations: list[Relation],
) -> dict[str, list[Relation]]:
    """Map each canonical_id to the list of Relation objects it participates in."""
    adj: dict[str, list[Relation]] = {r.canonical_id: [] for r in rules}
    for rel in relations:
        if rel.rule_id_a in adj:
            adj[rel.rule_id_a].append(rel)
        if rel.rule_id_b in adj:
            adj[rel.rule_id_b].append(rel)
    return adj


class ResolveStage:
    """
    RESOLVE stage: score and rank CanonicalRules into FinalRules.

    All logic is deterministic; no LLM calls are made.
    """

    def resolve(
        self,
        rules: list[CanonicalRule],
        relations: list[Relation],
        pmcid: str = "",
    ) -> list[FinalRule]:
        """
        Score each CanonicalRule and return a ranked list of FinalRules.

        Parameters
        ----------
        rules:
            Output of CanonicalizeStage.canonicalize().
        relations:
            Output of RelateStage.relate().
        pmcid:
            For logging only.

        Returns
        -------
        FinalRule list sorted by final_score descending.
        """
        if not rules:
            return []

        adj = _build_adjacency(rules, relations)

        # Detect whether RELATE produced any relational signal at all.
        # When it didn't (single-paper run, NLI skipped, or no rule pairs
        # exceeded the similarity threshold), use the wider grounding-dominant
        # formula so scores are meaningfully spread.
        relations_present = len(relations) > 0

        final_rules: list[FinalRule] = []

        for rule in rules:
            touching_rels = adj.get(rule.canonical_id, [])

            # Separate by type
            supports = [
                r for r in touching_rels
                if r.relation_type == RelationTypeLabel.SUPPORT
            ]
            contradicts = [
                r for r in touching_rels
                if r.relation_type == RelationTypeLabel.CONTRADICT
            ]
            scope_qualifies = [
                r for r in touching_rels
                if r.relation_type == RelationTypeLabel.SCOPE_QUALIFY
            ]

            grounding = (
                rule.mean_grounding_score
                if rule.mean_grounding_score is not None
                else _GROUNDING_DEFAULT
            )

            if relations_present:
                base           = grounding * _GROUNDING_WEIGHT
                finding_bonus  = min(rule.finding_count / _FINDING_BONUS_SCALE, 1.0) * _FINDING_BONUS_MAX
                support_bonus  = min(len(supports) * _SUPPORT_BOOST_PER_REL, _SUPPORT_BOOST_CAP)
                single_study_pen = (
                    _SINGLE_STUDY_PEN
                    if rule.study_coverage == "single_study"
                    else 0.0
                )
                contradict_pen = min(len(contradicts) * _CONTRADICT_PEN_PER_REL, _CONTRADICT_PEN_CAP)
            else:
                base           = grounding * _NO_REL_GROUNDING_WEIGHT
                finding_bonus  = min(rule.finding_count / _FINDING_BONUS_SCALE, 1.0) * _NO_REL_FINDING_BONUS_MAX
                support_bonus  = 0.0
                single_study_pen = (
                    _NO_REL_SINGLE_STUDY_PEN
                    if rule.study_coverage == "single_study"
                    else 0.0
                )
                contradict_pen = 0.0

            final_score = max(
                0.0,
                min(1.0, base + finding_bonus + support_bonus - single_study_pen - contradict_pen),
            )

            # Collect IDs of rules that contradict this one
            contradicted_by: list[str] = []
            for r in contradicts:
                other_id = r.rule_id_b if r.rule_id_a == rule.canonical_id else r.rule_id_a
                contradicted_by.append(other_id)

            final_rule = FinalRule(
                final_id=f"FR_{rule.canonical_id}",
                canonical_id=rule.canonical_id,
                group_id=rule.group_id,
                subject_entity=rule.subject_entity,
                outcome_entity=rule.outcome_entity,
                relation_type=rule.relation_type,
                direction=rule.direction,
                predicate_text=rule.predicate_text,
                is_conflicted=rule.is_conflicted,
                study_coverage=rule.study_coverage,
                category=rule.category,
                supporting_pmcids=rule.supporting_pmcids,
                member_normal_ids=rule.member_normal_ids,
                mean_grounding_score=rule.mean_grounding_score,
                finding_count=rule.finding_count,
                final_score=round(final_score, 4),
                support_count=len(supports),
                contradict_count=len(contradicts),
                scope_qualify_count=len(scope_qualifies),
                is_contradicted=len(contradicts) > 0,
                contradicted_by=contradicted_by,
            )
            final_rules.append(final_rule)

        # Sort by score descending
        final_rules.sort(key=lambda r: r.final_score, reverse=True)

        logger.info(
            "[%s] RESOLVE: %d CanonicalRules → %d FinalRules "
            "(top score=%.3f, contradicted=%d)",
            pmcid,
            len(rules),
            len(final_rules),
            final_rules[0].final_score if final_rules else 0.0,
            sum(1 for r in final_rules if r.is_contradicted),
        )
        return final_rules
