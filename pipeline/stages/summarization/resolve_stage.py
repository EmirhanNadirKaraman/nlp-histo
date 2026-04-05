"""
RESOLVE stage: CanonicalRule[] + Relation[] → FinalRule[]

Deterministic weighted confidence scoring.  No LLM required.

Scoring formula
---------------
  base          = mean_grounding_score * 0.60   (range 0–0.60; default 0.30 if unknown)
  finding_bonus = min(finding_count / 5, 1.0) * 0.10   (up to +0.10 for ≥5 findings)
  support_bonus = min(support_count * 0.08, 0.20)       (up to +0.20)
  single_study_pen = 0.10 if canonical_scope == single_study else 0.0  (−0.10)
  contradict_pen   = min(contradict_count * 0.15, 0.30)  (up to −0.30)

  final_score = clip(base + finding_bonus + support_bonus
                     − single_study_pen − contradict_pen, 0.0, 1.0)

Example scores (approximate):
  Well-grounded (0.9), 3 findings, 1 support, single_study, 0 contradictions:
    0.54 + 0.06 + 0.08 − 0.10 − 0.0 = 0.58
  Singleton (0.5), 1 finding, 0 support, single_study, 1 contradiction:
    0.30 + 0.02 + 0.0  − 0.10 − 0.15 = 0.07
  Multi-study (0.8), 5 findings, 2 supports, 0 contradictions:
    0.48 + 0.10 + 0.16 − 0.0  − 0.0  = 0.74

is_contradicted flag is set when any CONTRADICT relation touches the rule.

FinalRules are returned sorted by final_score descending.
"""
from __future__ import annotations

import logging

from .models import CanonicalRule, CanonicalScopeEnum, FinalRule, Relation, RelationTypeLabel

logger = logging.getLogger(__name__)

# Scoring constants — all explicit, no magic numbers elsewhere
_GROUNDING_WEIGHT       = 0.60   # base = mean_grounding_score * this
_GROUNDING_DEFAULT      = 0.50   # used when mean_grounding_score is None
_FINDING_BONUS_MAX      = 0.10   # max bonus from finding_count
_FINDING_BONUS_SCALE    = 5      # finding_count / this, capped at 1.0
_SUPPORT_BOOST_PER_REL  = 0.08
_SUPPORT_BOOST_CAP      = 0.20
_SINGLE_STUDY_PEN       = 0.10   # penalty for single-paper rules
_CONTRADICT_PEN_PER_REL = 0.15   # now active
_CONTRADICT_PEN_CAP     = 0.30


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
            base = grounding * _GROUNDING_WEIGHT

            finding_bonus = min(rule.finding_count / _FINDING_BONUS_SCALE, 1.0) * _FINDING_BONUS_MAX

            support_bonus = min(len(supports) * _SUPPORT_BOOST_PER_REL, _SUPPORT_BOOST_CAP)

            single_study_pen = (
                _SINGLE_STUDY_PEN
                if rule.canonical_scope == CanonicalScopeEnum.single_study
                else 0.0
            )

            contradict_pen = min(
                len(contradicts) * _CONTRADICT_PEN_PER_REL,
                _CONTRADICT_PEN_CAP,
            )

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
                canonical_scope=rule.canonical_scope,
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
