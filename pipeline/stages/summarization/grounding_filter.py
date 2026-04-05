"""
Grounding filter for MAP findings and extracted rules.

Uses a cross-encoder NLI model to check whether each claim is actually
entailed by its cited verbatim source text.  Applied at two points:

  1. After MAP   — filters Finding objects whose verbatim_support does not
                   entail the claim.
  2. After RULES — filters Rule objects whose evidence_chain items do not
                   entail the rule's condition+action.  Evidence items that
                   fail are trimmed; rules with no remaining evidence are
                   dropped entirely.
"""
from __future__ import annotations

import logging
from collections import Counter
from functools import cached_property

from .models import AuditableSummary, EvidenceChainItem, ExtractedRules, Finding, Rule, RuleAuditSummary, RuleCounts

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"


class GroundingFilter:
    """
    NLI-based grounding filter.

    Parameters
    ----------
    threshold:
        Minimum entailment score (0–1) to consider a claim supported.
        Default 0.5 means entailment must be the dominant class.
    model_name:
        HuggingFace model for text-classification NLI.
        Must expose labels including "entailment".
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self.threshold = threshold
        self._model_name = model_name

    # ── Public API ─────────────────────────────────────────────────────────────

    def filter_findings(self, summary: AuditableSummary) -> AuditableSummary:
        """
        Drop findings whose verbatim_support does not entail the claim.
        Returns a new AuditableSummary with only supported findings.
        """
        if not summary.findings:
            return summary

        pairs = [(f.verbatim_support, f.claim) for f in summary.findings]
        supported_mask = self._entailment_mask(pairs)

        kept, dropped = [], []
        for finding, supported in zip(summary.findings, supported_mask):
            if supported:
                kept.append(finding)
            else:
                dropped.append(finding.claim)

        if dropped:
            logger.debug(
                "Chunk %s: grounding dropped %d/%d findings: %s",
                summary.chunk_id, len(dropped), len(summary.findings), dropped,
            )

        return summary.model_copy(update={"findings": kept})

    def filter_rules(self, extracted: ExtractedRules) -> ExtractedRules:
        """
        For each rule:
          - Drop evidence_chain items whose verbatim does not entail the rule.
          - Drop the entire rule if no evidence remains.
        Returns a new ExtractedRules with updated audit_summary.
        """
        if not extracted.rules:
            return extracted

        surviving: list[Rule] = []
        for rule in extracted.rules:
            hypothesis = f"{rule.condition} {rule.action}"
            pairs = [(item.verbatim, hypothesis) for item in rule.evidence_chain]
            mask = self._entailment_mask(pairs)

            kept_evidence: list[EvidenceChainItem] = [
                item for item, ok in zip(rule.evidence_chain, mask) if ok
            ]

            if not kept_evidence:
                logger.debug(
                    "Grounding dropped rule %s — no supporting evidence for: %s %s",
                    rule.rule_id, rule.condition, rule.action,
                )
                continue

            if len(kept_evidence) < len(rule.evidence_chain):
                logger.debug(
                    "Rule %s: trimmed evidence %d → %d items",
                    rule.rule_id, len(rule.evidence_chain), len(kept_evidence),
                )

            surviving.append(rule.model_copy(update={"evidence_chain": kept_evidence}))

        return ExtractedRules(
            rules=surviving,
            audit_summary=_recompute_audit(surviving),
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    @cached_property
    def _pipe(self):
        """Lazy-load the NLI pipeline (cached after first access)."""
        from transformers import pipeline  # optional dep
        return pipeline(
            "text-classification",
            model=self._model_name,
            top_k=None,
        )

    def _entailment_mask(self, pairs: list[tuple[str, str]]) -> list[bool]:
        """
        Run NLI on all (premise, hypothesis) pairs in one batch.
        Returns a bool list: True if entailment score >= threshold.
        """
        scores = _score_pairs(pairs, self._pipe)
        return [s >= self.threshold for s in scores]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_pairs(pairs: list[tuple[str, str]], nli_pipe) -> list[float]:
    """
    Run NLI on a batch of (premise, hypothesis) pairs.
    Returns a float list of entailment scores in [0, 1].
    Empty-string premises score 0.0 without hitting the model.
    """
    scores: list[float] = []
    non_empty_indices: list[int] = []
    non_empty_inputs: list[dict] = []

    for i, (premise, hyp) in enumerate(pairs):
        if not premise.strip():
            scores.append(0.0)
        else:
            scores.append(0.0)  # placeholder
            non_empty_indices.append(i)
            non_empty_inputs.append({"text": premise, "text_pair": hyp})

    if non_empty_inputs:
        batch_results = nli_pipe(non_empty_inputs)
        for idx, result in zip(non_empty_indices, batch_results):
            scores[idx] = next(
                (s["score"] for s in result if s["label"].lower() == "entailment"),
                0.0,
            )

    return scores


def score_findings(findings: list[Finding], nli_pipe) -> None:
    """
    Write grounding_score in-place on every Finding without filtering any out.
    Use this when you want NLI scores available on all findings but do not want
    to drop anything — e.g. for caching the full scored set before Phase 2
    NORMALIZE decides its own threshold.
    """
    if not findings:
        return
    pairs = [(f.verbatim_support, f.claim) for f in findings]
    for finding, score in zip(findings, _score_pairs(pairs, nli_pipe)):
        finding.grounding_score = score


def filter_atomic_findings(
    findings: list[Finding],
    threshold: float,
    nli_pipe,
) -> list[Finding]:
    """
    Score each Finding's (verbatim_support, claim) pair via NLI, write
    grounding_score in-place, and return only findings at or above threshold.

    Parameters
    ----------
    findings:
        Flat list of Finding objects from one or more chunks.
    threshold:
        Minimum entailment score to retain a finding.
    nli_pipe:
        HuggingFace text-classification pipeline (already loaded).
        Pass GroundingFilter._pipe to reuse the cached model instance.
    """
    if not findings:
        return []

    pairs = [(f.verbatim_support, f.claim) for f in findings]
    scores = _score_pairs(pairs, nli_pipe)

    kept: list[Finding] = []
    for finding, score in zip(findings, scores):
        finding.grounding_score = score
        if score >= threshold:
            kept.append(finding)
        else:
            logger.debug(
                "filter_atomic_findings: dropped finding (score=%.3f < %.3f): %s",
                score, threshold, finding.claim,
            )

    return kept


def _recompute_audit(rules: list[Rule]) -> RuleAuditSummary:
    counts: Counter[str] = Counter(r.type for r in rules)
    all_pmcids = {item.pmcid for r in rules for item in r.evidence_chain}
    avg_evidence = (
        sum(len(r.evidence_chain) for r in rules) / len(rules) if rules else 0.0
    )
    return RuleAuditSummary(
        total_rules=len(rules),
        rules_by_type=RuleCounts(
            Diagnostic=counts["Diagnostic"],
            Prognostic=counts["Prognostic"],
            Management=counts["Management"],
        ),
        pmcids_supporting_rules=sorted(all_pmcids),
        average_evidence_per_rule=round(avg_evidence, 2),
    )
