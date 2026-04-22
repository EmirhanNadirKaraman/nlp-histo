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

from .models import AuditableSummary, EvidenceChainItem, ExtractedRules, Finding, Rule, RuleAuditSummary, RuleCounts

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
_DEFAULT_BATCH_SIZE = 16

# Module-level NLI pipeline singleton — shared across all GroundingFilter
# instances and reused by RelateStage via relate_stage._get_nli_pipe().
# Avoids reloading the model on each GroundingFilter instantiation.
_NLI_PIPE_CACHE: dict[str, object] = {}


def _get_device() -> int | str:
    """Return the best available device: CUDA GPU, MPS (Apple Silicon), or CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return -1


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
    batch_size:
        Number of (premise, hypothesis) pairs per model forward pass.
        Larger values improve GPU/MPS throughput. Default 16.
    device:
        Device for inference. None = auto-detect (CUDA → MPS → CPU).
        Pass 0 for CUDA, "mps" for Apple Silicon, -1 to force CPU.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: int | str | None = None,
    ) -> None:
        self.threshold = threshold
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device if device is not None else _get_device()

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

    def filter_findings_with_scores(
        self, summary: AuditableSummary
    ) -> tuple[AuditableSummary, list[Finding]]:
        """
        Like filter_findings() but scores ALL findings in one NLI pass,
        writing grounding_score in-place on every finding (kept and dropped),
        and returns (kept_summary, dropped_findings).

        Use this in the runner to replace the separate filter_findings() +
        score_findings() two-step calls — one NLI batch instead of two.
        """
        if not summary.findings:
            return summary, []

        pairs = [(f.verbatim_support, f.claim) for f in summary.findings]
        scores = _score_pairs(pairs, self._pipe)

        kept: list[Finding] = []
        dropped: list[Finding] = []
        for finding, score in zip(summary.findings, scores):
            finding.grounding_score = score
            if score >= self.threshold:
                kept.append(finding)
            else:
                dropped.append(finding)

        return summary.model_copy(update={"findings": kept}), dropped

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

    @property
    def _pipe(self):
        """Return the NLI pipeline, loading it if not already cached."""
        global _NLI_PIPE_CACHE
        if self._model_name not in _NLI_PIPE_CACHE:
            from transformers import pipeline  # optional dep
            logger.info(
                "GroundingFilter: loading NLI model %r on device=%r batch_size=%d",
                self._model_name, self._device, self._batch_size,
            )
            _NLI_PIPE_CACHE[self._model_name] = pipeline(
                "text-classification",
                model=self._model_name,
                top_k=None,
                device=self._device,
                batch_size=self._batch_size,
            )
        return _NLI_PIPE_CACHE[self._model_name]

    def _entailment_mask(self, pairs: list[tuple[str, str]]) -> list[bool]:
        """
        Run NLI on all (premise, hypothesis) pairs in one batch.
        Returns a bool list: True if entailment score >= threshold.
        """
        scores = _score_pairs(pairs, self._pipe)
        return [s >= self.threshold for s in scores]


# ── Helpers ────────────────────────────────────────────────────────────────────

# cross-encoder/nli-deberta-v3-base has a 512-token limit.  A long
# verbatim_support paragraph can exceed this and get silently truncated,
# making the entailment check unreliable.  We split the premise into
# overlapping windows of ~400 chars (well inside 512 tokens for English)
# and take the max entailment score across windows.
_WINDOW_CHARS = 400
_STEP_CHARS   = 200   # 50 % overlap so a sentence split across a boundary is covered


def _split_windows(text: str) -> list[str]:
    """
    Split *text* into overlapping character windows.
    Returns the original text as a single window if it is short enough.
    """
    if len(text) <= _WINDOW_CHARS:
        return [text]
    windows: list[str] = []
    start = 0
    while start < len(text):
        windows.append(text[start: start + _WINDOW_CHARS])
        start += _STEP_CHARS
    return windows


def _score_pairs(pairs: list[tuple[str, str]], nli_pipe) -> list[float]:
    """
    Run NLI on a batch of (premise, hypothesis) pairs.
    Returns a float list of entailment scores in [0, 1].

    Long premises are split into overlapping windows; the maximum entailment
    score across windows is returned so that a supporting sentence in the
    second half of a paragraph is not missed due to token-limit truncation.
    Empty-string premises score 0.0 without hitting the model.
    """
    # Build a flat list of (pair_index, window_text, hyp) inputs
    flat_inputs: list[dict] = []
    flat_pair_indices: list[int] = []

    scores: list[float] = [0.0] * len(pairs)

    for i, (premise, hyp) in enumerate(pairs):
        if not premise.strip():
            continue
        for window in _split_windows(premise):
            flat_inputs.append({"text": window, "text_pair": hyp})
            flat_pair_indices.append(i)

    if flat_inputs:
        batch_results = nli_pipe(flat_inputs)
        for pair_idx, result in zip(flat_pair_indices, batch_results):
            window_score = next(
                (s["score"] for s in result if s["label"].lower() == "entailment"),
                0.0,
            )
            # Keep the best window score for this pair
            if window_score > scores[pair_idx]:
                scores[pair_idx] = window_score

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
