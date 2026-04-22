"""
RELATE stage: CanonicalRule[] → Relation[]

NLI-based pairwise comparison between canonical rules.  Each pair is classified
as one of: SUPPORT, CONTRADICT, SCOPE_QUALIFY, UNRELATED.

Only non-UNRELATED pairs are returned.

Comparability gate (ALL four must hold for NLI to run):
  1. exact category match
  2. exact relation_type match
  3. exact normalized subject_entity match
  4. exact normalized outcome_entity match

  Rationale for condition 4: two has_feature rules sharing the same subject but
  describing different histological features (e.g. "giant cells" vs "open vascular
  channels") are coexisting attributes — not contradiction candidates.  Comparing
  them produces spurious CONTRADICT labels from NLI.  Only rules about the SAME
  outcome can meaningfully SUPPORT, CONTRADICT, or SCOPE_QUALIFY each other.

  Special rule: for has_feature relation_type, CONTRADICT is only emitted when
  outcome_entity also matches.  Since condition 4 already enforces outcome match
  before NLI runs, this is automatically satisfied.  The explicit check in
  _classify_pair is a safety belt for edge cases.

The NLI pipeline is shared with GroundingFilter via a module-level singleton
so the model is loaded only once per process.
"""
from __future__ import annotations

import itertools
import logging

from .models import CanonicalRule, Relation, RelationTypeLabel

logger = logging.getLogger(__name__)

MAX_PAIRS = 500  # safety cap — increase if your rule sets are large

# ── Module-level NLI singleton ─────────────────────────────────────────────────
# Delegates to the grounding_filter module-level cache so the same model
# instance is shared with GroundingFilter (loaded once per process).

_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def _get_nli_pipe(model_name: str = _NLI_MODEL, batch_size: int = 16, device: int | str | None = None):
    """Return cached NLI pipeline, sharing the grounding_filter singleton."""
    from .grounding_filter import _NLI_PIPE_CACHE, _get_device
    if model_name not in _NLI_PIPE_CACHE:
        from transformers import pipeline  # type: ignore
        resolved_device = device if device is not None else _get_device()
        logger.info(
            "RELATE: loading NLI model %r on device=%r batch_size=%d",
            model_name, resolved_device, batch_size,
        )
        _NLI_PIPE_CACHE[model_name] = pipeline(
            "text-classification", model=model_name, top_k=None,
            device=resolved_device, batch_size=batch_size,
        )
    return _NLI_PIPE_CACHE[model_name]



# ── NLI helpers ────────────────────────────────────────────────────────────────

def _nli_scores(pairs: list[tuple[str, str]], pipe) -> list[dict[str, float]]:
    """
    Run NLI on (premise, hypothesis) pairs.
    Returns a list of dicts {entailment, contradiction, neutral} per pair.
    """
    inputs = [{"text": p, "text_pair": h} for p, h in pairs]
    results = pipe(inputs)
    out = []
    for result in results:
        d = {item["label"].lower(): item["score"] for item in result}
        out.append(d)
    return out


def _classify_pair(
    rule_a: CanonicalRule,
    rule_b: CanonicalRule,
    scores_ab: dict[str, float],
    scores_ba: dict[str, float],
    entailment_threshold: float,
    contradiction_threshold: float,
) -> RelationTypeLabel | None:
    """
    Decide the relation type for a (rule_a, rule_b) pair using bidirectional
    NLI scores.

    Logic:
      - CONTRADICT   : both directions score high on 'contradiction',
                       AND the two rules do not share the same direction
                       (two positive rules or two negative rules cannot
                       contradict each other — they are coexisting findings).
                       The comparability gate already ensures same
                       (subject, outcome, relation_type, category), so a high
                       mutual contradiction score with opposite directions is a
                       genuine conflict.
      - SUPPORT      : both directions score high on 'entailment'
      - SCOPE_QUALIFY: one direction entails the other but not vice-versa
      - None         : UNRELATED — caller skips this pair
    """
    from .models import DirectionEnum

    ent_ab = scores_ab.get("entailment", 0.0)
    ent_ba = scores_ba.get("entailment", 0.0)
    con_ab = scores_ab.get("contradiction", 0.0)
    con_ba = scores_ba.get("contradiction", 0.0)

    # CONTRADICT guard: block when both rules point in the same direction.
    # Two positive (or two negative/absent) findings about the same entity
    # pair are coexisting observations — not contradictions — even if the NLI
    # model scores them high on 'contradiction' due to lexical overlap.
    # We use the structured direction field (set by the LLM) rather than
    # re-inferring polarity from text, which was unreliable and blocked almost
    # all genuine contradictions.
    _NEGATIVE_DIRECTIONS = {DirectionEnum.negative, DirectionEnum.absent}
    _POSITIVE_DIRECTIONS = {DirectionEnum.positive, DirectionEnum.partial}

    dir_a = rule_a.direction
    dir_b = rule_b.direction

    same_polarity = (
        # both clearly positive
        (dir_a in _POSITIVE_DIRECTIONS and dir_b in _POSITIVE_DIRECTIONS)
        # both clearly negative / absent
        or (dir_a in _NEGATIVE_DIRECTIONS and dir_b in _NEGATIVE_DIRECTIONS)
    )
    contradict_allowed = not same_polarity

    # Mutual contradiction
    if contradict_allowed and con_ab >= contradiction_threshold and con_ba >= contradiction_threshold:
        return RelationTypeLabel.CONTRADICT

    # Mutual entailment → support
    if ent_ab >= entailment_threshold and ent_ba >= entailment_threshold:
        return RelationTypeLabel.SUPPORT

    # Asymmetric entailment → scope qualification
    if ent_ab >= entailment_threshold or ent_ba >= entailment_threshold:
        return RelationTypeLabel.SCOPE_QUALIFY

    return None  # UNRELATED


# ── Pruning ────────────────────────────────────────────────────────────────────

def _norm_outcome(entity: str) -> str:
    """
    Lowercase + strip for outcome gate comparison.
    Used for has_feature and all other relation types except expression.
    """
    return entity.strip().lower()


def _norm_outcome_expression(entity: str) -> str:
    """
    For expression relation_type: strip marker suffixes before comparing.
    "CD31 expression" → "cd31", "CD34 expression" → "cd34".
    This ensures different markers are never considered the same outcome.
    """
    s = entity.strip().lower()
    for suffix in (" expression", " positivity", " staining", " immunoreactivity"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def _should_compare(a: CanonicalRule, b: CanonicalRule) -> tuple[bool, str]:
    """
    Strict 4-condition gate. A pair reaches NLI only when ALL hold:
      1. category matches exactly
      2. relation_type matches exactly
      3. subject_entity matches exactly
      4. normalized outcome_entity matches exactly
         (for expression: marker name after stripping suffixes must match)

    Returns (eligible, rejection_reason).
    rejection_reason is "" when eligible=True.

    Debug rejection reasons:
      category_mismatch
      relation_type_mismatch
      subject_mismatch
      outcome_incompatible
    """
    from .models import RelationTypeEnum  # local import avoids circular at module level
    if a.category != b.category:
        return False, "category_mismatch"
    if a.relation_type != b.relation_type:
        return False, "relation_type_mismatch"
    if a.subject_entity != b.subject_entity:
        return False, "subject_mismatch"
    # Outcome comparison — stricter normalization for expression rules
    if a.relation_type == RelationTypeEnum.expression:
        if _norm_outcome_expression(a.outcome_entity) != _norm_outcome_expression(b.outcome_entity):
            return False, "outcome_incompatible"
    else:
        if _norm_outcome(a.outcome_entity) != _norm_outcome(b.outcome_entity):
            return False, "outcome_incompatible"
    return True, ""


# ── Public API ─────────────────────────────────────────────────────────────────

class RelateStage:
    """
    RELATE stage: NLI-based pairwise comparison of CanonicalRules.

    Parameters
    ----------
    model_name:
        HuggingFace NLI model name.  Defaults to the same model used by
        GroundingFilter.  The pipeline is cached at module level.
    entailment_threshold:
        Minimum entailment score to classify as SUPPORT or SCOPE_QUALIFY.
    contradiction_threshold:
        Minimum contradiction score (both directions required) to classify
        as CONTRADICT.
    max_pairs:
        Hard cap on the number of pairs compared.
    """

    def __init__(
        self,
        model_name: str = _NLI_MODEL,
        entailment_threshold: float = 0.55,
        contradiction_threshold: float = 0.65,
        max_pairs: int = MAX_PAIRS,
        batch_size: int = 16,
        device: int | str | None = None,
    ) -> None:
        self._model_name = model_name
        self._entailment_threshold = entailment_threshold
        self._contradiction_threshold = contradiction_threshold
        self._max_pairs = max_pairs
        self._batch_size = batch_size
        self._device = device

    def relate(
        self,
        rules: list[CanonicalRule],
        pmcid: str = "",
        gate=None,
    ) -> list[Relation]:
        """
        Compare all eligible pairs of CanonicalRules and return non-UNRELATED
        Relation objects.

        Parameters
        ----------
        rules:
            Output of CanonicalizeStage.canonicalize().
        pmcid:
            For logging only.
        gate:
            Callable(a, b) -> (bool, str) used to filter pairs before NLI.
            Defaults to _should_compare (strict: category + relation_type +
            subject + outcome must all match).  Pass a looser function for
            cross-paper mode where entity names differ across papers.

        Returns
        -------
        List of Relation objects (SUPPORT | CONTRADICT | SCOPE_QUALIFY only).
        """
        if len(rules) < 2:
            return []

        _gate = gate if gate is not None else _should_compare
        pipe = _get_nli_pipe(self._model_name, self._batch_size, self._device)

        # Generate eligible pairs
        eligible: list[tuple[int, int]] = []
        rejection_counts: dict[str, int] = {}
        all_pairs = list(itertools.combinations(range(len(rules)), 2))

        for i, j in all_pairs:
            ok, reason = _gate(rules[i], rules[j])
            if ok:
                eligible.append((i, j))
            else:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                logger.debug(
                    "[%s] RELATE skip (%s): [%s/%s] %r  vs  [%s/%s] %r",
                    pmcid, reason,
                    rules[i].category, rules[i].relation_type, rules[i].subject_entity,
                    rules[j].category, rules[j].relation_type, rules[j].subject_entity,
                )

        if rejection_counts:
            logger.info(
                "[%s] RELATE gate rejected %d/%d pairs: %s",
                pmcid,
                sum(rejection_counts.values()),
                len(all_pairs),
                ", ".join(f"{k}={v}" for k, v in sorted(rejection_counts.items())),
            )

        if len(eligible) > self._max_pairs:
            logger.warning(
                "[%s] RELATE: %d eligible pairs exceeds cap %d — truncating.",
                pmcid, len(eligible), self._max_pairs,
            )
            eligible = eligible[: self._max_pairs]

        if not eligible:
            return []

        # Build bidirectional NLI input
        nli_inputs_ab: list[tuple[str, str]] = []
        nli_inputs_ba: list[tuple[str, str]] = []
        for i, j in eligible:
            nli_inputs_ab.append((rules[i].predicate_text, rules[j].predicate_text))
            nli_inputs_ba.append((rules[j].predicate_text, rules[i].predicate_text))

        scores_ab = _nli_scores(nli_inputs_ab, pipe)
        scores_ba = _nli_scores(nli_inputs_ba, pipe)

        relations: list[Relation] = []
        for (i, j), s_ab, s_ba in zip(eligible, scores_ab, scores_ba):
            label = _classify_pair(
                rules[i], rules[j], s_ab, s_ba,
                self._entailment_threshold,
                self._contradiction_threshold,
            )
            if label is None:
                continue  # UNRELATED — skip

            # Store the score that is semantically meaningful for this label:
            #   SUPPORT / SCOPE_QUALIFY → entailment score (the signal that fired)
            #   CONTRADICT             → contradiction score
            if label == RelationTypeLabel.CONTRADICT:
                score_ab = s_ab.get("contradiction", 0.0)
                score_ba = s_ba.get("contradiction", 0.0)
            else:
                score_ab = s_ab.get("entailment", 0.0)
                score_ba = s_ba.get("entailment", 0.0)

            relation = Relation(
                rule_id_a=rules[i].canonical_id,
                rule_id_b=rules[j].canonical_id,
                relation_type=label,
                nli_score_a_to_b=score_ab,
                nli_score_b_to_a=score_ba,
            )
            relations.append(relation)

        logger.info(
            "[%s] RELATE: %d pairs checked → %d non-UNRELATED relations "
            "(SUPPORT=%d, CONTRADICT=%d, SCOPE_QUALIFY=%d)",
            pmcid,
            len(eligible),
            len(relations),
            sum(1 for r in relations if r.relation_type == RelationTypeLabel.SUPPORT),
            sum(1 for r in relations if r.relation_type == RelationTypeLabel.CONTRADICT),
            sum(1 for r in relations if r.relation_type == RelationTypeLabel.SCOPE_QUALIFY),
        )
        return relations
