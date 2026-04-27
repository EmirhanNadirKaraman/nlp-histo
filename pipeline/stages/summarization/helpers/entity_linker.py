"""
Lazy-loaded UMLS entity linker singleton for CUI enrichment of CanonicalRules.

Reuses the same scispacy model and linker config as named_entity_recognition/ner.py
but is kept separate so the summarization pipeline does not hard-depend on scispacy.
If scispacy / en_core_sci_lg is unavailable the enrichment silently no-ops.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import CanonicalRule

from ..umls_utils import UMLS_THRESHOLD, best_cui as _best_cui

logger = logging.getLogger(__name__)

# Module-level singletons — loaded at most once per process.
_nlp = None
_load_attempted = False


def _load() -> bool:
    """Load the scispacy linker model. Returns True if ready."""
    global _nlp, _load_attempted
    if _load_attempted:
        return _nlp is not None
    _load_attempted = True
    try:
        import spacy
        import scispacy  # noqa: F401  — registers scispacy components
        from scispacy.linking import EntityLinker  # noqa: F401  — registers factory

        nlp = spacy.load(
            "en_core_sci_lg",
            disable=["parser", "attribute_ruler", "lemmatizer"],
        )
        nlp.add_pipe("scispacy_linker", config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": UMLS_THRESHOLD,
        })
        _nlp = nlp
        logger.info("Entity linker loaded for CUI enrichment")
        return True
    except Exception as exc:
        logger.warning(
            "UMLS linker unavailable — CUI enrichment skipped: %s", exc
        )
        return False


def _best_cui_from_doc(doc, kb) -> str | None:
    """Extract the highest-scoring non-junk CUI from a spacy Doc."""
    spans = doc.ents if doc.ents else [doc]
    kb_ents_by_span = [getattr(span._, "kb_ents", None) or [] for span in spans]
    return _best_cui(kb_ents_by_span, kb, doc.text)


def enrich_rules_with_cuis(rules: list[CanonicalRule]) -> None:
    """
    Fill in subject_cui and outcome_cui on CanonicalRule objects **in-place**.

    Only processes rules that are missing at least one CUI — rules whose CUIs
    were already set during NORMALIZE (via NormalFinding propagation) are
    skipped.  Silently no-ops when the linker is unavailable or enrichment fails.
    """
    rules_to_enrich = [r for r in rules if not (r.subject_cui and r.outcome_cui)]
    if not rules_to_enrich or not _load():
        return

    # Collect unique non-empty entity strings from rules that need enrichment
    texts: list[str] = list({
        t
        for r in rules_to_enrich
        for t in (r.subject_entity, r.outcome_entity)
        if t
    })
    if not texts:
        return

    kb = _nlp.get_pipe("scispacy_linker").kb
    cui_map: dict[str, str | None] = {}
    try:
        for doc in _nlp.pipe(texts, batch_size=32):
            cui_map[doc.text] = _best_cui_from_doc(doc, kb)
    except Exception as exc:
        logger.warning("CUI enrichment batch failed: %s", exc)
        return

    for rule in rules_to_enrich:
        if not rule.subject_cui:
            rule.subject_cui = cui_map.get(rule.subject_entity)
        if not rule.outcome_cui:
            rule.outcome_cui = cui_map.get(rule.outcome_entity)

    enriched = sum(1 for r in rules if r.subject_cui or r.outcome_cui)
    logger.info("CUI enrichment: %d/%d rules enriched", enriched, len(rules))
