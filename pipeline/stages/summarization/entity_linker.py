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
    from .models import CanonicalRule

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
            "threshold": 0.85,
        })
        _nlp = nlp
        logger.info("Entity linker loaded for CUI enrichment")
        return True
    except Exception as exc:
        logger.warning(
            "UMLS linker unavailable — CUI enrichment skipped: %s", exc
        )
        return False


def _best_cui_from_doc(doc) -> str | None:
    """Extract the highest-scoring CUI from a spacy Doc."""
    best_cui: str | None = None
    best_score = 0.0
    for ent in doc.ents:
        for cui, score in (getattr(ent._, "kb_ents", None) or []):
            if score > best_score:
                best_score, best_cui = score, cui
    if best_cui is None:
        for cui, score in (getattr(doc._, "kb_ents", None) or []):
            if score > best_score:
                best_score, best_cui = score, cui
    return best_cui


def enrich_rules_with_cuis(rules: list[CanonicalRule]) -> None:
    """
    Enrich CanonicalRule objects **in-place** with subject_cui and outcome_cui.

    Collects all unique entity strings, runs them through the UMLS linker in
    one batch, then writes the top CUI back onto each rule.  Silently skips
    when the linker is unavailable or enrichment fails.
    """
    if not rules or not _load():
        return

    # Collect unique non-empty entity strings
    texts: list[str] = list({
        t
        for r in rules
        for t in (r.subject_entity, r.outcome_entity)
        if t
    })
    if not texts:
        return

    cui_map: dict[str, str | None] = {}
    try:
        for doc in _nlp.pipe(texts, batch_size=32):
            cui_map[doc.text] = _best_cui_from_doc(doc)
    except Exception as exc:
        logger.warning("CUI enrichment batch failed: %s", exc)
        return

    for rule in rules:
        rule.subject_cui = cui_map.get(rule.subject_entity)
        rule.outcome_cui = cui_map.get(rule.outcome_entity)

    enriched = sum(1 for r in rules if r.subject_cui or r.outcome_cui)
    logger.info("CUI enrichment: %d/%d rules enriched", enriched, len(rules))
