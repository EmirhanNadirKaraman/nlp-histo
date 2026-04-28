"""Sample source cases from the TextElement table."""
from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Sections to exclude (references, acknowledgements, etc.)
_SKIP_SECTIONS = {
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "funding", "conflicts of interest", "conflict of interest",
    "author contributions", "supplementary", "appendix",
}

# Minimum paragraph length to be useful
_MIN_WORDS = 40


def _is_useful(te) -> bool:
    """Filter out boilerplate text elements."""
    if te.word_count and te.word_count < _MIN_WORDS:
        return False
    path_lower = (te.path_string or "").lower()
    for skip in _SKIP_SECTIONS:
        if skip in path_lower:
            return False
    return True


def sample_source_cases(
    n: int,
    seed: int = 42,
    pmcids: List[str] | None = None,
) -> List[dict]:
    """
    Sample n TextElement rows from the database.

    Returns list of dicts matching SourceCase schema.
    No pipeline imports — only database access.
    """
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

    from database import get_db_connection
    from database.models import TextElement, Document

    db = get_db_connection()
    with db.session_scope() as session:
        q = (
            session.query(TextElement, Document.pmcid)
            .join(Document, TextElement.document_id == Document.id)
        )
        if pmcids:
            q = q.filter(Document.pmcid.in_(pmcids))

        candidates = []
        for te, pmcid in q.all():
            text = te.text_content or ""
            wc = te.word_count or len(text.split())
            path = te.path_string or ""
            if wc < _MIN_WORDS:
                continue
            if any(skip in path.lower() for skip in _SKIP_SECTIONS):
                continue
            candidates.append({
                "te_id": te.id,
                "pmcid": pmcid,
                "text": text,
                "path_string": path,
                "word_count": wc,
            })

    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n, len(candidates)))

    cases = []
    for c in chosen:
        case_id = f"{c['pmcid']}|{c['te_id']}"
        cases.append({
            "case_id": case_id,
            "pmcid": c["pmcid"],
            "te_id": c["te_id"],
            "text": c["text"],
            "path_string": c["path_string"],
        })

    logger.info("Sampled %d / %d candidate paragraphs", len(cases), len(candidates))
    return cases
