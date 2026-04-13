"""
NORMALIZE stage: Finding[] → NormalFinding[]

Two operations:
1. Entity normalization — resolve surface-form variants to canonical names.
   Resolution order:
     a. UMLS linker via scispaCy (en_core_sci_lg, threshold 0.7) — loaded
        lazily and cached at module level; skipped if scispaCy unavailable.
     b. Hand-curated synonym dictionary (_SYNONYMS) — covers rare entities
        that UMLS does not link (CEAN, MGA, …).
     c. Identity fallback — stripped input returned unchanged.

2. Conditional dedup — findings that share the same
   (text_element_id, subject_entity, outcome_entity, relation_type) after
   normalization are merged into one NormalFinding (provenance union).
   Findings that differ on any of those keys stay separate even if they
   came from the same text element.  Findings where subject_entity or
   outcome_entity is None, or relation_type is unclear, are kept as
   individual NormalFindings without attempting dedup.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

from .models import DirectionEnum, Finding, FindingScope, NormalFinding, RelationTypeEnum, SourceSpan

logger = logging.getLogger(__name__)

# ── Synonym dictionary ─────────────────────────────────────────────────────────
# Canonical source of truth is synonyms.yaml in this directory.
# The hardcoded dict below is the fallback used when the YAML file is absent
# or unreadable (e.g. in tests that don't have the full repo on disk).

_SYNONYMS_YAML = Path(__file__).parent / "synonyms.yaml"

_SYNONYMS_FALLBACK: dict[str, str] = {
    # Overall survival
    "overall survival":          "OS",
    "os":                        "OS",
    "overall survival rate":     "OS",
    # Progression-free survival
    "progression-free survival": "PFS",
    "pfs":                       "PFS",
    "progression free survival": "PFS",
    # Disease-free survival
    "disease-free survival":     "DFS",
    "dfs":                       "DFS",
    # Common biomarkers — surface variants
    "cd30":                      "CD30",
    "ber-h2":                    "CD30",
    "ki-67":                     "Ki-67",
    "ki67":                      "Ki-67",
    "mib-1":                     "Ki-67",
    "mib1":                      "Ki-67",
    "bcl-2":                     "BCL2",
    "bcl2":                      "BCL2",
    "bcl-6":                     "BCL6",
    "bcl6":                      "BCL6",
    "myc":                       "MYC",
    "c-myc":                     "MYC",
    "pd-l1":                     "PD-L1",
    "pdl1":                      "PD-L1",
    "cd20":                      "CD20",
    "cd3":                       "CD3",
    "cd8":                       "CD8",
    "cd4":                       "CD4",
    "cd10":                      "CD10",
    "cd31":                      "CD31",
    "cd34":                      "CD34",
    "cd138":                     "CD138",
    "cd56":                      "CD56",
    "cd30 expression":           "CD30",
    "cd30 positivity":           "CD30",
    "bcl2 expression":           "BCL2",
    "myc expression":            "MYC",
    # Treatment
    "r-chop":                    "R-CHOP",
    "r chop":                    "R-CHOP",
    "rituximab-chop":            "R-CHOP",
    "chop":                      "CHOP",
    # Disease entity variants — extend as new papers are processed
    # CEAN: Cutaneous Epithelioid Angiomatous Nodule
    "caen":                                      "CEAN",
    "cean":                                      "CEAN",
    "cutaneous epithelioid angiomatous nodule":   "CEAN",
    "epithelioid angiomatous nodule":             "CEAN",
    # Cell population capitalisation variants
    "tumour cells":              "tumour cells",
    "tumor cells":               "tumour cells",
    "tumour cell":               "tumour cells",
    "tumor cell":                "tumour cells",
    "neoplastic cells":          "tumour cells",
    # Disease subtypes — keep as-is (subtype normalization is scope's job)
}


def _load_synonyms() -> dict[str, str]:
    """
    Load the synonym dictionary from synonyms.yaml if available,
    otherwise fall back to the hardcoded dict.
    """
    try:
        import yaml  # type: ignore
        with open(_SYNONYMS_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            loaded = {str(k).lower(): str(v) for k, v in data.items()}
            logger.debug("NORMALIZE: loaded %d synonyms from %s", len(loaded), _SYNONYMS_YAML)
            return loaded
    except FileNotFoundError:
        logger.debug("NORMALIZE: synonyms.yaml not found, using hardcoded fallback")
    except Exception as exc:
        logger.warning("NORMALIZE: failed to load synonyms.yaml (%s), using hardcoded fallback", exc)
    return dict(_SYNONYMS_FALLBACK)


_SYNONYMS: dict[str, str] = _load_synonyms()


# ── UMLS entity resolver ───────────────────────────────────────────────────────
# Module-level cache: entity surface text → canonical name string.
# Shared across all NormalizeStage instances in the same process.
_UMLS_CACHE: dict[str, str] = {}

# Module-level scispaCy model + linker singleton (None until first use).
_SPACY_NLP = None
_SPACY_LINKER = None
_SPACY_AVAILABLE: bool | None = None   # None = not yet probed

_UMLS_THRESHOLD = 0.85

# UMLS semantic types that are never valid entity normalizations in a
# histopathology / clinical text mining context.  A concept whose *only*
# semantic types fall in this set is discarded regardless of linker score.
_JUNK_SEMANTIC_TYPES = frozenset({
    "T001",  # Organism
    "T002",  # Plant
    "T004",  # Fungus
    "T005",  # Virus
    "T007",  # Bacterium
    "T008",  # Animal
    "T009",  # Invertebrate
    "T010",  # Vertebrate
    "T011",  # Amphibian
    "T012",  # Bird
    "T013",  # Fish
    "T014",  # Reptile
    "T015",  # Mammal  ← Cetacea lives here
    "T083",  # Geographic Area
    "T093",  # Health Care Related Organization
    "T097",  # Professional or Occupational Group
})

# Short all-caps tokens (≤5 chars) are acronyms; the linker almost always
# maps them to the wrong concept via string similarity.
import re as _re
_ACRONYM_RE = _re.compile(r'^[A-Z]{2,5}$')


def _is_junk_umls(canonical_name: str, types: list[str] | None, entity_text: str) -> bool:
    """Return True if a UMLS hit should be discarded."""
    type_set = set(types or [])
    if type_set and type_set.issubset(_JUNK_SEMANTIC_TYPES):
        return True
    if _ACRONYM_RE.match(entity_text.strip()) and type_set & _JUNK_SEMANTIC_TYPES:
        return True
    return False


def _probe_spacy() -> bool:
    """
    Try to load a scispaCy model with UMLS linker.  Attempts en_core_sci_lg
    first; falls back to en_core_sci_sm if lg is not installed.  Sets
    module-level _SPACY_NLP, _SPACY_LINKER, _SPACY_AVAILABLE.
    Returns True on success.
    """
    global _SPACY_NLP, _SPACY_LINKER, _SPACY_AVAILABLE
    try:
        import spacy                                # type: ignore
        import scispacy                             # noqa: F401  # type: ignore
        from scispacy.linking import EntityLinker   # noqa: F401  # type: ignore

        for model_name in ("en_core_sci_lg", "en_core_sci_sm"):
            try:
                logger.info("NORMALIZE: loading %s + UMLS linker (one-time)…", model_name)
                nlp = spacy.load(model_name, disable=["parser", "attribute_ruler", "lemmatizer"])
                nlp.add_pipe("scispacy_linker", config={
                    "resolve_abbreviations": True,
                    "linker_name": "umls",
                    "threshold": _UMLS_THRESHOLD,
                })
                _SPACY_NLP = nlp
                _SPACY_LINKER = nlp.get_pipe("scispacy_linker")
                _SPACY_AVAILABLE = True
                logger.info("NORMALIZE: scispaCy + UMLS linker ready (%s).", model_name)
                return True
            except OSError:
                logger.info("NORMALIZE: %s not found, trying next…", model_name)
                continue

        raise RuntimeError("No scispaCy model found (tried en_core_sci_lg, en_core_sci_sm)")
    except Exception as exc:
        logger.warning("NORMALIZE: scispaCy unavailable (%s) — using dict fallback only.", exc)
        _SPACY_AVAILABLE = False
        return False


def _umls_canonical(text: str) -> str | None:
    """
    Return the UMLS preferred name for *text*, or None if no confident link.

    Uses the module-level scispaCy model.  Results are cached in _UMLS_CACHE.
    """
    global _SPACY_AVAILABLE

    if text in _UMLS_CACHE:
        return _UMLS_CACHE[text]

    # Probe on first call
    if _SPACY_AVAILABLE is None:
        _probe_spacy()

    if not _SPACY_AVAILABLE:
        return None

    try:
        doc = _SPACY_NLP(text)  # type: ignore[arg-type]
        best_cui: str | None = None
        best_score: float = 0.0

        # Prefer entity spans; fall back to doc-level kb_ents
        ents = doc.ents if doc.ents else [doc]
        for span in ents:
            kb_ents = getattr(span._, "kb_ents", [])
            for cui, score in kb_ents:
                if score > best_score:
                    best_score = score
                    best_cui = cui

        canonical: str | None = None
        if best_cui and best_score >= _UMLS_THRESHOLD:
            concept = _SPACY_LINKER.kb.cui_to_entity.get(best_cui)  # type: ignore[union-attr]
            if concept:
                types = list(concept.types) if concept.types else []
                if not _is_junk_umls(concept.canonical_name, types, text):
                    canonical = concept.canonical_name

        _UMLS_CACHE[text] = canonical  # type: ignore[assignment]  # cache None too
        return canonical
    except Exception as exc:
        logger.debug("NORMALIZE: UMLS lookup failed for %r: %s", text, exc)
        return None


def normalize_entity(name: str | None) -> str | None:
    """
    Return the canonical form of an entity name, or None if input is None.

    Resolution order:
      1. UMLS linker (scispaCy) — preferred name from UMLS knowledge base.
      2. Hand-curated _SYNONYMS dict — covers rare/unlinked entities.
      3. Identity — stripped input returned unchanged.
    """
    if name is None:
        return None
    stripped = name.strip()
    # 1. UMLS
    umls = _umls_canonical(stripped)
    if umls:
        return umls
    # 2. Dict
    return _SYNONYMS.get(stripped.lower(), stripped)


# ── Assertion status inference ─────────────────────────────────────────────────

# Ordered: negative checked first so "not expressed" beats "expressed"
_NEGATIVE_TRIGGERS = (
    "not immunoreactive",
    "not reactive",
    "non-reactive",
    "not identified",
    "not detected",
    "not expressed",
    "no recurrence",
    "no metastasis",
    "no pleomorphism",
    "no cytologic atypia",
    "no atypia",
    "no mitoses",
    "not seen",
    "not observed",
    "non-",
    "absent",
    "negative",
    "lacking",
    "without",
    "no ",
    "not ",
)

_POSITIVE_TRIGGERS = (
    "present",
    "positive",
    "expressed",
    "immunoreactive",
    "reactive",
    "identified",
    "detected",
    "seen",
    "observed",
    "exhibits",
    "shows",
    "found",
)


def infer_direction(claim: str) -> DirectionEnum:
    """
    Infer direction from claim text using ordered keyword matching.

    Negative triggers are checked before positive triggers so that phrases
    like "not expressed" correctly return negative rather than positive.

    Returns DirectionEnum.unclear when no trigger matches.
    """
    lower = claim.lower()
    for trigger in _NEGATIVE_TRIGGERS:
        if trigger in lower:
            return DirectionEnum.negative
    for trigger in _POSITIVE_TRIGGERS:
        if trigger in lower:
            return DirectionEnum.positive
    return DirectionEnum.unclear



# ── Dedup key ──────────────────────────────────────────────────────────────────

def _dedup_key(
    text_element_id: int | None,
    subject: str | None,
    outcome: str | None,
    relation_type: RelationTypeEnum,
) -> str | None:
    """
    Return a grouping key for conditional dedup, or None if any field that
    would make the key meaningful is absent (→ finding kept as-is, not merged).

    text_element_id=None means the evidence string was missing or malformed;
    treat as ungroupable to avoid merging unrelated findings under a shared
    sentinel value.
    """
    if (
        subject is None
        or outcome is None
        or relation_type is RelationTypeEnum.unclear
        or text_element_id is None
    ):
        return None
    return f"{text_element_id}|{subject}|{outcome}|{relation_type.value}"


# ── Source span extraction ─────────────────────────────────────────────────────

def _spans_from_finding(f: Finding) -> list[SourceSpan]:
    """
    Convert a Finding's evidence list to SourceSpan objects.

    Evidence strings have the format "S{i}|{pmcid}|{te_id}".
    Malformed entries are skipped with a debug log.
    """
    spans = []
    for ev in f.evidence:
        parts = ev.split("|")
        if len(parts) != 3:
            logger.debug("Skipping malformed evidence string: %r", ev)
            continue
        sentence_id, pmcid, te_id_str = parts
        try:
            te_id = int(te_id_str)
        except ValueError:
            logger.debug("Non-integer text_element_id in evidence: %r", ev)
            continue
        spans.append(SourceSpan(
            sentence_id=sentence_id,
            pmcid=pmcid,
            text_element_id=te_id,
            verbatim=f.verbatim_support,
        ))
    return spans


def _normal_id(pmcid: str, subject: str | None, outcome: str | None,
               relation_type: RelationTypeEnum, te_ids: list[int]) -> str:
    raw = f"{subject}|{outcome}|{relation_type.value}|{'_'.join(map(str, sorted(te_ids)))}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"NF_{pmcid}_{digest}"


def _best_scope(findings: list[Finding]) -> FindingScope:
    """
    Return the scope from the finding with the highest grounding_score,
    falling back to the first finding if none have scores.
    """
    scored = [(f.grounding_score or 0.0, f) for f in findings]
    return max(scored, key=lambda x: x[0])[1].scope or FindingScope()


def _mean_score(findings: list[Finding]) -> float | None:
    scores = [f.grounding_score for f in findings if f.grounding_score is not None]
    return sum(scores) / len(scores) if scores else None


# ── Public API ─────────────────────────────────────────────────────────────────

class NormalizeStage:
    """
    NORMALIZE stage: entity normalization + conditional dedup.

    Parameters
    ----------
    extra_synonyms:
        Additional synonym mappings to merge with the built-in dictionary.
        Keys must be lowercase surface forms.
    """

    def __init__(self, extra_synonyms: dict[str, str] | None = None) -> None:
        self._synonyms = dict(_SYNONYMS)
        if extra_synonyms:
            self._synonyms.update(extra_synonyms)

    def normalize(self, findings: list[Finding], pmcid: str) -> list[NormalFinding]:
        """
        Normalize and deduplicate findings for one paper.

        Returns one NormalFinding per distinct (te_id, subject, outcome,
        direction) group, plus one per ungroupable finding (missing fields).
        """
        if not findings:
            return []

        # Step 1: normalize entity names in-place on a working copy
        normalized = self._normalize_entities(findings)

        # Step 2: partition into groupable and ungroupable
        groupable: dict[str, list[Finding]] = defaultdict(list)
        ungroupable: list[Finding] = []

        for f, te_id in normalized:
            key = _dedup_key(te_id, f.subject_entity, f.outcome_entity, f.relation_type)
            if key is None:
                ungroupable.append(f)
            else:
                groupable[key].append(f)

        results: list[NormalFinding] = []

        # Step 3: merge each groupable cluster
        for key, group in groupable.items():
            results.append(self._merge(group, pmcid))

        # Step 4: wrap each ungroupable finding individually
        for f in ungroupable:
            results.append(self._wrap_single(f, pmcid))

        logger.info(
            "[%s] NORMALIZE: %d findings → %d NormalFindings "
            "(%d grouped clusters, %d ungroupable)",
            pmcid, len(findings), len(results),
            len(groupable), len(ungroupable),
        )
        return results

    # ── Internals ──────────────────────────────────────────────────────────────

    def _normalize_entities(
        self, findings: list[Finding]
    ) -> list[tuple[Finding, int]]:
        """
        Return (finding_with_normalized_entities, text_element_id) pairs.
        text_element_id is extracted from the first evidence string; 0 if absent.
        """
        result = []
        for f in findings:
            # Extract te_id from first evidence string for grouping key.
            # None means evidence was absent or malformed → treated as
            # ungroupable so unrelated findings are never merged.
            te_id: int | None = None
            if f.evidence:
                parts = f.evidence[0].split("|")
                if len(parts) == 3:
                    try:
                        te_id = int(parts[2])
                    except ValueError:
                        pass

            # Apply the keyword heuristic when the LLM could not determine
            # direction — it may recover a clear polarity the LLM missed.
            # When the LLM supplied a concrete direction, preserve it.
            direction = (
                infer_direction(f.claim)
                if (f.direction is None or f.direction == DirectionEnum.unclear)
                else f.direction
            )
            normed = f.model_copy(update={
                "subject_entity": self._norm(f.subject_entity),
                "outcome_entity": self._norm(f.outcome_entity),
                "direction": direction,
            })
            result.append((normed, te_id))
        return result

    def _norm(self, name: str | None) -> str | None:
        """
        Resolve an entity name to its canonical form.

        Resolution order:
          1. Instance synonym dict (built-in + extra_synonyms) — domain
             knowledge takes priority; covers acronyms the linker gets wrong.
          2. UMLS linker (module-level scispaCy singleton).
          3. Identity fallback — stripped input returned unchanged.
        """
        if name is None:
            return None
        stripped = name.strip()
        # 1. Instance dict (includes extra_synonyms)
        from_dict = self._synonyms.get(stripped.lower())
        if from_dict is not None:
            return from_dict
        # 2. UMLS
        umls = _umls_canonical(stripped)
        if umls:
            return umls
        # 3. Identity
        return stripped

    def _merge(self, group: list[Finding], pmcid: str) -> NormalFinding:
        # Representative finding: highest grounding_score
        rep = max(group, key=lambda f: f.grounding_score or 0.0)

        # Deduplicated spans (by sentence_id + te_id)
        seen: set[tuple[str, int]] = set()
        spans: list[SourceSpan] = []
        te_ids: list[int] = []
        for f in group:
            for span in _spans_from_finding(f):
                key = (span.sentence_id, span.text_element_id)
                if key not in seen:
                    seen.add(key)
                    spans.append(span)
                if span.text_element_id not in te_ids:
                    te_ids.append(span.text_element_id)

        unique_pmcids = sorted({s.pmcid for s in spans}) or [pmcid]

        direction = (
            infer_direction(rep.claim)
            if (rep.direction is None or rep.direction == DirectionEnum.unclear)
            else rep.direction
        )
        return NormalFinding(
            normal_id=_normal_id(pmcid, rep.subject_entity, rep.outcome_entity,
                                 rep.relation_type, te_ids),
            subject_entity=rep.subject_entity,
            outcome_entity=rep.outcome_entity,
            relation_type=rep.relation_type,
            direction=direction,
            category=rep.category,
            predicate_text=rep.claim,
            scope=rep.scope or FindingScope(),
            source_finding_ids=[],   # populated by Phase 3+ when finding_id exists
            evidence=spans,
            pmcids=unique_pmcids,
            mean_grounding_score=_mean_score(group),
        )

    def _wrap_single(self, f: Finding, pmcid: str) -> NormalFinding:
        spans = _spans_from_finding(f)
        te_ids = [s.text_element_id for s in spans]
        unique_pmcids = sorted({s.pmcid for s in spans}) or [pmcid]
        direction = (
            infer_direction(f.claim)
            if (f.direction is None or f.direction == DirectionEnum.unclear)
            else f.direction
        )
        return NormalFinding(
            normal_id=_normal_id(pmcid, f.subject_entity, f.outcome_entity,
                                 f.relation_type, te_ids),
            subject_entity=f.subject_entity,
            outcome_entity=f.outcome_entity,
            relation_type=f.relation_type,
            direction=direction,
            category=f.category,
            predicate_text=f.claim,
            scope=f.scope or FindingScope(),
            source_finding_ids=[],
            evidence=spans,
            pmcids=unique_pmcids,
            mean_grounding_score=f.grounding_score,
        )
