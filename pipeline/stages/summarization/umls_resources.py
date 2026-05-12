"""
Process-wide singleton for the scispaCy + UMLS entity linker.

Before this module existed, ``normalize_stage`` and ``helpers/entity_linker``
each loaded their own copy of ``en_core_sci_lg`` + ``scispacy_linker``.  The
UMLS KB resident set is several GB; loading it twice in one process is the
quickest path to an OOM kill mid-pipeline.

Every component that needs the linker must obtain it through ``get_nlp()`` so
the model is loaded at most once per process.

The loader is silent-failing: if scispaCy / the model is unavailable,
``get_nlp()`` returns ``None`` and callers must degrade gracefully.
"""
from __future__ import annotations

import logging
import os
import threading

from pipeline.utils.memory_logging import MemoryLogger, get_default_memory_logger

from .umls_utils import UMLS_THRESHOLD

logger = logging.getLogger(__name__)

# Module-level MemoryLogger so scispaCy / UMLS load events share the unified
# MEMORY line format used by the per-paper pipeline runner.
_mem: MemoryLogger = get_default_memory_logger()

# ── Process-wide state ────────────────────────────────────────────────────────

_NLP = None
_LINKER = None
_AVAILABLE: bool | None = None       # None = not probed; True/False after
_LOAD_LOCK = threading.Lock()

# Disable scispaCy/UMLS entirely without uninstalling it. Useful for low-RAM
# machines and for the new --skip-umls-* CLI flags, which set this env var
# before calling into the pipeline.
_DISABLE_ENV = "NLP_HISTO_DISABLE_UMLS"


def umls_disabled() -> bool:
    """True when the kill-switch env var is set to a truthy value."""
    return os.environ.get(_DISABLE_ENV, "").lower() in {"1", "true", "yes"}


def _quiet_nmslib() -> None:
    """nmslib emits INFO lines on every index load — drop to WARNING."""
    for name in ("nmslib", "scispacy", "scispacy.candidate_generation"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _rss_mb() -> float | None:
    """Resident set size in MB, or None when psutil is not installed."""
    try:
        import psutil  # type: ignore
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _log_memory(label: str) -> None:
    """Emit a MEMORY checkpoint with stage=UMLS for the given label.

    ``label`` strings such as "before scispaCy load" / "after scispaCy load" /
    "before UMLS enrichment" map to the ``event`` field of the MEMORY line.
    """
    event = label.replace(" ", "_")
    _mem.checkpoint("UMLS", event)


def get_nlp(
    *,
    model_candidates: tuple[str, ...] = ("en_core_sci_lg", "en_core_sci_sm"),
    threshold: float = UMLS_THRESHOLD,
):
    """Return the shared scispaCy nlp object, loading it on first call.

    Returns ``None`` when scispaCy/UMLS is unavailable or has been disabled.
    Thread-safe: concurrent first-callers block on the load lock.
    """
    global _NLP, _LINKER, _AVAILABLE

    if _AVAILABLE is not None:
        return _NLP

    if umls_disabled():
        logger.info(
            "UMLS disabled via $%s — scispaCy linker will not be loaded",
            _DISABLE_ENV,
        )
        _AVAILABLE = False
        return None

    with _LOAD_LOCK:
        if _AVAILABLE is not None:        # raced another loader
            return _NLP
        _quiet_nmslib()
        _log_memory("before scispaCy load")
        try:
            import spacy                                  # type: ignore
            import scispacy                               # noqa: F401  # type: ignore
            from scispacy.linking import EntityLinker     # noqa: F401  # type: ignore

            nlp = None
            chosen: str | None = None
            for name in model_candidates:
                try:
                    logger.info("UMLS: loading %s (one-time, all stages reuse)", name)
                    nlp = spacy.load(
                        name,
                        disable=["parser", "attribute_ruler", "lemmatizer"],
                    )
                    chosen = name
                    break
                except OSError:
                    logger.info("UMLS: %s not installed, trying next", name)
                    continue
            if nlp is None:
                raise RuntimeError(
                    f"No scispaCy model found among {model_candidates!r}"
                )

            nlp.add_pipe("scispacy_linker", config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "threshold": threshold,
            })
            _NLP = nlp
            _LINKER = nlp.get_pipe("scispacy_linker")
            _AVAILABLE = True
            logger.info("UMLS: scispaCy + linker ready (%s)", chosen)
            _log_memory("after scispaCy load")
            return _NLP

        except Exception as exc:
            logger.warning(
                "UMLS: linker unavailable — downstream stages will skip CUI work: %s",
                exc,
            )
            _AVAILABLE = False
            return None


def get_linker():
    """Return the loaded scispacy_linker pipe, or None when unavailable."""
    if _AVAILABLE is None:
        get_nlp()
    return _LINKER


def is_available() -> bool:
    return bool(_AVAILABLE)


# Test / reset hook (do not call from production code).

def _reset_for_tests() -> None:
    global _NLP, _LINKER, _AVAILABLE
    _NLP = None
    _LINKER = None
    _AVAILABLE = None
