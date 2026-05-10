"""
Canonical voter model names, factory functions, and cascade profiles.

Single source of truth for all LLM model IDs used in the MAP cascade.
Both run_paper.py (batch + sync runners) and map_theta_sweep.py import
from here so the two are always in sync.

Tier usage (intended)
---------------------
  Haiku   — smoke tests / plumbing only. Use to verify wiring, schema, and
            cache-key behaviour without burning real money. Quality signal
            is poor; do NOT use Haiku-only output for thesis evaluation.
  Sonnet  — development-quality checks. Use during iteration to look at real
            extraction behaviour while staying cheaper than Opus.
  Opus    — final threshold calibration / judge labels. Use only when you
            need authoritative outputs: silver-label generation, calibration
            sweeps, gold-vs-silver comparisons, contradiction-judge calls.
            Never run Opus profiles unsupervised in a loop.

Profiles
--------
Pick one with ``get_profile(name)`` (or set ``NLP_HISTO_PROFILE``).

  smoke_haiku   — Haiku at every level. Cheapest possible smoke test for
                  plumbing. NOT for evaluating quality — agreement signal is
                  degraded because all voters are the same model. **DEFAULT
                  when no profile is explicitly selected** so accidental runs
                  fail safe (cheap) rather than expensive.
  dev_sonnet    — Haiku × 3 at L1, Sonnet at L2 / L3. Development-quality
                  signal while still cheaper than full heterogeneity at L1.
  final_opus    — Haiku × 3 at L1, Sonnet at L2, Opus at L3. Reserved for
                  final calibration / judge labels. Defined here for
                  reproducibility; **not used automatically** anywhere.
                  Run only when explicitly requested.
  default       — Heterogeneous cascade (Gemini + OpenAI + Claude L1/L2,
                  Claude Sonnet L3). Original production cascade. Kept for
                  back-compat / reproducibility of pre-Phase-1 runs; no
                  longer the default fallback.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# ── Model IDs ─────────────────────────────────────────────────────────────────

# L1 — cheapest tier
GEMINI_L1 = "gemini-2.5-flash-lite"
OPENAI_L1 = "gpt-4o-mini"
CLAUDE_L1 = "claude-haiku-4-5-20251001"

# L2 — mid tier
GEMINI_L2 = "gemini-2.5-flash"
OPENAI_L2 = "gpt-4.1-mini"
CLAUDE_L2 = "claude-haiku-4-5-20251001"

# L3 — escalation / REDUCE / RULES
CLAUDE_L3 = "claude-sonnet-4-6-20251001"

CLAUDE_HAIKU  = "claude-haiku-4-5-20251001"
CLAUDE_SONNET = "claude-sonnet-4-6-20251001"
# Latest Opus snapshot. Used ONLY by the final_opus profile and any explicit
# judge / silver-label code paths — never call this from a loop without
# explicit reason, it is by far the most expensive model in the cascade.
CLAUDE_OPUS   = "claude-opus-4-7"


# ── Default-profile factories (kept for back-compat) ──────────────────────────

def make_l1_voters():
    from .models import VoterBatchConfig
    return [
        VoterBatchConfig(GEMINI_L1, provider="gemini", temperature=0.1),
        VoterBatchConfig(OPENAI_L1, provider="openai", temperature=0.1),
        VoterBatchConfig(CLAUDE_L1, provider="claude", temperature=0.1),
    ]


def make_l2_voters():
    from .models import VoterBatchConfig
    return [
        VoterBatchConfig(GEMINI_L2, provider="gemini", temperature=0.1),
        VoterBatchConfig(OPENAI_L2, provider="openai", temperature=0.1),
        VoterBatchConfig(CLAUDE_L2, provider="claude", temperature=0.3),
    ]


def make_l3_voter():
    from .models import VoterBatchConfig
    return VoterBatchConfig(CLAUDE_L3, provider="claude")


# ── Profiles ──────────────────────────────────────────────────────────────────

@dataclass
class CascadeProfile:
    """Concrete cascade selection for the batch MAP runner."""
    name: str
    l1_voters: list  # list[VoterBatchConfig]
    l2_voters: list  # list[VoterBatchConfig]
    l3_voter:  object  # VoterBatchConfig


def _make_smoke_haiku_profile() -> CascadeProfile:
    """Cheapest possible smoke test: Haiku everywhere.

    Three Haiku voters at L1 with slightly different temperatures so the
    agreement scorer has *some* variation to consume. This is a plumbing test —
    it does NOT exercise heterogeneous-voter agreement and should not be used
    for quality evaluation.
    """
    from .models import VoterBatchConfig
    l1 = [
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.0),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.3),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.6),
    ]
    l2 = [VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.2)]
    l3 = VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.0)
    return CascadeProfile(name="smoke_haiku", l1_voters=l1, l2_voters=l2, l3_voter=l3)


def _make_dev_sonnet_profile() -> CascadeProfile:
    """Development-quality cascade: Haiku × 3 at L1, Sonnet at L2 / L3.

    Use this while iterating on prompts / thresholds when you need real
    extraction quality but Opus would be wasteful. Sonnet is the workhorse;
    Haiku at L1 only filters the easy chunks.
    """
    from .models import VoterBatchConfig
    l1 = [
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.0),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.3),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.6),
    ]
    l2 = [VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.1)]
    l3 = VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.0)
    return CascadeProfile(name="dev_sonnet", l1_voters=l1, l2_voters=l2, l3_voter=l3)


def _make_final_opus_profile() -> CascadeProfile:
    """Final-calibration cascade: Haiku × 3 at L1, Sonnet at L2, Opus at L3.

    Reserved for: silver-label generation, threshold calibration sweeps, and
    gold-vs-silver comparisons where we need authoritative outputs. Opus is
    the most expensive model in the cascade; this profile must NOT be the
    automatic fallback. ``get_profile`` will only return it when the caller
    (CLI flag or ``NLP_HISTO_PROFILE``) names it explicitly.
    """
    from .models import VoterBatchConfig
    l1 = [
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.0),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.3),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.6),
    ]
    l2 = [VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.1)]
    l3 = VoterBatchConfig(CLAUDE_OPUS, provider="claude", temperature=0.0)
    return CascadeProfile(name="final_opus", l1_voters=l1, l2_voters=l2, l3_voter=l3)


def _make_default_profile() -> CascadeProfile:
    return CascadeProfile(
        name="default",
        l1_voters=make_l1_voters(),
        l2_voters=make_l2_voters(),
        l3_voter=make_l3_voter(),
    )


# Default fallback. Deliberately the cheapest profile so an unconfigured run
# can never burn Opus by accident. To use any other profile, name it
# explicitly via the ``--profile`` CLI flag or ``$NLP_HISTO_PROFILE``.
DEFAULT_PROFILE_NAME: str = "smoke_haiku"

_PROFILE_BUILDERS = {
    "smoke_haiku": _make_smoke_haiku_profile,
    "dev_sonnet":  _make_dev_sonnet_profile,
    "final_opus":  _make_final_opus_profile,
    "default":     _make_default_profile,
}


def list_profiles() -> list[str]:
    return list(_PROFILE_BUILDERS.keys())


def get_profile(name: str | None = None) -> CascadeProfile:
    """Resolve a cascade profile by name.

    Resolution order: explicit ``name`` arg → ``$NLP_HISTO_PROFILE`` →
    ``DEFAULT_PROFILE_NAME`` ("smoke_haiku"). Raises ``ValueError`` for
    unknown names so typos surface immediately rather than silently picking
    the default.

    Note: ``final_opus`` is included in the registry but is never auto-selected
    by the fallback — you must name it explicitly.
    """
    resolved = name or os.environ.get("NLP_HISTO_PROFILE") or DEFAULT_PROFILE_NAME
    if resolved not in _PROFILE_BUILDERS:
        raise ValueError(
            f"Unknown cascade profile {resolved!r}. "
            f"Available: {sorted(_PROFILE_BUILDERS)}"
        )
    return _PROFILE_BUILDERS[resolved]()
