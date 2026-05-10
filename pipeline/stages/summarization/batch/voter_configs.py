"""
Canonical voter model names, factory functions, and cascade profiles.

Single source of truth for all LLM model IDs used in the MAP cascade.
Both run_paper.py (batch + sync runners) and map_theta_sweep.py import
from here so the two are always in sync.

Profiles
--------
Different profiles trade cost vs quality. Pick one with ``get_profile(name)``
or ``get_profile(env=NLP_HISTO_PROFILE)``.

  default       — Heterogeneous cascade (Gemini + OpenAI + Claude). Production.
  smoke_haiku   — Haiku at every level. Cheapest possible smoke test for plumbing.
                  NOT for evaluating quality — agreement signal is degraded
                  because all three voters are the same model.
  dev_sonnet    — Haiku × 3 at L1, Sonnet at L2 / L3. Development-quality signal
                  while still cheaper than production heterogeneity at L1.
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

CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
CLAUDE_SONNET = "claude-sonnet-4-6-20251001"


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
    """Development quality: Haiku × 3 at L1, Sonnet at L2 / L3."""
    from .models import VoterBatchConfig
    l1 = [
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.0),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.3),
        VoterBatchConfig(CLAUDE_HAIKU, provider="claude", temperature=0.6),
    ]
    l2 = [VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.1)]
    l3 = VoterBatchConfig(CLAUDE_SONNET, provider="claude", temperature=0.0)
    return CascadeProfile(name="dev_sonnet", l1_voters=l1, l2_voters=l2, l3_voter=l3)


def _make_default_profile() -> CascadeProfile:
    return CascadeProfile(
        name="default",
        l1_voters=make_l1_voters(),
        l2_voters=make_l2_voters(),
        l3_voter=make_l3_voter(),
    )


_PROFILE_BUILDERS = {
    "default":     _make_default_profile,
    "smoke_haiku": _make_smoke_haiku_profile,
    "dev_sonnet":  _make_dev_sonnet_profile,
}


def list_profiles() -> list[str]:
    return list(_PROFILE_BUILDERS.keys())


def get_profile(name: str | None = None) -> CascadeProfile:
    """Resolve a cascade profile by name.

    If ``name`` is None, falls back to ``$NLP_HISTO_PROFILE``, then ``"default"``.
    Raises ``ValueError`` for unknown names so typos surface immediately.
    """
    resolved = name or os.environ.get("NLP_HISTO_PROFILE") or "default"
    if resolved not in _PROFILE_BUILDERS:
        raise ValueError(
            f"Unknown cascade profile {resolved!r}. "
            f"Available: {sorted(_PROFILE_BUILDERS)}"
        )
    return _PROFILE_BUILDERS[resolved]()
