"""
Canonical voter model names and factory functions.

Single source of truth for all LLM model IDs used in the MAP cascade.
Both run_paper.py (batch + sync runners) and map_theta_sweep.py import
from here so the two are always in sync.
"""
from __future__ import annotations

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


# ── Batch voter factories ──────────────────────────────────────────────────────

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
