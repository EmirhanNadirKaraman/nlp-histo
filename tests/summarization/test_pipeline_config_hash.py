"""Tests for `compute_pipeline_config_hash` (PERSISTENCE_TODOS #8).

Verifies the hash is:
- deterministic for identical inputs
- sensitive to every input dimension
- key-order-independent for dict inputs
- coerces Pydantic / dataclasses / enums via _to_jsonable

No LLM / API calls.
"""
from __future__ import annotations

import dataclasses
from enum import Enum

from pipeline.stages.summarization.persistence import compute_pipeline_config_hash


# ── Helpers ──────────────────────────────────────────────────────────────────

def _base_kwargs() -> dict:
    return dict(
        config={"map": {"theta": 0.7}, "relate": {"entailment_threshold": 0.5}},
        thresholds={
            "grounding_threshold": 0.6,
            "entailment_threshold": 0.5,
            "contradiction_threshold": 0.55,
            "map_theta": 0.7,
            "map_reject_theta": 0.3,
            "contradiction_similarity_threshold": 0.8,
        },
        models={
            "voter_models": ["deepseek", "gemini-flash-lite", "mistral-large"],
            "level2_voter_models": ["gemini-flash", "kimi", "haiku"],
            "escalation_model": "sonnet-4-6",
            "scorer": "embedding",
        },
        schema_version="map_v1_explicit_direction",
        prompt_version="map_prompt_v1_explicit_enums",
        cascade_signature="abc123def456",
    )


# ── Determinism ──────────────────────────────────────────────────────────────

def test_identical_inputs_yield_identical_hash():
    h1 = compute_pipeline_config_hash(**_base_kwargs())
    h2 = compute_pipeline_config_hash(**_base_kwargs())
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 16


def test_hash_is_dict_key_order_independent():
    a = _base_kwargs()
    b = _base_kwargs()
    # Re-pack thresholds in a different key order.
    b["thresholds"] = {k: a["thresholds"][k] for k in reversed(list(a["thresholds"]))}
    assert compute_pipeline_config_hash(**a) == compute_pipeline_config_hash(**b)


# ── Sensitivity ──────────────────────────────────────────────────────────────

def test_changing_threshold_changes_hash():
    base = _base_kwargs()
    bumped = _base_kwargs()
    bumped["thresholds"]["grounding_threshold"] = 0.65  # was 0.6
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_changing_model_changes_hash():
    base = _base_kwargs()
    swapped = _base_kwargs()
    swapped["models"]["escalation_model"] = "opus-4-7"
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**swapped)


def test_changing_config_changes_hash():
    base = _base_kwargs()
    edited = _base_kwargs()
    edited["config"]["map"]["theta"] = 0.65  # was 0.7
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**edited)


def test_changing_schema_version_changes_hash():
    base = _base_kwargs()
    bumped = _base_kwargs()
    bumped["schema_version"] = "map_v2_new"
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_changing_prompt_version_changes_hash():
    base = _base_kwargs()
    bumped = _base_kwargs()
    bumped["prompt_version"] = "map_prompt_v2"
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_changing_cascade_signature_changes_hash():
    base = _base_kwargs()
    bumped = _base_kwargs()
    bumped["cascade_signature"] = "ffffffffffff"
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**bumped)


def test_adding_threshold_key_changes_hash():
    base = _base_kwargs()
    extended = _base_kwargs()
    extended["thresholds"]["new_dial"] = 0.42
    assert compute_pipeline_config_hash(**base) != compute_pipeline_config_hash(**extended)


# ── Coercion: dataclasses + enums survive through _to_jsonable ───────────────

class _Mode(Enum):
    A = "a"
    B = "b"


@dataclasses.dataclass
class _MiniCfg:
    threshold: float
    mode: _Mode


def test_dataclass_and_enum_inputs_are_coerced_deterministically():
    cfg_a = _MiniCfg(threshold=0.5, mode=_Mode.A)
    cfg_a2 = _MiniCfg(threshold=0.5, mode=_Mode.A)
    cfg_b = _MiniCfg(threshold=0.5, mode=_Mode.B)
    h_a  = compute_pipeline_config_hash(config=cfg_a)
    h_a2 = compute_pipeline_config_hash(config=cfg_a2)
    h_b  = compute_pipeline_config_hash(config=cfg_b)
    assert h_a == h_a2
    assert h_a != h_b


# ── Empty / None inputs still hash deterministically ─────────────────────────

def test_all_none_inputs_hash_deterministically():
    h1 = compute_pipeline_config_hash()
    h2 = compute_pipeline_config_hash()
    assert h1 == h2
    # And differs from anything with real content.
    assert h1 != compute_pipeline_config_hash(schema_version="x")
