"""
Data models for asynchronous batch processing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pipeline.stages.summarization.config import DEFAULT_MAX_TOKENS


class BatchPhase(str, Enum):
    """State machine phase for the cascading batch run."""
    L1_SUBMITTED = "l1_submitted"
    L2_SUBMITTED = "l2_submitted"
    L3_SUBMITTED = "l3_submitted"
    COMPLETE = "complete"


@dataclass
class VoterBatchConfig:
    """Config for one voter in batch mode."""
    model: str
    provider: str           # "azure" | "claude" | "gemini" | "vertex_gemini"
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.1
    strip_thinking: bool = False  # strip <think>…</think> from response


@dataclass
class BatchRequest:
    """One LLM call packaged for a batch submission."""
    custom_id: str          # "{pmcid}__{chunk_id}__{level}__{voter_idx}"
    messages: list[dict]    # OpenAI-format role/content dicts
    model: str
    provider: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.1


@dataclass
class BatchResult:
    """Raw result from one batch request."""
    custom_id: str
    content: str | None     # JSON string of the tool-call arguments
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ProviderJob:
    """One submitted batch job at a specific provider."""
    provider: str
    job_id: str
    status: str             # "submitted" | "in_progress" | "completed" | "failed"
    model: str
    request_count: int
    output_location: str | None = None  # file_id (Azure), batch_id (Claude), GCS prefix (Vertex)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "job_id": self.job_id,
            "status": self.status,
            "model": self.model,
            "request_count": self.request_count,
            "output_location": self.output_location,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProviderJob:
        return cls(**d)


@dataclass
class BatchHandle:
    """
    JSON-serialisable state object persisted to disk between runs.

    The handle advances through BatchPhase values as batches complete.
    Call BatchSummarizationRunner.advance(handle) to progress the pipeline.
    """
    pmcid: str
    phase: BatchPhase

    # Jobs belonging to the *current* phase (replaced on each advance)
    jobs: list[ProviderJob] = field(default_factory=list)

    # Original sentence provenance (needed for agreement re-computation)
    sentences: list[dict] = field(default_factory=list)

    # chunk_id → list of sentence dicts (built once on submit)
    chunk_map: dict[str, list[dict]] = field(default_factory=dict)

    # Voter index → strip_thinking flag, per level (needed when parsing results)
    l1_strip: list[bool] = field(default_factory=list)
    l2_strip: list[bool] = field(default_factory=list)
    l3_strip: bool = False

    # Raw batch content per custom_id (stored for debugging / re-parsing)
    l1_raw: dict[str, str] = field(default_factory=dict)
    l2_raw: dict[str, str] = field(default_factory=dict)
    l3_raw: dict[str, str] = field(default_factory=dict)

    # Which chunk_ids were escalated to each level
    l2_chunk_ids: list[str] = field(default_factory=list)
    l3_chunk_ids: list[str] = field(default_factory=list)

    # Final per-chunk AuditableSummary (serialised) — populated incrementally
    finalized: dict[str, dict] = field(default_factory=dict)

    # Actual token usage accumulated from API responses, keyed by level
    token_usage: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "l1": {"input": 0, "output": 0},
        "l2": {"input": 0, "output": 0},
        "l3": {"input": 0, "output": 0},
    })

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pmcid": self.pmcid,
            "phase": self.phase.value,
            "jobs": [j.to_dict() for j in self.jobs],
            "sentences": self.sentences,
            "chunk_map": self.chunk_map,
            "l1_strip": self.l1_strip,
            "l2_strip": self.l2_strip,
            "l3_strip": self.l3_strip,
            "l1_raw": self.l1_raw,
            "l2_raw": self.l2_raw,
            "l3_raw": self.l3_raw,
            "l2_chunk_ids": self.l2_chunk_ids,
            "l3_chunk_ids": self.l3_chunk_ids,
            "finalized": self.finalized,
            "token_usage": self.token_usage,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BatchHandle:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            pmcid=data["pmcid"],
            phase=BatchPhase(data["phase"]),
            jobs=[ProviderJob.from_dict(j) for j in data.get("jobs", [])],
            sentences=data.get("sentences", []),
            chunk_map=data.get("chunk_map", {}),
            l1_strip=data.get("l1_strip", []),
            l2_strip=data.get("l2_strip", []),
            l3_strip=data.get("l3_strip", False),
            l1_raw=data.get("l1_raw", {}),
            l2_raw=data.get("l2_raw", {}),
            l3_raw=data.get("l3_raw", {}),
            l2_chunk_ids=data.get("l2_chunk_ids", []),
            l3_chunk_ids=data.get("l3_chunk_ids", []),
            finalized=data.get("finalized", {}),
            token_usage=data.get("token_usage", {
                "l1": {"input": 0, "output": 0},
                "l2": {"input": 0, "output": 0},
                "l3": {"input": 0, "output": 0},
            }),
        )
