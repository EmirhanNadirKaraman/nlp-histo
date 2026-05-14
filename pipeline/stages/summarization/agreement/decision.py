"""Shared per-chunk cascade-decision function and decision log.

Used by both the sync ``MapStage`` and the batch ``BatchSummarizationRunner``
so the cascade-decision surface is identical by construction. The previous
state had each runner re-implementing the decision branch (router vs. bare
agreement, KEEP/escalate) inline, which was a recurring source of drift
between sync and batch.

The decision is intentionally per-level:
  - sync runs it for L1 (and again for L2 in the legacy path)
  - batch runs it for L1 (and again for L2 in the legacy path)

The orchestration around it (which voters to call next, when to escalate to
L3, how to chain the batch phases) stays in each runner — only the
``given these voters, do we KEEP or escalate?`` decision is shared.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pipeline.stages.summarization.interfaces.scoring import (
    ChunkDecision,
    ScoreBundle,
)
from pipeline.stages.summarization.models import AuditableSummary

logger = logging.getLogger(__name__)


@dataclass
class ChunkOutcome:
    """Result of one cascade-level evaluation for a single chunk."""

    keep: bool
    best: AuditableSummary | None
    agreement_bundle: ScoreBundle | None
    routing_decision: object | None = None  # RoutingDecision | None — avoid circular import
    valid_voter_indices: list[int] | None = None


def evaluate_chunk(
    voters: list[AuditableSummary],
    chunk: list[dict],
    pmcid: str,
    source_text: str,
    agreement,                          # AgreementChecker
    router=None,                        # MapOutputRouter | None
) -> ChunkOutcome:
    """Decide whether one cascade level's voters agree, and select the best.

    When ``router`` is provided, it runs schema + provenance validation per
    voter, drops UNUSABLE voters, and runs the agreement gate only over the
    eligible subset. Otherwise the agreement scorer is invoked directly over
    all voters.

    Returns a ``ChunkOutcome`` with:
      - keep=True and best populated when the gate decides KEEP;
      - keep=False and best=None otherwise (the caller is responsible for
        whatever escalation makes sense at its level).

    An empty ``voters`` list short-circuits to keep=False with no bundle
    so callers don't have to special-case it.
    """
    if not voters:
        return ChunkOutcome(
            keep=False,
            best=None,
            agreement_bundle=None,
            routing_decision=None,
        )

    if router is not None:
        decision = router.route(
            voters,
            chunk=chunk,
            pmcid=pmcid,
            source_text=source_text,
        )
        if decision.decision == ChunkDecision.KEEP:
            valid = (
                [voters[i] for i in decision.valid_voter_indices]
                if decision.valid_voter_indices
                else voters
            )
            best = agreement.best(valid, bundle=decision.agreement_details)
            return ChunkOutcome(
                keep=True,
                best=best,
                agreement_bundle=decision.agreement_details,
                routing_decision=decision,
                valid_voter_indices=decision.valid_voter_indices,
            )
        return ChunkOutcome(
            keep=False,
            best=None,
            agreement_bundle=decision.agreement_details,
            routing_decision=decision,
            valid_voter_indices=decision.valid_voter_indices,
        )

    bundle = agreement.compute(voters, source_text=source_text)
    if bundle.decision == ChunkDecision.KEEP:
        best = agreement.best(voters, bundle=bundle)
        return ChunkOutcome(
            keep=True,
            best=best,
            agreement_bundle=bundle,
            routing_decision=None,
        )
    return ChunkOutcome(
        keep=False,
        best=None,
        agreement_bundle=bundle,
        routing_decision=None,
    )


# ── Per-chunk decision log ─────────────────────────────────────────────────


@dataclass
class CascadeDecisionRecord:
    """One row per cascade-level decision for offline analysis.

    Always emitted, regardless of trace_enabled, so cascade behaviour is
    inspectable even when the heavyweight TraceCollector is off.
    """

    run_id: str
    pmcid: str
    chunk_id: str
    level: str                              # "l1" | "l2" | "l3"
    voter_count: int                        # voters that survived API/parse
    eligible_voter_count: int | None        # post-router; None when router off
    decision: str                           # "keep" | "escalate" | "reject"
    gate_origin: str | None                 # "schema_gate" | "grounding_gate" | "agreement_gate" | None
    reason_codes: list[str] = field(default_factory=list)
    deferral_score: float | None = None
    best_voter_index: int | None = None     # index into the original voter list
    selected_provider: str | None = None
    selected_model: str | None = None
    cascade_signature: str | None = None
    cascade_profile: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class CascadeDecisionLog:
    """Append-only JSONL writer for cascade decisions. Thread-safe.

    Writes one line per ``record()`` call to ``output_path``. When
    ``output_path`` is None, all calls are no-ops — useful for tests and
    for explicitly disabling logging without code branches at call sites.
    """

    def __init__(self, output_path: Path | None) -> None:
        self._path: Path | None = (
            Path(output_path) if output_path is not None else None
        )
        self._lock = threading.Lock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path | None:
        return self._path

    def record(self, record: CascadeDecisionRecord) -> None:
        if self._path is None:
            return
        line = record.to_jsonl()
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def producer_from_outcome(
    outcome: ChunkOutcome,
    voter_specs: list[tuple[str, str]] | None,
) -> tuple[str, str] | None:
    """Resolve the (provider, model) of the voter selected by ``outcome``.

    Handles both router (best_index indexes the eligible subset) and
    direct-agreement (best_index indexes the original voter list) paths.
    Returns None when the outcome did not select a best voter or when
    ``voter_specs`` is missing/short.
    """
    if voter_specs is None:
        return None
    bundle = outcome.agreement_bundle
    rd = outcome.routing_decision
    best_idx = bundle.best_index if bundle is not None else None
    if best_idx is None:
        return None
    if rd is not None and rd.valid_voter_indices is not None:
        if 0 <= best_idx < len(rd.valid_voter_indices):
            global_idx = rd.valid_voter_indices[best_idx]
            if 0 <= global_idx < len(voter_specs):
                return voter_specs[global_idx]
        return None
    if 0 <= best_idx < len(voter_specs):
        return voter_specs[best_idx]
    return None


def make_decision_record(
    outcome: ChunkOutcome,
    *,
    run_id: str,
    pmcid: str,
    chunk_id: str,
    level: str,
    voter_count: int,
    cascade_signature: str | None,
    cascade_profile: str | None,
    voter_specs: list[tuple[str, str]] | None,
) -> CascadeDecisionRecord:
    """Build a ``CascadeDecisionRecord`` from a ``ChunkOutcome`` and context.

    Handles the router/no-router cases: when a routing decision is present
    the gate_origin and reason_codes come from it; otherwise they default
    to ``agreement_gate`` and the bundle's threshold path.
    """
    bundle = outcome.agreement_bundle
    rd = outcome.routing_decision

    if outcome.keep:
        decision_str = "keep"
    elif rd is not None and rd.decision == ChunkDecision.REJECT:
        decision_str = "reject"
    else:
        decision_str = "escalate"

    gate_origin = rd.gate_origin.value if rd is not None else "agreement_gate"
    reason_codes: list[str] = (
        [r.value for r in rd.reason_codes] if rd is not None else []
    )

    deferral_score: float | None = None
    if bundle is not None:
        if bundle.confidence is not None:
            deferral_score = float(bundle.confidence)
        elif bundle.embedding_agreement is not None:
            deferral_score = float(bundle.embedding_agreement)

    best_idx = bundle.best_index if bundle is not None else None
    selected_provider: str | None = None
    selected_model: str | None = None
    if outcome.keep and best_idx is not None and voter_specs is not None:
        # Router path: best_idx indexes into the eligible subset; map back
        # to the original voter index so the provider/model is accurate.
        if rd is not None and rd.valid_voter_indices is not None:
            if 0 <= best_idx < len(rd.valid_voter_indices):
                global_idx = rd.valid_voter_indices[best_idx]
                if 0 <= global_idx < len(voter_specs):
                    selected_provider, selected_model = voter_specs[global_idx]
        elif 0 <= best_idx < len(voter_specs):
            selected_provider, selected_model = voter_specs[best_idx]

    eligible_count: int | None = None
    if rd is not None and rd.valid_voter_indices is not None:
        eligible_count = len(rd.valid_voter_indices)

    return CascadeDecisionRecord(
        run_id=run_id,
        pmcid=pmcid,
        chunk_id=chunk_id,
        level=level,
        voter_count=voter_count,
        eligible_voter_count=eligible_count,
        decision=decision_str,
        gate_origin=gate_origin,
        reason_codes=reason_codes,
        deferral_score=deferral_score,
        best_voter_index=best_idx,
        selected_provider=selected_provider,
        selected_model=selected_model,
        cascade_signature=cascade_signature,
        cascade_profile=cascade_profile,
    )
