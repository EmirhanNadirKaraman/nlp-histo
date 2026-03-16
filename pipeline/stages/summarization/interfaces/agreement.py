from __future__ import annotations

from typing import Protocol, runtime_checkable

from pipeline.stages.summarization.models import AuditableSummary


@runtime_checkable
class AgreementStrategy(Protocol):
    """
    Contract for voter-agreement scoring in the ABC MAP cascade.

    Implementations receive the list of AuditableSummary outputs from all
    Level-1 voters for a single chunk and return a scalar score in [0, 1].

    1.0 → all voters fully agree.
    0.0 → voters share nothing in common.
    """

    def compute(self, outputs: list[AuditableSummary]) -> float:
        """Return agreement score in [0, 1]."""
        ...
