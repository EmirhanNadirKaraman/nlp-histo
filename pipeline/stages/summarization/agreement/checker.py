"""AgreementChecker — thin wrapper that adds theta and helper methods."""
from __future__ import annotations

import logging
from typing import Optional

from pipeline.stages.summarization.interfaces.agreement import AgreementStrategy
from pipeline.stages.summarization.models import AuditableSummary
from .embedding import EmbeddingAgreement

logger = logging.getLogger(__name__)


class AgreementChecker:
    """
    Wraps an AgreementStrategy with a theta threshold and helper methods.

    Parameters
    ----------
    strategy:
        Any object implementing AgreementStrategy.  Defaults to
        EmbeddingAgreement (OpenAI text-embedding-3-small).
    theta:
        Minimum score to consider voters "confident".  Default 0.7; lower to
        ~0.6 when using same-provider voters with temperature diversity.
    """

    def __init__(
        self,
        strategy: Optional[AgreementStrategy] = None,
        theta: float = 0.7,
    ) -> None:
        self.theta = theta
        self._strategy = strategy or EmbeddingAgreement()

    def compute(self, outputs: list[AuditableSummary]) -> float:
        """Return agreement score in [0, 1] via the configured strategy."""
        if len(outputs) < 2:
            return 1.0
        score = self._strategy.compute(outputs)
        logger.debug(
            "Agreement [%s]: %.2f (theta=%.2f, voters=%d)",
            type(self._strategy).__name__,
            score,
            self.theta,
            len(outputs),
        )
        return score

    def confident(self, outputs: list[AuditableSummary]) -> bool:
        """True if compute(outputs) >= theta."""
        return self.compute(outputs) >= self.theta

    def best(self, outputs: list[AuditableSummary]) -> AuditableSummary:
        """Return the voter output with the most findings."""
        return max(outputs, key=lambda o: len(o.findings))
