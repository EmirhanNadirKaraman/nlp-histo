"""Protocol interfaces for the summarization pipeline."""
from .agreement import MapOutputScorer
from .contradiction import ContradictionChecker
from .grounding import GroundingChecker
from .scoring import AgreementContext, ChunkDecision, ScoreBundle, VoterContext
from .similarity import SimilarityStrategy

__all__ = [
    "MapOutputScorer",
    "GroundingChecker",
    "ContradictionChecker",
    "ScoreBundle",
    "ChunkDecision",
    "AgreementContext",
    "VoterContext",
    "SimilarityStrategy",
]
