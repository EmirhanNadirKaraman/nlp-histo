"""Protocol interfaces for the summarization pipeline."""
from .agreement import MapOutputScorer
from .contradiction import ContradictionChecker
from .grounding import GroundingChecker
from .scoring import ChunkDecision, ScoreBundle

__all__ = [
    "MapOutputScorer",
    "GroundingChecker",
    "ContradictionChecker",
    "ScoreBundle",
    "ChunkDecision",
]
