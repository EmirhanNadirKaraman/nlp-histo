"""Embedding provider implementations.

Keeps provider-specific API calls out of scorer logic so scorers stay
testable and provider-agnostic.
"""
from __future__ import annotations

from typing import Callable

# Type alias for any callable that maps strings to embedding vectors.
EmbedFn = Callable[[list[str]], list[list[float]]]


class OpenAIEmbedder:
    """
    Callable wrapper around the OpenAI Embeddings API.

    Implements EmbedFn so it can be injected into any scorer that accepts an
    embedding function.  The openai package is imported lazily so the rest of
    the pipeline does not require it at import time.

    Parameters
    ----------
    model:
        OpenAI embedding model name.  Defaults to text-embedding-3-small.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import openai  # lazy import — optional dependency

        response = openai.OpenAI().embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"
