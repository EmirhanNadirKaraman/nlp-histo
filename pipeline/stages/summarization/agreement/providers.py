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


class GeminiEmbedder:
    """
    Callable wrapper around the Google Gemini Embeddings API.

    Uses the google-genai SDK (google.genai).  Drop-in replacement for
    OpenAIEmbedder — implements the same EmbedFn interface.

    Parameters
    ----------
    model:
        Gemini embedding model name.  Defaults to gemini-embedding-001.
    task_type:
        Gemini task type hint.  "SEMANTIC_SIMILARITY" works well for
        agreement scoring between extracted claims.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        task_type: str = "SEMANTIC_SIMILARITY",
    ) -> None:
        self._model = model
        self._task_type = task_type

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import os
        import google.genai as genai  # lazy import — optional dependency

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        result = client.models.embed_content(
            model=self._model,
            contents=texts,
            config={"task_type": self._task_type},
        )
        return [e.values for e in result.embeddings]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"
