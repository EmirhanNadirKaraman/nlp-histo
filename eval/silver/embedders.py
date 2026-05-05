"""Embedding providers for the silver eval harness."""
from __future__ import annotations

OPENAI_MODEL = "text-embedding-3-small"
GEMINI_MODEL = "gemini-embedding-001"


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = OPENAI_MODEL) -> None:
        self._api_key = api_key
        self.model = model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=self._api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self.model!r})"


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = GEMINI_MODEL,
        task_type: str = "SEMANTIC_SIMILARITY",
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._task_type = task_type

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import google.genai as genai
        client = genai.Client(api_key=self._api_key)
        result = client.models.embed_content(
            model=self.model,
            contents=texts,
            config={"task_type": self._task_type},
        )
        return [e.values for e in result.embeddings]

    def __repr__(self) -> str:
        return f"GeminiEmbedder(model={self.model!r})"
