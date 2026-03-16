"""Shared OpenAI embedding function compatible with openai SDK v1+.

ChromaDB's built-in OpenAIEmbeddingFunction uses the removed openai.Embedding
API from v0.x. This module provides a drop-in replacement using the v1 API.
"""

from __future__ import annotations

from chromadb import Documents, EmbeddingFunction, Embeddings

from kb_arena.settings import settings


class OpenAIEmbedding(EmbeddingFunction[Documents]):
    """Embedding function using openai SDK v1+ (client.embeddings.create)."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        import openai

        self._client = openai.OpenAI(api_key=api_key or settings.openai_api_key)
        self._model = model or settings.embedding_model

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        resp = self._client.embeddings.create(model=self._model, input=list(input))
        return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]
