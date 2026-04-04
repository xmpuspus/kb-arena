"""Shared OpenAI embedding function compatible with openai SDK v1+.

ChromaDB's built-in OpenAIEmbeddingFunction uses the removed openai.Embedding
API from v0.x. This module provides a drop-in replacement using the v1 API.
"""

from __future__ import annotations

from chromadb import Documents, EmbeddingFunction, Embeddings

from kb_arena.settings import settings

_MAX_RETRIES = 3
_TIMEOUT_S = 30


class OpenAIEmbedding(EmbeddingFunction[Documents]):
    """Embedding function using openai SDK v1+ (client.embeddings.create).

    Includes retry with exponential backoff and per-request timeout.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        import openai

        self._client = openai.OpenAI(
            api_key=api_key or settings.openai_api_key,
            timeout=_TIMEOUT_S,
        )
        self._model = model or settings.embedding_model

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        import logging
        import time

        log = logging.getLogger(__name__)
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.embeddings.create(model=self._model, input=list(input))
                return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = (2**attempt) + 0.1  # exponential backoff
                    log.warning(
                        "Embedding API attempt %d failed (%s), retrying in %.1fs",
                        attempt + 1,
                        exc,
                        wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(f"Embedding API failed after {_MAX_RETRIES} attempts: {last_exc}")
