"""Embedding provider abstraction.

Vector strategies (`naive_vector`, `contextual_vector`, `qna_pairs`, `raptor`)
should not hard-code OpenAI. The provider is selected by
`KB_ARENA_EMBEDDING_PROVIDER`:

- `openai`  — text-embedding-3-large (default, current behaviour)
- `voyage`  — voyage-3-large (current MTEB retrieval leader, +10.58% over OpenAI)
- `cohere`  — cohere embed-v4
- `bge`     — BAAI/bge-large-en-v1.5 via sentence-transformers (local, no key)
- `ollama`  — Ollama embedding endpoint (privacy-friendly, no key)
- `gemini`  — text-embedding-004

Each provider exposes the same `__call__(Documents) -> Embeddings` ChromaDB
interface so existing strategies don't need conditional logic.

Concrete provider classes that need network/SDK lazy-import their dependency
inside `__init__` so missing optional packages only fail when that provider
is actually selected.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from chromadb import Documents, EmbeddingFunction, Embeddings

from kb_arena.settings import settings
from kb_arena.telemetry import traced_span

_MAX_RETRIES = 3
_TIMEOUT_S = 30
log = logging.getLogger(__name__)


def _retry(fn, *args, **kwargs):
    """Retry helper with exponential backoff + jitter for embedding calls.

    Reads the active provider straight from settings for the span label,
    since every caller here embeds through the one provider a run selected.
    """
    import random

    with traced_span("kb_arena.embedding", provider=settings.embedding_provider):
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — providers raise heterogeneous errors
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = (2**attempt) + random.uniform(0.0, 0.5)
                    log.warning(
                        "Embedding attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(f"Embedding failed after {_MAX_RETRIES} attempts: {last_exc}")


class OpenAIEmbedding(EmbeddingFunction[Documents]):
    """OpenAI embeddings via openai SDK v1+ (client.embeddings.create)."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        import openai

        self._client = openai.OpenAI(
            api_key=api_key or settings.openai_api_key,
            timeout=_TIMEOUT_S,
        )
        self._model = model or settings.embedding_model

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        def _call():
            resp = self._client.embeddings.create(model=self._model, input=list(input))
            return [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]

        return _retry(_call)


class VoyageEmbedding(EmbeddingFunction[Documents]):
    """Voyage AI embeddings — voyage-3-large is current MTEB retrieval leader."""

    def __init__(self, api_key: str | None = None, model: str = "voyage-3-large") -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover — only when chosen
            raise ImportError(
                "voyageai is required for KB_ARENA_EMBEDDING_PROVIDER=voyage. "
                "Install with: pip install voyageai"
            ) from exc
        self._client = voyageai.Client(api_key=api_key or settings.voyage_api_key or None)
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        def _call():
            resp = self._client.embed(list(input), model=self._model, input_type="document")
            return resp.embeddings

        return _retry(_call)


class CohereEmbedding(EmbeddingFunction[Documents]):
    """Cohere embed-v4."""

    def __init__(self, api_key: str | None = None, model: str = "embed-v4.0") -> None:
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "cohere is required for KB_ARENA_EMBEDDING_PROVIDER=cohere. "
                "Install with: pip install cohere"
            ) from exc
        self._client = cohere.Client(api_key=api_key or settings.cohere_api_key or None)
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        def _call():
            resp = self._client.embed(
                texts=list(input), model=self._model, input_type="search_document"
            )
            return list(resp.embeddings)

        return _retry(_call)


class BGEEmbedding(EmbeddingFunction[Documents]):
    """Local BGE-large via sentence-transformers — no API key, fully on-prem."""

    def __init__(self, model: str = "BAAI/bge-large-en-v1.5") -> None:
        self._model = model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for KB_ARENA_EMBEDDING_PROVIDER=bge. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._st = SentenceTransformer(model)

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        # sentence-transformers .encode is local and synchronous; no retry needed.
        with traced_span("kb_arena.embedding", provider="bge"):
            vecs = self._st.encode(list(input), normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vecs]


class OllamaEmbedding(EmbeddingFunction[Documents]):
    """Ollama embeddings — uses /api/embeddings with the configured model."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        import httpx

        self._base = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_embedding_model
        self._client = httpx.Client(timeout=_TIMEOUT_S)

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        def _call():
            out: list[list[float]] = []
            for text in input:
                resp = self._client.post(
                    f"{self._base}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                resp.raise_for_status()
                out.append(resp.json()["embedding"])
            return out

        return _retry(_call)


class GeminiEmbedding(EmbeddingFunction[Documents]):
    """Google Gemini text-embedding-004."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-004") -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "google-genai is required for KB_ARENA_EMBEDDING_PROVIDER=gemini. "
                "Install with: pip install google-genai"
            ) from exc
        self._client = genai.Client(api_key=api_key or settings.gemini_api_key or None)
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        def _call():
            results = self._client.models.embed_content(model=self._model, contents=list(input))
            return [list(r.values) for r in results.embeddings]

        return _retry(_call)


_PROVIDERS: dict[str, type[EmbeddingFunction[Documents]]] = {
    "openai": OpenAIEmbedding,
    "voyage": VoyageEmbedding,
    "cohere": CohereEmbedding,
    "bge": BGEEmbedding,
    "ollama": OllamaEmbedding,
    "gemini": GeminiEmbedding,
}


def get_embedding_function(**kwargs: Any) -> EmbeddingFunction[Documents]:
    """Return an embedding function selected by `KB_ARENA_EMBEDDING_PROVIDER`.

    Defaults to OpenAI for backward compatibility. Pass per-provider kwargs
    through (e.g. `model="bge-base-en-v1.5"`).
    """
    provider = (settings.embedding_provider or "openai").lower()
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown KB_ARENA_EMBEDDING_PROVIDER={provider!r}. " f"Valid: {sorted(_PROVIDERS)}"
        )
    inner = cls(**kwargs)
    if not settings.embedding_cache_enabled:
        return inner
    from kb_arena.strategies.embedding_cache import CachedEmbedding

    model = str(getattr(inner, "_model", "") or kwargs.get("model") or f"{provider}-default")
    # A self-hosted endpoint is part of the identity: another server with the
    # same model name can hold other weights.
    endpoint = ""
    if provider == "ollama":
        endpoint = str(kwargs.get("base_url") or settings.ollama_base_url)
    return CachedEmbedding(inner, provider=provider, model=model, endpoint=endpoint)
