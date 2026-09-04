"""A persistent embedding cache keyed by provider, model, and text.

Four strategies embed the same chunks of the same corpus. Without a cache
each build and each optimize trial pays the provider again for text it has
already embedded. The cache sits in front of any provider function, keys on
the provider, the model, and a hash of the text, and stores vectors in one
SQLite file, so a second build or a second strategy reuses every vector.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path

from chromadb import Documents, EmbeddingFunction, Embeddings

from kb_arena.settings import settings

log = logging.getLogger(__name__)
_SCHEMA = "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"


def cache_key(provider: str, model: str, text: str) -> str:
    """One key per provider, model, and exact text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{provider}:{model}:{digest}"


def default_cache_path() -> Path:
    if settings.embedding_cache_path:
        return Path(settings.embedding_cache_path)
    return Path(settings.chroma_path) / "embedding_cache.sqlite"


class CachedEmbedding(EmbeddingFunction[Documents]):
    """Wrap a provider function. Hits come from SQLite, misses go to the provider once."""

    def __init__(
        self,
        inner: EmbeddingFunction[Documents],
        *,
        provider: str,
        model: str,
        path: Path | str | None = None,
    ) -> None:
        self._inner = inner
        self._provider = provider
        self._model = model
        self._path = Path(path) if path else default_cache_path()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=30, check_same_thread=False)

    @property
    def identity(self) -> dict[str, str]:
        return {"provider": self._provider, "model": self._model, "path": str(self._path)}

    def _read(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with self._connect() as conn:
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                marks = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT key, vector FROM vectors WHERE key IN ({marks})", chunk
                )
                for key, raw in rows:
                    try:
                        vector = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(vector, list) and vector:
                        found[key] = vector
        return found

    def _write(self, pairs: list[tuple[str, list[float]]]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, vector) VALUES (?, ?)",
                [(key, json.dumps(vector)) for key, vector in pairs],
            )

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        texts = list(input)
        keys = [cache_key(self._provider, self._model, text) for text in texts]
        with self._lock:
            found = self._read(list(dict.fromkeys(keys)))
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        seen: set[str] = set()
        for key, text in zip(keys, texts, strict=True):
            if key in found or key in seen:
                continue
            seen.add(key)
            missing_keys.append(key)
            missing_texts.append(text)
        self.hits += len(texts) - len(missing_texts)
        self.misses += len(missing_texts)
        if missing_texts:
            fresh = self._inner(missing_texts)
            fresh = [list(map(float, vector)) for vector in fresh]
            if len(fresh) != len(missing_texts):
                raise RuntimeError(
                    f"embedding provider returned {len(fresh)} vectors "
                    f"for {len(missing_texts)} texts"
                )
            with self._lock:
                self._write(list(zip(missing_keys, fresh, strict=True)))
            found.update(zip(missing_keys, fresh, strict=True))
        return [found[key] for key in keys]
