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
# One lock per cache file for the whole process. Four strategies build at
# once in one benchmark, each with its own wrapper, and they share the file.
_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()
# Keys a provider call is computing right now, per cache file. A second
# caller that misses on one of them waits for that call instead of paying
# for the same text twice.
_INFLIGHT: dict[str, dict[str, threading.Event]] = {}


def _lock_for(path: Path) -> threading.Lock:
    key = str(Path(path).resolve())
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(key, threading.Lock())


def cache_key(provider: str, model: str, text: str, endpoint: str = "") -> str:
    """One key per provider, model, endpoint, and exact text.

    The endpoint matters for a self-hosted provider: two Ollama servers with
    one model name can hold different weights, so their vectors never share.
    The four parts go through one JSON digest, so a colon inside a model tag
    can never make two identities collide.
    """
    scope = json.dumps([provider, model, endpoint.rstrip("/"), text], separators=(",", ":"))
    return hashlib.sha256(scope.encode("utf-8")).hexdigest()


def default_cache_path() -> Path:
    """The cache file. It never lives under chroma_path.

    An optimize trial that rebuilds an index redirects chroma_path to a temp
    directory and deletes it afterwards. A cache there would be built in full
    and thrown away on every trial, which costs more than no cache.
    """
    if settings.embedding_cache_path:
        return Path(settings.embedding_cache_path)
    return Path("./embedding_cache.sqlite")


class CachedEmbedding(EmbeddingFunction[Documents]):
    """Wrap a provider function. Hits come from SQLite, misses go to the provider once."""

    def __init__(
        self,
        inner: EmbeddingFunction[Documents],
        *,
        provider: str,
        model: str,
        path: Path | str | None = None,
        endpoint: str = "",
    ) -> None:
        self._inner = inner
        self._provider = provider
        self._model = model
        self._endpoint = endpoint.rstrip("/")
        self._path = Path(path) if path else default_cache_path()
        self._lock = _lock_for(self._path)
        self.hits = 0
        self.misses = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30, check_same_thread=False)
        # WAL lets a second process read while one writes, so two benchmark
        # processes on one cache never wait thirty seconds for each other.
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @property
    def identity(self) -> dict[str, str]:
        return {
            "provider": self._provider,
            "model": self._model,
            "endpoint": self._endpoint,
            "path": str(self._path),
        }

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

    def _inflight(self) -> dict[str, threading.Event]:
        return _INFLIGHT.setdefault(str(self._path.resolve()), {})

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        texts = list(input)
        keys = [cache_key(self._provider, self._model, text, self._endpoint) for text in texts]
        unique = list(dict.fromkeys(keys))
        text_of = dict(zip(keys, texts, strict=True))
        # Under the lock: read what is stored, claim the misses nobody is
        # computing, and note the misses another caller is computing now.
        with self._lock:
            found = self._read(unique)
            inflight = self._inflight()
            mine: list[str] = []
            waits: list[threading.Event] = []
            for key in unique:
                if key in found:
                    continue
                pending = inflight.get(key)
                if pending is not None:
                    waits.append(pending)
                else:
                    inflight[key] = threading.Event()
                    mine.append(key)
        self.hits += sum(1 for key in keys if key in found)
        self.misses += sum(1 for key in keys if key not in found)
        try:
            if mine:
                fresh = self._inner([text_of[key] for key in mine])
                fresh = [list(map(float, vector)) for vector in fresh]
                if len(fresh) != len(mine):
                    raise RuntimeError(
                        f"embedding provider returned {len(fresh)} vectors for {len(mine)} texts"
                    )
                with self._lock:
                    self._write(list(zip(mine, fresh, strict=True)))
                found.update(zip(mine, fresh, strict=True))
        finally:
            with self._lock:
                for key in mine:
                    self._inflight().pop(key, None).set() if key in self._inflight() else None
        for event in waits:
            event.wait()
        still_missing = [key for key in unique if key not in found]
        if still_missing:
            with self._lock:
                found.update(self._read(still_missing))
            lost = [key for key in still_missing if key not in found]
            if lost:
                # The other caller failed. Compute the leftovers ourselves.
                fresh = self._inner([text_of[key] for key in lost])
                fresh = [list(map(float, vector)) for vector in fresh]
                if len(fresh) != len(lost):
                    raise RuntimeError(
                        f"embedding provider returned {len(fresh)} vectors for {len(lost)} texts"
                    )
                with self._lock:
                    self._write(list(zip(lost, fresh, strict=True)))
                found.update(zip(lost, fresh, strict=True))
        return [found[key] for key in keys]
