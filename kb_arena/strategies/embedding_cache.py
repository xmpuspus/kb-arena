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
import math
import sqlite3
import threading
import time
from pathlib import Path

from chromadb import Documents, EmbeddingFunction, Embeddings

from kb_arena.settings import settings

log = logging.getLogger(__name__)
_SCHEMA = "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
# A claim marks a key one process is computing right now. INSERT OR IGNORE
# on a primary key is atomic across processes, so two benchmark processes
# never both pay the provider for one text. A claim older than this many
# seconds belongs to a process that died, and the next caller takes it over.
_CLAIMS = "CREATE TABLE IF NOT EXISTS claims (key TEXT PRIMARY KEY, claimed_at REAL NOT NULL)"
# The vector length each identity produced the first time. A later row of
# another length is corrupt, whether it sits alone in a lookup or not.
_DIMS = "CREATE TABLE IF NOT EXISTS dims (identity TEXT PRIMARY KEY, dim INTEGER NOT NULL)"
_CLAIM_TTL_S = 120.0
_CLAIM_POLL_S = 0.05
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


def cache_key(provider: str, model: str, text: str, endpoint: str = "", salt: str = "") -> str:
    """One key per provider, model, endpoint, salt, and exact text.

    The endpoint matters for a self-hosted provider: two Ollama servers with
    one model name can hold different weights, so their vectors never share.
    The salt is KB_ARENA_EMBEDDING_CACHE_SALT, for a model that changed
    under the same tag: set it to the new revision and the old vectors are
    never read again. The parts go through one JSON digest, so a colon inside
    a model tag can never make two identities collide.
    """
    scope = json.dumps([provider, model, endpoint.rstrip("/"), salt, text], separators=(",", ":"))
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


def _checked_vectors(fresh) -> list[list[float]]:
    """Provider output as plain float lists, rejected when any element is not a finite number."""
    out: list[list[float]] = []
    for vector in fresh:
        items = list(vector)
        for x in items:
            if isinstance(x, bool) or isinstance(x, str) or not math.isfinite(float(x)):
                raise RuntimeError(
                    "embedding provider returned a vector with a non-numeric element"
                )
        out.append([float(x) for x in items])
    if out and len({len(v) for v in out}) != 1:
        raise RuntimeError("embedding provider returned vectors of different lengths")
    return out


def _valid_vector(vector) -> bool:
    """A non-empty list of finite numbers, and nothing else, counts as a vector."""
    if not isinstance(vector, list) or not vector:
        return False
    for x in vector:
        if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x):
            return False
    return True


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
            conn.execute(_CLAIMS)
            conn.execute(_DIMS)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30, check_same_thread=False)
        # WAL lets a second process read while one writes, so two benchmark
        # processes on one cache never wait thirty seconds for each other.
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _identity_key(self) -> str:
        return json.dumps(
            [self._provider, self._model, self._endpoint, settings.embedding_cache_salt],
            separators=(",", ":"),
        )

    def _known_dim(self, conn: sqlite3.Connection) -> int | None:
        row = conn.execute(
            "SELECT dim FROM dims WHERE identity = ?", (self._identity_key(),)
        ).fetchone()
        return int(row[0]) if row else None

    def _record_dim(self, conn: sqlite3.Connection, dim: int) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO dims (identity, dim) VALUES (?, ?)", (self._identity_key(), dim)
        )

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
            known_dim = self._known_dim(conn)
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
                    if _valid_vector(vector) and (known_dim is None or len(vector) == known_dim):
                        found[key] = vector
        # Every vector of one identity has one length. Before the first write
        # recorded it, the majority length in this lookup stands in for it.
        if known_dim is None:
            lengths = {len(v) for v in found.values()}
            if len(lengths) > 1:
                common = max(lengths, key=lambda n: sum(1 for v in found.values() if len(v) == n))
                found = {k: v for k, v in found.items() if len(v) == common}
        return found

    def _write(self, pairs: list[tuple[str, list[float]]]) -> None:
        with self._connect() as conn:
            if pairs:
                self._record_dim(conn, len(pairs[0][1]))
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, vector) VALUES (?, ?)",
                [(key, json.dumps(vector)) for key, vector in pairs],
            )

    def _inflight(self) -> dict[str, threading.Event]:
        return _INFLIGHT.setdefault(str(self._path.resolve()), {})

    def _claim(self, keys: list[str]) -> set[str]:
        """The keys this process now owns. A stale claim from a dead process is taken over."""
        now = time.time()
        with self._connect() as conn:
            conn.execute("DELETE FROM claims WHERE claimed_at < ?", (now - _CLAIM_TTL_S,))
            mine: set[str] = set()
            for key in keys:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO claims (key, claimed_at) VALUES (?, ?)", (key, now)
                )
                if cur.rowcount == 1:
                    mine.add(key)
        return mine

    def _release(self, keys: list[str]) -> None:
        with self._connect() as conn:
            conn.executemany("DELETE FROM claims WHERE key = ?", [(k,) for k in keys])

    def _wait_for_foreign(self, keys: list[str]) -> None:
        """Wait for another process to store the keys it claimed, or for its claim to expire."""
        deadline = time.time() + _CLAIM_TTL_S
        pending = set(keys)
        while pending and time.time() < deadline:
            time.sleep(_CLAIM_POLL_S)
            with self._lock:
                stored = self._read(list(pending))
                with self._connect() as conn:
                    marks = ",".join("?" for _ in pending)
                    live = {
                        row[0]
                        for row in conn.execute(
                            f"SELECT key FROM claims WHERE key IN ({marks})", list(pending)
                        )
                    }
            pending -= set(stored)
            pending -= {k for k in pending if k not in live}

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        texts = list(input)
        salt = settings.embedding_cache_salt
        keys = [
            cache_key(self._provider, self._model, text, self._endpoint, salt) for text in texts
        ]
        unique = list(dict.fromkeys(keys))
        text_of = dict(zip(keys, texts, strict=True))
        # Under the lock: read what is stored, claim the misses nobody is
        # computing, and note the misses another caller is computing now.
        # The claim lives in the database, so it holds across processes.
        with self._lock:
            found = self._read(unique)
            inflight = self._inflight()
            mine: list[str] = []
            waits: list[threading.Event] = []
            foreign: list[str] = []
            missing = [key for key in unique if key not in found]
            claimed = self._claim(missing) if missing else set()
            for key in missing:
                pending = inflight.get(key)
                if pending is not None:
                    waits.append(pending)
                elif key in claimed:
                    inflight[key] = threading.Event()
                    mine.append(key)
                else:
                    foreign.append(key)
        self.hits += sum(1 for key in keys if key in found)
        self.misses += sum(1 for key in keys if key not in found)
        try:
            if mine:
                fresh = self._inner([text_of[key] for key in mine])
                fresh = _checked_vectors(fresh)
                if len(fresh) != len(mine):
                    raise RuntimeError(
                        f"embedding provider returned {len(fresh)} vectors for {len(mine)} texts"
                    )
                with self._lock:
                    self._write(list(zip(mine, fresh, strict=True)))
                found.update(zip(mine, fresh, strict=True))
        finally:
            with self._lock:
                if mine:
                    self._release(mine)
                for key in mine:
                    event = self._inflight().pop(key, None)
                    if event is not None:
                        event.set()
        for event in waits:
            event.wait()
        if foreign:
            self._wait_for_foreign(foreign)
        still_missing = [key for key in unique if key not in found]
        if still_missing:
            with self._lock:
                found.update(self._read(still_missing))
            lost = [key for key in still_missing if key not in found]
            if lost:
                # The other caller failed. Compute the leftovers ourselves.
                fresh = self._inner([text_of[key] for key in lost])
                fresh = _checked_vectors(fresh)
                if len(fresh) != len(lost):
                    raise RuntimeError(
                        f"embedding provider returned {len(fresh)} vectors for {len(lost)} texts"
                    )
                with self._lock:
                    self._write(list(zip(lost, fresh, strict=True)))
                found.update(zip(lost, fresh, strict=True))
        return [found[key] for key in keys]
