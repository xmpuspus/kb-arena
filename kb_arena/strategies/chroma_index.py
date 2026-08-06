"""Atomic generation management shared by Chroma-backed strategies."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from kb_arena.settings import settings

logger = logging.getLogger(__name__)

INDEX_FORMAT_VERSION = 3
STATE_FORMAT_VERSION = 1
_STATE_FILENAME = ".kb_arena-index-state.json"
_LOCK_FILENAME = ".kb_arena-build.lock"
_ACTIVATION_LOCK_FILENAME = ".kb_arena-activation.lock"
_INACTIVE_GENERATION = "__kb_arena_inactive__"


class IndexStateError(RuntimeError):
    """The index activation manifest is invalid or cannot be updated safely."""


def new_generation() -> str:
    """Return an opaque identifier for one complete staged build."""
    return uuid4().hex


def backend_id(generation: str, stable_id: str) -> str:
    """Namespace a stable retrieval ID by its staged generation."""
    return f"{generation}::{stable_id}"


def index_metadata(generation: str) -> dict[str, int | str]:
    """Metadata required on every current-format Chroma record."""
    return {"index_version": INDEX_FORMAT_VERSION, "generation": generation}


def _index_directory() -> Path:
    return Path(settings.chroma_path).expanduser()


def _state_path() -> Path:
    return _index_directory() / _STATE_FILENAME


def _empty_state() -> dict[str, Any]:
    return {"format_version": STATE_FORMAT_VERSION, "collections": {}}


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return _empty_state()

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexStateError(f"Cannot read Chroma activation manifest: {path}") from exc

    if not isinstance(state, dict) or state.get("format_version") != STATE_FORMAT_VERSION:
        raise IndexStateError(f"Unsupported Chroma activation manifest: {path}")
    collections = state.get("collections")
    if not isinstance(collections, dict):
        raise IndexStateError(f"Invalid Chroma activation manifest: {path}")
    for collection_name, corpus_map in collections.items():
        if not isinstance(collection_name, str) or not isinstance(corpus_map, dict):
            raise IndexStateError(f"Invalid Chroma activation manifest: {path}")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in corpus_map.items()):
            raise IndexStateError(f"Invalid Chroma activation manifest: {path}")
    return state


def _write_state(state: dict[str, Any]) -> None:
    directory = _index_directory()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / _STATE_FILENAME
    fd, temporary_name = tempfile.mkstemp(prefix=f"{_STATE_FILENAME}.", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _active_generations(collection_name: str) -> dict[str, str]:
    state = _load_state()
    corpus_map = state["collections"].get(collection_name, {})
    return dict(corpus_map)


def _generation_clause(corpus: str, generation: str) -> dict[str, Any]:
    return {
        "$and": [
            {"index_version": INDEX_FORMAT_VERSION},
            {"corpus": corpus},
            {"generation": generation},
        ]
    }


def index_where(
    collection_name: str,
    corpus: str = "all",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select only atomically activated records for a collection and corpus."""
    active = _active_generations(collection_name)
    if corpus == "all":
        clauses = [_generation_clause(name, active[name]) for name in sorted(active)]
        if not clauses:
            base: dict[str, Any] = {
                "$and": [
                    {"index_version": INDEX_FORMAT_VERSION},
                    {"generation": _INACTIVE_GENERATION},
                ]
            }
        elif len(clauses) == 1:
            base = clauses[0]
        else:
            base = {"$or": clauses}
    else:
        base = _generation_clause(corpus, active.get(corpus, _INACTIVE_GENERATION))

    if extra:
        return {"$and": [base, extra]}
    return base


def staged_where(corpus: str, generation: str) -> dict[str, Any]:
    """Select a staged generation while a multi-level build is in progress."""
    return _generation_clause(corpus, generation)


def _try_lock(handle: Any) -> bool:
    if os.name == "nt":  # pragma: no cover - exercised on Windows release validation
        import msvcrt

        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _unlock(handle: Any) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows release validation
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_blocking(handle: Any, *, shared: bool) -> None:
    if os.name == "nt":  # pragma: no cover - exercised on Windows release validation
        import msvcrt

        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    fcntl.flock(handle.fileno(), mode)


@contextmanager
def _activation_lock(*, shared: bool) -> Iterator[None]:
    directory = _index_directory()
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / _ACTIVATION_LOCK_FILENAME).open("a+b")
    acquired = False
    try:
        _lock_blocking(handle, shared=shared)
        acquired = True
        yield
    finally:
        if acquired:
            _unlock(handle)
        handle.close()


@contextmanager
def index_read_lock() -> Iterator[None]:
    """Keep the selected generations alive until a complete query finishes."""
    with _activation_lock(shared=True):
        yield


@contextmanager
def index_activation_lock() -> Iterator[None]:
    """Exclude readers while activating generations and pruning their predecessors."""
    with _activation_lock(shared=False):
        yield


@asynccontextmanager
async def index_build_lock(poll_interval: float = 0.05) -> AsyncIterator[None]:
    """Serialize all Chroma publishers without blocking the event loop."""
    directory = _index_directory()
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / _LOCK_FILENAME).open("a+b")
    acquired = False
    try:
        while not acquired:
            acquired = _try_lock(handle)
            if not acquired:
                await asyncio.sleep(poll_interval)
        yield
    finally:
        if acquired:
            _unlock(handle)
        handle.close()


def upsert_staged_records(
    collection: Any,
    generation: str,
    stable_ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[Mapping[str, Any]],
    batch_size: int = 500,
) -> list[str]:
    """Write an invisible generation and remove it if any batch fails."""
    if not (len(stable_ids) == len(documents) == len(metadatas)):
        raise ValueError("Chroma record IDs, documents, and metadata must have equal lengths")

    staged_ids = [backend_id(generation, stable_id) for stable_id in stable_ids]
    staged_metadatas = [{**metadata, **index_metadata(generation)} for metadata in metadatas]
    try:
        for start in range(0, len(staged_ids), batch_size):
            collection.upsert(
                ids=staged_ids[start : start + batch_size],
                documents=list(documents[start : start + batch_size]),
                metadatas=staged_metadatas[start : start + batch_size],
            )
    except Exception:
        discard_staged_ids(collection, staged_ids)
        raise
    return staged_ids


def discard_staged_ids(collection: Any, staged_ids: Iterable[str]) -> None:
    """Best-effort cleanup for a generation that was never activated."""
    ids = list(staged_ids)
    if not ids:
        return
    try:
        collection.delete(ids=ids)
    except Exception as exc:  # pragma: no cover - cleanup failure is backend-specific
        logger.warning("Could not remove an inactive Chroma generation: %s", exc)


def activate_generations(activations: Mapping[str, Mapping[str, str]]) -> None:
    """Atomically publish one or more collection/corpus generations."""
    state = _load_state()
    collections = state["collections"]
    for collection_name, corpus_map in activations.items():
        current = collections.setdefault(collection_name, {})
        current.update(corpus_map)
    _write_state(state)


def prune_collection(collection: Any, collection_name: str, corpora: Iterable[str]) -> None:
    """Remove inactive records for rebuilt corpora after successful activation."""
    corpus_set = set(corpora)
    active = _load_state()["collections"].get(collection_name, {})
    existing = collection.get(include=["metadatas"])
    ids = list(existing.get("ids") or [])
    metadatas = list(existing.get("metadatas") or [])
    remove_ids: list[str] = []
    for index, record_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        corpus = metadata.get("corpus")
        if corpus not in corpus_set:
            continue
        if metadata.get("index_version") != INDEX_FORMAT_VERSION or metadata.get(
            "generation"
        ) != active.get(corpus):
            remove_ids.append(record_id)
    if remove_ids:
        collection.delete(ids=remove_ids)


def publish_collection_build(
    collection: Any,
    collection_name: str,
    corpora: Iterable[str],
    generation: str,
    stable_ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Stage and atomically activate a single collection build under the build lock."""
    corpus_list = list(dict.fromkeys(corpora))
    staged_ids = upsert_staged_records(collection, generation, stable_ids, documents, metadatas)
    try:
        with index_activation_lock():
            activate_generations({collection_name: {corpus: generation for corpus in corpus_list}})
            try:
                prune_collection(collection, collection_name, corpus_list)
            except Exception as exc:  # stale records remain hidden by the manifest
                logger.warning("Could not prune inactive %s records: %s", collection_name, exc)
    except Exception:
        discard_staged_ids(collection, staged_ids)
        raise
    return staged_ids
