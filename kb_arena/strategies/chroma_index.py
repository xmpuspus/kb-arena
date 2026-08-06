"""Versioning and corpus filters shared by Chroma-backed strategies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

INDEX_FORMAT_VERSION = 2


def index_metadata() -> dict[str, int]:
    """Metadata required on every current-format Chroma record."""
    return {"index_version": INDEX_FORMAT_VERSION}


def index_where(corpus: str = "all", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a Chroma filter that excludes legacy and cross-corpus records."""
    filters: list[dict[str, Any]] = [{"index_version": INDEX_FORMAT_VERSION}]
    if corpus != "all":
        filters.append({"corpus": corpus})
    if extra:
        filters.append(extra)
    return filters[0] if len(filters) == 1 else {"$and": filters}


def finalize_collection_build(
    collection: Any,
    corpora: Iterable[str],
    keep_ids: Iterable[str],
) -> None:
    """Remove legacy and stale records after a successful replacement upsert."""
    corpus_set = set(corpora)
    keep_id_set = set(keep_ids)
    existing = collection.get(include=["metadatas"])
    ids = list(existing.get("ids") or [])
    metadatas = list(existing.get("metadatas") or [])
    remove_ids = []
    for index, record_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        if metadata.get("index_version") != INDEX_FORMAT_VERSION or (
            metadata.get("corpus") in corpus_set and record_id not in keep_id_set
        ):
            remove_ids.append(record_id)
    if remove_ids:
        collection.delete(ids=remove_ids)
