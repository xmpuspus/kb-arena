"""Tests for atomic Chroma generation activation and cleanup."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kb_arena.strategies.chroma_index import (
    INDEX_FORMAT_VERSION,
    IndexStateError,
    activate_generations,
    index_build_lock,
    index_where,
    prune_collection,
    upsert_staged_records,
)


def test_index_where_requires_active_generation_and_selected_corpus():
    activate_generations({"naive_vector": {"nist": "generation-a"}})

    assert index_where("naive_vector", "nist") == {
        "$and": [
            {"index_version": INDEX_FORMAT_VERSION},
            {"corpus": "nist"},
            {"generation": "generation-a"},
        ]
    }


def test_index_where_all_selects_each_active_corpus_generation():
    activate_generations({"naive_vector": {"beta": "generation-b", "alpha": "generation-a"}})

    assert index_where("naive_vector") == {
        "$or": [
            {
                "$and": [
                    {"index_version": INDEX_FORMAT_VERSION},
                    {"corpus": "alpha"},
                    {"generation": "generation-a"},
                ]
            },
            {
                "$and": [
                    {"index_version": INDEX_FORMAT_VERSION},
                    {"corpus": "beta"},
                    {"generation": "generation-b"},
                ]
            },
        ]
    }


def test_index_where_fails_closed_without_an_active_generation():
    where = index_where("naive_vector", "nist")

    assert where["$and"][1] == {"corpus": "nist"}
    assert where["$and"][2]["generation"].startswith("__kb_arena_inactive")


def test_index_where_surfaces_corrupt_activation_manifest():
    from kb_arena.settings import settings

    state_path = Path(settings.chroma_path) / ".kb_arena-index-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("not json")

    with pytest.raises(IndexStateError, match="Cannot read"):
        index_where("naive_vector", "alpha")


def test_failed_second_batch_is_removed_without_changing_active_generation():
    activate_generations({"naive_vector": {"alpha": "old-generation"}})
    collection = MagicMock()
    collection.upsert.side_effect = [None, RuntimeError("second batch failed")]
    stable_ids = [f"alpha::chunk-{index}" for index in range(501)]

    with pytest.raises(RuntimeError, match="second batch failed"):
        upsert_staged_records(
            collection,
            "new-generation",
            stable_ids,
            [f"text-{index}" for index in range(501)],
            [{"corpus": "alpha"} for _ in range(501)],
        )

    assert collection.upsert.call_count == 2
    collection.delete.assert_called_once_with(
        ids=[f"new-generation::{stable_id}" for stable_id in stable_ids]
    )
    assert index_where("naive_vector", "alpha")["$and"][2] == {"generation": "old-generation"}


def test_prune_removes_only_inactive_records_for_rebuilt_corpora():
    activate_generations({"naive_vector": {"alpha": "generation-new"}})
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["legacy", "alpha::old", "alpha::new", "beta::current"],
        "metadatas": [
            {"corpus": "alpha"},
            {
                "corpus": "alpha",
                "index_version": INDEX_FORMAT_VERSION,
                "generation": "generation-old",
            },
            {
                "corpus": "alpha",
                "index_version": INDEX_FORMAT_VERSION,
                "generation": "generation-new",
            },
            {
                "corpus": "beta",
                "index_version": INDEX_FORMAT_VERSION,
                "generation": "generation-beta",
            },
        ],
    }

    prune_collection(collection, "naive_vector", ["alpha"])

    collection.delete.assert_called_once_with(ids=["legacy", "alpha::old"])


@pytest.mark.asyncio
async def test_build_lock_serializes_concurrent_publishers():
    active = 0
    maximum_active = 0

    async def publisher():
        nonlocal active, maximum_active
        async with index_build_lock(poll_interval=0.001):
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(publisher(), publisher())

    assert maximum_active == 1
