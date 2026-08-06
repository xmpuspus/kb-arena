"""Tests for Chroma index versioning and rebuild replacement."""

from unittest.mock import MagicMock

from kb_arena.strategies.chroma_index import (
    INDEX_FORMAT_VERSION,
    finalize_collection_build,
    index_where,
)


def test_index_where_requires_current_version_and_selected_corpus():
    assert index_where("nist") == {
        "$and": [{"index_version": INDEX_FORMAT_VERSION}, {"corpus": "nist"}]
    }
    assert index_where() == {"index_version": INDEX_FORMAT_VERSION}


def test_finalize_removes_legacy_and_stale_records_but_preserves_new_and_other_corpora():
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["legacy", "alpha::old", "alpha::new", "beta::current"],
        "metadatas": [
            {"source_id": "legacy"},
            {"corpus": "alpha", "index_version": INDEX_FORMAT_VERSION},
            {"corpus": "alpha", "index_version": INDEX_FORMAT_VERSION},
            {"corpus": "beta", "index_version": INDEX_FORMAT_VERSION},
        ],
    }

    finalize_collection_build(collection, ["alpha"], ["alpha::new"])

    collection.delete.assert_called_once_with(ids=["legacy", "alpha::old"])
