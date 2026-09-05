"""Tests for Strategy: Metadata Filtered — access-aware dense retrieval.

The filter must apply inside retrieval (pushed into the Chroma query or
applied to an over-fetched pool before the top_k cut), and it must fail
closed: an unresolvable filter raises instead of returning unfiltered
results.
"""

from __future__ import annotations

import pytest

from kb_arena.strategies.base import AnswerResult
from kb_arena.strategies.metadata_filtered import AccessFilter, MetadataFilteredStrategy


def _set_candidates(mock_chroma_client, rows: list[dict]) -> None:
    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.query.return_value = {
        "ids": [[r["id"] for r in rows]],
        "documents": [[r["content"] for r in rows]],
        "metadatas": [[r["metadata"] for r in rows]],
        "distances": [[r["distance"] for r in rows]],
    }


@pytest.mark.asyncio
async def test_happy_path_pushes_classification_into_the_chroma_query(
    mock_chroma_client, mock_llm_client
):
    _set_candidates(
        mock_chroma_client,
        [
            {
                "id": "c1",
                "content": "public content",
                "distance": 0.1,
                "metadata": {
                    "source_id": "doc-a",
                    "classification": "public",
                    "owner": "team-a",
                    "tags_csv": "billing",
                    "chunk_id": "doc-a::sec-1::0",
                },
            }
        ],
    )
    strategy = MetadataFilteredStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    result = await strategy.query(
        "What is the billing policy?",
        access_filter=AccessFilter(max_classification="internal"),
    )

    assert isinstance(result, AnswerResult)
    assert result.strategy == "metadata_filtered"
    assert len(result.retrieval.retrieved) == 1
    assert result.retrieval.retrieved[0].doc_id == "doc-a"

    collection = mock_chroma_client.get_or_create_collection.return_value
    passed_where = collection.query.call_args.kwargs["where"]
    # "internal" allows public and internal, but not confidential or restricted.
    assert {"classification": {"$in": ["public", "internal"]}} in passed_where["$and"]


@pytest.mark.asyncio
async def test_tag_filter_excludes_everything_after_over_fetching(
    mock_chroma_client, mock_llm_client
):
    _set_candidates(
        mock_chroma_client,
        [
            {
                "id": "c1",
                "content": "unrelated content",
                "distance": 0.1,
                "metadata": {
                    "source_id": "doc-a",
                    "classification": "public",
                    "owner": "team-a",
                    "tags_csv": "engineering",
                    "chunk_id": "doc-a::sec-1::0",
                },
            }
        ],
    )
    strategy = MetadataFilteredStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    result = await strategy.query(
        "What is the billing policy?",
        access_filter=AccessFilter(allowed_tags=frozenset({"finance"})),
    )

    collection = mock_chroma_client.get_or_create_collection.return_value
    assert collection.query.called, "tags are filtered after an over-fetch, not before"
    assert result.retrieval.retrieved == []
    assert result.sources == []


@pytest.mark.asyncio
async def test_empty_allow_list_excludes_everything_without_querying(
    mock_chroma_client, mock_llm_client
):
    strategy = MetadataFilteredStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    result = await strategy.query(
        "What is the billing policy?",
        access_filter=AccessFilter(allowed_owners=frozenset()),
    )

    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.query.assert_not_called()
    assert result.retrieval.retrieved == []


@pytest.mark.asyncio
async def test_unknown_classification_raises_instead_of_falling_open(
    mock_chroma_client, mock_llm_client
):
    strategy = MetadataFilteredStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    with pytest.raises(ValueError, match="Unknown classification level"):
        await strategy.query(
            "What is the billing policy?",
            access_filter=AccessFilter(max_classification="ultra-secret"),
        )

    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.query.assert_not_called()
