"""Tests for Strategy: Temporal — version-aware dense retrieval.

The strategy must prefer the newest eligible version of a document family,
support an "as of" date that surfaces an older version, and fail closed on a
filter it cannot evaluate.
"""

from __future__ import annotations

import pytest

from kb_arena.strategies.temporal import TemporalStrategy


def _two_versions(mock_chroma_client) -> None:
    """Two candidates from the same family: an older and a newer version."""
    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.query.return_value = {
        "ids": [["v1-c0", "v2-c0"]],
        "documents": [["policy v1 text", "policy v2 text"]],
        "metadatas": [
            [
                {
                    "source_id": "policy-v1",
                    "document_family": "policy",
                    "version": 1,
                    "effective_date": "2024-01-01",
                    "chunk_id": "policy-v1::sec-1::0",
                },
                {
                    "source_id": "policy-v2",
                    "document_family": "policy",
                    "version": 2,
                    "effective_date": "2025-01-01",
                    "chunk_id": "policy-v2::sec-1::0",
                },
            ]
        ],
        "distances": [[0.3, 0.2]],
    }


@pytest.mark.asyncio
async def test_happy_path_prefers_the_newest_version(mock_chroma_client, mock_llm_client):
    _two_versions(mock_chroma_client)
    strategy = TemporalStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    result = await strategy.query("What is the current policy?")

    assert result.strategy == "temporal"
    doc_ids = [c.doc_id for c in result.retrieval.retrieved]
    assert doc_ids == ["policy-v2"], "the older version must not outrank its replacement"


@pytest.mark.asyncio
async def test_as_of_an_earlier_date_returns_the_older_version(mock_chroma_client, mock_llm_client):
    _two_versions(mock_chroma_client)
    strategy = TemporalStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    result = await strategy.query("What was the policy?", as_of="2024-06-01")

    doc_ids = [c.doc_id for c in result.retrieval.retrieved]
    assert doc_ids == ["policy-v1"]


@pytest.mark.asyncio
async def test_as_of_before_any_version_excludes_everything(mock_chroma_client, mock_llm_client):
    _two_versions(mock_chroma_client)
    strategy = TemporalStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    result = await strategy.query("What was the policy?", as_of="1999-01-01")

    assert result.retrieval.retrieved == []
    assert result.sources == []


@pytest.mark.asyncio
async def test_unparseable_as_of_raises_instead_of_falling_open(
    mock_chroma_client, mock_llm_client
):
    strategy = TemporalStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    with pytest.raises(ValueError, match="as_of must be an ISO date"):
        await strategy.query("What was the policy?", as_of="not-a-date")

    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.query.assert_not_called()
