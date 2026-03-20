"""Tests for BM25 strategy."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.models.document import Document, Section
from kb_arena.strategies.bm25 import BM25Strategy


@pytest.fixture
def sample_docs():
    return [
        Document(
            id="doc1",
            source="lambda.md",
            corpus="test",
            title="Lambda Guide",
            sections=[
                Section(
                    id="s1",
                    title="Overview",
                    content=(
                        "AWS Lambda is a serverless compute service"
                        " that runs code in response to events."
                    ),
                ),
                Section(
                    id="s2",
                    title="Configuration",
                    content="Lambda functions can be configured with memory from 128MB to 10GB.",
                ),
            ],
            metadata={"corpus": "test"},
        ),
        Document(
            id="doc2",
            source="s3.md",
            corpus="test",
            title="S3 Guide",
            sections=[
                Section(
                    id="s3",
                    title="Overview",
                    content=(
                        "Amazon S3 is an object storage service"
                        " offering scalability and durability."
                    ),
                ),
            ],
            metadata={"corpus": "test"},
        ),
    ]


@pytest.fixture
def bm25_strategy():
    return BM25Strategy()


@pytest.fixture(autouse=True)
def _restore_datasets_path():
    """Restore settings.datasets_path after each test."""
    from kb_arena import settings

    original = settings.settings.datasets_path
    yield
    settings.settings.datasets_path = original


@pytest.mark.asyncio
async def test_build_index(bm25_strategy, sample_docs, tmp_path):
    from kb_arena import settings

    settings.settings.datasets_path = str(tmp_path)

    await bm25_strategy.build_index(sample_docs)
    assert bm25_strategy._bm25 is not None
    assert len(bm25_strategy._corpus_texts) == 3


@pytest.mark.asyncio
async def test_query_without_index(bm25_strategy, tmp_path):
    from kb_arena import settings

    settings.settings.datasets_path = str(tmp_path / "empty")
    result = await bm25_strategy.query("What is Lambda?")
    assert "not built" in result.answer


@pytest.mark.asyncio
async def test_query_returns_relevant(bm25_strategy, sample_docs, tmp_path):
    from kb_arena import settings

    settings.settings.datasets_path = str(tmp_path)

    await bm25_strategy.build_index(sample_docs)

    # Mock the LLM
    mock_llm = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.text = "Lambda is a serverless compute service."
    mock_resp.total_tokens = 50
    mock_resp.cost_usd = 0.001
    mock_llm.generate = AsyncMock(return_value=mock_resp)
    bm25_strategy._llm = mock_llm

    result = await bm25_strategy.query("What is Lambda?")
    assert result.answer == "Lambda is a serverless compute service."
    assert result.strategy == "bm25"
    assert result.latency_ms > 0
    assert result.retrieval_latency_ms > 0
    assert result.generation_latency_ms > 0
    assert len(result.sources) > 0


def test_bm25_name():
    s = BM25Strategy()
    assert s.name == "bm25"
