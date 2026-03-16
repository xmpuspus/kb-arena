"""Tests for all 7 retrieval strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kb_arena.generate.qna import parse_qna_json as _parse_qna_json
from kb_arena.models.document import Section
from kb_arena.strategies.base import AnswerResult
from kb_arena.strategies.contextual_vector import (
    ContextualVectorStrategy,
    _enrich_chunk,
    _heading_prefix,
    _section_metadata,
)
from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy, _mock_graph_context
from kb_arena.strategies.naive_vector import NaiveVectorStrategy, _chunk_text
from kb_arena.strategies.qna_pairs import QnAPairStrategy

# --- Chunking helpers ---


def test_chunk_text_basic():
    text = " ".join(str(i) for i in range(1000))
    chunks = _chunk_text(text, chunk_tokens=512, overlap_tokens=50)
    assert len(chunks) > 1
    # First chunk should be 512 tokens
    assert len(chunks[0].split()) == 512


def test_chunk_text_overlap():
    text = " ".join(str(i) for i in range(600))
    chunks = _chunk_text(text, chunk_tokens=512, overlap_tokens=50)
    assert len(chunks) == 2
    # Second chunk starts 50 tokens before end of first
    first_last_tokens = chunks[0].split()[-50:]
    second_first_tokens = chunks[1].split()[:50]
    assert first_last_tokens == second_first_tokens


def test_chunk_text_short():
    text = "short text"
    chunks = _chunk_text(text)
    assert chunks == ["short text"]


def test_chunk_text_empty():
    assert _chunk_text("") == []


# --- NaiveVectorStrategy ---


@pytest.mark.asyncio
async def test_naive_build_index(mock_chroma_client, sample_documents):
    strategy = NaiveVectorStrategy(chroma_client=mock_chroma_client)
    await strategy.build_index(sample_documents)
    # upsert should have been called at least once
    collection = mock_chroma_client.get_or_create_collection.return_value
    assert collection.upsert.called


@pytest.mark.asyncio
async def test_naive_build_index_empty(mock_chroma_client):
    strategy = NaiveVectorStrategy(chroma_client=mock_chroma_client)
    await strategy.build_index([])
    collection = mock_chroma_client.get_or_create_collection.return_value
    assert not collection.upsert.called


@pytest.mark.asyncio
async def test_naive_query(mock_chroma_client, mock_llm_client):
    strategy = NaiveVectorStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    result = await strategy.query("What does json.loads do?")

    assert isinstance(result, AnswerResult)
    assert result.strategy == "naive_vector"
    assert result.answer == "This is a generated answer."
    assert isinstance(result.sources, list)
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_naive_query_uses_top_k(mock_chroma_client, mock_llm_client):
    strategy = NaiveVectorStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    await strategy.query("What does json.loads do?", top_k=3)
    collection = mock_chroma_client.get_or_create_collection.return_value
    call_kwargs = collection.query.call_args
    assert call_kwargs.kwargs.get("n_results") == 3 or call_kwargs[1].get("n_results") == 3


# --- ContextualVectorStrategy ---


def test_heading_prefix_with_path(sample_section):
    prefix = _heading_prefix(sample_section)
    assert "Lambda" in prefix
    assert "Configuration" in prefix


def test_heading_prefix_fallback():
    section = Section(id="test-1", title="My Title", content="...")
    prefix = _heading_prefix(section)
    assert prefix == "My Title"


def test_enrich_chunk(sample_section):
    enriched = _enrich_chunk("some chunk text", sample_section)
    assert "##" in enriched
    assert "Lambda" in enriched
    assert "some chunk text" in enriched


def test_section_metadata(sample_document, sample_section):
    meta = _section_metadata(sample_document, sample_section)
    assert meta["source_id"] == sample_document.id
    assert meta["has_code"] is True
    assert meta["has_table"] is False
    assert "Lambda" in meta["section_path"]


@pytest.mark.asyncio
async def test_contextual_build_adds_heading(mock_chroma_client, sample_document):
    strategy = ContextualVectorStrategy(chroma_client=mock_chroma_client)
    await strategy.build_index([sample_document])

    collection = mock_chroma_client.get_or_create_collection.return_value
    assert collection.upsert.called

    # Verify enriched chunks contain the heading prefix
    call_args = collection.upsert.call_args_list[0]
    documents = (
        call_args.kwargs.get("documents") or call_args[1].get("documents") or call_args[0][1]
    )
    # At least one chunk should start with "##"
    assert any(doc.startswith("##") for doc in documents)


@pytest.mark.asyncio
async def test_contextual_query_with_where_filter(mock_chroma_client, mock_llm_client):
    strategy = ContextualVectorStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    where = {"module": "json"}
    await strategy.query("What is json.loads?", where=where)

    collection = mock_chroma_client.get_or_create_collection.return_value
    call_kwargs = collection.query.call_args
    # where filter should be passed through
    passed_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    assert "where" in passed_kwargs


# --- QnAPairStrategy ---


def test_parse_qna_json_valid():
    raw = '[{"question": "What is X?", "answer": "X is Y."}]'
    pairs = _parse_qna_json(raw)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "What is X?"


def test_parse_qna_json_with_markdown_fence():
    raw = '```json\n[{"question": "Q?", "answer": "A."}]\n```'
    pairs = _parse_qna_json(raw)
    assert len(pairs) == 1


def test_parse_qna_json_invalid():
    pairs = _parse_qna_json("not json at all")
    assert pairs == []


@pytest.mark.asyncio
async def test_qna_query_returns_pregenerated_answer(mock_chroma_client, mock_llm_client):
    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.query.return_value = {
        "ids": [["qna::doc1::sec1::0"]],
        "documents": [["What does json.loads do?"]],
        "metadatas": [
            [
                {
                    "answer": "json.loads deserializes a JSON string to Python.",
                    "source_id": "aws-compute-lambda",
                    "section_id": "json-loads",
                }
            ]
        ],
        "distances": [[0.05]],
    }

    strategy = QnAPairStrategy(chroma_client=mock_chroma_client, llm_client=mock_llm_client)
    # Exact question match — should return pre-generated answer without LLM rephrase
    result = await strategy.query("What does json.loads do?")

    assert isinstance(result, AnswerResult)
    assert result.strategy == "qna_pairs"
    assert "json.loads" in result.answer.lower() or "deserializes" in result.answer.lower()


@pytest.mark.asyncio
async def test_qna_query_empty_collection(mock_chroma_client, mock_llm_client):
    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    strategy = QnAPairStrategy(chroma_client=mock_chroma_client, llm_client=mock_llm_client)
    result = await strategy.query("What is X?")
    assert "No relevant" in result.answer


# --- KnowledgeGraphStrategy ---


def test_mock_graph_context():
    ctx = _mock_graph_context()
    assert len(ctx.nodes) >= 2
    assert len(ctx.edges) >= 1
    assert "MOCK" in ctx.cypher_used


@pytest.mark.asyncio
async def test_knowledge_graph_mock_fallback():
    """When Neo4j is None, strategy returns mock data with warning."""
    strategy = KnowledgeGraphStrategy(neo4j_driver=None)
    result = await strategy.query("What is json.loads?")

    assert result.mock is True
    assert result.graph_context is not None
    assert len(result.graph_context.nodes) >= 2
    assert "not connected" in result.answer.lower()
    assert result.strategy == "knowledge_graph"


@pytest.mark.asyncio
async def test_knowledge_graph_with_driver(mock_neo4j_driver, mock_llm_client):
    """With a connected driver, runs Cypher and generates an answer."""
    strategy = KnowledgeGraphStrategy(neo4j_driver=mock_neo4j_driver)
    strategy._llm = mock_llm_client

    result = await strategy.query("What is json.loads?")

    assert isinstance(result, AnswerResult)
    assert result.mock is False
    assert result.strategy == "knowledge_graph"


# --- HybridStrategy ---


@pytest.mark.asyncio
async def test_hybrid_routes_comparison_to_graph(
    mock_chroma_client, mock_neo4j_driver, mock_llm_client
):
    from kb_arena.strategies.hybrid import HybridStrategy

    strategy = HybridStrategy(
        neo4j_driver=mock_neo4j_driver,
        chroma_client=mock_chroma_client,
    )
    strategy._llm = mock_llm_client

    # Inject mocked sub-strategies
    mock_graph = AsyncMock()
    mock_graph.name = "knowledge_graph"
    mock_graph.last_sources = []
    mock_graph.last_graph_context = None
    mock_graph.last_latency_ms = 0.0
    mock_graph.query = AsyncMock(
        return_value=AnswerResult(answer="graph answer", sources=["g1"], strategy="knowledge_graph")
    )
    strategy._graph_strategy = mock_graph

    result = await strategy.query("compare json.loads vs yaml.safe_load")

    assert result.strategy == "hybrid"
    mock_graph.query.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_routes_factoid_to_vector(mock_chroma_client, mock_llm_client):
    from kb_arena.strategies.hybrid import HybridStrategy

    strategy = HybridStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    mock_vector = AsyncMock()
    mock_vector.name = "contextual_vector"
    mock_vector.last_sources = []
    mock_vector.last_latency_ms = 0.0
    mock_vector.query = AsyncMock(
        return_value=AnswerResult(
            answer="vector answer", sources=["v1"], strategy="contextual_vector"
        )
    )
    strategy._vector_strategy = mock_vector

    result = await strategy.query("what is json.loads?")

    assert result.strategy == "hybrid"
    mock_vector.query.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_procedural_fuses_both(mock_chroma_client, mock_neo4j_driver, mock_llm_client):
    from kb_arena.models.graph import GraphContext
    from kb_arena.strategies.hybrid import HybridStrategy

    strategy = HybridStrategy(
        neo4j_driver=mock_neo4j_driver,
        chroma_client=mock_chroma_client,
    )
    strategy._llm = mock_llm_client

    mock_vector = AsyncMock()
    mock_vector.query = AsyncMock(
        return_value=AnswerResult(
            answer="vector answer for procedure", sources=["v1"], strategy="contextual_vector"
        )
    )
    strategy._vector_strategy = mock_vector

    mock_graph = AsyncMock()
    mock_graph.query = AsyncMock(
        return_value=AnswerResult(
            answer="graph answer for procedure",
            sources=["g1"],
            strategy="knowledge_graph",
            graph_context=GraphContext(nodes=[], edges=[]),
        )
    )
    strategy._graph_strategy = mock_graph

    result = await strategy.query("how do I configure json encoder?")

    assert result.strategy == "hybrid"
    # Both sub-strategies should have been queried
    mock_vector.query.assert_called_once()
    mock_graph.query.assert_called_once()


# --- RaptorStrategy ---


@pytest.mark.asyncio
async def test_raptor_build_index(mock_chroma_client, sample_documents):
    """build_index should upsert L0 and attempt L1."""
    import numpy as np

    collection = mock_chroma_client.get_or_create_collection.return_value
    n_chunks = sum(len(s.content.split()) // 512 + 1 for d in sample_documents for s in d.sections)
    fake_ids = [f"chunk_{i}" for i in range(max(n_chunks, 3))]
    fake_emb = np.random.rand(len(fake_ids), 8).tolist()
    fake_docs = [f"chunk text {i}" for i in range(len(fake_ids))]
    collection.get.return_value = {"ids": fake_ids, "embeddings": fake_emb, "documents": fake_docs}
    collection.count.return_value = len(fake_ids)

    from unittest.mock import AsyncMock

    from kb_arena.llm.client import LLMResponse
    from kb_arena.strategies.raptor import RaptorStrategy

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        text="Cluster summary.", input_tokens=100, output_tokens=50, cost_usd=0.001
    )

    strategy = RaptorStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm

    await strategy.build_index(sample_documents)
    assert collection.upsert.called


@pytest.mark.asyncio
async def test_raptor_query(mock_chroma_client, mock_llm_client):
    """query should search all levels and return AnswerResult."""
    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.count.return_value = 3

    from kb_arena.strategies.raptor import RaptorStrategy

    strategy = RaptorStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    result = await strategy.query("What is Lambda?")
    assert result.strategy == "raptor"
    assert isinstance(result.answer, str)
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_raptor_query_empty_collection(mock_chroma_client, mock_llm_client):
    """query on empty collection returns helpful message."""
    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.count.return_value = 0

    from kb_arena.strategies.raptor import RaptorStrategy

    strategy = RaptorStrategy(chroma_client=mock_chroma_client)
    strategy._llm = mock_llm_client

    result = await strategy.query("What is Lambda?")
    assert "build-vectors" in result.answer or "No indexed" in result.answer


def test_cosine_kmeans_basic():
    """K-means returns one assignment per embedding."""
    import numpy as np

    from kb_arena.strategies.raptor import _cosine_kmeans

    rng = np.random.default_rng(42)
    embeddings = rng.random((20, 8)).astype(np.float32)
    assignments = _cosine_kmeans(embeddings, k=4)
    assert len(assignments) == 20
    assert all(0 <= a < 4 for a in assignments)


def test_cosine_kmeans_fewer_than_k():
    """When n <= k, each point is its own cluster."""
    import numpy as np

    from kb_arena.strategies.raptor import _cosine_kmeans

    embeddings = np.random.rand(3, 8).astype(np.float32)
    assignments = _cosine_kmeans(embeddings, k=5)
    assert assignments == [0, 1, 2]
