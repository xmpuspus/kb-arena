"""Each retrieval strategy populates AnswerResult.retrieval with stable chunk IDs.

These are mocked tests — no real Chroma/Neo4j. The point is to assert the
contract: every strategy's query() returns RetrievalTrace with rank-ordered
chunks carrying chunk_id, doc_id, content, score, and source_strategy.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.llm.client import LLMResponse
from kb_arena.models.retrieval import RetrievalTrace


def _fake_collection(ids, docs, source_id="d1", distances=None):
    """Build a Chroma-like collection that returns the given ids/docs."""
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [ids],
        "documents": [docs],
        "metadatas": [[{"source_id": source_id} for _ in ids]],
        "distances": [distances or [0.1 * (i + 1) for i in range(len(ids))]],
    }
    collection.count.return_value = len(ids)
    return collection


def _llm_resp() -> LLMResponse:
    return LLMResponse(text="answer", input_tokens=10, output_tokens=5, cost_usd=0.0001)


@pytest.mark.asyncio
async def test_naive_vector_populates_trace(monkeypatch):
    from kb_arena.strategies.naive_vector import NaiveVectorStrategy

    s = NaiveVectorStrategy()
    fake = _fake_collection(
        ids=["doc1::s1::0", "doc1::s1::1", "doc2::s2::0"],
        docs=["chunk a", "chunk b", "chunk c"],
    )
    s._collection = fake
    s._llm = AsyncMock()
    s._llm.generate = AsyncMock(return_value=_llm_resp())

    result = await s.query("test", top_k=3)

    assert result.retrieval is not None
    assert isinstance(result.retrieval, RetrievalTrace)
    assert len(result.retrieval.retrieved) == 3
    assert [c.rank for c in result.retrieval.retrieved] == [1, 2, 3]
    assert all(c.source_strategy == "naive_vector" for c in result.retrieval.retrieved)
    assert result.retrieval.retrieved[0].chunk_id == "doc1::s1::0"
    assert result.retrieval.retrieved[0].doc_id == "d1"


@pytest.mark.asyncio
async def test_contextual_vector_populates_trace():
    from kb_arena.strategies.contextual_vector import ContextualVectorStrategy

    s = ContextualVectorStrategy()
    s._collection = _fake_collection(ids=["a::1::0", "b::1::0"], docs=["one", "two"])
    s._llm = AsyncMock()
    s._llm.generate = AsyncMock(return_value=_llm_resp())

    result = await s.query("q", top_k=2)
    assert result.retrieval is not None
    assert len(result.retrieval.retrieved) == 2
    assert all(c.source_strategy == "contextual_vector" for c in result.retrieval.retrieved)


@pytest.mark.asyncio
async def test_bm25_populates_trace_with_stable_ids(tmp_path, monkeypatch):
    from kb_arena.strategies.bm25 import BM25Strategy

    s = BM25Strategy()
    # Manually populate index in-memory (skip persistence)
    s._corpus_texts = ["lambda timeout config", "s3 storage class", "ec2 instance type"]
    s._corpus_sources = ["lambda.html", "s3.html", "ec2.html"]
    s._chunk_ids = ["lambda.html::sect1", "s3.html::sect1", "ec2.html::sect1"]
    from rank_bm25 import BM25Okapi

    s._bm25 = BM25Okapi([t.lower().split() for t in s._corpus_texts])
    s._llm = AsyncMock()
    s._llm.generate = AsyncMock(return_value=_llm_resp())

    result1 = await s.query("lambda timeout", top_k=2)
    result2 = await s.query("lambda timeout", top_k=2)

    assert result1.retrieval is not None
    assert len(result1.retrieval.retrieved) == 2
    # Stable IDs across runs
    ids1 = [c.chunk_id for c in result1.retrieval.retrieved]
    ids2 = [c.chunk_id for c in result2.retrieval.retrieved]
    assert ids1 == ids2
    # ID format check
    assert all("::" in cid for cid in ids1)


@pytest.mark.asyncio
async def test_qna_pairs_populates_trace():
    from kb_arena.strategies.qna_pairs import QnAPairStrategy

    s = QnAPairStrategy()
    fake = MagicMock()
    fake.query.return_value = {
        "ids": [["pair-001", "pair-002"]],
        "documents": [["What is Lambda?", "How to invoke?"]],
        "metadatas": [
            [
                {"source_id": "lambda.html", "answer": "Lambda is..."},
                {"source_id": "lambda.html", "answer": "Use invoke API"},
            ]
        ],
        "distances": [[0.1, 0.2]],
    }
    s._collection = fake

    result = await s.query("What is Lambda?", top_k=2)
    assert result.retrieval is not None
    assert len(result.retrieval.retrieved) == 2
    assert result.retrieval.retrieved[0].chunk_id == "qna:pair-001"
    assert "Lambda is..." in result.retrieval.retrieved[0].content


@pytest.mark.asyncio
async def test_raptor_populates_trace_with_level_prefix():
    from kb_arena.strategies.raptor import RaptorStrategy

    s = RaptorStrategy()
    # Mock _get_collection to return level-tagged fakes
    fakes = {
        0: _fake_collection(ids=["c0a", "c0b"], docs=["x", "y"]),
        1: _fake_collection(ids=["c1a"], docs=["z"]),
        2: MagicMock(count=MagicMock(return_value=0)),
    }
    s._get_collection = lambda level: fakes[level]
    s._llm = AsyncMock()
    s._llm.generate = AsyncMock(return_value=_llm_resp())

    result = await s.query("q", top_k=2)
    assert result.retrieval is not None
    assert len(result.retrieval.retrieved) >= 2
    # Level prefix
    assert any(c.chunk_id.startswith("L0:") for c in result.retrieval.retrieved)
    assert any(c.chunk_id.startswith("L1:") for c in result.retrieval.retrieved)


@pytest.mark.asyncio
async def test_knowledge_graph_synthesizes_chunks():
    from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy

    s = KnowledgeGraphStrategy()
    # Fake driver returning records
    s._run_cypher = AsyncMock(
        return_value=[
            {"name": "Lambda", "fqn": "aws.lambda", "type": "Component", "score": 0.9},
            {"name": "S3", "fqn": "aws.s3", "type": "Component", "score": 0.7},
        ]
    )
    s._driver = MagicMock()  # not None so we don't fall into mock-mode
    s._llm = AsyncMock()
    s._llm.generate = AsyncMock(return_value=_llm_resp())

    result = await s.query("what is lambda", top_k=5)
    assert result.retrieval is not None
    assert len(result.retrieval.retrieved) == 2
    assert result.retrieval.retrieved[0].chunk_id == "graph:aws.lambda"
    assert result.retrieval.retrieved[0].source_strategy == "knowledge_graph"


@pytest.mark.asyncio
async def test_knowledge_graph_mock_mode_emits_empty_trace():
    from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy

    s = KnowledgeGraphStrategy(neo4j_driver=None)
    result = await s.query("anything", top_k=5)
    assert result.retrieval is not None
    assert result.retrieval.retrieved == []


@pytest.mark.asyncio
async def test_pageindex_populates_trace():
    from kb_arena.strategies.pageindex import PageIndexStrategy, TreeNode

    s = PageIndexStrategy()
    leaf1 = TreeNode(
        id="sec-1",
        title="Lambda",
        level=2,
        content="Lambda is a serverless compute service.",
        source_doc="lambda.html",
    )
    leaf2 = TreeNode(
        id="sec-2",
        title="S3",
        level=2,
        content="S3 is object storage.",
        source_doc="s3.html",
    )

    # Bypass tree loading and beam search — provide answers directly
    s._load_all_trees = lambda: [MagicMock(documents=[leaf1, leaf2])]

    async def fake_traverse(*args, **kwargs):
        return [leaf1, leaf2], 0.0001

    import kb_arena.strategies.pageindex as pi

    pi._beam_traverse = fake_traverse  # type: ignore[assignment]

    s._llm = AsyncMock()
    s._llm.generate = AsyncMock(return_value=_llm_resp())

    result = await s.query("what is lambda", top_k=2)
    assert result.retrieval is not None
    assert len(result.retrieval.retrieved) == 2
    assert result.retrieval.retrieved[0].chunk_id.startswith("pageindex:")
    assert result.retrieval.retrieved[0].source_strategy == "pageindex"


@pytest.mark.asyncio
async def test_hybrid_merges_sub_traces():
    from kb_arena.models.retrieval import RetrievalTrace as RetTrace
    from kb_arena.models.retrieval import RetrievedChunk
    from kb_arena.strategies.base import AnswerResult
    from kb_arena.strategies.hybrid import HybridStrategy

    s = HybridStrategy()

    vec_trace = RetTrace(
        query="q",
        top_k=5,
        retrieved=[
            RetrievedChunk(
                chunk_id="v1", doc_id="d1", content="x", score=0.9, rank=1, source_strategy="vec"
            ),
            RetrievedChunk(
                chunk_id="v2", doc_id="d2", content="y", score=0.8, rank=2, source_strategy="vec"
            ),
        ],
    )
    graph_trace = RetTrace(
        query="q",
        top_k=5,
        retrieved=[
            RetrievedChunk(
                chunk_id="g1", doc_id="d3", content="z", score=0.95, rank=1, source_strategy="graph"
            ),
        ],
    )

    fake_vector = AsyncMock()
    fake_vector.query = AsyncMock(
        return_value=AnswerResult(
            answer="vec answer", sources=["d1"], retrieval=vec_trace, strategy="vec"
        )
    )
    fake_graph = AsyncMock()
    fake_graph.query = AsyncMock(
        return_value=AnswerResult(
            answer="graph answer", sources=["d3"], retrieval=graph_trace, strategy="graph"
        )
    )
    s._vector_strategy = fake_vector
    s._graph_strategy = fake_graph
    s._llm = AsyncMock()
    s._llm.generate = AsyncMock(return_value=_llm_resp())
    # Force factoid intent → vector-primary path (single trace, but still merged)
    s._classify = AsyncMock(return_value="factoid")

    result = await s.query("q", top_k=5)
    assert result.retrieval is not None
    assert len(result.retrieval.retrieved) == 2
    # source_strategy preserved from sub-trace
    assert all(c.source_strategy == "vec" for c in result.retrieval.retrieved)


@pytest.mark.asyncio
async def test_hybrid_procedural_dedupes():
    """Procedural intent should fuse both, dedupe on chunk_id, and re-rank by score."""
    from kb_arena.models.retrieval import RetrievalTrace as RetTrace
    from kb_arena.models.retrieval import RetrievedChunk
    from kb_arena.strategies.base import AnswerResult
    from kb_arena.strategies.hybrid import HybridStrategy

    s = HybridStrategy()
    chunk = RetrievedChunk(
        chunk_id="shared", doc_id="d1", content="x", score=0.5, rank=1, source_strategy="vec"
    )
    vec = RetTrace(query="q", top_k=5, retrieved=[chunk])
    graph = RetTrace(query="q", top_k=5, retrieved=[chunk])  # same chunk_id

    fake_vector = AsyncMock()
    fake_vector.query = AsyncMock(
        return_value=AnswerResult(answer="a", retrieval=vec, strategy="vec")
    )
    fake_graph = AsyncMock()
    fake_graph.query = AsyncMock(
        return_value=AnswerResult(answer="b", retrieval=graph, strategy="graph")
    )
    s._vector_strategy = fake_vector
    s._graph_strategy = fake_graph
    s._llm = AsyncMock()
    s._llm.generate = AsyncMock(return_value=_llm_resp())
    s._classify = AsyncMock(return_value="procedural")

    result = await s.query("how do I", top_k=5)
    assert result.retrieval is not None
    assert len(result.retrieval.retrieved) == 1  # deduped
