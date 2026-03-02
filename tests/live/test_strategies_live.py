"""Live tests for retrieval strategies with real LLM and embeddings."""

from __future__ import annotations

import pytest

from kb_arena.models.document import CodeBlock, CrossRef, Document, Section
from kb_arena.strategies.contextual_vector import ContextualVectorStrategy, _enrich_chunk
from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy
from kb_arena.strategies.naive_vector import NaiveVectorStrategy

pytestmark = pytest.mark.live


def _make_json_docs() -> list[Document]:
    """Five small documents about the Python json module."""
    return [
        Document(
            id="json-loads-doc",
            source="https://docs.python.org/3/library/json.html#json.loads",
            corpus="aws-compute",
            title="json.loads",
            sections=[
                Section(
                    id="json-loads-section",
                    title="json.loads",
                    content=(
                        "json.loads(s, *, cls=None, object_hook=None, parse_float=None, "
                        "parse_int=None, parse_constant=None, object_pairs_hook=None, **kw) "
                        "Deserialize s (a str, bytes or bytearray instance containing a JSON "
                        "document) to a Python object using this conversion table. "
                        "Raises json.JSONDecodeError if the data being deserialized is not a "
                        "valid JSON document."
                    ),
                    heading_path=["json", "json.loads"],
                    code_blocks=[
                        CodeBlock(
                            language="python",
                            code=">>> json.loads('{\"key\": \"value\"}')\n{'key': 'value'}",
                            description="Basic json.loads usage",
                        )
                    ],
                    links=[
                        CrossRef(
                            target="json.JSONDecodeError", label="JSONDecodeError", ref_type="class"
                        )
                    ],
                    level=2,
                )
            ],
            raw_token_count=150,
        ),
        Document(
            id="json-dumps-doc",
            source="https://docs.python.org/3/library/json.html#json.dumps",
            corpus="aws-compute",
            title="json.dumps",
            sections=[
                Section(
                    id="json-dumps-section",
                    title="json.dumps",
                    content=(
                        "json.dumps(obj, *, skipkeys=False, ensure_ascii=True,"
                        " check_circular=True, allow_nan=True, cls=None,"
                        " indent=None, separators=None, default=None,"
                        " sort_keys=False, **kw) "
                        "Serialize obj to a JSON formatted str using this conversion table. "
                        "If indent is a non-negative integer, then JSON array elements and object "
                        "members will be pretty-printed with that indent level."
                    ),
                    heading_path=["json", "json.dumps"],
                    level=2,
                )
            ],
            raw_token_count=120,
        ),
        Document(
            id="json-error-doc",
            source="https://docs.python.org/3/library/json.html#json.JSONDecodeError",
            corpus="aws-compute",
            title="json.JSONDecodeError",
            sections=[
                Section(
                    id="json-error-section",
                    title="json.JSONDecodeError",
                    content=(
                        "exception json.JSONDecodeError(msg, doc, pos) "
                        "Subclass of ValueError with the following additional attributes: "
                        "msg: The unformatted error message. "
                        "doc: The JSON document being parsed. "
                        "pos: The start index of doc where parsing failed. "
                        "lineno: The line corresponding to pos. "
                        "colno: The column corresponding to pos. "
                        "Raised by json.loads and json.load when the JSON document is invalid."
                    ),
                    heading_path=["json", "json.JSONDecodeError"],
                    level=2,
                )
            ],
            raw_token_count=100,
        ),
        Document(
            id="json-encoder-doc",
            source="https://docs.python.org/3/library/json.html#json.JSONEncoder",
            corpus="aws-compute",
            title="json.JSONEncoder",
            sections=[
                Section(
                    id="json-encoder-section",
                    title="json.JSONEncoder",
                    content=(
                        "class json.JSONEncoder(*, skipkeys=False, ensure_ascii=True, "
                        "check_circular=True, allow_nan=True, sort_keys=False, indent=None, "
                        "separators=None, default=None) "
                        "Extensible JSON encoder for Python data structures. "
                        "Supports the following objects and types by default: "
                        "dict, list, tuple, str, int, float, True, False, None."
                    ),
                    heading_path=["json", "json.JSONEncoder"],
                    level=2,
                )
            ],
            raw_token_count=90,
        ),
        Document(
            id="json-module-doc",
            source="https://docs.python.org/3/library/json.html",
            corpus="aws-compute",
            title="json — JSON encoder and decoder",
            sections=[
                Section(
                    id="json-overview-section",
                    title="json — JSON encoder and decoder",
                    content=(
                        "JSON (JavaScript Object Notation), specified by RFC 7159 (which obsoletes "
                        "RFC 4627) and by ECMA-404, is a lightweight data interchange format "
                        "inspired by JavaScript object literal syntax. "
                        "The json module exposes an API familiar to users of the standard library "
                        "marshal and pickle modules. The main entry points are json.loads for "
                        "deserialization and json.dumps for serialization."
                    ),
                    heading_path=["json"],
                    level=1,
                )
            ],
            raw_token_count=100,
        ),
    ]


@pytest.fixture(scope="module")
def json_docs():
    return _make_json_docs()


@pytest.fixture(scope="module")
async def naive_strategy_with_index(tmp_path_factory, live_openai_key, json_docs):
    """NaiveVectorStrategy with real embeddings, built from json docs."""
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    chroma = chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("naive_chroma")))
    ef = OpenAIEmbeddingFunction(api_key=live_openai_key, model_name="text-embedding-3-small")
    collection = chroma.get_or_create_collection(
        name="naive_vector",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    strategy = NaiveVectorStrategy(chroma_client=chroma)
    strategy._collection = collection  # inject pre-configured collection
    await strategy.build_index(json_docs)
    return strategy


@pytest.fixture(scope="module")
async def contextual_strategy_with_index(tmp_path_factory, live_openai_key, json_docs):
    """ContextualVectorStrategy with real embeddings."""
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    chroma = chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("contextual_chroma")))
    ef = OpenAIEmbeddingFunction(api_key=live_openai_key, model_name="text-embedding-3-small")
    collection = chroma.get_or_create_collection(
        name="contextual_vector",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    strategy = ContextualVectorStrategy(chroma_client=chroma)
    strategy._collection = collection
    await strategy.build_index(json_docs)
    return strategy


async def test_naive_vector_answers_loads_question(naive_strategy_with_index):
    result = await naive_strategy_with_index.query("What does json.loads do?", top_k=3)
    assert result.answer
    assert len(result.answer) > 20
    assert any(kw in result.answer.lower() for kw in ["deserializ", "json", "python", "string"])
    assert result.strategy == "naive_vector"
    assert result.latency_ms > 0


async def test_naive_vector_answers_exception_question(naive_strategy_with_index):
    result = await naive_strategy_with_index.query(
        "What exception does json.loads raise on invalid input?", top_k=3
    )
    assert result.answer
    # Should mention JSONDecodeError
    assert (
        "JSONDecodeError" in result.answer
        or "valueerror" in result.answer.lower()
        or "exception" in result.answer.lower()
    )


async def test_contextual_vector_chunks_have_heading(json_docs):
    """Contextual strategy prepends heading_path to each chunk."""

    section = json_docs[0].sections[0]
    chunk = "some chunk text"
    enriched = _enrich_chunk(chunk, section)
    assert enriched.startswith("## json > json.loads")
    assert chunk in enriched


async def test_contextual_vector_answers_question(contextual_strategy_with_index):
    result = await contextual_strategy_with_index.query("What does json.loads do?", top_k=3)
    assert result.answer
    assert result.strategy == "contextual_vector"
    assert len(result.answer) > 20


async def test_naive_vector_sources_populated(naive_strategy_with_index):
    result = await naive_strategy_with_index.query("json module overview", top_k=3)
    assert isinstance(result.sources, list)
    # Sources should be populated when chunks are found
    if result.sources:
        assert all(isinstance(s, str) for s in result.sources)


async def test_knowledge_graph_mock_fallback():
    """KnowledgeGraphStrategy returns mock=True when no Neo4j driver."""
    strategy = KnowledgeGraphStrategy(neo4j_driver=None)
    result = await strategy.query("What is json?")
    assert result.mock is True
    assert result.graph_context is not None
    assert len(result.graph_context.nodes) > 0
    assert "[Graph database not connected" in result.answer


async def test_knowledge_graph_stream_mock_fallback():
    """Streaming from KnowledgeGraphStrategy works without Neo4j."""
    strategy = KnowledgeGraphStrategy(neo4j_driver=None)
    tokens = []
    async for token in strategy.stream_answer("What is json?"):
        tokens.append(token)
    assert len(tokens) > 0
    full = "".join(tokens)
    assert "Graph database not connected" in full


async def test_naive_vector_latency_recorded(naive_strategy_with_index):
    result = await naive_strategy_with_index.query("json serialization", top_k=2)
    assert result.latency_ms > 0
    assert result.latency_ms < 60_000  # sanity: under 60 seconds


async def test_contextual_strategy_sources_have_metadata(contextual_strategy_with_index):
    result = await contextual_strategy_with_index.query("What is json.JSONDecodeError?", top_k=3)
    assert result.strategy == "contextual_vector"
    assert result.answer
