"""Live tests for the embedding pipeline — OpenAI embeddings + ChromaDB round-trip."""

from __future__ import annotations

import math

import chromadb
import pytest

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def openai_client(live_openai_key):
    import openai
    return openai.OpenAI(api_key=live_openai_key)


@pytest.fixture(scope="module")
def embedding_model(live_settings):
    # Use the model from settings (may be overridden by .env)
    return live_settings.embedding_model


@pytest.fixture(scope="module")
def expected_dims(live_settings):
    # text-embedding-3-small → 1536, text-embedding-3-large → 3072
    model = live_settings.embedding_model
    if "large" in model:
        return 3072
    elif "small" in model:
        return 1536
    return 3072  # default to large


def _embed(client, text: str, model: str) -> list[float]:
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_embedding_dimensions(openai_client, embedding_model, expected_dims):
    vec = _embed(openai_client, "json.loads", embedding_model)
    assert len(vec) == expected_dims, (
        f"Expected {expected_dims} dims for {embedding_model}, got {len(vec)}"
    )


def test_embedding_is_float_array(openai_client, embedding_model):
    vec = _embed(openai_client, "test text", embedding_model)
    assert all(isinstance(v, float) for v in vec[:10])


def test_embedding_batch_dimensions(openai_client, embedding_model, expected_dims):
    texts = [
        "json.loads deserializes a JSON string",
        "json.dumps serializes a Python object",
        "os.path.join builds file paths",
        "pathlib.Path provides OOP file paths",
        "the asyncio event loop runs coroutines",
    ]
    response = openai_client.embeddings.create(input=texts, model=embedding_model)
    assert len(response.data) == len(texts)
    for item in response.data:
        assert len(item.embedding) == expected_dims


def test_similar_texts_closer_than_dissimilar(openai_client, embedding_model):
    """json.loads should be semantically closer to json.dumps than to os.path.join."""
    v_loads = _embed(openai_client, "json.loads: deserialize JSON string to Python object", embedding_model)
    v_dumps = _embed(openai_client, "json.dumps: serialize Python object to JSON string", embedding_model)
    v_path = _embed(openai_client, "os.path.join: join file system path components", embedding_model)

    sim_related = _cosine_sim(v_loads, v_dumps)
    sim_unrelated = _cosine_sim(v_loads, v_path)

    assert sim_related > sim_unrelated, (
        f"json.loads should be closer to json.dumps ({sim_related:.3f}) "
        f"than to os.path.join ({sim_unrelated:.3f})"
    )


def test_chroma_add_and_query_round_trip(tmp_path, live_settings, live_openai_key):
    """Add documents to ChromaDB with real embeddings, then query and recover them."""
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    ef = OpenAIEmbeddingFunction(
        api_key=live_openai_key,
        model_name=live_settings.embedding_model,
    )

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_test"))
    col = client.get_or_create_collection(
        name="test_live",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    docs = [
        "json.loads deserializes a JSON string to a Python object",
        "json.dumps serializes a Python object to a JSON string",
        "os.path.join combines path components into a single path",
        "pathlib.Path provides an object-oriented interface to filesystem paths",
        "asyncio.run executes a coroutine and returns the result",
    ]
    ids = [f"doc-{i}" for i in range(len(docs))]
    col.upsert(ids=ids, documents=docs)

    results = col.query(
        query_texts=["how do I parse a JSON string?"],
        n_results=2,
    )

    retrieved = results["documents"][0]
    assert len(retrieved) == 2
    # The most relevant result should be about json.loads
    assert any("json" in d.lower() for d in retrieved)


def test_contextual_chunks_differ_from_raw(openai_client, embedding_model):
    """A chunk with heading_path prepended embeds differently than raw chunk."""
    raw = "Deserialize s to a Python object."
    contextual = "## json > json.loads\n\nDeserialize s to a Python object."

    v_raw = _embed(openai_client, raw, embedding_model)
    v_ctx = _embed(openai_client, contextual, embedding_model)

    sim = _cosine_sim(v_raw, v_ctx)
    # They should be similar but not identical (contextual has extra context)
    assert sim < 1.0, "Raw and contextual embeddings should differ"
    assert sim > 0.7, "Embeddings should still be semantically close"


def test_chroma_query_returns_correct_n_results(tmp_path, live_settings, live_openai_key):
    """n_results parameter is respected."""
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    ef = OpenAIEmbeddingFunction(
        api_key=live_openai_key,
        model_name=live_settings.embedding_model,
    )

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_nresults"))
    col = client.get_or_create_collection(
        name="test_nresults",
        embedding_function=ef,
    )
    col.upsert(
        ids=["a", "b", "c", "d", "e"],
        documents=["alpha", "beta", "gamma", "delta", "epsilon"],
    )

    for n in [1, 3, 5]:
        results = col.query(query_texts=["test query"], n_results=n)
        assert len(results["documents"][0]) == n
