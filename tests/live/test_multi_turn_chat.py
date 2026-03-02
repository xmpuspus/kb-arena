"""Live tests for multi-turn conversation — session memory and context retention."""

from __future__ import annotations

import pytest

from kb_arena.chatbot.session import MAX_ASSISTANT_CHARS, MAX_TURNS, SessionMemory

pytestmark = pytest.mark.live


# --- SessionMemory unit tests (no LLM needed) ---


def test_session_memory_basic_add():
    mem = SessionMemory()
    mem.add_turn("user", "What is json?")
    mem.add_turn("assistant", "json is a module.")
    assert len(mem) == 2
    history = mem.get_history()
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_session_memory_truncates_assistant_message():
    mem = SessionMemory()
    long_answer = "A" * 1000
    mem.add_turn("assistant", long_answer)
    history = mem.get_history()
    assert len(history[0]["content"]) <= MAX_ASSISTANT_CHARS + 3  # +3 for "..."
    assert history[0]["content"].endswith("...")


def test_session_memory_does_not_truncate_short_assistant():
    mem = SessionMemory()
    short = "Short answer."
    mem.add_turn("assistant", short)
    assert mem.get_history()[0]["content"] == short


def test_session_memory_evicts_oldest_after_max_turns():
    mem = SessionMemory()
    for i in range(MAX_TURNS + 2):
        mem.add_turn("user", f"user message {i}")
        mem.add_turn("assistant", f"assistant reply {i}")

    history = mem.get_history()
    # Should cap at MAX_TURNS * 2 messages
    assert len(history) == MAX_TURNS * 2

    # Oldest messages should be gone
    contents = [h["content"] for h in history]
    assert "user message 0" not in contents
    assert f"user message {MAX_TURNS + 1}" in contents


def test_session_memory_get_history_returns_copy():
    mem = SessionMemory()
    mem.add_turn("user", "hello")
    h1 = mem.get_history()
    h1.append({"role": "user", "content": "injected"})
    h2 = mem.get_history()
    assert len(h2) == 1  # original not modified


def test_session_memory_clear():
    mem = SessionMemory()
    mem.add_turn("user", "hello")
    mem.clear()
    assert len(mem) == 0
    assert mem.get_history() == []


# --- LLM-backed multi-turn tests ---


@pytest.fixture(scope="module")
async def json_strategy_with_index(tmp_path_factory, live_openai_key):
    """Build a small NaiveVectorStrategy index for multi-turn tests."""
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    from kb_arena.models.document import Document, Section
    from kb_arena.strategies.naive_vector import NaiveVectorStrategy

    docs = [
        Document(
            id="json-mt-doc",
            source="test://json",
            corpus="python-stdlib",
            title="json module",
            sections=[
                Section(
                    id="json-loads-mt",
                    title="json.loads",
                    content=(
                        "json.loads(s) deserializes s to a Python object. "
                        "It raises json.JSONDecodeError if the input is invalid JSON. "
                        "Example: json.loads('{\"key\": 1}') returns {'key': 1}."
                    ),
                    heading_path=["json", "json.loads"],
                    level=2,
                ),
                Section(
                    id="json-dumps-mt",
                    title="json.dumps",
                    content=(
                        "json.dumps(obj) serializes obj to a JSON formatted string. "
                        "Use indent parameter for pretty-printing. "
                        "Example: json.dumps({'key': 1}) returns '{\"key\": 1}'."
                    ),
                    heading_path=["json", "json.dumps"],
                    level=2,
                ),
            ],
            raw_token_count=100,
        ),
        Document(
            id="ospath-mt-doc",
            source="test://os.path",
            corpus="python-stdlib",
            title="os.path module",
            sections=[
                Section(
                    id="ospath-join-mt",
                    title="os.path.join",
                    content=(
                        "os.path.join(path, *paths) joins path components intelligently. "
                        "It is used to construct file paths in a platform-independent way."
                    ),
                    heading_path=["os", "os.path", "os.path.join"],
                    level=3,
                ),
            ],
            raw_token_count=50,
        ),
    ]

    chroma = chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("mt_chroma")))
    ef = OpenAIEmbeddingFunction(api_key=live_openai_key, model_name="text-embedding-3-small")
    col = chroma.get_or_create_collection(
        name="naive_vector",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    strategy = NaiveVectorStrategy(chroma_client=chroma)
    strategy._collection = col
    await strategy.build_index(docs)
    return strategy


async def test_multi_turn_turn1_json_loads(json_strategy_with_index):
    result = await json_strategy_with_index.query("What is json.loads?")
    assert result.answer
    assert any(kw in result.answer.lower() for kw in ["deserializ", "json", "python"])


async def test_multi_turn_turn2_exceptions(json_strategy_with_index):
    """Second turn: exceptions question should retrieve JSONDecodeError info."""
    result = await json_strategy_with_index.query("What exceptions can json.loads raise?")
    assert result.answer
    # Should reference JSONDecodeError
    assert any(
        kw in result.answer for kw in ["JSONDecodeError", "ValueError", "exception", "invalid"]
    )


async def test_multi_turn_turn3_example(json_strategy_with_index):
    """Third turn: ask for example — should produce code-like response."""
    result = await json_strategy_with_index.query("Show me an example of json.loads")
    assert result.answer
    assert len(result.answer) > 10


async def test_multi_turn_turn4_topic_switch(json_strategy_with_index):
    """Turn 4: switch topic to os.path — should find the new context."""
    result = await json_strategy_with_index.query("What about os.path.join?")
    assert result.answer
    assert any(kw in result.answer.lower() for kw in ["os.path", "path", "join", "file"])


async def test_session_memory_10_turns_no_degradation(json_strategy_with_index):
    """10-turn conversation should not exceed memory limits."""
    mem = SessionMemory()
    questions = [
        "What is json.loads?",
        "What exceptions does it raise?",
        "Show me an example.",
        "What about json.dumps?",
        "How do I pretty-print JSON?",
        "What is json.JSONDecodeError?",
        "What does it inherit from?",
        "What about os.path.join?",
        "How is it different from pathlib?",
        "Summarize everything.",
    ]

    for i, q in enumerate(questions):
        result = await json_strategy_with_index.query(q)
        assert result.answer, f"Turn {i + 1} returned empty answer"
        mem.add_turn("user", q)
        mem.add_turn("assistant", result.answer)

    # Memory should not exceed MAX_TURNS * 2
    assert len(mem) <= MAX_TURNS * 2


async def test_session_memory_history_used_in_classify(live_llm_client):
    """History context changes classification for follow-up queries."""
    from kb_arena.chatbot.router import IntentRouter

    router = IntentRouter(llm=live_llm_client)
    history = [
        {"role": "user", "content": "Tell me about Python json module"},
        {"role": "assistant", "content": "The json module provides JSON encoding and decoding."},
    ]
    # "Compare it with pickle" — ambiguous without history context
    result = await router.classify("Compare it with pickle", history=history)
    assert result.value == "comparison"


async def test_conversation_all_5_intents(live_llm_client):
    """A conversation that naturally cycles through all 5 intent types."""
    from kb_arena.chatbot.router import IntentRouter, QueryIntent

    router = IntentRouter(llm=live_llm_client)
    intent_queries = [
        "What is json.loads?",  # factoid
        "Compare json.loads vs pickle.loads",  # comparison
        "What depends on json.JSONDecodeError?",  # relational
        "How do I configure JSON pretty-printing?",  # procedural
        "Tell me about the Python json module",  # exploratory
    ]
    expected_intents = [
        QueryIntent.FACTOID,
        QueryIntent.COMPARISON,
        QueryIntent.RELATIONAL,
        QueryIntent.PROCEDURAL,
        QueryIntent.EXPLORATORY,
    ]
    history = []
    for query, expected in zip(intent_queries, expected_intents):
        result = await router.classify(query, history=history)
        assert result == expected, (
            f"Query: {query!r} → expected {expected.value}, got {result.value}"
        )
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": f"Answer about {query}"})
