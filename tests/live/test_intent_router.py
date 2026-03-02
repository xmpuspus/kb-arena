"""Live tests for IntentRouter — keyword scan and LLM classification paths."""

from __future__ import annotations

import time

import pytest

from kb_arena.chatbot.router import IntentRouter, QueryIntent

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def router_no_llm():
    """Router without LLM — only keyword scan + regex fallback."""
    return IntentRouter(llm=None)


@pytest.fixture(scope="module")
def router_with_llm(live_llm_client):
    return IntentRouter(llm=live_llm_client)


# --- Keyword-stage tests (no LLM needed) ---


@pytest.mark.parametrize("query,expected", [
    ("Compare Deployment vs StatefulSet", QueryIntent.COMPARISON),
    ("What is the difference between Pods and Containers?", QueryIntent.COMPARISON),
    ("A vs B, which is better?", QueryIntent.COMPARISON),
    ("X versus Y", QueryIntent.COMPARISON),
    ("What depends on kube-proxy?", QueryIntent.RELATIONAL),
    ("What does json.loads require?", QueryIntent.RELATIONAL),
    ("How does X affect Y?", QueryIntent.RELATIONAL),
    ("How do I configure TLS for an Ingress?", QueryIntent.PROCEDURAL),
    ("How can I install Python?", QueryIntent.PROCEDURAL),
    ("Steps to configure RBAC", QueryIntent.PROCEDURAL),
    ("Setup a new project", QueryIntent.PROCEDURAL),
    ("Implement a retry mechanism", QueryIntent.PROCEDURAL),
    ("Create a Kubernetes cluster", QueryIntent.PROCEDURAL),
])
async def test_keyword_stage_catches_pattern(router_no_llm, query, expected):
    result = await router_no_llm.classify(query)
    assert result == expected, f"Query: {query!r} → expected {expected}, got {result}"


async def test_keyword_stage_is_fast(router_no_llm):
    """Keyword-matched queries should classify in under 5ms (no LLM call)."""
    t0 = time.perf_counter()
    for _ in range(20):
        await router_no_llm.classify("Compare X vs Y")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    # 20 keyword-scan classifications should take well under 200ms total
    assert elapsed_ms < 200, f"Keyword scan too slow: {elapsed_ms:.1f}ms for 20 calls"


# --- LLM-stage tests ---


@pytest.mark.parametrize("query,expected", [
    ("What is a Pod?", QueryIntent.FACTOID),
    ("Tell me about Python's asyncio", QueryIntent.EXPLORATORY),
    ("What is json.loads?", QueryIntent.FACTOID),
    ("Explain the GIL", QueryIntent.EXPLORATORY),
    ("What does the os module do?", QueryIntent.FACTOID),
])
async def test_llm_stage_classifies_correctly(router_with_llm, query, expected):
    result = await router_with_llm.classify(query)
    assert result == expected, f"Query: {query!r} → expected {expected}, got {result}"


async def test_llm_stage_latency(router_with_llm):
    """LLM classification should complete in under 5 seconds."""
    # Use a query that doesn't match keyword patterns
    t0 = time.perf_counter()
    await router_with_llm.classify("Tell me about asyncio event loops")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 5000, f"LLM classification too slow: {elapsed_ms:.0f}ms"


# --- Ambiguous queries ---


async def test_ambiguous_query_returns_valid_intent(router_with_llm):
    result = await router_with_llm.classify("json")
    assert isinstance(result, QueryIntent)


async def test_query_with_history_context(router_with_llm):
    history = [
        {"role": "user", "content": "Tell me about Python dictionaries"},
        {"role": "assistant", "content": "Dictionaries are key-value stores..."},
    ]
    result = await router_with_llm.classify("What about sets?", history=history)
    assert isinstance(result, QueryIntent)


async def test_empty_string_fallback(router_no_llm):
    result = await router_no_llm.classify("")
    # Should not crash — fallback returns FACTOID
    assert result == QueryIntent.FACTOID


async def test_single_word_query(router_with_llm):
    result = await router_with_llm.classify("json")
    assert isinstance(result, QueryIntent)


async def test_very_long_query(router_with_llm):
    long_query = "What is json.loads " + "and also " * 50 + "how does it work?"
    result = await router_with_llm.classify(long_query)
    assert isinstance(result, QueryIntent)


# --- All 5 intents reachable ---


async def test_all_intents_reachable(router_with_llm):
    queries_and_expected = [
        ("What is json.loads?", QueryIntent.FACTOID),
        ("Compare list vs tuple", QueryIntent.COMPARISON),
        ("What depends on os.path?", QueryIntent.RELATIONAL),
        ("How do I read a CSV?", QueryIntent.PROCEDURAL),
        ("Tell me about Python asyncio", QueryIntent.EXPLORATORY),
    ]
    seen_intents = set()
    for query, expected in queries_and_expected:
        result = await router_with_llm.classify(query)
        seen_intents.add(result)

    # All 5 intents should be reachable via the router
    assert len(seen_intents) == 5, f"Only {len(seen_intents)} of 5 intents reached: {seen_intents}"


# --- Edge cases ---


async def test_comparison_keyword_fast_path(router_no_llm):
    """'vs' keyword should not hit the LLM at all."""
    t0 = time.perf_counter()
    result = await router_no_llm.classify("json vs pickle performance")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert result == QueryIntent.COMPARISON
    assert elapsed_ms < 50, f"Should be sub-50ms keyword scan, got {elapsed_ms:.1f}ms"


async def test_procedural_keyword_fast_path(router_no_llm):
    t0 = time.perf_counter()
    result = await router_no_llm.classify("How do I install ChromaDB?")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert result == QueryIntent.PROCEDURAL
    assert elapsed_ms < 50


async def test_router_no_llm_returns_fallback_for_what(router_no_llm):
    # "What is X?" has no keyword pattern — falls through to regex fallback
    result = await router_no_llm.classify("What is json.loads?")
    assert result == QueryIntent.FACTOID  # regex fallback catches "what"
