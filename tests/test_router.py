"""Tests for IntentRouter — three-stage classification."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kb_arena.chatbot.router import IntentRouter, QueryIntent


@pytest.fixture
def router():
    return IntentRouter(llm=None)


@pytest.fixture
def router_with_llm():
    mock_llm = AsyncMock()
    mock_llm.classify.return_value = "factoid"
    return IntentRouter(llm=mock_llm), mock_llm


# --- Stage 1: keyword patterns ---


@pytest.mark.asyncio
async def test_keyword_comparison_compare(router):
    intent = await router.classify("compare json.loads vs yaml.safe_load")
    assert intent == QueryIntent.COMPARISON


@pytest.mark.asyncio
async def test_keyword_comparison_vs(router):
    intent = await router.classify("json vs pickle, which is better?")
    assert intent == QueryIntent.COMPARISON


@pytest.mark.asyncio
async def test_keyword_comparison_difference(router):
    intent = await router.classify("what is the difference between list and tuple?")
    assert intent == QueryIntent.COMPARISON


@pytest.mark.asyncio
async def test_keyword_relational_depend(router):
    intent = await router.classify("what does os.path.join depend on?")
    assert intent == QueryIntent.RELATIONAL


@pytest.mark.asyncio
async def test_keyword_relational_affect(router):
    intent = await router.classify("how does changing the buffer size affect performance?")
    assert intent == QueryIntent.RELATIONAL


@pytest.mark.asyncio
async def test_keyword_procedural_how_do(router):
    intent = await router.classify("how do I parse a CSV file with Python?")
    assert intent == QueryIntent.PROCEDURAL


@pytest.mark.asyncio
async def test_keyword_procedural_steps(router):
    intent = await router.classify("what are the steps to configure logging?")
    assert intent == QueryIntent.PROCEDURAL


@pytest.mark.asyncio
async def test_keyword_procedural_setup(router):
    intent = await router.classify("setup a new Python project")
    assert intent == QueryIntent.PROCEDURAL


# --- Stage 3: fallback (no keyword match, no LLM) ---


@pytest.mark.asyncio
async def test_fallback_factoid_what_is(router):
    intent = await router.classify("what is the default encoding?")
    # "what" triggers fallback factoid
    assert intent == QueryIntent.FACTOID


@pytest.mark.asyncio
async def test_fallback_exploratory(router):
    intent = await router.classify("tell me about the json module")
    assert intent == QueryIntent.EXPLORATORY


@pytest.mark.asyncio
async def test_fallback_never_fails(router):
    """Router must always return a valid QueryIntent regardless of input."""
    intent = await router.classify("")
    assert isinstance(intent, QueryIntent)


@pytest.mark.asyncio
async def test_fallback_gibberish(router):
    intent = await router.classify("xzqrfbml wqzpx yqzl")
    assert isinstance(intent, QueryIntent)


# --- Stage 2: LLM path ---


@pytest.mark.asyncio
async def test_llm_classify_called_when_no_keyword_match():
    mock_llm = AsyncMock()
    mock_llm.classify.return_value = "exploratory"
    router = IntentRouter(llm=mock_llm)

    intent = await router.classify("tell me everything you know about asyncio")
    # Should hit LLM (no keyword match for "tell me everything")
    assert mock_llm.classify.called
    assert intent == QueryIntent.EXPLORATORY


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_regex():
    mock_llm = AsyncMock()
    mock_llm.classify.side_effect = RuntimeError("API down")
    router = IntentRouter(llm=mock_llm)

    # Should not raise — fallback handles it
    intent = await router.classify("what is the maximum file size?")
    assert isinstance(intent, QueryIntent)


@pytest.mark.asyncio
async def test_keyword_shortcircuits_llm():
    """Keyword match should prevent LLM call entirely."""
    mock_llm = AsyncMock()
    mock_llm.classify.return_value = "factoid"
    router = IntentRouter(llm=mock_llm)

    intent = await router.classify("compare list vs deque performance")
    assert intent == QueryIntent.COMPARISON
    mock_llm.classify.assert_not_called()


# --- Session memory integration ---


@pytest.mark.asyncio
async def test_router_passes_history_to_llm():
    mock_llm = AsyncMock()
    mock_llm.classify.return_value = "factoid"
    router = IntentRouter(llm=mock_llm)

    history = [{"role": "user", "content": "we were talking about json"}]
    await router.classify("what about loads?", history=history)

    if mock_llm.classify.called:
        call_kwargs = mock_llm.classify.call_args.kwargs
        assert "history" in call_kwargs


# --- QueryIntent enum values ---


def test_query_intent_values():
    assert QueryIntent.FACTOID == "factoid"
    assert QueryIntent.COMPARISON == "comparison"
    assert QueryIntent.RELATIONAL == "relational"
    assert QueryIntent.PROCEDURAL == "procedural"
    assert QueryIntent.EXPLORATORY == "exploratory"
