"""Behavioral integration tests for session management, cost tracking, rate limiting,
CORS settings, corpus validation, and session TTL settings."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.llm.client import LLMResponse

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def test_session_store_get_creates_new_session():
    from kb_arena.chatbot.session import SessionStore

    store = SessionStore()
    session = store.get("abc123")
    assert session is not None
    assert len(store) == 1


def test_session_store_get_returns_existing_session():
    from kb_arena.chatbot.session import SessionStore

    store = SessionStore()
    first = store.get("abc123")
    second = store.get("abc123")
    assert first is second
    assert len(store) == 1


def test_session_store_cleanup_evicts_expired():
    from kb_arena.chatbot.session import SessionMemory, SessionStore

    # TTL of 0 means every session is immediately expired
    store = SessionStore(ttl_minutes=0)
    session = SessionMemory()
    session.last_accessed = time.time() - 1  # already in the past
    store._sessions["stale"] = session

    evicted = store.cleanup()
    assert evicted == 1
    assert len(store) == 0


def test_session_store_cleanup_keeps_active():
    from kb_arena.chatbot.session import SessionStore

    store = SessionStore(ttl_minutes=30)
    store.get("active")  # creates and touches the session

    evicted = store.cleanup()
    assert evicted == 0
    assert len(store) == 1


def test_session_memory_last_accessed_updated_on_add_turn():
    from kb_arena.chatbot.session import SessionMemory

    mem = SessionMemory()
    before = mem.last_accessed
    time.sleep(0.01)
    mem.add_turn("user", "hello")
    assert mem.last_accessed > before


def test_session_memory_last_accessed_updated_on_get_history():
    from kb_arena.chatbot.session import SessionMemory

    mem = SessionMemory()
    before = mem.last_accessed
    time.sleep(0.01)
    mem.get_history()
    assert mem.last_accessed > before


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_usd_propagated_through_answer_result():
    from kb_arena.strategies.naive_vector import NaiveVectorStrategy

    mock_chroma = MagicMock()
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["doc1-s1"]],
        "documents": [["chunk text"]],
        "metadatas": [[{"source_id": "doc1"}]],
    }
    mock_chroma.get_or_create_collection.return_value = collection

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        text="answer", input_tokens=100, output_tokens=50, cost_usd=0.0045
    )

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    strategy._llm = mock_llm

    result = await strategy.query("What is Lambda?")
    assert result.cost_usd == pytest.approx(0.0045)


@pytest.mark.asyncio
async def test_cost_usd_nonzero_when_llm_returns_cost():
    from kb_arena.strategies.naive_vector import NaiveVectorStrategy

    mock_chroma = MagicMock()
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["doc1-s1"]],
        "documents": [["chunk text"]],
        "metadatas": [[{"source_id": "doc1"}]],
    }
    mock_chroma.get_or_create_collection.return_value = collection

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = LLMResponse(
        text="answer", input_tokens=100, output_tokens=50, cost_usd=0.0045
    )

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    strategy._llm = mock_llm

    result = await strategy.query("What is Lambda?")
    assert result.cost_usd > 0


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def test_rate_limiter_allows_within_limit():
    from kb_arena.chatbot.api import _check_rate_limit, _rate_store

    ip = "test-ip-allow-1"
    _rate_store.pop(ip, None)

    allowed = _check_rate_limit(ip)
    assert allowed is True


def test_rate_limiter_blocks_after_60_requests():
    from kb_arena.chatbot.api import RATE_LIMIT_RPM, _check_rate_limit, _rate_store

    ip = "test-ip-block-1"
    _rate_store.pop(ip, None)

    for _ in range(RATE_LIMIT_RPM):
        _check_rate_limit(ip)

    blocked = _check_rate_limit(ip)
    assert blocked is False


def test_rate_limiter_resets_after_window_expires():
    from kb_arena.chatbot.api import RATE_LIMIT_RPM, _check_rate_limit, _rate_store

    ip = "test-ip-reset-1"
    # Simulate 60 calls that happened 61 seconds ago (outside the 60s window)
    old_time = time.time() - 61
    _rate_store[ip] = [old_time] * RATE_LIMIT_RPM

    # Should be allowed because all prior calls are outside the window
    allowed = _check_rate_limit(ip)
    assert allowed is True


# ---------------------------------------------------------------------------
# CORS settings
# ---------------------------------------------------------------------------


def test_settings_cors_origins_defaults_to_empty_list():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.cors_origins == []


def test_settings_cors_origins_set_via_env(monkeypatch):
    monkeypatch.setenv("KB_ARENA_CORS_ORIGINS", '["https://myapp.example.com"]')
    from kb_arena.settings import Settings

    s = Settings()
    assert "https://myapp.example.com" in s.cors_origins


# ---------------------------------------------------------------------------
# Corpus validation
# ---------------------------------------------------------------------------


def test_graph_build_request_rejects_dotdot():
    import pytest
    from pydantic import ValidationError

    from kb_arena.chatbot.api import _GraphBuildRequest

    with pytest.raises(ValidationError):
        _GraphBuildRequest(corpus="../etc")


def test_graph_build_request_rejects_slash():
    import pytest
    from pydantic import ValidationError

    from kb_arena.chatbot.api import _GraphBuildRequest

    with pytest.raises(ValidationError):
        _GraphBuildRequest(corpus="aws/compute")


def test_graph_build_request_accepts_valid_corpus():
    from kb_arena.chatbot.api import _GraphBuildRequest

    req = _GraphBuildRequest(corpus="aws-compute")
    assert req.corpus == "aws-compute"


# ---------------------------------------------------------------------------
# Session TTL settings
# ---------------------------------------------------------------------------


def test_settings_session_ttl_minutes_defaults_to_30():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.session_ttl_minutes == 30
