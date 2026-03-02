"""Integration tests for the FastAPI chatbot app using TestClient."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from kb_arena.models.graph import GraphContext
from kb_arena.strategies.base import AnswerResult


def _make_mock_strategy(name: str, answer: str = "mock answer", sources: list | None = None) -> MagicMock:
    strategy = MagicMock()
    strategy.name = name
    strategy.last_sources = sources or ["doc1"]
    strategy.last_graph_context = None
    strategy.last_latency_ms = 50.0
    strategy.last_tokens_used = 100
    strategy.last_cost_usd = 0.001

    result = AnswerResult(
        answer=answer,
        sources=sources or ["doc1"],
        strategy=name,
        latency_ms=50.0,
        tokens_used=100,
        cost_usd=0.001,
    )
    strategy.query = AsyncMock(return_value=result)

    async def _stream(question, history=None):
        for word in answer.split():
            yield word + " "
    strategy.stream_answer = _stream
    return strategy


@pytest.fixture
def app_client():
    from kb_arena.chatbot.api import app

    strategies = {
        "naive_vector": _make_mock_strategy("naive_vector", "Naive vector answer"),
        "contextual_vector": _make_mock_strategy("contextual_vector", "Contextual vector answer"),
        "qna_pairs": _make_mock_strategy("qna_pairs", "QnA pairs answer"),
        "knowledge_graph": _make_mock_strategy("knowledge_graph", "Knowledge graph answer"),
        "hybrid": _make_mock_strategy("hybrid", "Hybrid answer"),
    }

    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.strategies = strategies
        app.state.neo4j = None
        yield client


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200


def test_health_has_status_ok(app_client):
    r = app_client.get("/health")
    data = r.json()
    assert data["status"] == "ok"


def test_health_reports_neo4j_unavailable(app_client):
    r = app_client.get("/health")
    data = r.json()
    assert "unavailable" in data["neo4j"] or data["neo4j"] == "connected"


def test_health_lists_strategies(app_client):
    r = app_client.get("/health")
    data = r.json()
    assert "strategies" in data
    assert len(data["strategies"]) == 5


# ---------------------------------------------------------------------------
# GET /strategies
# ---------------------------------------------------------------------------

def test_strategies_returns_200(app_client):
    r = app_client.get("/strategies")
    assert r.status_code == 200


def test_strategies_returns_all_five(app_client):
    r = app_client.get("/strategies")
    data = r.json()
    assert "strategies" in data
    names = set(data["strategies"])
    assert "naive_vector" in names
    assert "contextual_vector" in names
    assert "qna_pairs" in names
    assert "knowledge_graph" in names
    assert "hybrid" in names


# ---------------------------------------------------------------------------
# POST /chat — happy path
# ---------------------------------------------------------------------------

def test_chat_returns_200(app_client):
    r = app_client.post("/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"})
    assert r.status_code == 200


def test_chat_response_has_answer(app_client):
    r = app_client.post("/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"})
    data = r.json()
    assert "answer" in data
    assert data["answer"]


def test_chat_response_has_strategy_used(app_client):
    r = app_client.post("/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"})
    data = r.json()
    assert data["strategy_used"] == "naive_vector"


def test_chat_response_has_sources(app_client):
    r = app_client.post("/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"})
    data = r.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_chat_response_has_latency_ms(app_client):
    r = app_client.post("/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"})
    data = r.json()
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (int, float))


def test_chat_response_has_tokens_used(app_client):
    r = app_client.post("/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"})
    data = r.json()
    assert "tokens_used" in data


def test_chat_response_has_cost_usd(app_client):
    r = app_client.post("/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"})
    data = r.json()
    assert "cost_usd" in data


def test_chat_default_strategy_is_hybrid(app_client):
    r = app_client.post("/chat", json={"query": "What does json.loads do?"})
    assert r.status_code == 200
    data = r.json()
    assert data["strategy_used"] == "hybrid"


# ---------------------------------------------------------------------------
# POST /chat — each strategy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", [
    "naive_vector",
    "contextual_vector",
    "qna_pairs",
    "knowledge_graph",
    "hybrid",
])
def test_chat_each_strategy(app_client, strategy):
    r = app_client.post("/chat", json={"query": "What is X?", "strategy": strategy})
    assert r.status_code == 200
    data = r.json()
    assert data["strategy_used"] == strategy


# ---------------------------------------------------------------------------
# POST /chat — error cases
# ---------------------------------------------------------------------------

def test_chat_invalid_strategy_returns_error(app_client):
    r = app_client.post("/chat", json={"query": "What is X?", "strategy": "nonexistent"})
    assert r.status_code in (400, 422)
    # If 400, should have error envelope
    if r.status_code == 400:
        data = r.json()
        assert "detail" in data


def test_chat_missing_query_returns_422(app_client):
    r = app_client.post("/chat", json={"strategy": "naive_vector"})
    assert r.status_code == 422


def test_chat_empty_body_returns_422(app_client):
    r = app_client.post("/chat", json={})
    assert r.status_code == 422


def test_chat_wrong_content_type_returns_422(app_client):
    r = app_client.post("/chat", content="not json", headers={"Content-Type": "text/plain"})
    assert r.status_code in (400, 415, 422)


# ---------------------------------------------------------------------------
# POST /chat — history handling
# ---------------------------------------------------------------------------

def test_chat_with_empty_history(app_client):
    r = app_client.post("/chat", json={
        "query": "What is X?",
        "strategy": "naive_vector",
        "history": [],
    })
    assert r.status_code == 200


def test_chat_with_history(app_client):
    r = app_client.post("/chat", json={
        "query": "Follow-up question.",
        "strategy": "naive_vector",
        "history": [
            {"role": "user", "content": "What is json.loads?"},
            {"role": "assistant", "content": "It parses JSON strings."},
        ],
    })
    assert r.status_code == 200


def test_chat_with_extra_fields_ignored(app_client):
    r = app_client.post("/chat", json={
        "query": "What is X?",
        "strategy": "naive_vector",
        "unknown_field": "ignored",
    })
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /chat/stream — SSE events
# ---------------------------------------------------------------------------

def test_chat_stream_returns_200(app_client):
    with app_client.stream("POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}) as r:
        assert r.status_code == 200


def test_chat_stream_content_type_is_sse(app_client):
    with app_client.stream("POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}) as r:
        ct = r.headers.get("content-type", "")
        assert "text/event-stream" in ct


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    current = {}
    for line in body.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:"):].strip()
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def test_chat_stream_has_message_id_event(app_client):
    with app_client.stream("POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    event_types = [e.get("event") for e in events]
    assert "message_id" in event_types


def test_chat_stream_message_id_is_uuid(app_client):
    import re
    with app_client.stream("POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    msg_id_event = next((e for e in events if e.get("event") == "message_id"), None)
    assert msg_id_event is not None
    data = json.loads(msg_id_event["data"])
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    assert uuid_pattern.match(data["id"])


def test_chat_stream_has_done_event(app_client):
    with app_client.stream("POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    event_types = [e.get("event") for e in events]
    assert "done" in event_types


def test_chat_stream_done_event_has_sources(app_client):
    with app_client.stream("POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    done_event = next((e for e in events if e.get("event") == "done"), None)
    assert done_event is not None
    data = json.loads(done_event["data"])
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_chat_stream_has_meta_event(app_client):
    with app_client.stream("POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    event_types = [e.get("event") for e in events]
    assert "meta" in event_types


def test_chat_stream_meta_has_latency(app_client):
    with app_client.stream("POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    meta_event = next((e for e in events if e.get("event") == "meta"), None)
    assert meta_event is not None
    data = json.loads(meta_event["data"])
    assert "latency_ms" in data
    assert "tokens_used" in data
    assert "cost_usd" in data


def test_chat_stream_event_order(app_client):
    """message_id must come before token events, done comes before meta."""
    with app_client.stream("POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    event_types = [e.get("event") for e in events]

    assert event_types[0] == "message_id"
    assert "done" in event_types
    assert "meta" in event_types

    done_idx = event_types.index("done")
    meta_idx = event_types.index("meta")
    assert done_idx < meta_idx


def test_chat_stream_invalid_strategy_returns_error(app_client):
    with app_client.stream("POST", "/chat/stream", json={"query": "X?", "strategy": "nonexistent"}) as r:
        assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

def test_cors_headers_present(app_client):
    r = app_client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    # FastAPI returns 200 for preflight when origin is allowed
    assert r.status_code in (200, 204)


# ---------------------------------------------------------------------------
# Special input handling
# ---------------------------------------------------------------------------

def test_chat_unicode_query(app_client):
    r = app_client.post("/chat", json={
        "query": "什么是 json.loads？ مرحبا 🐍",
        "strategy": "naive_vector",
    })
    assert r.status_code == 200


def test_chat_special_characters(app_client):
    r = app_client.post("/chat", json={
        "query": 'What about <script> tags & "quotes" and \'apostrophes\'?',
        "strategy": "naive_vector",
    })
    assert r.status_code == 200


def test_chat_very_short_query(app_client):
    r = app_client.post("/chat", json={"query": "hi", "strategy": "naive_vector"})
    assert r.status_code == 200


def test_chat_very_long_query(app_client):
    long_query = "What does json do? " * 600  # ~10000+ chars
    r = app_client.post("/chat", json={"query": long_query, "strategy": "naive_vector"})
    assert r.status_code in (200, 400, 422)


# ---------------------------------------------------------------------------
# Concurrent requests
# ---------------------------------------------------------------------------

def test_concurrent_chat_requests(app_client):
    import concurrent.futures

    def make_request(n):
        return app_client.post("/chat", json={
            "query": f"Question number {n}?",
            "strategy": "naive_vector",
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(make_request, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    status_codes = [r.status_code for r in results]
    assert all(c in (200, 429) for c in status_codes)
    # At least some should succeed
    assert any(c == 200 for c in status_codes)
