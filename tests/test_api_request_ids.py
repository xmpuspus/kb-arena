"""Every API error path logs with a request id, and the id reaches the client."""

from __future__ import annotations

import json
import logging
import re
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from kb_arena.strategies.base import AnswerResult

_ID = re.compile(r"^[A-Za-z0-9_.:-]{8,64}$")


def _strategy(name: str, answer: str = "answer") -> MagicMock:
    strategy = MagicMock()
    strategy.name = name
    result = AnswerResult(answer=answer, sources=["doc1"], strategy=name, latency_ms=1.0)
    strategy.query = AsyncMock(return_value=result)

    async def _stream(question, history=None, corpus="all"):
        for word in answer.split():
            yield word + " "

    strategy.stream_answer = _stream
    return strategy


@pytest.fixture
def client():
    from kb_arena.chatbot.api import _rate_store, app
    from kb_arena.chatbot.auth import require_auth
    from kb_arena.settings import settings

    _rate_store.clear()
    prior_demo = settings.demo_mode
    settings.demo_mode = False
    app.dependency_overrides[require_auth] = lambda: None
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.strategies = {"naive_vector": _strategy("naive_vector")}
            app.state.neo4j = None
            yield c
    finally:
        app.dependency_overrides.pop(require_auth, None)
        settings.demo_mode = prior_demo


def test_every_response_carries_a_request_id(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert _ID.match(r.headers["X-Request-ID"])


def test_a_browser_client_can_read_the_request_id_header(client):
    # CORS hides response headers from JS unless the server lists them in
    # Access-Control-Expose-Headers, so a cross-origin caller could not
    # otherwise read X-Request-ID even though the header is on the wire.
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert "X-Request-ID" in r.headers["access-control-expose-headers"]


def test_a_client_request_id_is_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
    assert r.headers["X-Request-ID"] == "trace-abc-123"


def test_an_invalid_client_request_id_is_replaced(client):
    r = client.get("/health", headers={"X-Request-ID": "bad id with spaces " + "x" * 100})
    assert r.headers["X-Request-ID"] != "bad id with spaces " + "x" * 100
    assert _ID.match(r.headers["X-Request-ID"])


def test_unhandled_exception_is_logged_with_the_request_id(client, caplog):
    # /chat has no try/except of its own, so a strategy failure reaches the
    # global handler exactly like any other unguarded route would.
    client.app.state.strategies["naive_vector"].query = AsyncMock(
        side_effect=RuntimeError("kaboom")
    )
    with caplog.at_level(logging.ERROR, logger="kb_arena.chatbot.api"):
        r = client.post(
            "/chat",
            json={"query": "q", "strategy": "naive_vector"},
            headers={"X-Request-ID": "boom-request-1"},
        )

    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["request_id"] == "boom-request-1"
    assert "boom-request-1" in caplog.text
    assert "kaboom" in caplog.text
    assert "kaboom" not in body["error"]["message"]


def test_stream_error_is_logged_and_carries_the_request_id(client, caplog):
    from kb_arena.chatbot.api import app

    async def failing_stream(question, history=None, corpus="all"):
        raise RuntimeError("stream broke")
        yield  # pragma: no cover

    app.state.strategies["naive_vector"].stream_answer = failing_stream
    with caplog.at_level(logging.ERROR, logger="kb_arena.chatbot.api"):
        with client.stream(
            "POST",
            "/chat/stream",
            json={"query": "q", "strategy": "naive_vector"},
            headers={"X-Request-ID": "stream-request-1"},
        ) as r:
            text = "".join(r.iter_text())

    assert "event: error" in text
    data_lines = [line for line in text.splitlines() if line.startswith("data:")]
    error_line = next(line for line in data_lines if "stream_error" in line)
    payload = json.loads(error_line[5:])
    assert payload["request_id"] == "stream-request-1"
    assert "stream broke" in caplog.text
    assert "stream broke" not in payload["message"]


def test_arena_match_failure_is_logged(client, caplog):
    from kb_arena.chatbot.api import app

    arena = MagicMock()
    arena.create_match = AsyncMock(side_effect=RuntimeError("arena exploded"))
    app.state.arena = arena
    with caplog.at_level(logging.ERROR, logger="kb_arena.chatbot.api"):
        r = client.post(
            "/api/arena/match",
            json={"question": "q"},
            headers={"X-Request-ID": "arena-request-1"},
        )

    assert r.status_code == 500
    assert r.json()["error"]["code"] == "match_failed"
    assert r.json()["error"]["request_id"] == "arena-request-1"
    assert "arena exploded" in caplog.text


def test_debug_explain_failures_are_logged(client, caplog, monkeypatch):
    from kb_arena.chatbot.api import app
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "debug", True)
    router = MagicMock()
    router.classify = AsyncMock(side_effect=RuntimeError("classifier down"))
    app.state.router = router
    app.state.strategies["naive_vector"].query = AsyncMock(side_effect=RuntimeError("query down"))
    with caplog.at_level(logging.WARNING, logger="kb_arena.chatbot.api"):
        r = client.post(
            "/api/debug/explain",
            json={"query": "q", "strategy": "naive_vector"},
            headers={"X-Request-ID": "explain-request-1"},
        )

    assert r.status_code == 200
    assert "classifier down" in caplog.text
    assert "query down" in caplog.text
    assert "explain-request-1" in caplog.text


def test_api_docs_are_closed_by_default(client):
    # The test process runs with debug off and api_docs_enabled unset, so the
    # docs routes never got registered on the app this client wraps.
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404


def test_api_docs_urls_resolve_from_the_docs_enabled_setting():
    from kb_arena.chatbot.api import _docs_urls

    assert _docs_urls(True) == ("/docs", "/redoc", "/openapi.json")
    assert _docs_urls(False) == (None, None, None)


def test_docs_enabled_follows_debug_when_unset():
    from kb_arena.chatbot.api import _resolve_docs_enabled

    assert _resolve_docs_enabled(None, debug=False) is False
    assert _resolve_docs_enabled(None, debug=True) is True


def test_an_explicit_docs_setting_overrides_debug():
    from kb_arena.chatbot.api import _resolve_docs_enabled

    assert _resolve_docs_enabled(True, debug=False) is True
    assert _resolve_docs_enabled(False, debug=True) is False
