"""Integration tests for the FastAPI chatbot app using TestClient."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from kb_arena.strategies.base import AnswerResult

_UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _make_mock_strategy(
    name: str, answer: str = "mock answer", sources: list | None = None
) -> MagicMock:
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

    strategy.stream_calls = []

    async def _stream(question, history=None, corpus="all"):
        strategy.stream_calls.append((question, history, corpus))
        for word in answer.split():
            yield word + " "

    strategy.stream_answer = _stream
    return strategy


@pytest.fixture
def app_client():
    from kb_arena.chatbot.api import _rate_store, app
    from kb_arena.chatbot.auth import require_auth
    from kb_arena.settings import settings

    strategies = {
        "naive_vector": _make_mock_strategy("naive_vector", "Naive vector answer"),
        "contextual_vector": _make_mock_strategy("contextual_vector", "Contextual vector answer"),
        "qna_pairs": _make_mock_strategy("qna_pairs", "QnA pairs answer"),
        "knowledge_graph": _make_mock_strategy("knowledge_graph", "Knowledge graph answer"),
        "hybrid": _make_mock_strategy("hybrid", "Hybrid answer"),
    }

    _rate_store.clear()

    # CI runs without API keys, which makes the lifespan auto-enable demo_mode
    # and the require_auth dependency return 503 on every /chat call. The
    # integration tests model the behaviour of a fully-configured deployment,
    # so we (a) override auth to a no-op, (b) force demo_mode off for the
    # duration of the test session.
    prior_demo_mode = settings.demo_mode
    settings.demo_mode = False
    app.dependency_overrides[require_auth] = lambda: None

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            app.state.strategies = strategies
            app.state.neo4j = None
            yield client
    finally:
        app.dependency_overrides.pop(require_auth, None)
        settings.demo_mode = prior_demo_mode


# GET /health


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
    # /health now returns a structured neo4j object: {connected, uri, last_error}
    neo4j = data["neo4j"]
    assert isinstance(neo4j, dict)
    assert "connected" in neo4j


def test_health_lists_strategies(app_client):
    r = app_client.get("/health")
    data = r.json()
    assert "strategies" in data
    # Fixture mocks five strategies; the live registry has nine. Just assert presence.
    assert isinstance(data["strategies"], list)
    assert "hybrid" in data["strategies"]


def test_arena_match_passes_selected_corpus(app_client):
    arena = MagicMock()
    arena.create_match = AsyncMock(
        return_value=SimpleNamespace(
            id="match-1",
            question="What is the control?",
            answer_a="A",
            answer_b="B",
            latency_a_ms=1.0,
            latency_b_ms=2.0,
            sources_a=[],
            sources_b=[],
        )
    )
    app_client.app.state.arena = arena

    response = app_client.post(
        "/api/arena/match",
        json={"question": "What is the control?", "corpus": "nist"},
    )

    assert response.status_code == 200
    arena.create_match.assert_awaited_once_with("What is the control?", corpus="nist")


def test_debug_explain_passes_selected_corpus(app_client):
    from kb_arena.settings import settings

    strategy = app_client.app.state.strategies["naive_vector"]
    prior_debug = settings.debug
    settings.debug = True
    try:
        response = app_client.post(
            "/api/debug/explain",
            json={"query": "What is the control?", "strategy": "naive_vector", "corpus": "nist"},
        )
    finally:
        settings.debug = prior_debug

    assert response.status_code == 200
    strategy.query.assert_awaited_once_with("What is the control?", top_k=5, corpus="nist")


def test_demo_lifespan_skips_external_clients_and_disables_chroma_telemetry(monkeypatch):
    import os

    import chromadb
    import neo4j

    from kb_arena.chatbot.api import app
    from kb_arena.llm import client as llm_client_module
    from kb_arena.settings import settings

    captured: dict = {"llm_initialized": False}

    def fake_persistent_client(*args, **kwargs):
        captured.update(kwargs)
        captured["anonymized_telemetry"] = os.environ.get("ANONYMIZED_TELEMETRY")
        return MagicMock()

    def fail_if_neo4j_connects(*args, **kwargs):
        raise AssertionError("demo mode must not connect to Neo4j")

    def record_llm_initialization(*args, **kwargs):
        captured["llm_initialized"] = True
        return MagicMock()

    prior_demo_mode = settings.demo_mode
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(chromadb, "PersistentClient", fake_persistent_client)
    monkeypatch.setattr(neo4j.AsyncGraphDatabase, "driver", fail_if_neo4j_connects)
    monkeypatch.setattr(llm_client_module, "LLMClient", record_llm_initialization)

    try:
        with TestClient(app) as client:
            assert "qiss" in app.state.strategies
            ready = client.get("/ready")
            assert ready.status_code == 200
            assert ready.json() == {
                "ready": True,
                "checks": {"demo_mode": True, "neo4j_required": False, "llm_required": False},
            }
    finally:
        settings.demo_mode = prior_demo_mode

    assert captured["anonymized_telemetry"] == "False"
    assert captured["llm_initialized"] is False


def test_lifespan_accepts_generic_key_for_selected_generation_provider(monkeypatch):
    import chromadb

    from kb_arena.chatbot.api import app
    from kb_arena.llm import client as llm_client_module
    from kb_arena.settings import settings

    initialized = {"llm": False}

    def fake_llm(*args, **kwargs):
        initialized["llm"] = True
        return MagicMock()

    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "llm_api_key", "generic-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "embedding-only")
    monkeypatch.setattr(llm_client_module, "LLMClient", fake_llm)
    monkeypatch.setattr(chromadb, "PersistentClient", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(
        "neo4j.AsyncGraphDatabase.driver", MagicMock(side_effect=OSError("Neo4j unavailable"))
    )

    with TestClient(app):
        assert settings.demo_mode is False

    assert initialized["llm"] is True


def test_configured_llm_initialization_failure_stops_startup(monkeypatch):
    from kb_arena.chatbot.api import app
    from kb_arena.llm import client as llm_client_module
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "llm_api_key", "configured-key")
    monkeypatch.setattr(llm_client_module, "LLMClient", MagicMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"), TestClient(app):
        pass


def test_unknown_generation_provider_stops_startup(monkeypatch):
    from kb_arena.chatbot.api import app
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "llm_provider", "unknown")

    with pytest.raises(ValueError, match="Unknown KB_ARENA_LLM_PROVIDER"), TestClient(app):
        pass


# GET /strategies


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


def test_strategies_returns_full_catalog_with_runtime_status(app_client):
    data = app_client.get("/strategies").json()

    assert len(data["catalog"]) == 11
    by_name = {record["name"]: record for record in data["catalog"]}
    assert by_name["naive_vector"]["status"] == "loaded"
    assert by_name["bm25"]["status"] == "unavailable"
    assert by_name["qiss"]["experimental"] is True
    assert by_name["sqr"]["optional_extra"] == "quantum"
    assert by_name["sqr"]["default_benchmark"] is False


# POST /chat happy path


def test_chat_returns_200(app_client):
    r = app_client.post(
        "/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"}
    )
    assert r.status_code == 200


def test_chat_response_has_answer(app_client):
    r = app_client.post(
        "/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"}
    )
    data = r.json()
    assert "answer" in data
    assert data["answer"]


def test_chat_passes_selected_corpus_to_strategy(app_client):
    from kb_arena.chatbot.api import app

    response = app_client.post(
        "/chat",
        json={"query": "What is X?", "strategy": "naive_vector", "corpus": "nist"},
    )

    assert response.status_code == 200
    app.state.strategies["naive_vector"].query.assert_awaited_with(
        "What is X?", top_k=5, corpus="nist"
    )


def test_graph_data_uses_corpus_qualified_ids_for_aggregate_view(app_client):
    from kb_arena.chatbot.api import app

    node_result = AsyncMock()
    node_result.data.return_value = [
        {"id": "alpha::shared", "name": "Shared", "type": "Topic", "description": ""},
        {"id": "beta::shared", "name": "Shared", "type": "Topic", "description": ""},
    ]
    edge_result = AsyncMock()
    edge_result.data.return_value = [
        {"source": "alpha::shared", "type": "CONNECTS_TO", "target": "beta::shared"}
    ]
    session = AsyncMock()
    session.run.side_effect = [node_result, edge_result]
    context = AsyncMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = context
    app.state.neo4j = driver

    try:
        response = app_client.get("/api/graph/data?corpus=all")
    finally:
        app.state.neo4j = None

    assert response.status_code == 200
    assert [node["id"] for node in response.json()["nodes"]] == [
        "alpha::shared",
        "beta::shared",
    ]
    assert response.json()["edges"] == [
        {"source": "alpha::shared", "target": "beta::shared", "type": "CONNECTS_TO"}
    ]
    node_query = session.run.call_args_list[0].args[0]
    edge_query = session.run.call_args_list[1].args[0]
    assert "MATCH (n:KBArenaEntity)" in node_query
    assert "n.entity_id AS id" in node_query
    assert "MATCH (a:KBArenaEntity)-[r]->(b:KBArenaEntity)" in edge_query
    assert "a.entity_id AS source" in edge_query
    assert "b.entity_id AS target" in edge_query


def test_chat_response_has_strategy_used(app_client):
    r = app_client.post(
        "/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"}
    )
    data = r.json()
    assert data["strategy_used"] == "naive_vector"


def test_chat_response_has_sources(app_client):
    r = app_client.post(
        "/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"}
    )
    data = r.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_chat_response_has_latency_ms(app_client):
    r = app_client.post(
        "/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"}
    )
    data = r.json()
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], int | float)


def test_chat_response_has_tokens_used(app_client):
    r = app_client.post(
        "/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"}
    )
    data = r.json()
    assert "tokens_used" in data


def test_chat_response_has_cost_usd(app_client):
    r = app_client.post(
        "/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"}
    )
    data = r.json()
    assert "cost_usd" in data


def test_chat_default_strategy_is_hybrid(app_client):
    r = app_client.post("/chat", json={"query": "What does json.loads do?"})
    assert r.status_code == 200
    data = r.json()
    assert data["strategy_used"] == "hybrid"


# POST /chat for each strategy


@pytest.mark.parametrize(
    "strategy",
    [
        "naive_vector",
        "contextual_vector",
        "qna_pairs",
        "knowledge_graph",
        "hybrid",
    ],
)
def test_chat_each_strategy(app_client, strategy):
    r = app_client.post("/chat", json={"query": "What is X?", "strategy": strategy})
    assert r.status_code == 200
    data = r.json()
    assert data["strategy_used"] == strategy


# POST /chat error cases


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


# POST /chat history handling


def test_chat_with_empty_history(app_client):
    r = app_client.post(
        "/chat",
        json={
            "query": "What is X?",
            "strategy": "naive_vector",
            "history": [],
        },
    )
    assert r.status_code == 200


def test_chat_with_history(app_client):
    r = app_client.post(
        "/chat",
        json={
            "query": "Follow-up question.",
            "strategy": "naive_vector",
            "history": [
                {"role": "user", "content": "What is json.loads?"},
                {"role": "assistant", "content": "It parses JSON strings."},
            ],
        },
    )
    assert r.status_code == 200


def test_chat_with_extra_fields_ignored(app_client):
    r = app_client.post(
        "/chat",
        json={
            "query": "What is X?",
            "strategy": "naive_vector",
            "unknown_field": "ignored",
        },
    )
    assert r.status_code == 200


# POST /chat/stream SSE events


def test_chat_stream_returns_200(app_client):
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}
    ) as r:
        assert r.status_code == 200


def test_chat_stream_content_type_is_sse(app_client):
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}
    ) as r:
        ct = r.headers.get("content-type", "")
        assert "text/event-stream" in ct


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    current = {}
    for line in body.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:") :].strip()
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def test_chat_stream_has_message_id_event(app_client):
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}
    ) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    event_types = [e.get("event") for e in events]
    assert "message_id" in event_types


def test_chat_stream_message_id_is_uuid(app_client):
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}
    ) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    msg_id_event = next((e for e in events if e.get("event") == "message_id"), None)
    assert msg_id_event is not None
    data = json.loads(msg_id_event["data"])
    assert _UUID_PATTERN.match(data["id"])


def test_chat_stream_has_done_event(app_client):
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}
    ) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    event_types = [e.get("event") for e in events]
    assert "done" in event_types


def test_chat_stream_done_event_has_sources(app_client):
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}
    ) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    done_event = next((e for e in events if e.get("event") == "done"), None)
    assert done_event is not None
    data = json.loads(done_event["data"])
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_chat_stream_has_meta_event(app_client):
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}
    ) as r:
        body = r.read().decode()

    events = _parse_sse_events(body)
    event_types = [e.get("event") for e in events]
    assert "meta" in event_types


def test_chat_stream_meta_has_latency(app_client):
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}
    ) as r:
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
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "What is X?", "strategy": "naive_vector"}
    ) as r:
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
    with app_client.stream(
        "POST", "/chat/stream", json={"query": "X?", "strategy": "nonexistent"}
    ) as r:
        assert r.status_code in (400, 422)


# CORS headers


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


# Special input handling


def test_chat_unicode_query(app_client):
    r = app_client.post(
        "/chat",
        json={
            "query": "什么是 json.loads？ مرحبا",
            "strategy": "naive_vector",
        },
    )
    assert r.status_code == 200


def test_chat_special_characters(app_client):
    r = app_client.post(
        "/chat",
        json={
            "query": "What about <script> tags & \"quotes\" and 'apostrophes'?",
            "strategy": "naive_vector",
        },
    )
    assert r.status_code == 200


def test_chat_very_short_query(app_client):
    r = app_client.post("/chat", json={"query": "hi", "strategy": "naive_vector"})
    assert r.status_code == 200


def test_chat_very_long_query(app_client):
    long_query = "What does json do? " * 600  # ~10000+ chars
    r = app_client.post("/chat", json={"query": long_query, "strategy": "naive_vector"})
    assert r.status_code in (200, 400, 422)


# Concurrent requests


def test_concurrent_chat_requests(app_client):
    import concurrent.futures

    def make_request(n):
        return app_client.post(
            "/chat",
            json={
                "query": f"Question number {n}?",
                "strategy": "naive_vector",
            },
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(make_request, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    status_codes = [r.status_code for r in results]
    assert all(c in (200, 429) for c in status_codes)
    # At least some should succeed
    assert any(c == 200 for c in status_codes)


@pytest.mark.asyncio
async def test_completed_graph_build_queue_expires_without_stream_client(tmp_path, monkeypatch):
    import asyncio

    from kb_arena.chatbot import api
    from kb_arena.settings import settings

    processed = tmp_path / "sample" / "processed"
    processed.mkdir(parents=True)
    (processed / "documents.jsonl").write_text("{}\n")
    extraction_finished = asyncio.Event()

    async def fake_extraction(corpus: str, event_callback) -> None:
        assert corpus == "sample"
        await event_callback(
            {"type": "complete", "data": {"total_entities": 0, "total_relationships": 0}}
        )
        extraction_finished.set()

    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    monkeypatch.setattr("kb_arena.graph.extractor.run_extraction", fake_extraction)
    monkeypatch.setattr(api, "_GRAPH_BUILD_QUEUE_TTL_SECONDS", 0.01)

    response = await api.trigger_graph_build(api._GraphBuildRequest(corpus="sample"))
    build_id = response["build_id"]
    try:
        await asyncio.wait_for(extraction_finished.wait(), timeout=1)
        await asyncio.sleep(0.03)
        assert build_id not in api._graph_build_queues
    finally:
        api._graph_build_queues.pop(build_id, None)


@pytest.mark.asyncio
async def test_graph_build_rejects_requests_above_active_limit(tmp_path, monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from kb_arena.chatbot import api
    from kb_arena.settings import settings

    processed = tmp_path / "sample" / "processed"
    processed.mkdir(parents=True)
    (processed / "documents.jsonl").write_text("{}\n")
    release_extraction = asyncio.Event()

    async def fake_extraction(corpus: str, event_callback) -> None:
        await release_extraction.wait()

    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    monkeypatch.setattr("kb_arena.graph.extractor.run_extraction", fake_extraction)
    monkeypatch.setattr(api, "_GRAPH_BUILD_MAX_ACTIVE", 2)

    responses = [
        await api.trigger_graph_build(api._GraphBuildRequest(corpus="sample")) for _ in range(2)
    ]
    tasks = [api._graph_build_tasks[response["build_id"]] for response in responses]
    try:
        with pytest.raises(HTTPException) as exc_info:
            await api.trigger_graph_build(api._GraphBuildRequest(corpus="sample"))
        assert exc_info.value.status_code == 429
    finally:
        release_extraction.set()
        await asyncio.gather(*tasks)
        for response in responses:
            api._graph_build_queues.pop(response["build_id"], None)


@pytest.mark.asyncio
async def test_hung_graph_build_times_out_and_releases_active_slot(tmp_path, monkeypatch):
    import asyncio

    from kb_arena.chatbot import api
    from kb_arena.settings import settings

    processed = tmp_path / "sample" / "processed"
    processed.mkdir(parents=True)
    (processed / "documents.jsonl").write_text("{}\n")

    async def fake_extraction(corpus: str, event_callback) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    monkeypatch.setattr("kb_arena.graph.extractor.run_extraction", fake_extraction)
    monkeypatch.setattr(api, "_GRAPH_BUILD_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(api, "_GRAPH_BUILD_QUEUE_TTL_SECONDS", 60.0)

    response = await api.trigger_graph_build(api._GraphBuildRequest(corpus="sample"))
    build_id = response["build_id"]
    task = api._graph_build_tasks[build_id]
    try:
        await asyncio.wait_for(task, timeout=1)
        assert build_id not in api._graph_build_tasks

        queue = api._graph_build_queues[build_id]
        retained = []
        while not queue.empty():
            retained.append(queue.get_nowait())

        assert retained[-1] is None
        assert retained[-2]["type"] == "error"
        assert "timed out" in retained[-2]["data"]["message"]
    finally:
        api._graph_build_queues.pop(build_id, None)


@pytest.mark.asyncio
async def test_graph_build_queue_stays_bounded_without_stream_client(tmp_path, monkeypatch):
    import asyncio

    from kb_arena.chatbot import api
    from kb_arena.settings import settings

    processed = tmp_path / "sample" / "processed"
    processed.mkdir(parents=True)
    (processed / "documents.jsonl").write_text("{}\n")
    events_emitted = asyncio.Event()
    release_extraction = asyncio.Event()
    extraction_finished = asyncio.Event()

    async def fake_extraction(corpus: str, event_callback) -> None:
        assert corpus == "sample"
        for index in range(10):
            await event_callback(
                {
                    "type": "entity",
                    "data": {"id": str(index), "name": str(index), "nodeType": "Topic"},
                }
            )
        events_emitted.set()
        await release_extraction.wait()
        await event_callback(
            {"type": "complete", "data": {"total_entities": 10, "total_relationships": 0}}
        )
        extraction_finished.set()

    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    monkeypatch.setattr("kb_arena.graph.extractor.run_extraction", fake_extraction)
    monkeypatch.setattr(api, "_GRAPH_BUILD_QUEUE_MAX_EVENTS", 3)
    monkeypatch.setattr(api, "_GRAPH_BUILD_QUEUE_TTL_SECONDS", 60.0)

    response = await api.trigger_graph_build(api._GraphBuildRequest(corpus="sample"))
    build_id = response["build_id"]
    try:
        await asyncio.wait_for(events_emitted.wait(), timeout=1)
        queue = api._graph_build_queues[build_id]
        assert queue.qsize() == 3

        while not queue.empty():
            queue.get_nowait()

        release_extraction.set()
        await asyncio.wait_for(extraction_finished.wait(), timeout=1)
        await asyncio.sleep(0.01)
        assert queue.qsize() <= 3

        retained = []
        while not queue.empty():
            retained.append(queue.get_nowait())

        assert retained[-1] is None
        assert retained[-2]["type"] == "complete"
    finally:
        api._graph_build_queues.pop(build_id, None)


@pytest.mark.asyncio
async def test_graph_build_stream_rejects_a_second_subscriber():
    import asyncio

    from kb_arena.chatbot import api

    build_id = "single-subscriber"
    queue: asyncio.Queue = asyncio.Queue()
    api._graph_build_queues[build_id] = queue
    try:
        first = await api.graph_build_stream(build_id)
        second = await api.graph_build_stream(build_id)
        event = await anext(second.body_iterator)

        assert event["event"] == "error"
        assert "already has a subscriber" in event["data"]
        await second.body_iterator.aclose()
        await first.body_iterator.aclose()
    finally:
        api._graph_build_queues.pop(build_id, None)
        api._graph_build_subscribers.discard(build_id)


@pytest.mark.asyncio
async def test_graph_build_stream_can_reconnect_after_client_disconnect():
    import asyncio

    from kb_arena.chatbot import api

    build_id = "reconnect-stream"
    queue: asyncio.Queue = asyncio.Queue()
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    api._graph_build_queues[build_id] = queue
    api._graph_build_tasks[build_id] = task
    try:
        first = await api.graph_build_stream(build_id)
        queue.put_nowait({"type": "started", "data": {"total_sections": 2}})
        assert (await anext(first.body_iterator))["event"] == "started"
        await first.body_iterator.aclose()

        assert build_id in api._graph_build_queues
        assert build_id not in api._graph_build_subscribers

        second = await api.graph_build_stream(build_id)
        queue.put_nowait(
            {
                "type": "complete",
                "data": {"total_entities": 3, "total_relationships": 2},
            }
        )
        queue.put_nowait(None)
        assert (await anext(second.body_iterator))["event"] == "complete"
        with pytest.raises(StopAsyncIteration):
            await anext(second.body_iterator)
    finally:
        release.set()
        await task
        api._graph_build_tasks.pop(build_id, None)
        api._graph_build_queues.pop(build_id, None)
        api._graph_build_subscribers.discard(build_id)
