"""Fixtures for API tests — mock client and live client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from kb_arena.models.graph import GraphContext
from kb_arena.strategies.base import AnswerResult


def _make_strategy(name: str, answer: str = "A helpful answer.", sources: list | None = None):
    strategy = MagicMock()
    strategy.name = name
    strategy.last_sources = sources or [f"doc-{name}"]
    strategy.last_graph_context = None
    strategy.last_latency_ms = 42.0
    strategy.last_tokens_used = 80
    strategy.last_cost_usd = 0.0008

    result = AnswerResult(
        answer=answer,
        sources=sources or [f"doc-{name}"],
        strategy=name,
        latency_ms=42.0,
        tokens_used=80,
        cost_usd=0.0008,
    )
    strategy.query = AsyncMock(return_value=result)

    async def _stream(question, history=None):
        for word in answer.split():
            yield word + " "

    strategy.stream_answer = _stream
    return strategy


def _make_graph_strategy(name: str = "knowledge_graph") -> MagicMock:
    strategy = _make_strategy(name, "Graph-based answer with context.")
    strategy.last_graph_context = GraphContext(
        nodes=[{"fqn": "json", "label": "Module"}, {"fqn": "json.loads", "label": "Function"}],
        edges=[{"src": "json", "dst": "json.loads", "rel": "CONTAINS"}],
        query_path=["json", "json.loads"],
        cypher_used="MATCH (n)-[:CONTAINS]->(m) RETURN n, m",
    )
    result = AnswerResult(
        answer="Graph-based answer with context.",
        sources=["json-docs"],
        strategy=name,
        latency_ms=120.0,
        tokens_used=200,
        cost_usd=0.002,
        graph_context=strategy.last_graph_context,
    )
    strategy.query = AsyncMock(return_value=result)
    return strategy


@pytest.fixture
def client():
    from kb_arena.chatbot.api import app

    strategies = {
        "naive_vector": _make_strategy("naive_vector", "Naive vector answer about json."),
        "contextual_vector": _make_strategy("contextual_vector", "Context-enriched answer."),
        "qna_pairs": _make_strategy("qna_pairs", "Pre-generated Q&A answer."),
        "knowledge_graph": _make_graph_strategy(),
        "hybrid": _make_strategy("hybrid", "Hybrid routing answer."),
    }

    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.strategies = strategies
        app.state.neo4j = None
        yield c


@pytest.fixture
def sample_chat_request():
    return {"query": "What does json.loads do?", "strategy": "naive_vector"}


@pytest.fixture
def factoid_request():
    return {"query": "What does json.loads do?", "strategy": "naive_vector"}


@pytest.fixture
def comparison_request():
    return {"query": "compare json.loads vs json.load", "strategy": "hybrid"}


@pytest.fixture
def history_request():
    return {
        "query": "What exceptions does it raise?",
        "strategy": "naive_vector",
        "history": [
            {"role": "user", "content": "What does json.loads do?"},
            {"role": "assistant", "content": "It deserializes JSON strings to Python objects."},
        ],
    }


@pytest.fixture
def live_client():
    """Live client using real API keys. Skipped unless KB_ARENA_ANTHROPIC_API_KEY is set."""
    import os

    if not os.environ.get("KB_ARENA_ANTHROPIC_API_KEY"):
        pytest.skip("Live tests require KB_ARENA_ANTHROPIC_API_KEY")

    from kb_arena.chatbot.api import app

    with TestClient(app) as c:
        yield c
