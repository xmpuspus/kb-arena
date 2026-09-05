"""traced_span must emit real spans when tracing is on, and never break
the caller when the `otel` extra is absent.

The core install carries no OpenTelemetry dependency, so this suite
proves both directions: the extra present (an in-memory exporter reads
back the expected span and attributes) and the extra absent (a simulated
ImportError still lets the wrapped code run and export nothing).
"""

from __future__ import annotations

import sys

import pytest

from kb_arena import telemetry
from kb_arena.settings import settings


def _in_memory_provider():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_traced_span_emits_a_span_with_the_expected_attributes(monkeypatch):
    provider, exporter = _in_memory_provider()
    monkeypatch.setattr(settings, "otel_enabled", True)
    monkeypatch.setattr(telemetry, "_tracer_provider", lambda: provider)

    with telemetry.traced_span(
        "kb_arena.retrieval", strategy="naive_vector", corpus="aws-compute", top_k=5
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "kb_arena.retrieval"
    assert span.attributes["strategy"] == "naive_vector"
    assert span.attributes["corpus"] == "aws-compute"
    assert span.attributes["top_k"] == 5


def test_traced_span_skips_a_none_attribute(monkeypatch):
    provider, exporter = _in_memory_provider()
    monkeypatch.setattr(settings, "otel_enabled", True)
    monkeypatch.setattr(telemetry, "_tracer_provider", lambda: provider)

    with telemetry.traced_span("kb_arena.judge", strategy="naive_vector", top_k=None):
        pass

    span = exporter.get_finished_spans()[0]
    assert "top_k" not in span.attributes


def test_traced_span_never_carries_question_or_document_text(monkeypatch):
    provider, exporter = _in_memory_provider()
    monkeypatch.setattr(settings, "otel_enabled", True)
    monkeypatch.setattr(telemetry, "_tracer_provider", lambda: provider)

    with telemetry.traced_span("kb_arena.embedding", provider="openai"):
        pass

    span = exporter.get_finished_spans()[0]
    assert set(span.attributes.keys()) == {"provider"}


def test_traced_span_is_a_no_op_when_tracing_is_off(monkeypatch):
    monkeypatch.setattr(settings, "otel_enabled", False)
    ran = False
    with telemetry.traced_span("kb_arena.retrieval", strategy="naive_vector"):
        ran = True
    assert ran


def test_traced_span_runs_the_block_when_the_otel_extra_is_absent(monkeypatch):
    """Simulate the `otel` extra being uninstalled by making its import fail.

    KB_ARENA_OTEL_ENABLED=true with the package missing must not raise; the
    wrapped code still runs and the span is simply never recorded anywhere.
    """
    monkeypatch.setattr(settings, "otel_enabled", True)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)

    ran = False
    with telemetry.traced_span("kb_arena.retrieval", strategy="naive_vector"):
        ran = True
    assert ran


def test_core_install_declares_no_opentelemetry_dependency():
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())[
        "project"
    ]
    assert not any("opentelemetry" in dep for dep in project["dependencies"])
    assert any("opentelemetry-api" in dep for dep in project["optional-dependencies"]["otel"])


@pytest.mark.asyncio
async def test_naive_vector_retrieval_is_traced(monkeypatch):
    """A real strategy call opens the retrieval span with strategy/corpus/top_k."""
    from kb_arena.models.retrieval import RetrievedChunk
    from kb_arena.strategies import naive_vector as naive_vector_module

    provider, exporter = _in_memory_provider()
    monkeypatch.setattr(settings, "otel_enabled", True)
    monkeypatch.setattr(telemetry, "_tracer_provider", lambda: provider)

    strategy_cls = naive_vector_module.NaiveVectorStrategy
    strategy = strategy_cls.__new__(strategy_cls)
    strategy._client = None
    strategy._collection = object()
    strategy._llm = None
    strategy.last_sources = []
    strategy.last_graph_context = None
    strategy.last_latency_ms = 0.0
    strategy.last_tokens_used = 0
    strategy.last_cost_usd = 0.0

    monkeypatch.setattr(strategy, "_get_collection", lambda: strategy._collection)

    async def fake_query_in_thread(collection, collection_name, corpus, query_kwargs):
        return {}

    monkeypatch.setattr(naive_vector_module, "query_in_thread", fake_query_in_thread)
    monkeypatch.setattr(
        naive_vector_module,
        "parse_query_result",
        lambda results: ([], [], [], []),
    )

    class _FakeLLM:
        async def generate(self, **kwargs):
            from kb_arena.llm.client import LLMResponse

            return LLMResponse(text="answer", input_tokens=0, output_tokens=0, cost_usd=0.0)

    monkeypatch.setattr(strategy, "_get_llm", lambda: _FakeLLM())

    await strategy.query("what is lambda?", top_k=3, corpus="aws-compute")

    retrieval_spans = [s for s in exporter.get_finished_spans() if s.name == "kb_arena.retrieval"]
    assert len(retrieval_spans) == 1
    assert retrieval_spans[0].attributes["strategy"] == "naive_vector"
    assert retrieval_spans[0].attributes["corpus"] == "aws-compute"
    assert retrieval_spans[0].attributes["top_k"] == 3
    assert not any(isinstance(c, RetrievedChunk) for c in retrieval_spans[0].attributes.values())
