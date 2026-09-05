"""Conformance tests for the bring-your-own-retriever HTTP contract.

Covers the four endpoint cases the strategy must handle: a compliant
endpoint, a schema-violating endpoint, a slow endpoint (per-request timeout
and total budget), and an endpoint that resolves to a private address. No
test makes a real network call; SSRF checks and DNS answers are monkeypatched
the same way tests/test_web_ssrf_dns_rebinding.py exercises the guard.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from kb_arena.exceptions import RetrieverContractError
from kb_arena.ingest.parsers.web import SSRFBlocked, _PinnedClient
from kb_arena.models.retrieval import RetrievalTrace
from kb_arena.strategies.base import AnswerResult
from kb_arena.strategies.http_retriever import HTTPRetrieverStrategy


def _answers(*ips: str):
    return lambda host, port: [(socket.AF_INET, None, None, "", (ip, 0)) for ip in ips]


def _strategy(handler, **kwargs) -> HTTPRetrieverStrategy:
    client = _PinnedClient(transport=httpx.MockTransport(handler))
    return HTTPRetrieverStrategy("https://retriever.example.com/query", client=client, **kwargs)


def _compliant_body() -> dict:
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "source_doc_id": "doc-1",
                "source_section_id": "section-1",
                "content": "Lambda memory settings.",
                "score": 0.9,
            },
            {
                "chunk_id": "c2",
                "source_doc_id": "doc-2",
                "source_section_id": "section-2",
                "content": "Lambda timeout settings.",
                "score": 0.5,
            },
        ]
    }


# --- Case 1: compliant endpoint ---


@pytest.mark.asyncio
async def test_compliant_endpoint_produces_a_retrieval_trace(monkeypatch, mock_llm_client):
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_compliant_body())

    strategy = _strategy(handler)
    strategy._llm = mock_llm_client

    result = await strategy.query("How do I set Lambda memory?", top_k=2)

    assert isinstance(result, AnswerResult)
    assert isinstance(result.retrieval, RetrievalTrace)
    assert [c.chunk_id for c in result.retrieval.retrieved] == ["c1", "c2"]
    assert result.retrieval.retrieved[0].doc_id == "doc-1"
    assert result.retrieval.retrieved[0].metadata["source_doc_id"] == "doc-1"
    assert result.retrieval.retrieved[0].metadata["source_section_id"] == "section-1"
    assert result.sources == ["doc-1", "doc-2"]
    assert result.answer == "This is a generated answer."


# --- Case 2: schema-violating endpoint ---


def test_schema_violating_endpoint_fails_closed_with_a_named_field(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"chunks": [{"chunk_id": "c1", "content": "no source ids here"}]},
        )

    strategy = _strategy(handler)

    with pytest.raises(RetrieverContractError, match="source_doc_id"):
        strategy._fetch_chunks("How do I set Lambda memory?", top_k=2, corpus="all")


def test_non_json_body_fails_closed(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    strategy = _strategy(handler)

    with pytest.raises(RetrieverContractError, match="not valid JSON"):
        strategy._fetch_chunks("q", top_k=2, corpus="all")


def test_error_status_fails_closed(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"chunks": []})

    strategy = _strategy(handler)

    with pytest.raises(RetrieverContractError, match="status 500"):
        strategy._fetch_chunks("q", top_k=2, corpus="all")


# --- Case 3: slow endpoint (per-request timeout and total budget) ---


def test_slow_endpoint_fails_the_call_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    strategy = _strategy(handler, request_timeout_s=1.0)

    with pytest.raises(RetrieverContractError, match="no reply within"):
        strategy._fetch_chunks("q", top_k=2, corpus="all")
    # A timed-out call still spends its share of the budget.
    assert strategy._spent_s > 0


def test_exhausted_budget_refuses_before_dialing(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("dialed the endpoint after the budget was already spent")

    strategy = _strategy(handler, total_budget_s=5.0)
    strategy._spent_s = 5.0

    with pytest.raises(RetrieverContractError, match="budget"):
        strategy._fetch_chunks("q", top_k=2, corpus="all")


# --- Case 4: private address ---


def test_private_address_is_blocked_before_any_request(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answers("10.0.0.5"))

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("dialed a private address the SSRF guard should have blocked")

    strategy = _strategy(handler)

    with pytest.raises(SSRFBlocked, match="private"):
        strategy._fetch_chunks("q", top_k=2, corpus="all")


# --- Case 5: an oversized body, and a shared budget ---


def test_an_oversized_reply_is_refused_before_it_is_parsed(monkeypatch):
    """`client.post` buffers and decompresses the whole body before any check.

    So a compliant-looking endpoint returning a gzip bomb exhausted memory for
    a `top_k` of 1. The reply streams under the same cap the web fetcher uses.
    """
    from kb_arena.ingest.parsers.web import _MAX_RESPONSE_BYTES

    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))
    oversized = b'{"chunks": [' + b" " * (_MAX_RESPONSE_BYTES + 1024) + b"]}"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized)

    with pytest.raises(RetrieverContractError, match="cap"):
        _strategy(handler)._fetch_chunks("q", top_k=1, corpus="all")


def test_two_concurrent_calls_cannot_spend_one_budget_twice(monkeypatch):
    """Reading the remainder and charging it later let both callers see it whole.

    A strategy instance is shared across concurrent requests, so this is the
    ordinary case rather than a corner one. The reservation is charged when it
    is taken, and settled against the time the call really took.
    """
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("no request should reach the endpoint in this test")

    strategy = _strategy(handler, total_budget_s=1.0, request_timeout_s=1.0)

    assert strategy._reserve_budget() == 1.0
    with pytest.raises(RetrieverContractError, match="budget"):
        strategy._reserve_budget()


def test_a_reservation_settles_against_the_time_the_call_took(monkeypatch):
    """A call that returns fast must not consume its whole allowance."""
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("no request should reach the endpoint in this test")

    strategy = _strategy(handler, total_budget_s=10.0, request_timeout_s=4.0)

    allowance = strategy._reserve_budget()
    strategy._charge_budget(0.25, allowance)

    assert strategy._spent_s == pytest.approx(0.25)


def test_a_slow_drip_reply_stops_at_the_deadline(monkeypatch):
    """httpx applies its timeout to each socket wait, not to the whole call.

    An endpoint dripping one byte at a time never waits long enough to trip
    that timeout, so it ran past the timeout AND past the budget and then
    succeeded. The deadline is wall-clock now.
    """
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))
    ticks = iter([0.0])
    monkeypatch.setattr(
        "kb_arena.strategies.http_retriever.time.perf_counter",
        lambda: next(ticks, 99.0),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"chunks": []}')

    with pytest.raises(RetrieverContractError, match="no reply within"):
        _strategy(handler, request_timeout_s=1.0)._fetch_chunks("q", top_k=1, corpus="all")


def test_the_request_asks_for_an_unencoded_body(monkeypatch):
    """httpx advertises `br` when Brotli is installed, and the reader has no Brotli decoder.

    A compliant endpoint honouring that header crashed the call with
    "unsupported content-encoding 'br'".
    """
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["accept-encoding"] = request.headers.get("accept-encoding", "")
        return httpx.Response(200, json=_compliant_body())

    _strategy(handler)._fetch_chunks("q", top_k=2, corpus="all")

    assert seen["accept-encoding"] == "identity"
