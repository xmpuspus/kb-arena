"""Three web-parser defects from the SSRF review: the crawl cap, a temporary
resolver failure, and a compressed body that inflates past the byte cap.
"""

from __future__ import annotations

import gzip
import socket

import httpx
import pytest

from kb_arena.ingest.parsers import web
from kb_arena.ingest.parsers.web import WebParser


@pytest.fixture
def pin_stub(monkeypatch):
    # The byte cap is independent of the SSRF guard. MockTransport never
    # dials the address, so the stub value is inert.
    monkeypatch.setattr(web, "_validate_url", lambda url: ["203.0.113.1"])


def _streamed(body: bytes, headers: dict | None = None) -> httpx.Client:
    # A body given as stream= reaches the reader raw, like a live response.
    # A body given as content= is read and decoded at construction instead.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers or {}, stream=httpx.ByteStream(body))

    return web._PinnedClient(transport=httpx.MockTransport(handler))


# N-13: the page cap counts fetch attempts, so empty pages cannot inflate it.


def test_the_crawl_cap_counts_fetch_attempts_not_extracted_pages(monkeypatch):
    fetched: list[str] = []

    def fake_fetch(url, client):
        fetched.append(url)
        return "<html></html>"

    monkeypatch.setattr(web, "_fetch_page", fake_fetch)
    monkeypatch.setattr(web, "_clean_html", lambda html, bs: "")
    monkeypatch.setattr(web, "_extract_links", lambda html, url, bs: [f"{url}/a", f"{url}/b"])

    docs = WebParser(max_depth=3, max_pages=1)._crawl("https://docs.example.com", "c", None, None)

    assert docs == []
    assert fetched == ["https://docs.example.com"]


# N-14: EAI_AGAIN is an outage, not a policy refusal.


def test_a_temporary_resolver_failure_raises_its_own_error(monkeypatch):
    def try_again(host, port):
        raise socket.gaierror(socket.EAI_AGAIN, "temporary failure in name resolution")

    monkeypatch.setattr(socket, "getaddrinfo", try_again)

    with pytest.raises(web.DNSTemporaryError, match="retry later"):
        web._validate_url("https://docs.example.com/page")


def test_a_permanent_resolver_failure_still_reads_as_blocked(monkeypatch):
    def no_such_host(host, port):
        raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", no_such_host)

    with pytest.raises(web.SSRFBlocked, match="dns resolution failed"):
        web._validate_url("https://docs.example.com/page")


def test_scrape_reports_a_resolver_outage_instead_of_an_empty_corpus(monkeypatch):
    def try_again(host, port):
        raise socket.gaierror(socket.EAI_AGAIN, "temporary failure in name resolution")

    monkeypatch.setattr(socket, "getaddrinfo", try_again)

    with pytest.raises(web.DNSTemporaryError):
        WebParser()._scrape("https://docs.example.com", "c")


# N-15: the cap holds on decoded bytes without inflating a compressed chunk first.


def test_the_request_asks_for_an_uncompressed_body(pin_stub):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["accept_encoding"] = request.headers.get("accept-encoding")
        return httpx.Response(200, stream=httpx.ByteStream(b"hello"))

    with web._PinnedClient(transport=httpx.MockTransport(handler)) as client:
        resp = web._safe_get(client, "https://docs.example.com/page")

    assert seen["accept_encoding"] == "identity"
    assert resp.content == b"hello"


def test_a_gzip_bomb_is_rejected_before_it_inflates_past_the_cap(pin_stub):
    bomb = gzip.compress(b"0" * (web._MAX_RESPONSE_BYTES + 4096))
    assert len(bomb) < 64 * 1024  # a few KB on the wire, over 10 MB inflated

    with _streamed(bomb, {"content-encoding": "gzip"}) as client:
        with pytest.raises(web.ResponseTooLargeError, match="exceeded"):
            web._safe_get(client, "https://docs.example.com/bomb")


def test_a_compressed_body_under_the_cap_still_decodes(pin_stub):
    body = b"<html>fine</html>" * 100

    with _streamed(gzip.compress(body), {"content-encoding": "gzip"}) as client:
        resp = web._safe_get(client, "https://docs.example.com/page")

    assert resp.content == body


class _CountingDecoder:
    """Wraps a zlib decompressor and records the largest output it emitted."""

    largest = 0

    def __init__(self, inner) -> None:
        self._inner = inner

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        out = self._inner.decompress(data, max_length)
        _CountingDecoder.largest = max(_CountingDecoder.largest, len(out))
        return out

    @property
    def unconsumed_tail(self) -> bytes:
        return self._inner.unconsumed_tail

    def flush(self) -> bytes:
        return self._inner.flush()


def test_the_decoder_never_emits_past_the_cap_even_for_one_chunk(pin_stub, monkeypatch):
    # One compressed chunk that inflates to three times the cap. The reader
    # must stop the decoder at the cap instead of taking the whole expansion.
    bomb = gzip.compress(b"0" * (web._MAX_RESPONSE_BYTES * 3))
    real_decoder_for = web._decoder_for
    _CountingDecoder.largest = 0

    def counting_decoder_for(encoding: str):
        inner = real_decoder_for(encoding)
        return None if inner is None else _CountingDecoder(inner)

    monkeypatch.setattr(web, "_decoder_for", counting_decoder_for)

    with _streamed(bomb, {"content-encoding": "gzip"}) as client:
        with pytest.raises(web.ResponseTooLargeError):
            web._safe_get(client, "https://docs.example.com/bomb")

    assert 0 < _CountingDecoder.largest <= web._MAX_RESPONSE_BYTES + 1


def test_an_encoding_the_reader_cannot_bound_is_refused(pin_stub):
    with _streamed(b"\x00", {"content-encoding": "br"}) as client:
        with pytest.raises(httpx.DecodingError, match="unsupported content-encoding"):
            web._safe_get(client, "https://docs.example.com/page")
