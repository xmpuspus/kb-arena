"""Three web-parser defects from the SSRF review: the crawl cap, a temporary
resolver failure, and a compressed body that inflates past the byte cap.
"""

from __future__ import annotations

import gzip
import socket
import zlib

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

    def fake_fetch(url, client, *, raise_transport=False):
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

    with pytest.raises(web.DNSFailureError, match="dns lookup"):
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

    with pytest.raises(web.DNSFailureError):
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

    @property
    def unused_data(self) -> bytes:
        return self._inner.unused_data

    @property
    def eof(self) -> bool:
        return self._inner.eof

    def flush(self) -> bytes:
        return self._inner.flush()


def test_the_decoder_never_emits_past_the_cap_even_for_one_chunk(pin_stub, monkeypatch):
    # One compressed chunk that inflates to three times the cap. The reader
    # must stop the decoder at the cap instead of taking the whole expansion.
    bomb = gzip.compress(b"0" * (web._MAX_RESPONSE_BYTES * 3))
    real_decoder_for = web._decoder_for
    _CountingDecoder.largest = 0

    def counting_decoder_for(encoding: str):
        make = real_decoder_for(encoding)
        return None if make is None else (lambda: _CountingDecoder(make()))

    monkeypatch.setattr(web, "_decoder_for", counting_decoder_for)

    with _streamed(bomb, {"content-encoding": "gzip"}) as client:
        with pytest.raises(web.ResponseTooLargeError):
            web._safe_get(client, "https://docs.example.com/bomb")

    assert 0 < _CountingDecoder.largest <= web._MAX_RESPONSE_BYTES + 1


def test_an_encoding_the_reader_cannot_bound_is_refused(pin_stub):
    with _streamed(b"\x00", {"content-encoding": "br"}) as client:
        with pytest.raises(httpx.DecodingError, match="unsupported content-encoding"):
            web._safe_get(client, "https://docs.example.com/page")


def test_a_gzip_body_with_several_members_decodes_in_full(pin_stub):
    # A valid gzip stream can hold members back to back. One decompressor
    # stops at the first member, so the reader must start a fresh one.
    body = gzip.compress(b"first,") + gzip.compress(b"second,") + gzip.compress(b"third")

    with _streamed(body, {"content-encoding": "gzip"}) as client:
        resp = web._safe_get(client, "https://docs.example.com/page")

    assert resp.content == b"first,second,third"


def test_a_bomb_hidden_in_a_later_gzip_member_is_still_rejected(pin_stub):
    body = gzip.compress(b"first") + gzip.compress(b"0" * (web._MAX_RESPONSE_BYTES + 1))

    with _streamed(body, {"content-encoding": "gzip"}) as client:
        with pytest.raises(web.ResponseTooLargeError):
            web._safe_get(client, "https://docs.example.com/bomb")


def test_a_resolver_outage_mid_crawl_still_reaches_the_operator(monkeypatch):
    # The entry check resolves fine. Every lookup after it hits EAI_AGAIN.
    # Both fetch paths catch generic errors, so the outage must be re-raised
    # ahead of them instead of turning into "no llms.txt" and "no page".
    calls = {"count": 0}

    def flaky_getaddrinfo(host, port):
        calls["count"] += 1
        if calls["count"] == 1:
            return [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))]
        raise socket.gaierror(socket.EAI_AGAIN, "temporary failure in name resolution")

    monkeypatch.setattr(socket, "getaddrinfo", flaky_getaddrinfo)
    monkeypatch.setattr(web, "_try_import_bs4", lambda: object)

    with pytest.raises(web.DNSFailureError):
        WebParser(max_pages=1)._scrape("https://docs.example.com", "c")


def test_a_non_recoverable_resolver_failure_is_not_a_policy_refusal(monkeypatch):
    def resolver_down(host, port):
        raise socket.gaierror(socket.EAI_FAIL, "non-recoverable failure in name resolution")

    monkeypatch.setattr(socket, "getaddrinfo", resolver_down)

    with pytest.raises(web.DNSFailureError, match="dns lookup"):
        web._validate_url("https://docs.example.com/page")


def test_a_gzip_stream_cut_off_early_is_not_accepted_as_complete(pin_stub):
    # zlib's flush() does not object to a truncated member, so only the
    # decoder's eof flag can tell a dropped connection from a whole page.
    cut = gzip.compress(b"<html><body>complete</body></html>")[:-8]

    with _streamed(cut, {"content-encoding": "gzip"}) as client:
        with pytest.raises(httpx.DecodingError, match="ended before its stream"):
            web._safe_get(client, "https://docs.example.com/page")


def test_a_raw_deflate_body_decodes_like_httpx_would(pin_stub):
    # Some servers send raw deflate under the "deflate" name. httpx retries
    # raw on the first zlib error, and the bounded reader must too.
    raw = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    body = raw.compress(b"<html>raw deflate</html>") + raw.flush()

    with _streamed(body, {"content-encoding": "deflate"}) as client:
        resp = web._safe_get(client, "https://docs.example.com/page")

    assert resp.content == b"<html>raw deflate</html>"


def test_a_zlib_deflate_body_decodes(pin_stub):
    with _streamed(zlib.compress(b"<html>zlib</html>"), {"content-encoding": "deflate"}) as client:
        resp = web._safe_get(client, "https://docs.example.com/page")

    assert resp.content == b"<html>zlib</html>"


def test_a_comma_separated_encoding_with_identity_still_decodes(pin_stub):
    with _streamed(
        gzip.compress(b"<html>ok</html>"), {"content-encoding": "gzip, identity"}
    ) as client:
        resp = web._safe_get(client, "https://docs.example.com/page")

    assert resp.content == b"<html>ok</html>"


def test_a_corrupt_compressed_body_raises_one_typed_error(pin_stub):
    with _streamed(b"\x1f\x8b" + b"not gzip at all", {"content-encoding": "gzip"}) as client:
        with pytest.raises(httpx.DecodingError, match="corrupt compressed body"):
            web._safe_get(client, "https://docs.example.com/page")


def test_the_scraper_ignores_environment_proxies(monkeypatch):
    # A proxy resolves the host itself, so it would carry a request past
    # the pinned socket. The client must not mount one from the environment.
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:3128")

    with web._PinnedClient() as client:
        assert client._mounts == {}
        assert isinstance(client._transport, web._PinnedTransport)


def test_an_unreachable_entry_page_reports_instead_of_an_empty_corpus(pin_stub, monkeypatch):
    def refuse(client, url, timeout=15, max_redirects=5):
        raise httpx.ConnectTimeout("connect timed out")

    monkeypatch.setattr(web, "_safe_get", refuse)
    monkeypatch.setattr(web, "_try_import_bs4", lambda: object)

    with pytest.raises(httpx.ConnectTimeout):
        WebParser()._scrape("https://docs.example.com", "c")


def test_a_slow_link_mid_crawl_keeps_the_pages_already_collected(monkeypatch):
    def fetch(url, client, *, raise_transport=False):
        if url.endswith("/slow"):
            if raise_transport:
                raise httpx.ReadTimeout("read timed out")
            return None
        return "<html>page</html>"

    monkeypatch.setattr(web, "_fetch_page", fetch)
    monkeypatch.setattr(web, "_clean_html", lambda html, bs: "some text")
    monkeypatch.setattr(
        web, "_extract_links", lambda html, url, bs: [f"{url}/slow"] if url.endswith("com") else []
    )

    docs = WebParser(max_depth=2, max_pages=5)._crawl("https://docs.example.com", "c", None, None)

    assert [d.source for d in docs] == ["https://docs.example.com"]
