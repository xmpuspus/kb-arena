"""A fetched web page over the size cap is rejected mid-stream, before it accumulates."""

from __future__ import annotations

import httpx
import pytest

from kb_arena.ingest.parsers import web


@pytest.fixture(autouse=True)
def _skip_ssrf_checks(monkeypatch):
    # The byte cap is independent of the SSRF guard, so tests skip DNS lookups.
    monkeypatch.setattr(web, "_validate_url", lambda url: None)


def _client(body: bytes, status_code: int = 200, headers: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, headers=headers or {}, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_response_under_the_cap_passes_through():
    with _client(b"hello world") as client:
        resp = web._safe_get(client, "https://docs.example.com/page")

    assert resp.content == b"hello world"
    assert resp.text == "hello world"


def test_a_response_over_the_cap_is_rejected_during_streaming_with_no_declared_length():
    # A server using chunked transfer encoding sends no Content-Length, so the
    # only signal available is the running byte count while the body streams in.
    oversized = b"x" * (web._MAX_RESPONSE_BYTES + 1)

    def handler(request: httpx.Request) -> httpx.Response:
        resp = httpx.Response(200, content=oversized)
        del resp.headers["content-length"]
        return resp

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(web.ResponseTooLargeError, match="exceeded"):
            web._safe_get(client, "https://docs.example.com/huge-page")


def test_a_declared_content_length_over_the_cap_is_rejected_without_reading_the_body():
    # A malicious server can lie in the body but tell the truth in the header;
    # either signal alone must stop the fetch before the cap is crossed.
    declared_size = web._MAX_RESPONSE_BYTES + 1
    headers = {"content-length": str(declared_size)}
    with _client(b"short body", headers=headers) as client:
        with pytest.raises(web.ResponseTooLargeError, match="content-length"):
            web._safe_get(client, "https://docs.example.com/lying-header")


def test_fetch_page_returns_none_instead_of_crashing_on_an_oversized_page(caplog):
    oversized = b"x" * (web._MAX_RESPONSE_BYTES + 1)
    with _client(oversized, headers={"content-type": "text/html"}) as client:
        with caplog.at_level("WARNING", logger="kb_arena.ingest.parsers.web"):
            result = web._fetch_page("https://docs.example.com/huge-page", client)

    assert result is None
    assert "oversized" in caplog.text


def test_llms_txt_check_skips_an_oversized_file():
    oversized = b"x" * (web._MAX_RESPONSE_BYTES + 1)
    with _client(oversized) as client:
        result = web._check_llms_txt("https://docs.example.com", client)

    assert result is None


def test_the_cap_applies_at_every_redirect_hop():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/start"):
            return httpx.Response(302, headers={"location": "/next"})
        return httpx.Response(200, content=b"x" * (web._MAX_RESPONSE_BYTES + 1))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(web.ResponseTooLargeError):
            web._safe_get(client, "https://docs.example.com/start")
