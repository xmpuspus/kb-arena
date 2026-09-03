"""A fetched web page over the size cap is rejected, not parsed into memory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kb_arena.ingest.parsers import web


def _fake_response(body: bytes, status_code: int = 200, headers: dict | None = None):
    resp = MagicMock()
    resp.content = body
    resp.text = body.decode("utf-8", errors="replace")
    resp.status_code = status_code
    resp.headers = headers or {}
    return resp


@pytest.fixture(autouse=True)
def _skip_ssrf_checks(monkeypatch):
    # The byte cap is independent of the SSRF guard, so tests skip DNS lookups.
    monkeypatch.setattr(web, "_validate_url", lambda url: None)


def test_a_response_under_the_cap_passes_through():
    client = MagicMock()
    client.get.return_value = _fake_response(b"hello world")

    resp = web._safe_get(client, "https://docs.example.com/page")

    assert resp.content == b"hello world"


def test_a_response_over_the_cap_is_rejected():
    client = MagicMock()
    oversized = b"x" * (web._MAX_RESPONSE_BYTES + 1)
    client.get.return_value = _fake_response(oversized)

    with pytest.raises(web.ResponseTooLargeError, match="exceeds"):
        web._safe_get(client, "https://docs.example.com/huge-page")


def test_fetch_page_returns_none_instead_of_crashing_on_an_oversized_page(caplog):
    client = MagicMock()
    oversized = b"x" * (web._MAX_RESPONSE_BYTES + 1)
    client.get.return_value = _fake_response(oversized, headers={"content-type": "text/html"})

    with caplog.at_level("WARNING", logger="kb_arena.ingest.parsers.web"):
        result = web._fetch_page("https://docs.example.com/huge-page", client)

    assert result is None
    assert "exceeds" in caplog.text


def test_llms_txt_check_skips_an_oversized_file():
    client = MagicMock()
    oversized = b"x" * (web._MAX_RESPONSE_BYTES + 1)
    client.get.return_value = _fake_response(oversized)

    result = web._check_llms_txt("https://docs.example.com", client)

    assert result is None


def test_the_cap_applies_at_every_redirect_hop():
    client = MagicMock()
    oversized = _fake_response(
        b"x" * (web._MAX_RESPONSE_BYTES + 1),
        status_code=302,
        headers={"location": "https://docs.example.com/next"},
    )
    client.get.return_value = oversized

    with pytest.raises(web.ResponseTooLargeError):
        web._safe_get(client, "https://docs.example.com/start")
