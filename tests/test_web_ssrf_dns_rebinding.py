"""_safe_get connects to the IP _validate_url checked, so a second DNS lookup
at connect time cannot return a different, private address.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from kb_arena.ingest.parsers import web


def test_validate_url_returns_a_checked_ip(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )

    ip = web._validate_url("https://docs.example.com/page")

    assert ip == "93.184.216.34"


def test_validate_url_still_blocks_a_private_answer(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("10.0.0.5", 0))],
    )

    with pytest.raises(web.SSRFBlocked, match="private"):
        web._validate_url("https://docs.example.com/page")


def test_safe_get_connects_to_the_checked_ip_not_the_hostname(monkeypatch):
    # The check sees a safe public IP. A rebinding attacker wants the actual
    # connection to resolve the same host to a private one instead.
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )

    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        return httpx.Response(200, content=b"hello")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        web._safe_get(client, "https://docs.example.com/page")

    assert seen_hosts == ["93.184.216.34"]


def test_safe_get_resolves_dns_exactly_once_per_hop(monkeypatch):
    # A rebinding host answers differently on a second lookup. If _safe_get
    # ever resolved again after the check, this second call would return a
    # private IP the check never saw.
    calls = {"count": 0}
    answers = iter(["93.184.216.34", "10.0.0.5"])

    def fake_getaddrinfo(host, port):
        calls["count"] += 1
        return [(socket.AF_INET, None, None, "", (next(answers), 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"hello")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        web._safe_get(client, "https://docs.example.com/page")

    assert calls["count"] == 1


def test_safe_get_sends_the_real_hostname_as_host_header_and_sni(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host_header"] = request.headers.get("host")
        seen["sni_hostname"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"hello")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        web._safe_get(client, "https://docs.example.com/page")

    assert seen["host_header"] == "docs.example.com"
    assert seen["sni_hostname"] == "docs.example.com"


def test_a_redirect_to_a_new_host_pins_to_that_host_own_checked_ip(monkeypatch):
    answers = {
        "docs.example.com": "93.184.216.34",
        "other.example.com": "1.1.1.1",
    }
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", (answers[host], 0))],
    )

    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.host == "93.184.216.34":
            return httpx.Response(302, headers={"location": "https://other.example.com/next"})
        return httpx.Response(200, content=b"hello")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        web._safe_get(client, "https://docs.example.com/start")

    assert seen_hosts == ["93.184.216.34", "1.1.1.1"]
