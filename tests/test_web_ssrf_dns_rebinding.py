"""_safe_get pins each socket to the IPs _validate_url checked, so a second DNS
lookup at connect time cannot steer the connection to a private address.
"""

from __future__ import annotations

import socket

import httpcore
import httpx
import pytest

from kb_arena.ingest.parsers import web


def _answers(*ips: str):
    return lambda host, port: [(socket.AF_INET, None, None, "", (ip, 0)) for ip in ips]


def test_validate_url_returns_every_checked_ip_once_in_resolver_order(monkeypatch):
    # getaddrinfo repeats an address once per socket type; the pin list must not.
    monkeypatch.setattr(
        socket, "getaddrinfo", _answers("93.184.216.34", "93.184.216.34", "1.1.1.1")
    )

    assert web._validate_url("https://docs.example.com/page") == ["93.184.216.34", "1.1.1.1"]


def test_validate_url_still_blocks_a_private_answer(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answers("10.0.0.5"))

    with pytest.raises(web.SSRFBlocked, match="private"):
        web._validate_url("https://docs.example.com/page")


def test_backend_dials_the_checked_ip_not_the_hostname(monkeypatch):
    dialed: list[tuple[str, int]] = []

    def fake_connect(self, host, port, timeout=None, local_address=None, socket_options=None):
        dialed.append((host, port))
        return "stream"

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", fake_connect)
    backend = web._PinnedBackend({"docs.example.com": ["93.184.216.34"]})

    assert backend.connect_tcp("docs.example.com", 443) == "stream"
    assert dialed == [("93.184.216.34", 443)]


def test_backend_falls_back_to_the_next_checked_ip_when_the_first_refuses(monkeypatch):
    dialed: list[str] = []

    def fake_connect(self, host, port, timeout=None, local_address=None, socket_options=None):
        dialed.append(host)
        if host == "2606:4700:10::6814:179a":
            raise httpcore.ConnectError("network unreachable")
        return "stream"

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", fake_connect)
    backend = web._PinnedBackend({"docs.example.com": ["2606:4700:10::6814:179a", "93.184.216.34"]})

    assert backend.connect_tcp("docs.example.com", 443) == "stream"
    assert dialed == ["2606:4700:10::6814:179a", "93.184.216.34"]


def test_backend_refuses_a_host_the_guard_never_checked(monkeypatch):
    def fake_connect(self, host, port, timeout=None, local_address=None, socket_options=None):
        pytest.fail(f"dialed {host} without a check")

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", fake_connect)
    backend = web._PinnedBackend({})

    with pytest.raises(httpcore.ConnectError, match="no checked address"):
        backend.connect_tcp("docs.example.com", 443)


def test_pinned_client_wires_its_pins_into_the_real_transport():
    with web._PinnedClient() as client:
        pool = client._transport._pool

    assert isinstance(client._transport, web._PinnedTransport)
    assert pool._network_backend._pins is client.pins


def test_safe_get_registers_the_checked_ips_and_keeps_the_real_hostname(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["pins"] = dict(client.pins)
        seen["url_host"] = request.url.host
        seen["host_header"] = request.headers["host"]
        return httpx.Response(200, content=b"hello")

    client = web._PinnedClient(transport=httpx.MockTransport(handler))
    with client:
        web._safe_get(client, "https://docs.example.com/page")

    assert seen["pins"] == {"docs.example.com": ["93.184.216.34"]}
    # The URL keeps the hostname, so httpx pools, scopes cookies, and starts
    # TLS by hostname. Only the socket target below it changes.
    assert seen["url_host"] == "docs.example.com"
    assert seen["host_header"] == "docs.example.com"


def test_safe_get_resolves_dns_exactly_once_per_hop(monkeypatch):
    # A rebinding host answers differently on a second lookup. If _safe_get
    # ever resolved again after the check, the second answer, a private IP
    # the check never saw, would reach the connection.
    calls = {"count": 0}
    answers = iter(["93.184.216.34", "10.0.0.5"])

    def fake_getaddrinfo(host, port):
        calls["count"] += 1
        return [(socket.AF_INET, None, None, "", (next(answers), 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"hello")

    with web._PinnedClient(transport=httpx.MockTransport(handler)) as client:
        web._safe_get(client, "https://docs.example.com/page")

    assert calls["count"] == 1


def test_a_redirect_to_a_new_host_gets_that_host_own_checked_ips(monkeypatch):
    answers = {"docs.example.com": "93.184.216.34", "other.example.com": "1.1.1.1"}
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", (answers[host], 0))],
    )
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.headers["host"])
        if request.url.host == "docs.example.com":
            return httpx.Response(302, headers={"location": "https://other.example.com/next"})
        return httpx.Response(200, content=b"hello")

    with web._PinnedClient(transport=httpx.MockTransport(handler)) as client:
        web._safe_get(client, "https://docs.example.com/start")

    assert seen_hosts == ["docs.example.com", "other.example.com"]
    assert client.pins == {
        "docs.example.com": ["93.184.216.34"],
        "other.example.com": ["1.1.1.1"],
    }


def test_a_plain_client_cannot_be_used_by_mistake(monkeypatch):
    # A client without a pin registry would connect wherever DNS says at
    # connect time, so _safe_get must fail loudly instead of silently unpinned.
    monkeypatch.setattr(socket, "getaddrinfo", _answers("93.184.216.34"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"hello")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AttributeError, match="pins"):
            web._safe_get(client, "https://docs.example.com/page")
