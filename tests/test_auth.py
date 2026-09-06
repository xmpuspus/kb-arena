"""Authentication and rate-limit regression tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from starlette.requests import Request


def _request(client_ip: str, headers: dict[str, str] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [
                (name.lower().encode(), value.encode()) for name, value in (headers or {}).items()
            ],
            "client": (client_ip, 12345),
            "server": ("testserver", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_rate_limit_store_evicts_least_recently_used_client(monkeypatch):
    from kb_arena.chatbot import auth
    from kb_arena.settings import settings

    auth._rate_store.clear()
    monkeypatch.setattr(auth, "_RATE_LIMIT_MAX_KEYS", 3)
    monkeypatch.setattr(settings, "trusted_proxy_header", "")

    try:
        for suffix in range(1, 5):
            auth.check_rate_limit(_request(f"192.0.2.{suffix}"))

        assert len(auth._rate_store) == 3
        assert "192.0.2.1" not in auth._rate_store
        assert list(auth._rate_store) == ["192.0.2.2", "192.0.2.3", "192.0.2.4"]
    finally:
        auth._rate_store.clear()


def test_rate_limit_consumption_is_atomic_at_capacity():
    from kb_arena.chatbot import auth

    client_id = "concurrent-client"
    with auth._rate_lock:
        auth._rate_store.clear()
        auth._rate_bucket(client_id).extend([100.0] * (auth.RATE_LIMIT_RPM - 1))

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            allowed = list(
                pool.map(
                    lambda _: auth._consume_rate_limit(client_id, now=101.0),
                    range(8),
                )
            )

        assert allowed.count(True) == 1
        assert allowed.count(False) == 7
        with auth._rate_lock:
            assert len(auth._rate_store[client_id]) == auth.RATE_LIMIT_RPM
    finally:
        with auth._rate_lock:
            auth._rate_store.clear()


def test_invalid_bearers_are_rate_limited(monkeypatch):
    from kb_arena.chatbot import auth
    from kb_arena.settings import settings

    auth._rate_store.clear()
    monkeypatch.setattr(settings, "api_token", "correct-token")
    monkeypatch.setattr(settings, "demo_mode", False)
    request = _request("192.0.2.50")

    try:
        for _ in range(auth.RATE_LIMIT_RPM):
            with pytest.raises(HTTPException) as exc_info:
                auth.require_auth(request, authorization="Bearer wrong-token")
            assert exc_info.value.status_code == 401

        with pytest.raises(HTTPException) as exc_info:
            auth.require_auth(request, authorization="Bearer wrong-token")
        assert exc_info.value.status_code == 429
    finally:
        auth._rate_store.clear()


def test_open_mode_allows_loopback_requests(monkeypatch):
    from kb_arena.chatbot import auth
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(settings, "demo_mode", False)

    assert auth.require_auth(_request("127.0.0.1")) is None


def test_open_mode_rejects_remote_requests(monkeypatch):
    from kb_arena.chatbot import auth
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(settings, "demo_mode", False)

    with pytest.raises(HTTPException) as exc_info:
        auth.require_auth(_request("192.0.2.10"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "api_token_required_for_remote_access"


def test_open_mode_rejects_remote_client_forwarded_by_loopback_proxy(monkeypatch):
    from kb_arena.chatbot import auth
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "trusted_proxy_header", "X-Forwarded-For")
    request = _request("127.0.0.1", {"X-Forwarded-For": "192.0.2.10"})

    with pytest.raises(HTTPException) as exc_info:
        auth.require_auth(request)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "api_token_required_for_remote_access"


def test_open_mode_rejects_loopback_proxy_without_client_header(monkeypatch):
    from kb_arena.chatbot import auth
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "trusted_proxy_header", "X-Forwarded-For")

    with pytest.raises(HTTPException) as exc_info:
        auth.require_auth(_request("127.0.0.1"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "api_token_required_for_remote_access"


def test_open_mode_rejects_spoofed_loopback_header_from_remote_peer(monkeypatch):
    from kb_arena.chatbot import auth
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "trusted_proxy_header", "X-Forwarded-For")
    request = _request("192.0.2.10", {"X-Forwarded-For": "127.0.0.1"})

    with pytest.raises(HTTPException) as exc_info:
        auth.require_auth(request)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "api_token_required_for_remote_access"


def test_every_public_read_route_carries_the_rate_limit():
    """A route that walks the results tree on every call needs the limiter.

    `require_read_auth` calls `check_rate_limit` itself, so a gated route is
    covered. The open aggregate routes had no limiter at all, and the hosted
    deployment note claimed 60 requests a minute for the whole API. A Codex pass
    proved the claim false against `/api/leaderboard`.
    """
    from kb_arena.chatbot import api
    from kb_arena.chatbot.auth import check_rate_limit, require_read_auth

    open_reads = {
        "/api/leaderboard",
        "/api/corpora",
        "/api/retriever-lab/runs",
        "/api/arena/leaderboard",
    }
    seen = set()
    for route in api.app.routes:
        path = getattr(route, "path", None)
        if path not in open_reads:
            continue
        seen.add(path)
        calls = {d.dependency for d in getattr(route, "dependencies", [])}
        assert (
            check_rate_limit in calls or require_read_auth in calls
        ), f"{path} answers without a token and without a rate limit"

    assert seen == open_reads, f"these routes are gone or renamed: {sorted(open_reads - seen)}"
