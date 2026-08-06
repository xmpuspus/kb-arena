"""Authentication and rate-limit regression tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from starlette.requests import Request


def _request(client_ip: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [],
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
