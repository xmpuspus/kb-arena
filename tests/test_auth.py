"""Authentication and rate-limit regression tests."""

from __future__ import annotations

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
