"""Bearer-token auth + rate limiting for LLM-triggering endpoints.

When `KB_ARENA_API_TOKEN` is unset, auth is disabled (localhost dev). When set,
every LLM-triggering endpoint must present `Authorization: Bearer <token>` and
the token is constant-time compared.

Demo mode (`KB_ARENA_DEMO_MODE=true`) returns 503 from any LLM-triggering endpoint
so a hosted public demo cannot drain credits. The static benchmark/leaderboard
pages still work because they read JSON without invoking LLMs.
"""

from __future__ import annotations

import hmac
import ipaddress
import time
from collections import OrderedDict, deque
from threading import RLock

from fastapi import Header, HTTPException, Request

from kb_arena.settings import settings

RATE_LIMIT_RPM = 60
_RATE_LIMIT_MAX_KEYS = 10_000
_rate_store: OrderedDict[str, deque[float]] = OrderedDict()
_rate_lock = RLock()


def _rate_bucket(client_id: str) -> deque[float]:
    """Return the client's bucket and evict the least recently used key when full."""
    with _rate_lock:
        bucket = _rate_store.get(client_id)
        if bucket is None:
            while len(_rate_store) >= _RATE_LIMIT_MAX_KEYS:
                _rate_store.popitem(last=False)
            bucket = deque(maxlen=RATE_LIMIT_RPM)
            _rate_store[client_id] = bucket
        elif not isinstance(bucket, deque):
            bucket = deque(bucket, maxlen=RATE_LIMIT_RPM)
            _rate_store[client_id] = bucket
        _rate_store.move_to_end(client_id)
        return bucket


def _consume_rate_limit(client_id: str, *, now: float | None = None) -> bool:
    """Atomically consume one request from the client's rolling-minute allowance."""
    with _rate_lock:
        timestamp = time.time() if now is None else now
        bucket = _rate_bucket(client_id)
        while bucket and timestamp - bucket[0] >= 60.0:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_RPM:
            return False
        bucket.append(timestamp)
        return True


def _client_key(request: Request) -> str:
    """Resolve client identity for rate limiting. Honors trusted proxy header when configured."""
    if settings.trusted_proxy_header:
        forwarded = request.headers.get(settings.trusted_proxy_header)
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    """Raise 429 if the caller exceeds RATE_LIMIT_RPM. Bounded memory."""
    client_id = _client_key(request)
    if not _consume_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="rate_limited")


def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Bearer-token auth + rate limit + demo-mode gate. Use as `Depends(require_auth)`."""
    if settings.demo_mode:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "demo_mode",
                "message": (
                    "This is a read-only public demo. "
                    "Run KB Arena locally to use chat, arena, tools, and graph endpoints."
                ),
            },
        )

    check_rate_limit(request)

    expected = settings.api_token
    if expected:
        provided = ""
        if authorization and authorization.startswith("Bearer "):
            provided = authorization[len("Bearer ") :].strip()
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="unauthorized")
    else:
        client_host = request.client.host if request.client else ""
        try:
            is_loopback = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            is_loopback = client_host.lower() == "localhost"
        if not is_loopback:
            raise HTTPException(status_code=401, detail="api_token_required_for_remote_access")
