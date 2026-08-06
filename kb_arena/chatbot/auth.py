"""Bearer-token auth + rate limiting for LLM-triggering endpoints.

When `KB_ARENA_API_TOKEN` is unset, auth is disabled (localhost dev). When set,
every LLM-triggering endpoint must present `Authorization: Bearer <token>` and
the token is constant-time compared.

Demo mode (`KB_ARENA_DEMO_MODE=true`) returns 503 from any LLM-triggering endpoint
so a hosted public demo cannot drain credits. The static benchmark/leaderboard
pages still work — they read JSON without invoking LLMs.
"""

from __future__ import annotations

import hmac
import time
from collections import OrderedDict, deque

from fastapi import Header, HTTPException, Request

from kb_arena.settings import settings

RATE_LIMIT_RPM = 60
_RATE_LIMIT_MAX_KEYS = 10_000
_rate_store: OrderedDict[str, deque[float]] = OrderedDict()


def _rate_bucket(client_id: str) -> deque[float]:
    """Return the client's bucket and evict the least recently used key when full."""
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
    now = time.time()
    window = 60.0
    bucket = _rate_bucket(client_id)
    # Pop entries older than the window
    while bucket and now - bucket[0] >= window:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="rate_limited")
    bucket.append(now)


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

    expected = settings.api_token
    if expected:
        provided = ""
        if authorization and authorization.startswith("Bearer "):
            provided = authorization[len("Bearer ") :].strip()
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    check_rate_limit(request)
