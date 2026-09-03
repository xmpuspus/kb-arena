"""Process-wide logging: level and format from settings, request ids on every record.

The CLI and the API both call ``configure_logging`` once. ``bind_request_id`` sets the
id that the filter stamps on each record, so one failing request can be traced from
the response header to the log line that explains it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import Mapping
from contextvars import ContextVar

from kb_arena.settings import Settings, settings

_request_id: ContextVar[str] = ContextVar("kb_arena_request_id", default="")

# A client may supply its own id so its traces join ours. Keep it short and printable.
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{8,64}$")
_ENV_PREFIX = "KB_ARENA_"
_NOISY_LOGGERS = ("chromadb.telemetry", "chromadb.telemetry.product.posthog")


def new_request_id() -> str:
    return uuid.uuid4().hex


def normalize_request_id(candidate: str | None) -> str:
    """Return the client id when it is safe to log, otherwise a fresh one."""
    if candidate and _REQUEST_ID_PATTERN.match(candidate):
        return candidate
    return new_request_id()


def bind_request_id(request_id: str):
    return _request_id.set(request_id)


def reset_request_id(token) -> None:
    _request_id.reset(token)


def current_request_id() -> str:
    return _request_id.get()


class RequestContextFilter(logging.Filter):
    """Stamp the active request id on every record so formatters can print it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def silence_third_party_noise() -> None:
    """ChromaDB 0.5.x prints telemetry callback errors that look like real failures."""
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)


_installed_handler: logging.Handler | None = None


def configure_logging(
    level: str | int | None = None,
    fmt: str | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """Install one root handler for this process. A later call swaps it, and only it.

    Handlers other code attached to the root logger (pytest capture, uvicorn) stay in
    place, so this can run inside a test or under a server without eating log records.
    """
    global _installed_handler
    resolved_level = level if level is not None else settings.log_level
    resolved_fmt = fmt or settings.log_format
    if handler is None:
        handler = logging.StreamHandler()
    if resolved_fmt == "json":
        handler.setFormatter(JsonFormatter())
    elif handler.formatter is None:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s")
        )
    handler.addFilter(RequestContextFilter())
    root = logging.getLogger()
    if _installed_handler is not None:
        root.removeHandler(_installed_handler)
    root.addHandler(handler)
    root.setLevel(resolved_level)
    _installed_handler = handler
    silence_third_party_noise()


def unknown_env_keys(environ: Mapping[str, str] | None = None) -> list[str]:
    """Return KB_ARENA_ variables that no settings field consumes, so typos surface."""
    env = os.environ if environ is None else environ
    known = {f"{_ENV_PREFIX}{name.upper()}" for name in Settings.model_fields}
    return sorted(key for key in env if key.startswith(_ENV_PREFIX) and key not in known)


def warn_unknown_env(logger: logging.Logger | None = None) -> list[str]:
    log = logger or logging.getLogger("kb_arena.settings")
    unknown = unknown_env_keys()
    for key in unknown:
        log.warning("Ignoring unknown environment variable %s (no matching setting)", key)
    return unknown
