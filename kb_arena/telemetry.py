"""Optional OpenTelemetry spans for retrieval, embedding, and judge calls.

Enabled by KB_ARENA_OTEL_ENABLED. The opentelemetry SDK is imported only
inside this module, and only when tracing is on, so the core install and
every code path work with the `otel` extra absent. When the extra is
missing but tracing is on, `traced_span` logs one warning and runs the
wrapped code plainly, instead of failing the caller.

A span carries structural attributes only: strategy name, corpus, top_k,
and a provider name. Never pass question text, document text, or an API
key here.

This module never configures a TracerProvider or an exporter itself. The
host process does that, for example with `opentelemetry-instrument` and
OTEL_EXPORTER_OTLP_ENDPOINT; kb_arena only opens spans on whatever tracer
provider is already registered.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from kb_arena.settings import settings

logger = logging.getLogger(__name__)

_TRACER_NAME = "kb_arena"


def _tracer_provider() -> Any:
    """Return the registered OpenTelemetry tracer provider.

    A thin indirection so a test can supply its own provider (an in-memory
    exporter) without touching global OpenTelemetry state.
    """
    from opentelemetry import trace

    return trace.get_tracer_provider()


@contextmanager
def traced_span(name: str, **attributes: Any) -> Iterator[None]:
    """Open a span named `name` when tracing is on, else run the block plainly.

    A `None` attribute value is skipped, so a caller can pass an optional
    field such as top_k without a branch.
    """
    if not settings.otel_enabled:
        yield
        return
    try:
        provider = _tracer_provider()
    except ImportError:
        logger.warning(
            "KB_ARENA_OTEL_ENABLED is set but opentelemetry is not installed. "
            "Install the 'otel' extra or unset KB_ARENA_OTEL_ENABLED."
        )
        yield
        return
    tracer = provider.get_tracer(_TRACER_NAME)
    # record_exception and set_status_on_exception default to True and would
    # copy str(exc) onto the span. A provider's error can echo the request
    # body, so that text can carry the question or the document it embedded.
    # A failure still marks the span ERROR; it just carries no message.
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield
        except BaseException:
            from opentelemetry import trace

            span.set_status(trace.StatusCode.ERROR)
            raise
