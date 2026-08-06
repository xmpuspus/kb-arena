"""Abstract base strategy — all retrieval strategies implement this interface.

The benchmark runner and chatbot API only interact via this interface.

Per-call state contract
-----------------------
Strategy instances are SHARED across concurrent HTTP requests. Anything attached
to `self` is visible to every in-flight request. To prevent a cross-tenant leak,
all per-call metrics travel two ways:

* `query()` returns an `AnswerResult` carrying answer, sources, graph_context,
  latency, tokens, cost, and the `RetrievalTrace`. Read these from the return
  value, not from the instance.
* `stream_answer()` yields tokens (str), then optionally yields ONE final
  `{"_kb_arena_meta": {...}}` dict so the SSE consumer can emit `done` + `meta`
  events with per-call values.

The legacy `last_*` attributes still exist so existing tests and plugins keep
compiling, but they MUST NOT be read from the chat/SSE path. The chatbot API
consumes only the meta packet yielded by stream_answer.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from kb_arena.models.document import Document
from kb_arena.models.graph import GraphContext
from kb_arena.models.retrieval import RetrievalTrace

# Bound fanout strategies before they allocate or request an amplified candidate set.
MAX_RETRIEVAL_CANDIDATES = 1000


class AnswerResult(BaseModel):
    """Unified answer result from any strategy."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    graph_context: GraphContext | None = None
    retrieval: RetrievalTrace | None = None
    strategy: str = ""
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    mock: bool = False


def meta_packet(result: AnswerResult) -> dict[str, Any]:
    """Build the final SSE meta packet from an AnswerResult.

    Streaming strategies should `yield {"_kb_arena_meta": meta_packet(result)}`
    once after the last token. The chatbot API consumes this and emits the
    SSE `done` + `meta` events.
    """
    return {
        "_kb_arena_meta": {
            "sources": result.sources,
            "graph_context": (result.graph_context.model_dump() if result.graph_context else None),
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
            "cost_usd": result.cost_usd,
        }
    }


class Strategy(ABC):
    """Abstract base for retrieval strategies.

    Every strategy must implement build_index() and query().
    Optionally implement stream_answer() for SSE streaming.
    """

    name: str = "base"

    def __init__(self) -> None:
        # Legacy per-instance fields — DO NOT read across concurrent requests.
        # Kept for backward compatibility with plugins; the chatbot SSE path
        # uses meta_packet() instead.
        self.last_sources: list[str] = []
        self.last_graph_context: GraphContext | None = None
        self.last_latency_ms: float = 0.0
        self.last_tokens_used: int = 0
        self.last_cost_usd: float = 0.0

    @abstractmethod
    async def build_index(self, documents: list[Document]) -> None:
        """Build the retrieval index from parsed documents.

        Called during setup, before any queries.
        """

    @abstractmethod
    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        """Answer a question using this strategy's retrieval approach.

        Returns a structured AnswerResult with answer, sources, metrics.
        """

    async def stream_answer(
        self,
        question: str,
        history: list[dict] | None = None,
        corpus: str = "all",
    ) -> AsyncIterator[str | dict]:
        """Stream answer tokens. Default: call query() and yield full answer + meta."""
        result = await self.query(question, corpus=corpus)
        yield result.answer
        yield meta_packet(result)

    def _start_timer(self) -> float:
        return time.perf_counter()

    def _record_metrics(
        self,
        start: float,
        tokens: int = 0,
        cost: float = 0.0,
        sources: list[str] | None = None,
        graph_context: GraphContext | None = None,
    ) -> float:
        """Record metrics on the instance (legacy path) and return elapsed ms.

        Note: the instance-level fields are not safe to read from concurrent SSE
        consumers — see module docstring. They remain populated only for tests
        and benchmark code that reads them on the same task that called this.
        """
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_latency_ms = elapsed_ms
        self.last_tokens_used = tokens
        self.last_cost_usd = cost
        self.last_sources = sources or []
        self.last_graph_context = graph_context
        return elapsed_ms
