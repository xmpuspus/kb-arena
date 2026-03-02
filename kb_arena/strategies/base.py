"""Abstract base strategy — all 5 retrieval strategies implement this interface.

The benchmark runner and chatbot API only interact via this interface.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import AsyncIterator

from pydantic import BaseModel, Field

from kb_arena.models.document import Document
from kb_arena.models.graph import GraphContext


class AnswerResult(BaseModel):
    """Unified answer result from any strategy."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    graph_context: GraphContext | None = None
    strategy: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    mock: bool = False


class Strategy(ABC):
    """Abstract base for retrieval strategies.

    Every strategy must implement build_index() and query().
    Optionally implement stream_answer() for SSE streaming.
    """

    name: str = "base"

    # Tracking fields populated after each query
    last_sources: list[str] = []
    last_graph_context: GraphContext | None = None
    last_latency_ms: float = 0.0
    last_tokens_used: int = 0
    last_cost_usd: float = 0.0

    @abstractmethod
    async def build_index(self, documents: list[Document]) -> None:
        """Build the retrieval index from parsed documents.

        Called during setup, before any queries.
        """

    @abstractmethod
    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        """Answer a question using this strategy's retrieval approach.

        Returns a structured AnswerResult with answer, sources, metrics.
        """

    async def stream_answer(
        self, question: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Stream answer tokens. Default: call query() and yield full answer."""
        result = await self.query(question)
        yield result.answer

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
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_latency_ms = elapsed_ms
        self.last_tokens_used = tokens
        self.last_cost_usd = cost
        self.last_sources = sources or []
        self.last_graph_context = graph_context
        return elapsed_ms
