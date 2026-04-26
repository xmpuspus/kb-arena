"""Retrieval trace models — chunk-level visibility for IR metrics and UI drill-down.

Every Strategy.query() result can attach a RetrievalTrace exposing exactly which
chunks surfaced, with rank, score, and source strategy. This is what powers IR
metrics and the /retriever-lab drill-down view.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single chunk surfaced by a retrieval strategy."""

    chunk_id: str
    doc_id: str
    content: str
    score: float = 0.0
    rank: int = Field(ge=1)
    source_strategy: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    """Complete retrieval result for a single query against a single strategy."""

    query: str
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    top_k: int = 5
