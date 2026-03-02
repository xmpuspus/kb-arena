"""API request/response models for the chatbot and benchmark endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kb_arena.models.graph import GraphContext


class Message(BaseModel):
    """A single chat message."""

    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""

    query: str
    strategy: str = "hybrid"
    history: list[Message] = Field(default_factory=list)
    corpus: str = "python-stdlib"


class ChatResponse(BaseModel):
    """Non-streaming response for /chat endpoint."""

    answer: str
    strategy_used: str
    sources: list[str] = Field(default_factory=list)
    graph_context: GraphContext | None = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0


class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Consistent error envelope (paper-trail-ph pattern)."""

    error: ErrorDetail
