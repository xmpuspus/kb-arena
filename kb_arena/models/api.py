"""API request/response models for the chatbot and benchmark endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from kb_arena.models.graph import GraphContext

MAX_QUERY_LEN = 4000
# Cap chat history at ~25 user+assistant turn pairs (50 messages). Existing
# tests assert 20 full turns are accepted; well above any realistic UI window.
MAX_HISTORY_TURNS = 50
# Past messages can be longer than a single new query (e.g. a previously-pasted
# document) — cap them generously but not unbounded.
MAX_MESSAGE_CONTENT_LEN = 32_000


class Message(BaseModel):
    """A single chat message."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_MESSAGE_CONTENT_LEN)


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LEN)
    strategy: str = Field(default="hybrid", max_length=64)
    history: list[Message] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    corpus: str = Field(default="aws-compute", max_length=64)

    @field_validator("corpus", "strategy")
    @classmethod
    def _validate_identifier(cls, v: str) -> str:
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Invalid identifier: must contain only letters, digits, hyphens, underscores"
            )
        return v


class ArenaMatchRequest(BaseModel):
    """Request body for /api/arena/match."""

    question: str = Field(min_length=1, max_length=MAX_QUERY_LEN)
    corpus: str = Field(default="aws-compute", max_length=64)
    # The rubric the voter judges by. Ratings never cross a rubric.
    rubric: str = Field(default="default", max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")

    @field_validator("corpus")
    @classmethod
    def _validate_corpus(cls, v: str) -> str:
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Invalid corpus name")
        return v


class ArenaVoteRequest(BaseModel):
    """Request body for /api/arena/vote."""

    match_id: str = Field(min_length=1, max_length=64)
    winner: Literal["a", "b", "tie"]
    # Who voted. "human" is the arena page. A named reviewer names itself.
    voter: str = Field(default="human", min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")


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
    # Echo of the X-Request-ID header, so a client can quote the log line that explains it.
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Consistent error envelope (paper-trail-ph pattern)."""

    error: ErrorDetail
