"""FastAPI chatbot API — SSE streaming, strategy routing, health check.

Lifespan pattern from paper-trail-ph: init services on startup, store on app.state.
Neo4j unavailability is handled gracefully — strategy falls back to mock data.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from kb_arena.models.api import ChatRequest, ChatResponse, ErrorDetail, ErrorResponse
from kb_arena.settings import settings

logger = logging.getLogger(__name__)

# Simple in-memory rate limiter (requests per minute per IP)
_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_RPM = 60


def _check_rate_limit(client_ip: str) -> bool:
    """Returns True if request is allowed, False if rate limited."""
    now = time.time()
    window = 60.0
    calls = _rate_store[client_ip]
    # Evict calls outside the window
    _rate_store[client_ip] = [t for t in calls if now - t < window]
    if len(_rate_store[client_ip]) >= RATE_LIMIT_RPM:
        return False
    _rate_store[client_ip].append(now)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services. Store on app.state. (Pattern 11 from PLAN.md)"""
    from kb_arena.chatbot.router import IntentRouter
    from kb_arena.llm.client import LLMClient
    from kb_arena.strategies.contextual_vector import ContextualVectorStrategy
    from kb_arena.strategies.hybrid import HybridStrategy
    from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy
    from kb_arena.strategies.naive_vector import NaiveVectorStrategy
    from kb_arena.strategies.qna_pairs import QnAPairStrategy

    # LLM client (shared across strategies)
    llm = LLMClient()
    app.state.llm = llm

    # Neo4j — fail gracefully (Pattern 15: mock fallback)
    app.state.neo4j = None
    try:
        import neo4j

        driver = neo4j.AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await driver.verify_connectivity()
        app.state.neo4j = driver
        logger.info("Neo4j connected at %s", settings.neo4j_uri)
    except Exception as exc:
        logger.warning("Neo4j not available (%s) — graph strategy will use mock data", exc)

    # ChromaDB (always available — local file)
    import chromadb

    chroma = chromadb.PersistentClient(path=settings.chroma_path)
    app.state.chroma = chroma

    # Intent router
    router = IntentRouter(llm=llm)
    app.state.router = router

    # Strategy map (Pattern 11)
    app.state.strategies = {
        "naive_vector": NaiveVectorStrategy(chroma_client=chroma),
        "contextual_vector": ContextualVectorStrategy(chroma_client=chroma),
        "qna_pairs": QnAPairStrategy(chroma_client=chroma, llm_client=llm),
        "knowledge_graph": KnowledgeGraphStrategy(neo4j_driver=app.state.neo4j),
        "hybrid": HybridStrategy(
            neo4j_driver=app.state.neo4j,
            chroma_client=chroma,
            router=router,
        ),
    }

    yield

    if app.state.neo4j is not None:
        await app.state.neo4j.close()


app = FastAPI(
    title="KB Arena API",
    description="Benchmark 5 retrieval strategies on real documentation.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow configurable origins, never wildcard in production
_cors_origins = getattr(settings, "cors_origins", None) or ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Consistent error envelope (Pattern 14 from PLAN.md)."""
    message = str(exc) if settings.debug else "An internal error occurred"
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(code="internal_error", message=message)
        ).model_dump(),
    )


def _resolve_strategy(strategy_name: str, request: Request):
    strategies = request.app.state.strategies
    if strategy_name not in strategies:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code="unknown_strategy",
                    message=f"Unknown strategy '{strategy_name}'. Available: {list(strategies)}",
                )
            ).model_dump(),
        )
    return strategies[strategy_name]


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Non-streaming answer from the requested strategy."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    strategy = _resolve_strategy(body.strategy, request)
    history = [{"role": m.role, "content": m.content} for m in body.history]

    result = await strategy.query(body.query, top_k=5)
    return ChatResponse(
        answer=result.answer,
        strategy_used=result.strategy,
        sources=result.sources,
        graph_context=result.graph_context,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
        cost_usd=result.cost_usd,
    )


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> EventSourceResponse:
    """SSE streaming with 4 event types (Pattern 10 from PLAN.md).

    Events: message_id → token* → done (sources + graph_context) → meta (timing)
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    strategy = _resolve_strategy(body.strategy, request)
    history = [{"role": m.role, "content": m.content} for m in body.history]

    async def event_generator() -> AsyncIterator[dict]:
        msg_id = str(uuid4())
        yield {"event": "message_id", "data": json.dumps({"id": msg_id})}

        try:
            async for token in strategy.stream_answer(body.query, history):
                yield {"event": "token", "data": json.dumps({"text": token})}
        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"code": "stream_error", "message": str(exc)}),
            }
            return

        graph_ctx = strategy.last_graph_context
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "sources": strategy.last_sources,
                    "graph_context": graph_ctx.model_dump() if graph_ctx else None,
                    "strategy_used": strategy.name,
                }
            ),
        }

        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "latency_ms": strategy.last_latency_ms,
                    "tokens_used": strategy.last_tokens_used,
                    "cost_usd": strategy.last_cost_usd,
                }
            ),
        }

    return EventSourceResponse(event_generator())


@app.get("/strategies")
async def list_strategies(request: Request) -> dict:
    """List available strategy names."""
    return {"strategies": list(request.app.state.strategies.keys())}


@app.get("/health")
async def health(request: Request) -> dict:
    """Health check — reports Neo4j connectivity status."""
    neo4j_ok = request.app.state.neo4j is not None
    return {
        "status": "ok",
        "neo4j": "connected" if neo4j_ok else "unavailable (mock mode)",
        "strategies": list(request.app.state.strategies.keys()),
    }
