"""FastAPI chatbot API: SSE streaming, strategy routing, health check.

Lifespan pattern from paper-trail-ph: init services on startup, store on app.state.
Neo4j unavailability is handled gracefully; the strategy falls back to mock data.
"""

from __future__ import annotations

import asyncio as _asyncio
import importlib.resources as _pkg_resources
import json
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path as _Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse

from kb_arena import __version__
from kb_arena.arena.engine import ArenaEngine, scope_key
from kb_arena.benchmark.compare import compare_result_files, resolve_result_path
from kb_arena.benchmark.manifest import compatibility_key, manifest_summary
from kb_arena.chatbot.auth import require_auth
from kb_arena.chatbot.session import SessionStore
from kb_arena.chatbot.tools_api import router as tools_router
from kb_arena.exceptions import ArenaError, StrategyError
from kb_arena.logging_config import bind_request_id, normalize_request_id, reset_request_id
from kb_arena.models.api import (
    ArenaMatchRequest,
    ArenaVoteRequest,
    ChatRequest,
    ChatResponse,
    ErrorDetail,
    ErrorResponse,
)
from kb_arena.settings import settings

_REQUEST_ID_HEADER = "X-Request-ID"

# Per-build UUID queues for streaming graph build events to SSE clients.
# Keyed by build_id (not corpus) so concurrent builds for the same corpus don't collide.
_graph_build_queues: dict[str, _asyncio.Queue] = {}
_graph_build_tasks: dict[str, _asyncio.Task[None]] = {}
_graph_build_subscribers: set[str] = set()
_GRAPH_BUILD_QUEUE_MAX_EVENTS = 1000
_GRAPH_BUILD_QUEUE_TTL_SECONDS = 300.0
_GRAPH_BUILD_MAX_ACTIVE = 4
_GRAPH_BUILD_TIMEOUT_SECONDS = 1800.0


def _enqueue_graph_build_event(queue: _asyncio.Queue, event: dict | None) -> None:
    """Add an event without allowing an unattended build queue to grow forever."""
    if queue.full():
        try:
            queue.get_nowait()
        except _asyncio.QueueEmpty:
            pass
    queue.put_nowait(event)


def _expire_graph_build(build_id: str) -> None:
    _graph_build_queues.pop(build_id, None)
    _graph_build_subscribers.discard(build_id)


class _GraphBuildRequest(BaseModel):
    corpus: str

    @field_validator("corpus")
    @classmethod
    def _validate(cls, v: str) -> str:
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Invalid corpus name: must contain only letters, digits, hyphens, underscores"
            )
        return v


logger = logging.getLogger(__name__)

# Per-session memory for multi-turn conversations (TTL-based eviction)
_session_store = SessionStore(ttl_minutes=settings.session_ttl_minutes)

# Back-compat re-exports for tests that imported the old rate-limiter store
# directly. The active store now lives in kb_arena.chatbot.auth.
from kb_arena.chatbot import auth as _auth_module  # noqa: E402

_rate_store = _auth_module._rate_store
RATE_LIMIT_RPM = _auth_module.RATE_LIMIT_RPM


def _public_error_message(exc: Exception) -> str:
    """Expose exception details only in explicitly enabled debug mode."""
    return str(exc) if settings.debug else "An internal error occurred"


def _current_request_id() -> str:
    from kb_arena.logging_config import current_request_id

    return current_request_id()


def _request_id_for(request: Request | None = None) -> str:
    """Read the id set by RequestIDMiddleware.

    Prefers ``request.state``, which the middleware's outer ServerErrorMiddleware
    layer can still read after an exception unwinds this middleware's own
    ``finally`` block. Falls back to the context var for code with no request.
    """
    if request is not None:
        state_id = getattr(request.state, "request_id", "")
        if state_id:
            return state_id
    return _current_request_id()


def _error_detail(code: str, exc: Exception, request: Request | None = None) -> ErrorDetail:
    """Build one error body shape for every failure path: message plus request id."""
    return ErrorDetail(
        code=code, message=_public_error_message(exc), request_id=_request_id_for(request) or None
    )


class RequestIDMiddleware:
    """Bind a request id for the whole request, echo it on the response.

    Plain ASGI, not ``BaseHTTPMiddleware``: that class buffers the response
    through its own task group, which breaks ``sse_starlette`` streaming and
    raises a cross-event-loop error under the test client. This wraps only
    ``send`` to add one header, so a streamed body passes through untouched.

    A client-sent id passes through only when it matches a safe pattern, so a
    request id can never carry attacker-controlled text into the log format.
    """

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        client_id = headers.get(_REQUEST_ID_HEADER.lower().encode()) or b""
        request_id = normalize_request_id(client_id.decode("latin-1") or None)
        # ServerErrorMiddleware wraps this middleware and calls the registered
        # Exception handler AFTER this call returns, once the exception has
        # already unwound through the `finally` below and reset the context
        # var. Scope state outlives that unwind, so the handler reads from
        # there; bind_request_id keeps the context var for everything that
        # runs before the reset, and for log lines with no Request object.
        scope.setdefault("state", {})["request_id"] = request_id
        token = bind_request_id(request_id)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message["headers"] = [
                    *message.get("headers", []),
                    (_REQUEST_ID_HEADER.encode(), request_id.encode()),
                ]
            await send(message)

        try:
            await self._app(scope, receive, send_with_id)
        finally:
            reset_request_id(token)


def _generation_configured() -> bool:
    """Return whether the selected generation provider has usable credentials."""
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return True
    if provider == "anthropic":
        return bool(settings.llm_api_key or settings.anthropic_api_key)
    if provider == "openai":
        return bool(settings.llm_api_key or settings.openai_api_key)
    return False


def _check_rate_limit(client_ip: str) -> bool:
    """Back-compat shim for old tests. Return True when a request is allowed.

    Tolerates plain-list assignment (`_rate_store[ip] = [...]`) used by older
    audit tests, even though the production store is a deque.
    """
    return _auth_module._consume_rate_limit(client_ip)


def _build_arena(strategies: dict) -> tuple[ArenaEngine | None, str]:
    """Build the arena, and report why when it cannot be built.

    A broken arena must not stop the rest of the API from serving, so the failure
    ends as None either way. The log line and the returned reason are what tell a
    broken arena apart from an absent one, on the log and on /health.
    """
    try:
        return ArenaEngine(strategies), ""
    except Exception as exc:
        # /health needs no token, so the reason it publishes follows the same
        # redaction rule as every other error surface. The log above keeps the
        # full traceback for whoever operates the deployment.
        logger.exception("Arena engine failed to initialize; arena endpoints return 503")
        return None, _public_error_message(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services. Store on app.state. (Pattern 11 from PLAN.md)"""
    from kb_arena.chatbot.router import IntentRouter
    from kb_arena.llm.client import LLMClient
    from kb_arena.strategies.bm25 import BM25Strategy
    from kb_arena.strategies.contextual_vector import ContextualVectorStrategy
    from kb_arena.strategies.hybrid import HybridStrategy
    from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy
    from kb_arena.strategies.naive_vector import NaiveVectorStrategy
    from kb_arena.strategies.pageindex import PageIndexStrategy
    from kb_arena.strategies.qna_pairs import QnAPairStrategy
    from kb_arena.strategies.quantum.qiss import QISSStrategy
    from kb_arena.strategies.raptor import RaptorStrategy
    from kb_arena.strategies.rerank_vector import RerankVectorStrategy

    # Detect zero-config demo: if no API keys are configured AND we're not on Ollama,
    # auto-enable demo_mode so /chat etc. return 503 instead of crashing on the
    # first request. The static dashboard, /api/benchmark/results, /api/corpora,
    # and the public arena/leaderboard read-only endpoints all keep working.
    if settings.llm_provider.lower() not in {"anthropic", "openai", "ollama"}:
        raise ValueError(
            f"Unknown KB_ARENA_LLM_PROVIDER={settings.llm_provider!r}. "
            "Valid: anthropic, ollama, openai."
        )
    if not _generation_configured():
        if not settings.demo_mode:
            logger.info(
                "No API key configured; auto-enabling KB_ARENA_DEMO_MODE. "
                "Static benchmark/leaderboard pages remain available; "
                "chat/arena/tools endpoints return 503 until a key is set."
            )
            settings.demo_mode = True

    # The read-only demo does not need a model client. Configured deployments
    # share one client across strategies; initialization failures stop startup.
    llm: LLMClient | None
    if settings.demo_mode:
        llm = None
        logger.info("LLM client skipped in demo mode")
    else:
        llm = LLMClient()
    app.state.llm = llm

    # Neo4j: the read-only demo does not use live graph queries.
    app.state.neo4j = None
    app.state.neo4j_error = ""
    if settings.demo_mode:
        app.state.neo4j_error = "disabled in demo mode"
        logger.info("Neo4j connection skipped in demo mode")
    else:
        try:
            import neo4j
        except ImportError:
            logger.warning("Neo4j driver not installed; graph strategy will use mock data")
            app.state.neo4j_error = "neo4j driver not installed"
            neo4j = None  # type: ignore[assignment]

        if neo4j is not None:
            driver = None
            try:
                driver = neo4j.AsyncGraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, settings.neo4j_password),
                )
                async with driver.session(database=settings.neo4j_database) as session:
                    result = await session.run("RETURN 1")
                    await result.consume()
                app.state.neo4j = driver
                logger.info("Neo4j connected at %s", settings.neo4j_uri)
            except (OSError, neo4j.exceptions.GqlError) as exc:
                if driver is not None:
                    await driver.close()
                app.state.neo4j_error = str(exc)
                logger.warning(
                    "Neo4j not available at %s (%s); knowledge_graph and hybrid will use mock "
                    "data. Run: docker compose up neo4j -d",
                    settings.neo4j_uri,
                    exc,
                )
            except BaseException:
                if driver is not None:
                    await driver.close()
                raise

    # ChromaDB 0.5.x can emit telemetry callback errors even when the client is local.
    import os

    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

    import chromadb

    chroma = chromadb.PersistentClient(path=settings.chroma_path)
    app.state.chroma = chroma

    # When the LLM is missing, hybrid routing falls back to keyword rules.
    router = IntentRouter(llm=llm) if llm is not None else None
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
            llm=llm,
        ),
        "raptor": RaptorStrategy(chroma_client=chroma),
        "pageindex": PageIndexStrategy(),
        "bm25": BM25Strategy(),
        "qiss": QISSStrategy(chroma_client=chroma, llm_client=llm),
    }

    from kb_arena.strategies.catalog import STRATEGY_CATALOG, missing_optional_modules

    rerank_spec = next(spec for spec in STRATEGY_CATALOG if spec.name == "rerank_vector")
    if not missing_optional_modules(rerank_spec):
        app.state.strategies["rerank_vector"] = RerankVectorStrategy(
            chroma_client=chroma, llm_client=llm
        )

    sqr_spec = next(spec for spec in STRATEGY_CATALOG if spec.name == "sqr")
    if not missing_optional_modules(sqr_spec):
        from kb_arena.strategies.quantum.sqr import SQRStrategy

        app.state.strategies["sqr"] = SQRStrategy(chroma_client=chroma, llm_client=llm)

    app.state.arena, app.state.arena_error = _build_arena(app.state.strategies)

    # Periodic session cleanup task
    async def _cleanup_sessions():
        while True:
            await _asyncio.sleep(300)  # every 5 minutes
            evicted = _session_store.cleanup()
            if evicted:
                logger.debug("Evicted %d expired sessions", evicted)

    cleanup_task = _asyncio.create_task(_cleanup_sessions())

    yield

    cleanup_task.cancel()
    if app.state.neo4j is not None:
        await app.state.neo4j.close()


def _docs_urls(enabled: bool) -> tuple[str | None, str | None, str | None]:
    """Resolve the three FastAPI doc routes from one setting, so a private
    deployment can drop them without three separate flags to keep in sync."""
    if not enabled:
        return None, None, None
    return "/docs", "/redoc", "/openapi.json"


def _resolve_docs_enabled(explicit: bool | None, debug: bool) -> bool:
    """Use the explicit setting when an operator gave one, else follow debug.

    The closed default means a production deployment that never sets
    KB_ARENA_API_DOCS_ENABLED and never turns on debug serves no /docs,
    /redoc, or /openapi.json.
    """
    return debug if explicit is None else explicit


_docs_url, _redoc_url, _openapi_url = _docs_urls(
    _resolve_docs_enabled(settings.api_docs_enabled, settings.debug)
)

app = FastAPI(
    title="KB Arena API",
    description="Compare retrieval architectures on your documentation with recorded evidence.",
    version=__version__,
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
)

# CORS is configurable via KB_ARENA_CORS_ORIGINS and defaults to localhost dev ports.
_cors_origins = settings.cors_origins or [
    "http://localhost:3000",
    "http://localhost:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(RequestIDMiddleware)

app.include_router(tools_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Consistent error envelope (Pattern 14 from PLAN.md), logged with the request id."""
    logger.exception(
        "Unhandled exception on %s %s [request_id=%s]",
        request.method,
        request.url.path,
        _request_id_for(request),
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=_error_detail("internal_error", exc, request)).model_dump(),
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


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_auth)])
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Non-streaming answer from the requested strategy."""
    strategy = _resolve_strategy(body.strategy, request)

    # Track conversation history per session (X-Session-ID header preferred, IP fallback)
    client_ip = request.client.host if request.client else "unknown"
    session_id = request.headers.get("x-session-id", client_ip)
    session_key = f"{session_id}:{body.corpus}:{body.strategy}"
    session = _session_store.get(session_key)
    session.add_turn("user", body.query)

    result = await strategy.query(body.query, top_k=5, corpus=body.corpus)

    session.add_turn("assistant", result.answer)

    return ChatResponse(
        answer=result.answer,
        strategy_used=result.strategy,
        sources=result.sources,
        graph_context=result.graph_context,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
        cost_usd=result.cost_usd,
    )


@app.post("/chat/stream", dependencies=[Depends(require_auth)])
async def chat_stream(body: ChatRequest, request: Request) -> EventSourceResponse:
    """SSE streaming with 4 event types (Pattern 10 from PLAN.md).

    Events: message_id, token*, done (sources + graph_context), meta (timing)

    The done/meta payloads come from the per-call `RetrievalTrace` and metrics that
    `stream_answer` records on the result side, not from shared instance state. See
    Strategy.stream_answer for the per-call snapshot.
    """
    strategy = _resolve_strategy(body.strategy, request)
    history = [{"role": m.role, "content": m.content} for m in body.history]

    async def event_generator() -> AsyncIterator[dict]:
        msg_id = str(uuid4())
        yield {"event": "message_id", "data": json.dumps({"id": msg_id})}

        snapshot: dict | None = None
        try:
            async for token in strategy.stream_answer(
                body.query,
                history,
                corpus=body.corpus,
            ):
                if isinstance(token, dict) and "_kb_arena_meta" in token:
                    # Final metadata packet; see the Strategy.stream_answer protocol.
                    snapshot = token["_kb_arena_meta"]
                    continue
                yield {"event": "token", "data": json.dumps({"text": token})}
        except Exception as exc:
            logger.exception(
                "Chat stream failed for strategy %s [request_id=%s]",
                body.strategy,
                _current_request_id(),
            )
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "code": "stream_error",
                        "message": _public_error_message(exc),
                        "request_id": _current_request_id() or None,
                    }
                ),
            }
            return

        # Snapshot is per-call (no shared state). Falls back to "" / 0 if the
        # strategy didn't emit one (kept for backward compat with any external plugins).
        snapshot = snapshot or {}
        sources = snapshot.get("sources", [])
        graph_ctx = snapshot.get("graph_context")
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "sources": sources,
                    "graph_context": graph_ctx,
                    "strategy_used": strategy.name,
                }
            ),
        }

        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "latency_ms": snapshot.get("latency_ms", 0.0),
                    "tokens_used": snapshot.get("tokens_used", 0),
                    "cost_usd": snapshot.get("cost_usd", 0.0),
                }
            ),
        }

    return EventSourceResponse(event_generator())


@app.get("/api/corpora")
async def list_corpora() -> dict:
    """Discover available corpora with pipeline status from the datasets directory."""
    from pathlib import Path

    datasets_dir = Path(settings.datasets_path)
    results_dir = Path(settings.results_path)
    corpora = []
    if datasets_dir.exists():
        for d in sorted(datasets_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            has_processed = (d / "processed").is_dir() and any((d / "processed").glob("*.jsonl"))
            total_questions = 0
            if (d / "questions").is_dir():
                for qf in (d / "questions").glob("*.yaml"):
                    try:
                        total_questions += qf.read_text().count("- id:")
                    except OSError:
                        pass
            has_results = results_dir.exists() and any(results_dir.glob(f"{d.name}_*.json"))
            qa_path = d / "qa-pairs" / "qa_pairs.jsonl"
            has_qa_pairs = qa_path.exists()
            qa_pair_count = 0
            if has_qa_pairs:
                try:
                    qa_pair_count = sum(
                        1 for line in qa_path.read_text().splitlines() if line.strip()
                    )
                except OSError:
                    pass
            corpora.append(
                {
                    "value": d.name,
                    "label": d.name.replace("-", " ").title(),
                    "questionCount": total_questions,
                    "hasProcessed": has_processed,
                    "hasResults": has_results,
                    "hasQaPairs": has_qa_pairs,
                    "qaPairCount": qa_pair_count,
                }
            )
    return {"corpora": corpora}


@app.get("/api/retriever-lab/runs")
async def retriever_lab_runs() -> dict:
    """List available retriever-lab runs (most recent first)."""
    base = _Path(settings.results_path)
    if not base.exists():
        return {"runs": []}
    runs: list[dict] = []
    for run_dir in sorted(base.glob("run_*"), reverse=True):
        path = run_dir / "retriever_lab.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        run_id = data.get("run_id") or run_dir.name.replace("run_", "")
        runs.append(
            {
                "run_id": run_id,
                "timestamp": data.get("timestamp", ""),
                "top_k": data.get("top_k", 5),
                "corpora": list(data.get("corpora", {}).keys()),
            }
        )
    return {"runs": runs}


@app.get("/api/retriever-lab/{run_id}")
async def retriever_lab_results(run_id: str) -> dict:
    """Return retriever-lab JSON for the given run."""
    import re as _re

    if not _re.match(r"^[a-zA-Z0-9_-]+$", run_id):
        raise HTTPException(status_code=400, detail="invalid run_id")
    path = _Path(settings.results_path) / f"run_{run_id}" / "retriever_lab.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="corrupt run file") from e


@app.get("/api/benchmark/results")
async def benchmark_results(corpus: str = "all") -> dict:
    """Load benchmark results from the results directory."""
    import json
    from pathlib import Path

    results_dir = Path(settings.results_path)
    if not results_dir.exists():
        return {"results": [], "source": "none"}

    all_results = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        if corpus != "all" and data.get("corpus") != corpus:
            continue
        all_results.append(data)

    if not all_results:
        return {"results": [], "source": "none"}

    # Aggregate per-strategy across all loaded result files
    from collections import defaultdict

    strategy_tiers: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    strategy_latency: dict[str, list[float]] = defaultdict(list)
    strategy_cost: dict[str, list[float]] = defaultdict(list)

    for result in all_results:
        strategy = result.get("strategy", "")
        records = result.get("records", [])
        if not isinstance(strategy, str) or not strategy or not isinstance(records, list):
            continue
        parsed_records: list[tuple[int, float, float, float]] = []
        try:
            for index, rec in enumerate(records):
                if not isinstance(rec, dict):
                    raise ValueError(f"records[{index}] must be an object")
                tier = rec.get("question_tier", 0)
                if isinstance(tier, bool) or not isinstance(tier, int):
                    raise ValueError(f"records[{index}].question_tier must be an integer")
                if not tier:
                    question_id = rec.get("question_id", "")
                    if not isinstance(question_id, str):
                        raise ValueError(f"records[{index}].question_id must be a string")
                    try:
                        tier = int(question_id.split("-t")[1].split("-")[0])
                    except (IndexError, ValueError):
                        tier = 0
                score = rec.get("score", {})
                if not isinstance(score, dict):
                    raise ValueError(f"records[{index}].score must be an object")
                parsed_records.append(
                    (
                        tier,
                        _finite_number(score.get("accuracy", 0.0), f"records[{index}].accuracy"),
                        _finite_number(rec.get("latency_ms", 0.0), f"records[{index}].latency_ms"),
                        _finite_number(rec.get("cost_usd", 0.0), f"records[{index}].cost_usd"),
                    )
                )
        except ValueError:
            continue

        for tier, accuracy, latency, cost in parsed_records:
            strategy_tiers[strategy][tier].append(accuracy)
            strategy_latency[strategy].append(latency)
            strategy_cost[strategy].append(cost)

    rows = []
    for strat, tiers in strategy_tiers.items():
        tier_avgs = []
        for t in range(1, 6):
            vals = tiers.get(t, [])
            tier_avgs.append(round(sum(vals) / len(vals) * 100) if vals else 0)
        latencies = strategy_latency[strat]
        costs = strategy_cost[strat]
        rows.append(
            {
                "strategy": strat,
                "tiers": tier_avgs,
                "latencyMs": round(sum(latencies) / len(latencies)) if latencies else 0,
                "costUsd": round(sum(costs) / len(costs), 4) if costs else 0,
            }
        )

    return {"results": rows, "source": "file"}


@app.get("/strategies")
async def list_strategies(request: Request) -> dict:
    """List loaded names and the status of every built-in strategy."""
    from kb_arena.strategies.catalog import public_catalog

    loaded = list(request.app.state.strategies)
    return {"strategies": loaded, "catalog": public_catalog(loaded)}


@app.get("/graph/stats")
async def graph_stats(request: Request) -> dict:
    """Return graph node and edge counts, centrality hubs, and communities."""
    if request.app.state.neo4j is None:
        return {"error": "Neo4j not connected", "stats": None}

    from kb_arena.graph.analyzer import GraphAnalyzer
    from kb_arena.graph.neo4j_store import Neo4jStore

    store = Neo4jStore(request.app.state.neo4j)
    analyzer = GraphAnalyzer(store)

    centrality = await analyzer.calculate_centrality()
    top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    communities = await analyzer.analyze_communities()

    return {
        "node_count": sum(1 for _ in centrality),
        "top_hubs": [
            {
                "entity_id": entity_id,
                "fqn": entity_id.split("::", 1)[-1],
                "centrality": round(centrality_score, 4),
            }
            for entity_id, centrality_score in top_hubs
        ],
        "community_count": len(communities),
    }


@app.get("/api/graph/data")
async def graph_data(
    request: Request,
    corpus: str = "all",
    limit: int = Query(default=200, ge=1, le=500),
) -> dict:
    """Return graph nodes and edges from Neo4j for visualization."""
    if request.app.state.neo4j is None:
        return {"nodes": [], "edges": [], "connected": False}

    driver = request.app.state.neo4j
    # Fetch nodes
    node_query = (
        "MATCH (n:KBArenaEntity) "
        + ("WHERE n.corpus = $corpus " if corpus != "all" else "")
        + "RETURN n.entity_id AS id, n.name AS name, "
        "head([label IN labels(n) WHERE label <> 'KBArenaEntity']) AS type, "
        "n.description AS description LIMIT $limit"
    )
    params = {"limit": limit}
    if corpus != "all":
        params["corpus"] = corpus

    nodes = []
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(node_query, params)
        records = await result.data()
        await result.consume()
        for r in records:
            nodes.append(
                {
                    "id": r["id"] or r["name"],
                    "name": r["name"] or r["id"],
                    "type": r["type"] or "Topic",
                    "description": r.get("description", ""),
                }
            )

    # Fetch edges between those nodes
    node_ids = {n["id"] for n in nodes}
    edges = []
    if node_ids:
        edge_query = (
            "MATCH (a:KBArenaEntity)-[r]->(b:KBArenaEntity) "
            + ("WHERE a.corpus = $corpus " if corpus != "all" else "")
            + "RETURN a.entity_id AS source, type(r) AS type, b.entity_id AS target "
            "LIMIT $edge_limit"
        )
        edge_params = {"edge_limit": limit * 2}
        if corpus != "all":
            edge_params["corpus"] = corpus

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(edge_query, edge_params)
            records = await result.data()
            await result.consume()
            for r in records:
                if r["source"] in node_ids and r["target"] in node_ids:
                    edges.append(
                        {
                            "source": r["source"],
                            "target": r["target"],
                            "type": r["type"],
                        }
                    )

    return {"nodes": nodes, "edges": edges, "connected": True}


@app.post("/api/graph/build", dependencies=[Depends(require_auth)])
async def trigger_graph_build(body: _GraphBuildRequest) -> dict:
    """Trigger graph build for a corpus.

    Returns a unique build_id; stream via /api/graph/build/stream/{build_id}.
    """
    corpus = body.corpus

    # Validate corpus exists and has processed documents
    corpus_dir = _Path(settings.datasets_path) / corpus
    processed_dir = corpus_dir / "processed"
    if not corpus_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Corpus '{corpus}' not found in datasets directory",
        )
    if not processed_dir.is_dir() or not any(processed_dir.glob("*.jsonl")):
        raise HTTPException(
            status_code=400,
            detail=f"Corpus '{corpus}' has no processed documents. Run 'kb-arena ingest' first.",
        )
    if len(_graph_build_tasks) >= _GRAPH_BUILD_MAX_ACTIVE:
        raise HTTPException(
            status_code=429,
            detail=f"Too many graph builds are active. Limit: {_GRAPH_BUILD_MAX_ACTIVE}.",
        )

    build_id = str(uuid4())
    queue: _asyncio.Queue = _asyncio.Queue(maxsize=_GRAPH_BUILD_QUEUE_MAX_EVENTS)
    _graph_build_queues[build_id] = queue

    async def _callback(event: dict | None) -> None:
        _enqueue_graph_build_event(queue, event)

    async def _run() -> None:
        from kb_arena.graph.extractor import run_extraction

        try:
            async with _asyncio.timeout(_GRAPH_BUILD_TIMEOUT_SECONDS):
                await run_extraction(corpus=corpus, event_callback=_callback)
        except TimeoutError:
            _enqueue_graph_build_event(
                queue,
                {
                    "type": "error",
                    "data": {
                        "message": (
                            "Graph build timed out after "
                            f"{_GRAPH_BUILD_TIMEOUT_SECONDS:g} seconds."
                        )
                    },
                },
            )
        except Exception as exc:
            logger.exception(
                "Graph build %s failed for corpus %s [request_id=%s]",
                build_id,
                corpus,
                _current_request_id(),
            )
            _enqueue_graph_build_event(
                queue,
                {
                    "type": "error",
                    "data": {
                        "message": _public_error_message(exc),
                        "request_id": _current_request_id() or None,
                    },
                },
            )
        finally:
            _graph_build_tasks.pop(build_id, None)
            _enqueue_graph_build_event(queue, None)  # Sentinel that signals stream end.
            _asyncio.get_running_loop().call_later(
                _GRAPH_BUILD_QUEUE_TTL_SECONDS,
                _expire_graph_build,
                build_id,
            )

    task = _asyncio.create_task(_run(), name=f"graph_build:{build_id}")
    _graph_build_tasks[build_id] = task
    return {"status": "started", "build_id": build_id, "corpus": corpus}


@app.get(
    "/api/graph/build/stream/{build_id}",
    dependencies=[Depends(require_auth)],
)
async def graph_build_stream(build_id: str) -> EventSourceResponse:
    """SSE stream of graph build events for a specific build."""
    queue = _graph_build_queues.get(build_id)
    if queue is None:

        async def _empty() -> AsyncIterator[dict]:
            # 410 Gone semantics: tell EventSource clients to stop reconnecting.
            yield {
                "event": "error",
                "retry": 99999999,
                "data": json.dumps({"message": "Build not found. POST to /api/graph/build first."}),
            }

        return EventSourceResponse(_empty())

    if build_id in _graph_build_subscribers:

        async def _duplicate() -> AsyncIterator[dict]:
            yield {
                "event": "error",
                "retry": 99999999,
                "data": json.dumps({"message": "Build stream already has a subscriber."}),
            }

        return EventSourceResponse(_duplicate())

    _graph_build_subscribers.add(build_id)

    async def event_generator() -> AsyncIterator[dict]:
        reached_end = False
        try:
            while True:
                try:
                    event = await _asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    yield {"event": "heartbeat", "data": "{}"}
                    continue
                if event is None:  # Sentinel that signals the build is complete.
                    reached_end = True
                    break
                yield {"event": event["type"], "data": json.dumps(event["data"])}
        finally:
            _graph_build_subscribers.discard(build_id)
            if reached_end:
                _graph_build_queues.pop(build_id, None)

    return EventSourceResponse(event_generator())


@app.post("/api/arena/match", dependencies=[Depends(require_auth)])
async def arena_create_match(body: ArenaMatchRequest, request: Request):
    """Create a blind A/B match between two random strategies."""
    arena = request.app.state.arena
    if not arena:
        return JSONResponse(
            {"error": {"code": "arena_unavailable", "message": "Arena not initialized"}}, 503
        )
    try:
        match = await arena.create_match(body.question, corpus=body.corpus, rubric=body.rubric)
        return {
            "match_id": match.id,
            "question": match.question,
            "answer_a": match.answer_a,
            "answer_b": match.answer_b,
            "latency_a_ms": round(match.latency_a_ms, 1),
            "latency_b_ms": round(match.latency_b_ms, 1),
            "sources_a": match.sources_a,
            "sources_b": match.sources_b,
        }
    except (ArenaError, StrategyError) as exc:
        # Two shapes of the same outage: a strategy answered from mock data, or it
        # failed outright. Rating either would record an outage in the leaderboard,
        # so both are unavailability rather than a server fault.
        logger.warning(
            "Arena match unavailable for corpus %s [request_id=%s]: %s",
            body.corpus,
            _current_request_id(),
            exc,
        )
        return JSONResponse({"error": _error_detail("strategy_unavailable", exc).model_dump()}, 503)
    except Exception as exc:
        logger.exception(
            "Arena match failed for corpus %s [request_id=%s]", body.corpus, _current_request_id()
        )
        return JSONResponse({"error": _error_detail("match_failed", exc).model_dump()}, 500)


@app.post("/api/arena/vote", dependencies=[Depends(require_auth)])
async def arena_vote(body: ArenaVoteRequest, request: Request):
    """Vote on an arena match. Body: {match_id, winner: 'a'|'b'|'tie'}."""
    arena = request.app.state.arena
    if not arena:
        return JSONResponse(
            {"error": {"code": "arena_unavailable", "message": "Arena not initialized"}}, 503
        )
    result = arena.vote(body.match_id, body.winner, voter=body.voter or "human")
    if "error" in result:
        return JSONResponse({"error": {"code": "vote_failed", "message": result["error"]}}, 400)
    return result


@app.get("/api/arena/leaderboard")
async def arena_leaderboard(request: Request, corpus: str = "", rubric: str = "default"):
    """The ELO leaderboard for one corpus and rubric. Votes from other scopes never count."""
    arena = request.app.state.arena
    if not arena:
        return {
            "leaderboard": [],
            "total_votes": 0,
            "scope": {"corpus": corpus or "all", "rubric": rubric},
        }
    board = arena.leaderboard(corpus=corpus, rubric=rubric)
    return {
        "leaderboard": board,
        "scope": {"corpus": corpus or "all", "rubric": rubric or "default"},
        "scopes": sorted(arena.state.elo_by_scope),
        # Each match sits on two rows, so count the matches, not the rows.
        "total_votes": sum(
            1
            for m in arena.state.matches
            if m.winner and scope_key(m.corpus, m.rubric) == scope_key(corpus, rubric)
        ),
    }


@app.get("/api/compare")
async def compare_strategies(
    corpus: str, a: str, b: str, run_a: str = "", run_b: str = "", metric: str = "accuracy"
):
    """Pair two strategies question by question. Delta is b minus a."""
    results_dir = _Path(settings.results_path)
    try:
        path_a = resolve_result_path(results_dir, corpus, a, run_a or None)
        path_b = resolve_result_path(results_dir, corpus, b, run_b or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path_a.exists() or not path_b.exists():
        raise HTTPException(status_code=404, detail="result not found")
    try:
        return compare_result_files(path_a, path_b, metric=metric)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"cannot compare: {exc}") from exc


@app.get("/api/leaderboard")
async def leaderboard(request: Request, corpus: str = "all") -> dict:
    """Public read-only leaderboard of all benchmark runs across all corpora.

    Aggregates `results/run_*` JSON files into a per-(corpus, strategy) leaderboard
    with mean accuracy, mean Recall@5, mean NDCG@5, mean cost, and run count.
    This unauthenticated endpoint supplies the hosted demo at kb-arena.dev.
    """
    import json
    from collections import defaultdict
    from pathlib import Path

    base = _Path(settings.results_path)
    if not base.exists():
        return {"corpora": [], "leaderboard": []}

    # Collect (corpus, strategy) -> list[per-run metrics]
    rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    seen_corpora: set[str] = set()
    # The runner writes each result twice, once at the top level and once
    # under its run directory. One run counts once.
    seen_runs: set[tuple[str, str, str]] = set()

    def _first_sighting(c: str, s: str, data: dict) -> bool:
        run_id = data.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return True
        if (c, s, run_id) in seen_runs:
            return False
        seen_runs.add((c, s, run_id))
        return True

    # Top-level files (legacy single-run shape)
    for path in sorted(base.glob("*.json")):
        if path.name == "arena_state.json":
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        c = data.get("corpus") or path.stem.split("_")[0]
        s = data.get("strategy") or path.stem.split("_", 1)[-1]
        if not isinstance(c, str) or not c or not isinstance(s, str) or not s:
            continue
        seen_corpora.add(c)
        if corpus != "all" and c != corpus:
            continue
        try:
            summary = _summarise_run(data)
        except ValueError:
            continue
        # A run counts as seen only once it summarised, so a bad top-level
        # copy never hides the good copy under its run directory.
        if not _first_sighting(c, s, data):
            continue
        rows[(c, s, compatibility_key(data))].append(summary)

    # New per-run subdirectories (results/run_<id>/<corpus>_<strategy>.json)
    for run_dir in sorted(base.glob("run_*")):
        for path in sorted(Path(run_dir).glob("*.json")):
            if path.name in {"retriever_lab.json", "report.md"}:
                continue
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            c = data.get("corpus") or path.stem.split("_")[0]
            s = data.get("strategy") or path.stem.split("_", 1)[-1]
            if not isinstance(c, str) or not c or not isinstance(s, str) or not s:
                continue
            seen_corpora.add(c)
            if corpus != "all" and c != corpus:
                continue
            try:
                summary = _summarise_run(data)
            except ValueError:
                continue
            if not _first_sighting(c, s, data):
                continue
            rows[(c, s, compatibility_key(data))].append(summary)

    # Runs made against different question sets, qrels, judges, or top_k
    # values never share a row. A row names its key, and lists the other keys
    # seen for the same corpus and strategy, so a reader can tell two
    # incomparable rows apart instead of reading one blended number.
    keys_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c, s, key in rows:
        keys_by_pair[(c, s)].append(key)

    leaderboard: list[dict] = []
    for (c, s, key), runs in sorted(rows.items()):
        if not runs:
            continue
        leaderboard.append(
            {
                "corpus": c,
                "strategy": s,
                "compatibility_key": key,
                "manifest": runs[0].get("manifest", {}),
                "mixed_with": sorted(k for k in keys_by_pair[(c, s)] if k != key),
                "runs": len(runs),
                "mean_accuracy": _avg(runs, "overall_accuracy"),
                "mean_recall_at_5": _avg(runs, "mean_recall_at_k"),
                "mean_ndcg_at_5": _avg(runs, "mean_ndcg_at_k"),
                "mean_cost_usd": _avg(runs, "total_cost_usd"),
                "mean_latency_ms": _avg(runs, "mean_latency_ms"),
            }
        )

    leaderboard.sort(key=lambda r: (r["corpus"], -(r["mean_accuracy"] or 0.0), r["strategy"]))

    return {
        "corpora": sorted(seen_corpora),
        "leaderboard": leaderboard,
        "filter": corpus,
    }


def _finite_number(value: object, field: str) -> float:
    """Return a finite real value without accepting Python's Boolean integers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _summarise_run(data: dict) -> dict:
    """Pull the leaderboard-relevant fields out of a benchmark JSON, tolerantly."""

    records = data.get("records", [])
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("records must be a list of objects")

    overall = data.get("overall_accuracy")
    if overall is None:
        # Older shape: derive from records.
        if records:
            accuracies = []
            for index, record in enumerate(records):
                score = record.get("score", {})
                if not isinstance(score, dict):
                    raise ValueError(f"records[{index}].score must be an object")
                accuracies.append(
                    _finite_number(score.get("accuracy", 0.0), f"records[{index}].score.accuracy")
                )
            overall = sum(accuracies) / len(accuracies)
        else:
            overall = 0.0
    else:
        overall = _finite_number(overall, "overall_accuracy")

    explicit_cost = data.get("total_cost_usd")
    if explicit_cost is not None:
        cost = _finite_number(explicit_cost, "total_cost_usd")
    else:
        cost = sum(
            _finite_number(record.get("cost_usd", 0.0), f"records[{index}].cost_usd")
            for index, record in enumerate(records)
        )
    latencies = [
        _finite_number(record.get("latency_ms", 0.0), f"records[{index}].latency_ms")
        for index, record in enumerate(records)
    ]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "manifest": manifest_summary(data),
        "overall_accuracy": overall,
        "mean_recall_at_k": _finite_number(data.get("mean_recall_at_k", 0.0), "mean_recall_at_k"),
        "mean_ndcg_at_k": _finite_number(data.get("mean_ndcg_at_k", 0.0), "mean_ndcg_at_k"),
        "total_cost_usd": cost,
        "mean_latency_ms": mean_latency,
    }


def _avg(items: list[dict], key: str) -> float | None:
    values = [it.get(key) for it in items if it.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


@app.get("/health")
async def health(request: Request) -> dict:
    """Return structured health fields for probes and dashboards."""
    neo4j_ok = request.app.state.neo4j is not None
    neo4j_error = getattr(request.app.state, "neo4j_error", "")
    return {
        "status": "ok",
        "version": __version__,
        "neo4j": {
            "connected": neo4j_ok,
            "uri": settings.neo4j_uri if neo4j_ok else None,
            "last_error": neo4j_error or None,
        },
        "llm": {
            "provider": settings.llm_provider,
            "configured": _generation_configured(),
            "available": request.app.state.llm is not None,
        },
        "arena": {
            "available": request.app.state.arena is not None,
            "last_error": getattr(request.app.state, "arena_error", "") or None,
        },
        "strategies": list(request.app.state.strategies.keys()),
        "demo_mode": settings.demo_mode,
    }


@app.get("/ready")
async def readiness(request: Request) -> JSONResponse:
    """Readiness probe that fails if Neo4j is configured but unreachable.

    Use for orchestrators (k8s, docker compose) that need to know when
    the service is actually ready to serve traffic, not just alive.
    """
    if settings.demo_mode:
        return JSONResponse(
            {
                "ready": True,
                "checks": {
                    "demo_mode": True,
                    "neo4j_required": False,
                    "llm_required": False,
                },
            }
        )

    checks: dict[str, bool] = {}
    ready = True

    # Neo4j: only required if a neo4j-dependent strategy is loaded
    neo4j_strategies = {"knowledge_graph", "hybrid"}
    loaded = set(request.app.state.strategies.keys())
    needs_neo4j = bool(loaded & neo4j_strategies)

    if needs_neo4j:
        driver = request.app.state.neo4j
        if driver is None:
            checks["neo4j"] = False
            ready = False
        else:
            try:
                async with driver.session(database=settings.neo4j_database) as session:
                    result = await session.run("RETURN 1")
                    await result.consume()
                checks["neo4j"] = True
            except Exception:
                checks["neo4j"] = False
                ready = False
    else:
        checks["neo4j"] = True  # not needed

    # LLM: check if at least one API key is configured
    checks["llm_configured"] = _generation_configured() and request.app.state.llm is not None
    if not checks["llm_configured"]:
        ready = False

    status_code = 200 if ready else 503
    return JSONResponse(
        {"ready": ready, "checks": checks},
        status_code=status_code,
    )


@app.post("/api/debug/explain", dependencies=[Depends(require_auth)])
async def debug_explain(body: ChatRequest, request: Request) -> dict:
    """Debug endpoint: show intent classification, retrieval chunks, and routing decision.

    Returns the full pipeline trace without generating a final answer.
    Gated behind KB_ARENA_DEBUG=true to avoid exposing internals in production.
    """
    if not settings.debug:
        raise HTTPException(status_code=404, detail="not_found")

    strategies = request.app.state.strategies
    router = request.app.state.router

    # Classify intent before resolving a strategy.
    intent = "unknown"
    try:
        intent = await router.classify(body.query)
    except Exception as exc:
        logger.warning(
            "Debug explain: intent classification failed [request_id=%s]: %s",
            _current_request_id(),
            exc,
        )
        intent = f"error: {exc}"

    # Resolve the requested or routed strategy.
    strategy_name = body.strategy
    strategy = strategies.get(strategy_name)
    if not strategy:
        return {
            "intent": intent,
            "strategy": strategy_name,
            "error": f"Unknown strategy: {strategy_name}",
        }

    # Query the strategy for retrieval results.
    import time as _time

    t0 = _time.perf_counter()
    try:
        result = await strategy.query(body.query, top_k=5, corpus=body.corpus)
        latency_ms = (_time.perf_counter() - t0) * 1000
    except Exception as exc:
        logger.warning(
            "Debug explain: strategy %s query failed [request_id=%s]: %s",
            strategy_name,
            _current_request_id(),
            exc,
        )
        return {
            "intent": intent,
            "strategy": strategy_name,
            "error": f"Query failed: {exc}",
        }

    return {
        "intent": intent,
        "strategy": strategy_name,
        "answer_preview": result.answer[:500] if result.answer else "",
        "sources": result.sources,
        "graph_context": result.graph_context.model_dump() if result.graph_context else None,
        "latency_ms": round(latency_ms, 1),
        "retrieval_latency_ms": round(result.retrieval_latency_ms, 1),
        "generation_latency_ms": round(result.generation_latency_ms, 1),
        "tokens_used": result.tokens_used,
        "cost_usd": result.cost_usd,
    }


# Mount bundled static frontend (must be AFTER all API routes)
# Check multiple locations: source tree, importlib, site-packages
_static_candidates = [
    _Path(_pkg_resources.files("kb_arena")) / "static",  # type: ignore[arg-type]
    _Path(__file__).resolve().parent.parent / "static",
]
# Also check relative to CWD (for editable installs where kb_arena/ is in project root)
_cwd_static = _Path("kb_arena") / "static"
if _cwd_static.is_dir():
    _static_candidates.insert(0, _cwd_static.resolve())

for _sd in _static_candidates:
    if _sd.is_dir() and (_sd / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(_sd), html=True), name="static")
        break
