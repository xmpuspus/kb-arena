"""FastAPI chatbot API — SSE streaming, strategy routing, health check.

Lifespan pattern from paper-trail-ph: init services on startup, store on app.state.
Neo4j unavailability is handled gracefully — strategy falls back to mock data.
"""

from __future__ import annotations

import asyncio as _asyncio
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
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse

from kb_arena.chatbot.session import SessionMemory
from kb_arena.chatbot.tools_api import router as tools_router
from kb_arena.models.api import ChatRequest, ChatResponse, ErrorDetail, ErrorResponse
from kb_arena.settings import settings

# Per-corpus queues for streaming graph build events to SSE clients
_graph_build_queues: dict[str, _asyncio.Queue] = {}


class _GraphBuildRequest(BaseModel):
    corpus: str

    @field_validator("corpus")
    @classmethod
    def _validate(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v or "\0" in v:
            raise ValueError("Invalid corpus name")
        return v


logger = logging.getLogger(__name__)

# Simple in-memory rate limiter (requests per minute per IP)
_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_RPM = 60

# Per-session memory for multi-turn conversations
_session_store: dict[str, SessionMemory] = defaultdict(SessionMemory)


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
    from kb_arena.strategies.pageindex import PageIndexStrategy
    from kb_arena.strategies.qna_pairs import QnAPairStrategy
    from kb_arena.strategies.raptor import RaptorStrategy

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
        "raptor": RaptorStrategy(chroma_client=chroma),
        "pageindex": PageIndexStrategy(),
    }

    yield

    if app.state.neo4j is not None:
        await app.state.neo4j.close()


app = FastAPI(
    title="KB Arena API",
    description="Benchmark retrieval strategies on your documentation.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow configurable origins, never wildcard in production
_cors_origins = getattr(settings, "cors_origins", None) or [
    "http://localhost:3000",
    "http://localhost:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(tools_router)


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

    # Track conversation history per session (keyed by IP + strategy)
    session_key = f"{client_ip}:{body.strategy}"
    session = _session_store[session_key]
    session.add_turn("user", body.query)

    result = await strategy.query(body.query, top_k=5)

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
        if corpus != "all" and corpus not in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            all_results.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if not all_results:
        return {"results": [], "source": "none"}

    # Aggregate per-strategy across all loaded result files
    from collections import defaultdict

    strategy_tiers: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    strategy_latency: dict[str, list[float]] = defaultdict(list)
    strategy_cost: dict[str, list[float]] = defaultdict(list)

    for result in all_results:
        strategy = result.get("strategy", "")
        for rec in result.get("records", []):
            try:
                tier = int(rec.get("question_id", "").split("-t")[1].split("-")[0])
            except (IndexError, ValueError):
                tier = 0
            score = rec.get("score", {})
            strategy_tiers[strategy][tier].append(score.get("accuracy", 0))
            strategy_latency[strategy].append(rec.get("latency_ms", 0))
            strategy_cost[strategy].append(rec.get("cost_usd", 0))

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
    """List available strategy names."""
    return {"strategies": list(request.app.state.strategies.keys())}


@app.get("/graph/stats")
async def graph_stats(request: Request) -> dict:
    """Graph statistics — node/edge counts, centrality hubs, communities."""
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
        "top_hubs": [{"fqn": fqn, "centrality": round(c, 4)} for fqn, c in top_hubs],
        "community_count": len(communities),
    }


@app.get("/api/graph/data")
async def graph_data(request: Request, corpus: str = "all", limit: int = 200) -> dict:
    """Return graph nodes and edges from Neo4j for visualization."""
    if request.app.state.neo4j is None:
        return {"nodes": [], "edges": [], "connected": False}

    driver = request.app.state.neo4j
    limit = min(limit, 500)

    # Fetch nodes
    node_query = (
        "MATCH (n) "
        + ("WHERE n.corpus = $corpus " if corpus != "all" else "")
        + "RETURN n.fqn AS id, n.name AS name, labels(n)[0] AS type, "
        "n.description AS description LIMIT $limit"
    )
    params = {"limit": limit}
    if corpus != "all":
        params["corpus"] = corpus

    nodes = []
    async with driver.session() as session:
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
            "MATCH (a)-[r]->(b) "
            + ("WHERE a.corpus = $corpus " if corpus != "all" else "")
            + "RETURN a.fqn AS source, type(r) AS type, b.fqn AS target "
            "LIMIT $edge_limit"
        )
        edge_params = {"edge_limit": limit * 2}
        if corpus != "all":
            edge_params["corpus"] = corpus

        async with driver.session() as session:
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


@app.post("/api/graph/build")
async def trigger_graph_build(body: _GraphBuildRequest) -> dict:
    """Trigger graph build for a corpus. Events streamed via /api/graph/build/stream/{corpus}."""
    corpus = body.corpus
    queue: _asyncio.Queue = _asyncio.Queue()
    _graph_build_queues[corpus] = queue

    async def _callback(event: dict | None) -> None:
        await queue.put(event)

    async def _run() -> None:
        from kb_arena.graph.extractor import run_extraction

        try:
            await run_extraction(corpus=corpus, event_callback=_callback)
        except Exception as exc:
            await queue.put({"type": "error", "data": {"message": str(exc)}})
        finally:
            await queue.put(None)  # sentinel — signals stream end

    _asyncio.create_task(_run())
    return {"status": "started", "corpus": corpus}


@app.get("/api/graph/build/stream/{corpus}")
async def graph_build_stream(corpus: str) -> EventSourceResponse:
    """SSE stream of graph build events for a corpus."""
    queue = _graph_build_queues.get(corpus)
    if queue is None:

        async def _empty() -> AsyncIterator[dict]:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": "No build in progress. POST to /api/graph/build first."}
                ),
            }

        return EventSourceResponse(_empty())

    async def event_generator() -> AsyncIterator[dict]:
        while True:
            try:
                event = await _asyncio.wait_for(queue.get(), timeout=30.0)
            except TimeoutError:
                yield {"event": "heartbeat", "data": "{}"}
                continue
            if event is None:  # sentinel — build complete
                break
            yield {"event": event["type"], "data": json.dumps(event["data"])}
        _graph_build_queues.pop(corpus, None)

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health(request: Request) -> dict:
    """Health check — reports Neo4j connectivity status."""
    neo4j_ok = request.app.state.neo4j is not None
    return {
        "status": "ok",
        "neo4j": "connected" if neo4j_ok else "unavailable (mock mode)",
        "strategies": list(request.app.state.strategies.keys()),
    }
