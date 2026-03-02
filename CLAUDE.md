# KB Arena

## Overview

Benchmark knowledge graphs vs vector RAG on AWS documentation. Proves empirically that knowledge graphs beat vector RAG on multi-hop, relational, and comparative queries across 3 AWS documentation corpora (Compute, Storage, Networking).

## Architecture

```
Raw docs → Parsers → Document model (JSONL) → 5 Strategies → Benchmark Engine → Results
                                                                    ↓
                                                            Chatbot Demo (SSE)
```

**5 strategies:** naive vector, contextual vector, QnA pairs, knowledge graph (Neo4j), hybrid (graph + vector with intent routing)

**Stack:** Python 3.12, FastAPI, Neo4j Community 5, ChromaDB, Claude Haiku/Sonnet, Next.js 14 frontend

## How to Run

```bash
# Full stack
docker compose up -d

# Or locally
pip install -e ".[dev]"
kb-arena ingest ./datasets/aws-compute/raw/ --corpus aws-compute
kb-arena build-graph --corpus aws-compute
kb-arena build-vectors --corpus aws-compute
kb-arena benchmark --corpus aws-compute
kb-arena serve
```

## Project Structure

- `kb_arena/models/` — Pydantic v2 models (Document, Entity, Question, etc.). Central interchange.
- `kb_arena/ingest/` — Document parsers (markdown, HTML, AWS docs) + pipeline orchestrator
- `kb_arena/graph/` — Neo4j graph engine: schema, extraction, resolution, batch loading, Cypher templates
- `kb_arena/strategies/` — 5 retrieval strategies, all implement Strategy base class
- `kb_arena/benchmark/` — YAML question loader, evaluator (structural + LLM judge), runner, reporter
- `kb_arena/chatbot/` — FastAPI app with SSE streaming, intent router, session memory
- `kb_arena/llm/` — Dual-model LLM client (Haiku for classification, Sonnet for generation)
- `cypher/` — Idempotent Neo4j schema DDL per corpus
- `datasets/` — Raw docs, processed JSONL, benchmark questions (YAML)
- `web/` — Next.js 14 frontend: demo, benchmark explorer, graph visualizer
- `tests/` — pytest, mock Neo4j/ChromaDB in conftest.py

## Conventions

- **Functions:** snake_case
- **Classes:** PascalCase
- **Enums:** NodeType, RelType — str Enum with string values
- **Models:** Pydantic v2 BaseModel everywhere
- **Config:** pydantic-settings, all from environment with KB_ARENA_ prefix
- **CLI:** Typer + Rich
- **Async:** FastAPI endpoints async, Neo4j driver async, LLM client async
- **Error handling:** Consistent `{"error": {"code": "...", "message": "..."}}` envelope
- **Intermediate files:** JSONL in datasets/{corpus}/processed/

## Testing

```bash
pytest tests/ -v
ruff check . && ruff format --check .
```

- Mock Neo4j and ChromaDB via conftest fixtures
- Test each parser with real sample documents
- Benchmark evaluator has structural checks (must_mention/must_not_claim) before LLM judge
- Never make real API calls in unit tests

## Key Patterns

- **Schema-constrained extraction:** NodeType/RelType enums in extraction prompts. Post-validate.
- **Two-threshold entity resolution:** Jaro-Winkler >=0.92 auto-merge, 0.85-0.91 review queue
- **UNWIND/MERGE batch loading:** batch_size=1000, always consume() results, nodes before edges
- **Three-stage intent classification:** keyword scan → Haiku LLM → regex fallback
- **SSE streaming:** 4 event types (message_id, token, done, meta)
- **Lifespan init:** Services on app.state, structured errors when Neo4j unavailable
- **Dual-model LLM:** Haiku for cheap classification, Sonnet for generation. cache_control on system prompts.

## Frontend (web/)

- **Theme:** Cloudwright light (CSS custom properties in globals.css)
- **Corpora:** aws-compute, aws-storage, aws-networking (defined in lib/api.ts)
- **GraphViewer:** Custom canvas force-directed graph (~600 lines), no external graph libs
- **Demo page:** Pre-filled with AWS Lambda/API Gateway/RDS showcase question
- **Screenshots:** Use `agent-browser` with networkidle wait (NOT Playwright CLI `--wait-for-timeout` which misses CSS hydration). Dev server must be running on port 3001.
- **Build:** `cd web && npm install && npx next build` — builds static pages

## Never

- Hardcode API keys — use .env via pydantic-settings
- Use `yaml.load()` without SafeLoader
- Create Neo4j edges before nodes exist
- Skip `await result.consume()` on Neo4j queries
- Use bare `except:` — catch specific exceptions
- Add `allow_origins=["*"]` in production CORS
