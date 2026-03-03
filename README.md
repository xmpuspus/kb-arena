# KB Arena

Which retrieval architecture works best for your documentation?

KB Arena benchmarks 5 retrieval strategies — naive vector, contextual vector, Q&A pairs, knowledge graph, and hybrid — on **your** documentation. Bring your docs in any format, run the pipeline, get empirical results. Ships with an AWS Compute corpus (75 questions across 5 difficulty tiers) as a built-in example.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue) ![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green) ![Tests](https://img.shields.io/badge/tests-367-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue)

![KB Arena Demo](docs/demo.gif)

---

## Quick Start — I Just Have My Docs

You have documentation files (markdown, HTML, text, PDFs). You want to know which retrieval strategy works best. Here's everything from zero.

### Prerequisites

1. **Python 3.11+** and **pip**
2. **Docker** (for Neo4j — the knowledge graph strategy needs it)
3. **API keys** for Anthropic (LLM) and OpenAI (embeddings)

That's it. No Neo4j expertise needed. No graph database experience required. KB Arena handles the schema, extraction, and querying.

### Step 1: Install

```bash
pip install kb-arena
```

### Step 2: Set API keys

Create a `.env` file or export directly:

```bash
export KB_ARENA_ANTHROPIC_API_KEY=sk-ant-...    # Claude for generation + evaluation
export KB_ARENA_OPENAI_API_KEY=sk-...           # OpenAI for text-embedding-3-large
```

### Step 3: Start Neo4j

KB Arena uses Neo4j for the knowledge graph strategy. One command:

```bash
docker compose up neo4j -d
```

This starts Neo4j on `localhost:7687` with default credentials (`neo4j` / `kbarena1`). No configuration needed — KB Arena creates the schema automatically.

If you don't have the `docker-compose.yml`, create one:

```yaml
services:
  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/kbarena1
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

**Don't want Docker?** KB Arena still works — the 3 vector strategies run without Neo4j. Only the knowledge graph and hybrid strategies need it.

### Step 4: Run the pipeline

```bash
# Scaffold a new corpus
kb-arena init-corpus my-docs

# Drop your docs into the raw/ directory
cp ~/my-documentation/*.md datasets/my-docs/raw/
# Also works: .html, .rst, .txt files — the parser auto-detects format

# Parse into the unified Document model (JSONL intermediate files)
kb-arena ingest datasets/my-docs/raw/ --corpus my-docs

# Build the knowledge graph in Neo4j (entities + relationships)
kb-arena build-graph --corpus my-docs

# Build vector indexes in ChromaDB (local, no server needed)
kb-arena build-vectors --corpus my-docs

# Auto-generate benchmark questions from your docs (10 per difficulty tier)
kb-arena generate-questions --corpus my-docs --count 50

# Run the benchmark (each question x 5 strategies, 4-pass evaluation)
kb-arena benchmark --corpus my-docs

# Launch the web UI to explore results
kb-arena serve
```

Open `http://localhost:8000` for the API, `http://localhost:3000` for the dashboard.

### Step 5: Read the results

The benchmark produces:
- **Accuracy by tier** — which strategy handles simple lookups vs multi-hop architecture questions
- **Latency percentiles** — p50, p95, p99 per strategy
- **Cost per query** — token usage and API cost
- **Composite ranking** — 0.5 * accuracy + 0.3 * reliability + 0.2 * latency

Results are saved to `results/` as JSON and displayed in the web dashboard.

---

## Full Stack with Docker Compose

Run everything — Neo4j, the API server, and the frontend — in one command:

```bash
# Set your API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Start all services
docker compose up -d

# Open the dashboard
open http://localhost:3000
```

The compose file starts Neo4j (port 7474/7687), the FastAPI backend (port 8000), and the Next.js frontend (port 3000).

---

## Using the Built-in AWS Example

The AWS Compute corpus ships ready to use (75 questions across 5 difficulty tiers):

```bash
kb-arena ingest ./datasets/aws-compute/raw/ --corpus aws-compute
kb-arena build-graph --corpus aws-compute
kb-arena build-vectors --corpus aws-compute
kb-arena benchmark --corpus aws-compute
kb-arena serve
```

---

## Screenshots

**Benchmark results** — Accuracy table by tier with grouped bar chart. Fetches real results from `kb-arena benchmark` or displays sample data.

![Benchmark results](docs/screenshot-benchmark.png)

**Strategy comparison** — Ask the same question to all 5 strategies simultaneously. Compare answers, sources, latency, and cost.

![Strategy comparison demo](docs/screenshot-demo.png)

**Knowledge graph explorer** — Interactive force-directed visualization of entities extracted from your documentation.

![Knowledge graph viewer](docs/screenshot-graph.png)

**Home page** — Overview of strategies, difficulty tiers, and evaluation methodology.

![Home page](docs/screenshot-home.png)

---

## The 5 Strategies

| # | Strategy | How it works | Best at |
|---|----------|-------------|---------|
| 1 | **Naive Vector** | Chunk → embed → cosine similarity → generate | Fast lookups, simple factoid questions |
| 2 | **Contextual Vector** | Chunk + parent context → embed → rank | Disambiguating domain-specific terms |
| 3 | **Q&A Pairs** | LLM pre-generates Q&A at index time → match | Common questions with known answers |
| 4 | **Knowledge Graph** | Entities → Neo4j → Cypher templates → generate | Multi-hop dependencies, cross-topic queries |
| 5 | **Hybrid** | Intent routing → vector or graph or both (RRF) | Adapts per question type |

---

## Question Tiers

Questions are organized into 5 difficulty tiers:

| Tier | Type | Hops | Where vector RAG breaks |
|------|------|------|--------------------------|
| 1 | Lookup | 1 | All strategies competitive |
| 2 | Procedural | 1-2 | Vector drops to ~60% |
| 3 | Comparative | 2-3 | Vector drops to ~30%, graph dominates |
| 4 | Relational | 3-4 | Only graph answers correctly |
| 5 | Multi-hop | 3-5 | Only graph + provenance answers |

Use `kb-arena generate-questions` to auto-generate questions from your docs, or write them by hand in YAML.

---

## Universal Documentation Schema

KB Arena extracts entities and relationships using a universal schema that works for any documentation domain:

**5 node types:** Topic, Component, Process, Config, Constraint
**7 relationship types:** DEPENDS_ON, CONTAINS, CONNECTS_TO, TRIGGERS, CONFIGURES, ALTERNATIVE_TO, EXTENDS

No per-domain configuration needed. The LLM maps your documentation concepts to these types automatically.

---

## CLI Reference

| Command | Description |
|---|---|
| `init-corpus <name>` | Scaffold `datasets/{name}/` directories |
| `ingest <path>` | Parse docs into JSONL. Options: `--corpus`, `--format` |
| `build-graph` | Extract entities/rels into Neo4j. Options: `--corpus` |
| `build-vectors` | Build ChromaDB indexes. Options: `--corpus`, `--strategy` |
| `generate-questions` | Auto-generate benchmark questions. Options: `--corpus`, `--count` |
| `benchmark` | Run evaluation. Options: `--corpus`, `--strategy`, `--tier` |
| `report` | Generate report. Options: `--corpus`, `--output` |
| `serve` | Launch API + frontend. Options: `--host`, `--port` |
| `health` | Pipeline status — per-corpus progress, service connectivity, API keys |

All commands are independently re-runnable. Each stage writes to disk so you can re-run any step without repeating earlier ones.

---

## Environment Variables

All prefixed with `KB_ARENA_`. Loaded from `.env` or environment.

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `ANTHROPIC_API_KEY` | — | Yes | Claude for generation, evaluation, extraction |
| `OPENAI_API_KEY` | — | Yes | OpenAI for text-embedding-3-large |
| `NEO4J_URI` | `bolt://localhost:7687` | No | Neo4j connection |
| `NEO4J_USER` | `neo4j` | No | Neo4j username |
| `NEO4J_PASSWORD` | `kbarena1` | No | Neo4j password |
| `GENERATE_MODEL` | `claude-sonnet-4-6` | No | Generation model |
| `FAST_MODEL` | `claude-haiku-4-5-20251001` | No | Classification model |
| `BENCHMARK_MAX_CONCURRENT` | `5` | No | Parallel benchmark queries |

---

## Development

```bash
# Install with dev dependencies
pip install -e '.[dev]'

# Run tests (367 tests)
pytest tests/ -v --ignore=tests/live

# Lint + format
ruff check . && ruff format --check .

# Frontend
cd web && npm install && npx next build
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Claude Sonnet 4.6 (generation) + Haiku 4.5 (classification) |
| Embeddings | text-embedding-3-large (3072-dim) |
| Vector store | ChromaDB 0.5 (local, no server) |
| Graph store | Neo4j 5 Community |
| Backend | FastAPI + SSE streaming |
| Frontend | Next.js 14 + Tailwind + Recharts |
| Models | Pydantic v2 |
| CLI | Typer + Rich |
| Testing | pytest (367 tests) |

---

## License

MIT
