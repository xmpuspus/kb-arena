# KB Arena

Benchmark knowledge graphs vs vector RAG on real documentation.

KB Arena runs the same 200 questions against 5 retrieval strategies — naive vector, contextual vector, Q&A pairs, knowledge graph, and hybrid — and measures accuracy, latency percentiles, and response reliability across 3 real-world documentation corpora. The result: empirical evidence for which data representation actually works, not opinions.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue) ![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green) ![Tests](https://img.shields.io/badge/tests-339-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue)

```
"What exception does json.loads raise for invalid input?"
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Same question → 5 strategies in parallel                    │
│                                                              │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────┐            │
│  │ Naive Vector│ │  Contextual  │ │  Q&A Pairs │            │
│  │   ChromaDB  │ │    Vector    │ │   ChromaDB │            │
│  │ Embed→Rank  │ │ Chunk+Parent │ │ LLM-gen QA │            │
│  └──────┬──────┘ └──────┬───────┘ └─────┬──────┘            │
│         │               │               │                    │
│  ┌──────┴──────┐ ┌──────┴───────┐                            │
│  │  Knowledge  │ │    Hybrid    │                            │
│  │    Graph    │ │ Vector + KG  │                            │
│  │ Neo4j+Cypher│ │  RRF Fusion  │                            │
│  └──────┬──────┘ └──────┬───────┘                            │
│         │               │                                    │
│         ▼               ▼                                    │
│  ┌──────────────────────────────────────────────────┐        │
│  │ 4-Pass Evaluator                                 │        │
│  │ Structural → Entity Coverage → Source Attr → LLM │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Report: accuracy by tier, latency p50/p95/p99,              │
│          reliability rates, cross-strategy ranking            │
└──────────────────────────────────────────────────────────────┘
```

---

## The Question This Answers

> "What should the data format be for freeform texts or multiple documents so we can have an accurate, reliable, and fast knowledge retrieval + chatbot system?"

The source format matters far less than the retrieval architecture. Knowledge graphs as an intermediate representation — extracted from ANY source format — beat every other approach on multi-hop, relational, and comparative queries. Pure vector RAG wins on simple lookups but collapses on queries touching 3+ interconnected concepts.

This project proves it with real data, reproducible benchmarks, and a live side-by-side chatbot demo.

---

## How It Compares

| Capability | KB Arena | RAGAS | BEIR | LlamaIndex Bench |
|---|:---:|:---:|:---:|:---:|
| Knowledge graph strategy | Y | - | - | - |
| Hybrid (vector + KG) strategy | Y | - | - | - |
| Side-by-side chatbot demo | Y | - | - | - |
| Per-tier difficulty breakdown | 5 tiers | - | - | - |
| Latency percentiles (p50/p95/p99) | Y | - | - | - |
| Response reliability tracking | Y | Y | - | - |
| 4-pass evaluator (structural + LLM) | Y | LLM-only | BM25 | LLM |
| Multi-corpus (3 domains) | Y | Custom | 18 datasets | Custom |
| Pip-installable CLI | Y | Y | Y | - |
| Entity coverage scoring | Y | - | - | - |
| Source attribution scoring | Y | Y | - | - |

RAGAS and BEIR benchmark retrieval quality but only test vector-based approaches. KB Arena adds knowledge graph and hybrid strategies to the comparison, with a difficulty-tiered question set that isolates exactly where each approach breaks down.

---

## Quick Start

```bash
pip install kb-arena
```

Set API keys:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...   # for text-embedding-3-large
```

Start Neo4j (for knowledge graph strategy):

```bash
docker compose up neo4j -d
```

Run the full pipeline:

```bash
# 1. Parse raw documentation into unified Document model
kb-arena ingest ./docs --corpus python-stdlib

# 2. Build knowledge graph in Neo4j
kb-arena build-graph --corpus python-stdlib

# 3. Build vector indexes in ChromaDB
kb-arena build-vectors --corpus python-stdlib

# 4. Run benchmark (200 questions × 5 strategies)
kb-arena benchmark --corpus all --strategy all

# 5. Generate report
kb-arena report

# 6. Launch side-by-side chatbot demo
kb-arena serve
```

The API runs at `http://localhost:8000` and the Next.js frontend at `http://localhost:3000`.

---

## Real-World Examples

### 1. Ingest Documentation — Any Format to JSONL

Parse a directory of raw docs into the unified Document model:

```bash
$ kb-arena ingest ./python-stdlib-docs --corpus python-stdlib --format html

Parsing ./python-stdlib-docs (format: html)
  json.html → 47 sections, 12,847 tokens
  os.html → 128 sections, 34,291 tokens
  asyncio.html → 89 sections, 27,103 tokens
  ...
Written 412 documents to datasets/python-stdlib/processed/docs.jsonl
```

Three parsers ship out of the box:

```bash
kb-arena ingest ./k8s-docs --corpus kubernetes --format markdown
kb-arena ingest ./sec-filings --corpus sec-edgar --format sec-edgar
kb-arena ingest ./anything --corpus custom --format auto  # auto-detects
```

### 2. Build the Knowledge Graph

Extract entities and relationships into Neo4j using corpus-specific schemas:

```bash
$ kb-arena build-graph --corpus python-stdlib

Extracting entities from 412 documents...
  json.loads → Function (raises: JSONDecodeError, returns: Any)
  json.JSONDecodeError → Exception (inherits: ValueError)
  os.getcwd → Function (returns: str)
  collections.OrderedDict → Class (inherits: dict)
  ...
Resolving duplicates (Jaro-Winkler threshold: 0.92)...
  Merged 23 duplicate entities
Writing to Neo4j: 1,847 nodes, 4,212 relationships
Done.
```

### 3. Build Vector Indexes

Build ChromaDB indexes for the three vector-backed strategies:

```bash
$ kb-arena build-vectors --corpus python-stdlib --strategy all

Building index: naive_vector (412 documents)
  Embedding with text-embedding-3-large (3072 dims)...
  412 chunks indexed in ChromaDB
Done: naive_vector

Building index: contextual_vector (412 documents)
  Adding parent context to each chunk...
  412 chunks indexed in ChromaDB
Done: contextual_vector

Building index: qna_pairs (412 documents)
  Generating Q&A pairs via Claude Sonnet...
  1,847 Q&A pairs indexed in ChromaDB
Done: qna_pairs
```

### 4. Run the Benchmark

Run 200 questions against all 5 strategies with per-query timeout and retry:

```bash
$ kb-arena benchmark --corpus all --strategy all

Running benchmark: python-stdlib × naive_vector (75 questions)
  ████████████████████████████████████████ 75/75 [00:42]
Running benchmark: python-stdlib × knowledge_graph (75 questions)
  ████████████████████████████████████████ 75/75 [01:14]
  ...

Results written to:
  results/python-stdlib_naive_vector.json
  results/python-stdlib_contextual_vector.json
  results/python-stdlib_qna_pairs.json
  results/python-stdlib_knowledge_graph.json
  results/python-stdlib_hybrid.json
  results/kubernetes_naive_vector.json
  ...
```

Filter by corpus, strategy, or tier:

```bash
kb-arena benchmark --corpus python-stdlib --strategy knowledge_graph --tier 4
```

### 5. Generate the Report

```bash
$ kb-arena report

Report written to results/report.md
Summary written to results/summary.json
```

The report contains 6 sections per corpus plus a cross-strategy ranking:

```
# KB Arena Benchmark Report

## python-stdlib

### Overall Accuracy by Strategy
| Strategy          | Accuracy | Completeness | Faithfulness | Avg Latency (ms) | Total Cost  | Cost/Correct |
|-------------------|----------|--------------|--------------|------------------|-------------|--------------|
| naive_vector      | ...      | ...          | ...          | ...              | $...        | $...         |
| contextual_vector | ...      | ...          | ...          | ...              | $...        | $...         |
| qna_pairs         | ...      | ...          | ...          | ...              | $...        | $...         |
| knowledge_graph   | ...      | ...          | ...          | ...              | $...        | $...         |
| hybrid            | ...      | ...          | ...          | ...              | $...        | $...         |

### Latency Distribution by Strategy
| Strategy | Avg | p50 | p95 | p99 | Min | Max |

### Response Reliability by Strategy
| Strategy | Success Rate | Error Rate | Empty Rate | Avg Faithfulness | Avg Source Attr | Avg Entity Cov |

### Accuracy by Tier
| Strategy          | Tier 1 - Factoid | Tier 2 - Multi-entity | Tier 3 - Comparative | Tier 4 - Relational | Tier 5 - Temporal |

### Accuracy by Question Type
| Strategy          | factoid | comparison | relational | temporal | causal |

### Latency by Tier (p50 ms)
| Strategy          | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 |

## Cross-Strategy Ranking
| Rank | Strategy         | Avg Accuracy | p50 Latency (ms) | Success Rate | Composite |
|------|------------------|-------------|------------------|-------------|-----------|
| 1    | hybrid           | ...         | ...              | ...         | ...       |
| 2    | knowledge_graph  | ...         | ...              | ...         | ...       |
| 3    | contextual_vector| ...         | ...              | ...         | ...       |
| 4    | naive_vector     | ...         | ...              | ...         | ...       |
| 5    | qna_pairs        | ...         | ...              | ...         | ...       |

*Composite = 0.5 * Accuracy + 0.3 * Reliability + 0.2 * Latency Score*
```

### 6. Side-by-Side Chatbot Demo

```bash
$ kb-arena serve --port 8000

INFO:     KB Arena API v0.1.0
INFO:     Neo4j connected at bolt://localhost:7687
INFO:     5 strategies loaded: naive_vector, contextual_vector, qna_pairs, knowledge_graph, hybrid
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The chatbot API supports both synchronous and SSE streaming:

```bash
# Synchronous — single strategy
$ curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "What exception does json.loads raise?", "strategy": "knowledge_graph"}' | python3 -m json.tool

{
  "answer": "json.loads() raises json.JSONDecodeError when given invalid JSON. JSONDecodeError is a subclass of ValueError.",
  "strategy_used": "knowledge_graph",
  "sources": ["json.html#json.JSONDecodeError", "json.html#json.loads"],
  "graph_context": {
    "entities": [
      {"name": "json.loads", "type": "Function"},
      {"name": "json.JSONDecodeError", "type": "Exception"}
    ],
    "relationships": [
      {"source": "json.loads", "type": "RAISES", "target": "json.JSONDecodeError"}
    ]
  },
  "latency_ms": 823.4,
  "tokens_used": 147,
  "cost_usd": 0.0012
}
```

SSE streaming returns 4 event types:

```
event: message_id
data: {"id": "a1b2c3d4"}

event: token
data: {"text": "json.loads()"}

event: token
data: {"text": " raises "}

event: token
data: {"text": "json.JSONDecodeError"}

event: done
data: {"sources": ["json.html#json.JSONDecodeError"], "strategy_used": "knowledge_graph"}

event: meta
data: {"latency_ms": 823.4, "tokens_used": 147, "cost_usd": 0.0012}
```

### 7. Health Check

```bash
$ curl -s http://localhost:8000/health | python3 -m json.tool

{
  "status": "ok",
  "neo4j": "connected",
  "strategies": ["naive_vector", "contextual_vector", "qna_pairs", "knowledge_graph", "hybrid"]
}
```

When Neo4j is unavailable, the graph strategy falls back to mock data gracefully:

```json
{
  "status": "ok",
  "neo4j": "unavailable (mock mode)",
  "strategies": ["naive_vector", "contextual_vector", "qna_pairs", "knowledge_graph", "hybrid"]
}
```

---

## Python API

### Evaluate a Single Answer

```python
import asyncio
from kb_arena.benchmark.evaluator import evaluate
from kb_arena.models.benchmark import GroundTruth, Constraints

async def main():
    score = await evaluate(
        answer="json.loads raises json.JSONDecodeError for invalid input.",
        ground_truth=GroundTruth(
            answer="json.loads() raises JSONDecodeError.",
            source_refs=["json.html#json.JSONDecodeError"],
            required_entities=["json.loads", "json.JSONDecodeError"],
        ),
        constraints=Constraints(
            must_mention=["JSONDecodeError", "ValueError"],
            must_not_claim=["TypeError", "SyntaxError"],
        ),
        sources=["json.html#json.JSONDecodeError"],
        llm=None,  # structural-only (no LLM cost)
    )
    print(f"Accuracy: {score.accuracy}")                    # 0.5 (1 of 2 must_mention)
    print(f"Entity coverage: {score.entity_coverage}")      # 1.0
    print(f"Source attribution: {score.source_attribution}") # 1.0
    print(f"Structural pass: {score.structural_pass}")       # True

asyncio.run(main())
```

### Query a Single Strategy

```python
import asyncio
from kb_arena.strategies import get_strategy

async def main():
    strategy = get_strategy("knowledge_graph")
    result = await strategy.query("What does json.loads return?")

    print(f"Answer: {result.answer}")
    print(f"Sources: {result.sources}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    print(f"Cost: ${result.cost_usd:.4f}")

    if result.graph_context:
        for entity in result.graph_context.entities:
            print(f"  Entity: {entity.name} ({entity.type})")
        for rel in result.graph_context.relationships:
            print(f"  Rel: {rel.source} --{rel.type}--> {rel.target}")

asyncio.run(main())
```

### Classify Query Intent

```python
import asyncio
from kb_arena.chatbot.router import IntentRouter
from kb_arena.llm.client import LLMClient

async def main():
    router = IntentRouter(llm=LLMClient())

    # Stage 1 (keyword scan, <1ms)
    intent = await router.classify("Compare json.dumps vs pickle.dumps")
    print(intent)  # QueryIntent.COMPARISON

    # Stage 2 (Haiku LLM, ~50ms) — only if keyword scan misses
    intent = await router.classify("What implications does asyncio have for database access?")
    print(intent)  # QueryIntent.RELATIONAL

asyncio.run(main())
```

### Run Benchmark Programmatically

```python
import asyncio
from kb_arena.benchmark.runner import run_benchmark
from kb_arena.benchmark.reporter import generate_report

async def main():
    # Run a subset
    await run_benchmark(corpus="python-stdlib", strategy="knowledge_graph", tier=4)

    # Generate report
    generate_report(corpus="python-stdlib", output="./my_report.md")

asyncio.run(main())
```

### Use the LLM Client Directly

```python
import asyncio
from kb_arena.llm.client import LLMClient

async def main():
    llm = LLMClient()

    # Classification (Haiku, ~20 tokens, <50ms, cached system prompt)
    intent = await llm.classify(
        query="What's the difference between threads and processes?",
        system_prompt="Classify as: factoid, comparison, relational",
        allowed_values=["factoid", "comparison", "relational"],
    )
    print(intent)  # "comparison"

    # Generation (Sonnet, cached system prompt)
    answer = await llm.generate(
        query="What does json.loads return?",
        context="json.loads deserializes a JSON string to a Python object...",
        system_prompt="You are a Python documentation assistant.",
    )
    print(answer)

    # LLM-as-judge (Sonnet)
    judgment = await llm.judge(
        answer="json.loads returns a dict",
        reference="json.loads returns a Python object — dict, list, str, int, float, bool, or None",
        system_prompt="Score accuracy, completeness, faithfulness as JSON.",
    )
    print(judgment)  # {"accuracy": 0.6, "completeness": 0.4, "faithfulness": 1.0}

asyncio.run(main())
```

### Work with Latency Stats

```python
from kb_arena.models.benchmark import LatencyStats

# From a list of query times
values = [120.5, 340.2, 180.7, 520.1, 90.3, 210.8, 150.4, 680.9, 310.6, 240.0]
stats = LatencyStats.from_values(values)

print(f"Avg: {stats.avg_ms:.0f}ms")   # 284ms
print(f"p50: {stats.p50_ms:.0f}ms")   # 240ms
print(f"p95: {stats.p95_ms:.0f}ms")   # 681ms (falls back to max for <20 samples)
print(f"Min: {stats.min_ms:.0f}ms")   # 90ms
print(f"Max: {stats.max_ms:.0f}ms")   # 681ms
```

---

## What Gets Measured

KB Arena evaluates across three dimensions. Each dimension is broken down by question tier and strategy.

### Accuracy (4-pass evaluation)

Every answer goes through four sequential evaluation passes:

```
Pass 1: Structural Check         <1ms    regex-based
  └── must_mention terms present? must_not_claim terms absent?
Pass 2: Entity Coverage          <1ms    regex-based
  └── what fraction of required_entities appear in the answer?
Pass 3: Source Attribution        <1ms    substring matching
  └── do returned sources match expected source_refs?
Pass 4: LLM-as-Judge            ~500ms   Claude Sonnet
  └── accuracy (0-1), completeness (0-1), faithfulness (0-1)
```

If Pass 1 fails (false claims detected), the answer scores 0.0 and the LLM judge is skipped — saving cost and preventing the LLM from rationalizing a wrong answer.

### Latency

Per-query wall-clock time with percentile breakdown:

| Metric | Description |
|--------|-------------|
| avg | Mean latency across all queries |
| p50 | Median — what most queries feel like |
| p95 | Tail latency — 1 in 20 queries is this slow |
| p99 | Worst-case latency (needs 100+ samples) |
| min/max | Bounds |

Reported per-strategy, per-tier, and cross-strategy. The p95 calculation requires 20+ samples to avoid noise; below that it falls back to max.

### Reliability

| Metric | Description |
|--------|-------------|
| Success rate | Queries that returned a non-error, non-empty answer |
| Error rate | Timeouts, API failures, malformed responses |
| Empty rate | Queries that returned nothing |
| Avg faithfulness | Mean faithfulness score (0-1) from LLM judge |
| Avg source attribution | Mean source attribution score |
| Avg entity coverage | Mean entity coverage ratio |

Each query runs with a configurable timeout (default 120s) and retry with exponential backoff (default 2 retries).

### Cross-Strategy Ranking

Strategies are ranked by a composite score:

```
Composite = 0.5 × Accuracy + 0.3 × Reliability + 0.2 × Latency Score
```

Where Latency Score = max(0, 1 - p50/10000), so a 10-second median maps to 0.0.

---

## The 5 Strategies

### Strategy 1: Naive Vector

The baseline every RAG tutorial teaches. Chunk documents, embed with `text-embedding-3-large` (3072 dimensions), store in ChromaDB, retrieve top-k by cosine similarity, generate answer.

**Strengths:** Fast, simple, good on factoid lookups.
**Weakness:** No understanding of relationships between concepts. "Which modules implement the iterator protocol?" returns chunks that mention iterators but can't connect them.

### Strategy 2: Contextual Vector

Same as naive vector, but each chunk is embedded with its parent document context prepended. A chunk from `json.html` about `JSONDecodeError` gets embedded as "json module documentation: JSONDecodeError is a subclass of ValueError..."

**Strengths:** Better at disambiguation — "loads" in json context vs "loads" in pickle context.
**Weakness:** Still can't traverse relationships.

### Strategy 3: Q&A Pairs

LLM pre-generates question-answer pairs from each document at index time. At query time, the question is matched against pre-generated questions via vector similarity.

**Strengths:** Direct question-to-answer mapping — no retrieval noise.
**Weakness:** Only answers questions the LLM thought to generate. Novel questions fall through.

### Strategy 4: Knowledge Graph

Extracts entities and relationships into Neo4j using corpus-specific schemas. Queries are classified by intent and routed to specialized Cypher templates. Entity resolution uses Jaro-Winkler similarity for fuzzy matching.

Four Cypher templates:

```cypher
-- Multi-hop traversal (tier 4-5 questions)
MATCH path = (start)-[*1..{depth}]-(connected)
WHERE start.fqn = $target
RETURN connected.name, length(path) AS hops

-- Comparison (tier 3 questions)
MATCH (a)-[r1]-(shared)-[r2]-(b)
WHERE a.fqn = $entity_a AND b.fqn = $entity_b
RETURN shared.name, type(r1), type(r2)

-- Dependency chain (tier 4 relational)
MATCH path = (source)-[:REQUIRES|IMPORTS|INHERITS*1..4]->(dep)
WHERE source.fqn = $start
RETURN dep.name, length(path) AS depth

-- Fulltext search (all tiers)
CALL db.index.fulltext.queryNodes('entity_search', $query) YIELD node, score
```

**Strengths:** Multi-hop reasoning, relationship queries, comparisons across entities.
**Weakness:** Slower (Neo4j round-trips), requires schema design per corpus.

### Strategy 5: Hybrid

Three-stage intent classification determines routing:

```
factoid / exploratory → contextual_vector (fast path)
comparison / relational → knowledge_graph (graph path)
procedural → both paths, fused via RRF, re-ranked by Sonnet
```

Results are deduplicated by leading-token overlap, re-ranked by Sonnet (each passage scored 0-1), and top 5 passages feed final generation.

**Strengths:** Best of both — vector for recall, graph for precision on structured queries.
**Weakness:** Highest latency (two retrieval paths + re-ranking), most complex.

---

## The 200 Questions

Questions are organized into 5 difficulty tiers across 3 corpora:

| Tier | Type | Hops | Description | Example |
|------|------|------|-------------|---------|
| 1 | Factoid | 1 | Single fact lookup | "What exception does `json.loads()` raise when given invalid JSON?" |
| 2 | Multi-entity | 2 | Involves 2+ entities | "Which classes in `collections` implement `__len__`?" |
| 3 | Comparative | 2-3 | Compare/contrast | "How does `json.dumps` differ from `pickle.dumps`?" |
| 4 | Relational | 3-4 | Traverse relationships | "What is the chain of modules involved when `urllib.request.urlopen()` makes an HTTPS connection?" |
| 5 | Temporal | 3-5 | Version/change tracking | "Which `asyncio` functions were deprecated between 3.8 and 3.12?" |

### Corpora

| Corpus | Questions | Domain | Source Format |
|--------|-----------|--------|---------------|
| python-stdlib | 75 | API reference documentation | HTML |
| kubernetes | 65 | Infrastructure documentation | Markdown |
| sec-edgar | 60 | Financial regulatory filings | SEC EDGAR XML/HTML |

Each question includes ground truth, required entities, source refs, `must_mention` terms, and `must_not_claim` terms — all hand-verified.

### Question Format

Questions are defined in YAML with full evaluation metadata:

```yaml
- id: "py-t4-001"
  tier: 4
  type: relational
  hops: 3
  question: "If I use json.loads() with a custom object_hook, what exceptions could propagate?"
  ground_truth:
    answer: "Any exception raised inside object_hook propagates through json.loads() unchanged..."
    source_refs:
      - "json.html#json.loads"
    required_entities:
      - "json.loads"
      - "object_hook"
      - "json.JSONDecodeError"
  constraints:
    must_mention:
      - "object_hook"
      - "propagate"
      - "JSONDecodeError"
    must_not_claim:
      - "json catches all exceptions"
      - "only JSONDecodeError"
```

---

## Architecture

```
kb_arena/
├── cli.py                        # Typer CLI — 7 commands
├── settings.py                   # Pydantic-settings, all config from env
├── ingest/                       # Stage 1: document parsing
│   ├── pipeline.py               # Orchestrator — detect format, dispatch parser
│   └── parsers/
│       ├── html.py               # BeautifulSoup HTML parser
│       ├── markdown.py           # Markdown section parser
│       └── sec_edgar.py          # SEC EDGAR filing parser
├── graph/                        # Stage 2: knowledge graph
│   ├── schema.py                 # Per-corpus node/rel enums (Python, K8s, SEC)
│   ├── extractor.py              # LLM entity/relationship extraction
│   ├── resolver.py               # Jaro-Winkler entity resolution
│   ├── neo4j_store.py            # Async Neo4j driver wrapper
│   ├── cypher_templates.py       # Intent → Cypher template mapping
│   ├── cypher_generator.py       # Dynamic Cypher generation
│   └── analyzer.py               # Graph statistics and validation
├── strategies/                   # Stage 3: the 5 retrieval approaches
│   ├── base.py                   # Abstract Strategy + AnswerResult model
│   ├── naive_vector.py           # Strategy 1: embed → rank
│   ├── contextual_vector.py      # Strategy 2: parent-context embedding
│   ├── qna_pairs.py              # Strategy 3: pre-generated Q&A
│   ├── knowledge_graph.py        # Strategy 4: Neo4j + Cypher
│   └── hybrid.py                 # Strategy 5: vector + KG + RRF fusion
├── benchmark/                    # Stage 4: evaluation
│   ├── questions.py              # YAML question loader
│   ├── evaluator.py              # 4-pass evaluator
│   ├── runner.py                 # Orchestrator with timeout/retry
│   └── reporter.py               # Markdown + JSON report generator
├── chatbot/                      # Stage 5: demo
│   ├── api.py                    # FastAPI with SSE streaming + rate limiting
│   ├── router.py                 # 3-stage intent classifier
│   └── session.py                # Conversation session management
├── llm/
│   └── client.py                 # Dual-model: Haiku classify, Sonnet generate
├── models/                       # Pydantic v2 (central interchange format)
│   ├── document.py               # Document, Section, Chunk
│   ├── graph.py                  # Entity, Relationship, GraphContext
│   ├── benchmark.py              # Question, Score, AnswerRecord, BenchmarkResult
│   └── api.py                    # ChatRequest, ChatResponse
└── viz/                          # Visualization utilities

web/                              # Next.js 14 frontend
├── app/                          # App router pages
├── components/
│   ├── ChatPanel.tsx             # Side-by-side strategy chat
│   ├── BenchmarkTable.tsx        # Sortable results table
│   ├── TierChart.tsx             # Recharts accuracy by tier
│   ├── GraphViewer.tsx           # Sigma.js knowledge graph viz
│   ├── Nav.tsx                   # Navigation
│   └── ThemeToggle.tsx           # Dark mode toggle
└── lib/
    └── api.ts                    # Typed API client

datasets/
├── python-stdlib/questions/      # 75 questions (5 tiers × 8-20 each)
├── kubernetes/questions/         # 65 questions (5 tiers × 8-15 each)
└── sec-edgar/questions/          # 60 questions (5 tiers × 12 each)
```

---

## Graph Schemas

Each corpus has a purpose-built schema. The extractor uses the schema to generate structured prompts for entity/relationship extraction.

### Python stdlib

10 node types, 10 relationship types:

```
Module ─CONTAINS─→ Class ─INHERITS─→ Class
                   Class ─CONTAINS─→ Function
                                     Function ─REQUIRES─→ Parameter
                                     Function ─RETURNS─→ ReturnType
                                     Function ─RAISES─→ Exception
Module ─REFERENCES─→ Module
Class ─IMPLEMENTS─→ Concept
```

### Kubernetes

7 node types, 8 relationship types:

```
APIGroup ←BELONGS_TO─ Resource ─REQUIRES─→ Resource
                       Controller ─MANAGES─→ Resource
                       Version ─SUPERSEDES─→ Version
```

### SEC EDGAR

8 node types, 8 relationship types:

```
Company ─EMPLOYS─→ Executive
Company ─OWNS─→ Subsidiary
Company ─HAS_RISK─→ RiskFactor
Company ─REPORTS_METRIC─→ FinancialMetric
Company ─OPERATES_SEGMENT─→ Segment
Company ─INVOLVED_IN─→ LegalProceeding
```

---

## CLI Reference

| Command | Description |
|---|---|
| `ingest <path>` | Parse raw docs into JSONL. Options: `--corpus`, `--format` (auto/html/markdown/sec-edgar) |
| `build-graph` | Extract entities/rels → Neo4j. Options: `--corpus`, `--schema` (auto/python/kubernetes/sec) |
| `build-vectors` | Build ChromaDB indexes. Options: `--corpus`, `--strategy` (all/naive/contextual/qna) |
| `benchmark` | Run evaluation. Options: `--corpus`, `--strategy`, `--tier` (0=all) |
| `report` | Generate report. Options: `--corpus`, `--output` |
| `serve` | Launch chatbot API. Options: `--host`, `--port`, `--reload` |
| `download <corpus>` | Download raw dataset files |

All commands are independently re-runnable. Each stage writes to disk (JSONL, Neo4j, ChromaDB, JSON results) so you can re-run any stage without repeating earlier ones.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Single question → answer from specified strategy |
| `POST` | `/chat/stream` | SSE streaming response (4 event types) |
| `GET` | `/strategies` | List available strategies and their status |
| `GET` | `/health` | Health check (Neo4j status, loaded strategies) |

### Request/Response Format

```json
// POST /chat
{
  "query": "What does json.loads return?",
  "strategy": "hybrid",
  "history": [
    {"role": "user", "content": "Tell me about json"},
    {"role": "assistant", "content": "The json module provides..."}
  ],
  "corpus": "python-stdlib"
}

// Response
{
  "answer": "json.loads() deserializes a JSON string...",
  "strategy_used": "hybrid",
  "sources": ["json.html#json.loads"],
  "graph_context": {
    "entities": [...],
    "relationships": [...]
  },
  "latency_ms": 1042.3,
  "tokens_used": 234,
  "cost_usd": 0.0018
}
```

### Error Envelope

All errors follow a consistent envelope:

```json
{
  "error": {
    "code": "unknown_strategy",
    "message": "Unknown strategy 'bad'. Available: ['naive_vector', 'contextual_vector', ...]"
  }
}
```

---

## Environment Variables

All prefixed with `KB_ARENA_`. Loaded from `.env` file or environment.

### Required

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (Sonnet for generation/evaluation, Haiku for classification) |
| `OPENAI_API_KEY` | OpenAI API key (text-embedding-3-large, 3072 dimensions) |

### Neo4j

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `kbarena` | Neo4j password |

### ChromaDB

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PATH` | `./chroma_data` | ChromaDB persistent storage path |

### Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `EMBEDDING_DIMENSIONS` | `3072` | Embedding vector dimensions |

### LLM Models

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATE_MODEL` | `claude-sonnet-4-6` | Model for generation, extraction, evaluation |
| `FAST_MODEL` | `claude-haiku-4-5-20251001` | Model for classification (~20 tokens, <50ms) |

### Benchmark

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_TEMPERATURE` | `0.0` | LLM temperature for benchmark runs |
| `BENCHMARK_MAX_CONCURRENT` | `5` | Max parallel queries |
| `BENCHMARK_QUERY_TIMEOUT_S` | `120` | Per-query timeout in seconds |
| `BENCHMARK_MAX_RETRIES` | `2` | Retry count with exponential backoff |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `8000` | API port |
| `DEBUG` | `false` | Debug mode (shows full error messages) |

---

## Deployment

### Docker Compose (all services)

```bash
# Set API keys
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "OPENAI_API_KEY=sk-..." >> .env

# Start Neo4j + API + Web
docker compose up -d
```

| Service | Image | Purpose | Port |
|---------|-------|---------|------|
| `neo4j` | `neo4j:5-community` | Knowledge graph (APOC plugin) | 7474, 7687 |
| `api` | Built from Dockerfile | FastAPI backend | 8000 |
| `web` | Built from `./web` | Next.js 14 frontend | 3000 |

### Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `neo4j_data` | `/data` | Neo4j persistent storage |
| `chroma_data` | `/data/chroma` | ChromaDB persistent storage |
| `./datasets` | `/datasets` | Question YAML files |
| `./results` | `/results` | Benchmark output |

---

## Testing

```bash
# Install dev dependencies
pip install -e '.[dev]'

# Run all unit tests (339 tests)
pytest tests/ -q

# Run specific test modules
pytest tests/test_benchmark.py -v     # evaluator, models, aggregation (24 tests)
pytest tests/test_strategies.py -v    # all 5 strategies (26 tests)
pytest tests/test_graph/ -v           # cypher, extractor, resolver (28 tests)
pytest tests/test_ingest.py -v        # all 3 parsers (18 tests)
pytest tests/test_router.py -v        # intent classification (12 tests)

# Integration tests (requires Neo4j + ChromaDB running)
pytest tests/integration/ -v

# API tests (mocked, no external dependencies)
pytest tests/api/ -v

# Live tests (requires API keys + Neo4j + ChromaDB)
ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... pytest tests/live/ -v

# Lint + format
ruff check . && ruff format --check .
```

| Suite | Tests | Requires |
|-------|-------|----------|
| `tests/test_*.py` | 177 | Nothing |
| `tests/api/` | ~30 | Nothing (mocked) |
| `tests/integration/` | ~50 | Neo4j, ChromaDB |
| `tests/live/` | ~80 | API keys, Neo4j, ChromaDB |

---

## Security

### API Security

- **Rate limiting** — In-memory per-IP rate limiter (60 requests/minute) on all chat endpoints
- **CORS** — Configurable allowed origins, defaults to `http://localhost:3000`. Never `*` in production
- **Error envelope** — Debug details only exposed when `KB_ARENA_DEBUG=true`. Production returns generic messages
- **Input validation** — All request bodies validated by Pydantic v2 with strict type checking

### Credential Management

- **No hardcoded secrets** — All API keys loaded from environment variables via `pydantic-settings`
- **Neo4j credentials** — Configurable via env vars, Docker secrets in production
- **LLM API keys** — Never logged, never included in error messages or benchmark output

### Dependencies

All 14 direct dependencies are pinned to exact versions in `pyproject.toml`:

```toml
dependencies = [
    "anthropic==0.42.0",
    "neo4j==5.27.0",
    "chromadb==0.5.23",
    "fastapi==0.115.6",
    "pydantic==2.10.4",
    # ...
]
```

### Database Security

- **Parameterized queries only** — No string interpolation in Cypher
- **Read-only access** from chatbot API — Queries use `MATCH`/`RETURN`, never `CREATE`/`DELETE`
- **Graph mutations** only during `build-graph` stage

---

## Contributing

### Development Setup

```bash
git clone https://github.com/xmpuspus/kb-arena
cd kb-arena
pip install -e '.[dev]'

# Start Neo4j for integration tests
docker compose up neo4j -d
```

### Code Style

Ruff handles both linting and formatting. Run before every commit:

```bash
ruff check .          # lint
ruff format .         # format
ruff check --fix .    # auto-fix
```

Configuration in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

### Adding a New Corpus

1. Create a parser in `kb_arena/ingest/parsers/` implementing document → `list[Document]`
2. Add node/rel enums in `kb_arena/graph/schema.py`
3. Register the schema in `_CORPUS_SCHEMA`
4. Write question YAML files in `datasets/{corpus}/questions/tier{1-5}_*.yaml`
5. Add tests for the parser and schema

### Adding a New Strategy

1. Create a class in `kb_arena/strategies/` extending `Strategy`
2. Implement `build_index(documents)` and `query(question, top_k)` methods
3. Register in `STRATEGY_REGISTRY` in `kb_arena/strategies/__init__.py`
4. Add to `get_strategy()` factory if it needs special initialization
5. Add to the chatbot lifespan in `kb_arena/chatbot/api.py`
6. Write tests in `tests/test_strategies.py`

### Writing Questions

Each question needs:

```yaml
- id: "corpus-tN-NNN"          # unique ID
  tier: 1-5                     # difficulty tier
  type: factoid|comparison|...  # question type
  hops: 1-5                     # reasoning hops required
  question: "..."               # the actual question
  ground_truth:
    answer: "..."               # reference answer
    source_refs: [...]          # expected source documents
    required_entities: [...]    # entities that must appear
  constraints:
    must_mention: [...]         # terms the answer must include
    must_not_claim: [...]       # false claims to detect
```

Guidelines:
- Tier 1-2 should be answerable by any strategy
- Tier 3-4 should specifically test graph traversal and relationships
- Tier 5 should require temporal reasoning (version changes, deprecations)
- `must_not_claim` should include common misconceptions, not obscure edge cases

### Commit Messages

Simple messages, no prefix format. Example: "Add SEC EDGAR benchmark questions"

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM (generation) | Claude Sonnet 4.6 | Answer generation, evaluation, entity extraction |
| LLM (classification) | Claude Haiku 4.5 | Intent classification (~20 tokens, <50ms) |
| Embeddings | text-embedding-3-large | 3072-dim vectors for ChromaDB |
| Vector store | ChromaDB 0.5 | Strategies 1-3 (persistent, local) |
| Graph store | Neo4j 5 (Community) | Strategy 4 (APOC plugin for graph algorithms) |
| Backend | FastAPI 0.115 + Uvicorn | Chatbot API with SSE streaming |
| Frontend | Next.js 14 + Tailwind | Side-by-side demo, benchmark tables, graph viz |
| Graph viz | Sigma.js | Interactive knowledge graph visualization |
| Charts | Recharts | Accuracy by tier charts |
| Models | Pydantic v2 | All data interchange — 15 models across 4 modules |
| CLI | Typer 0.12 + Rich | 7 pipeline commands |
| Entity resolution | Jellyfish | Jaro-Winkler fuzzy matching |
| Graph analysis | NetworkX | Graph statistics |
| Testing | pytest + pytest-asyncio | 339 tests across 18 files |
| Lint/Format | Ruff | Check + format in one tool |

---

## Repository Structure

```
kb-arena/
  kb_arena/            pip install kb-arena          CLI + library (41 Python modules, 10K lines)
  web/                 npm install && npm run dev    Next.js 14 frontend (10 components)
  datasets/            Question YAML files           200 questions, 3 corpora, 5 tiers
  tests/               pytest tests/                 339 tests across 18 files
  docker-compose.yml   docker compose up             Neo4j + API + Web
  pyproject.toml       Package metadata              Hatchling build, all deps pinned
```

---

## License

MIT
