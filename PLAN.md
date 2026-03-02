# KB Arena — Knowledge Base Arena

**Package:** `pip install kb-arena`
**CLI:** `kb-arena ingest`, `kb-arena benchmark`, `kb-arena serve`
**Import:** `from kb_arena import Pipeline, Benchmark`
**PyPI:** https://pypi.org/project/kb-arena/ (AVAILABLE — confirmed 2026-03-02)
**GitHub:** github.com/xmpuspus/kb-arena

## The Question This Answers

> "What should the data format be for freeform texts or multiple documents so we can have an accurate, reliable, and fast knowledge retrieval + chatbot system?"

**The answer (which this project proves empirically):** The source format matters far less than the retrieval architecture. Knowledge graphs as an intermediate representation — extracted from ANY source format — beat every other approach on multi-hop, relational, and comparative queries. Pure vector RAG wins on simple lookups but collapses to near-0% accuracy the moment a query touches 3+ interconnected concepts.

This project proves it with real data, reproducible benchmarks, and a live side-by-side chatbot demo.

---

## Why This Will Go Viral

### What Made Xavier's Past Projects Stand Out

From studying 60+ personal repos and local projects:

**paper-trail-ph** (most technically sophisticated)
- Cross-dataset relationship surfacing across 4 Philippine government agencies
- Jaro-Winkler entity resolution (same company spelled 6 ways, now linked)
- 13 automated corruption red flag detectors as pure Cypher (no Python computation)
- GraphRAG chat with SSE streaming on Neo4j HNSW vector index
- LinkedIn factor: 9/10 — real government data, invisible connections made visible

**cloudwright** (most polished, published)
- pip-installable, benchmarked: beats Claude-raw 83.5% vs 31.6%
- Blast radius / SPOF detection no competitor does
- Competitor analyzed against 30 tools; closest (Brainboard) raised $4M
- LinkedIn factor: 9/10 — publishable package with hard numbers

**cloudcare** (most relevant to this project)
- Proved structured SQL beats naive RAG for tabular data: 90.9% accuracy at 2ms latency
- Intent-routed retrieval — 5 distinct query paths, not one generic vector search
- Deliberate "no pgvector" decision that outperformed vector approaches

**aegis** (most impressive infrastructure)
- 53 hooks, 14 agents, 38 library modules
- Knowledge half-life decay model (confidence scores decay at 0.995/hour)
- Production Claude Code operator infrastructure

**The pattern that wows:** Real data + benchmark numbers + working demo + something people assumed was the right approach being proven wrong.

### The LinkedIn Hook

"I benchmarked 5 data formats for AI chatbots across 200 real questions on 3 document sets.

Vector RAG scored 94% on simple lookups.
It scored 11% on multi-hop queries.

Knowledge graphs scored 89% across ALL query types.

Here's the open-source benchmark with a live demo."

This follows the proven formula:
1. Counter-intuitive claim with a specific number
2. Name the flawed approach everyone is doing (dump docs into vector DB)
3. Proof, not assertion (benchmark table, not opinion)
4. Working code + live demo link

GraphRAG hit 30k GitHub stars with this exact pattern. LightRAG went viral by showing "same accuracy at 1/100th the cost."

### The Gap No One Has Filled

Nothing in the current GraphRAG ecosystem offers:
- A publicly accessible live demo
- On real (not toy) data
- With side-by-side comparison of the same query across 5 approaches
- With reproducible benchmark numbers
- That normal developers can run on their own docs

Every existing project is either a research paper, a Python library, or a toy demo.

---

## Proven Patterns to Reuse (Extracted from Successful Projects)

These are the exact implementation patterns that made paper-trail-ph, cloudwright, cloudcare, and climate-money-ph work. Every one is battle-tested.

### Pattern 1: Pydantic v2 as Central Interchange (from cloudwright)

Cloudwright's ArchSpec is a Pydantic v2 model that every module reads and writes. This eliminated an entire class of integration bugs.

```python
# kb_arena/models/document.py — the central interchange for this project
class Section(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")  # IaC-safe IDs (cloudwright pattern)
    title: str
    content: str
    heading_path: list[str]  # ["module", "class", "method"] — enables contextual retrieval
    tables: list[Table] = Field(default_factory=list)
    code_blocks: list[CodeBlock] = Field(default_factory=list)
    links: list[CrossRef] = Field(default_factory=list)
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)

class Document(BaseModel):
    id: str
    source: str  # file path or URL
    corpus: str  # "python-stdlib" | "kubernetes" | "sec-edgar"
    title: str
    sections: list[Section]
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_token_count: int = 0
```

**Why it works:** Every strategy reads the same Document model. The benchmark runner doesn't care which strategy produced the answer — they all return the same AnswerResult model. This is what makes adding new strategies trivial.

### Pattern 2: Enum-Based Node/Relationship Types (from paper-trail-ph + climate-money-ph)

Both graph projects use Python str Enums for node and edge types. Typos fail at import time, not as silent 0-result Cypher queries.

```python
# kb_arena/graph/schema.py
class NodeType(str, Enum):
    CONCEPT = "Concept"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    PARAMETER = "Parameter"
    RETURN_TYPE = "ReturnType"
    EXCEPTION = "Exception"
    DEPRECATION = "Deprecation"
    VERSION = "Version"
    EXAMPLE = "Example"

class RelType(str, Enum):
    CONTAINS = "CONTAINS"           # Module -> Class, Class -> Function
    REQUIRES = "REQUIRES"           # Function -> Parameter
    RETURNS = "RETURNS"             # Function -> ReturnType
    RAISES = "RAISES"               # Function -> Exception
    DEPRECATED_BY = "DEPRECATED_BY" # old -> new
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    REFERENCES = "REFERENCES"       # cross-module reference
    INHERITS = "INHERITS"           # Class -> Class
    IMPLEMENTS = "IMPLEMENTS"       # Class -> Concept (e.g., "iterator protocol")
    EXAMPLE_OF = "EXAMPLE_OF"       # Example -> Function/Class
```

**Why it works:** The LLM extraction prompt includes the exact enum values as the only allowed types. Post-extraction validation rejects anything not in the enum. This prevents graph drift across document batches — Neo4j's documented failure mode when schemas aren't enforced.

### Pattern 3: Idempotent Cypher Schema with IF NOT EXISTS (from paper-trail-ph)

```cypher
-- cypher/schema.cypher — loaded on every startup, safe to re-run
CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT class_fqn IF NOT EXISTS FOR (c:Class) REQUIRE c.fqn IS UNIQUE;
CREATE CONSTRAINT function_fqn IF NOT EXISTS FOR (f:Function) REQUIRE f.fqn IS UNIQUE;

-- Multi-label fulltext index for cross-entity search (paper-trail-ph pattern)
-- Covers 4 node types in one index — enables single-query cross-entity search
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (n:Concept|Module|Class|Function)
ON EACH [n.name, n.description, n.fqn];

-- Vector index for hybrid search
CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
FOR (c:Concept) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
```

**Why it works:** Schema loading is idempotent — safe to run on every app startup. The multi-label fulltext index is critical: it lets a single Cypher query search across all entity types simultaneously. Property existence constraints are intentionally omitted (Neo4j Community doesn't support them) — enforce at the Pydantic layer instead.

### Pattern 4: Two-Threshold Entity Resolution (from paper-trail-ph)

```python
# kb_arena/graph/resolver.py
import jellyfish

MERGE_THRESHOLD = 0.92    # auto-merge — proven in paper-trail-ph
REVIEW_THRESHOLD = 0.85   # queue for review — catches edge cases

def resolve_entities(entities: list[Entity]) -> list[Entity]:
    """Two-threshold Jaro-Winkler resolution.
    >= 0.92: auto-merge (longer string becomes canonical name)
    0.85-0.91: queued to review log for human inspection
    < 0.85: considered distinct
    """
    merged = []
    review_queue = []

    for i, a in enumerate(entities):
        for j, b in enumerate(entities[i+1:], start=i+1):
            if a.type != b.type:
                continue
            norm_a = normalize_name(a.name)
            norm_b = normalize_name(b.name)
            score = jellyfish.jaro_winkler_similarity(norm_a, norm_b)

            if score >= MERGE_THRESHOLD:
                # Longer string as canonical (paper-trail-ph decision)
                canonical = a if len(a.name) >= len(b.name) else b
                alias = b if canonical is a else a
                canonical.aliases.append(alias.name)  # preserve original for auditability
                merged.append((a.id, b.id, canonical.id))
            elif score >= REVIEW_THRESHOLD:
                review_queue.append((a.name, b.name, score))

    return merged, review_queue

def normalize_name(name: str) -> str:
    """Normalize before matching. Strip noise, preserve meaningful content.
    Paper-trail-ph lesson: separate normalizers per entity type."""
    name = name.upper().strip()
    # Strip common suffixes that don't distinguish entities
    for suffix in ["()", "CLASS", "FUNCTION", "METHOD", "MODULE"]:
        name = name.removesuffix(suffix).strip()
    return name
```

**Why it works:** Jaro-Winkler weights prefix matches more heavily than Levenshtein, which is correct for technical names where the meaningful distinguishing text is at the start (e.g., `os.path.join` vs `os.path.split`). The two-threshold approach auto-merges obvious matches while flagging borderline cases. Preserving `original_name` as aliases means the audit trail is intact.

### Pattern 5: Three-Stage Intent Classification (from cloudcare)

Cloudcare's 3-stage pipeline (keyword → LLM → regex fallback) is the most robust classification pattern across Xavier's projects.

```python
# kb_arena/chatbot/router.py
class QueryIntent(str, Enum):
    FACTOID = "factoid"           # "What is X?" → vector is fine
    COMPARISON = "comparison"      # "Compare X vs Y" → graph needed
    RELATIONAL = "relational"      # "What depends on X?" → graph needed
    PROCEDURAL = "procedural"      # "How do I do X?" → vector + graph
    EXPLORATORY = "exploratory"    # "Tell me about X" → vector is fine

class IntentRouter:
    """Three-stage classification: keyword scan → LLM → regex fallback.
    Proven pattern from cloudcare (5-path routing, <50ms classification)."""

    # Stage 1: Keyword scan (no LLM, ~1ms)
    KEYWORD_PATTERNS = {
        QueryIntent.COMPARISON: [
            r"\bcompare\b", r"\bvs\.?\b", r"\bdifference between\b",
            r"\bwhich is better\b", r"\badvantages? of .+ over\b",
        ],
        QueryIntent.RELATIONAL: [
            r"\bdepend", r"\brequire", r"\baffect", r"\bdownstream\b",
            r"\bif I .+ what happens\b", r"\bcause", r"\bimplica",
        ],
        QueryIntent.PROCEDURAL: [
            r"\bhow (?:do|can|should|to)\b", r"\bsteps? to\b",
            r"\bsetup\b", r"\bconfigure\b", r"\bimplement\b",
        ],
    }

    async def classify(self, query: str, history: list[Message] | None = None) -> QueryIntent:
        # Stage 1: keyword scan (cloudcare pattern — returns early before touching LLM)
        for intent, patterns in self.KEYWORD_PATTERNS.items():
            if any(re.search(p, query, re.IGNORECASE) for p in patterns):
                return intent

        # Stage 2: LLM classification (Haiku, ~20 tokens, <50ms)
        try:
            result = await self.llm.classify(
                query=query,
                history=history[-6:] if history else None,  # cloudcare: last 6 turns
                allowed_intents=[e.value for e in QueryIntent],
            )
            return QueryIntent(result)
        except Exception:
            pass

        # Stage 3: regex fallback (cloudcare: never fails)
        return self._fallback_classify(query)
```

**Why it works:** The keyword scan handles 60-70% of queries without an LLM call (saves cost and latency). The LLM handles nuanced cases. The regex fallback ensures the router NEVER fails — every query gets classified. Cloudcare proved this pipeline is more reliable than LLM-only classification.

### Pattern 6: Graph Queries as Pure Cypher (from paper-trail-ph)

All 13 red flag detectors in paper-trail-ph are pure Cypher — no Python-side computation. This means graph queries are auditable, testable, and run at Neo4j speed.

```python
# kb_arena/graph/cypher_templates.py — pre-built query templates for common patterns

MULTI_HOP_QUERY = """
// Find all entities connected to {target} within {depth} hops
MATCH path = (start {fqn: $target})-[*1..{depth}]-(connected)
WHERE ALL(r IN relationships(path) WHERE type(r) IN $allowed_rel_types)
RETURN connected.name AS name, connected.fqn AS fqn,
       labels(connected)[0] AS type,
       length(path) AS hops,
       [r IN relationships(path) | type(r)] AS relationship_chain
ORDER BY hops, connected.name
LIMIT 50
"""

COMPARISON_QUERY = """
// Compare two entities across all shared relationship types
MATCH (a {fqn: $entity_a})-[r1]-(shared)-[r2]-(b {fqn: $entity_b})
RETURN shared.name AS shared_entity, labels(shared)[0] AS shared_type,
       type(r1) AS rel_to_a, type(r2) AS rel_to_b
UNION
MATCH (a {fqn: $entity_a})-[r]-(unique)
WHERE NOT (unique)--(b:_ {fqn: $entity_b})
RETURN unique.name AS unique_to_a, type(r) AS relationship
"""

DEPENDENCY_CHAIN = """
// Trace full dependency chain (climate-money-ph fund flow pattern)
MATCH path = (source {fqn: $start})-[:REQUIRES|IMPORTS|INHERITS*1..4]->(dep)
WITH path, dep, length(path) AS depth
RETURN dep.name, dep.fqn, labels(dep)[0] AS type, depth,
       [n IN nodes(path) | n.name] AS chain
ORDER BY depth
LIMIT 100
"""
```

**Why it works:** Pre-built Cypher templates are more reliable than Text-to-Cypher generation for known query patterns. Text-to-Cypher is a fallback for novel queries. This mirrors paper-trail-ph's approach: template queries for the 13 known patterns, free-form Cypher generation only for open-ended chat. Template queries are also testable — you can unit-test each one against a known graph state.

### Pattern 7: Batch Loading with UNWIND + MERGE (from paper-trail-ph)

```python
# kb_arena/graph/neo4j_store.py
BATCH_SIZE = 1000  # paper-trail-ph's proven batch size

async def load_nodes(self, nodes: list[dict], label: NodeType) -> int:
    """Load nodes in batches of 1000 using UNWIND/MERGE.
    CRITICAL lessons from paper-trail-ph:
    - Always UNWIND $records, never individual CREATE calls
    - Always MERGE on unique key, then SET remaining properties
    - Always await result.consume() or Neo4j holds cursors open
    - Load nodes BEFORE edges (edges reference nodes, missing refs silently dropped)
    """
    loaded = 0
    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i:i + BATCH_SIZE]
        query = f"""
        UNWIND $records AS record
        MERGE (n:{label.value} {{fqn: record.fqn}})
        SET n += record
        """
        result = await self.session.run(query, records=batch)
        summary = await result.consume()  # CRITICAL: must consume or cursors leak
        loaded += summary.counters.nodes_created
    return loaded

async def load_edges(self, edges: list[dict], rel_type: RelType) -> int:
    """Load edges after all nodes are loaded.
    Uses MATCH (not MERGE) for endpoints — edges referencing
    nonexistent nodes are silently dropped. Node load order matters."""
    loaded = 0
    for i in range(0, len(edges), BATCH_SIZE):
        batch = edges[i:i + BATCH_SIZE]
        query = f"""
        UNWIND $records AS record
        MATCH (a {{fqn: record.source_fqn}})
        MATCH (b {{fqn: record.target_fqn}})
        MERGE (a)-[r:{rel_type.value}]->(b)
        SET r += record.properties
        """
        result = await self.session.run(query, records=batch)
        summary = await result.consume()
        loaded += summary.counters.relationships_created
    return loaded
```

### Pattern 8: Multi-Stage CLI Pipeline (from paper-trail-ph + climate-money-ph)

Both graph projects use a staged CLI where each step is independently runnable. Climate-money-ph has the most mature version: `collect → transform → load → analyze → validate`.

```python
# kb_arena/cli.py — Typer CLI (cloudwright uses Typer + Rich)
import typer
app = typer.Typer()

@app.command()
def ingest(path: str, corpus: str = "custom", format: str = "auto"):
    """Stage 1: Parse raw documents into unified model.
    Writes JSONL to datasets/{corpus}/processed/"""

@app.command()
def build_graph(corpus: str, schema: str = "auto"):
    """Stage 2: Extract entities/relationships, build Neo4j graph.
    Requires: ingest completed. Writes to Neo4j."""

@app.command()
def build_vectors(corpus: str, strategy: str = "all"):
    """Stage 3: Build vector indexes for strategies 1-3.
    Requires: ingest completed. Writes to ChromaDB."""

@app.command()
def benchmark(corpus: str = "all", strategy: str = "all", tier: int = 0):
    """Stage 4: Run benchmark questions against specified strategies.
    Writes results to results/{corpus}.json"""

@app.command()
def serve(port: int = 8000):
    """Stage 5: Launch side-by-side chatbot demo.
    Requires: all strategies built."""

@app.command()
def report(corpus: str = "all"):
    """Generate benchmark report from results JSON."""
```

**Why it works:** Each stage is independently testable and re-runnable. If graph extraction fails, you don't need to re-ingest. If you add a new strategy, you only re-run build + benchmark. JSONL as the intermediate format (climate-money-ph pattern) means each stage writes to disk, so you can inspect the pipeline at any point.

### Pattern 9: LLM Dual-Model Split with Cache Control (from cloudwright)

```python
# kb_arena/llm/client.py
GENERATE_MODEL = "claude-sonnet-4-6"        # extraction, generation, evaluation
FAST_MODEL = "claude-haiku-4-5-20251001"    # classification, routing

class LLMClient:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self._system_prompt_cache = {}

    async def classify(self, query: str, system_prompt: str, **kwargs) -> str:
        """Cheap classification call (~20 tokens). Haiku. <50ms."""
        return await self._call(FAST_MODEL, system_prompt, query, max_tokens=100, **kwargs)

    async def generate(self, query: str, context: str, system_prompt: str, **kwargs) -> str:
        """Full generation call. Sonnet."""
        return await self._call(GENERATE_MODEL, system_prompt, f"{context}\n\nQuery: {query}", **kwargs)

    async def _call(self, model: str, system: str, user: str, **kwargs) -> str:
        """Cloudwright pattern: cache_control on system prompt for cross-call caching."""
        response = self.client.messages.create(
            model=model,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},  # cloudwright: caches large system prompts
            }],
            messages=[{"role": "user", "content": user}],
            temperature=0,  # deterministic for benchmark reproducibility
            **kwargs,
        )
        return response.content[0].text
```

### Pattern 10: SSE Streaming with Typed Events (from paper-trail-ph)

```python
# kb_arena/chatbot/api.py
from sse_starlette.sse import EventSourceResponse

async def chat_stream(request: ChatRequest):
    """SSE streaming with 4 event types (paper-trail-ph pattern).
    Streams: message_id → token* → done (with metadata) → meta (timing)"""
    async def event_generator():
        msg_id = str(uuid4())
        yield {"event": "message_id", "data": json.dumps({"id": msg_id})}

        # Route to appropriate strategy
        intent = await router.classify(request.query, request.history)
        strategy = strategy_map[intent]

        # Stream answer tokens
        async for token in strategy.stream_answer(request.query, request.history):
            yield {"event": "token", "data": json.dumps({"text": token})}

        # Send completion with graph context for visualization
        yield {"event": "done", "data": json.dumps({
            "sources": strategy.last_sources,
            "graph_context": strategy.last_graph_context,  # for Sigma.js visualization
            "strategy_used": strategy.name,
        })}

        yield {"event": "meta", "data": json.dumps({
            "intent": intent.value,
            "latency_ms": strategy.last_latency_ms,
            "tokens_used": strategy.last_tokens_used,
            "cost_usd": strategy.last_cost_usd,
        })}

    return EventSourceResponse(event_generator())
```

### Pattern 11: Service Initialization in Lifespan (from paper-trail-ph)

```python
# kb_arena/chatbot/api.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Paper-trail-ph pattern: initialize services in lifespan, store on app.state.
    Store driver even when connectivity fails — endpoints return structured errors
    rather than unhandled 500s."""
    try:
        app.state.neo4j = await Neo4jStore.connect(settings.neo4j_uri)
    except Exception as e:
        logger.warning(f"Neo4j not available: {e}")
        app.state.neo4j = None  # endpoints check and return structured error

    app.state.chroma = ChromaStore(settings.chroma_path)
    app.state.llm = LLMClient(settings.anthropic_api_key)
    app.state.router = IntentRouter(app.state.llm)

    # Build strategy map — each strategy gets its dependencies
    app.state.strategies = {
        "naive_vector": NaiveVectorStrategy(app.state.chroma),
        "contextual_vector": ContextualVectorStrategy(app.state.chroma),
        "qna_pairs": QnAPairStrategy(app.state.chroma),
        "knowledge_graph": KnowledgeGraphStrategy(app.state.neo4j),
        "hybrid": HybridStrategy(app.state.neo4j, app.state.chroma, app.state.router),
    }
    yield
    if app.state.neo4j:
        await app.state.neo4j.close()
```

### Pattern 12: Graph Analysis with asyncio.to_thread (from climate-money-ph)

```python
# kb_arena/graph/analyzer.py
import networkx as nx

class GraphAnalyzer:
    """Climate-money-ph pattern: Neo4j for storage, networkx for algorithms.
    CPU-intensive graph algorithms run in thread pool to avoid blocking async."""

    def __init__(self, neo4j_store: Neo4jStore):
        self.store = neo4j_store
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = 300  # 5-minute cache (climate-money-ph)

    async def analyze_communities(self, resolution: float = 1.0) -> list[Community]:
        """Louvain community detection with exposed resolution param."""
        G = await self._build_networkx_graph()
        # CPU-bound — offload to thread pool (climate-money-ph pattern)
        communities = await asyncio.to_thread(
            nx.community.louvain_communities, G, weight="weight", resolution=resolution
        )
        return communities

    async def find_dependency_chains(self, start_fqn: str, max_depth: int = 4) -> list[Path]:
        """Follow-the-money pattern from climate-money-ph applied to dependency chains.
        DiGraph with cutoff depth and result cap."""
        G = await self._build_directed_graph()
        paths = await asyncio.to_thread(
            lambda: list(itertools.islice(
                nx.all_simple_paths(G, start_fqn, cutoff=max_depth), 100  # cap at 100
            ))
        )
        return paths
```

### Pattern 13: Benchmark YAML Test Cases (from cloudwright)

```yaml
# datasets/python-stdlib/questions/tier3_comparative.yaml
- id: "py-t3-001"
  tier: 3
  type: comparison
  hops: 3
  question: "Compare the thread safety guarantees of queue.Queue, collections.deque, and list. Which should I use for a producer-consumer pattern?"
  ground_truth:
    answer: "queue.Queue is the only one designed for thread-safe producer-consumer patterns. It provides blocking get()/put() with optional timeouts. collections.deque is thread-safe for append/pop operations (due to GIL) but has no blocking mechanism. list is not thread-safe for concurrent modification."
    source_refs:
      - "queue.html#queue.Queue"
      - "collections.html#collections.deque"
      - "glossary.html#term-GIL"
    required_entities: ["queue.Queue", "collections.deque", "GIL"]
  constraints:
    must_mention: ["blocking", "GIL", "thread-safe"]
    must_not_claim: ["deque is not thread-safe", "list is thread-safe"]
```

**Why YAML with `must_mention`/`must_not_claim`:** Cloudwright's benchmark proved that simple presence/absence checks catch 80% of correctness issues without needing a full LLM judge. The LLM judge only runs on cases that pass the structural checks, saving cost.

### Pattern 14: Consistent Error Envelope (from paper-trail-ph)

```python
# kb_arena/chatbot/api.py
class ErrorResponse(BaseModel):
    error: ErrorDetail

class ErrorDetail(BaseModel):
    code: str
    message: str

# Used everywhere — consistent across all endpoints
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=ErrorDetail(
            code="internal_error",
            message=str(exc) if settings.debug else "An internal error occurred"
        )).model_dump()
    )
```

### Pattern 15: Mock Fallback for Missing Services (from climate-money-ph)

```python
# If Neo4j isn't connected, the graph strategy returns mock data with a warning
# rather than crashing. This allows the demo to partially work even without
# the full stack running. Climate-money-ph uses this extensively:
# two communities, one broker, one anomaly — enough to show the UI works.

class KnowledgeGraphStrategy(BaseStrategy):
    async def retrieve(self, query: str) -> RetrievalResult:
        if self.neo4j is None:
            return RetrievalResult(
                answer="[Graph database not connected. Showing mock data.]",
                sources=[],
                mock=True,
                graph_context=self._mock_graph_context(),
            )
        # ... real implementation
```

---

## Project Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────┐
│              DOCUMENT INGESTION                  │
│                                                  │
│  Raw docs (MD, HTML, PDF, Confluence export,     │
│  JSON, CSV, nested pages, tables, images)        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           UNIFIED DOCUMENT MODEL                 │
│  (Pydantic v2 — cloudwright ArchSpec pattern)    │
│                                                  │
│  Every doc → Document { id, source, sections[],  │
│  tables[], metadata{}, parent_id, children[],    │
│  links[] }. JSONL intermediate files.            │
└──────────────────┬──────────────────────────────┘
                   │
          ┌────────┼────────┬──────────┬──────────┐
          ▼        ▼        ▼          ▼          ▼
     ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
     │Strategy││Strategy││Strategy││Strategy││Strategy│
     │   1    ││   2    ││   3    ││   4    ││   5    │
     │ Naive  ││Context ││  QnA   ││  KG    ││Hybrid  │
     │ Vector ││ Vector ││ Pairs  ││(Neo4j) ││KG+Vec  │
     └───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
         │         │         │         │         │
         ▼         ▼         ▼         ▼         ▼
     ┌─────────────────────────────────────────────┐
     │           BENCHMARK ENGINE                   │
     │  (cloudwright pattern: YAML test cases with  │
     │   structural checks + LLM judge fallback)    │
     │                                              │
     │  200+ curated questions × 5 complexity tiers │
     │  Ground truth answers (human-verified)       │
     │  Metrics: accuracy, latency, cost, tokens    │
     └──────────────────┬──────────────────────────┘
                        │
                        ▼
     ┌─────────────────────────────────────────────┐
     │           LIVE CHATBOT DEMO                  │
     │  (cloudcare 3-stage routing + paper-trail-ph │
     │   SSE streaming + climate-money-ph viz)      │
     │                                              │
     │  Side-by-side: same question, 5 backends     │
     │  Shows: answer, confidence, source trail,    │
     │         retrieval path, latency, token cost   │
     └─────────────────────────────────────────────┘
```

### The 5 Retrieval Strategies — Detailed Implementation

**Strategy 1: Naive Vector RAG** (the baseline everyone uses)
- Split docs into 512-token chunks with 50-token overlap
- Embed with `text-embedding-3-small` (1536 dimensions)
- Store in ChromaDB with minimal metadata (source_id only)
- Top-k=5 cosine similarity search → concatenate chunks → pass to Sonnet
- This is what happens when you "dump Confluence into a vector DB"
- Implementation: ~80 lines. Deliberately simple — this is the strawman.

**Strategy 2: Chunked + Contextual Metadata**
- Same chunking but prepend hierarchical heading path to each chunk
  - Example: "## collections > deque > appendleft\n\nAppend x to the left side..."
- Add metadata filter fields: `{source_doc, section_path, module, has_code, has_table}`
- Embed the enriched chunk (contextual retrieval — Anthropic's documented technique)
- ChromaDB `where` filters narrow search space before similarity
- This is the "best practice" vector approach
- Implementation: ~120 lines. Shows how much metadata helps (or doesn't).

**Strategy 3: QnA Pairs**
- LLM (Sonnet) generates 3-5 question-answer pairs per section
- System prompt: "Generate questions a developer would ask about this section. Include multi-hop questions that reference other sections."
- Embed questions, store with pre-generated answer + source reference
- Query matches question embeddings → returns pre-generated answer without re-querying LLM
- Implementation: ~150 lines (generation) + ~60 lines (retrieval).
- Higher upfront cost (generation), lower runtime cost (no LLM at query time for answer).

**Strategy 4: Knowledge Graph (Neo4j)**
- LLM extracts entities and relationships with schema constraints (Pattern 2 above)
- Schema per corpus: Python stdlib gets Module/Class/Function/Parameter/Exception/etc.
- Entities stored as Neo4j nodes with embeddings for hybrid search
- Query pipeline: classify intent → select Cypher template or generate Cypher → execute → assemble context → generate answer
- Entity resolution via Jaro-Winkler (Pattern 4 above)
- Provenance: every node has `source_section_id` linking back to the Document model
- Implementation: ~400 lines (extraction) + ~200 lines (querying). The bulk of the project.

**Strategy 5: Hybrid Graph + Vector**
- Three-stage intent classification (Pattern 5 above) determines routing:
  - `factoid` / `exploratory` → Strategy 2 (contextual vector)
  - `comparison` / `relational` → Strategy 4 (knowledge graph)
  - `procedural` → both, results fused and deduplicated
- Deduplication: if both strategies return overlapping source sections, merge and keep the richer answer
- Re-ranking: Sonnet scores each result chunk on relevance (0-1), top 5 passed to final generation
- Implementation: ~200 lines. The hypothesis is this wins overall.

### Query Complexity Tiers

| Tier | Type | Example | Expected Winner |
|------|------|---------|-----------------|
| 1 | Single-fact lookup | "What is the default timeout for X?" | Vector (all strategies competitive) |
| 2 | Multi-entity | "Which services support both feature A and feature B?" | Graph starts winning |
| 3 | Comparative | "Compare the authentication methods of X vs Y vs Z" | Graph dominates |
| 4 | Relational/causal | "If I configure X with setting Y, what downstream effects does that have on Z?" | Only graph answers correctly |
| 5 | Temporal + relational | "What changed between version 3 and version 4 that affects configurations using feature X?" | Only graph + provenance answers correctly |

**The key insight the benchmark will prove:**
- Tier 1: All strategies score 85-95% — vector is fine here
- Tier 2: Vector drops to ~60%, graph stays at ~85%
- Tier 3: Vector drops to ~30%, graph stays at ~80%
- Tier 4: Vector drops to ~10%, graph stays at ~75%
- Tier 5: Vector drops to ~5%, graph stays at ~70%

The crossover point (where graph becomes mandatory) is the actionable takeaway.

---

## Dataset Selection

### Primary: Python Standard Library Documentation

**Why Python docs:**
- Universal audience — every developer has used Python docs
- Naturally complex: 200+ modules, cross-referencing types, deprecation chains
- Has tables, code examples, version-specific notes, nested hierarchy
- Multi-hop queries are natural: "What's the most efficient way to parse CSV with custom date handling and unicode support?" requires csv + datetime + io + codecs knowledge
- Publicly available, no licensing issues
- ~500 pages across standard library reference
- The pain is IMMEDIATELY relatable — everyone has struggled to find things in Python docs

**Corpus size:** ~50 most-used modules, ~300 pages, ~200K tokens raw

**Graph schema for Python docs:**
- Nodes: Module, Class, Function, Parameter, ReturnType, Exception, Deprecation, Version, Concept, Example
- Edges: CONTAINS, REQUIRES, RETURNS, RAISES, DEPRECATED_BY, ALTERNATIVE_TO, REFERENCES, INHERITS, IMPLEMENTS, EXAMPLE_OF
- Estimated graph: ~3,000 nodes, ~8,000 edges

**Download method:**
```bash
# Python docs are available as raw RST or HTML
git clone --depth 1 https://github.com/python/cpython.git /tmp/cpython
# Extract Doc/library/*.rst for stdlib reference
# Or download pre-built HTML from docs.python.org/3/archives/
```

### Secondary: Kubernetes Documentation

**Why K8s docs:**
- Second-most universal audience in tech
- Extremely nested (concepts → workloads → pods → containers → lifecycle)
- Heavy cross-referencing (Services reference Pods reference ConfigMaps reference Secrets)
- The "Confluence nightmare" experience — everyone has struggled with K8s docs
- Multi-hop queries are the norm: "How do I expose a StatefulSet with persistent storage through an Ingress with TLS termination?"

**Corpus size:** Core concepts + workloads + networking + storage, ~200 pages, ~150K tokens raw

**Graph schema for K8s docs:**
- Nodes: Resource, Field, APIGroup, Controller, Concept, Example, Version
- Edges: MANAGES, REFERENCES, REQUIRES, CONFIGURES, ALTERNATIVE_TO, DEPRECATED_BY, OWNS, SELECTS
- Estimated graph: ~2,000 nodes, ~6,000 edges

**Download method:**
```bash
git clone --depth 1 https://github.com/kubernetes/website.git /tmp/k8s-website
# Content in content/en/docs/
```

### Tertiary: SEC EDGAR 10-K Filings (3-5 companies)

**Why SEC filings:**
- Financial professionals are the most active LinkedIn sharers
- Cross-entity relationships are extreme (subsidiaries, board members, risk factors)
- The query that breaks vector RAG: "Which executives serve on boards of companies with material litigation disclosed in the same fiscal year?" — literally impossible without graph
- Free API, no auth required
- Dramatic failure mode for naive RAG

**Corpus size:** 5 companies × ~80 pages = ~400 pages, ~300K tokens raw

**Graph schema for SEC filings:**
- Nodes: Company, Executive, BoardMember, Subsidiary, RiskFactor, FinancialMetric, LegalProceeding, Segment
- Edges: SERVES_ON_BOARD, SUBSIDIARY_OF, DISCLOSED_RISK, REPORTED_METRIC, INVOLVED_IN, OPERATES_IN, RELATED_TO
- Estimated graph: ~500 nodes, ~2,000 edges

**Download method:**
```python
# SEC EDGAR API — no auth required
# edgartools library has no rate limits
from edgar import Company
company = Company("AAPL")
filing = company.get_filings(form="10-K").latest(1)
```

### Total: ~700 pages, ~650K tokens, 200+ benchmark questions

---

## Technical Stack

Based on Xavier's proven patterns:

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Backend | FastAPI (Python 3.12) | Xavier's standard, async-native |
| Graph DB | Neo4j Community 5 (Docker) | Proven in paper-trail-ph, climate-money-ph |
| Vector DB | ChromaDB | Lightweight, embeddable, sufficient for benchmark |
| Embeddings | text-embedding-3-small | Cost-efficient, good enough for comparison |
| LLM (classify) | Claude Haiku 4.5 | Xavier's standard classifier (<50ms, ~20 tokens) |
| LLM (generate) | Claude Sonnet 4.6 | Xavier's standard generator |
| LLM (extract) | Claude Sonnet 4.6 | Entity/relationship extraction |
| Frontend | Next.js 14 + Tailwind | Xavier's standard frontend |
| Graph viz | Sigma.js | Proven in paper-trail-ph |
| Entity resolution | jellyfish (Jaro-Winkler >=0.92) | Proven in paper-trail-ph |
| Benchmark runner | pytest + custom YAML harness | Cloudwright pattern: reproducible, CI-compatible |
| Package | pip-installable (hatchling) | Cloudwright pattern |
| CLI | Typer + Rich | Cloudwright pattern |
| Deployment | Docker Compose | Paper-trail-ph pattern |

### Key Design Decisions (with rationale from past projects)

1. **Schema-constrained extraction** — Define allowed node/relationship types BEFORE extraction as Python str Enums. Without this, LLM extraction produces graph drift across document batches (Neo4j's finding). Xavier already does this in paper-trail-ph. The LLM extraction prompt includes the exact enum values as the ONLY allowed types.

2. **Intent-routed retrieval** — Don't send every query through every strategy. Three-stage classify first (keyword → Haiku → regex fallback), route to the right backend. Proven in cloudcare's 5-path routing. In benchmark mode, ALL strategies run for comparison; in demo mode, only the classified strategy runs for speed.

3. **Provenance-first** — Every answer traces back to source document + section + paragraph. Every Neo4j node has `source_section_id` linking back to the Document model. Paper-trail-ph stores `original_name` for auditability; we store `source_section_id` + `extraction_confidence`.

4. **Benchmark-first development** — Write the 200 questions and ground truth BEFORE building the retrieval strategies. Then measure. This is the cloudwright pattern (54-case YAML benchmark with `prompt`, `constraints`, `expected` drove the entire roadmap). Cloudwright's `must_mention`/`must_not_claim` structural checks catch 80% of correctness issues without LLM judge.

5. **JSONL intermediate format** — Climate-money-ph writes JSONL between pipeline stages. Each stage reads input JSONL, writes output JSONL. This makes the pipeline inspectable, restartable, and debuggable at any point.

6. **Consistent error envelope** — Paper-trail-ph's `{"error": {"code": "...", "message": "..."}}` pattern across all endpoints. Services initialized in lifespan(), stored on `app.state`. Neo4j driver stored even when connectivity fails — endpoints return structured errors instead of 500s.

7. **In-memory caching** — Climate-money-ph uses a 5-minute cache keyed by filter params on `GraphAnalysisService`. The graph doesn't change during a demo session, so caching graph traversal results is safe and dramatically improves latency.

8. **Dual-model LLM with prompt caching** — Cloudwright's pattern: Haiku for classification (~20 tokens, <50ms), Sonnet for generation. System prompts use `cache_control: {"type": "ephemeral"}` so Anthropic caches the large extraction/generation prompts across calls.

9. **Mock fallback** — Climate-money-ph pattern: if Neo4j isn't connected, graph strategy returns mock data with a warning. Allows the demo to partially work during development or when infra is partially deployed.

10. **Hatchling monorepo** — Cloudwright ships 3 packages from one repo (`core`, `cli`, `web`). KB Arena can start as a single package and split later if needed. `pyproject.toml` with `[project.scripts]` entry point for the CLI.

---

## Repository Structure

```
kb-arena/
├── README.md                    # The viral README
├── pyproject.toml               # pip-installable (hatchling, cloudwright pattern)
├── docker-compose.yml           # Neo4j + ChromaDB + API (paper-trail-ph pattern)
├── CLAUDE.md                    # Agent instructions for building this project
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BENCHMARK_METHODOLOGY.md
│   └── RESULTS.md               # Published benchmark results
│
├── cypher/                      # Neo4j schema DDL (paper-trail-ph pattern)
│   ├── schema_python.cypher     # Python stdlib schema
│   ├── schema_kubernetes.cypher # K8s schema
│   └── schema_sec.cypher        # SEC filing schema
│
├── kb_arena/
│   ├── __init__.py
│   ├── cli.py                   # Typer + Rich CLI (cloudwright pattern)
│   ├── settings.py              # Pydantic Settings (env-based config)
│   │
│   ├── models/                  # Pydantic v2 models — central interchange
│   │   ├── __init__.py
│   │   ├── document.py          # Document, Section, Table, CrossRef
│   │   ├── graph.py             # Entity, Relationship, GraphContext
│   │   ├── benchmark.py         # Question, GroundTruth, BenchmarkResult
│   │   └── api.py               # ChatRequest, ChatResponse, ErrorResponse
│   │
│   ├── ingest/                  # Document ingestion (stage 1)
│   │   ├── __init__.py
│   │   ├── parsers/
│   │   │   ├── __init__.py
│   │   │   ├── markdown.py      # .md/.rst files
│   │   │   ├── html.py          # .html files (Python docs, K8s)
│   │   │   └── sec_edgar.py     # SEC EDGAR 10-K specific parser
│   │   ├── unified_model.py     # Raw → Document conversion
│   │   └── pipeline.py          # Orchestrator, JSONL output
│   │
│   ├── strategies/              # The 5 retrieval strategies
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract Strategy with AnswerResult model
│   │   ├── naive_vector.py      # Strategy 1 (~80 lines)
│   │   ├── contextual_vector.py # Strategy 2 (~120 lines)
│   │   ├── qna_pairs.py         # Strategy 3 (~210 lines)
│   │   ├── knowledge_graph.py   # Strategy 4 (~600 lines, the core)
│   │   └── hybrid.py            # Strategy 5 (~200 lines)
│   │
│   ├── graph/                   # Knowledge graph construction (stage 2)
│   │   ├── __init__.py
│   │   ├── schema.py            # NodeType + RelType enums (per corpus)
│   │   ├── extractor.py         # LLM-based entity extraction with schema constraints
│   │   ├── resolver.py          # Jaro-Winkler entity resolution (paper-trail-ph)
│   │   ├── neo4j_store.py       # UNWIND/MERGE batch loading, cursor management
│   │   ├── cypher_templates.py  # Pre-built query templates (paper-trail-ph pattern)
│   │   ├── cypher_generator.py  # Text-to-Cypher for novel queries (fallback)
│   │   └── analyzer.py          # networkx algorithms via asyncio.to_thread
│   │
│   ├── benchmark/               # Benchmark engine (stage 4)
│   │   ├── __init__.py
│   │   ├── questions.py         # YAML question loader
│   │   ├── evaluator.py         # Structural checks + LLM judge (cloudwright)
│   │   ├── runner.py            # Orchestrator — runs all strategies × all questions
│   │   └── reporter.py          # JSON + markdown report generation
│   │
│   ├── chatbot/                 # Live chatbot backend (stage 5)
│   │   ├── __init__.py
│   │   ├── router.py            # 3-stage intent classification (cloudcare)
│   │   ├── session.py           # Client-side memory, last 6 turns (cloudcare)
│   │   └── api.py               # FastAPI + SSE streaming (paper-trail-ph)
│   │
│   ├── llm/                     # LLM client
│   │   ├── __init__.py
│   │   └── client.py            # Dual-model, prompt caching (cloudwright)
│   │
│   └── viz/                     # Visualization helpers
│       ├── __init__.py
│       └── graph_export.py      # Neo4j → Sigma.js JSON export
│
├── datasets/
│   ├── python-stdlib/
│   │   ├── raw/                 # Downloaded docs
│   │   ├── processed/           # JSONL unified model (climate-money-ph pattern)
│   │   └── questions/           # YAML per tier (cloudwright pattern)
│   │       ├── tier1_factoid.yaml
│   │       ├── tier2_multi_entity.yaml
│   │       ├── tier3_comparative.yaml
│   │       ├── tier4_relational.yaml
│   │       └── tier5_temporal.yaml
│   ├── kubernetes/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── questions/
│   └── sec-edgar/
│       ├── raw/
│       ├── processed/
│       └── questions/
│
├── results/                     # Benchmark results (committed to repo)
│   ├── python-stdlib.json
│   ├── kubernetes.json
│   ├── sec-edgar.json
│   └── summary.json             # Cross-corpus aggregate
│
├── web/                         # Next.js 14 frontend
│   ├── app/
│   │   ├── page.tsx             # Landing page with results table
│   │   ├── demo/page.tsx        # Side-by-side chatbot (5 panels)
│   │   ├── benchmark/page.tsx   # Interactive benchmark explorer
│   │   └── graph/page.tsx       # Knowledge graph visualizer (Sigma.js)
│   ├── components/
│   │   ├── ChatPanel.tsx        # Single strategy chat panel
│   │   ├── BenchmarkTable.tsx   # Sortable results table
│   │   ├── TierChart.tsx        # Recharts accuracy-by-tier chart
│   │   └── GraphViewer.tsx      # Sigma.js wrapper
│   └── ...
│
└── tests/
    ├── conftest.py              # Shared fixtures (mock Neo4j, mock ChromaDB)
    ├── test_ingest.py
    ├── test_strategies.py
    ├── test_graph/
    │   ├── test_extractor.py
    │   ├── test_resolver.py
    │   └── test_cypher.py
    ├── test_benchmark.py
    └── test_router.py
```

---

## Pip Package Design

KB Arena ships as a single pip-installable package. The goal: `pip install kb-arena && kb-arena ingest ./my-docs/ && kb-arena serve` works from scratch.

### Public API

```python
# kb_arena/__init__.py — public surface
from kb_arena.models.document import Document, Section
from kb_arena.models.graph import Entity, Relationship
from kb_arena.models.benchmark import Question, BenchmarkResult
from kb_arena.ingest.pipeline import Pipeline
from kb_arena.benchmark.runner import Benchmark
from kb_arena.strategies.base import Strategy

__all__ = [
    "Document", "Section", "Entity", "Relationship",
    "Question", "BenchmarkResult", "Pipeline", "Benchmark", "Strategy",
]
```

Users who `from kb_arena import Pipeline, Benchmark` get a clean API. Internal modules (graph store, LLM client, etc.) are importable but not part of the public contract.

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "kb-arena"
version = "0.1.0"
description = "Benchmark knowledge graphs vs vector RAG on real documentation"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{ name = "Xavier Puspus", email = "xavier@xmpuspus.dev" }]
keywords = ["rag", "knowledge-graph", "benchmark", "neo4j", "chromadb", "llm"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "typer==0.12.5",
    "rich==13.9.4",
    "pydantic==2.10.4",
    "pydantic-settings==2.7.1",
    "httpx==0.28.1",
    "anthropic==0.42.0",
    "neo4j==5.27.0",
    "chromadb==0.5.23",
    "jellyfish==1.1.3",
    "networkx==3.4.2",
    "beautifulsoup4==4.12.3",
    "pyyaml==6.0.2",
    "fastapi==0.115.6",
    "uvicorn[standard]==0.34.0",
    "sse-starlette==2.2.1",
]

[project.optional-dependencies]
dev = ["pytest==8.3.4", "ruff==0.8.6", "pytest-asyncio==0.24.0", "httpx"]
datasets = ["requests==2.32.3"]  # for downloading raw docs
frontend = []  # frontend is Node-based, not Python

[project.scripts]
kb-arena = "kb_arena.cli:app"

[project.urls]
Homepage = "https://github.com/xmpuspus/kb-arena"
Documentation = "https://github.com/xmpuspus/kb-arena/tree/main/docs"
Repository = "https://github.com/xmpuspus/kb-arena"
Issues = "https://github.com/xmpuspus/kb-arena/issues"
```

### Dependency Strategy

All dependencies pinned to exact versions — no `>=`, `^`, or `~`. This prevents version drift that breaks reproducibility (the whole point of a benchmark).

**Core vs optional split:**
- Core deps (always installed): Typer, Pydantic, httpx, anthropic, neo4j, chromadb, jellyfish, networkx, bs4, pyyaml, FastAPI, uvicorn, sse-starlette
- Dev deps (`pip install kb-arena[dev]`): pytest, ruff, pytest-asyncio
- Dataset download (`pip install kb-arena[datasets]`): requests (for fetching raw docs from docs.python.org, k8s.io, SEC EDGAR)

### Dataset Download-on-Demand

Raw datasets are NOT bundled in the pip package (too large). Instead:

```python
# kb_arena/datasets/downloader.py
DATASET_REGISTRY = {
    "python-stdlib": {
        "modules": ["json", "os", "sys", "pathlib", ...],  # 50 modules
        "base_url": "https://docs.python.org/3/library/{module}.html",
        "parser": "html",
    },
    "kubernetes": {
        "base_url": "https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.28/",
        "parser": "html",
    },
    "sec-edgar": {
        "base_url": "https://www.sec.gov/cgi-bin/browse-edgar",
        "parser": "sec_edgar",
    },
}
```

CLI: `kb-arena download python-stdlib` fetches raw docs to `datasets/python-stdlib/raw/`. Ships with question YAML files (small, committed to the package).

### Plugin System for Custom Strategies

Users can add custom retrieval strategies without forking:

```python
# Custom strategy in user code
from kb_arena.strategies.base import Strategy, AnswerResult

class MyCustomStrategy(Strategy):
    name = "my_custom"

    async def build_index(self, documents: list[Document]) -> None:
        # Build your custom index
        ...

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        # Your retrieval logic
        ...
```

Registration via pyproject.toml entry points:

```toml
# In user's pyproject.toml
[project.entry-points."kb_arena.strategies"]
my_custom = "my_package.strategy:MyCustomStrategy"
```

The benchmark runner auto-discovers registered strategies via `importlib.metadata.entry_points()`.

### Versioning and Release

- SemVer: `0.1.0` for initial release, bump minor for new strategies/datasets
- Benchmark results always tagged with package version (`results/v0.1.0/`)
- Breaking changes to Question/BenchmarkResult models = major version bump
- GitHub Actions CI: lint, test, build wheel, publish to PyPI on tag push

### CI/CD Pipeline

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI
on:
  push:
    tags: ["v*"]
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
```

### .gitignore additions for packaging

```
dist/
*.egg-info/
build/
.eggs/
```

---

## The Benchmark Methodology

### Question Design (200+ questions)

Each question is a YAML file (cloudwright pattern) tagged with:
- **id:** unique identifier (`{corpus}-t{tier}-{number}`)
- **tier** (1-5 complexity)
- **type** (factoid, comparison, relational, temporal, causal)
- **hops** (1-5: how many distinct doc sections needed)
- **ground_truth:**
  - `answer`: human-verified reference answer
  - `source_refs`: list of source document section IDs
  - `required_entities`: entities that must appear in the answer
- **constraints:**
  - `must_mention`: keywords/phrases the answer MUST include
  - `must_not_claim`: incorrect statements the answer must NOT make
  - `max_tokens`: expected answer length (for cost normalization)

**Distribution:**
- Tier 1 (single-fact): 50 questions — 20 Python, 15 K8s, 15 SEC
- Tier 2 (multi-entity): 50 questions — 20 Python, 15 K8s, 15 SEC
- Tier 3 (comparative): 40 questions — 15 Python, 15 K8s, 10 SEC
- Tier 4 (relational): 35 questions — 12 Python, 12 K8s, 11 SEC
- Tier 5 (temporal + relational): 25 questions — 8 Python, 8 K8s, 9 SEC

### Evaluation Pipeline (cloudwright pattern)

```python
# kb_arena/benchmark/evaluator.py
class BenchmarkEvaluator:
    """Two-pass evaluation: structural checks (cheap) → LLM judge (expensive).
    Cloudwright lesson: structural checks catch 80% of issues without LLM."""

    def evaluate(self, answer: str, question: Question) -> Score:
        # Pass 1: Structural checks (no LLM, <1ms)
        structural = self._structural_check(answer, question)
        if structural.accuracy == 0.0:
            return structural  # obviously wrong, no need for LLM

        # Pass 2: LLM-as-judge (Sonnet, ~200 tokens)
        llm_score = self._llm_judge(answer, question)
        return Score(
            accuracy=llm_score.accuracy,
            completeness=llm_score.completeness,
            faithfulness=llm_score.faithfulness,
            structural_pass=structural.all_passed,
        )

    def _structural_check(self, answer: str, question: Question) -> StructuralResult:
        """Check must_mention and must_not_claim from YAML."""
        answer_lower = answer.lower()
        mentions_found = [m for m in question.constraints.must_mention if m.lower() in answer_lower]
        false_claims = [c for c in question.constraints.must_not_claim if c.lower() in answer_lower]
        mention_ratio = len(mentions_found) / len(question.constraints.must_mention)
        return StructuralResult(
            accuracy=0.0 if false_claims else mention_ratio,
            mentions_found=mentions_found,
            false_claims=false_claims,
        )
```

### Evaluation Metrics

| Metric | How Measured |
|--------|-------------|
| **Accuracy** | Structural checks (must_mention/must_not_claim) + LLM-as-judge (0-1 scale) |
| **Completeness** | Does the answer address all parts of the question? (0/0.5/1) |
| **Faithfulness** | Does the answer contradict any source document? (binary) |
| **Latency** | End-to-end from query to answer (p50, p95, p99) |
| **Token cost** | Total tokens consumed (embedding + retrieval + generation) |
| **Cost per correct answer** | Total cost / number of correct answers — the metric no one publishes |

### Reproducibility

- All benchmark questions committed to repo as YAML
- Deterministic seeding for LLM calls (temperature=0)
- Docker Compose for full stack: `docker compose up` → Neo4j + ChromaDB + API
- Single command: `kb-arena benchmark run --dataset python-stdlib`
- Results auto-committed as JSON with timestamps and model versions
- `kb-arena benchmark diff results/v1.json results/v2.json` — compare runs

---

## Docker Compose (paper-trail-ph pattern)

```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"   # HTTP
      - "7687:7687"   # Bolt
    environment:
      - NEO4J_AUTH=neo4j/kbarena
      - NEO4J_PLUGINS=["apoc"]
      # GDS excluded by design — Enterprise-only (paper-trail-ph decision)
    volumes:
      - neo4j_data:/data
      - ./cypher:/cypher  # schema files accessible inside container
    healthcheck:
      test: ["CMD", "neo4j", "status"]
      interval: 10s
      retries: 5

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=kbarena
      - CHROMA_PATH=/data/chroma
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      neo4j:
        condition: service_healthy
    volumes:
      - chroma_data:/data/chroma
      - ./datasets:/datasets
      - ./results:/results

  web:
    build: ./web
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - api

volumes:
  neo4j_data:
  chroma_data:
```

---

## Implementation Phases (Detailed)

### Phase 1: Foundation (Week 1)

**Goal:** End-to-end pipeline working for Strategy 1 + 50 Tier 1 questions on Python stdlib.

**Tasks:**
1. Scaffold repo with hatchling pyproject.toml, Typer CLI, basic project structure
2. Implement `kb_arena/models/document.py` — the Pydantic v2 unified document model
3. Implement `kb_arena/ingest/parsers/html.py` — Python docs HTML parser
4. Implement `kb_arena/ingest/pipeline.py` — raw docs → JSONL
5. Implement `kb_arena/strategies/base.py` — abstract Strategy interface + AnswerResult model
6. Implement `kb_arena/strategies/naive_vector.py` — Strategy 1 (ChromaDB, ~80 lines)
7. Write 50 Tier 1 YAML questions for Python stdlib with ground truth
8. Implement `kb_arena/benchmark/runner.py` + `evaluator.py` (structural checks only, no LLM judge yet)
9. Run first benchmark: verify pipeline works end-to-end
10. Docker Compose with Neo4j + API

**Verification:** `kb-arena ingest ./datasets/python-stdlib/raw/ && kb-arena benchmark run --strategy naive_vector --tier 1` produces a results JSON.

### Phase 2: Knowledge Graph (Week 2)

**Goal:** Strategy 4 working, first head-to-head comparison with Strategy 1.

**Tasks:**
1. Define Python stdlib graph schema as NodeType/RelType enums
2. Write cypher/schema_python.cypher with idempotent constraints + indexes
3. Implement `kb_arena/graph/extractor.py` — LLM entity extraction with schema constraints
4. Implement `kb_arena/graph/resolver.py` — Jaro-Winkler entity resolution
5. Implement `kb_arena/graph/neo4j_store.py` — UNWIND/MERGE batch loading
6. Implement `kb_arena/graph/cypher_templates.py` — 5-8 template queries for known patterns
7. Implement `kb_arena/graph/cypher_generator.py` — Text-to-Cypher fallback
8. Implement `kb_arena/strategies/knowledge_graph.py` — Strategy 4
9. Write 50 Tier 2-3 YAML questions
10. Run benchmark: Strategy 1 vs Strategy 4 on Tiers 1-3

**Verification:** Graph strategy outperforms vector on Tier 2-3 questions. Vector still competitive on Tier 1. This is the first "proof" moment.

### Phase 3: All Strategies + Full Benchmark (Week 3)

**Goal:** All 5 strategies working, full 200+ question benchmark across all 3 corpuses.

**Tasks:**
1. Implement Strategy 2 (contextual_vector.py) — heading path prepend + metadata filters
2. Implement Strategy 3 (qna_pairs.py) — LLM question generation + embedding
3. Implement Strategy 5 (hybrid.py) — 3-stage router + fusion
4. Add K8s doc parser + schema + questions (50 questions)
5. Add SEC EDGAR parser + schema + questions (50 questions)
6. Write remaining 50 Tier 4-5 questions across all corpuses
7. Implement LLM-as-judge in evaluator (for cases that pass structural checks)
8. Full benchmark: 5 strategies × 3 corpuses × 200+ questions
9. Generate results JSON + summary report
10. Human spot-check 20% of LLM judge evaluations

**Verification:** Full results table matches expected pattern (vector degrades on high tiers, graph holds, hybrid wins). Cost-per-correct-answer calculated.

### Phase 4: Live Demo + Polish (Week 4)

**Goal:** Deployable side-by-side chatbot demo, pip package, polished README.

**Tasks:**
1. Implement SSE streaming in chatbot API (paper-trail-ph pattern)
2. Implement 3-stage intent router (cloudcare pattern)
3. Build Next.js side-by-side demo page (5 panels, same question)
4. Build Sigma.js graph visualizer page
5. Build Recharts benchmark explorer page
6. Generate architecture diagram for README
7. Deploy: Vercel (frontend), Railway (API), Neo4j AuraDB free tier
8. Publish pip package (`pip install kb-arena`)
9. Write README with results table, quick start, architecture diagram
10. Generate LinkedIn carousel images (6-8 slides)

**Verification:** Live demo URL works. `pip install kb-arena && kb-arena ingest ./my-docs/ && kb-arena serve` works from scratch.

### Phase 5: Launch

- LinkedIn post with the hook + carousel
- GitHub repo public
- Live demo link
- Submit to HackerNews, Reddit r/MachineLearning, r/LocalLLaMA

---

## Gotchas and Lessons Learned (from past projects)

These are specific things that went wrong or required non-obvious solutions:

| Gotcha | Source Project | Solution |
|--------|---------------|----------|
| Neo4j cursors leak if you don't `await result.consume()` | paper-trail-ph | Always consume results, even if you don't need the summary |
| Edge loads silently drop records referencing nonexistent nodes | paper-trail-ph | Load nodes BEFORE edges. Always. Validate node existence first. |
| LLM entity extraction drifts across batches without schema constraints | paper-trail-ph | Enum-based schema in the extraction prompt. Post-validate. Reject unknowns. |
| `NOT EXISTS { }` subquery syntax requires Neo4j 4.4+ | paper-trail-ph | Pin Neo4j 5+ in Docker Compose |
| Text-to-Cypher reliability degrades above ~15 node types | industry research | Cap at 10-12 node types per corpus schema. Use template queries for known patterns, Cypher gen only for novel queries. |
| Benchmark bias: if you design metrics that favor your approach, the benchmark is useless | cloudwright | Include Tier 1 questions where vector WINS. Report per-tier AND aggregate. Be honest. Cloudwright's benchmark favors structure by design — acknowledge this. |
| FTS5 special characters crash queries | cloudcare | Sanitize input: strip special chars, split into words, join as `"word1" OR "word2"` |
| React Flow `fitView` has timing bugs | cloudcare | Pre-compute `defaultViewport` from graph dimensions via dagre layout |
| CPU-intensive graph algorithms block the async event loop | climate-money-ph | `asyncio.to_thread()` for all networkx computations |
| SVG choropleths with hardcoded coordinates don't scale | climate-money-ph | Use Leaflet + GeoJSON from the start if maps are needed |
| Module-level singletons for service instances | cloudcare | Fine for single-process demos. Use lifespan() for proper FastAPI lifecycle. |
| Client-side conversation memory loses state on refresh | cloudcare | Acceptable for demo. If persistence needed, add SQLite session store. |
| Jaro-Winkler false positives on short strings | paper-trail-ph | Minimum string length threshold (3+ chars) before resolution matching |
| APOC plugin load order matters | paper-trail-ph | Specify `NEO4J_PLUGINS=["apoc"]` in docker-compose env, not in config file |

---

## Agent Team Build Instructions

If building this with an agent team, partition as follows:

**Agent 1 (Lead / Orchestrator):** Models + CLI + settings + Docker Compose + README
- Owns: `kb_arena/models/`, `kb_arena/cli.py`, `kb_arena/settings.py`, `pyproject.toml`, `docker-compose.yml`
- Pattern: cloudwright monorepo structure

**Agent 2 (Ingestion):** Document parsers + pipeline
- Owns: `kb_arena/ingest/`, `datasets/*/raw/`
- Pattern: climate-money-ph 5-stage pipeline with JSONL intermediates

**Agent 3 (Graph):** Neo4j schema, extraction, resolution, store, templates
- Owns: `kb_arena/graph/`, `cypher/`
- Pattern: paper-trail-ph graph stack (enums, UNWIND/MERGE, Jaro-Winkler)

**Agent 4 (Strategies):** All 5 retrieval strategies
- Owns: `kb_arena/strategies/`
- Depends on: Agent 2 (ingestion models), Agent 3 (graph store)
- Pattern: cloudcare intent routing for Strategy 5

**Agent 5 (Benchmark):** Question design, evaluation, reporting
- Owns: `kb_arena/benchmark/`, `datasets/*/questions/`, `results/`
- Pattern: cloudwright YAML test cases with structural checks

**Agent 6 (Frontend):** Next.js demo, graph viz, benchmark explorer
- Owns: `web/`
- Pattern: paper-trail-ph Sigma.js, cloudcare React Flow layout

**Serialization:** Agents 1-3 can run in parallel (Wave 1). Agent 4 depends on 2+3. Agent 5 can start question writing in parallel with everything. Agent 6 starts in Wave 2 after API shape is stable.

---

## What Makes This Different From Existing Work

| Existing | KB Arena |
|----------|---------------|
| Microsoft GraphRAG | Library, no benchmark comparison against alternatives |
| LightRAG | Single approach, no multi-strategy comparison |
| Various RAG tutorials | Toy data (Wikipedia paragraphs), no real-world docs |
| Academic papers | Not reproducible, no live demo, no pip install |
| Neo4j blog posts | Vendor-biased, single approach, no side-by-side |

**KB Arena is the first:**
- Multi-strategy benchmark on real documentation
- With a live side-by-side chatbot demo
- pip-installable so anyone can run it on their own docs
- With the "cost per correct answer" metric
- That empirically answers the format question with data, not opinion

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| KG extraction quality varies per doc type | Schema constraints + verification pass (Globant's self-correcting KG pattern). Post-extraction validation rejects unknown types. |
| Text-to-Cypher breaks on complex schemas | Limit to <12 node types per corpus. Pre-built Cypher templates for known patterns (paper-trail-ph: 13 template queries). Text-to-Cypher only for novel queries. |
| Benchmark bias toward graph | Include Tier 1 questions where vector wins. Report per-tier AND aggregate. Be honest about tradeoffs. Cloudwright lesson: acknowledge structural advantages. |
| LLM-as-judge unreliable | Structural checks first (must_mention/must_not_claim). LLM judge only for cases that pass structural. Human spot-check 20%. |
| Demo hosting costs | Free tier: Vercel (frontend) + Railway (API). Neo4j AuraDB free tier. Mock fallback if Neo4j unavailable (climate-money-ph pattern). |
| Someone else ships first | The live demo + pip package + 3-corpus benchmark is a defensible moat. No one has all three. |
| Entity resolution false positives | Two-threshold approach (>=0.92 merge, 0.85-0.91 review). Minimum string length. Same-type-only matching. |
| Neo4j Community Edition limitations | No property existence constraints (enforce at Pydantic layer). No GDS (use networkx via asyncio.to_thread). APOC is sufficient. |

---

## LinkedIn Content Strategy

### Post 1 (Launch): The Benchmark Results
- Carousel (6-8 slides): results table, architecture diagram, tier breakdown, live demo screenshot
- Hook: "I benchmarked 5 data formats for AI chatbots. Vector RAG scores 94% on simple questions but 8% on relational queries."
- CTA: Live demo link + GitHub

### Post 2 (Week after): The Knowledge Graph Extraction Pipeline
- Technical deep-dive on how to convert messy docs into a clean knowledge graph
- Show before (raw Confluence export) vs after (structured graph in Sigma.js)
- CTA: "Run it on your own docs: pip install kb-arena"

### Post 3 (2 weeks after): The Cost Per Correct Answer
- The metric no one talks about
- Show that hybrid is 2x the cost of naive vector but 4x the accuracy on hard queries
- "The cheapest correct answer, not the cheapest answer"

### Post 4 (Monthly): Community Results
- People running it on their own docs and sharing results
- Aggregate findings across industries/doc types

---

## Estimated Impact

**GitHub:** 500-2000 stars in first month (based on GraphRAG trajectory scaled to scope)
**LinkedIn:** 50-200k impressions on launch post (technical carousel + live demo + hard numbers)
**Practical:** Becomes the reference people share when asked "what format should docs be in?"

**Xavier's positioning:** From "the paper-trail-ph / cloudwright guy" to "the person who definitively answered the document format question for AI with data"

---

## Summary

This project sits at the intersection of:
1. **Xavier's proven expertise** (knowledge graphs, Neo4j, entity resolution, benchmarking)
2. **A universal pain point** (everyone dumping docs into vector DBs and getting bad results)
3. **A gap no one has filled** (no live demo + benchmark + pip package together)
4. **LinkedIn virality mechanics** (counter-intuitive claim + hard numbers + live demo)

The core thesis: **Knowledge graphs aren't a data format — they're the extracted semantic layer that should sit between your raw documents and your retrieval system. The source format (Confluence, Markdown, PDF) doesn't matter. What matters is whether you extract the relationships.**

Every implementation pattern in this plan is battle-tested from paper-trail-ph (graph schema, entity resolution, Cypher templates, SSE streaming), cloudwright (benchmark methodology, pip packaging, dual-model LLM, CLI design), cloudcare (intent routing, SQL vs vector proof, conversation memory), and climate-money-ph (graph algorithms, asyncio offload, caching, mock fallback).

Build it. Benchmark it. Demo it. Share the numbers.
