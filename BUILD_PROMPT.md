# DocGraph Bench — Build Prompt

Copy this entire prompt into a new Claude Code session from `~/Desktop/docgraph-bench/`.

---

## PROMPT START

Ultrawork this. Build the DocGraph Bench project from PLAN.md. Read PLAN.md first, then execute in parallel agent teams across worktrees. Don't stop until everything is built and verified.

### Phase 0: Scaffold (orchestrator — no worktree)

Before spinning up agents, create the skeleton they all write into:

```
docgraph-bench/
├── pyproject.toml
├── docker-compose.yml
├── CLAUDE.md
├── .env.example
├── cypher/
├── docgraph/__init__.py
├── docgraph/settings.py
├── docgraph/cli.py
├── docgraph/models/__init__.py
├── docgraph/models/document.py
├── docgraph/models/graph.py
├── docgraph/models/benchmark.py
├── docgraph/models/api.py
├── docgraph/llm/__init__.py
├── docgraph/llm/client.py
├── docgraph/strategies/__init__.py
├── docgraph/strategies/base.py
├── datasets/python-stdlib/raw/.gitkeep
├── datasets/python-stdlib/processed/.gitkeep
├── datasets/python-stdlib/questions/.gitkeep
├── datasets/kubernetes/raw/.gitkeep
├── datasets/kubernetes/processed/.gitkeep
├── datasets/kubernetes/questions/.gitkeep
├── datasets/sec-edgar/raw/.gitkeep
├── datasets/sec-edgar/processed/.gitkeep
├── datasets/sec-edgar/questions/.gitkeep
├── results/.gitkeep
├── tests/conftest.py
├── web/.gitkeep
└── .gitignore
```

Write the shared models (document.py, graph.py, benchmark.py, api.py), settings.py, llm/client.py, strategies/base.py, and pyproject.toml FIRST — these are the contracts all agents depend on. Commit to main.

Write a CLAUDE.md for this project with:
- Project overview and architecture
- How to run (docker compose up, docgraph ingest, docgraph benchmark, docgraph serve)
- Python 3.12, FastAPI, Neo4j Community 5, ChromaDB, Claude Haiku/Sonnet
- Naming conventions: snake_case functions, PascalCase classes, NodeType/RelType enums
- Testing: pytest, mock Neo4j with conftest fixtures
- Never hardcode API keys — use .env via pydantic-settings

Commit scaffold to main. Then launch Wave 1.

---

### Wave 1: BUILD (4 parallel agents in worktrees)

Spin up a team. Create 4 agents, each in their own worktree. Each agent gets their own branch. Agents work in PARALLEL.

**Agent 1: "ingest" (implementer, worktree, sonnet)**

Branch: `feat/ingest-pipeline`

Build the document ingestion pipeline. Read PLAN.md "Pattern 8: Multi-Stage CLI Pipeline" and "Dataset Selection" sections.

Files to create:
- `docgraph/ingest/__init__.py`
- `docgraph/ingest/pipeline.py` — orchestrator, reads raw docs, writes JSONL to processed/
- `docgraph/ingest/parsers/__init__.py`
- `docgraph/ingest/parsers/markdown.py` — parses .md and .rst files into Document model
- `docgraph/ingest/parsers/html.py` — parses HTML files (Python docs format) into Document model
- `docgraph/ingest/parsers/sec_edgar.py` — parses SEC EDGAR 10-K HTML into Document model
- `tests/test_ingest.py` — test each parser with sample documents

Implementation notes:
- Every parser outputs the shared `Document` model from `docgraph/models/document.py`
- Write JSONL intermediates to `datasets/{corpus}/processed/` (climate-money-ph pattern)
- HTML parser must handle Python docs format: `<dl>` for function signatures, `<table>` for params, nested `<div class="section">`
- RST parser: handle directives (.. function::, .. class::, .. deprecated::), cross-references (:func:, :class:, :mod:)
- Heading path extraction: track the h1 > h2 > h3 hierarchy as `section.heading_path`
- Download Python stdlib HTML docs from `docs.python.org/3/library/` for 50 most-used modules
- Tests: parse a real Python docs page (json.html or os.html), verify sections, tables, cross-refs extracted

Wire the `ingest` CLI command in `docgraph/cli.py`.

---

**Agent 2: "graph" (implementer, worktree, sonnet)**

Branch: `feat/graph-engine`

Build the knowledge graph engine. Read PLAN.md "Pattern 2-4, 6-7" and "Graph schema for Python docs" sections.

Files to create:
- `docgraph/graph/__init__.py`
- `docgraph/graph/schema.py` — NodeType + RelType enums per corpus (Python, K8s, SEC). Include validation.
- `docgraph/graph/extractor.py` — LLM-based entity/relationship extraction with schema constraints. System prompt includes enum values as ONLY allowed types. Post-validation rejects unknowns.
- `docgraph/graph/resolver.py` — Two-threshold Jaro-Winkler entity resolution (>=0.92 auto-merge, 0.85-0.91 review). Use jellyfish library.
- `docgraph/graph/neo4j_store.py` — UNWIND/MERGE batch loading (batch_size=1000), cursor management (always consume results), node-before-edge loading order.
- `docgraph/graph/cypher_templates.py` — 8+ pre-built Cypher query templates: single_entity_lookup, multi_hop, comparison, dependency_chain, deprecation_chain, cross_reference, type_hierarchy, usage_examples.
- `docgraph/graph/cypher_generator.py` — Text-to-Cypher via LLM for novel queries. System prompt includes schema. Fallback to template matching.
- `docgraph/graph/analyzer.py` — networkx graph algorithms (communities, centrality, dependency chains) via asyncio.to_thread. 5-minute in-memory cache.
- `cypher/schema_python.cypher` — idempotent DDL with IF NOT EXISTS. Uniqueness constraints, fulltext index across 4+ label types, vector index for embeddings.
- `cypher/schema_kubernetes.cypher`
- `cypher/schema_sec.cypher`
- `tests/test_graph/test_extractor.py`
- `tests/test_graph/test_resolver.py`
- `tests/test_graph/test_cypher.py`

Wire the `build-graph` CLI command.

---

**Agent 3: "strategies" (implementer, worktree, sonnet)**

Branch: `feat/strategies`

Build all 5 retrieval strategies. Read PLAN.md "The 5 Retrieval Strategies — Detailed Implementation" and "Pattern 5, 9" sections.

Files to create:
- `docgraph/strategies/naive_vector.py` — Strategy 1. ChromaDB, 512-token chunks, 50-token overlap, top-k=5. ~80 lines. Deliberately simple.
- `docgraph/strategies/contextual_vector.py` — Strategy 2. Prepend heading_path to chunks before embedding. Metadata filters on ChromaDB where clause. ~120 lines.
- `docgraph/strategies/qna_pairs.py` — Strategy 3. LLM generates 3-5 QnA pairs per section at build time. Embed questions. At query time, match question embeddings, return pre-generated answers. ~210 lines.
- `docgraph/strategies/knowledge_graph.py` — Strategy 4. Uses graph/ module. Intent → template query or Cypher gen → execute → assemble context → generate answer. ~600 lines.
- `docgraph/strategies/hybrid.py` — Strategy 5. 3-stage intent classification (keyword → Haiku → regex fallback). Routes factoid/exploratory to vector, comparison/relational to graph, procedural to both with fusion. ~200 lines.
- `docgraph/chatbot/__init__.py`
- `docgraph/chatbot/router.py` — IntentRouter with 3-stage classification. QueryIntent enum.
- `docgraph/chatbot/session.py` — Client-side memory helper. Last 6 turns, truncated to 500 chars per assistant message.
- `docgraph/chatbot/api.py` — FastAPI app with SSE streaming. Lifespan init. Consistent error envelope. CORS. Rate limiter. Mock fallback if Neo4j unavailable.
- `tests/test_strategies.py`
- `tests/test_router.py`

Wire `build-vectors` and `serve` CLI commands.

---

**Agent 4: "benchmark" (implementer, worktree, sonnet)**

Branch: `feat/benchmark`

Build the benchmark engine AND write all 200+ questions. Read PLAN.md "Benchmark Methodology", "Pattern 13", and "Query Complexity Tiers" sections.

Files to create:
- `docgraph/benchmark/questions.py` — YAML question loader. Validates against Question model.
- `docgraph/benchmark/evaluator.py` — Two-pass: structural checks (must_mention / must_not_claim) then LLM-as-judge. Cloudwright pattern.
- `docgraph/benchmark/runner.py` — Orchestrator. Runs all specified strategies against all questions. Captures accuracy, latency, tokens, cost. Writes results JSON.
- `docgraph/benchmark/reporter.py` — Generates markdown report + summary JSON from results.
- `datasets/python-stdlib/questions/tier1_factoid.yaml` — 20 questions. Single-fact lookups about Python stdlib. Real questions with verified ground truth from docs.python.org.
- `datasets/python-stdlib/questions/tier2_multi_entity.yaml` — 20 questions. Multi-entity queries spanning 2+ modules.
- `datasets/python-stdlib/questions/tier3_comparative.yaml` — 15 questions. Compare stdlib alternatives.
- `datasets/python-stdlib/questions/tier4_relational.yaml` — 12 questions. Dependency/causation chains.
- `datasets/python-stdlib/questions/tier5_temporal.yaml` — 8 questions. Version changes + relationships.
- Same 5 files for `kubernetes/` (15, 15, 15, 12, 8 questions)
- Same 5 files for `sec-edgar/` (15, 15, 10, 11, 9 questions)
- `tests/test_benchmark.py`

Wire `benchmark` and `report` CLI commands.

Each question YAML entry must have: id, tier, type, hops, question, ground_truth (answer, source_refs, required_entities), constraints (must_mention, must_not_claim). Use real questions with real answers verified against actual documentation.

---

### Wave 1 Merge (orchestrator)

After all 4 agents complete:
1. Merge `feat/ingest-pipeline` → main (no conflicts expected — owns ingest/)
2. Merge `feat/graph-engine` → main (no conflicts — owns graph/ and cypher/)
3. Merge `feat/strategies` → main (may touch cli.py — resolve if needed)
4. Merge `feat/benchmark` → main (no conflicts — owns benchmark/ and datasets/)
5. Run `pytest tests/` on merged main — fix any integration issues
6. Commit merged main

---

### Wave 2: QA + FRONTEND (3 parallel agents in worktrees)

**Agent 5: "qa-integration" (qa-verifier, worktree, sonnet)**

Branch: `qa/integration-tests`

Full integration test suite. Read the merged codebase on main.

Tasks:
1. Run `pytest tests/` — fix any failures from merging
2. Write integration tests that exercise the full pipeline:
   - `tests/integration/test_ingest_to_vector.py` — ingest sample doc → build vector index → query → verify answer
   - `tests/integration/test_ingest_to_graph.py` — ingest sample doc → extract entities → load Neo4j → query via Cypher → verify
   - `tests/integration/test_benchmark_e2e.py` — run benchmark on 5 sample questions across 2 strategies → verify results JSON schema
   - `tests/integration/test_chatbot_api.py` — FastAPI TestClient: POST /chat, verify SSE events, verify error envelope on bad input
3. Verify all CLI commands work: `docgraph ingest --help`, `docgraph build-graph --help`, `docgraph benchmark --help`, `docgraph serve --help`
4. Verify docker-compose.yml is valid: `docker compose config`
5. Lint with ruff: `ruff check . && ruff format --check .`
6. Fix all issues found

---

**Agent 6: "qa-benchmark" (qa-verifier, worktree, sonnet)**

Branch: `qa/benchmark-validation`

Validate all 200+ benchmark questions for quality.

Tasks:
1. Load every YAML question file — verify valid YAML and matches Question model schema
2. Check for duplicate question IDs across all files
3. Verify every `source_ref` points to a real page in the documentation (check URL or file exists)
4. Verify `must_mention` terms are actually present in the `ground_truth.answer`
5. Verify `must_not_claim` terms are NOT present in the `ground_truth.answer`
6. Verify question distribution matches PLAN.md targets per tier per corpus
7. Spot-check 20 random questions: is the ground_truth.answer actually correct per the real documentation?
8. Flag any questions that are ambiguous, have incorrect ground truth, or have weak constraints
9. Write a validation report to `datasets/VALIDATION_REPORT.md`
10. Fix any issues found in question files

---

**Agent 7: "frontend" (implementer, worktree, sonnet)**

Branch: `feat/frontend`

Build the Next.js 14 frontend. Read PLAN.md "Repository Structure > web/" section.

Tasks:
1. Scaffold Next.js 14 + Tailwind + TypeScript in `web/`
2. `web/app/page.tsx` — Landing page: project title, results table (hardcoded initially from PLAN.md expected values), architecture diagram (static image or ASCII), "Try the Demo" button, "Quick Start" section
3. `web/app/demo/page.tsx` — Side-by-side chatbot. 5 panels (one per strategy). Single input box at top. On submit, POST to each strategy endpoint in parallel. Show streaming answers via SSE. Below each answer: latency, tokens, cost, source refs.
4. `web/app/benchmark/page.tsx` — Interactive benchmark explorer. Recharts bar chart: accuracy by tier for each strategy. Dropdown to filter by corpus. Sortable results table.
5. `web/app/graph/page.tsx` — Knowledge graph visualizer. Sigma.js component. Load graph JSON from API. Node colors by type. Click node to see details panel.
6. `web/components/ChatPanel.tsx` — Single strategy chat panel with SSE streaming, loading state, source attribution
7. `web/components/BenchmarkTable.tsx` — Sortable, filterable table
8. `web/components/TierChart.tsx` — Recharts grouped bar chart (wrap in ResponsiveContainer!)
9. `web/components/GraphViewer.tsx` — Sigma.js wrapper with zoom/pan, node type legend
10. Keep it clean — Tailwind utility classes, no component library deps, dark/light toggle

---

### Wave 2 Merge (orchestrator)

1. Merge `qa/integration-tests` → main (test files + fixes)
2. Merge `qa/benchmark-validation` → main (question fixes + validation report)
3. Merge `feat/frontend` → main (web/ directory, no conflicts)
4. Run full test suite on merged main
5. Fix any remaining issues

---

### Wave 3: FINAL QA (2 parallel agents)

**Agent 8: "qa-final" (qa-verifier, worktree, sonnet)**

Branch: `qa/final-verification`

Final verification pass:
1. Run `ruff check . && ruff format --check .` — zero warnings
2. Run `pytest tests/ -v` — all pass
3. Verify `pip install -e .` works and all CLI commands are accessible
4. Verify `docgraph --help` shows all subcommands
5. Test the naive pipeline end-to-end: create a small test corpus (3 markdown files), ingest, build vectors, run 3 benchmark questions, verify results JSON
6. Verify docker-compose.yml works: `docker compose up -d && sleep 10 && docker compose ps` — all healthy
7. Check for hardcoded paths, API keys, or secrets — none should exist
8. Verify .gitignore covers: .env, __pycache__, *.pyc, node_modules, .next, neo4j_data, chroma_data
9. Verify pyproject.toml metadata: name, version, description, author, license, python requires, dependencies
10. Write final QA report to `QA_REPORT.md`

**Agent 9: "readme" (implementer, worktree, sonnet)**

Branch: `feat/readme`

Write the viral README. Read PLAN.md "The README Structure (Viral Optimization)" section.

Create:
- `README.md` — follow the exact structure from PLAN.md. Results table (use placeholder values marked [PENDING BENCHMARK]). Architecture diagram as ASCII art matching PLAN.md. Quick start with pip install + 3 commands. "How It Works" section. Comparison table vs existing work. Contributing section. License (MIT).
- `.env.example` — all required env vars with placeholder values
- `docs/ARCHITECTURE.md` — detailed architecture doc referencing the patterns from PLAN.md
- `docs/BENCHMARK_METHODOLOGY.md` — how benchmarks are designed, evaluated, reproduced

---

### Wave 3 Merge + Final Commit

1. Merge both branches → main
2. Run final `pytest && ruff check .`
3. Commit: "Complete initial build of docgraph-bench"

---

### Routing and constraints

- All agents: restrict searches to project directory only. Never search ~/Desktop or ~/.claude recursively.
- All agents: read PLAN.md before starting. Follow the exact patterns described.
- All agents: use the shared models from docgraph/models/ — do NOT create parallel model definitions.
- Build agents (Wave 1): implementer type, worktree isolation, sonnet model.
- QA agents (Wave 2-3): qa-verifier type, worktree isolation, sonnet model.
- Frontend agent: implementer type, worktree isolation, sonnet model.
- Orchestrator budget: stay under 25% context. Write summaries, not full file contents.
- Max 4 concurrent agents per wave. Wait for wave completion before starting next wave.
- If an agent hits 3+ failures on same issue: stop, report to orchestrator, move on.

## PROMPT END
