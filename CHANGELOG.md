# Changelog

All notable changes to KB Arena.

## [Unreleased]

## [0.10.0] - 2026-08-05 - Retrieval architecture decision lab

### Added
- Added a runtime strategy catalog that distinguishes registered, default, loaded, optional, and
  experimental strategies.
- Added a deterministic NIST SP 800-171 Revision 3 corpus with 130 controls, 80 draft questions,
  source hashes, control anchors, and section-level relevance labels.
- Added reproducible README media, source tapes, checksums, and visual validation notes.
- Added contribution, conduct, issue-template, method, onboarding, and strategy guides.

### Changed
- Reframed the project as a retrieval architecture decision lab and replaced fixed-count,
  competitor, universal-winner, and quantum-first public claims with run-scoped evidence.
- Updated the dashboard to load strategy availability from the API and limit the precomputed demo
  to strategies that have sample output.
- Upgraded the static frontend to Next.js 16.3.0, Node.js 20.9+, and ESLint 9 flat configuration.
- Backfilled release history from 0.7.0 through 0.9.3 and refreshed citation, package, security, and
  archive metadata.

### Fixed
- Made `kb-arena run` ingest populated corpus `raw/` directories automatically and continue after
  a graph-stage failure without writing a successful graph checkpoint.
- Validated generation and embedding credentials independently for mixed hosted and local provider
  configurations.
- Forced `kb-arena demo` into read-only mode, skipped model and Neo4j clients there, and disabled
  noisy ChromaDB telemetry callbacks.
- Stopped unavailable graph strategies and empty artifacts from producing successful one-shot
  pipeline evidence.
- Preserved development, validation, and holdout labels through benchmark and optimization runs.
- Made capped benchmarks stop launching queued queries when tracked spend reaches the boundary.
- Fixed provider-aware demo detection, demo readiness, live graph build streaming, and stale
  dashboard result updates.
- Passed retrieved context and reference-free mode into answer evaluation, and made benchmark dry
  runs independent of provider credentials.
- Honored configured dataset roots in corpus scaffolding, ingest previews, and one-shot runs, and
  made zero-document ingestion fail instead of printing a next step.
- Added session-only browser token support for protected requests, bounded rate-limit storage,
  automatic graph-build queue expiry, and stale-build isolation when the selected corpus changes.
- Limited concurrent live graph builds, timed out hung extraction tasks, and bounded each event
  queue so unattended builds cannot accumulate without limit.
- Kept optimization retrieval-only by reusing prebuilt QnA Pairs and RAPTOR indexes instead of
  regenerating their LLM-built artifacts during a sweep.
- Stopped retrieval execution failures from becoming valid zero-score observations or optimizer
  recommendations, and made concurrent rate-limit consumption atomic.
- Updated Typer, Click, FastAPI, and Starlette to compatible releases that address current
  command-execution and HTTP request-processing advisories.
- Added tested Python 3.13 support and bounded package metadata to Python 3.11 through 3.13.

## [0.9.3] - 2026-06-22 - Retrieval ceiling and cost efficiency

### Added
- `retriever-lab --ceiling-k` compares base-retriever Recall@top-k with a deeper cutoff.
- Benchmark reports include tokens per query, cost per query, and NDCG per 1,000 tokens.

### Changed
- Cost-efficiency values now come from recorded run output instead of assumed pricing.

## [0.9.2] - 2026-06-22 - QISS subspace projection

### Changed
- Replaced the QISS multi-query centroid operator with projection onto the span of the question and
  its subqueries.
- Kept the single-query contract equal to squared cosine ranking.

## [0.9.1] - 2026-06-22 - Quantum and retrieval robustness

### Fixed
- Fixed the QISS decomposition request, which used an invalid empty system prompt.
- Added a Retriever Lab guard that separates a crashed strategy from a valid low-recall result.
- Updated the editable frontend from eight to ten core choices at that release.

## [0.9.0] - 2026-06-22 - Experimental quantum rerankers

### Added
- QISS, a pure NumPy state-fidelity reranker over dense candidates.
- SQR, an optional Qiskit Aer SWAP-test reranker in the `quantum` dependency group.
- `kb-arena quantum-diagnostics` for PCA variance, shot error, and runtime overhead.

### Fixed
- Corrected a `rerank_vector` source identifier bug that produced empty retrieval traces.
- Allowed comma-separated strategy filters.

## [0.8.1] - 2026-05-21 - First archived release

### Changed
- Bumped package and citation metadata to trigger the first Zenodo source archive.
- No runtime code changed.

## [0.8.0] - 2026-05-20 - Statistical evaluation

### Added
- MAP, R-Precision, bpref, graded NDCG, and Rank-Biased Overlap.
- Bootstrap confidence intervals and per-tier Retriever Lab summaries.
- Wilcoxon paired tests, win rate, quality-latency efficiency, and Pareto markers in optimization.

### Changed
- Added SciPy as a runtime dependency for the statistical layer.

## [0.7.1] - 2026-05-20 - Optimizer rebuild fix

### Fixed
- Compared each trial with the persistent baseline instead of the prior trial, so top-k-only sweeps
  no longer rebuild vector indexes.

### Changed
- Added the first tracked AWS Compute optimization table and a reproducible demo recording.

## [0.7.0] - 2026-05-19 - Strategy optimization and graph provenance

### Added
- `kb-arena optimize` with scoped grid or random search, dry-run planning, and retrieval-only
  scoring.
- Configurable chunk size and overlap settings.

### Fixed
- Carried graph source document and section identifiers through extraction, Neo4j, query results,
  and retrieval traces so graph results can match qrels.

## [0.6.1] - 2026-05-02 - Docs

### Changed
- README gains a full "What's New in v0.6.0" section (was missing from the
  v0.6.0 PyPI listing because PyPI doesn't allow republishing the same version).
- Strategy count bumped from 8 to 9 across the README; the strategy table now
  lists the new `rerank_vector` row; changelog table gains a v0.6.0 entry.
- No code changes. The package, tests, and CLI surface are the same as v0.6.0.

## [0.6.0] - 2026-05-02 - Hardening, 9th strategy, embedding providers, public leaderboard

### Added
- **Strategy #9: `rerank_vector`** - Naive Vector + cross-encoder reranking with three backends:
  `bge` (BAAI/bge-reranker-v2-m3, local, free, default), `cohere` (Rerank v3.5/v4),
  `voyage` (Rerank 2.5). Selects via `KB_ARENA_RERANKER_BACKEND`.
- **Embedding provider abstraction** - `KB_ARENA_EMBEDDING_PROVIDER` selects
  `openai` (default), `voyage` (current MTEB retrieval leader), `cohere`, `bge`
  (local, no key), `ollama` (local, no key), or `gemini`. All four vector
  strategies route through `get_embedding_function()` instead of hard-coded OpenAI.
- **`kb-arena run --corpus my-docs --resume`** - one-shot orchestrator that
  ingests, builds graph, builds vectors, generates questions, and benchmarks,
  with a checkpoint at `datasets/{corpus}/.pipeline_state.json` so a re-run
  with `--resume` skips finished stages.
- **Public read-only `/api/leaderboard`** + Next.js `/leaderboard` page -
  aggregates every benchmark run in `results/run_*` per (corpus, strategy)
  with mean accuracy, Recall@5, NDCG@5, cost, and latency. No auth.
- **Bearer-token auth** (`KB_ARENA_API_TOKEN`) on every LLM-triggering endpoint;
  bounded-deque rate limiter with optional trusted-proxy header support.
- **Demo mode** (`KB_ARENA_DEMO_MODE`). When no API key exists, the API turns on demo mode.
  Every LLM-triggering endpoint returns 503 while the static dashboard,
  leaderboard, benchmark results, and corpora endpoints stay available.
- `kb-arena --version` flag.
- `deploy/vercel.json` and `deploy/huggingface_space.yaml` for hosted demos.
- `docs/tapes/hero-demo.tape`, `docs/tapes/retriever-lab.tape`, and
  `docs/tapes/record-ui.py` so demo GIFs regenerate deterministically.

### Changed
- **Hybrid strategy** - procedural branch now reranks **passages** (real
  `RetrievedChunk.content`) instead of answer strings generated before,
  and uses Reciprocal Rank Fusion (k=60) instead of LLM-pairwise rerank.
  Vector + graph queries now run via `asyncio.gather`. IntentRouter is wired
  in `get_strategy("hybrid")` so the advertised three-stage classification
  actually fires.
- **Knowledge graph extraction** - cross-section relationships are no longer
  dropped at section validation. The extractor checks the global FQN union after
  it finishes all sections, which restores multi-hop graph queries.
- **Ground-truth labelling** - `expected_chunks.yaml` candidate pool widened
  from BM25 alone to BM25 union naive_vector union contextual_vector top-N when the
  vector indexes are built. Closes the circular-method critique.
- **Default `KB_ARENA_BENCHMARK_COST_CAP_USD` is 10.0** (was 0 / unlimited).
- **`SECURITY.md`** rewritten to match implementation; supported versions
  refreshed to 0.6.x.
- **Dockerfile** runs as non-root `kbarena` user with HEALTHCHECK and a
  default `KB_ARENA_DEMO_MODE=true` so a freshly built image cannot drain
  credits without explicit opt-in.
- **`docker-compose.yml`** fail-closes when `KB_ARENA_NEO4J_PASSWORD` is
  unset, binds Neo4j to 127.0.0.1, adds resource limits and api healthcheck.
- README hero rewritten with the question-frame pitch
  ("Should you use Graph RAG, Vector RAG, or Hybrid?") and a No-API-Keys
  Quick Start using Ollama.

### Fixed
- **Cross-tenant data leak** - concurrent SSE consumers overwrote `Strategy.last_*` fields.
  Per-call metrics now travel with the streamed
  tokens via a `_kb_arena_meta` packet; the `last_*` fields stay only as a
  back-compat surface for plugins.
- **`bm25` strategy missing from bundled demo** - `kb_arena/data/aws-compute_bm25.json`
  is now shipped, plus a hatch `force-include` glob so future strategies are
  picked up automatically.
- **`kb-arena demo` zero-config gate** - lifespan tolerates missing API keys,
  enables `demo_mode`, and continues serving the dashboard.
- **Ollama free path** - `_preflight()` reads `settings.llm_provider` and
  skips Anthropic/OpenAI key checks when set to `ollama`.
- **APOC Cypher write bypass** - write regex now rejects
  `apoc.create|merge|refactor|delete|remove|set|drop|iterate|cypher.runWrite|export|trigger`,
  and every read path opens the Neo4j session with `default_access_mode=READ_ACCESS`.
- **SSRF in `kb-arena ingest <url>`** - `WebParser` rejects `file://`, private,
  loopback, link-local, multicast, and reserved IPs (post-DNS); blocks AWS / GCE
  metadata hostnames; disables auto-redirect with per-hop validation.
- **Cost-bomb on chat / arena / tools** - every LLM-triggering endpoint is now
  rate-limited and `Field(max_length=4000)`-bounded; arena endpoints use
  Pydantic models instead of raw `request.json()`.
- **Benchmark runner retry** distinguishes retryable transients (rate limit,
  5xx, network, timeout) from permanent errors (auth, validation,
  missing model) - bad keys fail fast instead of burning 7 minutes per run.
- **Two sources of truth for version** - `chatbot/api.py` now reads
  `__version__` from `kb_arena` package metadata.

### Tests
- Test suite still 558 tests; updated 4 stale tests that asserted old contracts
  (cost cap default, cross-section edge dropping, health response shape,
  strategy count).

## [0.5.0] - 2026-04-26 - Retriever Lab

### Added
- Classical IR metrics computed for every benchmark query: Recall@k, Precision@k, Hit@k, MRR, NDCG@k.
- `RetrievalTrace` and `RetrievedChunk` models on `AnswerResult` - every strategy now exposes the chunks it surfaced with rank, score, and source strategy.
- `Question.expected_chunks` field; `load_questions()` merges `expected_chunks.yaml` automatically.
- `RetrievalMetrics` model attached to `AnswerRecord.retrieval_metrics`; `BenchmarkResult` gains aggregate `mean_recall_at_k`, `mean_precision_at_k`, `mean_hit_at_k`, `mean_mrr`, `mean_ndcg_at_k`.
- New CLI command `kb-arena retriever-lab` - retrieval-only benchmark with a live Rich table. It costs about 10x less than the full benchmark because it stubs LLM generation.
- New CLI command `kb-arena label-chunks` - generate `expected_chunks.yaml` ground truth via BM25 + Haiku judge. Idempotent and cost-capped.
- New `--top-k` flag on `kb-arena benchmark` (default 5).
- New web page `/retriever-lab` - aggregate metrics card per strategy, plus per-question drill-down with HIT/MISS chunk highlighting.
- New API endpoints `GET /api/retriever-lab/runs` and `GET /api/retriever-lab/{run_id}`.
- Hierarchical chunk-id matching: section-level expected IDs match sub-chunk retrievals (`doc::sec` matches `doc::sec::0`). Matching strips the `L0:`, `qna:`, `graph:`, and `pageindex:` strategy prefixes.
- Doc-level fallback in IR metrics: when chunk labels are absent, match against `chunk.doc_id ∈ ground_truth.source_refs`.

### Changed
- All 8 strategies now populate `AnswerResult.retrieval` with stable chunk IDs.
- Benchmark Markdown report gains a "Retrieval Quality (top-k)" section.
- Hybrid strategy preserves sub-strategy `source_strategy` per chunk during fusion.
- BM25 index format includes `chunk_ids` for stable identity across runs (older indexes still load with synthesized IDs).
- ChromaDB telemetry warnings suppressed in retriever-lab to keep terminal output clean.

### Fixed
- BM25 chunk identifiers now stay stable across runs (index-position only before).
- `is_hit` flag in retriever-lab JSON now uses hierarchical matching so vector sub-chunks correctly tag as HIT against section-level labels.

### Tests
- Test suite grows from 514 to 558 tests; coverage adds `tests/test_ir_metrics.py`, `tests/test_retrieval_trace.py`, `tests/test_retriever_lab_runner.py`, `tests/test_label_chunks_cli.py`.
