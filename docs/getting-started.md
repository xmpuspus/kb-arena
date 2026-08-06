# Getting started

## Explore the packaged sample

```bash
pip install kb-arena
kb-arena demo
```

The command copies packaged AWS Compute result files into `./results` when needed, selects an open
port, starts the API, and opens the bundled dashboard. This path does not run a benchmark or call an
LLM.

## Create a corpus

```bash
kb-arena init-corpus my-docs
cp -R /path/to/docs/. datasets/my-docs/raw/
```

Supported input includes Markdown, HTML, text, CSV, and TSV in the core install. Install
`kb-arena[pdf]`, `kb-arena[docx]`, or `kb-arena[all-formats]` for other file types.

You can ingest a URL or public GitHub repository with the same command:

```bash
kb-arena ingest https://docs.example.com --corpus my-docs
kb-arena ingest github:owner/repository --corpus my-docs
```

## Choose generation and embedding providers

Generation and embeddings use separate settings.

### Fully local Ollama

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text

export KB_ARENA_LLM_PROVIDER=ollama
export KB_ARENA_EMBEDDING_PROVIDER=ollama
```

### Anthropic generation and OpenAI embeddings

```bash
export KB_ARENA_LLM_PROVIDER=anthropic
export KB_ARENA_ANTHROPIC_API_KEY=...
export KB_ARENA_EMBEDDING_PROVIDER=openai
export KB_ARENA_OPENAI_API_KEY=...
```

Valid embedding providers are `openai`, `voyage`, `cohere`, `bge`, `ollama`, and `gemini`.
Local BGE can download model weights on first use.

## Run without a graph

```bash
kb-arena run --corpus my-docs --skip-graph
```

When `processed/` is empty, `run` ingests files from `raw/` automatically. It writes successful
stage names to `datasets/my-docs/.pipeline_state.json`. A later run resumes from that state.

## Add graph and hybrid retrieval

Set a password and start Neo4j:

```bash
export KB_ARENA_NEO4J_PASSWORD=choose-a-password
docker compose up neo4j -d
kb-arena run --corpus my-docs
```

If graph extraction fails, the pipeline leaves that stage incomplete and continues with the
vector-capable stages. A later run retries graph extraction.

## Create and review relevance labels

Generated questions are useful for coverage discovery. Review them and add production questions
when possible.

```bash
kb-arena generate-questions --corpus my-docs --count 50
kb-arena label-chunks --corpus my-docs
```

The label command creates `datasets/my-docs/questions/expected_chunks.yaml`. Inspect each expected
chunk before you use the result as public evidence.

## Evaluate

Retrieval-only:

```bash
kb-arena retriever-lab --corpus my-docs --top-k 5
```

Retrieval plus generated answers:

```bash
kb-arena benchmark --corpus my-docs --top-k 5
kb-arena report --corpus my-docs --format markdown
```

The default benchmark runs the core strategy set. Run optional SQR explicitly after installing its
dependencies:

```bash
pip install 'kb-arena[quantum]'
kb-arena benchmark --corpus my-docs --strategy sqr
```

## Control cost and concurrency

Important settings:

| Variable | Default | Purpose |
|---|---:|---|
| `KB_ARENA_BENCHMARK_COST_CAP_USD` | `10.0` | Stops launching queries after observed cumulative cost reaches the cap |
| `KB_ARENA_BENCHMARK_MAX_CONCURRENT` | `5` | Limits concurrent benchmark queries |
| `KB_ARENA_BENCHMARK_QUERY_TIMEOUT_S` | `120` | Limits each strategy query |
| `KB_ARENA_BENCHMARK_MAX_RETRIES` | `2` | Retries transient failures |
| `KB_ARENA_CHROMA_PATH` | `./chroma_data` | Stores local vector indexes |
| `KB_ARENA_DATASETS_PATH` | `./datasets` | Stores corpus inputs and questions |
| `KB_ARENA_RESULTS_PATH` | `./results` | Stores benchmark artifacts |

Run `kb-arena health` to inspect corpus state and local service connectivity.

Rebuild vector and graph indexes created before 0.10.0. Current indexes store corpus metadata and
corpus-namespaced backend IDs so a request cannot retrieve records from another corpus. Vector
rebuilds remove legacy records from KB Arena collections. Graph rebuilds label current entities and
exclude legacy nodes from current reads without deleting them. Use a Neo4j database dedicated to
KB Arena because schema setup removes the legacy KB Arena constraints by name. Rebuild every graph
corpus you still need after upgrading.

Capped runs launch one query at a time so queued work stops at the boundary. The final in-flight
query can make recorded cost exceed the cap. Set the cap to `0` to use parallel execution without
a spend boundary.
