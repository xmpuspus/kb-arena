# KB Arena

Compare retrieval architectures on your own documentation and choose with evidence.

[![PyPI](https://img.shields.io/pypi/v/kb-arena)](https://pypi.org/project/kb-arena/)
[![Python](https://img.shields.io/pypi/pyversions/kb-arena)](https://pypi.org/project/kb-arena/)
[![CI](https://github.com/xmpuspus/kb-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/xmpuspus/kb-arena/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![DOI](https://zenodo.org/badge/1182030516.svg)](https://zenodo.org/badge/latestdoi/1182030516)

KB Arena runs the same corpus and question set through 19 built-in retrieval strategies,
lexical, dense, graph, hybrid, hierarchical, reranked, and more. It records retrieval quality,
answer quality, latency, cost, and run artifacts, so you can decide which design fits your data
before you build on it.

Use it before you commit to a retrieval architecture, or as a regression lab after your corpus,
chunking, model, or index changes.

![Historical KB Arena retrieval result](https://raw.githubusercontent.com/xmpuspus/kb-arena/main/docs/demo.gif)

## Explore a checked result

The packaged demo uses precomputed AWS Compute results. It needs no API key, Docker service, or
Neo4j instance.

```bash
pip install kb-arena
kb-arena demo
```

Open the URL printed by the command. The demo ships 8 benchmark result files for the
`aws-compute` corpus, one per strategy, so the benchmark table, the per-tier breakdown, the
source drill-down and the strategy comparison all have real numbers behind them.

The Retriever Lab and the spread across repeated runs are empty until you produce them. They
read run directories, and the package bundles none. Run `kb-arena retriever-lab` and
`kb-arena benchmark --runs 3` against a corpus to fill them.

## Point it at your own documents

Ingest your files, build the strategy your provider budget allows, and score retrieval for free
through BM25, the one built-in strategy that needs no key. [Run it on your
documents](#run-it-on-your-documents) below has the setup for local and hosted providers, and
[the own-corpus walkthrough](https://github.com/xmpuspus/kb-arena/blob/main/docs/own-corpus-walkthrough.md)
runs ingest through an exported evidence bundle with the real console output from each step.

## Four ways to run a comparison

- **CLI** runs `kb-arena`, with commands from `ingest` through `evidence`. See [the command
  reference](https://github.com/xmpuspus/kb-arena/blob/main/docs/reference-cli.md).
- **HTTP API** starts with `kb-arena serve`, which runs the FastAPI server behind the dashboard.
  See [the HTTP reference](https://github.com/xmpuspus/kb-arena/blob/main/docs/reference-http.md)
  for every route and its auth gate.
- **MCP server** installs with `pip install 'kb-arena[mcp]'`, then runs with `python3 -m
  kb_arena.mcp.server`. It exposes corpus, strategy, benchmark, and evidence tools over stdio to
  an MCP client such as Claude Code or Codex. See
  [the server module](https://github.com/xmpuspus/kb-arena/blob/main/kb_arena/mcp/server.py).
- **GitHub Action**
  [`retrieval-regression-gate`](https://github.com/xmpuspus/kb-arena/tree/main/.github/actions/retrieval-regression-gate)
  ingests a corpus, builds an index, and fails a pull request when a named metric drops past a
  threshold. See
  [the example workflow](https://github.com/xmpuspus/kb-arena/blob/main/.github/workflows/retrieval-regression-example.yml).

## What KB Arena helps decide

| Question | Comparison |
|---|---|
| Do exact terms matter more than semantic similarity? | BM25 against dense retrieval |
| Does document context improve chunk retrieval? | Naive against contextual vector |
| Do cross-document relationships justify a graph? | Dense against graph and hybrid |
| Does hierarchy help on broad questions? | Dense against RAPTOR and PageIndex |
| Is a reranker worth its latency and cost? | Dense against reranked dense |
| Is a proposed method meaningfully different? | Paired scores, confidence intervals, and rank overlap |

KB Arena does not give a universal leaderboard. A result applies to the corpus, questions,
ground truth, configuration, and models named in that run.

## Evidence included in this repository

The tracked Retriever Lab run `855aac4e` is a historical, reproducible example:

- Corpus: `aws-compute`, three documents and 1,549 words
- Run date: 2026-04-26
- Questions: 75 across five tiers
- Cutoff: top 5 chunks
- Chunk-level labels: 35 questions have labels, and 40 do not
- Scope: eight strategies from the version available on the run date

| Strategy | Recall@5 | MRR | NDCG@5 |
|---|---:|---:|---:|
| Contextual Vector | 0.355 | 0.433 | 0.388 |
| Naive Vector | 0.352 | 0.414 | 0.367 |
| RAPTOR | 0.352 | 0.414 | 0.367 |
| BM25 | 0.275 | 0.352 | 0.278 |
| Hybrid | 0.080 | 0.093 | 0.086 |
| PageIndex | 0.061 | 0.111 | 0.076 |

![Historical retrieval metrics from run 855aac4e](https://raw.githubusercontent.com/xmpuspus/kb-arena/main/docs/benchmark-evidence.png)

Source: [tracked report](https://github.com/xmpuspus/kb-arena/blob/main/results/run_855aac4e/retriever_lab.md) and
[run artifact](https://github.com/xmpuspus/kb-arena/blob/main/results/run_855aac4e/retriever_lab.json).

These numbers show the report format and calculation path. The corpus is too small, and
its chunk labels are too incomplete, to support a general winner claim. Q&A Pairs and Knowledge
Graph from that run have zero chunk-level scores because their retrieved identifiers did not map
to the available labels. Those zeroes reflect gaps in the current evaluation, not evidence that the
methods cannot retrieve useful context.

## A larger public corpus for method development

The repository includes a deterministic NIST SP 800-171 Revision 3 corpus built from the
official publication. It has 130 control documents and 80 questions across direct,
paraphrased, scenario, boundary, and multi-control categories. Each question maps to source control
sections, with 48 development, 12 validation, and 20 holdout items.

The question set is not a human-approved benchmark but a machine-generated draft. Do not publish a
strategy winner from it until a qualified reviewer checks the questions, answers, constraints, and
holdout isolation. See the [corpus notes](https://github.com/xmpuspus/kb-arena/blob/main/datasets/nist-800-171-r3/README.md),
[source manifest](https://github.com/xmpuspus/kb-arena/blob/main/datasets/nist-800-171-r3/source-manifest.json), and
[evaluation method](https://github.com/xmpuspus/kb-arena/blob/main/docs/methodology.md).

## Run it on your documents

### Local models with Ollama

Install and start [Ollama](https://ollama.com/), then pull both generation and embedding models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text

export KB_ARENA_LLM_PROVIDER=ollama
export KB_ARENA_EMBEDDING_PROVIDER=ollama
```

Create a corpus, place files in `raw/`, and run the pipeline:

```bash
pip install 'kb-arena[all-formats]'
kb-arena init-corpus my-docs
cp -R /path/to/docs/. datasets/my-docs/raw/
kb-arena run --corpus my-docs --skip-graph
```

![Create and ingest an example documentation corpus](https://raw.githubusercontent.com/xmpuspus/kb-arena/main/docs/demo-own-docs.gif)

`run` ingests a populated `raw/` directory automatically. Remove `--skip-graph` after starting
Neo4j when you want graph and hybrid comparisons.

### Hosted generation and embeddings

The generation and embedding providers are independent:

```bash
export KB_ARENA_LLM_PROVIDER=anthropic
export KB_ARENA_ANTHROPIC_API_KEY=...
export KB_ARENA_EMBEDDING_PROVIDER=openai
export KB_ARENA_OPENAI_API_KEY=...

kb-arena run --corpus my-docs
```

Supported embedding providers are OpenAI, Voyage, Cohere, Gemini, local BGE, and Ollama. See the
[getting-started guide](https://github.com/xmpuspus/kb-arena/blob/main/docs/getting-started.md) for formats, Neo4j setup, checkpoints, and provider
configuration.

## Evaluation paths

Use the retrieval-only path when you need to isolate the index and ranking behavior:

```bash
kb-arena label-chunks --corpus my-docs
kb-arena retriever-lab --corpus my-docs --top-k 5
```

It reports Recall@k, Precision@k, Hit@k, MRR, NDCG@k, MAP, R-Precision, bpref,
bootstrap confidence intervals, and per-tier breakdowns.

Use the full benchmark for generated-answer scoring, source attribution, latency,
cost, and reliability:

```bash
kb-arena benchmark --corpus my-docs --top-k 5
kb-arena report --corpus my-docs --format markdown
```

Use optimization after you define a development split. Keep a separate holdout for the published
comparison:

```bash
kb-arena optimize \
  --corpus my-docs \
  --split development \
  --strategies bm25,naive_vector,contextual_vector,raptor \
  --top-ks 3,5,10 \
  --metric ndcg
```

After you choose the configuration, run the public comparison with
`kb-arena benchmark --split holdout`.

Optimization remains retrieval-only. QnA Pairs and RAPTOR reuse their prebuilt indexes and sweep
top-k only. They do not regenerate pairs or summaries during a search.

Read [the evaluation method](https://github.com/xmpuspus/kb-arena/blob/main/docs/methodology.md) before interpreting small score differences or
synthetic question sets.

## Strategy catalog

The catalog holds 19 strategies. The default `all` benchmark runs 9 of them and leaves
out 10: LightRAG, Metadata Filtered, Temporal, Rerank Vector, SQR, HyDE, Multi-Query, Late
Interaction, SPLADE and Agentic. Each one is out for a reason a default run cannot meet.
Rerank Vector needs the dependency for its selected backend. SQR needs the optional `quantum`
dependency group. Late Interaction needs `late-interaction`, and SPLADE needs `splade`.
LightRAG needs a reachable Neo4j, and a degraded result there fails the whole run rather
than the strategy. Metadata Filtered needs a filter the caller passes. Temporal needs an
as-of date. HyDE, Multi-Query and Agentic each cost one or more LLM calls per question.
The API reports loaded and unavailable strategies at `GET /strategies`.

| Strategy | Architecture | Default | Notes |
|---|---|:---:|---|
| Naive Vector | Dense | Yes | Chunk, embed, cosine retrieval |
| Contextual Vector | Dense | Yes | Adds parent context before embedding |
| Q&A Pairs | Generated index | Yes | Creates likely questions at index time |
| Knowledge Graph | Graph | Yes | Retrieves through Neo4j entities and relationships |
| LightRAG | Experimental | No | Local entity neighborhood plus a global community summary, needs Neo4j |
| Hybrid | Hybrid | Yes | Routes and fuses vector and graph results with RRF |
| RAPTOR | Hierarchical | Yes | Retrieves chunks and recursive summaries |
| PageIndex | Hierarchical | Yes | Uses document structure and LLM tree traversal |
| BM25 | Lexical | Yes | Keyless keyword baseline |
| Metadata Filtered | Access-aware dense | No | Applies a tag, owner, classification, and doc ID filter inside retrieval |
| Temporal | Version-aware dense | No | Prefers the newest document version and supports an as-of date |
| Rerank Vector | Reranked dense | No | The BGE backend uses `kb-arena[rerank]`. |
| QISS | Experimental | Yes | Pure NumPy fidelity reranker over dense candidates |
| SQR | Experimental | No | Qiskit Aer SWAP-test reranker, install `kb-arena[quantum]` |
| HyDE | Experimental | No | Embeds an LLM-written hypothetical answer instead of the question |
| Multi-Query | Experimental | No | Asks the LLM for several sub-queries and fuses their results with RRF |
| Late Interaction | Token-level dense | No | ColBERT-style MaxSim reranker, install `kb-arena[late-interaction]` |
| SPLADE | Learned sparse | No | Term-weight expansion over its own sparse index, install `kb-arena[splade]` |
| Agentic | Experimental | No | Retrieve-judge-refine loop under a hard iteration and call budget |

See [strategy details](https://github.com/xmpuspus/kb-arena/blob/main/docs/strategy-catalog.md) and the
[plugin guide](https://github.com/xmpuspus/kb-arena/blob/main/CONTRIBUTING.md#add-a-strategy).

## Data and method limits

- Auto-generated questions help expand coverage, but production queries and human review give
  stronger deployment evidence.
- An LLM judge can introduce model bias. Use a different judge family, keep the prompts and model
  versions, and inspect disagreements.
- Architecture-native indexes do not always return the same chunk identifiers. Validate qrel
  mappings before comparing retrieval metrics.
- Cost and latency depend on provider, model, cache state, hardware, concurrency, and region.
- Tune on development data. Publish results only from a sealed holdout.
- Quantum strategies are experiments. The AWS sample does not show a Recall@5 gain over the dense
  baseline.

The [method guide](https://github.com/xmpuspus/kb-arena/blob/main/docs/methodology.md) defines the evidence that belongs with a public result.

## Project references

- [Getting started](https://github.com/xmpuspus/kb-arena/blob/main/docs/getting-started.md)
- [Evaluation method](https://github.com/xmpuspus/kb-arena/blob/main/docs/methodology.md)
- [Retriever Lab](https://github.com/xmpuspus/kb-arena/blob/main/docs/retriever-lab.md)
- [Strategy catalog](https://github.com/xmpuspus/kb-arena/blob/main/docs/strategy-catalog.md)
- [Command reference](https://github.com/xmpuspus/kb-arena/blob/main/docs/reference-cli.md), generated from the code
- [HTTP reference](https://github.com/xmpuspus/kb-arena/blob/main/docs/reference-http.md), every route and what it asks of a caller
- [Environment reference](https://github.com/xmpuspus/kb-arena/blob/main/docs/reference-environment.md), every setting
- [Dataset adapters](https://github.com/xmpuspus/kb-arena/blob/main/docs/datasets.md)
- [Changelog](https://github.com/xmpuspus/kb-arena/blob/main/CHANGELOG.md)
- [Security policy](https://github.com/xmpuspus/kb-arena/blob/main/SECURITY.md)
- [Contributing](https://github.com/xmpuspus/kb-arena/blob/main/CONTRIBUTING.md)

## Checking a result

A benchmark number is only worth as much as what sits beside it. These three
commands are how KB Arena says what a number is, and what it is not.

```bash
# Repeat a benchmark, so a difference can be told from noise
kb-arena benchmark --corpus my-docs --runs 3 --seed 7

# Read the spread across those repeats
kb-arena variance --corpus my-docs

# Write the record that travels with a run
kb-arena evidence --corpus my-docs --run-id <id>
```

![Spread across repeats](https://raw.githubusercontent.com/xmpuspus/kb-arena/main/docs/demo-variance.gif)

`variance` groups runs by experiment and by build. Two runs from different
commits measured different code, so it lists their values and reports no mean.
Two runs is a range a reader can misread as a bound, so it says when a row rests
on fewer than three.

`evidence` writes the command, the package version, the commit, the platform and
the seed beside the result. It also writes whether the run may be cited:

```json
"citable": false,
"why_not_citable": "publishable is true only when every scored question is human-reviewed."
```

![What a run says about itself](https://raw.githubusercontent.com/xmpuspus/kb-arena/main/docs/demo-evidence.gif)

`kb-arena evidence --check <path>` reads a bundle back. It refuses one that calls
itself citable while its own review says otherwise, and one that is not citable
and does not say why.

The committed example lives at
[`results/run_422209dd`](https://github.com/xmpuspus/kb-arena/tree/main/results/run_422209dd).
It needs no API key to repeat.

## Public datasets

```bash
kb-arena datasets                                  # what is available, and its terms
kb-arena datasets --name crag --destination ~/data/crag
```

An adapter records who made the data, which revision, under what licence, and
what KB Arena did to it before scoring. A moving revision such as `latest` is
refused, because a run against one cannot be repeated.

A dataset whose licence forbids redistribution is never bundled. CRAG is CC BY-NC
4.0, so its adapter ships nothing, fetches nothing for you, and refuses to write
inside the checkout. See [dataset adapters](https://github.com/xmpuspus/kb-arena/blob/main/docs/datasets.md).

## The sealed holdout

`kb-arena optimize` refuses `--split holdout` without `--confirm-holdout`. Every
run that reads holdout questions appends to `results/holdout_uses.jsonl`, judged
by the questions it read rather than by the split it named. `kb-arena
holdout-uses` prints that ledger.

## Development

```bash
git clone https://github.com/xmpuspus/kb-arena
cd kb-arena
pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest tests/ -q --ignore=tests/live
```

The frontend uses Next.js 16 and needs Node.js 20.9 or later:

```bash
cd web
npm ci
npm run lint
npm run build
```

## Citation

The canonical citation metadata is in [CITATION.cff](https://github.com/xmpuspus/kb-arena/blob/main/CITATION.cff). GitHub can export it through
the repository's **Cite this repository** action. The archived software record is available through
the DOI badge above.

## License

[MIT](https://github.com/xmpuspus/kb-arena/blob/main/LICENSE)
