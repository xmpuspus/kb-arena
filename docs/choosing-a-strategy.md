# Choosing a strategy to try

The catalog holds 19 built-in strategies. This page maps an experiment you
might want to run to the strategy or strategies that run it, and names what
each one costs beyond the baseline query.

Read [the strategy catalog](strategy-catalog.md) for what each strategy does
in detail, and [the evaluation method](methodology.md) for how to measure
strategies against each other fairly.

## This page names no winner

The bundled `aws-compute` corpus has 75 reviewed questions. Its own
[README](../datasets/aws-compute/README.md) records that the corpus holds
enough source material to answer only 17 of them. Forty questions carry an
empty `expected_chunks` list, because no chunk in the corpus answers them. A
recall number from this corpus states as much about the corpus as about a
strategy.

The evaluation method states this directly. The bundled example, in its
words, "is not large enough to justify a general architecture winner." Use
it to check that your pipeline runs end to end. Then run every strategy you
care about against your own documents before you trust a comparison.

## Cost categories used below

Every strategy answers a query with at least one call to the generation
model. The categories below name what a strategy adds on top of that
baseline.

- **Nothing added.** The strategy changes what gets embedded, indexed, or
  filtered, but it makes no extra model call at query time.
- **LLM call per query.** The strategy calls the generation model one or more
  extra times before it can answer, for example to rewrite a query or judge a
  retrieval round.
- **Optional extra, heavy model.** The strategy needs a `pip install
  'kb-arena[extra]'` group with a large dependency such as `torch` or
  `qiskit`, or a paid reranker API.
- **Neo4j instance.** The strategy reads a live graph database. `/ready`
  checks Neo4j only when the app loads `knowledge_graph`, `lightrag`, or
  `hybrid`.

`catalog.py` records one more field, `needs_embeddings`, which is `False`
only for BM25 and SPLADE. That flag governs whether a CLI run needs an embedding-provider
key before it starts. It is a narrower question than the cost categories
above, so BM25 and SPLADE are the two strategies you can try with no
embedding-provider key at all, though a full `benchmark` run still needs a
generation-model key for every strategy.

## Experiment to strategy

| What you want to try | Strategy | Cost beyond the baseline |
|---|---|---|
| A dense cosine-similarity baseline over chunk embeddings | Naive Vector | Nothing added |
| Prepend heading context to each chunk before embedding | Contextual Vector | Nothing added |
| Answer from question-and-answer pairs generated at index time | Q&A Pairs | LLM calls at index time, nothing added per query |
| Retrieve through entity relationships in a graph | Knowledge Graph | A Neo4j instance |
| Read a graph as a local walk and a community summary | LightRAG | A Neo4j instance, marked experimental |
| Route a question to vector or graph retrieval by intent | Hybrid | A Neo4j instance for its graph and procedural paths |
| Search cluster summaries built above the source chunks | RAPTOR | LLM calls at index time, nothing added per query |
| Traverse a heading tree with an LLM beam search | PageIndex | LLM calls per query, for the tree traversal |
| Rank by lexical term matching, no embeddings | BM25 | Nothing added, no embedding-provider key needed |
| Filter retrieval by tag, owner, or classification level | Metadata Filtered | Nothing added. Unreachable through `/chat` |
| Prefer the newest document version, or one as of a date | Temporal | Nothing added, marked experimental |
| Rescore dense candidates with a cross-encoder | Rerank Vector | Optional extra, `kb-arena[rerank]`, or a Cohere or Voyage key |
| Rescore dense candidates by quantum-state fidelity in NumPy | QISS | Nothing added by default |
| Rescore dense candidates with a Qiskit Aer SWAP-test circuit | SQR | Optional extra, `kb-arena[quantum]` |
| Rewrite the question into a hypothetical answer before retrieving | HyDE | One LLM call per query, marked experimental |
| Split the question into sub-queries, fuse results with RRF | Multi-Query | LLM calls per query, one per sub-query |
| Score a query against a passage per token | Late Interaction | Optional extra, `kb-arena[late-interaction]` |
| Score a query against a passage by learned term weights | SPLADE | Optional extra, `kb-arena[splade]`, no embedding key needed |
| Retrieve, judge the context, and retrieve again if needed | Agentic | LLM calls per query, capped at construction time |

Two catalog facts do not fit a table cell. First, `qiss` carries
`experimental=True` but no `default_benchmark` override, so it runs inside
the default `all` set today, unlike every other experimental strategy above.
Its config-gated multi-query mode, `KB_ARENA_QISS_DECOMPOSE`, adds LLM calls
per query on top of the default nothing-added cost. Second, no bundled
corpus carries the `classification`, `tags`, `document_family`, or `version`
fields that Metadata Filtered and Temporal read, so on those corpora both
retrieve what Naive Vector retrieves. Add those fields to your own corpus to
see either strategy diverge from the baseline.

## Read the retrieval trace next to the generated-answer score

[The evaluation method](methodology.md) explains why a full-answer score can
hide a retrieval failure. Read it again before you publish a result. It
names what an evidence record needs, including the corpus, the question
split, the qrel source, and the model versions behind the run.
