# Strategy catalog

`kb_arena.strategies.catalog` is the source for built-in strategy names, default benchmark status,
experiment labels, and optional dependencies. `GET /strategies` adds the loaded status for the
current API process.

## Core architectures

### Naive Vector

Chunks documents, embeds each chunk, and ranks by cosine similarity. Use it as the dense baseline.

### Contextual Vector

Adds parent headings and document context before embedding. Compare it with Naive Vector to measure
whether local context resolves ambiguous terms.

### Q&A Pairs

Generates likely questions and answers at index time, then retrieves those pairs. It can do
well on recurring questions but may miss novel phrasing and has extra index-time cost.

### Knowledge Graph

Extracts entities and relationships into Neo4j and retrieves through graph query templates. It is
most relevant when a question depends on explicit links across topics.

### Hybrid

Routes by intent and combines vector and graph passages with Reciprocal Rank Fusion. Report its
added latency and graph dependency with its quality result.

### RAPTOR

Builds recursive summaries above the source chunks and retrieves across tree levels. It targets
broad or cross-document questions.

### PageIndex

Builds a hierarchy from document structure and uses an LLM to traverse it. It gives a
vector-free comparison for well-structured documents.

### BM25

Ranks by lexical term matching without embeddings. It is the keyless baseline and often remains
competitive when identifiers and exact terms dominate.

### Metadata Filtered

Applies an access filter, tags, owner, classification, and a document ID allow-list, inside
retrieval. Scalar fields go into the Chroma `where` clause. Tags go into a Python check against an
over-fetched candidate pool, then the result is cut to top_k. A chunk outside the filter never
reaches the ranked list. An unknown classification level raises instead of allowing or denying
everything by guess.

### Temporal

Prefers the newest eligible version of each document family and accepts an as-of date. Once a
newer eligible version is present among the candidates, every chunk from an older version is
dropped, so a superseded chunk cannot outrank its replacement. An unparseable as-of date raises.

### Rerank Vector

Retrieves a wider dense candidate set and rescores it with BGE, Cohere, or Voyage. Compare the
quality lift with its latency and provider cost. It is excluded from the default benchmark because
the local BGE backend needs the `kb-arena[rerank]` dependency group. BGE is the default. For Cohere
or Voyage, install `cohere` or `voyageai`, set `KB_ARENA_RERANKER_BACKEND` to `cohere` or `voyage`,
and set `KB_ARENA_COHERE_API_KEY` or `KB_ARENA_VOYAGE_API_KEY`. Run the strategy explicitly after
installing the selected backend.

## Experimental methods

### LightRAG

Reads the same Neo4j graph as Knowledge Graph, through two paths at once. Local retrieval walks the
one-hop neighborhood of entities matched in the question. Global retrieval groups fulltext-matched
candidates into a community by their shared neighbor links, then reads the community's member names
and descriptions as a summary. Every retrieved chunk records which path produced it. It is excluded
from the default benchmark because a degraded (Neo4j-unreachable) result fails the benchmark run,
unlike knowledge_graph and hybrid, which still carry that same risk in the default set today.

### QISS

Reranks dense candidates with pure NumPy state-fidelity calculations. Its single-query score is a
monotonic transform of cosine similarity, so a recall gain needs a separate multi-query operator.

### SQR

Reduces embeddings to a power-of-two dimension, amplitude-encodes them, and runs a Qiskit Aer
SWAP-test circuit. It is excluded from the default benchmark and needs `kb-arena[quantum]`.

### HyDE

Asks the LLM for a hypothetical answer to the question, then retrieves Naive Vector's index with
that hypothetical answer instead of the question text. It is excluded from the default benchmark
because the rewrite step adds one LLM call per query.

### Multi-Query

Asks the LLM for several sub-queries, retrieves Naive Vector's index once per sub-query, and fuses
the ranked chunk lists with Reciprocal Rank Fusion. It is excluded from the default benchmark
because each sub-query adds an LLM call per query.
### Late Interaction

Keeps one embedding per token instead of one pooled vector per passage, then reranks Naive Vector
candidates by MaxSim: for every query token, the best cosine match among the passage tokens,
averaged across query tokens. It is excluded from the default benchmark and needs
`kb-arena[late-interaction]`.

### SPLADE

Expands a query and each indexed passage into a weighted set of vocabulary terms, then scores a
query against a passage by the dot product of their term weights. It builds and reads its own
term-weight index, so it needs no embedding provider. It is excluded from the default benchmark and
needs `kb-arena[splade]`.
### Agentic

Retrieves, then asks the LLM whether the gathered context already answers the question, and
retrieves again with a refined query when it does not. A maximum iteration count and a maximum
LLM-call count are set at construction time and enforced every round, so the loop always stops even
when the judge keeps asking for another round. It is excluded from the default benchmark because it
costs several LLM calls per question.

The experiments answer research questions about operators and overhead. They do not define KB
Arena's main product category.

## Runtime states

The following terms are distinct:

- Registered: implemented in the installed package.
- Default: included when a benchmark uses `all`.
- Loaded: instantiated by the current API process.
- Optional: needs an extra dependency group.
- Experimental: available for research, with no general performance claim.

Clients should read `GET /strategies` rather than infer loaded status from a fixed number.
