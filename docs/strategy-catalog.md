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

### Rerank Vector

Retrieves a wider dense candidate set and rescores it with BGE, Cohere, or Voyage. Compare the
quality lift with its latency and provider cost. It is excluded from the default benchmark because
the local BGE backend needs the `kb-arena[rerank]` dependency group. BGE is the default. For Cohere
or Voyage, install `cohere` or `voyageai`, set `KB_ARENA_RERANKER_BACKEND` to `cohere` or `voyage`,
and set `KB_ARENA_COHERE_API_KEY` or `KB_ARENA_VOYAGE_API_KEY`. Run the strategy explicitly after
installing the selected backend.

## Experimental methods

### QISS

Reranks dense candidates with pure NumPy state-fidelity calculations. Its single-query score is a
monotonic transform of cosine similarity, so a recall gain needs a separate multi-query operator.

### SQR

Reduces embeddings to a power-of-two dimension, amplitude-encodes them, and runs a Qiskit Aer
SWAP-test circuit. It is excluded from the default benchmark and needs `kb-arena[quantum]`.

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
