# Retriever Lab

KB Arena answers a higher-level question — *which retrieval architecture fits these docs?* Retriever Lab adds the chunk-level evidence: for every benchmark question, exactly which chunks each strategy surfaced, which it missed, and how that translates to classical IR metrics.

This page walks through the metrics, when to look at one over another, how to add chunk-level ground truth to your own corpus, and what the aws-compute results actually mean.

## Why chunk-level visibility matters

`AnswerResult.sources: list[str]` (the pre-v0.5.0 contract) only exposes the docs that surfaced — not the chunk text, score, or rank. So when one strategy beats another, you couldn't say *why*. v0.5.0 adds `AnswerResult.retrieval: RetrievalTrace` with full chunk detail. Every strategy populates it, including knowledge_graph (synthesizes "chunks" from records) and qna_pairs (treats matched Q-A pairs as chunks).

## The five metrics

For each query and strategy, we compute:

| Metric | What it measures | When you care |
|---|---|---|
| **Recall@k** | Fraction of expected chunks that appeared anywhere in the top-k | Coverage. "Did the system find the relevant chunks at all?" |
| **Precision@k** | Fraction of top-k chunks that are relevant | Noise. "How many irrelevant chunks does the LLM have to wade through?" |
| **Hit@k** | 1 if any expected chunk is in top-k, else 0 | Pass/fail. "Did at least one relevant chunk surface?" |
| **MRR** | 1 / rank of the first relevant chunk (averaged across queries) | Ranking quality at the top. Sensitive to whether the best chunk is at rank 1 vs rank 3. |
| **NDCG@k** | Position-discounted cumulative gain, normalized | Combines hit and ranking. The metric most sensitive to *where* in the top-k each relevant chunk lands. |

### When to look at which

- **High Recall@k, low Precision@k** → strategy finds the right chunks but also drags in noise; consider tighter top-k or better re-ranking.
- **High Hit@k, low MRR** → relevant chunks are *in* the top-k but not at rank 1; the LLM may still fail because it weights early chunks more.
- **Recall and MRR both low, NDCG also low** → the strategy is missing the topic entirely; check chunk granularity, embedding model, or whether the corpus even contains the answer.

## Generating chunk-level ground truth (`label-chunks`)

The candidate pool is the union of BM25 and every retrieval-only index that answers a
probe query, which can be naive vector, contextual vector, Q and A pairs and RAPTOR,
plus a seeded random sample of the rest of the corpus. A probe cannot tell a missing
index from a provider that did not answer, so an index you built can still drop out.
The `pool` record in the output names the retrievers that actually answered, and that
record, not this sentence, is what a reader should trust for a given file.

If nothing but BM25 answers, `label-chunks` refuses to write. Labels drawn only from
what BM25 ranks high carry BM25's bias, and every strategy is then scored against
them for as long as that file lives. Build the indexes and label again, or pass
`--allow-bm25-only` to write a BM25-shaped gold set deliberately. The pool record
says which of the two happened. That flag also skips the embedding preflight, because
the reason to reach for it is usually that the provider is not answering.

The file carries one pool record, so it describes one pool. Labeling a corpus whose
stored labels were judged with a different pool stops and names what changed. Re-label
with `--force`, which regenerates every label under the current pool, or restore the
settings the earlier run used. A pool made only of what one retriever ranks high never
shows the judge a chunk it missed, and the random sample is what gives the labels their
negatives.

The judge is the model `KB_ARENA_GENERATE_MODEL` names, because labeling calls
`LLMClient.extract`. It is not the fast model and not the judge model; those two score
answers elsewhere. The prompt asks it to grade every candidate: 2 answers the
question, 1 supports the answer, 0 means the judge read the chunk and rejected it.
A judge that returns grades for only some of the candidates is accepted, and the
ones it left out are simply absent from the labels rather than recorded as 0. So a
missing chunk means unjudged, not rejected.

Output is a `{version, pool, labels}` file written to
`datasets/{corpus}/questions/expected_chunks.yaml`, where `labels` maps a question id to a
grade per chunk and `pool` records which retrievers and how many random chunks fed the judge.

```bash
# Build BM25 first (label-chunks needs it)
kb-arena build-vectors --corpus aws-compute --strategy bm25

# Label (cost-capped via KB_ARENA_BENCHMARK_COST_CAP_USD)
kb-arena label-chunks --corpus aws-compute
# 75 labeled, $0.34, ~2 minutes
```

The command is idempotent — re-running picks up where it left off. Pass `--force` to relabel.

## Hierarchical chunk matching

Strategies emit chunks at different granularities:

- BM25: section-level (`doc::section`)
- naive_vector / contextual_vector: sub-chunked (`doc::section::0`, `doc::section::1`)
- RAPTOR: prefixed by level (`L0:doc::section::0`)
- QnA: pair-id prefixed (`qna:pair-001`)
- Knowledge graph: FQN-prefixed (`graph:aws.lambda`)
- PageIndex: leaf-id prefixed (`pageindex:doc::section`)

The IR metrics module performs hierarchical matching: a section-level expected ID matches any sub-chunk under it. It also strips known strategy-namespace prefixes (`L0:`, `L1:`, `L2:`, `qna:`, `graph:`, `pageindex:`) so RAPTOR L0 chunks score equivalently to naive_vector chunks pointing at the same content.

Doc-level fallback: if a question has no chunk-level labels, the matcher falls back to checking `chunk.doc_id ∈ ground_truth.source_refs`. Useful when chunk labels don't exist yet but you have document-level references.

## Adding labels to your own corpus

1. Ingest your docs and build the BM25 index: `kb-arena ingest ... && kb-arena build-vectors --strategy bm25`
2. Generate questions if you don't have them: `kb-arena generate-questions --corpus my-docs --count 50`
3. Label: `kb-arena label-chunks --corpus my-docs`
4. Spot-check 5 random labels in `datasets/my-docs/questions/expected_chunks.yaml`. If the LLM judge was too strict for your domain, tune `JUDGE_PROMPT` in `kb_arena/benchmark/expected_chunks.py`.

## Interpreting the aws-compute results

From run `855aac4e` (top-5):

| Strategy | R@5 | P@5 | Hit@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|
| **contextual_vector** | **35.5%** | **24.5%** | 46.7% | **0.433** | **0.388** |
| naive_vector | 35.2% | 23.2% | 46.7% | 0.414 | 0.367 |
| raptor | 35.2% | 23.2% | 46.7% | 0.414 | 0.367 |
| bm25 | 27.5% | 17.1% | 44.0% | 0.352 | 0.278 |
| hybrid | 8.0% | 4.8% | 9.3% | 0.093 | 0.086 |
| pageindex | 6.1% | 5.0% | 14.7% | 0.111 | 0.076 |
| qna_pairs | 0.0% | 0.0% | 0.0% | 0.000 | 0.000 |
| knowledge_graph | 0.0% | 0.0% | 0.0% | 0.000 | 0.000 |

What this one run shows, and what it does not:

Read every line below as an observation about run `855aac4e` on a three-document
corpus, not as a finding about the strategies. One run has no spread, so a gap of a
few points is not distinguishable from noise.

Repeat the run and read the spread:

```bash
kb-arena retriever-lab --corpus aws-compute   # each run gets its own id
kb-arena variance --corpus aws-compute --metric mean_recall_at_k
```

`variance` reads a Retriever Lab run and a benchmark run alike. Name the metric,
because the default is `accuracy_by_tier` and these are retrieval metrics.

Two runs give a range a reader can misread as a bound. Three is the smallest
number that says anything about spread, and the command says when a row rests on
fewer.

Runs from different commits are listed and never averaged. The gap between two
builds is a change, not noise.

1. **Contextual Vector and Naive Vector are not separated by this run.** Their Recall
   and Hit are within 0.3 points and their MRR within 0.02. That is a difference this
   run cannot resolve, so it is not evidence that either ranks better.
2. **RAPTOR reports the same numbers as Naive Vector.** RAPTOR's L0 chunks share
   identity with naive vector's chunks, so at top-5 the two retrieve the same set.
   This says nothing about RAPTOR's higher layers, which chunk-level labels on a
   three-document corpus do not reach.
3. **BM25 is 7.7 points below the embedding strategies on Recall@5 in this run**, and
   0.06 below on MRR. Whether that ordering holds on another corpus is an open
   question this run does not answer.
4. **Hybrid's 8.0% is not a measurement of hybrid retrieval.** Hybrid fuses a vector
   leg and a graph leg, and the graph leg needs Neo4j. This run's stored result does
   not record whether Neo4j answered, so the number cannot be attributed either to
   the strategy or to the deployment. Treat the row as unusable and re-run it with
   `kb-arena retriever-lab --strategies hybrid` against a Neo4j you can see.
5. **QnA Pairs and Knowledge Graph score 0.0% because they are not being measured.**
   Both emit ids in their own namespace, Q and A pairs and entity names, and the
   chunk-level labels of this corpus contain neither. A zero here means unmeasured,
   not bad.

Forty of the 75 questions have empty `expected_chunks`. They ask about EC2, EKS and
Batch, which the three-document demo corpus does not cover, so no chunk in the corpus
could be correct for them. Those questions contribute nothing to the numbers above,
which is why the table is a demonstration of the metrics and not a benchmark result.

## Roadmap

- **v1.1**: Reranker comparison — drop a cross-encoder / Cohere rerank / bge-reranker between retrieval and generation, measure how much each lifts MRR / NDCG.
- Per-tier and per-question-type IR breakdowns.
- Graded relevance (1.0 / 0.5 / 0.0) instead of binary, with re-labeling support in `label-chunks`.
