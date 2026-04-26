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

Run BM25 over the corpus to retrieve top-N candidates per question, then ask Claude Haiku to mark which are actually relevant. Output is a `{question_id: [chunk_id, ...]}` map written to `datasets/{corpus}/questions/expected_chunks.yaml`.

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

Five things this tells us:

1. **Contextual Vector wins on ranking, not coverage.** Contextual and Naive have nearly identical Recall and Hit; the win shows up in MRR and NDCG. The heading-path prefix nudges the *first* relevant chunk into a higher rank, which matters for downstream answer quality.
2. **RAPTOR's L0 layer is doing the work.** RAPTOR's numbers track naive_vector exactly because L0 chunks share identity with naive_vector chunks. The L1/L2 cluster summaries don't have section-level labels (they're synthetic content), so they don't contribute to chunk-level recall — but they would help on tier 4/5 multi-hop questions where the answer requires synthesis across sections.
3. **BM25 trails the embeddings by ~8 points on Recall@5** but the gap on MRR is much smaller — when BM25 hits, it tends to hit at rank 1 because exact keyword matches dominate. For "look up exact term" questions BM25 is fine; for synonyms/paraphrasing it loses.
4. **Hybrid drops to 8% because Neo4j wasn't running.** The graph leg returns empty traces, dragging down the fused result. With Neo4j connected and a graph built, hybrid usually beats vector on relational questions. This is itself a useful diagnostic — IR metrics expose infrastructure gaps fast.
5. **QnA and Knowledge Graph score 0% chunk-level.** Both operate on different identity spaces (Q-A pairs and entity FQNs respectively). They need doc-level ground truth, or labels in their own identity space. The doc-level fallback path will rescue them when `ground_truth.source_refs` matches the doc IDs in your corpus (it doesn't for aws-compute — the source_refs use AWS docs URL paths but doc IDs are short slugs).

The 40 questions out of 75 with empty `expected_chunks` reference EC2, EKS, Batch, etc. — services not in the 3-doc demo corpus. They contribute 0 to every metric, which pulls means down. Filtering to labeled questions only would roughly double these percentages — but the unfiltered numbers are the honest signal: *this corpus is incomplete for this question set*.

## Roadmap

- **v1.1**: Reranker comparison — drop a cross-encoder / Cohere rerank / bge-reranker between retrieval and generation, measure how much each lifts MRR / NDCG.
- Per-tier and per-question-type IR breakdowns.
- Graded relevance (1.0 / 0.5 / 0.0) instead of binary, with re-labeling support in `label-chunks`.
