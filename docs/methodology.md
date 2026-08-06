# Evaluation method

KB Arena compares retrieval architectures on a shared corpus and question set. A result is useful
only when the experiment keeps its data, judgments, configuration, and limits visible.

## Two evaluation tracks

### Retrieval-only

`kb-arena retriever-lab` asks each strategy for ranked chunks and compares them with qrels. This
track isolates indexing and ranking from answer generation.

Metrics include Recall@k, Precision@k, Hit@k, MRR, NDCG@k, MAP, R-Precision, and bpref. Reports can
also include per-tier results, bootstrap confidence intervals, and a deeper retrieval ceiling.

### Full answer evaluation

`kb-arena benchmark` retrieves context, generates an answer, and records answer scores, sources,
latency, tokens, cost, retries, and reliability. Optional RAGAS metrics add faithfulness, context
precision, context recall, and answer relevancy.

Full answer scores can hide retrieval failures. Always read them with the retrieval trace.

## Question and qrel design

Use real user questions when available. Remove sensitive data, stratify by intent, and keep a source
record for each item. Synthetic questions can fill coverage gaps, but they do not prove production
fit by themselves.

Each qrel should use a stable source identifier. For a hierarchical corpus, map retrieved subchunks
to their canonical parent. For graph retrieval, keep the source document and section on every
entity or relationship result.

Review every public qrel. Record the reviewer, source anchor, reason, and relevance grade.

## Development and holdout data

Split questions before tuning:

- Development: choose strategies, chunking, models, and search ranges.
- Validation: select among configurations without opening the holdout.
- Holdout: run once for the public comparison.

Keep related question families in one split when possible. This reduces leakage from repeated source
sections or near-duplicate questions.

## Fair comparisons

Use the same corpus snapshot, question set, qrels, metric implementation, and top-k definition.
Record architecture-native preprocessing because graph, generated-index, and hierarchical methods
do different index-time work.

Report both common and native tracks when needed:

- Common track: shared retrieval cutoff and qrels for direct comparison.
- Native track: the configuration each architecture would use in practice.

Do not turn a failure to map source identifiers into a zero-quality product claim. Mark the run
invalid for that strategy and repair the provenance path.

## Statistical interpretation

Prefer paired, per-question comparisons. Report the effect size, win rate, confidence interval, and
paired test with the mean score. A small positive mean with a wide interval is not a decision.

Use Rank-Biased Overlap when two strategies return different rankings and gold labels are sparse.
Use a Pareto view when quality, latency, and cost trade off.

## Evidence record

Every public result should include:

- corpus name, source version, retrieval date, and content hash;
- question count, type distribution, and split;
- qrel source and review method;
- loaded strategy names and optional dependencies;
- generation, judge, embedding, and reranker model versions;
- chunking, top-k, graph schema, and relevant settings;
- metric definitions and confidence method;
- run ID, timestamp, cost, latency, and known failures;
- result files and the command used to create them.

## Current bundled evidence

The AWS Compute example has three short documents and 75 questions. Only 35 questions have
chunk-level labels. It is useful for learning the workflow and checking report calculations. It is
not large enough to justify a general architecture winner.

The next public comparison uses NIST SP 800-171 Revision 3. Stable control identifiers, official
hierarchy, and source cross-references give auditable qrels for lexical, dense, graph, hybrid, and
hierarchical retrieval.
