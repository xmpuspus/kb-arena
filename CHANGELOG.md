# Changelog

All notable changes to KB Arena.

## [0.5.0] — 2026-04-26 — Retriever Lab

### Added
- Classical IR metrics computed for every benchmark query: Recall@k, Precision@k, Hit@k, MRR, NDCG@k.
- `RetrievalTrace` and `RetrievedChunk` models on `AnswerResult` — every strategy now exposes the chunks it surfaced with rank, score, and source strategy.
- `Question.expected_chunks` field; `load_questions()` merges `expected_chunks.yaml` automatically.
- `RetrievalMetrics` model attached to `AnswerRecord.retrieval_metrics`; `BenchmarkResult` gains aggregate `mean_recall_at_k`, `mean_precision_at_k`, `mean_hit_at_k`, `mean_mrr`, `mean_ndcg_at_k`.
- New CLI command `kb-arena retriever-lab` — retrieval-only benchmark with live Rich metrics table; ~10x cheaper than full `benchmark` because LLM generation is stubbed.
- New CLI command `kb-arena label-chunks` — generate `expected_chunks.yaml` ground truth via BM25 + Haiku judge. Idempotent and cost-capped.
- New `--top-k` flag on `kb-arena benchmark` (default 5).
- New web page `/retriever-lab` — aggregate metrics card per strategy, plus per-question drill-down with HIT/MISS chunk highlighting.
- New API endpoints `GET /api/retriever-lab/runs` and `GET /api/retriever-lab/{run_id}`.
- Hierarchical chunk-id matching: section-level expected IDs match sub-chunk retrievals (`doc::sec` matches `doc::sec::0`); strategy-namespace prefixes (`L0:`, `qna:`, `graph:`, `pageindex:`) are stripped before matching.
- Doc-level fallback in IR metrics: when chunk labels are absent, match against `chunk.doc_id ∈ ground_truth.source_refs`.

### Changed
- All 8 strategies now populate `AnswerResult.retrieval` with stable chunk IDs.
- Benchmark Markdown report gains a "Retrieval Quality (top-k)" section.
- Hybrid strategy preserves sub-strategy `source_strategy` per chunk during fusion.
- BM25 index format includes `chunk_ids` for stable identity across runs (older indexes still load with synthesized IDs).
- ChromaDB telemetry warnings suppressed in retriever-lab to keep terminal output clean.

### Fixed
- BM25 chunk identifiers now stable across runs (previously index-position only).
- `is_hit` flag in retriever-lab JSON now uses hierarchical matching so vector sub-chunks correctly tag as HIT against section-level labels.

### Tests
- Test suite grows from 514 to 558 tests; coverage adds `tests/test_ir_metrics.py`, `tests/test_retrieval_trace.py`, `tests/test_retriever_lab_runner.py`, `tests/test_label_chunks_cli.py`.
