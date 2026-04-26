"""Classical IR metrics — pure functions, no I/O.

All five metrics operate on a list of retrieved IDs and a set of expected IDs.
compute_all() falls back to doc-level matching (chunk.doc_id) when chunk-level
expected IDs are absent — preserves usefulness when a corpus has no chunk
labels but does have ground-truth source_refs.
"""

from __future__ import annotations

import math

from kb_arena.models.benchmark import RetrievalMetrics
from kb_arena.models.retrieval import RetrievedChunk


def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Fraction of expected items that appear in the top-k retrieval.

    Duplicates in retrieved_ids count once — recall is over the unique set.
    """
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & expected_ids)
    return hits / len(expected_ids)


def precision_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Fraction of top-k items that are expected. Counts duplicates as separate
    positions (a doc retrieved twice in top-k counts twice).
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in expected_ids)
    return hits / min(k, len(top_k))


def hit_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> int:
    """1 if any expected item is in top-k, else 0."""
    top_k = retrieved_ids[:k]
    return 1 if any(rid in expected_ids for rid in top_k) else 0


def mrr(retrieved_ids: list[str], expected_ids: set[str]) -> float:
    """Reciprocal rank of the first expected item (1.0 if rank 1, 0.5 if rank 2, ...)."""
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    expected_relevance: dict[str, float],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain over top-k.

    Uses graded relevance from expected_relevance; binary (1.0 / 0.0) is fine
    for the default callers. Repeated IDs in retrieved_ids contribute relevance
    only on first occurrence so DCG cannot exceed IDCG.
    Returns 0.0 if no relevant items exist.
    """
    if not expected_relevance or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    seen: set[str] = set()
    dcg = 0.0
    for i, rid in enumerate(top_k, start=1):
        if rid in seen:
            continue
        seen.add(rid)
        dcg += expected_relevance.get(rid, 0.0) / math.log2(i + 1)
    ideal_relevances = sorted(expected_relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_relevances, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_all(
    retrieved: list[RetrievedChunk],
    expected_ids: set[str],
    k: int = 5,
    expected_doc_ids: set[str] | None = None,
) -> RetrievalMetrics:
    """Compute all five IR metrics for one query.

    If expected_ids is empty and expected_doc_ids is provided, falls back to
    doc-level matching: chunk.doc_id is checked against expected_doc_ids.
    This makes the metrics useful even before chunk labels exist.
    """
    fallback = False
    if not expected_ids and expected_doc_ids:
        ids_in_top_k = [c.doc_id for c in retrieved]
        target = expected_doc_ids
        fallback = True
    else:
        ids_in_top_k = [c.chunk_id for c in retrieved]
        target = expected_ids

    relevance = {rid: 1.0 for rid in target}
    hits_set = sorted({rid for rid in ids_in_top_k[:k] if rid in target})

    return RetrievalMetrics(
        k=k,
        recall_at_k=recall_at_k(ids_in_top_k, target, k),
        precision_at_k=precision_at_k(ids_in_top_k, target, k),
        hit_at_k=hit_at_k(ids_in_top_k, target, k),
        mrr=mrr(ids_in_top_k, target),
        ndcg_at_k=ndcg_at_k(ids_in_top_k, relevance, k),
        expected_count=len(target),
        retrieved_count=len(retrieved),
        hits=hits_set,
        fallback_doc_level=fallback,
    )
