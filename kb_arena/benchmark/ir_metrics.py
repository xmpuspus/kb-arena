"""Classical IR metrics — pure functions, no I/O.

All five metrics operate on a list of retrieved IDs and a set of expected IDs.
compute_all() falls back to doc-level matching (chunk.doc_id) when chunk-level
expected IDs are absent — preserves usefulness when a corpus has no chunk
labels but does have ground-truth source_refs.

Identifier hierarchy: chunk IDs are "::"-delimited paths (e.g. doc::section::0).
When an expected_id is a strict prefix of a retrieved chunk_id (e.g. expected
doc::section matches retrieved doc::section::0), it counts as a match. This
lets section-level ground truth match against sub-chunk retrievals.
"""

from __future__ import annotations

import math

from kb_arena.models.benchmark import RetrievalMetrics
from kb_arena.models.retrieval import RetrievedChunk

_STRATEGY_NAMESPACE_PREFIXES = ("L0:", "L1:", "L2:", "qna:", "graph:", "pageindex:")


def _candidate_ids(chunk_id: str) -> list[str]:
    """Yield matchable forms of a chunk_id.

    1. The chunk_id itself.
    2. The chunk_id with a known strategy-namespace prefix stripped — RAPTOR's
       'L0:doc::sec' is the same chunk as naive_vector's 'doc::sec', and
       expected labels are written without a strategy prefix.
    """
    candidates = [chunk_id]
    for p in _STRATEGY_NAMESPACE_PREFIXES:
        if chunk_id.startswith(p):
            candidates.append(chunk_id[len(p) :])
            break
    return candidates


def _match_expected(chunk_id: str, expected: set[str]) -> str | None:
    """Return the expected_id matched by chunk_id (exact or hierarchical prefix), else None.

    Tries the raw chunk_id first; if no match, retries with strategy-namespace
    prefixes stripped. Hierarchical match: each '::'-delimited prefix of a
    candidate counts as a match if it exists in expected.
    """
    if not expected:
        return None
    for cand in _candidate_ids(chunk_id):
        if cand in expected:
            return cand
        parts = cand.split("::")
        for n in range(len(parts) - 1, 0, -1):
            prefix = "::".join(parts[:n])
            if prefix in expected:
                return prefix
    return None


def recall_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Fraction of expected items that appear in the top-k retrieval.

    Hierarchical matching: a retrieved id matches an expected id if they're equal
    or the expected id is a "::"-prefix of the retrieved id. Duplicates count once.
    """
    if not expected_ids:
        return 0.0
    matched: set[str] = set()
    for rid in retrieved_ids[:k]:
        m = _match_expected(rid, expected_ids)
        if m:
            matched.add(m)
    return len(matched) / len(expected_ids)


def precision_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Fraction of top-k items that are expected (with hierarchical matching).

    Counts duplicates as separate positions.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if _match_expected(rid, expected_ids) is not None)
    return hits / min(k, len(top_k))


def hit_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> int:
    """1 if any expected item is in top-k (with hierarchical matching), else 0."""
    top_k = retrieved_ids[:k]
    return 1 if any(_match_expected(rid, expected_ids) is not None for rid in top_k) else 0


def mrr(retrieved_ids: list[str], expected_ids: set[str]) -> float:
    """Reciprocal rank of the first expected item (1.0 if rank 1, 0.5 if rank 2, ...)."""
    for i, rid in enumerate(retrieved_ids, start=1):
        if _match_expected(rid, expected_ids) is not None:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    expected_relevance: dict[str, float],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain over top-k.

    Uses graded relevance from expected_relevance with hierarchical matching:
    a retrieved id earns the relevance of the first expected id it matches
    (exact or "::"-prefix). Each expected id contributes only on first match
    so DCG cannot exceed IDCG.
    Returns 0.0 if no relevant items exist.
    """
    if not expected_relevance or k <= 0:
        return 0.0
    expected_set = set(expected_relevance)
    top_k = retrieved_ids[:k]
    seen: set[str] = set()
    dcg = 0.0
    for i, rid in enumerate(top_k, start=1):
        m = _match_expected(rid, expected_set)
        if m is None or m in seen:
            continue
        seen.add(m)
        dcg += expected_relevance.get(m, 0.0) / math.log2(i + 1)
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
    hits_set = sorted(
        {m for rid in ids_in_top_k[:k] if (m := _match_expected(rid, target)) is not None}
    )

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
