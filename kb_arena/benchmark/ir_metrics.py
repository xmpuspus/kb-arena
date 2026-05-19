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
    exponential_gain: bool = False,
) -> float:
    """Normalized Discounted Cumulative Gain over top-k.

    Uses graded relevance from expected_relevance with hierarchical matching:
    a retrieved id earns the relevance of the first expected id it matches
    (exact or "::"-prefix). Each expected id contributes only on first match
    so DCG cannot exceed IDCG.

    `exponential_gain=True` switches to Burges et al. 2^rel - 1 gain, which
    amplifies the cost of mis-ranking a highly-graded item versus a low-graded
    one (SIGIR-standard formulation for graded relevance).
    Returns 0.0 if no relevant items exist.
    """
    if not expected_relevance or k <= 0:
        return 0.0

    def _gain(rel: float) -> float:
        return (2.0**rel - 1.0) if exponential_gain else rel

    expected_set = set(expected_relevance)
    top_k = retrieved_ids[:k]
    seen: set[str] = set()
    dcg = 0.0
    for i, rid in enumerate(top_k, start=1):
        m = _match_expected(rid, expected_set)
        if m is None or m in seen:
            continue
        seen.add(m)
        dcg += _gain(expected_relevance.get(m, 0.0)) / math.log2(i + 1)
    ideal_relevances = sorted(expected_relevance.values(), reverse=True)[:k]
    idcg = sum(_gain(rel) / math.log2(i + 1) for i, rel in enumerate(ideal_relevances, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(retrieved_ids: list[str], expected_ids: set[str]) -> float:
    """Average Precision — precision averaged over the ranks where relevant items appear.

    The per-query building block of MAP (Mean Average Precision). Captures
    where in the ranking the relevant items sit, weighted by precision at
    each hit position; complements NDCG which weights by log-discounted gain.
    Hierarchical matching consistent with `_match_expected`.
    """
    if not expected_ids:
        return 0.0
    seen: set[str] = set()
    s = 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        m = _match_expected(rid, expected_ids)
        if m is not None and m not in seen:
            seen.add(m)
            s += len(seen) / i
    return s / len(expected_ids)


def r_precision(retrieved_ids: list[str], expected_ids: set[str]) -> float:
    """Precision at rank R, where R = number of expected (relevant) items.

    Self-tuning k: a query with 3 relevant items is scored at P@3, a query
    with 8 at P@8. Removes the arbitrary fixed-k choice when comparing across
    queries with very different relevance-set sizes.
    """
    if not expected_ids:
        return 0.0
    r = len(expected_ids)
    matched: set[str] = set()
    for rid in retrieved_ids[:r]:
        m = _match_expected(rid, expected_ids)
        if m is not None:
            matched.add(m)
    return len(matched) / r


def bpref(
    retrieved_ids: list[str],
    expected_relevant: set[str],
    judged_nonrelevant: set[str],
) -> float:
    """Binary Preference (Buckley & Voorhees 2004) — robust to incomplete pools.

    Counts only judged non-relevant items ranked above each relevant item;
    unjudged items are ignored. When the judgment pool is partial (KB Arena
    typically labels a subset of queries), MAP silently penalizes systems for
    surfacing unjudged-but-truly-relevant items; bpref does not.

    bpref = (1/R) * Σ_{r ∈ relevant retrieved} (1 - |n above r| / min(R, N))

    R = |expected_relevant|, N = |judged_nonrelevant|, n = judged nonrel
    ranked above r.
    """
    if not expected_relevant:
        return 0.0
    r_count = len(expected_relevant)
    n_count = len(judged_nonrelevant)
    denom = max(1, min(r_count, n_count))
    seen_rel: set[str] = set()
    n_above = 0
    s = 0.0
    for rid in retrieved_ids:
        m = _match_expected(rid, expected_relevant)
        if m is not None and m not in seen_rel:
            seen_rel.add(m)
            # TREC bpref clamps the inner term so it stays in [0, 1] when
            # many non-relevant items rank above a single relevant one.
            n_clamped = min(n_above, denom)
            s += 1.0 - (n_clamped / denom)
        elif rid in judged_nonrelevant:
            n_above += 1
    return s / r_count


def compute_all(
    retrieved: list[RetrievedChunk],
    expected_ids: set[str],
    k: int = 5,
    expected_doc_ids: set[str] | None = None,
    judged_nonrelevant: set[str] | None = None,
    expected_relevance: dict[str, float] | None = None,
    exponential_gain: bool = False,
) -> RetrievalMetrics:
    """Compute the IR metric bundle for one query.

    If `expected_ids` is empty and `expected_doc_ids` is provided, falls back
    to doc-level matching. `judged_nonrelevant` is the set of chunks the
    labeler explicitly judged not relevant (bpref input); when omitted, a
    TREC-bpref-10-style proxy is used (retrieved-but-unjudged items).
    `expected_relevance` lifts NDCG to graded relevance when provided;
    `exponential_gain` selects the 2^rel - 1 gain function.
    """
    fallback = False
    if not expected_ids and expected_doc_ids:
        ids_in_top_k = [c.doc_id for c in retrieved]
        target = expected_doc_ids
        fallback = True
    else:
        ids_in_top_k = [c.chunk_id for c in retrieved]
        target = expected_ids

    relevance = expected_relevance if expected_relevance else {rid: 1.0 for rid in target}
    hits_set = sorted(
        {m for rid in ids_in_top_k[:k] if (m := _match_expected(rid, target)) is not None}
    )

    # bpref non-relevant proxy when caller did not supply explicit judgments:
    # treat retrieved-but-not-matched as judged non-relevant (TREC bpref-10).
    if judged_nonrelevant is None:
        nonrel = {rid for rid in ids_in_top_k if _match_expected(rid, target) is None}
    else:
        nonrel = judged_nonrelevant

    return RetrievalMetrics(
        k=k,
        recall_at_k=recall_at_k(ids_in_top_k, target, k),
        precision_at_k=precision_at_k(ids_in_top_k, target, k),
        hit_at_k=hit_at_k(ids_in_top_k, target, k),
        mrr=mrr(ids_in_top_k, target),
        ndcg_at_k=ndcg_at_k(ids_in_top_k, relevance, k, exponential_gain=exponential_gain),
        average_precision=average_precision(ids_in_top_k, target),
        r_precision=r_precision(ids_in_top_k, target),
        bpref=bpref(ids_in_top_k, target, nonrel),
        expected_count=len(target),
        retrieved_count=len(retrieved),
        hits=hits_set,
        fallback_doc_level=fallback,
    )
