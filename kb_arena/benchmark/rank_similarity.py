"""Rank-Biased Overlap — Webber, Moffat & Zobel 2010.

Top-weighted similarity between two ranked lists. Does not need a gold
ranking, so it answers questions the gold-relative metrics can't:
"Are these two strategies actually different in what they surface?",
"Did the new release perturb rankings beyond what we expected?", and
"Is hybrid actually fusing or just mirroring vector?".

Implements the extrapolated form (§4.3) so identical lists score 1.0
regardless of the evaluation depth — without extrapolation, top-k RBO
on identical lists tops out at 1 - p^k, which surprises everyone who
hasn't read the paper carefully.
"""

from __future__ import annotations

from collections.abc import Iterable


def rank_biased_overlap(
    list_a: Iterable[str],
    list_b: Iterable[str],
    p: float = 0.9,
    k: int | None = None,
) -> float:
    """Return RBO in [0, 1]. `p` is the user-persistence parameter (0.9 is the
    SIGIR default — heavy weight on the top of each list)."""
    a = list(list_a)
    b = list(list_b)
    if not a or not b:
        return 0.0
    depth = min(len(a), len(b))
    if k is not None:
        depth = min(depth, k)
    if depth == 0:
        return 0.0

    seen_a: set[str] = set()
    seen_b: set[str] = set()
    overlap = 0
    weighted_sum = 0.0
    for d in range(1, depth + 1):
        x, y = a[d - 1], b[d - 1]
        if x == y:
            overlap += 1
        else:
            if x in seen_b:
                overlap += 1
            if y in seen_a:
                overlap += 1
        seen_a.add(x)
        seen_b.add(y)
        agreement = overlap / d
        weighted_sum += agreement * (p ** (d - 1))

    base = (1.0 - p) * weighted_sum
    extrapolation = (overlap / depth) * (p**depth)
    return base + extrapolation
