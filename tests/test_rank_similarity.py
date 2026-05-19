"""Rank-Biased Overlap — measure ranking similarity between two strategies."""

from __future__ import annotations

import pytest

from kb_arena.benchmark.rank_similarity import rank_biased_overlap


def test_identical_top_k_is_one():
    a = ["x", "y", "z", "w"]
    b = ["x", "y", "z", "w"]
    assert rank_biased_overlap(a, b, p=0.9) == pytest.approx(1.0)


def test_completely_disjoint_is_zero():
    a = ["a", "b", "c", "d"]
    b = ["e", "f", "g", "h"]
    assert rank_biased_overlap(a, b, p=0.9) == 0.0


def test_symmetric():
    a = ["a", "b", "c", "d"]
    b = ["a", "c", "b", "e"]
    assert rank_biased_overlap(a, b, p=0.9) == pytest.approx(rank_biased_overlap(b, a, p=0.9))


def test_top_match_weighted_more_than_tail():
    # Same item at rank 1 in both, different elsewhere — should beat the
    # mirror case where the shared item sits at rank 4.
    top_shared = (["a", "x", "y", "z"], ["a", "p", "q", "r"])
    tail_shared = (["x", "y", "z", "a"], ["p", "q", "r", "a"])
    rbo_top = rank_biased_overlap(*top_shared, p=0.9)
    rbo_tail = rank_biased_overlap(*tail_shared, p=0.9)
    assert rbo_top > rbo_tail


def test_partial_overlap_in_range():
    a = ["a", "b", "c", "d"]
    b = ["a", "b", "x", "y"]
    rbo = rank_biased_overlap(a, b, p=0.9)
    assert 0.0 < rbo < 1.0


def test_empty_inputs_return_zero():
    assert rank_biased_overlap([], ["a"], p=0.9) == 0.0
    assert rank_biased_overlap(["a"], [], p=0.9) == 0.0


def test_unequal_length_uses_shorter():
    a = ["x", "y"]
    b = ["x", "y", "z", "w", "u"]
    # Top-2 are identical; with extrapolated RBO (Webber 2010 §4.3) identical
    # within the evaluation depth means RBO = 1.0.
    assert rank_biased_overlap(a, b, p=0.9, k=2) == pytest.approx(1.0)
