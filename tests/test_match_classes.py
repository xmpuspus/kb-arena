"""A recall number says how it matched: a strict id, a parent label, or a document fallback."""

from __future__ import annotations

from kb_arena.benchmark.ir_metrics import MATCH_CLASSES, match_class
from kb_arena.benchmark.retriever_lab import _strict_share, count_match_classes


def test_match_class_names_the_kind_of_match():
    expected = {"doc::sec", "other"}
    assert match_class("doc::sec", expected) == "strict"
    assert (
        match_class("L0:doc::sec", expected) == "strict"
    ), "a strategy prefix is not a looser match"
    assert match_class("doc::sec::chunk3", expected) == "parent"
    assert match_class("doc::elsewhere", expected) == "unmapped"
    assert match_class("anything", set()) == "unmapped"


def test_doc_level_fallback_is_its_own_class():
    expected = {"doc"}
    assert match_class("doc::sec", expected, doc_level=True, doc_id="doc") == "doc"
    assert match_class("doc::sec", expected, doc_level=True, doc_id="another") == "unmapped"


def test_counts_cover_every_class_and_the_strict_share_is_honest():
    counts = count_match_classes(["strict", "parent", "parent", "unmapped"])
    assert counts == {"strict": 1, "parent": 2, "doc": 0, "unmapped": 1}
    assert set(counts) == set(MATCH_CLASSES)
    assert _strict_share(counts) == round(1 / 3, 4)
    assert _strict_share(count_match_classes(["unmapped"])) is None
    assert _strict_share(count_match_classes([])) is None
