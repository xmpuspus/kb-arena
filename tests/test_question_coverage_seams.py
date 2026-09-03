"""Excerpt pieces never merge words, and sections keep their separators."""

from __future__ import annotations

import json

import pytest

from kb_arena.benchmark import question_gen
from kb_arena.benchmark.question_gen import _DOC_SEPARATOR, _cut, _load_doc_excerpts
from kb_arena.settings import settings


def _corpus(tmp_path, monkeypatch, docs: list[dict], name: str = "seams") -> str:
    processed = tmp_path / name / "processed"
    processed.mkdir(parents=True)
    with open(processed / "docs.jsonl", "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    return name


def _doc(title: str, *sections: str) -> dict:
    return {
        "title": title,
        "sections": [{"title": f"s{n}", "content": body} for n, body in enumerate(sections)],
    }


def _body_words(text: str) -> set[str]:
    lines = [line for line in text.splitlines() if not line.startswith("[") and line != "---"]
    return set(" ".join(lines).split())


def test_a_second_pass_piece_starts_on_a_fresh_word(tmp_path, monkeypatch):
    words = [f"w{n:03d}" for n in range(400)]
    long_body = " ".join(words)
    docs = [
        _doc("Long", long_body),
        _doc("Short", "short body of fifty plus characters, nothing more here"),
    ]
    corpus = _corpus(tmp_path, monkeypatch, docs)

    _, text, coverage = _load_doc_excerpts(corpus, max_chars=1200)

    # Pass one cut Long at its share. Pass two gave it the room Short left.
    assert coverage["documents_in_prompt"] == 2
    assert _body_words(text) <= set(words) | set(
        "short body of fifty plus characters, nothing more here".split()
    )
    assert len(text) <= 1200


def test_sections_of_one_document_keep_their_separator(tmp_path, monkeypatch):
    docs = [
        _doc(
            "One",
            "a" * 80 + " end of first",
            "b" * 80 + " end of second",
            "c" * 80 + " end of third",
        )
    ]
    corpus = _corpus(tmp_path, monkeypatch, docs)

    _, text, _ = _load_doc_excerpts(corpus, max_chars=5000)

    assert text.count(_DOC_SEPARATOR) == 2
    assert f"end of first{_DOC_SEPARATOR}[One / s1]" in text
    assert "first[One" not in text


def test_section_separators_count_against_the_cap(tmp_path, monkeypatch):
    docs = [_doc("One", *[f"section {n} " + "word " * 40 for n in range(12)])]
    corpus = _corpus(tmp_path, monkeypatch, docs)

    _, text, _ = _load_doc_excerpts(corpus, max_chars=900)

    assert len(text) <= 900


def test_the_section_slice_ends_on_a_word_boundary(tmp_path, monkeypatch):
    body = " ".join(f"token{n:04d}" for n in range(600))  # about 5,400 chars
    corpus = _corpus(tmp_path, monkeypatch, [_doc("Big", body)])

    _, text, _ = _load_doc_excerpts(corpus, max_chars=50000)

    assert _body_words(text) <= set(body.split())
    assert len(text) <= question_gen._SECTION_SLICE + len("[Big / s0]\n")


def test_cut_treats_a_newline_as_a_boundary():
    text = "line one\nline two\nline three\nline four\n" + "x" * 30
    piece = _cut(text, 50)
    assert piece == "line one\nline two\nline three\nline four\n"


def test_cut_keeps_the_boundary_whitespace():
    assert _cut("alpha beta gamma", 12) == "alpha beta "


@pytest.mark.parametrize("count,max_chars", [(5, 260), (7, 300), (3, 210)])
def test_the_stride_re_checks_its_own_floor(tmp_path, monkeypatch, count, max_chars):
    docs = [_doc(f"D{n}", f"document {n} " + "word " * 30) for n in range(count)]
    corpus = _corpus(tmp_path, monkeypatch, docs, name=f"floor{count}")

    _, text, coverage = _load_doc_excerpts(corpus, max_chars=max_chars)

    assert coverage["per_document_share"] >= question_gen._MIN_DOC_SHARE
    assert len(text) <= max_chars
