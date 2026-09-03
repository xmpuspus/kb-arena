"""Every document with a usable section reaches the question generator's prompt."""

from __future__ import annotations

import json
from pathlib import Path

from kb_arena.benchmark.question_gen import _load_doc_excerpts
from kb_arena.settings import settings

ROOT = Path(__file__).resolve().parents[1]


def _write_corpus(tmp_path, monkeypatch, docs: list[dict], corpus: str = "c") -> None:
    processed = tmp_path / corpus / "processed"
    processed.mkdir(parents=True)
    (processed / "documents.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))


def _doc(i: int, sections: int = 3, size: int = 1500) -> dict:
    return {
        "id": f"doc{i}",
        "title": f"Document {i}",
        "source": f"doc{i}.md",
        "sections": [
            {"id": f"s{j}", "title": f"Section {j}", "content": f"D{i}S{j} " * (size // 6)}
            for j in range(sections)
        ],
    }


def test_every_document_contributes_under_a_cap_the_first_few_would_have_filled(
    tmp_path, monkeypatch
):
    # Ten documents of three 1,500-char sections each. A global 6,000-char
    # cap in corpus order stops inside the second document. A per-document
    # share of 600 chars reaches all ten.
    _write_corpus(tmp_path, monkeypatch, [_doc(i) for i in range(10)])

    _, text, coverage = _load_doc_excerpts("c", max_chars=6000)

    assert coverage["documents"] == 10
    assert coverage["documents_in_prompt"] == 10
    assert coverage["documents_without_usable_section"] == 0
    for i in range(10):
        assert f"[Document {i} /" in text
    assert coverage["chars"] <= 6000


def test_a_document_with_only_short_sections_is_counted_not_hidden(tmp_path, monkeypatch):
    docs = [
        _doc(0),
        {
            "id": "tiny",
            "title": "Tiny",
            "source": "tiny.md",
            "sections": [{"id": "s0", "title": "Stub", "content": "too short"}],
        },
    ]
    _write_corpus(tmp_path, monkeypatch, docs)

    _, text, coverage = _load_doc_excerpts("c", max_chars=4000)

    assert coverage["documents"] == 2
    assert coverage["documents_in_prompt"] == 1
    assert coverage["documents_without_usable_section"] == 1
    assert "[Tiny /" not in text


def test_document_order_does_not_change_which_documents_contribute(tmp_path, monkeypatch):
    forward = [_doc(i) for i in range(8)]
    _write_corpus(tmp_path, monkeypatch, forward)
    _, text_forward, cov_forward = _load_doc_excerpts("c", max_chars=5000)

    _write_corpus(tmp_path / "again", monkeypatch, list(reversed(forward)))
    _, text_reversed, cov_reversed = _load_doc_excerpts("c", max_chars=5000)

    assert cov_forward["documents_in_prompt"] == cov_reversed["documents_in_prompt"] == 8
    seen_forward = {i for i in range(8) if f"[Document {i} /" in text_forward}
    seen_reversed = {i for i in range(8) if f"[Document {i} /" in text_reversed}
    assert seen_forward == seen_reversed == set(range(8))


def test_the_real_nist_corpus_reaches_the_prompt_in_full(monkeypatch):
    # The audit measured 36 of 130 NIST documents in the prompt under the
    # old global cap. The per-document share must carry every control.
    monkeypatch.setattr(settings, "datasets_path", str(ROOT / "datasets"))

    docs, _, coverage = _load_doc_excerpts("nist-800-171-r3")

    assert coverage["documents"] == len(docs) == 130
    # 28 controls are withdrawn. Each holds one stub section of 36 to 41
    # characters, so they cannot feed a question and the record says so.
    assert coverage["documents_without_usable_section"] == 28
    assert coverage["documents_in_prompt"] == 102
    # The room the 28 stubs leave unused flows to the other documents, so the
    # prompt fills the cap instead of leaving a fifth of it unspent.
    assert 45000 <= coverage["chars"] <= 50000


def test_a_small_corpus_keeps_every_section_the_old_walk_sent(tmp_path, monkeypatch):
    # The shape of aws-compute: 3 documents, 35 sections, about 8,200 chars,
    # far under the cap. An equal share that stopped at 2,000 chars per
    # document dropped eleven of those sections. Every section must stay.
    docs = [
        _doc(0, sections=12, size=234),
        _doc(1, sections=12, size=234),
        _doc(2, sections=11, size=234),
    ]
    _write_corpus(tmp_path, monkeypatch, docs)

    _, text, coverage = _load_doc_excerpts("c")

    assert text.count("]\n") == 35
    assert coverage["documents_in_prompt"] == 3
    assert coverage["documents_without_room"] == 0
    for d in docs:
        for s in d["sections"]:
            assert s["content"] in text


def test_a_huge_corpus_never_passes_the_cap_and_reports_the_subset(tmp_path, monkeypatch):
    _write_corpus(tmp_path, monkeypatch, [_doc(i, sections=1, size=600) for i in range(1000)])

    _, text, coverage = _load_doc_excerpts("c", max_chars=50000)

    assert len(text) <= 50000
    assert coverage["documents"] == 1000
    assert coverage["documents_selected"] < 1000
    assert coverage["documents_in_prompt"] == coverage["documents_selected"]
    assert coverage["per_document_share"] >= 200


def test_excerpts_end_on_a_word_boundary(tmp_path, monkeypatch):
    _write_corpus(tmp_path, monkeypatch, [_doc(0, sections=1, size=5000)])

    _, text, _ = _load_doc_excerpts("c", max_chars=700)

    assert len(text) <= 700
    assert not text.endswith("D0S")  # a cut inside a token would leave this stub
