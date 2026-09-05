"""MultiHop-RAG: the parse, and the guards it must not skip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_arena.adapters import ADAPTERS, ChecksumMismatchError
from kb_arena.adapters.multihop_rag import MultiHopRagAdapter


def test_parse_document_keeps_the_article_url_as_the_doc_id():
    raw = {
        "category": "technology",
        "author": "A. Writer",
        "published_at": "2024-01-01T00:00:00+00:00",
        "body": "A small news article body, written for this test.",
        "title": "A Small Article",
        "url": "https://example.org/news/small-article",
        "source": "Example Times",
    }

    section = MultiHopRagAdapter.parse_document(raw)

    assert section["source_doc_id"] == "https://example.org/news/small-article"
    assert section["source_section_id"] == "0"
    assert section["text"] == raw["body"]


def test_parse_question_keeps_the_evidence_trail_to_the_same_doc_ids():
    raw = {
        "query": "Which outlet reported the small article?",
        "answer": "Example Times",
        "question_type": "inference_query",
        "evidence_list": [
            {"url": "https://example.org/news/small-article", "fact": "..."},
            {"url": "https://example.org/news/other-article", "fact": "..."},
        ],
    }

    question = MultiHopRagAdapter.parse_question(raw)

    assert question["question"] == raw["query"]
    assert question["source_doc_ids"] == [
        "https://example.org/news/small-article",
        "https://example.org/news/other-article",
    ]


def test_a_moving_revision_is_refused():
    template = ADAPTERS["multihop-rag"]().manifest_template()
    with pytest.raises(ValidationError, match="moving target"):
        type(template)(**{**template.model_dump(), "revision": "latest"})


def test_a_checksum_mismatch_stops_the_run(tmp_path):
    adapter = ADAPTERS["multihop-rag"]()
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"source_doc_id": "https://example.org/a"}\n')

    with pytest.raises(ChecksumMismatchError, match="cannot be cited"):
        adapter.verify(path, "b" * 64)


def test_download_only_and_says_the_terms():
    adapter = ADAPTERS["multihop-rag"]()
    template = adapter.manifest_template()

    assert adapter.download_only is True
    assert template.license == "ODC-BY-1.0"
    assert template.redistributable is True
    assert "opendatacommons.org" in template.license_url


def test_refuses_to_fetch_and_says_what_to_do_instead():
    with pytest.raises(NotImplementedError) as refused:
        ADAPTERS["multihop-rag"]().build(Path("/tmp/multihop-rag"))

    message = str(refused.value)
    assert "download-only" in message
    assert "huggingface.co/datasets/yixuantt/MultiHopRAG" in message
