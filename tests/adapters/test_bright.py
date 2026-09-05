"""BRIGHT: the parse, and the guards it must not skip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_arena.adapters import ADAPTERS, ChecksumMismatchError
from kb_arena.adapters.bright import BrightAdapter


def test_parse_document_splits_the_chunk_suffix_from_the_article_id():
    raw = {"id": "sample_topic/Sample_Article_3.txt", "content": "A chunk of text."}

    section = BrightAdapter.parse_document(raw)

    assert section["source_doc_id"] == "sample_topic/Sample_Article"
    assert section["source_section_id"] == "3"
    assert section["text"] == "A chunk of text."


def test_parse_question_resolves_gold_ids_to_the_same_doc_ids():
    raw = {
        "query": "Why does the sample article matter?",
        "gold_answer": "Because it says so.",
        "gold_ids": [
            "sample_topic/Sample_Article_0.txt",
            "sample_topic/Sample_Article_1.txt",
        ],
    }

    question = BrightAdapter.parse_question(raw)

    assert question["gold_sections"] == [
        {"source_doc_id": "sample_topic/Sample_Article", "source_section_id": "0"},
        {"source_doc_id": "sample_topic/Sample_Article", "source_section_id": "1"},
    ]


def test_a_moving_revision_is_refused():
    template = ADAPTERS["bright"]().manifest_template()
    with pytest.raises(ValidationError, match="moving target"):
        type(template)(**{**template.model_dump(), "revision": "master"})


def test_a_checksum_mismatch_stops_the_run(tmp_path):
    adapter = ADAPTERS["bright"]()
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"source_doc_id": "sample_topic/Sample_Article"}\n')

    with pytest.raises(ChecksumMismatchError, match="cannot be cited"):
        adapter.verify(path, "d" * 64)


def test_download_only_and_says_the_terms():
    adapter = ADAPTERS["bright"]()
    template = adapter.manifest_template()

    assert adapter.download_only is True
    assert template.license == "CC-BY-4.0"
    assert template.redistributable is True
    assert "creativecommons.org" in template.license_url


def test_refuses_to_fetch_and_says_what_to_do_instead():
    with pytest.raises(NotImplementedError) as refused:
        ADAPTERS["bright"]().build(Path("/tmp/bright"))

    message = str(refused.value)
    assert "download-only" in message
    assert "huggingface.co/datasets/xlangai/BRIGHT" in message
