"""MIRACL: the parse, and the guards it must not skip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_arena.adapters import ADAPTERS, ChecksumMismatchError
from kb_arena.adapters.miracl import MiraclAdapter


def test_parse_document_splits_the_docid_into_article_and_passage():
    raw = {"docid": "39#0", "title": "Sample Article", "text": "A sample passage."}

    section = MiraclAdapter.parse_document(raw)

    assert section["source_doc_id"] == "39"
    assert section["source_section_id"] == "0"
    assert section["text"] == "A sample passage."


def test_parse_qrel_reads_the_trec_line_and_matches_the_same_ids():
    raw = {"line": "8 Q0 39#0 1"}

    judgment = MiraclAdapter.parse_qrel(raw)

    assert judgment["query_id"] == "8"
    assert judgment["source_doc_id"] == "39"
    assert judgment["source_section_id"] == "0"
    assert judgment["relevance"] == 1


def test_a_moving_revision_is_refused():
    template = ADAPTERS["miracl"]().manifest_template()
    with pytest.raises(ValidationError, match="moving target"):
        type(template)(**{**template.model_dump(), "revision": "latest"})


def test_a_checksum_mismatch_stops_the_run(tmp_path):
    adapter = ADAPTERS["miracl"]()
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"source_doc_id": "39"}\n')

    with pytest.raises(ChecksumMismatchError, match="cannot be cited"):
        adapter.verify(path, "f" * 64)


def test_download_only_and_says_the_terms():
    """The corpus is Wikipedia text, so its licence governs, not the packaging tag."""
    adapter = ADAPTERS["miracl"]()
    template = adapter.manifest_template()

    assert adapter.download_only is True
    assert template.license == "CC-BY-SA-4.0"
    assert template.redistributable is True
    assert "Apache-2.0" in template.attribution, "the packaging's own licence is named too"


def test_refuses_to_fetch_and_says_what_to_do_instead():
    with pytest.raises(NotImplementedError) as refused:
        ADAPTERS["miracl"]().build(Path("/tmp/miracl"))

    message = str(refused.value)
    assert "download-only" in message
    assert "huggingface.co/datasets/miracl/miracl-corpus" in message
