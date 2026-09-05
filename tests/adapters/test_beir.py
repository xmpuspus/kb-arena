"""BEIR (scifact): the parse, and the guards it must not skip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_arena.adapters import ADAPTERS, ChecksumMismatchError
from kb_arena.adapters.beir import BeirScifactAdapter


def test_parse_document_keeps_the_corpus_id_as_the_doc_id():
    raw = {"_id": "4983", "title": "A sample abstract title", "text": "A sample abstract body."}

    section = BeirScifactAdapter.parse_document(raw)

    assert section["source_doc_id"] == "4983"
    assert section["source_section_id"] == "0"
    assert section["text"] == "A sample abstract body."


def test_parse_qrel_keeps_the_same_corpus_id_parse_document_emits():
    raw = {"query-id": "1", "corpus-id": "4983", "score": "1"}

    judgment = BeirScifactAdapter.parse_qrel(raw)

    assert judgment["query_id"] == "1"
    assert judgment["source_doc_id"] == "4983"
    assert judgment["relevance"] == 1


def test_a_moving_revision_is_refused():
    template = ADAPTERS["beir-scifact"]().manifest_template()
    with pytest.raises(ValidationError, match="moving target"):
        type(template)(**{**template.model_dump(), "revision": "main"})


def test_a_checksum_mismatch_stops_the_run(tmp_path):
    adapter = ADAPTERS["beir-scifact"]()
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"source_doc_id": "4983"}\n')

    with pytest.raises(ChecksumMismatchError, match="cannot be cited"):
        adapter.verify(path, "e" * 64)


def test_download_only_and_says_the_terms():
    """The corpus licence is scifact's own, not a blanket BEIR licence."""
    adapter = ADAPTERS["beir-scifact"]()
    template = adapter.manifest_template()

    assert adapter.download_only is True
    assert template.license == "ODC-BY-1.0"
    assert template.redistributable is True
    assert "CC-BY-4.0" in template.attribution, "the claims' own, different licence is named too"


def test_refuses_to_fetch_and_says_what_to_do_instead():
    with pytest.raises(NotImplementedError) as refused:
        ADAPTERS["beir-scifact"]().build(Path("/tmp/beir-scifact"))

    message = str(refused.value)
    assert "download-only" in message
    assert "github.com/allenai/scifact" in message
    assert "5f7d1de60b170fc8027bb7898e2efca1" in message
