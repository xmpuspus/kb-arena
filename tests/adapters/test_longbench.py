"""LongBench v2: the parse, and the guards it must not skip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_arena.adapters import ADAPTERS, ChecksumMismatchError
from kb_arena.adapters.longbench import LongBenchV2Adapter


def test_parse_document_cuts_the_context_into_sequential_sections():
    raw = {"_id": "sample-001", "context": "x" * 9000}

    sections = LongBenchV2Adapter.parse_document(raw)

    assert [s["source_doc_id"] for s in sections] == ["sample-001"] * 3
    assert [s["source_section_id"] for s in sections] == ["0", "1", "2"]
    assert sum(len(s["text"]) for s in sections) == 9000


def test_parse_question_keeps_the_same_doc_id_the_sections_carry():
    raw = {
        "_id": "sample-001",
        "question": "What happened in chapter one?",
        "choice_A": "Nothing",
        "choice_B": "Something",
        "choice_C": "Everything",
        "choice_D": "Unclear",
        "answer": "B",
        "difficulty": "hard",
    }

    question = LongBenchV2Adapter.parse_question(raw)

    assert question["source_doc_id"] == "sample-001"
    assert question["choices"]["B"] == "Something"
    assert question["answer"] == "B"


def test_a_moving_revision_is_refused():
    template = ADAPTERS["longbench-v2"]().manifest_template()
    with pytest.raises(ValidationError, match="moving target"):
        type(template)(**{**template.model_dump(), "revision": "latest"})


def test_a_checksum_mismatch_stops_the_run(tmp_path):
    adapter = ADAPTERS["longbench-v2"]()
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"source_doc_id": "sample-001"}\n')

    with pytest.raises(ChecksumMismatchError, match="cannot be cited"):
        adapter.verify(path, "0" * 63 + "1")


def test_download_only_and_says_the_terms():
    adapter = ADAPTERS["longbench-v2"]()
    template = adapter.manifest_template()

    assert adapter.download_only is True
    assert template.license == "Apache-2.0"
    assert template.redistributable is True
    assert "apache.org" in template.license_url


def test_refuses_to_fetch_and_says_what_to_do_instead():
    with pytest.raises(NotImplementedError) as refused:
        ADAPTERS["longbench-v2"]().build(Path("/tmp/longbench-v2"))

    message = str(refused.value)
    assert "download-only" in message
    assert "huggingface.co/datasets/zai-org/LongBench-v2" in message
