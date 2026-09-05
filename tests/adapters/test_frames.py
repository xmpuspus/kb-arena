"""FRAMES: the parse, and the guards it must not skip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_arena.adapters import ADAPTERS, ChecksumMismatchError
from kb_arena.adapters.frames import _LINK_COLUMNS, FramesAdapter


def test_parse_question_keeps_only_the_filled_wikipedia_links():
    raw = {
        "Prompt": "Who succeeded the first president of Testland?",
        "Answer": "The second president of Testland",
        "wikipedia_link_1": "https://en.wikipedia.org/wiki/Testland",
        "wikipedia_link_2": "https://en.wikipedia.org/wiki/List_of_presidents_of_Testland",
        "wikipedia_link_3": None,
        "wikipedia_link_4": None,
        "wikipedia_link_5": None,
        "wikipedia_link_6": None,
        "wikipedia_link_7": None,
        "wikipedia_link_8": None,
        "wikipedia_link_9": None,
        "wikipedia_link_10": None,
        "wikipedia_link_11+": None,
        "reasoning_types": "Multiple constraints",
    }

    question = FramesAdapter.parse_question(raw)

    assert question["question"] == raw["Prompt"]
    assert question["evidence"] == [
        {"source_doc_id": "https://en.wikipedia.org/wiki/Testland", "source_section_id": "0"},
        {
            "source_doc_id": "https://en.wikipedia.org/wiki/List_of_presidents_of_Testland",
            "source_section_id": "0",
        },
    ]


def test_a_moving_revision_is_refused():
    template = ADAPTERS["frames"]().manifest_template()
    with pytest.raises(ValidationError, match="moving target"):
        type(template)(**{**template.model_dump(), "revision": "HEAD"})


def test_a_checksum_mismatch_stops_the_run(tmp_path):
    adapter = ADAPTERS["frames"]()
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"source_doc_id": "https://en.wikipedia.org/wiki/Testland"}\n')

    with pytest.raises(ChecksumMismatchError, match="cannot be cited"):
        adapter.verify(path, "c" * 64)


def test_download_only_and_says_the_terms():
    adapter = ADAPTERS["frames"]()
    template = adapter.manifest_template()

    assert adapter.download_only is True
    assert template.license == "Apache-2.0"
    assert template.redistributable is True
    assert "apache.org" in template.license_url


def test_refuses_to_fetch_and_says_what_to_do_instead():
    with pytest.raises(NotImplementedError) as refused:
        ADAPTERS["frames"]().build(Path("/tmp/frames"))

    message = str(refused.value)
    assert "download-only" in message
    assert "huggingface.co/datasets/google/frames-benchmark" in message


def test_an_overflow_cell_holding_several_links_becomes_several_evidence_rows():
    """The last link column is an overflow cell, and it holds comma-separated URLs.

    Reading it as one URL made three Wikipedia pages into one `source_doc_id`
    that resolves to nothing.
    """
    row = {
        "Prompt": "q",
        "Answer": "a",
        "reasoning_types": "numerical",
        "wikipedia_link_1": "https://en.wikipedia.org/wiki/A",
        _LINK_COLUMNS[-1]: ("https://en.wikipedia.org/wiki/B, https://en.wikipedia.org/wiki/C"),
    }

    evidence = FramesAdapter.parse_question(row)["evidence"]

    assert [e["source_doc_id"] for e in evidence] == [
        "https://en.wikipedia.org/wiki/A",
        "https://en.wikipedia.org/wiki/B",
        "https://en.wikipedia.org/wiki/C",
    ]
