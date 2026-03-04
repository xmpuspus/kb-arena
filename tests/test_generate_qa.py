"""Tests for Q&A pair generation (shared module + CLI runner)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.generate.qna import (
    generate_pairs_for_documents,
    generate_pairs_for_section,
    parse_qna_json,
)
from kb_arena.models.document import Document, Section

# --- parse_qna_json ---


def test_parse_qna_json_valid():
    raw = '[{"question": "What is X?", "answer": "X is Y."}]'
    pairs = parse_qna_json(raw)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "What is X?"


def test_parse_qna_json_with_fences():
    raw = '```json\n[{"question": "Q", "answer": "A"}]\n```'
    pairs = parse_qna_json(raw)
    assert len(pairs) == 1
    assert pairs[0]["answer"] == "A"


def test_parse_qna_json_empty():
    assert parse_qna_json("not json") == []
    assert parse_qna_json("") == []
    assert parse_qna_json("{}") == []  # dict, not list


def test_parse_qna_json_multiple_pairs():
    raw = json.dumps(
        [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
    )
    pairs = parse_qna_json(raw)
    assert len(pairs) == 3


# --- generate_pairs_for_section ---


@pytest.fixture
def mock_section():
    return Section(
        id="sec-1",
        title="VPC Configuration",
        content="Configure VPC peering to connect Lambda to RDS.",
        heading_path=["Networking", "VPC Configuration"],
        code_blocks=[],
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.extract = AsyncMock(
        return_value=MagicMock(
            text=json.dumps(
                [
                    {
                        "question": "How do I configure VPC peering?",
                        "answer": "Use the VPC console.",
                    },
                    {"question": "What is VPC?", "answer": "Virtual Private Cloud."},
                ]
            )
        )
    )
    return llm


@pytest.mark.asyncio
async def test_generate_pairs_for_section(mock_section, mock_llm):
    pairs = await generate_pairs_for_section(mock_section, "doc-1", mock_llm)
    assert len(pairs) == 2
    assert all(p["source_id"] == "doc-1" for p in pairs)
    assert all(p["section_id"] == "sec-1" for p in pairs)
    assert pairs[0]["question"] == "How do I configure VPC peering?"


@pytest.mark.asyncio
async def test_generate_pairs_attaches_heading_path(mock_section, mock_llm):
    pairs = await generate_pairs_for_section(mock_section, "doc-1", mock_llm)
    assert all(p["section_ref"] == "Networking > VPC Configuration" for p in pairs)


# --- generate_pairs_for_documents ---


@pytest.fixture
def mock_documents():
    return [
        Document(
            id="doc-1",
            title="Networking Guide",
            source="networking.md",
            corpus="test",
            sections=[
                Section(id="s1", title="VPC", content="VPC content here."),
                Section(id="s2", title="Subnets", content="Subnet content here."),
                Section(id="s3", title="Empty", content="   "),  # should be skipped
            ],
        ),
        Document(
            id="doc-2",
            title="Compute Guide",
            source="compute.md",
            corpus="test",
            sections=[
                Section(id="s4", title="EC2", content="EC2 instances run workloads."),
            ],
        ),
    ]


@pytest.mark.asyncio
async def test_generate_pairs_for_documents(mock_documents, mock_llm):
    pairs = await generate_pairs_for_documents(mock_documents, mock_llm)
    # 3 non-empty sections * 2 pairs each = 6
    assert len(pairs) == 6
    assert mock_llm.extract.call_count == 3  # skipped empty section


@pytest.mark.asyncio
async def test_generate_pairs_skips_empty_sections(mock_documents, mock_llm):
    pairs = await generate_pairs_for_documents(mock_documents, mock_llm)
    section_ids = {p["section_id"] for p in pairs}
    assert "s3" not in section_ids  # empty section skipped


@pytest.mark.asyncio
async def test_generate_pairs_handles_llm_error(mock_documents):
    llm = MagicMock()
    llm.extract = AsyncMock(side_effect=Exception("API error"))
    pairs = await generate_pairs_for_documents(mock_documents, llm)
    assert pairs == []  # all sections failed gracefully


@pytest.mark.asyncio
async def test_generate_pairs_on_progress_callback(mock_documents, mock_llm):
    progress_calls = []

    def on_progress(doc_id, count):
        progress_calls.append((doc_id, count))

    await generate_pairs_for_documents(mock_documents, mock_llm, on_progress=on_progress)
    assert len(progress_calls) == 2
    assert progress_calls[0] == ("doc-1", 4)  # 2 sections * 2 pairs
    assert progress_calls[1] == ("doc-2", 2)  # 1 section * 2 pairs


# --- Output format ---


def test_output_jsonl_format(mock_section, mock_llm, tmp_path):
    """Verify JSONL output is valid line-delimited JSON."""
    import asyncio

    pairs = asyncio.get_event_loop().run_until_complete(
        generate_pairs_for_section(mock_section, "doc-1", mock_llm)
    )

    out = tmp_path / "qa.jsonl"
    with open(out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "question" in parsed
        assert "answer" in parsed
        assert "source_id" in parsed
