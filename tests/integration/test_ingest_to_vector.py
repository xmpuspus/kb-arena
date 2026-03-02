"""Integration test: raw markdown files → ingest pipeline → naive vector index → query."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.ingest.parsers.markdown import MarkdownParser
from kb_arena.ingest.pipeline import run_ingest
from kb_arena.models.document import Document
from kb_arena.strategies.naive_vector import NaiveVectorStrategy

SAMPLE_DOC_A = """\
# Python json Module

The json module provides JSON encoding and decoding.

## json.loads

Deserialize a JSON string to a Python object.
Use json.loads to parse JSON data from an API response.

## json.dumps

Serialize a Python object to a JSON string.
"""

SAMPLE_DOC_B = """\
# Python os Module

The os module provides operating system interfaces.

## os.path.join

Join path components intelligently using os.path.join.
Returns a path string combining the given components.
"""

SAMPLE_DOC_C = """\
# Python pathlib Module

The pathlib module offers object-oriented filesystem paths.

## Path.read_text

Read the file contents as a string using Path.read_text.
Supports encoding parameter for non-UTF-8 files.
"""


@pytest.fixture
def md_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "json.md").write_text(SAMPLE_DOC_A, encoding="utf-8")
    (raw / "os.md").write_text(SAMPLE_DOC_B, encoding="utf-8")
    (raw / "pathlib.md").write_text(SAMPLE_DOC_C, encoding="utf-8")
    return tmp_path, raw


def test_ingest_writes_jsonl_with_all_docs(md_corpus):
    tmp_path, raw = md_corpus
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(str(raw), corpus="test-vector", format="markdown")
    finally:
        os.chdir(old_cwd)

    out = tmp_path / "datasets" / "test-vector" / "processed" / "documents.jsonl"
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 3


def test_ingest_each_line_is_valid_document(md_corpus):
    tmp_path, raw = md_corpus
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(str(raw), corpus="test-valid", format="markdown")
    finally:
        os.chdir(old_cwd)

    out = tmp_path / "datasets" / "test-valid" / "processed" / "documents.jsonl"
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        doc = Document.model_validate(data)
        assert doc.id
        assert doc.corpus == "test-valid"
        assert len(doc.sections) >= 1
        assert doc.raw_token_count > 0


def test_ingest_sections_have_content(md_corpus):
    tmp_path, raw = md_corpus
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(str(raw), corpus="test-sections", format="markdown")
    finally:
        os.chdir(old_cwd)

    out = tmp_path / "datasets" / "test-sections" / "processed" / "documents.jsonl"
    docs = [
        Document.model_validate(json.loads(l)) for l in out.read_text().splitlines() if l.strip()
    ]
    all_sections = [s for doc in docs for s in doc.sections]
    assert all(s.content for s in all_sections)
    assert all(s.id for s in all_sections)


def test_ingest_preserves_headings(md_corpus):
    tmp_path, raw = md_corpus
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(str(raw), corpus="test-headings", format="markdown")
    finally:
        os.chdir(old_cwd)

    out = tmp_path / "datasets" / "test-headings" / "processed" / "documents.jsonl"
    docs = [
        Document.model_validate(json.loads(l)) for l in out.read_text().splitlines() if l.strip()
    ]
    all_titles = {s.title for doc in docs for s in doc.sections}
    assert "json.loads" in all_titles
    assert "os.path.join" in all_titles
    assert "Path.read_text" in all_titles


@pytest.mark.asyncio
async def test_naive_vector_build_index_from_parsed_docs(md_corpus):
    tmp_path, raw = md_corpus
    parser = MarkdownParser()
    docs = []
    for md_file in raw.glob("*.md"):
        docs.extend(parser.parse(md_file, "test-e2e"))

    mock_chroma = MagicMock()
    collection = MagicMock()
    mock_chroma.get_or_create_collection.return_value = collection

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    await strategy.build_index(docs)

    assert collection.upsert.called
    # 3 docs with 3 sections each → multiple chunks
    total_upserted = sum(
        len(call.kwargs.get("ids") or call[1].get("ids") or [])
        for call in collection.upsert.call_args_list
    )
    assert total_upserted > 0


@pytest.mark.asyncio
async def test_naive_vector_query_references_source(md_corpus):
    tmp_path, raw = md_corpus
    parser = MarkdownParser()
    docs = []
    for md_file in raw.glob("*.md"):
        docs.extend(parser.parse(md_file, "test-e2e"))

    mock_chroma = MagicMock()
    collection = MagicMock()
    # Simulate ChromaDB returning the json doc's content
    collection.query.return_value = {
        "ids": [["json::json-loads::0"]],
        "documents": [["Use json.loads to parse JSON data from an API response."]],
        "metadatas": [[{"source_id": "json"}]],
        "distances": [[0.05]],
    }
    mock_chroma.get_or_create_collection.return_value = collection

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "json.loads parses JSON data from a string."

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    strategy._llm = mock_llm

    result = await strategy.query("How do I parse JSON data?")

    assert result.answer == "json.loads parses JSON data from a string."
    assert "json" in result.sources
    assert result.strategy == "naive_vector"
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_naive_vector_query_uses_context_from_retrieval(md_corpus):
    """Verify retrieved chunks are passed as context to the LLM."""
    tmp_path, raw = md_corpus

    mock_chroma = MagicMock()
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["pathlib::path-read-text::0"]],
        "documents": [["Read the file contents as a string using Path.read_text."]],
        "metadatas": [[{"source_id": "pathlib"}]],
        "distances": [[0.08]],
    }
    mock_chroma.get_or_create_collection.return_value = collection

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Path.read_text reads file contents."

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    strategy._llm = mock_llm

    await strategy.query("How do I read a file with pathlib?")

    generate_call = mock_llm.generate.call_args
    context_arg = (
        generate_call.kwargs.get("context")
        or generate_call[1].get("context")
        or generate_call[0][1]
    )
    assert "Path.read_text" in context_arg


@pytest.mark.asyncio
async def test_naive_vector_deduplicates_sources():
    """Multiple chunks from the same source_id should appear once in sources."""
    mock_chroma = MagicMock()
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["json::s1::0", "json::s1::1", "json::s2::0"]],
        "documents": [["chunk1", "chunk2", "chunk3"]],
        "metadatas": [
            [
                {"source_id": "json"},
                {"source_id": "json"},
                {"source_id": "json"},
            ]
        ],
        "distances": [[0.1, 0.2, 0.3]],
    }
    mock_chroma.get_or_create_collection.return_value = collection

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "answer"

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    strategy._llm = mock_llm

    result = await strategy.query("question")
    assert result.sources.count("json") == 1


@pytest.mark.asyncio
async def test_full_pipeline_e2e(md_corpus):
    """Full path: write files → ingest → parse JSONL → build index → query."""
    tmp_path, raw = md_corpus

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(str(raw), corpus="e2e-corpus", format="markdown")
    finally:
        os.chdir(old_cwd)

    jsonl_path = tmp_path / "datasets" / "e2e-corpus" / "processed" / "documents.jsonl"
    docs = [
        Document.model_validate(json.loads(line))
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(docs) == 3

    mock_chroma = MagicMock()
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["os::os-path-join::0"]],
        "documents": [["Join path components intelligently using os.path.join."]],
        "metadatas": [[{"source_id": "os"}]],
        "distances": [[0.03]],
    }
    mock_chroma.get_or_create_collection.return_value = collection

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Use os.path.join to combine path components."

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    await strategy.build_index(docs)
    strategy._llm = mock_llm

    result = await strategy.query("How do I join file paths?")

    assert result.answer == "Use os.path.join to combine path components."
    assert "os" in result.sources
