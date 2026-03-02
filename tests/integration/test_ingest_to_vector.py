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
# AWS Lambda

AWS Lambda lets you run code without provisioning or managing servers.

## Lambda Configuration

Configure your Lambda function's memory, timeout, and runtime settings.
Use the AWS Management Console or AWS CLI to update function configuration.

## Lambda Execution Role

A Lambda function's execution role grants it permission to access AWS services.
"""

SAMPLE_DOC_B = """\
# Amazon S3

Amazon S3 is an object storage service offering industry-leading scalability.

## Bucket Policies

Use bucket policies to grant permissions to your Amazon S3 resources.
Bucket policies are resource-based policies attached to the bucket itself.
"""

SAMPLE_DOC_C = """\
# Amazon VPC

Amazon Virtual Private Cloud lets you provision a logically isolated network.

## Security Groups

Security groups act as a virtual firewall for your EC2 instances.
They control inbound and outbound traffic at the instance level.
"""


@pytest.fixture
def md_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "lambda.md").write_text(SAMPLE_DOC_A, encoding="utf-8")
    (raw / "s3.md").write_text(SAMPLE_DOC_B, encoding="utf-8")
    (raw / "vpc.md").write_text(SAMPLE_DOC_C, encoding="utf-8")
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
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
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
        Document.model_validate(json.loads(ln)) for ln in out.read_text().splitlines() if ln.strip()
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
        Document.model_validate(json.loads(ln)) for ln in out.read_text().splitlines() if ln.strip()
    ]
    all_titles = {s.title for doc in docs for s in doc.sections}
    assert "Lambda Configuration" in all_titles
    assert "Bucket Policies" in all_titles
    assert "Security Groups" in all_titles


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
    # Simulate ChromaDB returning the Lambda doc's content
    collection.query.return_value = {
        "ids": [["lambda::lambda-configuration::0"]],
        "documents": [["Configure your Lambda function's memory, timeout, and runtime settings."]],
        "metadatas": [[{"source_id": "lambda"}]],
        "distances": [[0.05]],
    }
    mock_chroma.get_or_create_collection.return_value = collection

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Lambda timeout is configured via the AWS CLI or Console."

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    strategy._llm = mock_llm

    result = await strategy.query("How do I configure Lambda timeout?")

    assert result.answer == "Lambda timeout is configured via the AWS CLI or Console."
    assert "lambda" in result.sources
    assert result.strategy == "naive_vector"
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_naive_vector_query_uses_context_from_retrieval(md_corpus):
    """Verify retrieved chunks are passed as context to the LLM."""
    tmp_path, raw = md_corpus

    mock_chroma = MagicMock()
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["vpc::security-groups::0"]],
        "documents": [["Security groups act as a virtual firewall for your EC2 instances."]],
        "metadatas": [[{"source_id": "vpc"}]],
        "distances": [[0.08]],
    }
    mock_chroma.get_or_create_collection.return_value = collection

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Security groups control inbound and outbound traffic."

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    strategy._llm = mock_llm

    await strategy.query("How do security groups work in VPC?")

    generate_call = mock_llm.generate.call_args
    context_arg = (
        generate_call.kwargs.get("context")
        or generate_call[1].get("context")
        or generate_call[0][1]
    )
    assert "Security groups" in context_arg


@pytest.mark.asyncio
async def test_naive_vector_deduplicates_sources():
    """Multiple chunks from the same source_id should appear once in sources."""
    mock_chroma = MagicMock()
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["lambda::s1::0", "lambda::s1::1", "lambda::s2::0"]],
        "documents": [["chunk1", "chunk2", "chunk3"]],
        "metadatas": [
            [
                {"source_id": "lambda"},
                {"source_id": "lambda"},
                {"source_id": "lambda"},
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
    assert result.sources.count("lambda") == 1


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
        "ids": [["s3::bucket-policies::0"]],
        "documents": [["Use bucket policies to grant permissions to your Amazon S3 resources."]],
        "metadatas": [[{"source_id": "s3"}]],
        "distances": [[0.03]],
    }
    mock_chroma.get_or_create_collection.return_value = collection

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Use bucket policies to control access to S3 resources."

    strategy = NaiveVectorStrategy(chroma_client=mock_chroma)
    await strategy.build_index(docs)
    strategy._llm = mock_llm

    result = await strategy.query("How do I control access to S3?")

    assert result.answer == "Use bucket policies to control access to S3 resources."
    assert "s3" in result.sources
