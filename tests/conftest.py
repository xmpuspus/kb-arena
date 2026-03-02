"""Shared test fixtures — mock Neo4j, mock ChromaDB, sample documents."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.models.document import CodeBlock, CrossRef, Document, Section, Table


@pytest.fixture
def sample_section():
    return Section(
        id="json-loads",
        title="json.loads",
        content="Deserialize s (a str, bytes or bytearray instance containing a JSON document) to a Python object.",
        heading_path=["json", "json.loads"],
        tables=[],
        code_blocks=[
            CodeBlock(
                language="python",
                code=">>> import json\n>>> json.loads('{\"key\": \"value\"}')\n{'key': 'value'}",
                description="Basic usage of json.loads",
            )
        ],
        links=[CrossRef(target="json.JSONDecodeError", label="JSONDecodeError", ref_type="class")],
        level=2,
    )


@pytest.fixture
def sample_document(sample_section):
    return Document(
        id="python-stdlib-json",
        source="https://docs.python.org/3/library/json.html",
        corpus="python-stdlib",
        title="json — JSON encoder and decoder",
        sections=[
            Section(
                id="json-module",
                title="json — JSON encoder and decoder",
                content="JSON (JavaScript Object Notation) is a lightweight data interchange format.",
                heading_path=["json"],
                level=1,
                children=["json-loads", "json-dumps"],
            ),
            sample_section,
            Section(
                id="json-dumps",
                title="json.dumps",
                content="Serialize obj to a JSON formatted str.",
                heading_path=["json", "json.dumps"],
                tables=[
                    Table(
                        headers=["Parameter", "Type", "Description"],
                        rows=[
                            ["obj", "Any", "Object to serialize"],
                            ["indent", "int | None", "Number of spaces for indentation"],
                        ],
                    )
                ],
                level=2,
            ),
        ],
        metadata={"version": "3.12", "module_type": "standard_library"},
        raw_token_count=1500,
    )


@pytest.fixture
def sample_documents(sample_document):
    """Multiple documents for batch testing."""
    os_doc = Document(
        id="python-stdlib-os",
        source="https://docs.python.org/3/library/os.html",
        corpus="python-stdlib",
        title="os — Miscellaneous operating system interfaces",
        sections=[
            Section(
                id="os-module",
                title="os — Miscellaneous operating system interfaces",
                content="This module provides a portable way of using operating system dependent functionality.",
                heading_path=["os"],
                level=1,
            ),
            Section(
                id="os-path-join",
                title="os.path.join",
                content="Join one or more path components intelligently.",
                heading_path=["os", "os.path", "os.path.join"],
                level=3,
            ),
        ],
        raw_token_count=5000,
    )
    return [sample_document, os_doc]


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j async driver."""
    driver = MagicMock()
    session = AsyncMock()
    result = AsyncMock()
    summary = MagicMock()
    summary.counters.nodes_created = 5
    summary.counters.relationships_created = 3
    result.consume.return_value = summary
    result.data.return_value = []
    session.run.return_value = result
    # session() returns an async context manager
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx
    return driver


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    client = MagicMock()
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["doc1-s1", "doc1-s2"]],
        "documents": [["Content chunk 1", "Content chunk 2"]],
        "metadatas": [[{"source": "json.html"}, {"source": "json.html"}]],
        "distances": [[0.1, 0.3]],
    }
    client.get_or_create_collection.return_value = collection
    return client


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without API calls."""
    client = AsyncMock()
    client.classify.return_value = "factoid"
    client.generate.return_value = "This is a generated answer."
    client.extract.return_value = '{"entities": [], "relationships": []}'
    client.judge.return_value = '{"accuracy": 0.9, "completeness": 0.8, "faithfulness": 1.0}'
    return client
