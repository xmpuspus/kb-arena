"""Shared test fixtures — mock Neo4j, mock ChromaDB, sample documents."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.models.document import CodeBlock, CrossRef, Document, Section, Table


@pytest.fixture
def sample_section():
    return Section(
        id="lambda-configuration",
        title="Lambda Function Configuration",
        content=(
            "You can configure your Lambda function's memory, timeout, and runtime"
            " settings using the AWS Management Console or AWS CLI."
        ),
        heading_path=["AWS Lambda", "Configuration"],
        tables=[],
        code_blocks=[
            CodeBlock(
                language="bash",
                code=(
                    "aws lambda update-function-configuration \\\n"
                    "  --function-name my-function \\\n"
                    "  --timeout 300 \\\n"
                    "  --memory-size 512"
                ),
                description="Configure Lambda function timeout and memory",
            )
        ],
        links=[CrossRef(target="lambda-permissions", label="Execution Role", ref_type="concept")],
        level=2,
    )


@pytest.fixture
def sample_document(sample_section):
    return Document(
        id="aws-compute-lambda",
        source="https://docs.aws.amazon.com/lambda/latest/dg/configuration-function-common.html",
        corpus="aws-compute",
        title="Configuring AWS Lambda Functions",
        sections=[
            Section(
                id="lambda-overview",
                title="Configuring AWS Lambda Functions",
                content="AWS Lambda lets you run code without provisioning or managing servers.",
                heading_path=["AWS Lambda"],
                level=1,
                children=["lambda-configuration", "lambda-permissions"],
            ),
            sample_section,
            Section(
                id="lambda-permissions",
                title="Lambda Execution Role",
                content=(
                    "A Lambda function's execution role grants it"
                    " permission to access AWS services and resources."
                ),
                heading_path=["AWS Lambda", "Execution Role"],
                tables=[
                    Table(
                        headers=["Setting", "Type", "Description"],
                        rows=[
                            ["Timeout", "int", "Maximum execution time in seconds (1-900)"],
                            ["MemorySize", "int", "Memory allocation in MB (128-10240)"],
                        ],
                    )
                ],
                level=2,
            ),
        ],
        metadata={"service": "lambda", "doc_type": "developer_guide"},
        raw_token_count=1500,
    )


@pytest.fixture
def sample_documents(sample_document):
    """Multiple documents for batch testing."""
    s3_doc = Document(
        id="aws-storage-s3",
        source="https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        corpus="aws-storage",
        title="Amazon S3 User Guide",
        sections=[
            Section(
                id="s3-overview",
                title="Amazon S3 User Guide",
                content=(
                    "Amazon Simple Storage Service (Amazon S3) is an object storage service"
                    " offering industry-leading scalability, data availability, and security."
                ),
                heading_path=["Amazon S3"],
                level=1,
            ),
            Section(
                id="s3-bucket-policies",
                title="Bucket Policies",
                content="You use bucket policies to grant permissions to your Amazon S3 resources.",
                heading_path=["Amazon S3", "Security", "Bucket Policies"],
                level=3,
            ),
        ],
        raw_token_count=5000,
    )
    return [sample_document, s3_doc]


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
        "metadatas": [[{"source": "lambda-dg.html"}, {"source": "lambda-dg.html"}]],
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
