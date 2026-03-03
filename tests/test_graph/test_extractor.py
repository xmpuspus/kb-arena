"""Tests for entity/relationship extraction with mocked LLM."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kb_arena.graph.extractor import _validate_result, extract_document
from kb_arena.models.graph import ExtractionResult

AWS_CORPUS = "aws-compute"

_VALID_LLM_RESPONSE = {
    "entities": [
        {
            "id": "aws.lambda.invoke",
            "name": "InvokeFunction",
            "fqn": "aws.lambda.invoke",
            "type": "Process",
            "description": "Invokes an AWS Lambda function.",
            "properties": {},
            "aliases": [],
        },
        {
            "id": "aws.iam.execution-role",
            "name": "ExecutionRole",
            "fqn": "aws.iam.execution-role",
            "type": "Constraint",
            "description": "IAM role that Lambda assumes for execution.",
            "properties": {},
            "aliases": [],
        },
    ],
    "relationships": [
        {
            "source_fqn": "aws.lambda.invoke",
            "target_fqn": "aws.iam.execution-role",
            "type": "DEPENDS_ON",
            "properties": {},
        }
    ],
}


def test_validate_result_accepts_valid_types():
    result = _validate_result(_VALID_LLM_RESPONSE, AWS_CORPUS, "lambda-invoke")
    assert len(result.entities) == 2
    assert len(result.relationships) == 1
    assert result.entities[0].fqn == "aws.lambda.invoke"
    assert result.relationships[0].type == "DEPENDS_ON"


def test_validate_result_rejects_unknown_node_type():
    bad = {
        "entities": [
            {
                "id": "x",
                "name": "X",
                "fqn": "x",
                "type": "UnknownNode",
                "description": "",
                "properties": {},
                "aliases": [],
            }
        ],
        "relationships": [],
    }
    result = _validate_result(bad, AWS_CORPUS, "s1")
    assert result.entities == []


def test_validate_result_rejects_unknown_rel_type():
    bad = {
        "entities": list(_VALID_LLM_RESPONSE["entities"]),
        "relationships": [
            {
                "source_fqn": "aws.lambda.invoke",
                "target_fqn": "aws.iam.execution-role",
                "type": "INVENTED_REL",
                "properties": {},
            }
        ],
    }
    result = _validate_result(bad, AWS_CORPUS, "s1")
    assert result.relationships == []


def test_validate_result_drops_dangling_relationships():
    """Relationships referencing fqns not in the entity list are dropped."""
    payload = {
        "entities": [_VALID_LLM_RESPONSE["entities"][0]],  # only aws.lambda.invoke
        "relationships": list(_VALID_LLM_RESPONSE["relationships"]),  # refs aws.iam.execution-role
    }
    result = _validate_result(payload, AWS_CORPUS, "s1")
    assert result.relationships == []


@pytest.mark.asyncio
async def test_extract_document_calls_llm_per_section(sample_document):
    mock_llm = AsyncMock()
    mock_llm.extract.return_value = json.dumps(_VALID_LLM_RESPONSE)

    from kb_arena.graph.extractor import _build_system_prompt

    result = await extract_document(sample_document, mock_llm, _build_system_prompt(AWS_CORPUS))

    # Called once per section in sample_document
    assert mock_llm.extract.call_count == len(sample_document.sections)
    assert isinstance(result, ExtractionResult)


@pytest.mark.asyncio
async def test_extract_document_handles_bad_json(sample_document):
    mock_llm = AsyncMock()
    mock_llm.extract.return_value = "not json at all"

    from kb_arena.graph.extractor import _build_system_prompt

    result = await extract_document(sample_document, mock_llm, _build_system_prompt(AWS_CORPUS))
    # Should not crash — returns empty result
    assert isinstance(result, ExtractionResult)
