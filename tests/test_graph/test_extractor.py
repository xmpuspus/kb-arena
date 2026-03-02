"""Tests for entity/relationship extraction with mocked LLM."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kb_arena.graph.extractor import _validate_result, extract_document
from kb_arena.models.graph import ExtractionResult

PYTHON_CORPUS = "python-stdlib"

_VALID_LLM_RESPONSE = {
    "entities": [
        {
            "id": "json.loads",
            "name": "json.loads",
            "fqn": "json.loads",
            "type": "Function",
            "description": "Deserialize a JSON string to a Python object.",
            "properties": {},
            "aliases": [],
        },
        {
            "id": "json.JSONDecodeError",
            "name": "json.JSONDecodeError",
            "fqn": "json.JSONDecodeError",
            "type": "Exception",
            "description": "Raised when JSON parsing fails.",
            "properties": {},
            "aliases": [],
        },
    ],
    "relationships": [
        {
            "source_fqn": "json.loads",
            "target_fqn": "json.JSONDecodeError",
            "type": "RAISES",
            "properties": {},
        }
    ],
}


def test_validate_result_accepts_valid_types():
    result = _validate_result(_VALID_LLM_RESPONSE, PYTHON_CORPUS, "json-loads")
    assert len(result.entities) == 2
    assert len(result.relationships) == 1
    assert result.entities[0].fqn == "json.loads"
    assert result.relationships[0].type == "RAISES"


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
    result = _validate_result(bad, PYTHON_CORPUS, "sec-x")
    assert result.entities == []


def test_validate_result_rejects_unknown_rel_type():
    bad = {
        "entities": list(_VALID_LLM_RESPONSE["entities"]),
        "relationships": [
            {
                "source_fqn": "json.loads",
                "target_fqn": "json.JSONDecodeError",
                "type": "INVENTED_REL",
                "properties": {},
            }
        ],
    }
    result = _validate_result(bad, PYTHON_CORPUS, "sec-x")
    assert result.relationships == []


def test_validate_result_drops_dangling_relationships():
    """Relationships referencing fqns not in the entity list are dropped."""
    payload = {
        "entities": [_VALID_LLM_RESPONSE["entities"][0]],  # only json.loads
        "relationships": list(_VALID_LLM_RESPONSE["relationships"]),  # refs json.JSONDecodeError
    }
    result = _validate_result(payload, PYTHON_CORPUS, "s1")
    assert result.relationships == []


@pytest.mark.asyncio
async def test_extract_document_calls_llm_per_section(sample_document):
    mock_llm = AsyncMock()
    mock_llm.extract.return_value = json.dumps(_VALID_LLM_RESPONSE)

    from kb_arena.graph.extractor import _build_system_prompt

    result = await extract_document(sample_document, mock_llm, _build_system_prompt(PYTHON_CORPUS))

    # Called once per section in sample_document
    assert mock_llm.extract.call_count == len(sample_document.sections)
    assert isinstance(result, ExtractionResult)


@pytest.mark.asyncio
async def test_extract_document_handles_bad_json(sample_document):
    mock_llm = AsyncMock()
    mock_llm.extract.return_value = "not json at all"

    from kb_arena.graph.extractor import _build_system_prompt

    result = await extract_document(sample_document, mock_llm, _build_system_prompt(PYTHON_CORPUS))
    # Should not crash — returns empty result
    assert isinstance(result, ExtractionResult)
