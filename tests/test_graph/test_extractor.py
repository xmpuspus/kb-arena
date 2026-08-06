"""Tests for entity/relationship extraction with mocked LLM."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from kb_arena.exceptions import GraphError
from kb_arena.graph.extractor import (
    _load_schema,
    _validate_result,
    extract_document,
    run_extraction,
)
from kb_arena.models.document import Document, Section
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


def test_validate_result_keeps_cross_section_relationships():
    """As of v0.6.0, cross-section relationships are NOT dropped at validation time.

    Per-section validation no longer requires both endpoints to live in the same
    extraction batch. The global FQN union check happens later in run_extraction
    (after every section has been extracted), so multi-hop graph queries
    structurally work.
    """
    payload = {
        "entities": [_VALID_LLM_RESPONSE["entities"][0]],  # only aws.lambda.invoke
        "relationships": list(_VALID_LLM_RESPONSE["relationships"]),  # refs aws.iam.execution-role
    }
    result = _validate_result(payload, AWS_CORPUS, "s1")
    # Edge with both endpoints set is kept; the global pass deduplicates later.
    assert len(result.relationships) == len(_VALID_LLM_RESPONSE["relationships"])


@pytest.mark.asyncio
async def test_extract_document_calls_llm_per_section(sample_document):
    mock_llm = AsyncMock()
    from kb_arena.llm.client import LLMResponse

    mock_llm.extract.return_value = LLMResponse(text=json.dumps(_VALID_LLM_RESPONSE))

    from kb_arena.graph.extractor import _build_system_prompt

    result = await extract_document(sample_document, mock_llm, _build_system_prompt(AWS_CORPUS))

    # Called once per section in sample_document
    assert mock_llm.extract.call_count == len(sample_document.sections)
    assert isinstance(result, ExtractionResult)


@pytest.mark.asyncio
async def test_extract_document_schedules_at_most_five_sections(sample_document, monkeypatch):
    from kb_arena.graph import extractor

    active = 0
    maximum_active = 0

    async def fake_extract(section, corpus, llm, system_prompt):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.005)
        active -= 1
        return ExtractionResult(section_id=section.id)

    base_section = sample_document.sections[0]
    document = sample_document.model_copy(
        update={
            "sections": [
                base_section.model_copy(update={"id": f"section-{index}"}) for index in range(12)
            ]
        }
    )
    monkeypatch.setattr(extractor, "_extract_section", fake_extract)

    await extractor.extract_document(document, AsyncMock(), "prompt")

    assert maximum_active == 5


@pytest.mark.asyncio
async def test_extract_document_stamps_source_doc_id(sample_document):
    """Entities must carry their source document id so graph retrieval can map
    back to section-level ground truth (graph IR fix)."""
    mock_llm = AsyncMock()
    from kb_arena.llm.client import LLMResponse

    mock_llm.extract.return_value = LLMResponse(text=json.dumps(_VALID_LLM_RESPONSE))

    from kb_arena.graph.extractor import _build_system_prompt

    result = await extract_document(sample_document, mock_llm, _build_system_prompt(AWS_CORPUS))

    assert result.entities
    assert all(e.source_doc_id == sample_document.id for e in result.entities)
    # source_section_id is still set per-section
    assert all(e.source_section_id for e in result.entities)


@pytest.mark.asyncio
async def test_extract_document_rejects_bad_json(sample_document):
    mock_llm = AsyncMock()
    from kb_arena.llm.client import LLMResponse

    mock_llm.extract.return_value = LLMResponse(text="not json at all")

    from kb_arena.graph.extractor import _build_system_prompt

    with pytest.raises(GraphError, match="Invalid extraction response"):
        await extract_document(sample_document, mock_llm, _build_system_prompt(AWS_CORPUS))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"entities": null, "relationships": []}',
        '{"entities": [1], "relationships": []}',
    ],
)
async def test_extract_document_rejects_invalid_json_shapes(sample_document, payload):
    from kb_arena.graph.extractor import _build_system_prompt
    from kb_arena.llm.client import LLMResponse

    mock_llm = AsyncMock()
    mock_llm.extract.return_value = LLMResponse(text=payload)

    with pytest.raises(GraphError, match="Invalid extraction response"):
        await extract_document(sample_document, mock_llm, _build_system_prompt(AWS_CORPUS))


@pytest.mark.asyncio
async def test_extract_document_cancels_siblings_after_failure():
    from kb_arena.graph.extractor import _build_system_prompt
    from kb_arena.llm.client import LLMResponse

    blocked_started = asyncio.Event()
    blocked_cancelled = asyncio.Event()
    never_finish = asyncio.Event()

    class FakeLLM:
        async def extract(self, text, system_prompt):
            if "Blocked" in text:
                blocked_started.set()
                try:
                    await never_finish.wait()
                except asyncio.CancelledError:
                    blocked_cancelled.set()
                    raise
            await blocked_started.wait()
            return LLMResponse(text="not json")

    document = Document(
        id="doc",
        source="test",
        corpus=AWS_CORPUS,
        title="Cancellation",
        sections=[
            Section(id="blocked", title="Blocked", content="Blocked", level=2),
            Section(id="failed", title="Failed", content="Failed", level=2),
        ],
    )

    with pytest.raises(GraphError, match="Invalid extraction response"):
        await extract_document(document, FakeLLM(), _build_system_prompt(AWS_CORPUS))

    assert blocked_cancelled.is_set()


@pytest.mark.asyncio
async def test_run_extraction_closes_store_and_does_not_complete_after_bad_json(
    tmp_path, monkeypatch, sample_document
):
    from kb_arena.llm.client import LLMResponse
    from kb_arena.settings import settings

    processed = tmp_path / AWS_CORPUS / "processed"
    processed.mkdir(parents=True)
    (processed / "documents.jsonl").write_text(sample_document.model_dump_json() + "\n")
    llm = AsyncMock()
    llm.extract.return_value = LLMResponse(text="not json")
    store = AsyncMock()
    events: list[dict] = []

    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    monkeypatch.setattr("kb_arena.graph.extractor.LLMClient", lambda: llm)
    monkeypatch.setattr(
        "kb_arena.graph.extractor.Neo4jStore.connect", AsyncMock(return_value=store)
    )
    monkeypatch.setattr("kb_arena.graph.extractor._load_schema", AsyncMock())

    async def capture(event: dict) -> None:
        events.append(event)

    with pytest.raises(GraphError, match="Invalid extraction response"):
        await run_extraction(AWS_CORPUS, event_callback=capture)

    store.close.assert_awaited_once()
    store.load_nodes.assert_not_awaited()
    store.load_edges.assert_not_awaited()
    assert all(event["type"] != "complete" for event in events)


@pytest.mark.asyncio
async def test_load_schema_uses_packaged_resource_from_any_working_directory(tmp_path, monkeypatch):
    loaded: list[str] = []
    store = AsyncMock()
    store.legacy_constraint_names.return_value = []

    async def capture_schema(path):
        loaded.append(path.read_text(encoding="utf-8"))

    store.load_schema.side_effect = capture_schema
    monkeypatch.chdir(tmp_path)

    await _load_schema(store, "custom")

    store.load_schema.assert_awaited_once()
    assert "CREATE CONSTRAINT kb_arena_entity_id" in loaded[0]
    assert "DROP CONSTRAINT" not in loaded[0]


@pytest.mark.asyncio
async def test_load_schema_rejects_implicit_legacy_constraint_migration():
    store = AsyncMock()
    store.legacy_constraint_names.return_value = ["topic_fqn"]

    with pytest.raises(GraphError, match="--database <name> --confirm-dedicated-database"):
        await _load_schema(store, "custom")

    store.load_schema.assert_not_awaited()
