"""Graph IR metrics — knowledge_graph retrieval must map back to source sections.

Regression coverage for the v0.6.x defect where knowledge_graph scored 0.0 on
every IR metric because it emitted chunk_id="graph:{fqn}" (entity FQN) while
ground truth labels are section ids ("doc::section"). The fix carries each
entity's source doc + section through Neo4j so the existing section-level
expected_chunks.yaml matches.
"""

from __future__ import annotations

from kb_arena.benchmark.ir_metrics import compute_all
from kb_arena.models.graph import Entity
from kb_arena.strategies.knowledge_graph import _records_to_chunks


def test_records_with_source_section_map_to_section_chunk_id():
    records = [
        {
            "name": "AWS Lambda",
            "fqn": "aws.lambda",
            "type": "Service",
            "source_doc_id": "lambda-overview",
            "source_section_id": "aws-lambda",
            "score": 0.9,
        }
    ]
    chunks = _records_to_chunks(records, "knowledge_graph", top_k=5)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "graph:lambda-overview::aws-lambda"
    assert chunks[0].doc_id == "lambda-overview"


def test_records_without_source_fall_back_to_fqn():
    records = [{"name": "AWS Lambda", "fqn": "aws.lambda", "type": "Service"}]
    chunks = _records_to_chunks(records, "knowledge_graph", top_k=5)
    assert chunks[0].chunk_id == "graph:aws.lambda"


def test_graph_chunks_score_nonzero_recall_against_section_ground_truth():
    records = [
        {
            "fqn": "aws.lambda",
            "name": "AWS Lambda",
            "type": "Service",
            "source_doc_id": "lambda-overview",
            "source_section_id": "aws-lambda",
        },
        {
            "fqn": "aws.lambda.layers",
            "name": "Lambda Layers",
            "type": "Component",
            "source_doc_id": "lambda-overview",
            "source_section_id": "layers",
        },
    ]
    chunks = _records_to_chunks(records, "knowledge_graph", top_k=5)
    metrics = compute_all(
        retrieved=chunks,
        expected_ids={"lambda-overview::aws-lambda", "lambda-overview::layers"},
        k=5,
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg_at_k > 0.0
    assert metrics.hit_at_k == 1


def test_entity_model_carries_source_doc_id():
    e = Entity(id="x", name="X", fqn="a.b", type="Topic", source_doc_id="doc1")
    assert e.source_doc_id == "doc1"


def test_entity_returning_cypher_templates_expose_source_provenance():
    """Every template that returns entity nodes must RETURN source_doc_id and
    source_section_id, otherwise graph chunks can never match section ground
    truth no matter what extraction stores."""
    from kb_arena.strategies import knowledge_graph as kg

    for tmpl_name in (
        "MULTI_HOP_QUERY",
        "COMPARISON_QUERY",
        "DEPENDENCY_CHAIN",
        "FULLTEXT_SEARCH",
        "ENTITY_LOOKUP",
    ):
        tmpl = getattr(kg, tmpl_name)
        assert "source_doc_id" in tmpl, f"{tmpl_name} missing source_doc_id"
        assert "source_section_id" in tmpl, f"{tmpl_name} missing source_section_id"

    # Text-to-Cypher fallback must instruct the LLM to return provenance too.
    assert "source_doc_id" in kg.CYPHER_GEN_PROMPT_TEMPLATE
    assert "source_section_id" in kg.CYPHER_GEN_PROMPT_TEMPLATE
