"""Tests for corpus-qualified graph analysis identities."""

from unittest.mock import AsyncMock

import pytest

from kb_arena.graph.analyzer import GraphAnalyzer


@pytest.mark.asyncio
async def test_analyzer_keeps_same_fqn_entities_separate_by_corpus():
    store = AsyncMock()
    store.execute_query.side_effect = [
        [
            {"entity_id": "alpha::shared", "fqn": "shared", "label": "Topic"},
            {"entity_id": "alpha::leaf", "fqn": "leaf", "label": "Component"},
            {"entity_id": "beta::shared", "fqn": "shared", "label": "Topic"},
            {"entity_id": "beta::leaf", "fqn": "leaf", "label": "Component"},
        ],
        [
            {"src": "alpha::shared", "dst": "alpha::leaf", "rel": "CONTAINS"},
            {"src": "beta::shared", "dst": "beta::leaf", "rel": "CONTAINS"},
        ],
    ]
    analyzer = GraphAnalyzer(store)

    graph = await analyzer._build_networkx_graph()

    assert set(graph.nodes) == {
        "alpha::shared",
        "alpha::leaf",
        "beta::shared",
        "beta::leaf",
    }
    assert graph.number_of_edges() == 2
    assert all("KBArenaEntity" in call.args[0] for call in store.execute_query.await_args_list)
