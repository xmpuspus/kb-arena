"""The analyzer loads a bounded slice and samples centrality above the budget, and says so."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kb_arena.graph.analyzer import GraphAnalyzer
from kb_arena.settings import settings


def _chain(n: int):
    nodes = [{"entity_id": f"c::n{i}", "fqn": f"n{i}", "label": "Topic"} for i in range(n)]
    edges = [{"src": f"c::n{i}", "dst": f"c::n{i + 1}", "rel": "DEPENDS_ON"} for i in range(n - 1)]
    return nodes, edges


def _store(nodes, edges):
    store = AsyncMock()

    async def execute_query(cypher, params=None):
        limit = (params or {}).get("limit")
        rows = nodes if "MATCH (n:KBArenaEntity)" in cypher else edges
        return rows[:limit] if limit else rows

    store.execute_query.side_effect = execute_query
    return store


@pytest.mark.asyncio
async def test_a_graph_inside_the_budget_loads_whole_and_uses_exact_centrality(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 100)
    monkeypatch.setattr(settings, "graph_centrality_exact_max_nodes", 100)
    nodes, edges = _chain(10)
    analyzer = GraphAnalyzer(_store(nodes, edges))

    centrality = await analyzer.calculate_centrality()

    assert len(centrality) == 10
    assert analyzer.last_load["truncated"] is False
    assert analyzer.last_load["nodes_loaded"] == 10
    assert analyzer.last_centrality == {"method": "exact", "nodes": 10, "samples": None}
    assert centrality["c::n5"] > centrality["c::n0"]  # the middle of a chain is the hub


@pytest.mark.asyncio
async def test_a_graph_over_the_node_budget_loads_a_slice_and_drops_edges_outside_it(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 3)
    monkeypatch.setattr(settings, "graph_edge_budget", 100)
    nodes, edges = _chain(6)
    analyzer = GraphAnalyzer(_store(nodes, edges))

    graph = await analyzer._build_networkx_graph()

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert analyzer.last_load["truncated"] is True
    assert analyzer.last_load["edges_dropped_outside_slice"] == 3
    # the store was asked for one row past the budget, never the whole graph
    limits = [call.args[1]["limit"] for call in analyzer._store.execute_query.await_args_list]
    assert limits == [4, 101]


@pytest.mark.asyncio
async def test_an_edge_budget_alone_marks_the_load_truncated(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 100)
    monkeypatch.setattr(settings, "graph_edge_budget", 2)
    nodes, edges = _chain(6)
    analyzer = GraphAnalyzer(_store(nodes, edges))

    graph = await analyzer._build_networkx_graph()

    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 2
    assert analyzer.last_load["truncated"] is True


@pytest.mark.asyncio
async def test_a_large_graph_samples_centrality_with_a_seed(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 5000)
    monkeypatch.setattr(settings, "graph_centrality_exact_max_nodes", 50)
    monkeypatch.setattr(settings, "graph_centrality_samples", 20)
    nodes, edges = _chain(300)
    analyzer = GraphAnalyzer(_store(nodes, edges))

    first = await analyzer.calculate_centrality()
    analyzer._cache.clear()
    second = await analyzer.calculate_centrality()

    assert len(first) == 300
    assert analyzer.last_centrality == {"method": "approximate", "nodes": 300, "samples": 20}
    assert first == second, "a seeded sample gives the same numbers twice"
    assert max(first, key=first.get) not in ("c::n0", "c::n299")


@pytest.mark.asyncio
async def test_graph_stats_reports_the_method_and_the_load(monkeypatch):
    from kb_arena.chatbot import api

    class _FakeAnalyzer:
        def __init__(self, store):
            self.last_load = {"truncated": True, "nodes_loaded": 3}
            self.last_centrality = {"method": "approximate", "nodes": 3, "samples": 2}

        async def calculate_centrality(self):
            return {"c::a": 0.5, "c::b": 0.2, "c::c": 0.0}

        async def analyze_communities(self):
            return [{"c::a", "c::b"}, {"c::c"}]

    monkeypatch.setattr("kb_arena.graph.analyzer.GraphAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr("kb_arena.graph.neo4j_store.Neo4jStore", lambda driver: object())
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(neo4j=object())))

    stats = await api.graph_stats(request)

    assert stats["node_count"] == 3
    assert stats["community_count"] == 2
    assert stats["centrality"]["method"] == "approximate"
    assert stats["load"]["truncated"] is True
