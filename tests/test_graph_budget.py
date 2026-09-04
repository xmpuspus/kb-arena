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
        params = params or {}
        prefix = params.get("prefix", "")
        if "count(n)" in cypher:
            return [{"n": sum(1 for n in nodes if n["entity_id"].startswith(prefix))}]
        limit = params.get("limit")
        if "MATCH (n:KBArenaEntity)" in cypher:
            rows = [n for n in nodes if n["entity_id"].startswith(prefix)]
        else:
            ids = set(params.get("ids") or [])
            rows = [e for e in edges if e["src"] in ids and e["dst"] in ids]
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
    # the store only returns edges among the loaded nodes, so none are dropped here
    assert analyzer.last_load["edges_dropped_outside_slice"] == 0
    edge_call = next(c for c in analyzer._store.execute_query.await_args_list if "ids" in c.args[1])
    assert sorted(edge_call.args[1]["ids"]) == ["c::n0", "c::n1", "c::n2"]
    # the store was asked for one row past the budget, never the whole graph
    calls = analyzer._store.execute_query.await_args_list
    limits = [c.args[1]["limit"] for c in calls if "limit" in c.args[1]]
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

        async def calculate_centrality(self, corpus=None):
            assert corpus == "alpha"
            return {"c::a": 0.5, "c::b": 0.0031, "c::c": 0.0}

        async def analyze_communities(self, resolution=1.0, corpus=None):
            return [{"c::a", "c::b"}, {"c::c"}]

    monkeypatch.setattr("kb_arena.graph.analyzer.GraphAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr("kb_arena.graph.neo4j_store.Neo4jStore", lambda driver: object())
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(neo4j=object())))

    stats = await api.graph_stats(request, corpus="alpha")

    assert stats["node_count"] == 3
    assert stats["corpus"] == "alpha"
    assert stats["community_count"] == 2
    assert stats["centrality"]["method"] == "approximate"
    assert stats["load"]["truncated"] is True
    assert stats["top_hubs"][0]["centrality"] == 0.5
    # a sampled score keeps two significant digits, so a small hub is not erased
    assert stats["top_hubs"][1]["centrality"] == 0.0031


@pytest.mark.asyncio
async def test_the_directed_graph_loads_under_the_same_budget(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 3)
    monkeypatch.setattr(settings, "graph_edge_budget", 100)
    nodes, edges = _chain(6)
    analyzer = GraphAnalyzer(_store(nodes, edges))

    graph = await analyzer._build_directed_graph()

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    calls = analyzer._store.execute_query.await_args_list
    limits = [c.args[1]["limit"] for c in calls if "limit" in c.args[1]]
    assert limits == [4, 101]
    chains = await analyzer.find_dependency_chains("c::n0", max_depth=4)
    assert chains and all(len(path) <= 3 for path in chains)


def test_a_zero_budget_or_sample_count_is_rejected():
    from kb_arena.settings import Settings

    with pytest.raises(ValueError):
        Settings(graph_node_budget=0)
    with pytest.raises(ValueError):
        Settings(graph_centrality_samples=0)
    assert Settings(graph_node_budget=1).graph_node_budget == 1


def _two_corpora():
    nodes, edges = [], []
    for corpus, n in (("alpha", 6), ("beta", 6)):
        nodes += [
            {"entity_id": f"{corpus}::n{i}", "fqn": f"n{i}", "label": "Topic"} for i in range(n)
        ]
        edges += [
            {"src": f"{corpus}::n{i}", "dst": f"{corpus}::n{i + 1}", "rel": "DEPENDS_ON"}
            for i in range(n - 1)
        ]
    return nodes, edges


@pytest.mark.asyncio
async def test_a_truncated_multi_corpus_load_names_its_corpora_and_the_total(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 8)
    nodes, edges = _two_corpora()
    analyzer = GraphAnalyzer(_store(nodes, edges))

    await analyzer.calculate_centrality()

    assert analyzer.last_load["truncated"] is True
    assert analyzer.last_load["corpora_in_slice"] == ["alpha", "beta"]
    assert analyzer.last_load["nodes_total"] == 12
    assert analyzer.last_load["nodes_loaded"] == 8


@pytest.mark.asyncio
async def test_a_corpus_filter_loads_that_corpus_only(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 8)
    nodes, edges = _two_corpora()
    analyzer = GraphAnalyzer(_store(nodes, edges))

    centrality = await analyzer.calculate_centrality("beta")

    assert set(centrality) == {f"beta::n{i}" for i in range(6)}
    assert analyzer.last_load["corpus"] == "beta"
    assert analyzer.last_load["truncated"] is False
    assert analyzer.last_load["corpora_in_slice"] == ["beta"]


@pytest.mark.asyncio
async def test_above_the_ceiling_the_run_is_a_sample_and_a_cache_hit_keeps_the_record(monkeypatch):
    monkeypatch.setattr(settings, "graph_centrality_exact_max_nodes", 5)
    monkeypatch.setattr(settings, "graph_centrality_samples", 500)
    nodes, edges = _chain(20)
    analyzer = GraphAnalyzer(_store(nodes, edges))

    first = await analyzer.calculate_centrality()
    # the pivots cover every node, the numbers equal the exact ones, and the
    # label still says sample, because the ceiling is what bounds the cost
    assert analyzer.last_centrality == {"method": "approximate", "nodes": 20, "samples": 20}
    analyzer.last_centrality = {}
    analyzer.last_load = {}

    second = await analyzer.calculate_centrality()

    assert second == first
    assert analyzer.last_centrality["method"] == "approximate"
    assert analyzer.last_load["nodes_loaded"] == 20


@pytest.mark.asyncio
async def test_communities_are_seeded():
    nodes, edges = _chain(60)
    analyzer = GraphAnalyzer(_store(nodes, edges))

    first = await analyzer.analyze_communities()
    analyzer._cache.clear()
    second = await analyzer.analyze_communities()

    assert [sorted(c) for c in first] == [sorted(c) for c in second]


@pytest.mark.asyncio
async def test_an_edge_that_leaves_the_slice_never_spends_the_edge_budget(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 3)
    monkeypatch.setattr(settings, "graph_edge_budget", 2)
    nodes, edges = _chain(6)
    # three edges from the slice into the rest of the graph, listed first
    edges = [{"src": "c::n0", "dst": f"c::n{i}", "rel": "X"} for i in (3, 4, 5)] + edges
    analyzer = GraphAnalyzer(_store(nodes, edges))

    graph = await analyzer._build_networkx_graph()

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2, "the budget went to edges inside the slice"


@pytest.mark.asyncio
async def test_dependency_chains_report_their_load_so_a_missing_entity_is_explained(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 3)
    nodes, edges = _chain(6)
    analyzer = GraphAnalyzer(_store(nodes, edges))

    chains = await analyzer.find_dependency_chains("c::n5", max_depth=4)

    assert chains == []
    assert analyzer.last_load["truncated"] is True, "the entity sits past the budget, not absent"
    assert analyzer.last_load["nodes_total"] == 6
    again = await analyzer.find_dependency_chains("c::n0", max_depth=4)
    assert again and analyzer.last_load["nodes_loaded"] == 3


@pytest.mark.asyncio
async def test_parallel_relationships_are_counted_as_rows_and_as_one_edge(monkeypatch):
    monkeypatch.setattr(settings, "graph_node_budget", 100)
    nodes, edges = _chain(3)
    edges = edges + [{"src": "c::n0", "dst": "c::n1", "rel": "CONFIGURES"}]  # a second relationship
    analyzer = GraphAnalyzer(_store(nodes, edges))

    graph = await analyzer._build_networkx_graph()

    assert analyzer.last_load["edge_rows_loaded"] == 3
    assert analyzer.last_load["edges_loaded"] == 2
    assert graph.number_of_edges() == 2
