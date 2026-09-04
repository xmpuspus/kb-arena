"""Graph analysis using networkx for algorithms, Neo4j for storage.

CPU-intensive algorithms run via asyncio.to_thread (climate-money-ph pattern).
Results are cached in-memory for 5 minutes to avoid repeated graph pulls.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from typing import Any

import networkx as nx

from kb_arena.graph.neo4j_store import Neo4jStore
from kb_arena.settings import settings

logger = logging.getLogger(__name__)

_CACHE_TTL = 300  # seconds


class GraphAnalyzer:
    def __init__(self, store: Neo4jStore) -> None:
        self._store = store
        # cache: key -> (timestamp, value)
        self._cache: dict[Any, tuple[float, Any]] = {}
        # What the last load and the last centrality run did, for the caller
        # that reports them. A truncated graph or a sampled centrality must
        # never pass as the whole picture.
        self.last_load: dict[str, Any] = {}
        self.last_centrality: dict[str, Any] = {}

    async def _load_rows(
        self,
        cypher: str,
        budget: int,
        corpus: str | None = None,
        ids: list[str] | None = None,
    ) -> tuple[list[dict], bool]:
        """Rows up to the budget, and whether more existed beyond it.

        One extra row tells truncation apart from a graph that fits. An entity
        id starts with its corpus, so a corpus filter is a prefix match. An
        edge query gets the loaded node ids, so an edge that leaves the slice
        never spends the edge budget.
        """
        params: dict[str, Any] = {"limit": budget + 1, "prefix": f"{corpus}::" if corpus else ""}
        if ids is not None:
            params["ids"] = ids
        rows = await self._store.execute_query(cypher, params)
        rows = list(rows)
        truncated = len(rows) > budget
        return rows[:budget], truncated

    def _cache_get(self, key: Any) -> Any | None:
        entry = self._cache.get(key)
        if entry and time.monotonic() - entry[0] < _CACHE_TTL:
            return entry[1]
        return None

    def _cache_set(self, key: Any, value: Any) -> None:
        self._cache[key] = (time.monotonic(), value)

    async def _count_nodes(self, corpus: str | None) -> int | None:
        """How many entities the store holds, so a slice can say what it left out."""
        try:
            rows = await self._store.execute_query(
                "MATCH (n:KBArenaEntity) WHERE $prefix = '' OR n.entity_id STARTS WITH $prefix "
                "RETURN count(n) AS n",
                {"prefix": f"{corpus}::" if corpus else ""},
            )
        except Exception:  # noqa: BLE001 - a count is a courtesy, never a failure
            return None
        rows = list(rows)
        if not rows:
            return None
        value = rows[0].get("n") if isinstance(rows[0], dict) else None
        return int(value) if isinstance(value, int) else None

    # ── Graph builders ────────────────────────────────────────────────────────

    async def _build_networkx_graph(self, corpus: str | None = None) -> nx.Graph:
        """Pull nodes and edges from Neo4j into an undirected networkx graph.

        With a corpus, only that corpus's entities load. Without one, the
        slice is ordered by entity id, which starts with the corpus name, so
        a truncated multi-corpus load names the corpora it holds.
        """
        cached = self._cache_get(("undirected", corpus))
        if cached is not None:
            graph, self.last_load = cached
            return graph

        nodes, nodes_truncated = await self._load_rows(
            "MATCH (n:KBArenaEntity) WHERE $prefix = '' OR n.entity_id STARTS WITH $prefix "
            "RETURN n.entity_id AS entity_id, n.fqn AS fqn, "
            "head([label IN labels(n) WHERE label <> 'KBArenaEntity']) AS label "
            "ORDER BY n.entity_id LIMIT $limit",
            settings.graph_node_budget,
            corpus,
        )
        edges, edges_truncated = await self._load_rows(
            "MATCH (a:KBArenaEntity)-[r]->(b:KBArenaEntity) "
            "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
            "RETURN a.entity_id AS src, b.entity_id AS dst, type(r) AS rel "
            "ORDER BY a.entity_id, b.entity_id LIMIT $limit",
            settings.graph_edge_budget,
            corpus,
            ids=[row["entity_id"] for row in nodes],
        )

        graph: nx.Graph = nx.Graph()
        for row in nodes:
            graph.add_node(row["entity_id"], fqn=row["fqn"], label=row["label"])
        dropped_edges = 0
        for row in edges:
            # An edge to a node outside the loaded slice would add a bare node
            # and inflate the count, so it stays out.
            if row["src"] not in graph or row["dst"] not in graph:
                dropped_edges += 1
                continue
            graph.add_edge(row["src"], row["dst"], rel=row["rel"], weight=1)

        corpora = sorted({str(n).split("::", 1)[0] for n in graph.nodes if "::" in str(n)})
        total = await self._count_nodes(corpus) if nodes_truncated else graph.number_of_nodes()
        self.last_load = {
            "corpus": corpus or "all",
            "corpora_in_slice": corpora,
            "nodes_loaded": graph.number_of_nodes(),
            "nodes_total": total,
            "edges_loaded": graph.number_of_edges(),
            "node_budget": settings.graph_node_budget,
            "edge_budget": settings.graph_edge_budget,
            "truncated": nodes_truncated or edges_truncated,
            "edges_dropped_outside_slice": dropped_edges,
        }
        if self.last_load["truncated"]:
            logger.warning(
                "Graph load truncated at the budget: %d nodes, %d edges loaded. "
                "Raise KB_ARENA_GRAPH_NODE_BUDGET or KB_ARENA_GRAPH_EDGE_BUDGET "
                "for the whole graph.",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
        self._cache_set(("undirected", corpus), (graph, dict(self.last_load)))
        logger.debug(
            "Built undirected graph: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    async def _build_directed_graph(self, corpus: str | None = None) -> nx.DiGraph:
        """Directed graph for path finding, loaded under the same budgets."""
        cached = self._cache_get(("directed", corpus))
        if cached is not None:
            graph, self.last_load = cached
            return graph

        nodes, nodes_truncated = await self._load_rows(
            "MATCH (n:KBArenaEntity) WHERE $prefix = '' OR n.entity_id STARTS WITH $prefix "
            "RETURN n.entity_id AS entity_id, n.fqn AS fqn, "
            "head([label IN labels(n) WHERE label <> 'KBArenaEntity']) AS label "
            "ORDER BY n.entity_id LIMIT $limit",
            settings.graph_node_budget,
            corpus,
        )
        edges, edges_truncated = await self._load_rows(
            "MATCH (a:KBArenaEntity)-[r]->(b:KBArenaEntity) "
            "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
            "RETURN a.entity_id AS src, b.entity_id AS dst, type(r) AS rel "
            "ORDER BY a.entity_id, b.entity_id LIMIT $limit",
            settings.graph_edge_budget,
            corpus,
            ids=[row["entity_id"] for row in nodes],
        )

        graph: nx.DiGraph = nx.DiGraph()
        for row in nodes:
            graph.add_node(row["entity_id"], fqn=row["fqn"], label=row["label"])
        for row in edges:
            if row["src"] not in graph or row["dst"] not in graph:
                continue
            graph.add_edge(row["src"], row["dst"], rel=row["rel"])
        # The same record the undirected load keeps, so a caller of
        # find_dependency_chains can tell an absent entity from one the
        # budget left out.
        self.last_load = {
            "corpus": corpus or "all",
            "corpora_in_slice": sorted(
                {str(n).split("::", 1)[0] for n in graph.nodes if "::" in str(n)}
            ),
            "nodes_loaded": graph.number_of_nodes(),
            "nodes_total": (
                await self._count_nodes(corpus) if nodes_truncated else graph.number_of_nodes()
            ),
            "edges_loaded": graph.number_of_edges(),
            "node_budget": settings.graph_node_budget,
            "edge_budget": settings.graph_edge_budget,
            "truncated": nodes_truncated or edges_truncated,
            "edges_dropped_outside_slice": 0,
        }
        if self.last_load["truncated"]:
            logger.warning(
                "Directed graph load truncated at the budget: %d nodes, %d edges loaded.",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
        self._cache_set(("directed", corpus), (graph, dict(self.last_load)))
        return graph

    async def analyze_communities(
        self, resolution: float = 1.0, corpus: str | None = None
    ) -> list[set[str]]:
        """Louvain community detection, seeded so two calls agree.

        CPU-bound: runs in a thread so the event loop stays free.
        Returns list of sets, each containing corpus-qualified entity IDs.
        """
        graph = await self._build_networkx_graph(corpus)
        communities = await asyncio.to_thread(
            nx.community.louvain_communities,
            graph,
            weight="weight",
            resolution=resolution,
            seed=0,
        )
        return [set(c) for c in communities]

    async def find_dependency_chains(
        self, start_fqn: str, max_depth: int = 4, corpus: str | None = None
    ) -> list[list[str]]:
        """Find paths from an entity ID or FQN up to max_depth hops.

        Caps at 100 paths to avoid combinatorial explosion (climate-money-ph lesson).
        """
        graph = await self._build_directed_graph(corpus)
        start_nodes = (
            [start_fqn]
            if start_fqn in graph
            else [
                node_id for node_id, data in graph.nodes(data=True) if data.get("fqn") == start_fqn
            ]
        )
        if not start_nodes:
            return []

        paths = await asyncio.to_thread(
            lambda: list(
                itertools.islice(
                    (
                        p
                        for start_node in start_nodes
                        for target in graph.nodes
                        if target != start_node
                        for p in nx.all_simple_paths(graph, start_node, target, cutoff=max_depth)
                    ),
                    100,
                )
            )
        )
        return paths

    async def calculate_centrality(self, corpus: str | None = None) -> dict[str, float]:
        """Betweenness centrality for all nodes.

        High centrality = conceptual hub (important for graph-guided RAG).
        """
        cached = self._cache_get(("centrality", corpus))
        if cached is not None:
            centrality, self.last_centrality = cached
            await self._build_networkx_graph(corpus)  # restores last_load from its cache
            return centrality

        graph = await self._build_networkx_graph(corpus)
        node_count = graph.number_of_nodes()
        samples = min(settings.graph_centrality_samples, node_count)
        # k pivots equal to every node is the exact computation at the same
        # cost, so it is labeled exact. The ceiling still bounds the cost:
        # above it the run never visits more than graph_centrality_samples
        # pivots.
        exact = node_count <= settings.graph_centrality_exact_max_nodes or samples >= node_count
        if exact:
            centrality: dict[str, float] = await asyncio.to_thread(
                nx.betweenness_centrality, graph, normalized=True
            )
            self.last_centrality = {"method": "exact", "nodes": node_count, "samples": None}
        else:
            # Brandes sampling: k pivot nodes instead of all of them. Seeded,
            # so two calls on one graph agree.
            centrality = await asyncio.to_thread(
                nx.betweenness_centrality, graph, k=samples, normalized=True, seed=0
            )
            self.last_centrality = {
                "method": "approximate",
                "nodes": node_count,
                "samples": samples,
            }
        self._cache_set(("centrality", corpus), (centrality, dict(self.last_centrality)))
        return centrality
