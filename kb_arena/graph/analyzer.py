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

logger = logging.getLogger(__name__)

_CACHE_TTL = 300  # seconds


class GraphAnalyzer:
    def __init__(self, store: Neo4jStore) -> None:
        self._store = store
        # cache: key -> (timestamp, value)
        self._cache: dict[str, tuple[float, Any]] = {}

    def _cache_get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry and time.monotonic() - entry[0] < _CACHE_TTL:
            return entry[1]
        return None

    def _cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = (time.monotonic(), value)

    # ── Graph builders ────────────────────────────────────────────────────────

    async def _build_networkx_graph(self) -> nx.Graph:
        """Pull nodes and edges from Neo4j into an undirected networkx graph."""
        cached = self._cache_get("undirected")
        if cached is not None:
            return cached

        nodes = await self._store.execute_query(
            "MATCH (n) RETURN n.fqn AS fqn, labels(n)[0] AS label"
        )
        edges = await self._store.execute_query(
            "MATCH (a)-[r]->(b) RETURN a.fqn AS src, b.fqn AS dst, type(r) AS rel"
        )

        graph: nx.Graph = nx.Graph()
        for row in nodes:
            graph.add_node(row["fqn"], label=row["label"])
        for row in edges:
            graph.add_edge(row["src"], row["dst"], rel=row["rel"], weight=1)

        self._cache_set("undirected", graph)
        logger.debug(
            "Built undirected graph: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    async def _build_directed_graph(self) -> nx.DiGraph:
        """Pull nodes and edges from Neo4j into a directed networkx graph."""
        cached = self._cache_get("directed")
        if cached is not None:
            return cached

        nodes = await self._store.execute_query(
            "MATCH (n) RETURN n.fqn AS fqn, labels(n)[0] AS label"
        )
        edges = await self._store.execute_query(
            "MATCH (a)-[r]->(b) RETURN a.fqn AS src, b.fqn AS dst, type(r) AS rel"
        )

        graph: nx.DiGraph = nx.DiGraph()
        for row in nodes:
            graph.add_node(row["fqn"], label=row["label"])
        for row in edges:
            graph.add_edge(row["src"], row["dst"], rel=row["rel"])

        self._cache_set("directed", graph)
        return graph

    # ── Algorithms ────────────────────────────────────────────────────────────

    async def analyze_communities(self, resolution: float = 1.0) -> list[set[str]]:
        """Louvain community detection.

        CPU-bound — runs in thread pool to avoid blocking the event loop.
        Returns list of sets, each containing fqn strings in that community.
        """
        graph = await self._build_networkx_graph()
        communities = await asyncio.to_thread(
            nx.community.louvain_communities, graph, weight="weight", resolution=resolution
        )
        return [set(c) for c in communities]

    async def find_dependency_chains(
        self, start_fqn: str, max_depth: int = 4
    ) -> list[list[str]]:
        """Find all simple paths from start_fqn up to max_depth hops.

        Caps at 100 paths to avoid combinatorial explosion (climate-money-ph lesson).
        """
        graph = await self._build_directed_graph()
        if start_fqn not in graph:
            return []

        targets = [n for n in graph.nodes if n != start_fqn]

        paths = await asyncio.to_thread(
            lambda: list(
                itertools.islice(
                    (
                        p
                        for t in targets
                        for p in nx.all_simple_paths(graph, start_fqn, t, cutoff=max_depth)
                    ),
                    100,
                )
            )
        )
        return paths

    async def calculate_centrality(self) -> dict[str, float]:
        """Betweenness centrality for all nodes.

        High centrality = conceptual hub (important for graph-guided RAG).
        """
        cached = self._cache_get("centrality")
        if cached is not None:
            return cached

        graph = await self._build_networkx_graph()
        centrality: dict[str, float] = await asyncio.to_thread(
            nx.betweenness_centrality, graph, normalized=True
        )
        self._cache_set("centrality", centrality)
        return centrality
