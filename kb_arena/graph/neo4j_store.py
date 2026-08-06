"""Async Neo4j batch loader — UNWIND/MERGE pattern with cursor safety."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from kb_arena.graph.schema import NodeType, RelType
from kb_arena.settings import settings

logger = logging.getLogger(__name__)


def _prepare_for_neo4j(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize non-primitive values for Neo4j property storage.

    Neo4j properties must be primitives or arrays of primitives.
    Dicts get serialized to JSON strings; lists of dicts get serialized too.
    """
    prepared = []
    for record in records:
        clean = {}
        for k, v in record.items():
            if isinstance(v, dict):
                clean[k] = json.dumps(v) if v else ""
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                clean[k] = json.dumps(v)
            else:
                clean[k] = v
        prepared.append(clean)
    return prepared


BATCH_SIZE = 1000
LEGACY_CONSTRAINT_NAMES = (
    "topic_fqn",
    "component_fqn",
    "process_fqn",
    "config_fqn",
    "constraint_fqn",
)


class Neo4jStore:
    def __init__(self, driver: AsyncDriver, database: str | None = None) -> None:
        self._driver = driver
        self._database = database or settings.neo4j_database

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @classmethod
    async def connect(
        cls,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> Neo4jStore:
        target_database = database or settings.neo4j_database
        driver = AsyncGraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )
        try:
            async with driver.session(database=target_database) as session:
                result = await session.run("RETURN 1")
                await result.consume()
        except BaseException:
            await driver.close()
            raise
        return cls(driver, database=target_database)

    async def close(self) -> None:
        await self._driver.close()

    # ── Generic query ─────────────────────────────────────────────────────────

    async def execute_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a Cypher query and return records as plain dicts.

        Always consumes the result to avoid cursor leaks.
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            await result.consume()
            return records

    # ── Schema DDL ────────────────────────────────────────────────────────────

    async def load_schema(self, cypher_file: str | Path) -> None:
        """Execute idempotent DDL from a .cypher file.

        Splits on semicolons so multi-statement files work correctly.
        """
        text = Path(cypher_file).read_text()
        statements = [s.strip() for s in text.split(";") if s.strip()]
        async with self._driver.session(database=self._database) as session:
            for stmt in statements:
                result = await session.run(stmt)
                await result.consume()
        logger.info("Loaded schema from %s (%d statements)", cypher_file, len(statements))

    async def legacy_constraint_names(self) -> list[str]:
        """Return pre-0.10 KB Arena constraints present in the target database."""
        rows = await self.execute_query(
            "SHOW CONSTRAINTS YIELD name " "WHERE name IN $names " "RETURN name ORDER BY name",
            {"names": list(LEGACY_CONSTRAINT_NAMES)},
        )
        return [row["name"] for row in rows if row.get("name") in LEGACY_CONSTRAINT_NAMES]

    async def drop_legacy_constraints(self) -> list[str]:
        """Drop only known pre-0.10 constraints after explicit operator confirmation."""
        names = await self.legacy_constraint_names()
        if not names:
            return []

        async with self._driver.session(database=self._database) as session:
            for name in names:
                result = await session.run(f"DROP CONSTRAINT `{name}` IF EXISTS")
                await result.consume()
        return names

    # ── Batch node loading ────────────────────────────────────────────────────

    async def load_nodes(self, nodes: list[dict[str, Any]], label: NodeType) -> int:
        """UNWIND/MERGE batch load for a single node label.

        MERGE on the corpus-qualified entity ID, then SET remaining properties.
        Returns count of newly created nodes.
        """
        if not nodes:
            return 0

        created = 0
        safe_nodes = _prepare_for_neo4j(nodes)
        for node in safe_nodes:
            node.setdefault("corpus", "default")
            node.setdefault("entity_id", f"{node['corpus']}::{node['fqn']}")
        query = f"""
        UNWIND $records AS record
        MERGE (n:{label.value}:KBArenaEntity {{entity_id: record.entity_id}})
        SET n += record
        """
        async with self._driver.session(database=self._database) as session:
            for i in range(0, len(safe_nodes), BATCH_SIZE):
                batch = safe_nodes[i : i + BATCH_SIZE]
                result = await session.run(query, records=batch)
                summary = await result.consume()
                created += summary.counters.nodes_created

        logger.debug("load_nodes %s: %d created", label.value, created)
        return created

    # ── Batch edge loading ────────────────────────────────────────────────────

    async def load_edges(self, edges: list[dict[str, Any]], rel_type: RelType) -> int:
        """UNWIND/MERGE batch load for a single relationship type.

        Edges referencing non-existent nodes are silently dropped by MATCH.
        Always call load_nodes first.
        """
        if not edges:
            return 0

        created = 0
        safe_edges = _prepare_for_neo4j(edges)
        for edge in safe_edges:
            edge.setdefault("corpus", "default")
            edge.setdefault(
                "source_entity_id",
                f"{edge['corpus']}::{edge['source_fqn']}",
            )
            edge.setdefault(
                "target_entity_id",
                f"{edge['corpus']}::{edge['target_fqn']}",
            )
        query = f"""
        UNWIND $records AS record
        MATCH (a:KBArenaEntity {{entity_id: record.source_entity_id}})
        MATCH (b:KBArenaEntity {{entity_id: record.target_entity_id}})
        MERGE (a)-[r:{rel_type.value}]->(b)
        SET r.source_section_id = record.source_section_id,
            r.extraction_confidence = record.extraction_confidence,
            r.properties = record.properties
        """
        async with self._driver.session(database=self._database) as session:
            for i in range(0, len(safe_edges), BATCH_SIZE):
                batch = safe_edges[i : i + BATCH_SIZE]
                result = await session.run(query, records=batch)
                summary = await result.consume()
                created += summary.counters.relationships_created

        logger.debug("load_edges %s: %d created", rel_type.value, created)
        return created
