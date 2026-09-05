"""Strategy 13: LightRAG — local entity neighborhood + global community summary.

Reads the same Neo4j graph knowledge_graph reads, through the same read-only
Cypher path (Pattern 15 mock fallback included). It does not replace
knowledge_graph; it adds a second, LightRAG-style way to read that graph:

* Local retrieval walks the one-hop neighborhood of entities matched in the
  question — the same kind of lookup ENTITY_LOOKUP does in knowledge_graph.py.
* Global retrieval runs a fulltext search for candidate entities, then groups
  them into connected components using the neighbor links returned by the
  same query. The largest component is the "community", and its members'
  names and descriptions are the community-level summary.

Every retrieved chunk keeps `metadata["retrieval_mode"]` set to "local" or
"global", so a caller can tell which path produced it. Both paths reuse
`_records_to_chunks` from knowledge_graph.py, which is what keeps
source_doc_id and source_section_id intact end to end.
"""

from __future__ import annotations

import re
import time

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy, validate_top_k
from kb_arena.strategies.knowledge_graph import _mock_graph_context, _records_to_chunks

# One-hop neighborhood of the matched entity, or the entity alone with no match.
LOCAL_NEIGHBORHOOD = """
MATCH (n:KBArenaEntity)
WHERE (n.fqn = $fqn OR toLower(n.name) = toLower($name))
  AND ($corpus = 'all' OR n.corpus = $corpus)
OPTIONAL MATCH (n)-[r]-(neighbor:KBArenaEntity)
WHERE $corpus = 'all' OR neighbor.corpus = $corpus
WITH n, collect(DISTINCT neighbor) AS neighbors
UNWIND ([n] + neighbors) AS entity
WITH DISTINCT entity
WHERE entity IS NOT NULL
RETURN entity.name AS name, entity.fqn AS fqn,
       head([label IN labels(entity) WHERE label <> 'KBArenaEntity']) AS type,
       entity.description AS description,
       entity.source_doc_id AS source_doc_id,
       entity.source_section_id AS source_section_id
LIMIT 30
"""

# Fulltext candidates plus their neighbor ids, so a component can be built from
# this one query without a second round trip to Neo4j.
GLOBAL_COMMUNITY_QUERY = """
CALL db.index.fulltext.queryNodes('entity_search', $query) YIELD node, score
WHERE node:KBArenaEntity AND ($corpus = 'all' OR node.corpus = $corpus)
WITH node, score ORDER BY score DESC LIMIT 15
OPTIONAL MATCH (node)-[r]-(neighbor:KBArenaEntity)
WHERE $corpus = 'all' OR neighbor.corpus = $corpus
RETURN node.entity_id AS entity_id, node.name AS name, node.fqn AS fqn,
       head([label IN labels(node) WHERE label <> 'KBArenaEntity']) AS type,
       node.description AS description,
       node.source_doc_id AS source_doc_id,
       node.source_section_id AS source_section_id,
       collect(DISTINCT neighbor.entity_id) AS neighbor_ids
"""

SYSTEM_PROMPT = """You are a documentation assistant with access to a knowledge graph read two ways.
Local context is the immediate neighborhood of entities named in the question.
Global context is a community summary — the entities most related to the question's topic.
Answer using both. If one is empty, rely on the other and say so."""


def _extract_entities(question: str) -> list[str]:
    """Heuristic entity extraction, same shape as knowledge_graph's."""
    entities = []
    entities.extend(re.findall(r'["\']([^"\']+)["\']', question))
    entities.extend(re.findall(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)\b", question))
    entities.extend(re.findall(r"\b([a-z][a-z0-9]*(?:-[a-z][a-z0-9]*)+)\b", question))
    entities.extend(re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", question))
    return list(dict.fromkeys(entities))[:3]


def _largest_component(records: list[dict]) -> list[dict]:
    """Group global-query records by neighbor overlap; return the largest group.

    Uses only the neighbor ids returned by GLOBAL_COMMUNITY_QUERY itself, so
    grouping candidate entities into a community costs no second Neo4j call.
    """
    by_id = {r["entity_id"]: r for r in records if r.get("entity_id")}
    visited: set[str] = set()
    components: list[list[dict]] = []
    for entity_id in by_id:
        if entity_id in visited:
            continue
        stack = [entity_id]
        component: list[dict] = []
        while stack:
            current = stack.pop()
            if current in visited or current not in by_id:
                continue
            visited.add(current)
            component.append(by_id[current])
            stack.extend(by_id[current].get("neighbor_ids") or [])
        components.append(component)
    if not components:
        return []
    return max(components, key=len)


def _drop_adjacency_fields(records: list[dict]) -> list[dict]:
    """Strip the ids used only to build the community, before chunking."""
    return [{k: v for k, v in r.items() if k not in ("entity_id", "neighbor_ids")} for r in records]


class LightRAGStrategy(Strategy):
    """Local entity neighborhood + global community summary over one Neo4j graph."""

    name = "lightrag"

    def __init__(self, neo4j_driver=None):
        super().__init__()
        self._driver = neo4j_driver
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    async def build_index(self, documents: list[Document]) -> None:
        """Graph is built by build_graph CLI command — nothing to do here."""
        pass

    async def _run_cypher(self, cypher: str, params: dict) -> list[dict]:
        """Execute Cypher in a read-only session — same guard as knowledge_graph."""
        if self._driver is None:
            return []
        try:
            import neo4j as _neo4j

            session_kwargs = {"default_access_mode": _neo4j.READ_ACCESS}
        except ImportError:  # pragma: no cover — neo4j is a hard dep
            session_kwargs = {}
        async with self._driver.session(
            database=settings.neo4j_database,
            **session_kwargs,
        ) as session:
            result = await session.run(cypher, parameters=params)
            records = await result.data()
            await result.consume()
        return records

    async def _local_retrieval(self, question: str, corpus: str) -> list[dict]:
        entities = _extract_entities(question)
        if not entities:
            return []
        primary = entities[0]
        return await self._run_cypher(
            LOCAL_NEIGHBORHOOD, {"fqn": primary, "name": primary, "corpus": corpus}
        )

    async def _global_retrieval(self, question: str, corpus: str) -> list[dict]:
        records = await self._run_cypher(
            GLOBAL_COMMUNITY_QUERY, {"query": question, "corpus": corpus}
        )
        return _drop_adjacency_fields(_largest_component(records))

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        validate_top_k(top_k)
        start = self._start_timer()

        if self._driver is None:
            graph_ctx = _mock_graph_context()
            latency_ms = self._record_metrics(start, graph_context=graph_ctx)
            return AnswerResult(
                answer="[Graph database not connected. Showing mock data for demo purposes.]",
                sources=[],
                graph_context=graph_ctx,
                retrieval=RetrievalTrace(query=question, retrieved=[], latency_ms=0.0, top_k=top_k),
                strategy=self.name,
                latency_ms=latency_ms,
                mock=True,
            )

        retrieval_start = time.perf_counter()
        local_records = await self._local_retrieval(question, corpus)
        global_records = await self._global_retrieval(question, corpus)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        local_chunks = _records_to_chunks(local_records, self.name, top_k)
        for chunk in local_chunks:
            chunk.metadata["retrieval_mode"] = "local"
        global_chunks = _records_to_chunks(global_records, self.name, top_k)
        for chunk in global_chunks:
            chunk.metadata["retrieval_mode"] = "global"

        merged: dict[str, RetrievedChunk] = {}
        for chunk in local_chunks + global_chunks:
            merged.setdefault(chunk.chunk_id, chunk)
        ranked = list(merged.values())[:top_k]
        for i, chunk in enumerate(ranked):
            chunk.rank = i + 1

        trace = RetrievalTrace(
            query=question, retrieved=ranked, latency_ms=retrieval_ms, top_k=top_k
        )
        sources = list(dict.fromkeys(c.doc_id for c in ranked if c.doc_id))

        local_text = "\n".join(c.content for c in local_chunks) or "No local neighborhood match."
        global_text = "\n".join(c.content for c in global_chunks) or "No global community match."
        context = f"Local neighborhood:\n{local_text}\n\nGlobal community summary:\n{global_text}"

        llm = self._get_llm()
        gen_start = time.perf_counter()
        resp = await llm.generate(query=question, context=context, system_prompt=SYSTEM_PROMPT)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency_ms = self._record_metrics(
            start, tokens=resp.total_tokens, cost=resp.cost_usd, sources=sources
        )
        return AnswerResult(
            answer=resp.text,
            sources=sources,
            retrieval=trace,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )
