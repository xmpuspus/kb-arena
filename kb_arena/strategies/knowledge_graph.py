"""Strategy 4: Knowledge Graph (Neo4j).

Intent → select Cypher template or generate Cypher → execute → assemble context → Sonnet.
Falls back to mock data with warning when Neo4j is unavailable (Pattern 15).
Tracks graph_context for Sigma.js visualization.
"""

from __future__ import annotations

import logging
import re

from kb_arena.graph.schema import node_type_values, rel_type_values
from kb_arena.models.document import Document
from kb_arena.models.graph import GraphContext
from kb_arena.strategies.base import AnswerResult, Strategy

logger = logging.getLogger(__name__)

# Reject LLM-generated Cypher that contains write operations
_WRITE_CYPHER_RE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH|CALL\s+apoc\.schema)\b",
    re.IGNORECASE,
)

# --- Cypher templates (Pattern 6 from PLAN.md) ---

MULTI_HOP_QUERY = """
MATCH path = (start)-[*1..{depth}]-(connected)
WHERE start.fqn = $target AND ALL(r IN relationships(path) WHERE type(r) IN $allowed_rel_types)
RETURN connected.name AS name, connected.fqn AS fqn,
       labels(connected)[0] AS type,
       length(path) AS hops,
       [r IN relationships(path) | type(r)] AS relationship_chain
ORDER BY hops, connected.name
LIMIT 50
"""

COMPARISON_QUERY = """
MATCH (a)-[r1]-(shared)-[r2]-(b)
WHERE a.fqn = $entity_a AND b.fqn = $entity_b
RETURN shared.name AS shared_entity, labels(shared)[0] AS shared_type,
       type(r1) AS rel_to_a, type(r2) AS rel_to_b
LIMIT 50
"""

DEPENDENCY_CHAIN = """
MATCH path = (source)-[:DEPENDS_ON|CONNECTS_TO|TRIGGERS|EXTENDS|CONFIGURES*1..4]->(dep)
WHERE source.fqn = $start
WITH path, dep, length(path) AS depth
RETURN dep.name AS name, dep.fqn AS fqn, labels(dep)[0] AS type, depth,
       [n IN nodes(path) | n.name] AS chain
ORDER BY depth
LIMIT 100
"""

FULLTEXT_SEARCH = """
CALL db.index.fulltext.queryNodes('entity_search', $query) YIELD node, score
RETURN node.name AS name, node.fqn AS fqn,
       labels(node)[0] AS type, node.description AS description, score
ORDER BY score DESC
LIMIT 20
"""

ENTITY_LOOKUP = """
MATCH (n)
WHERE n.fqn = $fqn OR toLower(n.name) = toLower($name)
OPTIONAL MATCH (n)-[r]-(neighbor)
RETURN n.name AS name, n.fqn AS fqn, labels(n)[0] AS type,
       n.description AS description,
       collect({neighbor: neighbor.name, rel: type(r),
           dir: CASE WHEN startNode(r)=n THEN 'out' ELSE 'in' END}) AS neighbors
LIMIT 1
"""

SYSTEM_PROMPT = """You are a documentation assistant with access to a knowledge graph.
The context contains entities, relationships, and paths extracted from the graph.
Answer the question accurately using the graph context provided.
Be specific about relationships and dependencies. If the graph context is incomplete, say so."""

CYPHER_GEN_PROMPT_TEMPLATE = """\
You are a Neo4j Cypher expert. The graph contains documentation entities.

Node types: {node_types}
Relationship types: {rel_types}

Write a Cypher query to answer: {{question}}

Rules:
- Return only the Cypher query, no explanation
- Use LIMIT 50 to cap results
- Always return: name, fqn, type, and any relevant relationship fields
- Use $params for parameters (available: $query string)
"""


def _extract_cypher(raw: str) -> str:
    """Pull Cypher from LLM output that may include prose or markdown."""
    raw = raw.strip()
    # Strip markdown fences
    match = re.search(r"```(?:cypher)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Assume the whole response is Cypher if it starts with MATCH/CALL/WITH
    if re.match(r"^\s*(MATCH|CALL|WITH|RETURN|OPTIONAL)", raw, re.IGNORECASE):
        return raw
    return raw


def _mock_graph_context() -> GraphContext:
    """Two-node mock graph — enough to show visualization works (Pattern 15)."""
    return GraphContext(
        nodes=[
            {
                "id": "example-topic",
                "name": "Example Topic",
                "type": "Topic",
                "description": "A primary subject in the documentation",
            },
            {
                "id": "example-component",
                "name": "Example Component",
                "type": "Component",
                "description": "A building block that depends on the topic",
            },
        ],
        edges=[
            {"source": "example-component", "target": "example-topic", "type": "DEPENDS_ON"},
        ],
        query_path=["example-topic", "example-component"],
        cypher_used="MOCK — Neo4j not connected",
    )


def _results_to_context(records: list[dict]) -> str:
    """Format Neo4j records as readable context for LLM."""
    if not records:
        return "No relevant graph data found."
    lines = []
    for r in records[:30]:  # cap context size
        parts = []
        for k, v in r.items():
            if v is not None:
                parts.append(f"{k}: {v}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _records_to_graph_context(records: list[dict], cypher: str) -> GraphContext:
    """Convert Neo4j result records to GraphContext for visualization."""
    nodes_seen: dict[str, dict] = {}
    edges = []
    path_names = []

    for r in records:
        fqn = str(r.get("fqn", r.get("name", "")))
        name = str(r.get("name", fqn))
        node_type = str(r.get("type", "Unknown"))
        if fqn and fqn not in nodes_seen:
            nodes_seen[fqn] = {"id": fqn, "name": name, "type": node_type}

        # Handle relationship_chain from multi-hop queries
        chain = r.get("relationship_chain", [])
        if chain and isinstance(chain, list) and len(chain) > 0:
            path_names.append(name)

        # Handle neighbors from entity lookup
        neighbors = r.get("neighbors", [])
        if neighbors and isinstance(neighbors, list):
            for nb in neighbors:
                if not nb or not nb.get("neighbor"):
                    continue
                nb_id = str(nb.get("neighbor", ""))
                if nb_id and nb_id not in nodes_seen:
                    nodes_seen[nb_id] = {"id": nb_id, "name": nb_id, "type": "Unknown"}
                rel = nb.get("rel", "RELATED")
                direction = nb.get("dir", "out")
                src, tgt = (fqn, nb_id) if direction == "out" else (nb_id, fqn)
                edges.append({"source": src, "target": tgt, "type": rel})

    return GraphContext(
        nodes=list(nodes_seen.values()),
        edges=edges,
        query_path=path_names[:10],
        cypher_used=cypher,
    )


class KnowledgeGraphStrategy(Strategy):
    """Neo4j-backed retrieval — templates first, Text-to-Cypher fallback."""

    name = "knowledge_graph"

    def __init__(self, neo4j_driver=None):
        super().__init__()
        # neo4j_driver is the async neo4j.AsyncDriver instance (or None)
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
        """Execute Cypher and return list of record dicts."""
        if self._driver is None:
            return []
        async with self._driver.session() as session:
            result = await session.run(cypher, parameters=params)
            records = await result.data()
            await result.consume()
        return records

    async def _classify_intent(self, question: str) -> str:
        """Quick keyword-based intent classification to pick Cypher template."""
        q = question.lower()
        if any(kw in q for kw in ["compare", "vs", "difference", "vs.", "versus"]):
            return "comparison"
        if any(kw in q for kw in ["depend", "require", "affect", "downstream", "cause"]):
            return "dependency"
        if any(kw in q for kw in ["what is", "define", "explain", "describe"]):
            return "entity_lookup"
        if any(kw in q for kw in ["related", "connected", "linked", "path"]):
            return "multi_hop"
        return "fulltext"

    def _extract_entities(self, question: str) -> list[str]:
        """Heuristic: extract likely entity names (quoted, CamelCase, dotted, or hyphenated)."""
        entities = []
        # Quoted names
        entities.extend(re.findall(r'["\']([^"\']+)["\']', question))
        # Dotted paths like s3.put_object or aws.lambda
        entities.extend(re.findall(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)\b", question))
        # Hyphenated names like api-gateway, step-functions
        entities.extend(re.findall(r"\b([a-z][a-z0-9]*(?:-[a-z][a-z0-9]*)+)\b", question))
        # CamelCase names like Lambda, DynamoDB, CloudWatch
        entities.extend(re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", question))
        return list(dict.fromkeys(entities))[:3]  # deduplicate, limit

    async def _template_query(self, question: str, intent: str) -> tuple[list[dict], str]:
        """Run the appropriate Cypher template and return (records, cypher_used)."""
        entities = self._extract_entities(question)
        primary = entities[0] if entities else ""
        secondary = entities[1] if len(entities) > 1 else ""

        allowed_rels = rel_type_values("")

        if intent == "comparison" and primary and secondary:
            cypher = COMPARISON_QUERY
            params = {"entity_a": primary, "entity_b": secondary}
        elif intent == "dependency" and primary:
            cypher = DEPENDENCY_CHAIN
            params = {"start": primary}
        elif intent == "entity_lookup" and primary:
            cypher = ENTITY_LOOKUP
            params = {"fqn": primary, "name": primary}
        elif intent == "multi_hop" and primary:
            cypher = MULTI_HOP_QUERY.format(depth=3)
            params = {"target": primary, "allowed_rel_types": allowed_rels}
        else:
            # Fulltext search as catch-all
            cypher = FULLTEXT_SEARCH
            params = {"query": question}

        records = await self._run_cypher(cypher, params)
        return records, cypher

    async def _generate_cypher(self, question: str) -> tuple[list[dict], str]:
        """Text-to-Cypher fallback for novel queries."""
        llm = self._get_llm()
        cypher_gen_prompt = CYPHER_GEN_PROMPT_TEMPLATE.format(
            node_types=", ".join(node_type_values("")),
            rel_types=", ".join(rel_type_values("")),
        )
        prompt = cypher_gen_prompt.format(question=question)
        resp = await llm.extract(text=prompt, system_prompt="Output only Cypher. No prose.")
        cypher = _extract_cypher(resp.text)

        if _WRITE_CYPHER_RE.search(cypher):
            logger.warning("Blocked LLM-generated write Cypher: %.200s", cypher)
            records = await self._run_cypher(FULLTEXT_SEARCH, {"query": question})
            return records, FULLTEXT_SEARCH

        try:
            records = await self._run_cypher(cypher, {"query": question})
            return records, cypher
        except Exception:
            # Cypher generation can produce invalid queries — fall through to fulltext
            records = await self._run_cypher(FULLTEXT_SEARCH, {"query": question})
            return records, FULLTEXT_SEARCH

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        """Intent → Cypher template → execute → LLM answer."""
        start = self._start_timer()

        # Mock fallback: Neo4j not connected
        if self._driver is None:
            graph_ctx = _mock_graph_context()
            latency_ms = self._record_metrics(start, graph_context=graph_ctx)
            return AnswerResult(
                answer="[Graph database not connected. Showing mock data for demo purposes.]",
                sources=[],
                graph_context=graph_ctx,
                strategy=self.name,
                latency_ms=latency_ms,
                mock=True,
            )

        intent = await self._classify_intent(question)
        records, cypher_used = await self._template_query(question, intent)

        # If template returns nothing, try Text-to-Cypher
        if not records:
            records, cypher_used = await self._generate_cypher(question)

        context = _results_to_context(records)
        graph_ctx = _records_to_graph_context(records, cypher_used)
        sources = list({r.get("source_id", r.get("fqn", "")) for r in records if r})
        sources = [s for s in sources if s]

        llm = self._get_llm()
        resp = await llm.generate(
            query=question,
            context=context,
            system_prompt=SYSTEM_PROMPT,
        )

        latency_ms = self._record_metrics(
            start,
            tokens=resp.total_tokens,
            cost=resp.cost_usd,
            sources=sources,
            graph_context=graph_ctx,
        )
        return AnswerResult(
            answer=resp.text,
            sources=sources,
            graph_context=graph_ctx,
            strategy=self.name,
            latency_ms=latency_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )

    async def stream_answer(self, question: str, history: list[dict] | None = None):
        """Stream the final answer after synchronous graph retrieval."""
        start = self._start_timer()

        if self._driver is None:
            yield "[Graph database not connected. Showing mock data.]"
            self._record_metrics(start, graph_context=_mock_graph_context())
            return

        intent = await self._classify_intent(question)
        records, cypher_used = await self._template_query(question, intent)
        if not records:
            records, cypher_used = await self._generate_cypher(question)

        context = _results_to_context(records)
        graph_ctx = _records_to_graph_context(records, cypher_used)
        sources = [r.get("fqn", "") for r in records if r.get("fqn")]

        self.last_sources = sources
        self.last_graph_context = graph_ctx

        llm = self._get_llm()
        async for token in llm.stream(
            query=question,
            context=context,
            system_prompt=SYSTEM_PROMPT,
        ):
            yield token

        self._record_metrics(start, sources=sources, graph_context=graph_ctx)
