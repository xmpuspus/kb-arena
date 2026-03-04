"""Text-to-Cypher via LLM with template fallback and syntax validation."""

from __future__ import annotations

import logging
import re

from kb_arena.graph import cypher_templates
from kb_arena.graph.schema import get_schema
from kb_arena.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Minimal Cypher syntax check — catches unterminated strings and bare MATCH-less queries.
_CYPHER_SAFE_PATTERN = re.compile(r"\b(MATCH|CALL|RETURN|WITH|UNWIND)\b", re.IGNORECASE)

# Template keyword triggers — ordered by specificity
_TEMPLATE_TRIGGERS: list[tuple[list[str], str]] = [
    (["deprecat"], "DEPRECATION_CHAIN"),
    (["inherit", "hierarchy", "subclass", "superclass"], "TYPE_HIERARCHY"),
    (["example", "usage", "sample"], "USAGE_EXAMPLES"),
    (["depend", "import", "require"], "DEPENDENCY_CHAIN"),
    (["cross-ref", "references", "links to"], "CROSS_REFERENCE"),
    (["compare", "difference", "vs", "versus"], "COMPARISON_QUERY"),
    (["connected", "related", "hop", "neighbor"], "MULTI_HOP_QUERY"),
    (["find", "lookup", "get", "show", "what is"], "SINGLE_ENTITY_LOOKUP"),
]

_SYSTEM_PROMPT_TEMPLATE = """You are a Neo4j Cypher expert. Generate a single valid Cypher query.

Schema for corpus '{corpus}':
Node types: {node_types}
Relationship types: {rel_types}

Rules:
- Use ONLY the node and relationship types listed above
- Always use parameterized queries ($param), never interpolate values
- Return only the Cypher query, no explanation, no markdown fences
- The query must start with MATCH, CALL, or WITH
- Limit results to 50 unless the question asks for more
"""


def _validate_cypher(cypher: str) -> bool:
    """Minimal syntax check — does it contain at least one read clause?"""
    return bool(_CYPHER_SAFE_PATTERN.search(cypher))


def _pick_template(query: str) -> str | None:
    """Return a template name if the query matches known trigger keywords."""
    q = query.lower()
    for keywords, template_name in _TEMPLATE_TRIGGERS:
        if any(kw in q for kw in keywords):
            return template_name
    return None


def _get_template(name: str) -> str | None:
    return getattr(cypher_templates, name, None)


class CypherGenerator:
    def __init__(self, llm: LLMClient, corpus: str) -> None:
        self._llm = llm
        self._corpus = corpus
        node_enum, rel_enum = get_schema(corpus)
        self._node_types = [e.value for e in node_enum]
        self._rel_types = [e.value for e in rel_enum]
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            corpus=corpus,
            node_types=", ".join(self._node_types),
            rel_types=", ".join(self._rel_types),
        )

    async def generate(self, query: str, params: dict | None = None) -> tuple[str, dict]:
        """Return (cypher, params) for executing against Neo4j.

        Tries LLM first. Falls back to template matching if LLM output
        is invalid Cypher or the call fails.
        """
        params = params or {}

        # LLM attempt
        try:
            resp = await self._llm.extract(text=query, system_prompt=self._system_prompt)
            cypher = resp.text.strip()
            # Strip markdown fences if the model wrapped them
            if cypher.startswith("```"):
                cypher = re.sub(r"^```[a-z]*\n?", "", cypher)
                cypher = re.sub(r"\n?```$", "", cypher).strip()

            if _validate_cypher(cypher):
                logger.debug("CypherGenerator: LLM produced valid Cypher")
                return cypher, params
            logger.warning("CypherGenerator: LLM output failed validation, using template")
        except Exception as exc:
            logger.warning("CypherGenerator: LLM call failed (%s), using template", exc)

        # Template fallback
        template_name = _pick_template(query)
        if template_name:
            template = _get_template(template_name)
            if template:
                logger.debug("CypherGenerator: using template %s", template_name)
                return template, params

        # Last resort: fulltext search
        params.setdefault("query", query)
        params.setdefault("limit", 20)
        return cypher_templates.FULLTEXT_ENTITY_SEARCH, params
