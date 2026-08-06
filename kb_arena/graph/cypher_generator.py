"""Allowlisted parameterized Cypher template selection."""

from __future__ import annotations

import re

from kb_arena.graph import cypher_templates
from kb_arena.llm.client import LLMClient

# Template keyword triggers, ordered by specificity.
_TEMPLATE_TRIGGERS: list[tuple[list[str], str]] = [
    (["inherit", "hierarchy", "subclass", "superclass", "extends"], "TYPE_HIERARCHY"),
    (["depend", "import", "require"], "DEPENDENCY_CHAIN"),
    (["cross-ref", "references", "links to", "connected to"], "CROSS_REFERENCE"),
    (["compare", "difference", "vs", "versus"], "COMPARISON_QUERY"),
    (["connected", "related", "hop", "neighbor"], "MULTI_HOP_QUERY"),
    (["find", "lookup", "get", "show", "what is"], "SINGLE_ENTITY_LOOKUP"),
]

_REQUIRED_PARAMS: dict[str, frozenset[str]] = {
    "TYPE_HIERARCHY": frozenset({"fqn"}),
    "DEPENDENCY_CHAIN": frozenset({"start"}),
    "CROSS_REFERENCE": frozenset({"fqn"}),
    "COMPARISON_QUERY": frozenset({"entity_a", "entity_b"}),
    "MULTI_HOP_QUERY": frozenset({"target", "depth", "allowed_rel_types"}),
    "SINGLE_ENTITY_LOOKUP": frozenset({"fqn"}),
}


def _normalize_cypher(cypher: str) -> str:
    return re.sub(r"\s+", " ", cypher).strip()


_ALLOWED_CYPHER = {
    _normalize_cypher(value)
    for name, value in vars(cypher_templates).items()
    if name.isupper() and isinstance(value, str)
}


def _validate_cypher(cypher: str) -> bool:
    """Accept only an exact query from the project's parameterized allowlist."""
    return _normalize_cypher(cypher) in _ALLOWED_CYPHER


def _pick_template(query: str) -> str | None:
    """Return a template name if the query matches known trigger keywords."""
    normalized_query = query.lower()
    for keywords, template_name in _TEMPLATE_TRIGGERS:
        if any(keyword in normalized_query for keyword in keywords):
            return template_name
    return None


def _get_template(name: str) -> str | None:
    template = getattr(cypher_templates, name, None)
    return template if isinstance(template, str) and _validate_cypher(template) else None


class CypherGenerator:
    def __init__(self, llm: LLMClient, corpus: str) -> None:
        # Keep the constructor contract for callers; query text never becomes executable Cypher.
        self._llm = llm
        self._corpus = corpus

    async def generate(self, query: str, params: dict | None = None) -> tuple[str, dict]:
        """Return a parameterized allowlisted template and its parameters."""
        resolved_params = dict(params or {})
        template_name = _pick_template(query)
        if template_name and _REQUIRED_PARAMS[template_name] <= resolved_params.keys():
            template = _get_template(template_name)
            if template:
                return template, resolved_params

        resolved_params.setdefault("query", query)
        resolved_params.setdefault("limit", 20)
        return cypher_templates.FULLTEXT_ENTITY_SEARCH, resolved_params
