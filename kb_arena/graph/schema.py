"""Universal node and relationship type enums, with validation helpers.

The schema is domain-agnostic — 5 node types and 7 relationship types that
cover the structure of any documentation corpus. AWS services map to Topic,
resources to Component, policies to Constraint, etc.
"""

from __future__ import annotations

from enum import StrEnum


class NodeType(StrEnum):
    TOPIC = "Topic"  # Primary subject (AWS Lambda, a framework, a concept)
    COMPONENT = "Component"  # Sub-element or building block (Lambda layer, subnet)
    PROCESS = "Process"  # Procedure, workflow, or operation (deploy, configure)
    CONFIG = "Config"  # Setting, option, parameter (timeout, retention period)
    CONSTRAINT = "Constraint"  # Limit, requirement, or prerequisite (quotas, min versions)


class RelType(StrEnum):
    DEPENDS_ON = "DEPENDS_ON"
    CONTAINS = "CONTAINS"
    CONNECTS_TO = "CONNECTS_TO"
    TRIGGERS = "TRIGGERS"
    CONFIGURES = "CONFIGURES"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    EXTENDS = "EXTENDS"


def get_schema(corpus: str) -> tuple[type, type]:
    """Return (NodeType enum, RelType enum). Same universal schema for all corpora."""
    return (NodeType, RelType)


def valid_node_type(corpus: str, type_str: str) -> bool:
    """True if type_str is a valid node type for this corpus."""
    node_enum, _ = get_schema(corpus)
    return type_str in {e.value for e in node_enum}


def valid_rel_type(corpus: str, type_str: str) -> bool:
    """True if type_str is a valid relationship type for this corpus."""
    _, rel_enum = get_schema(corpus)
    return type_str in {e.value for e in rel_enum}


def node_type_values(corpus: str) -> list[str]:
    node_enum, _ = get_schema(corpus)
    return [e.value for e in node_enum]


def rel_type_values(corpus: str) -> list[str]:
    _, rel_enum = get_schema(corpus)
    return [e.value for e in rel_enum]
