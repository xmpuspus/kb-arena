"""Node and relationship type enums per corpus, with validation helpers."""

from __future__ import annotations

from enum import StrEnum

# ── AWS documentation schema ──────────────────────────────────────────────────


class NodeType(StrEnum):
    SERVICE = "Service"  # AWS service (Lambda, S3, EC2)
    RESOURCE = "Resource"  # Infrastructure resource (VPC, Subnet, Security Group)
    POLICY = "Policy"  # IAM policies, bucket policies, resource policies
    FEATURE = "Feature"  # Service feature (versioning, encryption, auto-scaling)
    CONFIGURATION = "Configuration"  # Config items (timeout, memory, runtime)
    LIMIT = "Limit"  # Service limits/quotas
    API_ACTION = "APIAction"  # AWS API actions (PutObject, InvokeFunction)
    ARN_PATTERN = "ARNPattern"  # ARN format patterns


class RelType(StrEnum):
    DEPENDS_ON = "DEPENDS_ON"
    INVOKES = "INVOKES"
    CONNECTS_TO = "CONNECTS_TO"
    ASSUMES = "ASSUMES"  # Role assumption
    CONTAINS = "CONTAINS"
    PROTECTS = "PROTECTS"  # Security group -> resource
    ROUTES_TO = "ROUTES_TO"
    LOGS_TO = "LOGS_TO"
    TRIGGERS = "TRIGGERS"
    DEPLOYED_IN = "DEPLOYED_IN"
    MANAGES = "MANAGES"
    READS_FROM = "READS_FROM"
    WRITES_TO = "WRITES_TO"


# ── Corpus dispatch ───────────────────────────────────────────────────────────

_CORPUS_SCHEMA: dict[str, tuple[type, type]] = {
    "aws-compute": (NodeType, RelType),
    "aws-storage": (NodeType, RelType),
    "aws-networking": (NodeType, RelType),
}


def get_schema(corpus: str) -> tuple[type, type]:
    """Return (NodeType enum, RelType enum) for the given corpus."""
    if corpus not in _CORPUS_SCHEMA:
        raise ValueError(f"Unknown corpus '{corpus}'. Valid: {list(_CORPUS_SCHEMA)}")
    return _CORPUS_SCHEMA[corpus]


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
