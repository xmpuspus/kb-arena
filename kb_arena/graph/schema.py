"""Node and relationship type enums per corpus, with validation helpers."""

from __future__ import annotations

from enum import Enum

# ── Python stdlib schema ──────────────────────────────────────────────────────

class NodeType(str, Enum):
    CONCEPT = "Concept"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    PARAMETER = "Parameter"
    RETURN_TYPE = "ReturnType"
    EXCEPTION = "Exception"
    DEPRECATION = "Deprecation"
    VERSION = "Version"
    EXAMPLE = "Example"


class RelType(str, Enum):
    CONTAINS = "CONTAINS"           # Module -> Class, Class -> Function
    REQUIRES = "REQUIRES"           # Function -> Parameter
    RETURNS = "RETURNS"             # Function -> ReturnType
    RAISES = "RAISES"               # Function -> Exception
    DEPRECATED_BY = "DEPRECATED_BY"  # old -> new
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    REFERENCES = "REFERENCES"       # cross-module reference
    INHERITS = "INHERITS"           # Class -> Class
    IMPLEMENTS = "IMPLEMENTS"       # Class -> Concept (e.g., "iterator protocol")
    EXAMPLE_OF = "EXAMPLE_OF"       # Example -> Function/Class


# ── Kubernetes schema ─────────────────────────────────────────────────────────

class K8sNodeType(str, Enum):
    RESOURCE = "Resource"
    FIELD = "Field"
    API_GROUP = "APIGroup"
    CONTROLLER = "Controller"
    CONCEPT = "Concept"
    EXAMPLE = "Example"
    VERSION = "Version"


class K8sRelType(str, Enum):
    CONTAINS = "CONTAINS"
    BELONGS_TO = "BELONGS_TO"       # Resource -> APIGroup
    MANAGES = "MANAGES"             # Controller -> Resource
    REFERENCES = "REFERENCES"
    REQUIRES = "REQUIRES"
    EXAMPLE_OF = "EXAMPLE_OF"
    SUPERSEDES = "SUPERSEDES"       # newer Version -> older Version
    RELATED_TO = "RELATED_TO"


# ── SEC EDGAR schema ──────────────────────────────────────────────────────────

class SecNodeType(str, Enum):
    COMPANY = "Company"
    EXECUTIVE = "Executive"
    BOARD_MEMBER = "BoardMember"
    SUBSIDIARY = "Subsidiary"
    RISK_FACTOR = "RiskFactor"
    FINANCIAL_METRIC = "FinancialMetric"
    LEGAL_PROCEEDING = "LegalProceeding"
    SEGMENT = "Segment"


class SecRelType(str, Enum):
    EMPLOYS = "EMPLOYS"             # Company -> Executive
    HAS_BOARD_MEMBER = "HAS_BOARD_MEMBER"
    OWNS = "OWNS"                   # Company -> Subsidiary
    HAS_RISK = "HAS_RISK"
    REPORTS_METRIC = "REPORTS_METRIC"
    INVOLVED_IN = "INVOLVED_IN"     # Company -> LegalProceeding
    OPERATES_SEGMENT = "OPERATES_SEGMENT"
    REFERENCES = "REFERENCES"


# ── Corpus dispatch ───────────────────────────────────────────────────────────

_CORPUS_SCHEMA: dict[str, tuple[type, type]] = {
    "python-stdlib": (NodeType, RelType),
    "kubernetes": (K8sNodeType, K8sRelType),
    "sec-edgar": (SecNodeType, SecRelType),
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
