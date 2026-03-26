"""Custom exception hierarchy for KB Arena."""


class KBArenaError(Exception):
    """Base exception for all KB Arena errors."""


class IngestError(KBArenaError):
    """Error during document ingestion or parsing."""


class GraphError(KBArenaError):
    """Error during Neo4j graph operations."""


class StrategyError(KBArenaError):
    """Error during strategy query or index building."""


class EvaluationError(KBArenaError):
    """Error during benchmark evaluation."""


class LLMError(KBArenaError):
    """Error during LLM API calls."""
