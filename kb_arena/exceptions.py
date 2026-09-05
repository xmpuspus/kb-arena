"""Custom exception hierarchy for KB Arena."""


class KBArenaError(Exception):
    """Base exception for all KB Arena errors."""


class IngestError(KBArenaError):
    """Error during document ingestion or parsing."""


class GraphError(KBArenaError):
    """Error during Neo4j graph operations."""


class StrategyError(KBArenaError):
    """Error during strategy query or index building."""


class RerankerError(StrategyError):
    """A reranker backend failed before it could produce valid scores."""


class RetrieverContractError(StrategyError):
    """A bring-your-own-retriever endpoint broke the request/response contract.

    Covers a non-2xx status, a non-JSON body, a body that fails schema
    validation, a per-request timeout, and an exhausted time budget. A
    response that does not match the schema is always an error, never a
    partial result.
    """


class ArenaError(KBArenaError):
    """A match cannot be rated, so no vote may follow it."""


class EvaluationError(KBArenaError):
    """Error during benchmark evaluation."""


class LLMError(KBArenaError):
    """Error during LLM API calls."""
