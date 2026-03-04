"""Retrieval strategies — 5 approaches to answering questions from documentation."""

from __future__ import annotations

import logging
from pathlib import Path

from kb_arena.models.document import Document
from kb_arena.settings import settings
from kb_arena.strategies.contextual_vector import ContextualVectorStrategy
from kb_arena.strategies.hybrid import HybridStrategy
from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy
from kb_arena.strategies.naive_vector import NaiveVectorStrategy
from kb_arena.strategies.qna_pairs import QnAPairStrategy

logger = logging.getLogger(__name__)

STRATEGY_REGISTRY: dict[str, type] = {
    "naive_vector": NaiveVectorStrategy,
    "contextual_vector": ContextualVectorStrategy,
    "qna_pairs": QnAPairStrategy,
    "knowledge_graph": KnowledgeGraphStrategy,
    "hybrid": HybridStrategy,
}


def load_documents(corpus: str) -> list[Document]:
    """Load processed JSONL documents for a corpus."""
    base = Path(settings.datasets_path)
    if corpus == "all":
        paths = list(base.glob("*/processed/*.jsonl"))
    else:
        paths = list((base / corpus / "processed").glob("*.jsonl"))

    documents: list[Document] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        documents.append(Document.model_validate_json(line))
                    except Exception as exc:
                        logger.warning("Skipping malformed JSONL line in %s: %s", path, exc)

    logger.info("Loaded %d documents for corpus=%s", len(documents), corpus)
    return documents


async def build_vector_indexes(corpus: str = "all", strategy: str = "all") -> None:
    """Build vector indexes for strategies 1-3.

    Called by the CLI build-vectors command.
    Loads processed JSONL, instantiates each strategy, calls build_index().
    """
    import chromadb

    chroma = chromadb.PersistentClient(path=settings.chroma_path)
    documents = load_documents(corpus)

    if not documents:
        logger.warning("No documents found for corpus=%s — skipping index build", corpus)
        return

    # Only vector-backed strategies need explicit build_index()
    buildable = {
        "naive_vector": NaiveVectorStrategy(chroma_client=chroma),
        "contextual_vector": ContextualVectorStrategy(chroma_client=chroma),
        "qna_pairs": QnAPairStrategy(chroma_client=chroma),
    }

    targets = (
        buildable
        if strategy == "all"
        else {name: inst for name, inst in buildable.items() if name == strategy}
    )

    for name, inst in targets.items():
        logger.info("Building index: %s (%d documents)", name, len(documents))
        await inst.build_index(documents)
        logger.info("Done: %s", name)


def get_strategy(name: str):
    """Instantiate a strategy by name. Used by the benchmark runner."""
    import chromadb
    from neo4j import AsyncGraphDatabase

    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")

    # Vector-backed strategies need a ChromaDB client
    if name in ("naive_vector", "contextual_vector", "qna_pairs"):
        chroma = chromadb.PersistentClient(path=settings.chroma_path)
        return cls(chroma_client=chroma)

    # Graph-backed strategies need an async Neo4j driver
    if name == "knowledge_graph":
        try:
            driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            return cls(neo4j_driver=driver)
        except Exception as e:
            logger.warning("Neo4j not available for %s: %s — using mock fallback", name, e)
            return cls()

    # Hybrid needs both
    if name == "hybrid":
        chroma = chromadb.PersistentClient(path=settings.chroma_path)
        try:
            driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            return cls(chroma_client=chroma, neo4j_driver=driver)
        except Exception:
            return cls(chroma_client=chroma)

    return cls()


__all__ = [
    "STRATEGY_REGISTRY",
    "get_strategy",
    "NaiveVectorStrategy",
    "ContextualVectorStrategy",
    "QnAPairStrategy",
    "KnowledgeGraphStrategy",
    "HybridStrategy",
    "build_vector_indexes",
    "load_documents",
]
