"""Retrieval strategies — 7 approaches to answering questions from documentation."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from kb_arena.models.document import Document
from kb_arena.settings import settings
from kb_arena.strategies.bm25 import BM25Strategy
from kb_arena.strategies.contextual_vector import ContextualVectorStrategy
from kb_arena.strategies.hybrid import HybridStrategy
from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy
from kb_arena.strategies.naive_vector import NaiveVectorStrategy
from kb_arena.strategies.pageindex import PageIndexStrategy
from kb_arena.strategies.qna_pairs import QnAPairStrategy
from kb_arena.strategies.raptor import RaptorStrategy

logger = logging.getLogger(__name__)
_console = Console()

STRATEGY_REGISTRY: dict[str, type] = {
    "naive_vector": NaiveVectorStrategy,
    "contextual_vector": ContextualVectorStrategy,
    "qna_pairs": QnAPairStrategy,
    "knowledge_graph": KnowledgeGraphStrategy,
    "hybrid": HybridStrategy,
    "raptor": RaptorStrategy,
    "pageindex": PageIndexStrategy,
    "bm25": BM25Strategy,
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
    """Build vector indexes for strategies 1-3 plus RAPTOR.

    Called by the CLI build-vectors command.
    Loads processed JSONL, instantiates each strategy, calls build_index().
    """
    import chromadb

    from kb_arena.llm.client import LLMClient

    chroma = chromadb.PersistentClient(path=settings.chroma_path)
    documents = load_documents(corpus)

    if not documents:
        logger.warning("No documents found for corpus=%s — skipping index build", corpus)
        return

    llm = LLMClient()
    raptor = RaptorStrategy(chroma_client=chroma)
    raptor._llm = llm
    pageindex = PageIndexStrategy()
    pageindex._llm = llm

    # Strategies that need explicit build_index()
    buildable = {
        "naive_vector": NaiveVectorStrategy(chroma_client=chroma),
        "contextual_vector": ContextualVectorStrategy(chroma_client=chroma),
        "qna_pairs": QnAPairStrategy(chroma_client=chroma),
        "raptor": raptor,
        "pageindex": pageindex,
        "bm25": BM25Strategy(),
    }

    targets = (
        buildable
        if strategy == "all"
        else {name: inst for name, inst in buildable.items() if name == strategy}
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} strategies"),
        console=_console,
    ) as progress:
        task = progress.add_task("Building vector indexes", total=len(targets))
        for name, inst in targets.items():
            progress.update(task, description=f"Building [bold]{name}[/bold]")
            await inst.build_index(documents)
            progress.advance(task)

    _console.print(
        f"[green]Done.[/green] Built {len(targets)} vector index(es) "
        f"from {len(documents)} documents"
    )


def register_plugin_strategy(module_path: str) -> None:
    """Import a user module and register its Strategy subclass.

    Usage: --strategy-module my_package.my_strategy
    The module must contain exactly one class that subclasses Strategy.
    """
    import importlib

    from kb_arena.strategies.base import Strategy as _Base

    mod = importlib.import_module(module_path)
    candidates = [
        obj
        for name in dir(mod)
        if not name.startswith("_")
        for obj in [getattr(mod, name)]
        if isinstance(obj, type) and issubclass(obj, _Base) and obj is not _Base
    ]
    if not candidates:
        raise ValueError(f"No Strategy subclass found in {module_path}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple Strategy subclasses in {module_path}: "
            f"{[c.__name__ for c in candidates]}. Export exactly one."
        )

    cls = candidates[0]
    name = getattr(cls, "name", module_path.split(".")[-1])
    STRATEGY_REGISTRY[name] = cls
    logger.info("Registered plugin strategy: %s from %s", name, module_path)


def get_strategy(name: str):
    """Instantiate a strategy by name. Used by the benchmark runner."""
    import chromadb
    from neo4j import AsyncGraphDatabase

    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")

    # No-dependency strategies
    if name in ("pageindex", "bm25"):
        return cls()

    # Vector-backed strategies need a ChromaDB client
    if name in ("naive_vector", "contextual_vector", "qna_pairs", "raptor"):
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
    "RaptorStrategy",
    "PageIndexStrategy",
    "BM25Strategy",
    "build_vector_indexes",
    "load_documents",
]
