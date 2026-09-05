"""Built-in retrieval strategies and index helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from kb_arena.models.document import Document
from kb_arena.settings import settings
from kb_arena.strategies.agentic import AgenticStrategy
from kb_arena.strategies.bm25 import BM25Strategy
from kb_arena.strategies.catalog import (
    STRATEGY_CATALOG,
    missing_optional_modules,
    optional_install_command,
)
from kb_arena.strategies.contextual_vector import ContextualVectorStrategy
from kb_arena.strategies.hybrid import HybridStrategy
from kb_arena.strategies.hyde import HydeStrategy
from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy
from kb_arena.strategies.late_interaction import LateInteractionStrategy
from kb_arena.strategies.lightrag import LightRAGStrategy
from kb_arena.strategies.metadata_filtered import MetadataFilteredStrategy
from kb_arena.strategies.multi_query import MultiQueryStrategy
from kb_arena.strategies.naive_vector import NaiveVectorStrategy
from kb_arena.strategies.pageindex import PageIndexStrategy
from kb_arena.strategies.qna_pairs import QnAPairStrategy
from kb_arena.strategies.quantum.qiss import QISSStrategy
from kb_arena.strategies.quantum.sqr import SQRStrategy
from kb_arena.strategies.raptor import RaptorStrategy
from kb_arena.strategies.rerank_vector import RerankVectorStrategy
from kb_arena.strategies.splade import SPLADEStrategy
from kb_arena.strategies.temporal import TemporalStrategy

logger = logging.getLogger(__name__)
_console = Console()

STRATEGY_REGISTRY: dict[str, type] = {
    "naive_vector": NaiveVectorStrategy,
    "contextual_vector": ContextualVectorStrategy,
    "qna_pairs": QnAPairStrategy,
    "knowledge_graph": KnowledgeGraphStrategy,
    "lightrag": LightRAGStrategy,
    "hybrid": HybridStrategy,
    "raptor": RaptorStrategy,
    "pageindex": PageIndexStrategy,
    "bm25": BM25Strategy,
    "metadata_filtered": MetadataFilteredStrategy,
    "temporal": TemporalStrategy,
    "rerank_vector": RerankVectorStrategy,
    "qiss": QISSStrategy,
    "sqr": SQRStrategy,
    "hyde": HydeStrategy,
    "multi_query": MultiQueryStrategy,
    "late_interaction": LateInteractionStrategy,
    "splade": SPLADEStrategy,
    "agentic": AgenticStrategy,
}

# Optional-dependency strategies: name -> (modules required, extra name).
# get_strategy raises a clear install hint when any module is missing, so the
# benchmark/retriever-lab loaders skip them instead of emitting empty traces.
_OPTIONAL_DEP_STRATEGIES = {
    spec.name: spec for spec in STRATEGY_CATALOG if spec.optional_extra is not None
}


def load_documents(corpus: str, *, strict: bool = False) -> list[Document]:
    """Load processed JSONL documents, optionally rejecting the first malformed row."""
    base = Path(settings.datasets_path)
    if corpus == "all":
        paths = list(base.glob("*/processed/*.jsonl"))
    else:
        paths = list((base / corpus / "processed").glob("*.jsonl"))

    documents: list[Document] = []
    for path in paths:
        with open(path) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        documents.append(Document.model_validate_json(line))
                    except Exception as exc:
                        if strict:
                            raise ValueError(
                                f"Malformed processed document at {path}:{line_number}"
                            ) from exc
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

    documents = load_documents(corpus, strict=True)

    if not documents:
        raise ValueError(f"No processed documents found for corpus={corpus}")

    buildable_names = (
        "naive_vector",
        "contextual_vector",
        "qna_pairs",
        "raptor",
        "pageindex",
        "bm25",
        "metadata_filtered",
        "temporal",
        "qiss",
        "sqr",
        "hyde",
        "multi_query",
        "splade",
    )
    if strategy != "all" and strategy not in buildable_names:
        raise ValueError(f"Unknown build strategy: {strategy}")
    # splade builds its own term-weight index and needs the optional [splade]
    # extra to do it, unlike qiss/sqr, which only rebuild the naive_vector
    # collection they wrap. So "all" must not fail on a plain install, splade
    # is a build target only when a caller names it explicitly.
    all_names = tuple(name for name in buildable_names if name != "splade")
    target_names = all_names if strategy == "all" else (strategy,)

    chroma_strategies = {
        "naive_vector",
        "contextual_vector",
        "qna_pairs",
        "raptor",
        "metadata_filtered",
        "temporal",
        "qiss",
        "sqr",
        "hyde",
        "multi_query",
    }
    chroma = (
        chromadb.PersistentClient(path=settings.chroma_path)
        if set(target_names) & chroma_strategies
        else None
    )
    llm_build_strategies = {"contextual_vector", "qna_pairs", "raptor", "pageindex"}
    llm = LLMClient() if set(target_names) & llm_build_strategies else None

    def _contextual():
        instance = ContextualVectorStrategy(chroma_client=chroma)
        instance._llm = llm
        return instance

    def _raptor():
        instance = RaptorStrategy(chroma_client=chroma)
        instance._llm = llm
        return instance

    def _pageindex():
        instance = PageIndexStrategy()
        instance._llm = llm
        return instance

    # qiss/sqr/hyde/multi_query build through the naive_vector collection they
    # wrap, so building them is idempotent with the dense index.
    factories = {
        "naive_vector": lambda: NaiveVectorStrategy(chroma_client=chroma),
        "contextual_vector": _contextual,
        "qna_pairs": lambda: QnAPairStrategy(chroma_client=chroma, llm_client=llm),
        "raptor": _raptor,
        "pageindex": _pageindex,
        "bm25": BM25Strategy,
        "metadata_filtered": lambda: MetadataFilteredStrategy(chroma_client=chroma),
        "temporal": lambda: TemporalStrategy(chroma_client=chroma),
        "qiss": lambda: QISSStrategy(chroma_client=chroma),
        "sqr": lambda: SQRStrategy(chroma_client=chroma),
        "hyde": lambda: HydeStrategy(chroma_client=chroma),
        "multi_query": lambda: MultiQueryStrategy(chroma_client=chroma),
        "splade": SPLADEStrategy,
    }
    targets = {name: factories[name]() for name in target_names}

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


# Every plugin module this process loaded, in the order it loaded them. A run
# using a plugin strategy cannot be repeated without the same import, so the
# recorded command has to name it. The path is an importable module name, not a
# file on one machine, so a reader who installs that package can replay it.
LOADED_PLUGIN_MODULES: list[str] = []


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
    if module_path not in LOADED_PLUGIN_MODULES:
        LOADED_PLUGIN_MODULES.append(module_path)
    logger.info("Registered plugin strategy: %s from %s", name, module_path)


def get_strategy(name: str):
    """Instantiate a strategy by name. Used by the benchmark runner."""
    import chromadb
    from neo4j import AsyncGraphDatabase

    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")

    # Give optional-dependency strategies such as sqr a clear install hint.
    # when the extra is absent, so loaders skip them rather than running empty.
    if name in _OPTIONAL_DEP_STRATEGIES:
        spec = _OPTIONAL_DEP_STRATEGIES[name]
        missing = missing_optional_modules(spec)
        if missing:
            raise ImportError(
                f"Strategy '{name}' is missing an optional dependency "
                f"({', '.join(missing)} not installed). "
                f"Install with: {optional_install_command(spec)}"
            )

    # No-dependency strategies. splade builds its own term-weight index, so it
    # needs no ChromaDB client, the same as bm25.
    if name in ("pageindex", "bm25", "splade"):
        return cls()

    # Vector-backed strategies need a ChromaDB client. qiss/sqr/hyde/multi_query
    # wrap naive_vector (like rerank_vector) and query the same Chroma index.
    # Vector-backed strategies need a ChromaDB client. qiss/sqr/late_interaction
    # wrap naive_vector for coarse retrieval (like rerank_vector) and rerank the
    # same Chroma index.
    # Vector-backed strategies need a ChromaDB client. qiss/sqr/agentic wrap
    # naive_vector for coarse retrieval (like rerank_vector) and read or rerank
    # the same Chroma index.
    if name in (
        "naive_vector",
        "contextual_vector",
        "qna_pairs",
        "raptor",
        "metadata_filtered",
        "temporal",
        "rerank_vector",
        "qiss",
        "sqr",
        "hyde",
        "multi_query",
        "late_interaction",
        "agentic",
    ):
        chroma = chromadb.PersistentClient(path=settings.chroma_path)
        return cls(chroma_client=chroma)

    # Graph-backed strategies need an async Neo4j driver
    if name in ("knowledge_graph", "lightrag"):
        try:
            driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            return cls(neo4j_driver=driver)
        except Exception as e:
            logger.warning("Neo4j not available for %s: %s; using mock fallback", name, e)
            return cls()

    # Hybrid needs both, plus the IntentRouter for the advertised three-stage classification.
    if name == "hybrid":
        from kb_arena.chatbot.router import IntentRouter
        from kb_arena.llm.client import LLMClient

        chroma = chromadb.PersistentClient(path=settings.chroma_path)
        llm = LLMClient()
        router = IntentRouter(llm=llm)
        try:
            driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            return cls(chroma_client=chroma, neo4j_driver=driver, router=router, llm=llm)
        except Exception:
            return cls(chroma_client=chroma, router=router, llm=llm)

    return cls()


__all__ = [
    "LOADED_PLUGIN_MODULES",
    "STRATEGY_REGISTRY",
    "STRATEGY_CATALOG",
    "get_strategy",
    "NaiveVectorStrategy",
    "ContextualVectorStrategy",
    "QnAPairStrategy",
    "KnowledgeGraphStrategy",
    "LightRAGStrategy",
    "HybridStrategy",
    "RaptorStrategy",
    "PageIndexStrategy",
    "BM25Strategy",
    "MetadataFilteredStrategy",
    "TemporalStrategy",
    "RerankVectorStrategy",
    "QISSStrategy",
    "SQRStrategy",
    "HydeStrategy",
    "MultiQueryStrategy",
    "LateInteractionStrategy",
    "SPLADEStrategy",
    "AgenticStrategy",
    "build_vector_indexes",
    "load_documents",
]
