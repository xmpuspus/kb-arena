"""Authoritative metadata for built-in retrieval strategies."""

from __future__ import annotations

import importlib.util
from collections.abc import Collection
from dataclasses import asdict, dataclass

from kb_arena.settings import settings


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Public and runtime properties of one built-in strategy."""

    name: str
    label: str
    architecture: str
    default_benchmark: bool = True
    api_supported: bool = True
    experimental: bool = False
    optional_extra: str | None = None
    required_modules: tuple[str, ...] = ()
    # Whether the strategy calls the embedding provider to answer a query. BM25
    # is lexical and calls nothing, so a bm25-only run needs no API key. The
    # command used to ask for one anyway, which made the documented
    # "needs no API key" run fail on a fresh checkout.
    needs_embeddings: bool = True


STRATEGY_CATALOG: tuple[StrategySpec, ...] = (
    StrategySpec("naive_vector", "Naive Vector", "dense"),
    StrategySpec("contextual_vector", "Contextual Vector", "dense"),
    StrategySpec("qna_pairs", "Q&A Pairs", "generated index"),
    StrategySpec("knowledge_graph", "Knowledge Graph", "graph"),
    StrategySpec("hybrid", "Hybrid", "hybrid"),
    StrategySpec("raptor", "RAPTOR", "hierarchical"),
    StrategySpec("pageindex", "PageIndex", "hierarchical"),
    StrategySpec("bm25", "BM25", "lexical", needs_embeddings=False),
    StrategySpec(
        "rerank_vector",
        "Rerank Vector",
        "reranked dense",
        default_benchmark=False,
        optional_extra="rerank",
        required_modules=("sentence_transformers",),
    ),
    StrategySpec("qiss", "QISS", "quantum-inspired", experimental=True),
    StrategySpec(
        "sqr",
        "SQR",
        "simulated quantum",
        default_benchmark=False,
        experimental=True,
        optional_extra="quantum",
        required_modules=("qiskit", "qiskit_aer", "sklearn"),
    ),
)


def default_strategy_names() -> tuple[str, ...]:
    """Return strategies used by the core ``all`` benchmark."""
    return tuple(spec.name for spec in STRATEGY_CATALOG if spec.default_benchmark)


def runtime_required_modules(spec: StrategySpec) -> tuple[str, ...]:
    """Return dependencies for the selected runtime backend."""
    if spec.name == "rerank_vector":
        return {
            "bge": ("sentence_transformers",),
            "cohere": ("cohere",),
            "voyage": ("voyageai",),
        }[settings.reranker_backend]
    return spec.required_modules


def optional_install_command(spec: StrategySpec) -> str:
    """Return the install command for an unavailable optional strategy."""
    if spec.name == "rerank_vector":
        return {
            "bge": "pip install 'kb-arena[rerank]'",
            "cohere": "pip install cohere",
            "voyage": "pip install voyageai",
        }[settings.reranker_backend]
    return f"pip install 'kb-arena[{spec.optional_extra}]'"


def missing_optional_modules(spec: StrategySpec) -> tuple[str, ...]:
    """Return optional modules that are unavailable in this interpreter."""
    return tuple(
        name for name in runtime_required_modules(spec) if importlib.util.find_spec(name) is None
    )


def public_catalog(loaded_names: Collection[str]) -> list[dict]:
    """Return catalog records with status for the current API process."""
    loaded = set(loaded_names)
    records: list[dict] = []
    for spec in STRATEGY_CATALOG:
        record = asdict(spec)
        record["required_modules"] = runtime_required_modules(spec)
        missing = missing_optional_modules(spec)
        if spec.name in loaded:
            record.update(status="loaded", unavailable_reason=None)
        elif missing:
            record.update(
                status="unavailable",
                unavailable_reason=(
                    f"Run {optional_install_command(spec)} to add {', '.join(missing)}."
                ),
            )
        elif not spec.api_supported:
            record.update(status="unavailable", unavailable_reason="Not supported by the API.")
        else:
            record.update(status="unavailable", unavailable_reason="Not loaded by this runtime.")
        records.append(record)
    return records


def selection_needs_embeddings(selection: str) -> bool:
    """Whether a `--strategies` selection calls the embedding provider.

    `all` and any unknown name count as needing embeddings, because refusing a
    key the run turns out to need fails later and deeper than refusing it here.
    """
    names = [n.strip() for n in (selection or "all").split(",") if n.strip()]
    if not names or "all" in names:
        return True
    by_name = {spec.name: spec for spec in STRATEGY_CATALOG}
    return any(name not in by_name or by_name[name].needs_embeddings for name in names)
