"""Authoritative metadata for built-in retrieval strategies."""

from __future__ import annotations

import importlib.util
from collections.abc import Collection
from dataclasses import asdict, dataclass


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


STRATEGY_CATALOG: tuple[StrategySpec, ...] = (
    StrategySpec("naive_vector", "Naive Vector", "dense"),
    StrategySpec("contextual_vector", "Contextual Vector", "dense"),
    StrategySpec("qna_pairs", "Q&A Pairs", "generated index"),
    StrategySpec("knowledge_graph", "Knowledge Graph", "graph"),
    StrategySpec("hybrid", "Hybrid", "hybrid"),
    StrategySpec("raptor", "RAPTOR", "hierarchical"),
    StrategySpec("pageindex", "PageIndex", "hierarchical"),
    StrategySpec("bm25", "BM25", "lexical"),
    StrategySpec("rerank_vector", "Rerank Vector", "reranked dense"),
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


def missing_optional_modules(spec: StrategySpec) -> tuple[str, ...]:
    """Return optional modules that are unavailable in this interpreter."""
    return tuple(name for name in spec.required_modules if importlib.util.find_spec(name) is None)


def public_catalog(loaded_names: Collection[str]) -> list[dict]:
    """Return catalog records with status for the current API process."""
    loaded = set(loaded_names)
    records: list[dict] = []
    for spec in STRATEGY_CATALOG:
        record = asdict(spec)
        missing = missing_optional_modules(spec)
        if spec.name in loaded:
            record.update(status="loaded", unavailable_reason=None)
        elif missing:
            record.update(
                status="unavailable",
                unavailable_reason=(
                    f"Install kb-arena[{spec.optional_extra}] to add " f"{', '.join(missing)}."
                ),
            )
        elif not spec.api_supported:
            record.update(status="unavailable", unavailable_reason="Not supported by the API.")
        else:
            record.update(status="unavailable", unavailable_reason="Not loaded by this runtime.")
        records.append(record)
    return records
