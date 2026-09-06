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
    # A degraded (Neo4j-unreachable) query returns mock=True, and the benchmark
    # runner treats a mock result as a failure. Unlike knowledge_graph/hybrid,
    # lightrag stays out of `all` so a fresh checkout without Neo4j can still
    # run the default benchmark end to end.
    StrategySpec(
        "lightrag",
        "LightRAG",
        "local + global graph",
        default_benchmark=False,
        experimental=True,
    ),
    StrategySpec("hybrid", "Hybrid", "hybrid"),
    StrategySpec("raptor", "RAPTOR", "hierarchical"),
    StrategySpec("pageindex", "PageIndex", "hierarchical"),
    StrategySpec("bm25", "BM25", "lexical", needs_embeddings=False),
    # Both push their filter into the Chroma query instead of cutting a fixed
    # top_k after the fact, so they need embeddings the way naive_vector does.
    # Both stay OUT of the default set. No corpus here carries the
    # `classification`, `tags`, `document_family` or `version` fields they
    # read, and no call site passes a filter or an `as_of` date. So under
    # `--strategies all` today they retrieve exactly what naive_vector
    # retrieves. The report would print three matching rows, and the arena
    # would gain two entrants that are the baseline under another name.
    StrategySpec(
        "metadata_filtered",
        "Metadata Filtered",
        "access-aware dense",
        default_benchmark=False,
        # NOT api_supported. `/chat` carries no access fields, so the API could
        # only construct this with an empty `AccessFilter`, which allows
        # everything. A strategy whose whole purpose is refusing documents must
        # not be reachable through a route that cannot say what to refuse. That
        # is the fail-open shape, and an access rule that fails open is worse
        # than no access rule, because it reads as one.
        api_supported=False,
        experimental=True,
    ),
    StrategySpec(
        "temporal",
        "Temporal",
        "version-aware dense",
        default_benchmark=False,
        experimental=True,
    ),
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
    # HyDE calls the LLM once to rewrite the query, before it calls naive_vector.
    # Multi-Query calls the LLM once per sub-query on top of that. Both add real
    # LLM cost to every question in a benchmark run, so both stay out of the
    # default set on purpose. That is a cost decision, not a sign of unfinished work.
    StrategySpec("hyde", "HyDE", "query rewrite", default_benchmark=False, experimental=True),
    StrategySpec(
        "multi_query",
        "Multi-Query",
        "multi-query fusion",
        default_benchmark=False,
        experimental=True,
    ),
    StrategySpec(
        "late_interaction",
        "Late Interaction",
        "token-level dense",
        default_benchmark=False,
        optional_extra="late-interaction",
        required_modules=("transformers", "torch"),
    ),
    StrategySpec(
        "splade",
        "SPLADE",
        "learned sparse",
        default_benchmark=False,
        optional_extra="splade",
        required_modules=("transformers", "torch"),
        needs_embeddings=False,
    ),
    # The retrieve-judge-refine loop costs several LLM calls per question, so it
    # stays out of the default `all` benchmark: an unbounded loop over 75
    # questions is a bill, not a benchmark run. It carries no recall claim over
    # the other strategies yet, so it is marked experimental like qiss/sqr.
    StrategySpec(
        "agentic",
        "Agentic",
        "iterative",
        default_benchmark=False,
        experimental=True,
    ),
)


def default_strategy_names() -> tuple[str, ...]:
    """Return strategies used by the core ``all`` benchmark."""
    return tuple(spec.name for spec in STRATEGY_CATALOG if spec.default_benchmark)


# Rerank Vector picks its reranker at run time, so one module tuple on the spec
# cannot describe it. Two helpers below and the reference generator all read this
# table, so a fourth backend needs one edit rather than three.
RERANK_BACKENDS: dict[str, tuple[tuple[str, ...], str]] = {
    "bge": (("sentence_transformers",), "pip install 'kb-arena[rerank]'"),
    "cohere": (("cohere",), "pip install cohere"),
    "voyage": (("voyageai",), "pip install voyageai"),
}


def runtime_required_modules(spec: StrategySpec) -> tuple[str, ...]:
    """Return dependencies for the selected runtime backend."""
    if spec.name == "rerank_vector":
        return RERANK_BACKENDS[settings.reranker_backend][0]
    return spec.required_modules


def optional_install_command(spec: StrategySpec) -> str:
    """Return the install command for an unavailable optional strategy."""
    if spec.name == "rerank_vector":
        return RERANK_BACKENDS[settings.reranker_backend][1]
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
