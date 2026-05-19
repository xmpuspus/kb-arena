"""Automated retrieval-strategy hyperparameter search.

`kb-arena optimize` sweeps chunk size, top-k, embedding provider and reranker
backend per strategy, scores each configuration on a retrieval IR metric
(retrieval-only, ~10x cheaper than the answer benchmark), and reports the tuned
optimum and its delta versus the current defaults.

This module is split into a pure search-space core (fully unit-tested, no I/O)
and an async orchestrator (`run_optimize`) that applies per-trial overrides,
rebuilds indexes only when a rebuild-affecting dimension changed, and runs the
retrieval-only loop.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from kb_arena.benchmark.ir_metrics import compute_all
from kb_arena.benchmark.questions import load_questions
from kb_arena.settings import settings
from kb_arena.strategies import load_documents

log = logging.getLogger(__name__)

# Which dimensions each strategy actually consumes. Sweeping a dimension a
# strategy ignores just burns wall-clock on duplicate trials.
CHUNKING_STRATEGIES = frozenset({"naive_vector", "contextual_vector", "raptor"})
EMBEDDING_STRATEGIES = frozenset(
    {"naive_vector", "contextual_vector", "qna_pairs", "raptor", "rerank_vector"}
)
RERANKER_STRATEGIES = frozenset({"rerank_vector"})

# A change in either of these requires rebuilding the strategy's index;
# top_k and reranker_backend are query-time only (free to vary).
REBUILD_DIMS = frozenset({"chunk_tokens", "embedding_provider"})

_METRIC_FIELDS = {
    "ndcg": "mean_ndcg_at_k",
    "recall": "mean_recall_at_k",
    "precision": "mean_precision_at_k",
    "mrr": "mean_mrr",
    "hit": "mean_hit_at_k",
}


class TrialConfig(BaseModel):
    """One point in the search space for a single strategy."""

    strategy: str
    top_k: int
    chunk_tokens: int | None = None
    embedding_provider: str | None = None
    reranker_backend: str | None = None

    model_config = {"frozen": True}


class OptimizeResult(BaseModel):
    """Per-strategy outcome: best config and its lift over the baseline."""

    strategy: str
    metric: str = "ndcg"
    best_config: TrialConfig
    best_score: float
    baseline_config: TrialConfig
    baseline_score: float
    n_trials: int = 0
    scored: list[tuple[TrialConfig, float]] = Field(default_factory=list)

    @property
    def delta(self) -> float:
        return round(self.best_score - self.baseline_score, 6)

    @property
    def improved(self) -> bool:
        return self.best_score > self.baseline_score


def applicable_dims(strategy: str) -> set[str]:
    """Dimensions worth sweeping for `strategy`. top_k always applies."""
    dims = {"top_k"}
    if strategy in CHUNKING_STRATEGIES:
        dims.add("chunk_tokens")
    if strategy in EMBEDDING_STRATEGIES:
        dims.add("embedding_provider")
    if strategy in RERANKER_STRATEGIES:
        dims.add("reranker_backend")
    return dims


def _axis(values: list, baseline_value, active: bool) -> list:
    """A swept axis: the requested values if this dim is active and non-empty,
    else a single-element axis pinned to the baseline."""
    if active and values:
        vals = list(dict.fromkeys(values))  # dedupe, keep order
        if baseline_value not in vals:
            vals = [baseline_value, *vals]
        return vals
    return [baseline_value]


def build_trials(
    strategy: str,
    *,
    top_ks: list[int],
    chunk_sizes: list[int],
    embedding_providers: list[str],
    reranker_backends: list[str],
    baseline: TrialConfig,
    method: str = "grid",
    max_trials: int = 0,
    seed: int = 0,
) -> list[TrialConfig]:
    """Enumerate trial configs for one strategy.

    Non-applicable dimensions collapse to the baseline value. The baseline
    config is always the first trial so the reported delta is honest. `grid`
    is the full cartesian product; `random` samples `max_trials` distinct
    configs (baseline always kept) with a seeded RNG.
    """
    if method not in ("grid", "random"):
        raise ValueError(f"Unknown method {method!r}. Use 'grid' or 'random'.")

    dims = applicable_dims(strategy)
    top_k_axis = _axis(top_ks, baseline.top_k, "top_k" in dims)
    chunk_axis = _axis(chunk_sizes, baseline.chunk_tokens, "chunk_tokens" in dims)
    emb_axis = _axis(embedding_providers, baseline.embedding_provider, "embedding_provider" in dims)
    rer_axis = _axis(reranker_backends, baseline.reranker_backend, "reranker_backend" in dims)

    combos = [
        TrialConfig(
            strategy=strategy,
            top_k=tk,
            chunk_tokens=ck,
            embedding_provider=ep,
            reranker_backend=rb,
        )
        for tk, ck, ep, rb in itertools.product(top_k_axis, chunk_axis, emb_axis, rer_axis)
    ]

    # Dedupe while preserving order, baseline first.
    seen: set[TrialConfig] = set()
    ordered: list[TrialConfig] = [baseline]
    seen.add(baseline)
    for c in combos:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    if method == "random" and max_trials and len(ordered) > max_trials:
        rest = ordered[1:]
        rng = random.Random(seed)
        rng.shuffle(rest)
        ordered = [baseline, *rest[: max_trials - 1]]
    elif max_trials and len(ordered) > max_trials:
        ordered = ordered[:max_trials]

    return ordered


def needs_rebuild(prev: TrialConfig | None, cfg: TrialConfig) -> bool:
    """True if going from `prev` to `cfg` changes a rebuild-affecting dim."""
    if prev is None:
        return True
    return any(getattr(prev, d) != getattr(cfg, d) for d in REBUILD_DIMS)


def select_best(
    strategy: str,
    scored: list[tuple[TrialConfig, float]],
    baseline: TrialConfig,
    metric: str = "ndcg",
) -> OptimizeResult:
    """Pick the highest-scoring config; delta is measured against the baseline.

    Ties keep the earliest config (baseline-first ordering means a tie with the
    baseline reports zero improvement, never a spurious 'win')."""
    if not scored:
        raise ValueError("no scored trials")
    by_cfg = {cfg: score for cfg, score in scored}
    baseline_score = by_cfg.get(baseline, scored[0][1])

    best_cfg, best_score = scored[0]
    for cfg, score in scored:
        if score > best_score:
            best_cfg, best_score = cfg, score

    return OptimizeResult(
        strategy=strategy,
        metric=metric,
        best_config=best_cfg,
        best_score=best_score,
        baseline_config=baseline,
        baseline_score=baseline_score,
        n_trials=len(scored),
        scored=scored,
    )


# ── Orchestration ─────────────────────────────────────────────────────────────


def baseline_config(strategy: str) -> TrialConfig:
    """The current-settings configuration — what the user gets today."""
    return TrialConfig(
        strategy=strategy,
        top_k=5,
        chunk_tokens=settings.chunk_tokens,
        embedding_provider=settings.embedding_provider,
        reranker_backend=settings.reranker_backend,
    )


def _trials_for(
    strategy: str,
    *,
    top_ks,
    chunk_sizes,
    embedding_providers,
    reranker_backends,
    method,
    max_trials,
    seed,
) -> list[TrialConfig]:
    return build_trials(
        strategy,
        top_ks=top_ks or [5],
        chunk_sizes=chunk_sizes,
        embedding_providers=embedding_providers,
        reranker_backends=reranker_backends,
        baseline=baseline_config(strategy),
        method=method,
        max_trials=max_trials,
        seed=seed,
    )


def plan_optimize(
    strategies: list[str],
    *,
    top_ks,
    chunk_sizes,
    embedding_providers,
    reranker_backends,
    method: str = "grid",
    max_trials: int = 0,
    seed: int = 0,
) -> list[dict]:
    """Cost preview: trial + rebuild counts per strategy. No execution."""
    plan: list[dict] = []
    for s in strategies:
        trials = _trials_for(
            s,
            top_ks=top_ks,
            chunk_sizes=chunk_sizes,
            embedding_providers=embedding_providers,
            reranker_backends=reranker_backends,
            method=method,
            max_trials=max_trials,
            seed=seed,
        )
        # A trial needs a rebuild iff its rebuild dims differ from the
        # persistent-index (baseline) config; the persistent index already
        # matches the baseline, so trials that match it on chunk/embedding
        # reuse it for free.
        base = baseline_config(s)
        rebuilds = sum(1 for t in trials if needs_rebuild(t, base))
        plan.append(
            {
                "strategy": s,
                "n_trials": len(trials),
                "n_rebuilds": rebuilds,
                "dims": sorted(applicable_dims(s)),
            }
        )
    return plan


class _ApplyOverrides:
    """Patch the Settings singleton for one trial; restore on exit.

    chunk/embedding changes also redirect chroma_path to an isolated temp dir
    so a sweep never corrupts the user's persistent indexes.
    """

    def __init__(self, cfg: TrialConfig, isolate_chroma: bool):
        self._cfg = cfg
        self._isolate = isolate_chroma
        self._saved: dict = {}
        self._tmp: tempfile.TemporaryDirectory | None = None

    def __enter__(self):
        for attr, val in (
            ("chunk_tokens", self._cfg.chunk_tokens),
            ("embedding_provider", self._cfg.embedding_provider),
            ("reranker_backend", self._cfg.reranker_backend),
        ):
            if val is not None:
                self._saved[attr] = getattr(settings, attr)
                setattr(settings, attr, val)
        if self._isolate:
            self._saved["chroma_path"] = settings.chroma_path
            self._tmp = tempfile.TemporaryDirectory(prefix="kb-arena-opt-")
            settings.chroma_path = self._tmp.name
        return self

    def __exit__(self, *exc):
        for attr, val in self._saved.items():
            setattr(settings, attr, val)
        if self._tmp is not None:
            self._tmp.cleanup()
        return False


async def _score_trial(strategy, cfg, documents, questions, metric, baseline) -> float:
    """Mean IR metric for one (strategy, config) over the corpus questions.

    Retrieval-only (LLM generation stubbed) so a full sweep stays ~10x cheaper
    than the answer benchmark. Rebuilds the index only when the trial's
    rebuild dims differ from the baseline (the persistent-index config).
    """
    from kb_arena.benchmark.retriever_lab import _PatchLLMClient, _retrieve_only
    from kb_arena.strategies import get_strategy

    rebuild = needs_rebuild(cfg, baseline)
    field = _METRIC_FIELDS.get(metric, "mean_ndcg_at_k").replace("mean_", "")

    with _ApplyOverrides(cfg, isolate_chroma=rebuild):
        inst = get_strategy(strategy)
        if rebuild and hasattr(inst, "build_index"):
            try:
                await inst.build_index(documents)
            except Exception as exc:  # noqa: BLE001 — a dead config scores 0, not crash
                log.warning("optimize: build_index failed for %s %s: %s", strategy, cfg, exc)
                return 0.0
        scores: list[float] = []
        with _PatchLLMClient():
            for q in questions:
                trace = await _retrieve_only(inst, q.question, cfg.top_k)
                m = compute_all(
                    retrieved=trace.retrieved,
                    expected_ids=set(getattr(q, "expected_chunks", []) or []),
                    k=cfg.top_k,
                    expected_doc_ids=set(
                        getattr(getattr(q, "ground_truth", None), "source_refs", []) or []
                    ),
                )
                scores.append(getattr(m, field))
    return sum(scores) / len(scores) if scores else 0.0


def _resolve_strategies(strategies_filter: str) -> list[str]:
    from kb_arena.benchmark.runner import STRATEGY_NAMES

    if strategies_filter == "all":
        return list(STRATEGY_NAMES)
    return [s.strip() for s in strategies_filter.split(",") if s.strip()]


async def run_optimize(
    corpus: str,
    strategies_filter: str = "all",
    *,
    top_ks: list[int] | None = None,
    chunk_sizes: list[int] | None = None,
    embedding_providers: list[str] | None = None,
    reranker_backends: list[str] | None = None,
    metric: str = "ndcg",
    method: str = "grid",
    max_trials: int = 0,
    seed: int = 0,
    dry_run: bool = False,
    out_dir: str | None = None,
) -> int:
    """Sweep, score, report. Returns 0 on success, 1 on hard failure."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    top_ks = top_ks or [5]
    chunk_sizes = chunk_sizes or []
    embedding_providers = embedding_providers or []
    reranker_backends = reranker_backends or []
    if metric not in _METRIC_FIELDS:
        console.print(f"[red]Unknown metric {metric!r}. Use one of {sorted(_METRIC_FIELDS)}[/red]")
        return 1

    strategies = _resolve_strategies(strategies_filter)

    if dry_run:
        plan = plan_optimize(
            strategies,
            top_ks=top_ks,
            chunk_sizes=chunk_sizes,
            embedding_providers=embedding_providers,
            reranker_backends=reranker_backends,
            method=method,
            max_trials=max_trials,
            seed=seed,
        )
        t = Table(title=f"optimize plan — {corpus} (metric={metric}, method={method})")
        t.add_column("Strategy", style="bold")
        t.add_column("Trials", justify="right")
        t.add_column("Rebuilds", justify="right")
        t.add_column("Swept dims")
        total = 0
        for p in plan:
            total += p["n_trials"]
            t.add_row(p["strategy"], str(p["n_trials"]), str(p["n_rebuilds"]), ", ".join(p["dims"]))
        console.print(t)
        console.print(f"[dim]{total} trials total. Re-run without --dry-run to execute.[/dim]")
        return 0

    documents = load_documents(corpus)
    try:
        questions = load_questions(corpus)
    except FileNotFoundError:
        console.print(f"[red]No questions for {corpus}. Run generate-questions first.[/red]")
        return 1
    if not questions:
        console.print(f"[red]No questions for {corpus}.[/red]")
        return 1

    run_id = uuid4().hex[:8]
    results: dict[str, OptimizeResult] = {}
    for s in strategies:
        trials = _trials_for(
            s,
            top_ks=top_ks,
            chunk_sizes=chunk_sizes,
            embedding_providers=embedding_providers,
            reranker_backends=reranker_backends,
            method=method,
            max_trials=max_trials,
            seed=seed,
        )
        scored: list[tuple[TrialConfig, float]] = []
        base = baseline_config(s)
        for cfg in trials:
            start = time.perf_counter()
            score = await _score_trial(s, cfg, documents, questions, metric, base)
            scored.append((cfg, score))
            console.print(
                f"[dim]{s} {cfg.model_dump(exclude={'strategy'})} "
                f"{metric}={score:.4f} ({(time.perf_counter() - start):.1f}s)[/dim]"
            )
        results[s] = select_best(s, scored, baseline_config(s), metric=metric)

    out = Path(out_dir) if out_dir else Path(settings.results_path) / f"run_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus": corpus,
        "metric": metric,
        "method": method,
        "strategies": {
            name: {
                "best_config": r.best_config.model_dump(),
                "best_score": r.best_score,
                "baseline_config": r.baseline_config.model_dump(),
                "baseline_score": r.baseline_score,
                "delta": r.delta,
                "improved": r.improved,
                "n_trials": r.n_trials,
            }
            for name, r in results.items()
        },
    }
    (out / "optimize.json").write_text(json.dumps(report, indent=2))

    table = Table(title=f"optimize — {corpus} (metric={metric})")
    table.add_column("Strategy", style="bold")
    table.add_column(f"default {metric}", justify="right")
    table.add_column(f"best {metric}", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("best config")
    for name, r in results.items():
        sign = "+" if r.delta >= 0 else ""
        bc = r.best_config.model_dump(exclude={"strategy"})
        bc = {k: v for k, v in bc.items() if v is not None}
        table.add_row(
            name,
            f"{r.baseline_score:.4f}",
            f"{r.best_score:.4f}",
            f"[green]{sign}{r.delta:.4f}[/green]" if r.improved else f"{sign}{r.delta:.4f}",
            str(bc),
        )
    console.print(table)
    console.print(f"[green]Report: {out / 'optimize.json'}[/green]")
    return 0
