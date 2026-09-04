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
import math
import random
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from uuid import uuid4

from pydantic import BaseModel, Field

from kb_arena.benchmark.ir_metrics import compute_all
from kb_arena.benchmark.questions import load_questions
from kb_arena.settings import settings
from kb_arena.strategies import load_documents

# Which dimensions each strategy actually consumes. Sweeping a dimension a
# strategy ignores just burns wall-clock on duplicate trials.
CHUNKING_STRATEGIES = frozenset({"naive_vector", "contextual_vector"})
EMBEDDING_STRATEGIES = frozenset({"naive_vector", "contextual_vector", "rerank_vector"})
RERANKER_STRATEGIES = frozenset({"rerank_vector"})
MAX_UNBOUNDED_TRIALS = 10_000

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


class TrialResult(BaseModel):
    """Per-question scores and latencies for one (strategy, config) trial.

    Carrying the per-question vectors (not just the mean) is what lets the
    summarizer compute bootstrap CIs, paired-significance tests, win-rates,
    and latency-normalized efficiency. Without this, every "improvement"
    the optimizer prints is a point estimate with no honesty layer.
    """

    cfg: TrialConfig
    per_question_scores: list[float] = Field(default_factory=list)
    per_question_latency_ms: list[float] = Field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return fmean(self.per_question_scores) if self.per_question_scores else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return fmean(self.per_question_latency_ms) if self.per_question_latency_ms else 0.0


class OptimizationTrialError(RuntimeError):
    """A trial failed before it could produce a complete score vector."""


class OptimizeResult(BaseModel):
    """Per-strategy outcome: best config, its lift over the baseline, and the
    statistical layer that makes the lift trustworthy (CIs, p-value, win-rate)."""

    strategy: str
    metric: str = "ndcg"
    best_config: TrialConfig
    best_score: float
    baseline_config: TrialConfig
    baseline_score: float
    n_trials: int = 0
    scored: list[tuple[TrialConfig, float]] = Field(default_factory=list)

    # v0.8.0 statistical layer
    best_score_ci: tuple[float, float] = (0.0, 0.0)
    baseline_score_ci: tuple[float, float] = (0.0, 0.0)
    p_value: float | None = None
    significant: bool = False
    # Multiplicity control. The best trial is the maximum over every trial,
    # so its raw p-value is biased. p_value above is the Holm-adjusted one
    # over n_comparisons, and p_value_raw is what one test alone would say.
    p_value_raw: float | None = None
    trial_p_values: list[float | None] = Field(default_factory=list)
    n_comparisons: int = 1
    correction: str = "none"
    exploratory: bool = False
    publishable: bool = False
    win_rate_vs_baseline: float = 0.0
    best_metric_per_ms: float = 0.0
    baseline_metric_per_ms: float = 0.0
    pareto_optimal: bool = False

    best_trial_index: int | None = None

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
    if max_trials < 0:
        raise ValueError("max_trials must be nonnegative")

    dims = applicable_dims(strategy)
    top_k_axis = _axis(top_ks, baseline.top_k, "top_k" in dims)
    chunk_axis = _axis(chunk_sizes, baseline.chunk_tokens, "chunk_tokens" in dims)
    emb_axis = _axis(embedding_providers, baseline.embedding_provider, "embedding_provider" in dims)
    rer_axis = _axis(reranker_backends, baseline.reranker_backend, "reranker_backend" in dims)

    axes = (top_k_axis, chunk_axis, emb_axis, rer_axis)
    total_combinations = math.prod(len(axis) for axis in axes)
    if max_trials > MAX_UNBOUNDED_TRIALS:
        raise ValueError(f"max_trials must be {MAX_UNBOUNDED_TRIALS:,} or fewer")
    if max_trials == 0 and total_combinations > MAX_UNBOUNDED_TRIALS:
        raise ValueError(
            f"Search space has {total_combinations:,} trials; set --max-trials "
            f"to {MAX_UNBOUNDED_TRIALS:,} or fewer."
        )
    if max_trials == 1:
        return [baseline]

    def _config(values) -> TrialConfig:
        tk, ck, ep, rb = values
        return TrialConfig(
            strategy=strategy,
            top_k=tk,
            chunk_tokens=ck,
            embedding_provider=ep,
            reranker_backend=rb,
        )

    # Baseline stays first so every reported delta has a measured reference.
    ordered: list[TrialConfig] = [baseline]
    if method == "random" and max_trials and max_trials < total_combinations:
        baseline_coords = (
            top_k_axis.index(baseline.top_k),
            chunk_axis.index(baseline.chunk_tokens),
            emb_axis.index(baseline.embedding_provider),
            rer_axis.index(baseline.reranker_backend),
        )
        baseline_index = 0
        for coordinate, axis in zip(baseline_coords, axes, strict=True):
            baseline_index = baseline_index * len(axis) + coordinate

        rng = random.Random(seed)
        sampled = rng.sample(range(total_combinations - 1), max_trials - 1)
        for compressed_index in sampled:
            flat_index = (
                compressed_index if compressed_index < baseline_index else compressed_index + 1
            )
            coordinates: list[int] = []
            for axis in reversed(axes):
                flat_index, coordinate = divmod(flat_index, len(axis))
                coordinates.append(coordinate)
            values = [axis[i] for axis, i in zip(axes, reversed(coordinates), strict=True)]
            ordered.append(_config(values))
        return ordered

    for values in itertools.product(*axes):
        candidate = _config(values)
        if candidate == baseline:
            continue
        ordered.append(candidate)
        if max_trials and len(ordered) >= max_trials:
            break

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


# ── Statistical summarization ────────────────────────────────────────────────


def _bootstrap_ci(
    values: list[float], n_resamples: int = 1000, ci: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean, following Sakai's standard for IR."""
    if not values:
        return (0.0, 0.0)
    if all(v == values[0] for v in values):
        mean = fmean(values)
        return (mean, mean)
    try:
        import numpy as np
        from scipy.stats import bootstrap as scipy_bootstrap

        res = scipy_bootstrap(
            (np.asarray(values),),
            statistic=np.mean,
            n_resamples=n_resamples,
            confidence_level=ci,
            method="percentile",
            random_state=0,
        )
        return (float(res.confidence_interval.low), float(res.confidence_interval.high))
    except ImportError:  # optional scientific stack is unavailable
        m = fmean(values)
        return (m, m)


def _wilcoxon(baseline: list[float], best: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value for paired samples; None if degenerate."""
    if len(baseline) != len(best) or len(baseline) < 2:
        return None
    if all(b == a for a, b in zip(baseline, best, strict=True)):
        return 1.0  # No paired difference means no significant difference.
    try:
        from scipy.stats import wilcoxon

        stat = wilcoxon(best, baseline, zero_method="zsplit", alternative="two-sided")
        return float(stat.pvalue)
    except Exception:  # noqa: BLE001
        return None


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Holm step-down adjustment. None entries stay None and do not count."""
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(indexed)):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def _apply_family(result: OptimizeResult, adjusted_best: float | None, n_comparisons: int) -> None:
    """Set the adjusted p, the flags, and the family size on one result."""
    result.p_value = adjusted_best
    result.n_comparisons = n_comparisons
    result.correction = "holm" if n_comparisons > 1 else "none"
    result.significant = (
        adjusted_best is not None
        and adjusted_best < 0.05
        and result.best_score > result.baseline_score
    )
    # A sweep picks the best of many. Its finding is a lead to confirm on the
    # holdout split, not a result to publish on its own.
    result.exploratory = n_comparisons > 1
    result.publishable = result.significant and not result.exploratory


def apply_run_wide_holm(results: dict[str, OptimizeResult]) -> int:
    """One Holm family over every trial-versus-baseline test in the run.

    Returns the family size. Each result's p_value becomes the run-wide
    adjusted p of its best trial.
    """
    order: list[tuple[str, int]] = []
    raw: list[float | None] = []
    for name, r in results.items():
        for i, p in enumerate(r.trial_p_values):
            order.append((name, i))
            raw.append(p)
    adjusted = holm_adjust(raw)
    m = sum(1 for p in raw if p is not None)
    by_key = {key: p for key, p in zip(order, adjusted, strict=True)}
    for name, r in results.items():
        best_index = r.best_trial_index
        best_adjusted = by_key.get((name, best_index)) if best_index is not None else None
        _apply_family(r, best_adjusted, max(1, m))
    return m


def _win_rate(baseline: list[float], best: list[float]) -> float:
    if not baseline or len(baseline) != len(best):
        return 0.0
    return sum(1 for a, b in zip(baseline, best, strict=True) if b > a) / len(baseline)


def summarize_optimization(
    strategy: str,
    trials: list[TrialResult],
    baseline: TrialConfig,
    metric: str = "ndcg",
) -> OptimizeResult:
    """Pick the best trial and attach the statistical layer (CI, p, win-rate, efficiency).

    Delta is computed against the baseline trial, whose config
    equals `baseline`, falling back to the first trial when none matches.
    """
    if not trials:
        raise ValueError("no trials to summarize")

    baseline_trial = next((t for t in trials if t.cfg == baseline), trials[0])
    best_trial = max(trials, key=lambda t: t.mean_score)

    baseline_score = baseline_trial.mean_score
    best_score = best_trial.mean_score
    base_lat = baseline_trial.mean_latency_ms or 1.0
    best_lat = best_trial.mean_latency_ms or 1.0

    same = best_trial is baseline_trial
    # Every trial is tested against the baseline, not only the winner. The
    # winner is the maximum over all of them, so its own p-value alone is
    # biased. The whole family gets a Holm adjustment.
    trial_p_values = [
        None
        if t is baseline_trial
        else _wilcoxon(baseline_trial.per_question_scores, t.per_question_scores)
        for t in trials
    ]
    best_index = None if same else trials.index(best_trial)
    p_value_raw = None if same else trial_p_values[best_index]
    win_rate = (
        0.0
        if same
        else _win_rate(baseline_trial.per_question_scores, best_trial.per_question_scores)
    )

    result = OptimizeResult(
        strategy=strategy,
        metric=metric,
        best_config=best_trial.cfg,
        best_score=best_score,
        baseline_config=baseline,
        baseline_score=baseline_score,
        n_trials=len(trials),
        scored=[(t.cfg, t.mean_score) for t in trials],
        best_score_ci=_bootstrap_ci(best_trial.per_question_scores),
        baseline_score_ci=_bootstrap_ci(baseline_trial.per_question_scores),
        p_value_raw=p_value_raw,
        trial_p_values=trial_p_values,
        best_trial_index=best_index,
        win_rate_vs_baseline=win_rate,
        best_metric_per_ms=best_score / best_lat,
        baseline_metric_per_ms=baseline_score / base_lat,
    )
    adjusted = holm_adjust(trial_p_values)
    family = sum(1 for p in trial_p_values if p is not None)
    _apply_family(result, None if same else adjusted[best_index], max(1, family))
    return result


def strategy_report(r: OptimizeResult) -> dict:
    """The JSON a run writes for one strategy."""
    return {
        "best_config": r.best_config.model_dump(),
        "best_score": r.best_score,
        "best_score_ci": list(r.best_score_ci),
        "baseline_config": r.baseline_config.model_dump(),
        "baseline_score": r.baseline_score,
        "baseline_score_ci": list(r.baseline_score_ci),
        "delta": r.delta,
        "improved": r.improved,
        "p_value": r.p_value,
        "p_value_raw": r.p_value_raw,
        "n_comparisons": r.n_comparisons,
        "correction": r.correction,
        "exploratory": r.exploratory,
        "publishable": r.publishable,
        "significant": r.significant,
        "win_rate_vs_baseline": r.win_rate_vs_baseline,
        "best_metric_per_ms": r.best_metric_per_ms,
        "baseline_metric_per_ms": r.baseline_metric_per_ms,
        "pareto_optimal": r.pareto_optimal,
        "n_trials": r.n_trials,
    }


def pareto_optimal_strategies(results: list[OptimizeResult]) -> list[OptimizeResult]:
    """Return the Pareto frontier in (latency, score): higher best_score is
    better; lower mean latency (implied by higher metric_per_ms) is better.

    A result is dominated iff some other result has score >= this one AND
    metric_per_ms >= this one, with at least one strict inequality. Ties are
    not dominated."""
    keep: list[OptimizeResult] = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            if (
                other.best_score >= r.best_score
                and other.best_metric_per_ms >= r.best_metric_per_ms
                and (
                    other.best_score > r.best_score
                    or other.best_metric_per_ms > r.best_metric_per_ms
                )
            ):
                dominated = True
                break
        if not dominated:
            keep.append(r)
    for r in keep:
        r.pareto_optimal = True
    return keep


# ── Orchestration ─────────────────────────────────────────────────────────────


def baseline_config(strategy: str) -> TrialConfig:
    """Return the configuration produced by the current settings."""
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


async def _score_trial(
    strategy, cfg, documents, questions, metric, baseline, corpus: str = "all"
) -> TrialResult:
    """Per-question scores + latencies for one (strategy, config) trial.

    Retrieval-only (LLM generation stubbed) so a full sweep stays ~10x cheaper
    than the answer benchmark. Rebuilds the index only when the trial's
    rebuild dims differ from the baseline (the persistent-index config).
    Returns a TrialResult carrying every per-question score so the summarizer
    can compute CIs, paired significance, and win-rate.
    """
    from kb_arena.benchmark.retriever_lab import _PatchLLMClient, _retrieve_only
    from kb_arena.strategies import get_strategy

    rebuild = needs_rebuild(cfg, baseline)
    field = _METRIC_FIELDS.get(metric, "mean_ndcg_at_k").replace("mean_", "")

    with _ApplyOverrides(cfg, isolate_chroma=rebuild):
        inst = get_strategy(strategy)
        with _PatchLLMClient():
            if rebuild and hasattr(inst, "build_index"):
                try:
                    await inst.build_index(documents)
                except Exception as exc:
                    raise OptimizationTrialError(
                        f"build failed for strategy {strategy!r}, config {cfg}: {exc}"
                    ) from exc
            scores: list[float] = []
            latencies: list[float] = []
            for q in questions:
                try:
                    trace = await _retrieve_only(inst, q.question, cfg.top_k, corpus=corpus)
                except Exception as exc:
                    raise OptimizationTrialError(
                        f"retrieval failed for strategy {strategy!r}, config {cfg}, "
                        f"question {getattr(q, 'id', '<unknown>')!r}: {exc}"
                    ) from exc
                m = compute_all(
                    retrieved=trace.retrieved,
                    expected_ids=set(getattr(q, "expected_chunks", []) or []),
                    k=cfg.top_k,
                    expected_doc_ids=set(
                        getattr(getattr(q, "ground_truth", None), "source_refs", []) or []
                    ),
                )
                scores.append(getattr(m, field))
                latencies.append(float(trace.latency_ms or 0.0))
    return TrialResult(cfg=cfg, per_question_scores=scores, per_question_latency_ms=latencies)


def _resolve_strategies(strategies_filter: str) -> list[str]:
    from kb_arena.benchmark.runner import STRATEGY_NAMES

    if strategies_filter == "all":
        return list(STRATEGY_NAMES)
    return [s.strip() for s in strategies_filter.split(",") if s.strip()]


def validate_optimize_inputs(
    strategies_filter: str,
    *,
    top_ks: list[int],
    chunk_sizes: list[int],
    method: str,
    max_trials: int,
) -> list[str]:
    """Validate an optimization plan before credentials or services are used."""
    from kb_arena.strategies.catalog import STRATEGY_CATALOG

    if method not in {"grid", "random"}:
        raise ValueError("Unknown search method. Use 'grid' or 'random'.")
    if max_trials < 0:
        raise ValueError("--max-trials must be nonnegative.")
    if any(top_k < 1 for top_k in top_ks):
        raise ValueError("Top-k values must be positive.")
    if any(size <= settings.chunk_overlap_tokens for size in chunk_sizes):
        raise ValueError(
            "Chunk sizes must be greater than the configured overlap "
            f"({settings.chunk_overlap_tokens})."
        )

    strategies = _resolve_strategies(strategies_filter)
    if not strategies:
        raise ValueError("Choose at least one strategy.")
    known = {spec.name for spec in STRATEGY_CATALOG}
    unknown = sorted(set(strategies) - known)
    if unknown:
        raise ValueError(f"Unknown strategy: {', '.join(unknown)}.")
    return strategies


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
    split: str = "auto",
) -> int:
    """Sweep, score, report. Returns 0 on success, 1 on hard failure."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    top_ks = top_ks or [5]
    chunk_sizes = chunk_sizes or []
    embedding_providers = embedding_providers or []
    reranker_backends = reranker_backends or []
    try:
        strategies = validate_optimize_inputs(
            strategies_filter,
            top_ks=top_ks,
            chunk_sizes=chunk_sizes,
            method=method,
            max_trials=max_trials,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    if metric not in _METRIC_FIELDS:
        console.print(f"[red]Unknown metric {metric!r}. Use one of {sorted(_METRIC_FIELDS)}[/red]")
        return 1

    valid_splits = {"auto", "all", "development", "validation", "holdout", "unspecified"}
    if split not in valid_splits:
        console.print(f"[red]Unknown split {split!r}. Use one of {sorted(valid_splits)}[/red]")
        return 1

    if dry_run:
        try:
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
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        t = Table(title=f"optimize plan: {corpus} (metric={metric}, method={method})")
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

    documents = load_documents(corpus, strict=True)
    try:
        all_questions = load_questions(corpus)
    except FileNotFoundError:
        console.print(f"[red]No questions for {corpus}. Run generate-questions first.[/red]")
        return 1
    effective_split = split
    if split == "auto":
        labeled = {getattr(q, "split", "unspecified") for q in all_questions} - {"unspecified"}
        effective_split = "development" if labeled else "all"
    questions = (
        all_questions
        if effective_split == "all"
        else [q for q in all_questions if getattr(q, "split", "unspecified") == effective_split]
    )
    if not questions:
        console.print(f"[red]No questions for {corpus}.[/red]")
        return 1

    run_id = uuid4().hex[:8]
    results: dict[str, OptimizeResult] = {}
    for s in strategies:
        try:
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
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        trial_results: list[TrialResult] = []
        base = baseline_config(s)
        for cfg in trials:
            start = time.perf_counter()
            try:
                tr = await _score_trial(s, cfg, documents, questions, metric, base, corpus=corpus)
            except Exception as exc:
                console.print(f"[red]Optimization aborted[/red] {exc}")
                console.print("[red]No recommendation report was written.[/red]")
                return 1
            trial_results.append(tr)
            console.print(
                f"[dim]{s} {cfg.model_dump(exclude={'strategy'})} "
                f"{metric}={tr.mean_score:.4f} ({(time.perf_counter() - start):.1f}s)[/dim]"
            )
        results[s] = summarize_optimization(s, trial_results, base, metric=metric)

    pareto_optimal_strategies(list(results.values()))  # marks pareto_optimal in-place
    family_size = apply_run_wide_holm(results)

    out = Path(out_dir) if out_dir else Path(settings.results_path) / f"run_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus": corpus,
        "metric": metric,
        "method": method,
        "question_split": effective_split,
        "n_comparisons": family_size,
        "correction": "holm" if family_size > 1 else "none",
        "note": (
            "Every p_value is Holm-adjusted over n_comparisons trial-versus-baseline tests "
            "in this run. p_value_raw is the single-test value. A sweep is exploratory: "
            "confirm a lead on the holdout split before you publish it."
        ),
        "strategies": {name: strategy_report(r) for name, r in results.items()},
    }
    (out / "optimize.json").write_text(json.dumps(report, indent=2))

    table = Table(title=f"optimize: {corpus} (metric={metric})")
    table.add_column("Strategy", style="bold")
    table.add_column(f"default {metric}", justify="right")
    table.add_column(f"best {metric} [95% CI]", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("p (Holm)", justify="right")
    table.add_column("win-rate", justify="right")
    table.add_column(f"{metric}/ms", justify="right")
    table.add_column("best config")
    for name, r in results.items():
        sign = "+" if r.delta >= 0 else ""
        bc = r.best_config.model_dump(exclude={"strategy"})
        bc = {k: v for k, v in bc.items() if v is not None}
        ci_lo, ci_hi = r.best_score_ci
        p_str = f"{r.p_value:.3g}" if r.p_value is not None else "n/a"
        sig_color = "green" if r.significant else "dim"
        delta_fmt = f"[{sig_color}]{sign}{r.delta:.4f}[/{sig_color}]"
        pareto_tag = " [Pareto]" if r.pareto_optimal else ""
        table.add_row(
            name + pareto_tag,
            f"{r.baseline_score:.4f}",
            f"{r.best_score:.4f} [{ci_lo:.3f}, {ci_hi:.3f}]",
            delta_fmt,
            p_str,
            f"{r.win_rate_vs_baseline:.0%}",
            f"{r.best_metric_per_ms:.3g}",
            str(bc),
        )
    console.print(table)
    console.print(
        f"[dim]p-values Holm-adjusted over {family_size} trial-versus-baseline tests. "
        "A sweep is exploratory: confirm a lead on the holdout split before you publish it.[/dim]"
    )
    console.print(
        "[dim][Pareto] = Pareto-optimal on (score, score/ms). "
        "Significance: green delta = Wilcoxon p<0.05 + positive lift.[/dim]"
    )
    console.print(f"[green]Report: {out / 'optimize.json'}[/green]")
    return 0
