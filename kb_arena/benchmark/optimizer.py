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
        return (
            sum(self.per_question_scores) / len(self.per_question_scores)
            if self.per_question_scores
            else 0.0
        )

    @property
    def mean_latency_ms(self) -> float:
        return (
            sum(self.per_question_latency_ms) / len(self.per_question_latency_ms)
            if self.per_question_latency_ms
            else 0.0
        )


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
    win_rate_vs_baseline: float = 0.0
    best_metric_per_ms: float = 0.0
    baseline_metric_per_ms: float = 0.0
    pareto_optimal: bool = False

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


# ── Statistical summarization ────────────────────────────────────────────────


def _bootstrap_ci(
    values: list[float], n_resamples: int = 1000, ci: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean — Sakai's standard for IR."""
    if not values:
        return (0.0, 0.0)
    try:
        import numpy as np
        from scipy.stats import bootstrap as scipy_bootstrap

        if all(v == values[0] for v in values):
            return (values[0], values[0])
        res = scipy_bootstrap(
            (np.asarray(values),),
            statistic=np.mean,
            n_resamples=n_resamples,
            confidence_level=ci,
            method="percentile",
            random_state=0,
        )
        return (float(res.confidence_interval.low), float(res.confidence_interval.high))
    except Exception:  # noqa: BLE001 — graceful degradation if scipy missing
        m = sum(values) / len(values)
        return (m, m)


def _wilcoxon(baseline: list[float], best: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value for paired samples; None if degenerate."""
    if len(baseline) != len(best) or len(baseline) < 2:
        return None
    if all(b == a for a, b in zip(baseline, best, strict=True)):
        return 1.0  # no paired difference — definitely not significant
    try:
        from scipy.stats import wilcoxon

        stat = wilcoxon(best, baseline, zero_method="zsplit", alternative="two-sided")
        return float(stat.pvalue)
    except Exception:  # noqa: BLE001
        return None


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

    Delta is computed against the baseline trial — the trial whose config
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
    p_value = (
        None
        if same
        else _wilcoxon(baseline_trial.per_question_scores, best_trial.per_question_scores)
    )
    win_rate = (
        0.0
        if same
        else _win_rate(baseline_trial.per_question_scores, best_trial.per_question_scores)
    )

    return OptimizeResult(
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
        p_value=p_value,
        significant=(p_value is not None and p_value < 0.05 and best_score > baseline_score),
        win_rate_vs_baseline=win_rate,
        best_metric_per_ms=best_score / best_lat,
        baseline_metric_per_ms=baseline_score / base_lat,
    )


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


async def _score_trial(strategy, cfg, documents, questions, metric, baseline) -> TrialResult:
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
        if rebuild and hasattr(inst, "build_index"):
            try:
                await inst.build_index(documents)
            except Exception as exc:  # noqa: BLE001 — a dead config scores 0, not crash
                log.warning("optimize: build_index failed for %s %s: %s", strategy, cfg, exc)
                return TrialResult(cfg=cfg, per_question_scores=[0.0] * len(questions))
        scores: list[float] = []
        latencies: list[float] = []
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
                latencies.append(float(trace.latency_ms or 0.0))
    return TrialResult(cfg=cfg, per_question_scores=scores, per_question_latency_ms=latencies)


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
        trial_results: list[TrialResult] = []
        base = baseline_config(s)
        for cfg in trials:
            start = time.perf_counter()
            tr = await _score_trial(s, cfg, documents, questions, metric, base)
            trial_results.append(tr)
            console.print(
                f"[dim]{s} {cfg.model_dump(exclude={'strategy'})} "
                f"{metric}={tr.mean_score:.4f} ({(time.perf_counter() - start):.1f}s)[/dim]"
            )
        results[s] = summarize_optimization(s, trial_results, base, metric=metric)

    pareto_optimal_strategies(list(results.values()))  # marks pareto_optimal in-place

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
                "best_score_ci": list(r.best_score_ci),
                "baseline_config": r.baseline_config.model_dump(),
                "baseline_score": r.baseline_score,
                "baseline_score_ci": list(r.baseline_score_ci),
                "delta": r.delta,
                "improved": r.improved,
                "p_value": r.p_value,
                "significant": r.significant,
                "win_rate_vs_baseline": r.win_rate_vs_baseline,
                "best_metric_per_ms": r.best_metric_per_ms,
                "baseline_metric_per_ms": r.baseline_metric_per_ms,
                "pareto_optimal": r.pareto_optimal,
                "n_trials": r.n_trials,
            }
            for name, r in results.items()
        },
    }
    (out / "optimize.json").write_text(json.dumps(report, indent=2))

    table = Table(title=f"optimize — {corpus} (metric={metric})")
    table.add_column("Strategy", style="bold")
    table.add_column(f"default {metric}", justify="right")
    table.add_column(f"best {metric} [95% CI]", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("p", justify="right")
    table.add_column("win-rate", justify="right")
    table.add_column(f"{metric}/ms", justify="right")
    table.add_column("best config")
    for name, r in results.items():
        sign = "+" if r.delta >= 0 else ""
        bc = r.best_config.model_dump(exclude={"strategy"})
        bc = {k: v for k, v in bc.items() if v is not None}
        ci_lo, ci_hi = r.best_score_ci
        p_str = f"{r.p_value:.3g}" if r.p_value is not None else "—"
        sig_color = "green" if r.significant else "dim"
        delta_fmt = f"[{sig_color}]{sign}{r.delta:.4f}[/{sig_color}]"
        pareto_tag = " ★" if r.pareto_optimal else ""
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
        "[dim]★ = Pareto-optimal on (score, score/ms). "
        "Significance: green delta = Wilcoxon p<0.05 + positive lift.[/dim]"
    )
    console.print(f"[green]Report: {out / 'optimize.json'}[/green]")
    return 0
