"""Retriever Lab retrieval-only benchmark with classical IR metrics.

Strategies emit retrieval traces without LLM generation. IR metrics are computed
against ground truth, results streamed to a Rich table. Roughly an order of
magnitude cheaper than `kb-arena benchmark` since the generator step is skipped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.live import Live
from rich.table import Table

# ChromaDB 0.5.23 has a buggy telemetry callback that floods stderr. Disable it.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


from kb_arena.benchmark.atomic import atomic_write_text  # noqa: E402
from kb_arena.benchmark.holdout import record_holdout_use, touches_holdout  # noqa: E402
from kb_arena.benchmark.ir_metrics import (  # noqa: E402  # noqa: E402
    MATCH_CLASSES,
    _match_expected,
    compute_all,
    match_class,
)
from kb_arena.benchmark.manifest import build_manifest  # noqa: E402
from kb_arena.benchmark.questions import discover_corpora, load_questions  # noqa: E402
from kb_arena.models.benchmark import RetrievalMetrics  # noqa: E402
from kb_arena.models.retrieval import RetrievalTrace  # noqa: E402
from kb_arena.settings import settings  # noqa: E402
from kb_arena.strategies.base import Strategy  # noqa: E402

console = Console()
log = logging.getLogger(__name__)


class RetrievalExecutionError(RuntimeError):
    """A strategy query failed before it could produce a retrieval trace."""


class _PatchLLMClient:
    """Enable zero-cost LLM responses only in the current async context.

    Strategies invoke query() which always runs LLM generation. retriever-lab
    needs only the retrieval trace, so generation is disabled without changing
    LLM behavior in concurrent chat or benchmark tasks.
    """

    def __enter__(self):
        from kb_arena.llm.client import _retrieval_only_mode

        self._mode = _retrieval_only_mode
        self._token = self._mode.set(True)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._mode.reset(self._token)
        return False


async def _retrieve_only(
    strategy: Strategy,
    question_text: str,
    top_k: int,
    corpus: str = "all",
) -> RetrievalTrace:
    """Run query() under the LLM-stub patch and return the retrieval trace."""
    if strategy.name == "pageindex":
        raise RetrievalExecutionError(
            "strategy 'pageindex' is not supported in zero-LLM Retriever Lab runs"
        )
    try:
        if corpus == "all":
            result = await strategy.query(question_text, top_k=top_k)
        else:
            result = await strategy.query(question_text, top_k=top_k, corpus=corpus)
    except Exception as exc:
        raise RetrievalExecutionError(
            f"strategy {strategy.name!r} failed to retrieve for {question_text!r}: {exc}"
        ) from exc
    if result.retrieval is not None:
        return result.retrieval
    raise RetrievalExecutionError(
        f"strategy {strategy.name!r} returned no retrieval trace for {question_text!r}"
    )


def _build_table(
    title: str, top_k: int, strategies: list[Strategy], rows: dict[str, list[RetrievalMetrics]]
) -> Table:
    t = Table(title=title)
    t.add_column("Strategy", style="bold")
    t.add_column(f"Recall@{top_k}", justify="right")
    t.add_column(f"P@{top_k}", justify="right")
    t.add_column(f"Hit@{top_k}", justify="right")
    t.add_column("MRR", justify="right")
    t.add_column(f"NDCG@{top_k}", justify="right")
    t.add_column("n", justify="right")
    for s in strategies:
        records = rows.get(s.name, [])
        if records:
            n = len(records)
            t.add_row(
                s.name,
                f"{sum(r.recall_at_k for r in records) / n:.3f}",
                f"{sum(r.precision_at_k for r in records) / n:.3f}",
                f"{sum(r.hit_at_k for r in records) / n:.3f}",
                f"{sum(r.mrr for r in records) / n:.3f}",
                f"{sum(r.ndcg_at_k for r in records) / n:.3f}",
                str(n),
            )
        else:
            t.add_row(s.name, "n/a", "n/a", "n/a", "n/a", "n/a", "0")
    return t


def _aggregate_means(records: list[RetrievalMetrics]) -> dict[str, float | int]:
    if not records:
        return {
            "mean_recall_at_k": 0.0,
            "mean_precision_at_k": 0.0,
            "mean_hit_at_k": 0.0,
            "mean_mrr": 0.0,
            "mean_ndcg_at_k": 0.0,
            "mean_average_precision": 0.0,
            "mean_r_precision": 0.0,
            "mean_bpref": 0.0,
            "questions": 0,
        }
    n = len(records)
    return {
        "mean_recall_at_k": sum(r.recall_at_k for r in records) / n,
        "mean_precision_at_k": sum(r.precision_at_k for r in records) / n,
        "mean_hit_at_k": sum(r.hit_at_k for r in records) / n,
        "mean_mrr": sum(r.mrr for r in records) / n,
        "mean_ndcg_at_k": sum(r.ndcg_at_k for r in records) / n,
        "mean_average_precision": sum(r.average_precision for r in records) / n,
        "mean_r_precision": sum(r.r_precision for r in records) / n,
        "mean_bpref": sum(r.bpref for r in records) / n,
        "questions": n,
    }


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean."""
    if not values:
        return (0.0, 0.0)
    if all(v == values[0] for v in values):
        return (values[0], values[0])
    import numpy as np
    from scipy.stats import bootstrap

    res = bootstrap(
        (np.asarray(values),),
        statistic=np.mean,
        n_resamples=1000,
        confidence_level=0.95,
        method="percentile",
        random_state=0,
    )
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


def _summarize_with_tiers(
    rows: list[tuple[int, RetrievalMetrics]],
) -> dict:
    """Aggregate means + per-tier breakdown + 95% bootstrap CIs on key metrics.

    Per-tier breakdown answers whether the win holds on hard queries, a
    strategy can dominate on tier-1 lookups and lose badly on tier-5
    architecture queries while the overall mean masks it.
    """
    out = _aggregate_means([m for _, m in rows])

    # Use Sakai's standard 95% CIs on the headline metrics.
    out["ci_ndcg_at_k"] = _bootstrap_ci([m.ndcg_at_k for _, m in rows])
    out["ci_recall_at_k"] = _bootstrap_ci([m.recall_at_k for _, m in rows])
    out["ci_average_precision"] = _bootstrap_ci([m.average_precision for _, m in rows])

    by_tier: dict[int, dict] = {}
    tiers = sorted({t for t, _ in rows if t is not None})
    for tier in tiers:
        tier_rows = [m for t, m in rows if t == tier]
        if tier_rows:
            by_tier[tier] = _aggregate_means(tier_rows)
    out["by_tier"] = by_tier
    return out


def _negatives_of(question) -> set[str] | None:
    """Chunks a judge called irrelevant, for bpref.

    None when the labels name none. `compute_all` reads None as "nobody judged
    negatives here" and falls back to the TREC bpref-10 proxy. An empty set
    would instead say "the judge found no negatives", which turns the proxy off
    and scores a bad ranking as perfect.
    """
    named = getattr(question, "judged_negatives", None) or []
    return set(named) or None


def _grades_of(question) -> dict[str, float] | None:
    """Graded relevance for the IR metrics, or None when the labels carry no grades."""
    grades = getattr(question, "expected_grades", None) or {}
    return {c: float(g) for c, g in grades.items()} if grades else None


def count_match_classes(classes) -> dict[str, int]:
    """Count retrieved chunks per match class, every class present."""
    counts = dict.fromkeys(MATCH_CLASSES, 0)
    for cls in classes:
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _strict_share(counts: dict[str, int]) -> float | None:
    """The share of hits that matched a strict id. None when nothing matched."""
    hits = counts.get("strict", 0) + counts.get("parent", 0) + counts.get("doc", 0)
    return round(counts.get("strict", 0) / hits, 4) if hits else None


def _summarize(records: list[RetrievalMetrics]) -> dict[str, float | int]:
    """Back-compat shim used by the live Rich table during the run."""
    return _aggregate_means(records)


def _summarize_ceiling(
    top_recalls: list[float], ceiling_recalls: list[float], top_k: int, ceiling_k: int
) -> dict:
    """Aggregate the base retriever's recall at the shallow vs deep cutoff.

    `ranking_headroom` = Recall@ceiling_k - Recall@top_k is the share of relevant
    chunks the retriever already surfaces but ranks below top_k. A large headroom
    means the bottleneck is RANKING, not retrieval; a small Recall@ceiling_k means
    the chunks are genuinely missing from the pool. The interpretation is left to
    the reader. This function only reports the measured numbers.
    """
    n = len(top_recalls) or 1
    rt = sum(top_recalls) / n
    rc = sum(ceiling_recalls) / n
    return {
        "top_k": top_k,
        "ceiling_k": ceiling_k,
        "recall_at_top_k": rt,
        "recall_at_ceiling_k": rc,
        "ranking_headroom": max(rc - rt, 0.0),
        "questions": len(top_recalls),
    }


def _ranks_over_naive_pool(strategy) -> bool:
    """True when the strategy's candidates come from the naive_vector collection.

    The retrieval-ceiling diagnostic measures that collection, so it only says
    something about naive_vector itself and the rerankers that wrap it. Every
    reranker in the tree, plugin or built in, keeps its base as `_base`, so
    the check reads the type instead of a closed list of names. hybrid ranks
    over contextual_vector, which has its own collection, so it does not count.
    """
    from kb_arena.strategies.naive_vector import NaiveVectorStrategy

    if strategy.name == "naive_vector":
        return True
    return isinstance(getattr(strategy, "_base", None), NaiveVectorStrategy)


async def _retrieval_ceiling(
    questions,
    top_k: int,
    ceiling_k: int,
    corpus: str = "all",
) -> dict:
    """Base-retriever (naive_vector) recall at top_k vs ceiling_k over `questions`.

    naive_vector is the pool every vector reranker shares, so its recall ceiling
    is the shared retrieval ceiling. Pure retrieval (LLM is already stubbed by the
    caller); one extra deep query per question, no generation cost.
    """
    from kb_arena.strategies import get_strategy

    try:
        base = get_strategy("naive_vector")
    except Exception as exc:  # noqa: BLE001 - report initialization failures in the artifact
        log.warning("Retrieval ceiling failed (naive_vector unavailable): %s", exc)
        return {
            "status": "error",
            "top_k": top_k,
            "ceiling_k": ceiling_k,
            "questions": 0,
            "execution_error": {"type": type(exc).__name__, "message": str(exc)},
        }

    top_recalls: list[float] = []
    ceiling_recalls: list[float] = []
    for q in questions:
        try:
            trace = await _retrieve_only(base, q.question, ceiling_k, corpus=corpus)
        except RetrievalExecutionError as exc:
            log.warning("Retrieval ceiling failed: %s", exc)
            cause = exc.__cause__ or exc
            return {
                "status": "error",
                "top_k": top_k,
                "ceiling_k": ceiling_k,
                "questions": len(top_recalls),
                "execution_error": {"type": type(cause).__name__, "message": str(exc)},
            }
        retrieved = trace.retrieved
        expected = set(q.expected_chunks or [])
        doc_ids = set(q.ground_truth.source_refs)
        grades = _grades_of(q)
        m_top = compute_all(
            retrieved=retrieved[:top_k],
            expected_ids=expected,
            k=top_k,
            expected_doc_ids=doc_ids,
            expected_relevance=grades,
            judged_nonrelevant=_negatives_of(q),
        )
        m_ceil = compute_all(
            retrieved=retrieved[:ceiling_k],
            expected_ids=expected,
            k=ceiling_k,
            expected_doc_ids=doc_ids,
            expected_relevance=grades,
            judged_nonrelevant=_negatives_of(q),
        )
        top_recalls.append(m_top.recall_at_k)
        ceiling_recalls.append(m_ceil.recall_at_k)

    return {
        "status": "complete",
        **_summarize_ceiling(top_recalls, ceiling_recalls, top_k, ceiling_k),
    }


async def run_retriever_lab(
    corpus: str = "all",
    strategies_filter: str = "all",
    top_k: int = 5,
    min_recall: float = 0.30,
    ceiling_k: int | None = None,
    *,
    split: str = "",
) -> int:
    """Run retrieval-only benchmark. Returns 0 on success, 1 if min_recall floor breached."""
    from kb_arena.benchmark.runner import _load_strategies

    if not math.isfinite(min_recall) or not 0.0 <= min_recall <= 1.0:
        console.print("[red]min_recall must be a finite number between 0 and 1.[/red]")
        return 1

    run_id = uuid4().hex[:8]
    timestamp = datetime.now(UTC).isoformat()
    ceiling_requested = ceiling_k is not None
    ceiling_k = ceiling_k if ceiling_k and ceiling_k > top_k else top_k * 4

    corpora = discover_corpora() if corpus == "all" else [corpus]
    strategies = _load_strategies(strategies_filter)
    if strategies_filter == "all":
        strategies = [strategy for strategy in strategies if strategy.name != "pageindex"]
    if not strategies:
        console.print("[red]No strategies available. Run build-vectors first.[/red]")
        return 1
    run_ceiling = ceiling_requested or any(_ranks_over_naive_pool(s) for s in strategies)

    results_dir = Path(settings.results_path) / f"run_{run_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    overall: dict = {
        "run_id": run_id,
        "timestamp": timestamp,
        "top_k": top_k,
        "ceiling_k": ceiling_k,
        "question_split": split or "all",
        "corpora": {},
        "retrieval_ceiling": {},
        "manifests": {},
    }
    per_question_rows: list[dict] = []

    console.print(f"[dim]Run ID: {run_id} | top-k: {top_k} | ceiling-k: {ceiling_k}[/dim]")

    _llm_patch = _PatchLLMClient()
    _llm_patch.__enter__()
    run_error: Exception | None = None
    try:
        _exit_code = await _run_corpora_loop(
            corpora,
            strategies,
            top_k,
            min_recall,
            overall,
            per_question_rows,
            ceiling_k,
            split,
            run_ceiling=run_ceiling,
        )
        overall["status"] = "complete"
    except Exception as exc:
        run_error = exc
        _exit_code = 1
        overall["status"] = "incomplete"
        overall["execution_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        console.print(f"[red]Retriever Lab failed: {type(exc).__name__}: {exc}[/red]")
    finally:
        _llm_patch.__exit__(None, None, None)

    json_path = results_dir / "retriever_lab.json"
    atomic_write_text(
        json_path,
        json.dumps({**overall, "questions": per_question_rows}, indent=2, ensure_ascii=False),
    )

    md_lines = [f"# Retriever Lab: run {run_id}", "", f"Top-k: {top_k}", ""]
    for corp_name, by_strategy in overall["corpora"].items():
        md_lines += [
            f"## {corp_name}",
            "",
            (
                f"| Strategy | Recall@{top_k} | P@{top_k} | Hit@{top_k} "
                f"| MRR | NDCG@{top_k} | n | Errors |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ]
        for sname, m in by_strategy.items():
            if m["questions"]:
                metric_cells = (
                    f"{m['mean_recall_at_k']:.3f} | {m['mean_precision_at_k']:.3f} "
                    f"| {m['mean_hit_at_k']:.3f} | {m['mean_mrr']:.3f} "
                    f"| {m['mean_ndcg_at_k']:.3f}"
                )
            else:
                metric_cells = "n/a | n/a | n/a | n/a | n/a"
            md_lines.append(
                f"| {sname} | {metric_cells} | {m['questions']} | {m['execution_errors']} |"
            )
        md_lines.append("")
    md_path = results_dir / "retriever_lab.md"
    atomic_write_text(md_path, "\n".join(md_lines))

    floor_violation = False
    for corp_name, by_strategy in overall["corpora"].items():
        for sname, m in by_strategy.items():
            if m["execution_errors"]:
                console.print(
                    f"[red]FAIL[/red] {corp_name}/{sname} had "
                    f"{m['execution_errors']} retrieval execution error(s); "
                    "failed queries were excluded from metrics"
                )
                floor_violation = True
            elif m["mean_recall_at_k"] < min_recall:
                console.print(
                    f"[red]FAIL[/red] {corp_name}/{sname} "
                    f"Recall@{top_k}={m['mean_recall_at_k']:.3f} < {min_recall}"
                )
                floor_violation = True

    no_questions = not overall["corpora"]
    if run_error is not None:
        console.print(f"[red]Incomplete run written to {results_dir}/[/red]")
    elif no_questions:
        console.print(
            f"[red]No questions were selected; incomplete run written to {results_dir}/[/red]"
        )
    else:
        console.print(f"[green]Run {run_id} written to {results_dir}/[/green]")
    return 1 if run_error is not None or no_questions or floor_violation else _exit_code


async def _run_corpora_loop(
    corpora,
    strategies,
    top_k,
    min_recall,
    overall,
    per_question_rows,
    ceiling_k,
    split,
    run_ceiling: bool = True,
):
    for corp in corpora:
        try:
            questions = load_questions(corp, split=split)
        except FileNotFoundError:
            console.print(f"[yellow]No questions for {corp}; skipping[/yellow]")
            continue
        if touches_holdout(questions):
            record_holdout_use(
                settings.results_path,
                tool="retriever-lab",
                corpus=corp,
                run_id=str(overall.get("run_id", "")),
                strategies=[s.name for s in strategies],
            )
        if not questions:
            continue

        per_strategy_rows: dict[str, list[RetrievalMetrics]] = {s.name: [] for s in strategies}
        per_strategy_classes: dict[str, dict[str, int]] = {
            s.name: dict.fromkeys(MATCH_CLASSES, 0) for s in strategies
        }
        per_strategy_tier_rows: dict[str, list[tuple[int, RetrievalMetrics]]] = {
            s.name: [] for s in strategies
        }
        # Empty retrievals are valid observations; execution errors are tracked separately.
        per_strategy_empty: dict[str, int] = {s.name: 0 for s in strategies}
        per_strategy_errors: dict[str, int] = {s.name: 0 for s in strategies}
        title = f"Retriever Lab: {corp} (top-{top_k})"

        # Live updates are TTY-only; in non-TTY contexts (CI, pipes, captured
        # subprocesses) Rich.Live silently buffers everything until exit, which
        # makes the run look hung. Detect and fall back to plain progress prints.
        is_tty = sys.stdout.isatty()
        live = (
            Live(
                _build_table(title, top_k, strategies, per_strategy_rows),
                refresh_per_second=2,
                console=console,
            )
            if is_tty
            else None
        )
        if live is not None:
            live.start()

        # This loop no longer serializes the awaits. How much overlaps depends
        # on the strategy: one that searches on a worker thread overlaps fully,
        # one that blocks the loop does not. Results are recorded as they
        # complete so the live table keeps moving, then written out in
        # question order so the rows and the bootstrap draws stay the same
        # from run to run.
        limit = max(1, settings.benchmark_max_concurrent)
        semaphore = asyncio.Semaphore(limit)

        async def _retrieve(index, s, q):
            async with semaphore:
                try:
                    return index, await _retrieve_only(s, q.question, top_k, corpus=corp), None
                except RetrievalExecutionError as exc:
                    return index, None, exc
                except Exception as exc:  # noqa: BLE001 - a raw error must not drop finished rows
                    return index, None, exc

        def _row(q, trace, metrics) -> dict:
            hits_set = set(metrics.hits)

            def _is_hit(chunk):
                if metrics.fallback_doc_level:
                    return chunk.doc_id in hits_set
                return _match_expected(chunk.chunk_id, hits_set) is not None

            classes = {
                c.chunk_id: match_class(
                    c.chunk_id, hits_set, doc_level=metrics.fallback_doc_level, doc_id=c.doc_id
                )
                for c in trace.retrieved
            }

            return {
                "corpus": corp,
                "strategy": s.name,
                "question_id": q.id,
                "question": q.question,
                "recall_at_k": metrics.recall_at_k,
                "precision_at_k": metrics.precision_at_k,
                "hit_at_k": metrics.hit_at_k,
                "mrr": metrics.mrr,
                "ndcg_at_k": metrics.ndcg_at_k,
                "fallback_doc_level": metrics.fallback_doc_level,
                "hits": list(hits_set),
                # How each hit matched: a strict id, a parent label, or a
                # document-level fallback. A recall built on parent and doc
                # matches is looser than one built on strict ones.
                "match_classes": count_match_classes(classes.values()),
                "retrieved": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "rank": c.rank,
                        "score": c.score,
                        "source_strategy": c.source_strategy,
                        "is_hit": _is_hit(c),
                        "match": classes[c.chunk_id],
                    }
                    for c in trace.retrieved
                ],
            }

        def _error_row(q, exc) -> dict:
            cause = exc.__cause__ if isinstance(exc, RetrievalExecutionError) else None
            return {
                "corpus": corp,
                "strategy": s.name,
                "question_id": q.id,
                "question": q.question,
                "execution_error": {
                    "type": type(cause).__name__ if cause else type(exc).__name__,
                    "message": str(exc),
                },
            }

        try:
            for s in strategies:
                tasks = [asyncio.create_task(_retrieve(i, s, q)) for i, q in enumerate(questions)]
                finished: dict[int, tuple] = {}
                try:
                    for fut in asyncio.as_completed(tasks):
                        index, trace, exc = await fut
                        if exc is not None:
                            finished[index] = (None, None, exc)
                            console.print(
                                f"[red]ERROR[/red] {corp}/{s.name}/{questions[index].id}: {exc}"
                            )
                            continue
                        q = questions[index]
                        metrics = compute_all(
                            retrieved=trace.retrieved,
                            expected_ids=set(q.expected_chunks or []),
                            expected_relevance=_grades_of(q),
                            judged_nonrelevant=_negatives_of(q),
                            k=top_k,
                            expected_doc_ids=set(q.ground_truth.source_refs),
                        )
                        finished[index] = (trace, metrics, None)
                        per_strategy_rows[s.name].append(metrics)
                        if live is not None:
                            live.update(_build_table(title, top_k, strategies, per_strategy_rows))
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Rewrite this strategy's rows in question order.
                per_strategy_rows[s.name] = []
                for index, q in enumerate(questions):
                    trace, metrics, exc = finished[index]
                    if exc is not None:
                        per_strategy_errors[s.name] += 1
                        per_question_rows.append(_error_row(q, exc))
                        continue
                    per_strategy_rows[s.name].append(metrics)
                    per_strategy_tier_rows[s.name].append((int(q.tier or 0), metrics))
                    if not trace.retrieved:
                        per_strategy_empty[s.name] += 1
                    row = _row(q, trace, metrics)
                    for cls, n in row["match_classes"].items():
                        per_strategy_classes[s.name][cls] += n
                    per_question_rows.append(row)
                if live is not None:
                    live.update(_build_table(title, top_k, strategies, per_strategy_rows))
                if live is None:
                    n = len(per_strategy_rows[s.name])
                    avg_recall = (
                        sum(r.recall_at_k for r in per_strategy_rows[s.name]) / n if n else 0.0
                    )
                    console.print(f"  {s.name}: n={n}, mean Recall@{top_k}={avg_recall:.3f}")
        finally:
            if live is not None:
                live.stop()

        summary = {}
        for s in strategies:
            stats = _summarize_with_tiers(per_strategy_tier_rows[s.name])
            stats["empty_retrieval"] = per_strategy_empty[s.name]
            stats["execution_errors"] = per_strategy_errors[s.name]
            stats["match_classes"] = dict(per_strategy_classes[s.name])
            stats["strict_share_of_hits"] = _strict_share(per_strategy_classes[s.name])
            summary[s.name] = stats
        overall["corpora"][corp] = summary
        overall["manifests"][corp] = build_manifest(
            corp, questions, top_k=top_k, split=split, reference_free=True
        )

        if not run_ceiling:
            ceiling = {
                "status": "skipped",
                "reason": "no strategy in this run ranks over the naive_vector pool; "
                "pass --ceiling-k to force the diagnostic",
                "top_k": top_k,
                "ceiling_k": ceiling_k,
                "questions": 0,
            }
            console.print(f"[dim]Retrieval ceiling skipped for {corp}: {ceiling['reason']}[/dim]")
        else:
            ceiling = await _retrieval_ceiling(questions, top_k, ceiling_k, corpus=corp)
        overall["retrieval_ceiling"][corp] = ceiling
        if ceiling.get("status") == "error":
            error = ceiling["execution_error"]
            console.print(
                f"[red]ERROR[/red] retrieval ceiling for {corp}: "
                f"{error['type']}: {error['message']}"
            )
        elif ceiling.get("questions"):
            console.print(
                f"[cyan]Retrieval ceiling[/cyan] {corp}: naive Recall@{ceiling['top_k']}="
                f"{ceiling['recall_at_top_k']:.3f}, Recall@{ceiling['ceiling_k']}="
                f"{ceiling['recall_at_ceiling_k']:.3f}, ranking headroom="
                f"{ceiling['ranking_headroom']:.3f}"
            )

    return (
        1
        if any(
            stats["execution_errors"]
            for by_strategy in overall["corpora"].values()
            for stats in by_strategy.values()
        )
        or any(
            ceiling.get("status") == "error" for ceiling in overall["retrieval_ceiling"].values()
        )
        else 0
    )


async def run_retriever_lab_async(*args, **kwargs) -> int:
    """Asyncio entry point alias."""
    return await run_retriever_lab(*args, **kwargs)


def run_retriever_lab_sync(*args, **kwargs) -> int:
    """Synchronous wrapper for the CLI."""
    return asyncio.run(run_retriever_lab(*args, **kwargs))
