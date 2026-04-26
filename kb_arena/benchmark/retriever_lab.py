"""Retriever Lab — retrieval-only benchmark with classical IR metrics.

No LLM generation calls — strategies emit retrieval traces, IR metrics computed
against ground truth, results streamed to a Rich table. Roughly an order of
magnitude cheaper than `kb-arena benchmark` since the generator step is skipped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.live import Live
from rich.table import Table

from kb_arena.benchmark.ir_metrics import compute_all
from kb_arena.benchmark.questions import discover_corpora, load_questions
from kb_arena.models.benchmark import RetrievalMetrics
from kb_arena.models.retrieval import RetrievalTrace
from kb_arena.settings import settings
from kb_arena.strategies.base import Strategy

console = Console()
log = logging.getLogger(__name__)


async def _retrieve_only(
    strategy: Strategy, question_text: str, top_k: int
) -> RetrievalTrace:
    """Call the strategy's query() and extract the retrieval trace.

    Strategies don't expose a separate retrieve() entry point — we run query()
    and discard the answer. The retrieval trace is what we care about.
    """
    start = time.perf_counter()
    try:
        result = await strategy.query(question_text, top_k=top_k)
    except Exception as exc:
        log.warning("Strategy %s failed on question: %s", strategy.name, exc)
        return RetrievalTrace(
            query=question_text, retrieved=[], latency_ms=0.0, top_k=top_k
        )
    elapsed = (time.perf_counter() - start) * 1000
    if result.retrieval is not None:
        return result.retrieval
    return RetrievalTrace(
        query=question_text, retrieved=[], latency_ms=elapsed, top_k=top_k
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
            t.add_row(s.name, "—", "—", "—", "—", "—", "0")
    return t


def _summarize(records: list[RetrievalMetrics]) -> dict[str, float | int]:
    if not records:
        return {
            "mean_recall_at_k": 0.0,
            "mean_precision_at_k": 0.0,
            "mean_hit_at_k": 0.0,
            "mean_mrr": 0.0,
            "mean_ndcg_at_k": 0.0,
            "questions": 0,
        }
    n = len(records)
    return {
        "mean_recall_at_k": sum(r.recall_at_k for r in records) / n,
        "mean_precision_at_k": sum(r.precision_at_k for r in records) / n,
        "mean_hit_at_k": sum(r.hit_at_k for r in records) / n,
        "mean_mrr": sum(r.mrr for r in records) / n,
        "mean_ndcg_at_k": sum(r.ndcg_at_k for r in records) / n,
        "questions": n,
    }


async def run_retriever_lab(
    corpus: str = "all",
    strategies_filter: str = "all",
    top_k: int = 5,
    min_recall: float = 0.30,
) -> int:
    """Run retrieval-only benchmark. Returns 0 on success, 1 if min_recall floor breached."""
    from kb_arena.benchmark.runner import _load_strategies

    run_id = uuid4().hex[:8]
    timestamp = datetime.now(UTC).isoformat()

    corpora = discover_corpora() if corpus == "all" else [corpus]
    strategies = _load_strategies(strategies_filter)
    if not strategies:
        console.print("[red]No strategies available. Run build-vectors first.[/red]")
        return 1

    results_dir = Path(settings.results_path) / f"run_{run_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    overall: dict = {
        "run_id": run_id,
        "timestamp": timestamp,
        "top_k": top_k,
        "corpora": {},
    }
    per_question_rows: list[dict] = []

    console.print(f"[dim]Run ID: {run_id} | top-k: {top_k}[/dim]")

    for corp in corpora:
        try:
            questions = load_questions(corp)
        except FileNotFoundError:
            console.print(f"[yellow]No questions for {corp}; skipping[/yellow]")
            continue
        if not questions:
            continue

        per_strategy_rows: dict[str, list[RetrievalMetrics]] = {s.name: [] for s in strategies}
        title = f"Retriever Lab — {corp} (top-{top_k})"

        with Live(
            _build_table(title, top_k, strategies, per_strategy_rows),
            refresh_per_second=2,
            console=console,
        ) as live:
            for s in strategies:
                for q in questions:
                    trace = await _retrieve_only(s, q.question, top_k)
                    metrics = compute_all(
                        retrieved=trace.retrieved,
                        expected_ids=set(q.expected_chunks or []),
                        k=top_k,
                        expected_doc_ids=set(q.ground_truth.source_refs),
                    )
                    per_strategy_rows[s.name].append(metrics)
                    per_question_rows.append(
                        {
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
                            "retrieved": [
                                {
                                    "chunk_id": c.chunk_id,
                                    "doc_id": c.doc_id,
                                    "rank": c.rank,
                                    "score": c.score,
                                    "source_strategy": c.source_strategy,
                                    "is_hit": c.chunk_id in metrics.hits
                                    or c.doc_id in metrics.hits,
                                }
                                for c in trace.retrieved
                            ],
                        }
                    )
                    live.update(_build_table(title, top_k, strategies, per_strategy_rows))

        overall["corpora"][corp] = {
            s.name: _summarize(per_strategy_rows[s.name]) for s in strategies
        }

    json_path = results_dir / "retriever_lab.json"
    json_path.write_text(
        json.dumps(
            {**overall, "questions": per_question_rows}, indent=2, ensure_ascii=False
        )
    )

    md_lines = [f"# Retriever Lab — run {run_id}", "", f"Top-k: {top_k}", ""]
    for corp, by_strategy in overall["corpora"].items():
        md_lines += [
            f"## {corp}",
            "",
            f"| Strategy | Recall@{top_k} | P@{top_k} | Hit@{top_k} | MRR | NDCG@{top_k} | n |",
            "|---|---|---|---|---|---|---|",
        ]
        for sname, m in by_strategy.items():
            md_lines.append(
                f"| {sname} | {m['mean_recall_at_k']:.3f} | {m['mean_precision_at_k']:.3f} "
                f"| {m['mean_hit_at_k']:.3f} | {m['mean_mrr']:.3f} | {m['mean_ndcg_at_k']:.3f} "
                f"| {m['questions']} |"
            )
        md_lines.append("")
    md_path = results_dir / "retriever_lab.md"
    md_path.write_text("\n".join(md_lines))

    floor_violation = False
    for corp, by_strategy in overall["corpora"].items():
        for sname, m in by_strategy.items():
            if m["mean_recall_at_k"] < min_recall:
                console.print(
                    f"[red]FAIL[/red] {corp}/{sname} "
                    f"Recall@{top_k}={m['mean_recall_at_k']:.3f} < {min_recall}"
                )
                floor_violation = True

    console.print(f"[green]Run {run_id} written to {results_dir}/[/green]")
    return 1 if floor_violation else 0


async def run_retriever_lab_async(*args, **kwargs) -> int:
    """Asyncio entry point alias."""
    return await run_retriever_lab(*args, **kwargs)


def run_retriever_lab_sync(*args, **kwargs) -> int:
    """Synchronous wrapper for the CLI."""
    return asyncio.run(run_retriever_lab(*args, **kwargs))
