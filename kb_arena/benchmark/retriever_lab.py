"""Retriever Lab retrieval-only benchmark with classical IR metrics.

Strategies emit retrieval traces without LLM generation. IR metrics are computed
against ground truth, results streamed to a Rich table. Roughly an order of
magnitude cheaper than `kb-arena benchmark` since the generator step is skipped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
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


from kb_arena.benchmark.ir_metrics import _match_expected, compute_all  # noqa: E402
from kb_arena.benchmark.questions import discover_corpora, load_questions  # noqa: E402
from kb_arena.models.benchmark import RetrievalMetrics  # noqa: E402
from kb_arena.models.retrieval import RetrievalTrace  # noqa: E402
from kb_arena.settings import settings  # noqa: E402
from kb_arena.strategies.base import Strategy  # noqa: E402

console = Console()
log = logging.getLogger(__name__)


async def _stub_generate(self, *args, **kwargs):
    """Replacement LLMClient.generate that returns an empty, zero-cost response."""
    from kb_arena.llm.client import LLMResponse

    return LLMResponse(text="", input_tokens=0, output_tokens=0, cost_usd=0.0)


async def _stub_extract(self, *args, **kwargs):
    from kb_arena.llm.client import LLMResponse

    return LLMResponse(text="", input_tokens=0, output_tokens=0, cost_usd=0.0)


async def _stub_classify(self, *args, **kwargs):
    return ""


class _PatchLLMClient:
    """Context manager that replaces LLMClient methods with no-op stubs.

    Strategies invoke query() which always runs LLM generation. retriever-lab
    needs only the retrieval trace, so we patch the LLM globally to make
    generation effectively free while retrieval still runs normally.
    """

    def __enter__(self):
        from kb_arena.llm import client as llm_module

        self._originals = {
            "generate": llm_module.LLMClient.generate,
            "extract": llm_module.LLMClient.extract,
            "classify": llm_module.LLMClient.classify,
        }
        llm_module.LLMClient.generate = _stub_generate
        llm_module.LLMClient.extract = _stub_extract
        llm_module.LLMClient.classify = _stub_classify
        return self

    def __exit__(self, exc_type, exc, tb):
        from kb_arena.llm import client as llm_module

        for name, original in self._originals.items():
            setattr(llm_module.LLMClient, name, original)
        return False


async def _retrieve_only(strategy: Strategy, question_text: str, top_k: int) -> RetrievalTrace:
    """Run query() under the LLM-stub patch and return the retrieval trace."""
    start = time.perf_counter()
    try:
        result = await strategy.query(question_text, top_k=top_k)
    except Exception as exc:
        log.warning("Strategy %s failed on question: %s", strategy.name, exc)
        return RetrievalTrace(query=question_text, retrieved=[], latency_ms=0.0, top_k=top_k)
    elapsed = (time.perf_counter() - start) * 1000
    if result.retrieval is not None:
        return result.retrieval
    return RetrievalTrace(query=question_text, retrieved=[], latency_ms=elapsed, top_k=top_k)


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
    """Percentile bootstrap 95% CI on the mean; degenerate-band when scipy
    missing or all values identical so the summary never crashes."""
    if not values:
        return (0.0, 0.0)
    if all(v == values[0] for v in values):
        return (values[0], values[0])
    try:
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
    except Exception:  # noqa: BLE001
        m = sum(values) / len(values)
        return (m, m)


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


async def _retrieval_ceiling(questions, top_k: int, ceiling_k: int) -> dict:
    """Base-retriever (naive_vector) recall at top_k vs ceiling_k over `questions`.

    naive_vector is the pool every vector reranker shares, so its recall ceiling
    is the shared retrieval ceiling. Pure retrieval (LLM is already stubbed by the
    caller); one extra deep query per question, no generation cost.
    """
    from kb_arena.strategies import get_strategy

    try:
        base = get_strategy("naive_vector")
    except Exception as exc:  # noqa: BLE001 - ceiling is a best-effort diagnostic
        log.warning("Retrieval ceiling skipped (naive_vector unavailable): %s", exc)
        return {}

    top_recalls: list[float] = []
    ceiling_recalls: list[float] = []
    for q in questions:
        trace = await _retrieve_only(base, q.question, ceiling_k)
        retrieved = trace.retrieved
        expected = set(q.expected_chunks or [])
        doc_ids = set(q.ground_truth.source_refs)
        m_top = compute_all(
            retrieved=retrieved[:top_k], expected_ids=expected, k=top_k, expected_doc_ids=doc_ids
        )
        m_ceil = compute_all(
            retrieved=retrieved[:ceiling_k],
            expected_ids=expected,
            k=ceiling_k,
            expected_doc_ids=doc_ids,
        )
        top_recalls.append(m_top.recall_at_k)
        ceiling_recalls.append(m_ceil.recall_at_k)

    return _summarize_ceiling(top_recalls, ceiling_recalls, top_k, ceiling_k)


def _empty_trace_failures(by_strategy: dict, threshold: float = 0.5) -> list[str]:
    """Strategy names whose empty-retrieval count is >= `threshold` of questions.

    An empty trace means the strategy raised and `_retrieve_only` swallowed it,
    not that it retrieved irrelevant chunks. A strategy empty on most questions is
    crashing rather than merely scoring low, so it is handled separately from the recall floor.
    """
    failures: list[str] = []
    for sname, m in by_strategy.items():
        n_q = m.get("questions", 0)
        n_empty = m.get("empty_retrieval", 0)
        if n_q and n_empty >= threshold * n_q:
            failures.append(sname)
    return failures


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

    run_id = uuid4().hex[:8]
    timestamp = datetime.now(UTC).isoformat()
    ceiling_k = ceiling_k if ceiling_k and ceiling_k > top_k else top_k * 4

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
        "ceiling_k": ceiling_k,
        "question_split": split or "all",
        "corpora": {},
        "retrieval_ceiling": {},
    }
    per_question_rows: list[dict] = []

    console.print(f"[dim]Run ID: {run_id} | top-k: {top_k} | ceiling-k: {ceiling_k}[/dim]")

    _llm_patch = _PatchLLMClient()
    _llm_patch.__enter__()
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
        )
    finally:
        _llm_patch.__exit__(None, None, None)

    json_path = results_dir / "retriever_lab.json"
    json_path.write_text(
        json.dumps({**overall, "questions": per_question_rows}, indent=2, ensure_ascii=False)
    )

    md_lines = [f"# Retriever Lab: run {run_id}", "", f"Top-k: {top_k}", ""]
    for corp_name, by_strategy in overall["corpora"].items():
        md_lines += [
            f"## {corp_name}",
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
    for corp_name, by_strategy in overall["corpora"].items():
        # Dead-strategy guard runs first: empty retrieval on most questions is a
        # crash signature (the strategy raised and _retrieve_only swallowed it
        # into an empty trace), NOT a genuine low-recall result. Flag it
        # distinctly so a silently-broken strategy can't masquerade as "0 recall"
        # This is how the rerank_vector c.source bug previously went undetected.
        crashed = set(_empty_trace_failures(by_strategy))
        for sname, m in by_strategy.items():
            if sname in crashed:
                console.print(
                    f"[red]FAIL[/red] {corp_name}/{sname} returned EMPTY retrieval for "
                    f"{m['empty_retrieval']}/{m['questions']} questions, likely a crash, "
                    "not a result"
                )
                floor_violation = True
            elif m["mean_recall_at_k"] < min_recall:
                console.print(
                    f"[red]FAIL[/red] {corp_name}/{sname} "
                    f"Recall@{top_k}={m['mean_recall_at_k']:.3f} < {min_recall}"
                )
                floor_violation = True

    console.print(f"[green]Run {run_id} written to {results_dir}/[/green]")
    return 1 if floor_violation else _exit_code


async def _run_corpora_loop(
    corpora,
    strategies,
    top_k,
    min_recall,
    overall,
    per_question_rows,
    ceiling_k,
    split,
):
    for corp in corpora:
        try:
            questions = load_questions(corp, split=split)
        except FileNotFoundError:
            console.print(f"[yellow]No questions for {corp}; skipping[/yellow]")
            continue
        if not questions:
            continue

        per_strategy_rows: dict[str, list[RetrievalMetrics]] = {s.name: [] for s in strategies}
        per_strategy_tier_rows: dict[str, list[tuple[int, RetrievalMetrics]]] = {
            s.name: [] for s in strategies
        }
        # Crash signature: count questions where a strategy surfaced zero chunks.
        per_strategy_empty: dict[str, int] = {s.name: 0 for s in strategies}
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

        try:
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
                    per_strategy_tier_rows[s.name].append((int(q.tier or 0), metrics))
                    if not trace.retrieved:
                        per_strategy_empty[s.name] += 1
                    hits_set = set(metrics.hits)

                    def _is_hit(chunk):
                        if metrics.fallback_doc_level:
                            return chunk.doc_id in hits_set
                        return _match_expected(chunk.chunk_id, hits_set) is not None

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
                            "hits": list(hits_set),
                            "retrieved": [
                                {
                                    "chunk_id": c.chunk_id,
                                    "doc_id": c.doc_id,
                                    "rank": c.rank,
                                    "score": c.score,
                                    "source_strategy": c.source_strategy,
                                    "is_hit": _is_hit(c),
                                }
                                for c in trace.retrieved
                            ],
                        }
                    )
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
            summary[s.name] = stats
        overall["corpora"][corp] = summary

        ceiling = await _retrieval_ceiling(questions, top_k, ceiling_k)
        if ceiling.get("questions"):
            overall["retrieval_ceiling"][corp] = ceiling
            console.print(
                f"[cyan]Retrieval ceiling[/cyan] {corp}: naive Recall@{ceiling['top_k']}="
                f"{ceiling['recall_at_top_k']:.3f}, Recall@{ceiling['ceiling_k']}="
                f"{ceiling['recall_at_ceiling_k']:.3f}, ranking headroom="
                f"{ceiling['ranking_headroom']:.3f}"
            )

    return 0


async def run_retriever_lab_async(*args, **kwargs) -> int:
    """Asyncio entry point alias."""
    return await run_retriever_lab(*args, **kwargs)


def run_retriever_lab_sync(*args, **kwargs) -> int:
    """Synchronous wrapper for the CLI."""
    return asyncio.run(run_retriever_lab(*args, **kwargs))
