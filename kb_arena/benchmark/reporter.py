"""Benchmark reporter — generates markdown reports and summary.json from results."""

from __future__ import annotations

import json
from pathlib import Path

from kb_arena.models.benchmark import BenchmarkResult
from kb_arena.settings import settings

CORPUS_NAMES = ["python-stdlib", "kubernetes", "sec-edgar"]
STRATEGY_NAMES = ["naive_vector", "contextual_vector", "qna_pairs", "knowledge_graph", "hybrid"]
TIER_LABELS = {
    1: "Tier 1 — Factoid",
    2: "Tier 2 — Multi-entity",
    3: "Tier 3 — Comparative",
    4: "Tier 4 — Relational",
    5: "Tier 5 — Temporal",
}


def _load_results(corpus: str) -> list[BenchmarkResult]:
    results_dir = Path(settings.results_path)
    loaded = []
    corpora = CORPUS_NAMES if corpus == "all" else [corpus]
    for corp in corpora:
        for strat in STRATEGY_NAMES:
            path = results_dir / f"{corp}_{strat}.json"
            if path.exists():
                data = json.loads(path.read_text())
                loaded.append(BenchmarkResult.model_validate(data))
    return loaded


def _format_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _build_markdown(results: list[BenchmarkResult]) -> str:
    lines = ["# KB Arena Benchmark Report", ""]

    # Group by corpus
    by_corpus: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_corpus.setdefault(r.corpus, []).append(r)

    for corpus, corpus_results in by_corpus.items():
        lines += [f"## {corpus}", ""]

        # Overall accuracy table
        lines.append("### Overall Accuracy by Strategy")
        lines.append("")
        lines.append("| Strategy | Accuracy | Avg Latency (ms) | Total Cost | Cost/Correct |")
        lines.append("|----------|----------|------------------|------------|--------------|")
        for r in corpus_results:
            overall = (
                sum(r.accuracy_by_tier.values()) / len(r.accuracy_by_tier)
                if r.accuracy_by_tier
                else 0.0
            )
            lines.append(
                f"| {r.strategy} "
                f"| {_format_pct(overall)} "
                f"| {r.avg_latency_ms:.0f} "
                f"| ${r.total_cost_usd:.4f} "
                f"| ${r.cost_per_correct:.4f} |"
            )
        lines.append("")

        # Per-tier accuracy table
        tiers = sorted({t for r in corpus_results for t in r.accuracy_by_tier})
        if tiers:
            lines.append("### Accuracy by Tier")
            lines.append("")
            header = "| Strategy | " + " | ".join(TIER_LABELS.get(t, f"Tier {t}") for t in tiers) + " |"
            sep = "|----------|" + "|".join("----------" for _ in tiers) + "|"
            lines.append(header)
            lines.append(sep)
            for r in corpus_results:
                row = f"| {r.strategy} | "
                row += " | ".join(_format_pct(r.accuracy_by_tier.get(t, 0.0)) for t in tiers)
                row += " |"
                lines.append(row)
            lines.append("")

    return "\n".join(lines)


def _build_summary(results: list[BenchmarkResult]) -> dict:
    summary: dict = {"corpora": {}}
    for r in results:
        corp = r.corpus
        if corp not in summary["corpora"]:
            summary["corpora"][corp] = {}
        overall = (
            sum(r.accuracy_by_tier.values()) / len(r.accuracy_by_tier)
            if r.accuracy_by_tier
            else 0.0
        )
        summary["corpora"][corp][r.strategy] = {
            "accuracy_overall": round(overall, 4),
            "accuracy_by_tier": {str(k): round(v, 4) for k, v in r.accuracy_by_tier.items()},
            "avg_latency_ms": round(r.avg_latency_ms, 1),
            "total_cost_usd": round(r.total_cost_usd, 6),
            "cost_per_correct": round(r.cost_per_correct, 6),
            "total_questions": r.total_questions,
        }
    return summary


def generate_report(corpus: str = "all", output: str | None = None) -> None:
    """Load results JSON files and generate markdown report + summary.json."""
    results = _load_results(corpus)
    if not results:
        print(f"No results found in {settings.results_path}. Run benchmark first.")
        return

    markdown = _build_markdown(results)
    summary = _build_summary(results)

    results_dir = Path(settings.results_path)

    # Write markdown
    out_path = Path(output) if output else results_dir / "report.md"
    out_path.write_text(markdown)
    print(f"Report written to {out_path}")

    # Write summary JSON alongside the report
    summary_path = out_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")
