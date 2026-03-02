"""Benchmark reporter — generates markdown reports and summary.json from results.

Enhanced with latency percentile tables, reliability section,
per-type accuracy, and cross-strategy ranking.
"""

from __future__ import annotations

import json
from pathlib import Path

from kb_arena.models.benchmark import BenchmarkResult
from kb_arena.settings import settings

CORPUS_NAMES = ["python-stdlib", "kubernetes", "sec-edgar"]
STRATEGY_NAMES = ["naive_vector", "contextual_vector", "qna_pairs", "knowledge_graph", "hybrid"]
TIER_LABELS = {
    1: "Tier 1 - Factoid",
    2: "Tier 2 - Multi-entity",
    3: "Tier 3 - Comparative",
    4: "Tier 4 - Relational",
    5: "Tier 5 - Temporal",
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


def _format_ms(v: float) -> str:
    return f"{v:.0f}"


def _build_markdown(results: list[BenchmarkResult]) -> str:
    lines = ["# KB Arena Benchmark Report", ""]

    # Group by corpus
    by_corpus: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_corpus.setdefault(r.corpus, []).append(r)

    for corpus, corpus_results in by_corpus.items():
        lines += [f"## {corpus}", ""]

        # 1. Overall Accuracy by Strategy
        lines.append("### Overall Accuracy by Strategy")
        lines.append("")
        lines.append(
            "| Strategy | Accuracy | Completeness | Faithfulness "
            "| Avg Latency (ms) | Total Cost | Cost/Correct |"
        )
        lines.append(
            "|----------|----------|--------------|------------- "
            "|------------------|------------|--------------|"
        )
        for r in corpus_results:
            overall_acc = (
                sum(r.accuracy_by_tier.values()) / len(r.accuracy_by_tier)
                if r.accuracy_by_tier
                else 0.0
            )
            overall_comp = (
                sum(r.completeness_by_tier.values()) / len(r.completeness_by_tier)
                if r.completeness_by_tier
                else 0.0
            )
            overall_faith = (
                sum(r.faithfulness_by_tier.values()) / len(r.faithfulness_by_tier)
                if r.faithfulness_by_tier
                else 0.0
            )
            lines.append(
                f"| {r.strategy} "
                f"| {_format_pct(overall_acc)} "
                f"| {_format_pct(overall_comp)} "
                f"| {_format_pct(overall_faith)} "
                f"| {r.avg_latency_ms:.0f} "
                f"| ${r.total_cost_usd:.4f} "
                f"| ${r.cost_per_correct:.4f} |"
            )
        lines.append("")

        # 2. Latency Distribution by Strategy
        lines.append("### Latency Distribution by Strategy")
        lines.append("")
        lines.append("| Strategy | Avg | p50 | p95 | p99 | Min | Max |")
        lines.append("|----------|-----|-----|-----|-----|-----|-----|")
        for r in corpus_results:
            lat = r.latency
            lines.append(
                f"| {r.strategy} "
                f"| {_format_ms(lat.avg_ms)} "
                f"| {_format_ms(lat.p50_ms)} "
                f"| {_format_ms(lat.p95_ms)} "
                f"| {_format_ms(lat.p99_ms)} "
                f"| {_format_ms(lat.min_ms)} "
                f"| {_format_ms(lat.max_ms)} |"
            )
        lines.append("")

        # 3. Response Reliability by Strategy
        lines.append("### Response Reliability by Strategy")
        lines.append("")
        lines.append(
            "| Strategy | Success Rate | Error Rate | Empty Rate "
            "| Avg Faithfulness | Avg Source Attr | Avg Entity Cov |"
        )
        lines.append(
            "|----------|-------------|------------|----------- "
            "|-----------------|----------------|----------------|"
        )
        for r in corpus_results:
            rel = r.reliability
            lines.append(
                f"| {r.strategy} "
                f"| {_format_pct(rel.success_rate)} "
                f"| {_format_pct(rel.error_rate)} "
                f"| {_format_pct(rel.empty_rate)} "
                f"| {_format_pct(rel.avg_faithfulness)} "
                f"| {_format_pct(rel.avg_source_attribution)} "
                f"| {_format_pct(rel.avg_entity_coverage)} |"
            )
        lines.append("")

        # 4. Accuracy by Tier
        tiers = sorted({t for r in corpus_results for t in r.accuracy_by_tier})
        if tiers:
            lines.append("### Accuracy by Tier")
            lines.append("")
            header = (
                "| Strategy | " + " | ".join(TIER_LABELS.get(t, f"Tier {t}") for t in tiers) + " |"
            )
            sep = "|----------|" + "|".join("----------" for _ in tiers) + "|"
            lines.append(header)
            lines.append(sep)
            for r in corpus_results:
                row = f"| {r.strategy} | "
                row += " | ".join(_format_pct(r.accuracy_by_tier.get(t, 0.0)) for t in tiers)
                row += " |"
                lines.append(row)
            lines.append("")

        # 5. Accuracy by Question Type
        all_types = sorted({t for r in corpus_results for t in r.accuracy_by_type})
        if all_types:
            lines.append("### Accuracy by Question Type")
            lines.append("")
            header = "| Strategy | " + " | ".join(all_types) + " |"
            sep = "|----------|" + "|".join("----------" for _ in all_types) + "|"
            lines.append(header)
            lines.append(sep)
            for r in corpus_results:
                row = f"| {r.strategy} | "
                row += " | ".join(_format_pct(r.accuracy_by_type.get(t, 0.0)) for t in all_types)
                row += " |"
                lines.append(row)
            lines.append("")

        # 6. Latency by Tier
        if tiers:
            lines.append("### Latency by Tier (p50 ms)")
            lines.append("")
            header = (
                "| Strategy | " + " | ".join(TIER_LABELS.get(t, f"Tier {t}") for t in tiers) + " |"
            )
            sep = "|----------|" + "|".join("----------" for _ in tiers) + "|"
            lines.append(header)
            lines.append(sep)
            for r in corpus_results:
                row = f"| {r.strategy} | "
                row += " | ".join(
                    _format_ms(r.latency_by_tier[t].p50_ms) if t in r.latency_by_tier else "-"
                    for t in tiers
                )
                row += " |"
                lines.append(row)
            lines.append("")

    # Cross-strategy ranking (all corpora)
    if len(results) > 1:
        lines += ["## Cross-Strategy Ranking", ""]
        _add_ranking_section(lines, results)

    return "\n".join(lines)


def _add_ranking_section(lines: list[str], results: list[BenchmarkResult]) -> None:
    """Rank strategies across dimensions."""
    by_strategy: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy, []).append(r)

    rankings: list[tuple[str, float, float, float, float]] = []
    for strat, strat_results in by_strategy.items():
        accs = []
        lats = []
        reliabs = []
        for r in strat_results:
            if r.accuracy_by_tier:
                accs.append(sum(r.accuracy_by_tier.values()) / len(r.accuracy_by_tier))
            lats.append(r.latency.p50_ms if r.latency.p50_ms > 0 else r.avg_latency_ms)
            reliabs.append(r.reliability.success_rate)

        avg_acc = sum(accs) / len(accs) if accs else 0.0
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        avg_rel = sum(reliabs) / len(reliabs) if reliabs else 0.0
        # Composite: weighted average (accuracy=0.5, reliability=0.3, latency=0.2)
        # Latency is inverted (lower is better) — normalize to 0-1 scale
        lat_score = max(0.0, 1.0 - (avg_lat / 10000))  # 10s = 0 score
        composite = 0.5 * avg_acc + 0.3 * avg_rel + 0.2 * lat_score
        rankings.append((strat, avg_acc, avg_lat, avg_rel, composite))

    rankings.sort(key=lambda x: x[4], reverse=True)

    lines.append("| Rank | Strategy | Avg Accuracy | p50 Latency (ms) | Success Rate | Composite |")
    lines.append("|------|----------|-------------|------------------|-------------|-----------|")
    for i, (strat, acc, lat, rel, comp) in enumerate(rankings, 1):
        lines.append(
            f"| {i} | {strat} "
            f"| {_format_pct(acc)} "
            f"| {_format_ms(lat)} "
            f"| {_format_pct(rel)} "
            f"| {comp:.3f} |"
        )
    lines.append("")
    lines.append("*Composite = 0.5 * Accuracy + 0.3 * Reliability + 0.2 * Latency Score*")
    lines.append("")


def _build_summary(results: list[BenchmarkResult]) -> dict:
    summary: dict = {"corpora": {}, "rankings": {}}
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
            "accuracy_by_type": {str(k): round(v, 4) for k, v in r.accuracy_by_type.items()},
            "completeness_by_tier": {
                str(k): round(v, 4) for k, v in r.completeness_by_tier.items()
            },
            "faithfulness_by_tier": {
                str(k): round(v, 4) for k, v in r.faithfulness_by_tier.items()
            },
            "latency": {
                "avg_ms": round(r.latency.avg_ms, 1),
                "p50_ms": round(r.latency.p50_ms, 1),
                "p95_ms": round(r.latency.p95_ms, 1),
                "p99_ms": round(r.latency.p99_ms, 1),
                "min_ms": round(r.latency.min_ms, 1),
                "max_ms": round(r.latency.max_ms, 1),
            },
            "latency_by_tier": {
                str(k): {
                    "avg_ms": round(v.avg_ms, 1),
                    "p50_ms": round(v.p50_ms, 1),
                    "p95_ms": round(v.p95_ms, 1),
                }
                for k, v in r.latency_by_tier.items()
            },
            "reliability": {
                "success_rate": round(r.reliability.success_rate, 4),
                "error_rate": round(r.reliability.error_rate, 4),
                "empty_rate": round(r.reliability.empty_rate, 4),
                "timeout_count": r.reliability.timeout_count,
                "avg_faithfulness": round(r.reliability.avg_faithfulness, 4),
                "avg_source_attribution": round(r.reliability.avg_source_attribution, 4),
                "avg_entity_coverage": round(r.reliability.avg_entity_coverage, 4),
                "avg_response_length": round(r.reliability.avg_response_length, 1),
            },
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
