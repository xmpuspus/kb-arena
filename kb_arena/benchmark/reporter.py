"""Benchmark reporter — generates markdown reports and summary.json from results.

Enhanced with latency percentile tables, reliability section,
per-type accuracy, and cross-strategy ranking.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from kb_arena.models.benchmark import BenchmarkResult
from kb_arena.settings import settings

console = Console()

STRATEGY_NAMES = [
    "naive_vector",
    "contextual_vector",
    "qna_pairs",
    "knowledge_graph",
    "hybrid",
    "raptor",
    "pageindex",
]
TIER_LABELS = {
    1: "Tier 1 - Factoid",
    2: "Tier 2 - Procedural",
    3: "Tier 3 - Comparative",
    4: "Tier 4 - Relational",
    5: "Tier 5 - Multi-hop",
}


def _discover_result_corpora() -> list[str]:
    """Find all corpora that have result files, regardless of question availability."""
    results_dir = Path(settings.results_path)
    if not results_dir.exists():
        return []
    corpora = set()
    for f in results_dir.glob("*.json"):
        # Filenames follow pattern: {corpus}_{strategy}.json
        for strat in STRATEGY_NAMES:
            suffix = f"_{strat}.json"
            if f.name.endswith(suffix):
                corpora.add(f.name[: -len(suffix)])
                break
    return sorted(corpora)


def _load_results(corpus: str | None = None, run_id: str | None = None) -> list[BenchmarkResult]:
    results_dir = Path(settings.results_path)
    if run_id:
        results_dir = results_dir / f"run_{run_id}"
    loaded = []
    resolved_corpus = corpus or "all"
    corpora = _discover_result_corpora() if resolved_corpus == "all" else [resolved_corpus]
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

        # 1b. Retrieval Quality (IR metrics) — only render when populated
        ir_results = [r for r in corpus_results if r.mean_recall_at_k or r.mean_mrr]
        if ir_results:
            k = ir_results[0].ir_top_k
            lines.append(f"### Retrieval Quality (top-{k})")
            lines.append("")
            lines.append(
                f"| Strategy | Recall@{k} | Precision@{k} | Hit@{k} | MRR | NDCG@{k} |"
            )
            lines.append(
                "|----------|------------|---------------|---------|-----|----------|"
            )
            for r in ir_results:
                lines.append(
                    f"| {r.strategy} "
                    f"| {_format_pct(r.mean_recall_at_k)} "
                    f"| {_format_pct(r.mean_precision_at_k)} "
                    f"| {_format_pct(r.mean_hit_at_k)} "
                    f"| {r.mean_mrr:.3f} "
                    f"| {r.mean_ndcg_at_k:.3f} |"
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


def _build_csv(results: list[BenchmarkResult]) -> str:
    """CSV export - one row per strategy."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "corpus",
            "strategy",
            "run_id",
            "accuracy",
            "completeness",
            "faithfulness",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "success_rate",
            "error_rate",
            "total_cost_usd",
        ]
    )
    for r in results:
        acc = sum(r.accuracy_by_tier.values()) / max(len(r.accuracy_by_tier), 1)
        comp = sum(r.completeness_by_tier.values()) / max(len(r.completeness_by_tier), 1)
        faith = sum(r.faithfulness_by_tier.values()) / max(len(r.faithfulness_by_tier), 1)
        writer.writerow(
            [
                r.corpus,
                r.strategy,
                r.run_id,
                f"{acc:.4f}",
                f"{comp:.4f}",
                f"{faith:.4f}",
                f"{r.latency.avg_ms:.1f}" if r.latency else "0",
                f"{r.latency.p50_ms:.1f}" if r.latency else "0",
                f"{r.latency.p95_ms:.1f}" if r.latency else "0",
                f"{r.reliability.success_rate:.4f}" if r.reliability else "0",
                f"{r.reliability.error_rate:.4f}" if r.reliability else "0",
                f"{r.total_cost_usd:.6f}",
            ]
        )
    return buf.getvalue()


def _build_html(results: list[BenchmarkResult], corpus: str = "") -> str:
    """Self-contained HTML report card."""
    rows = []
    for r in results:
        acc = sum(r.accuracy_by_tier.values()) / max(len(r.accuracy_by_tier), 1)
        comp = sum(r.completeness_by_tier.values()) / max(len(r.completeness_by_tier), 1)
        faith = sum(r.faithfulness_by_tier.values()) / max(len(r.faithfulness_by_tier), 1)
        lat = r.latency.avg_ms if r.latency else 0
        cost = r.total_cost_usd
        rows.append(f"""<tr>
            <td>{r.strategy}</td>
            <td>{acc:.1%}</td><td>{comp:.1%}</td><td>{faith:.1%}</td>
            <td>{lat:.0f}ms</td><td>${cost:.4f}</td>
        </tr>""")

    table_html = "\n".join(rows)
    title = f"KB Arena Report - {corpus}" if corpus else "KB Arena Report"

    font_stack = '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif'
    body_style = (
        f"font-family:{font_stack};max-width:1000px;margin:0 auto;"
        "padding:2rem;background:#f8fafc;color:#1e293b"
    )
    table_style = (
        "border-collapse:collapse;width:100%;background:white;"
        "border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1)"
    )
    thead_row = (
        "<tr><th>Strategy</th><th>Accuracy</th><th>Completeness</th>"
        "<th>Faithfulness</th><th>Avg Latency</th><th>Cost</th></tr>"
    )
    footer_link = '<a href="https://github.com/xpuspus/kb-arena">KB Arena</a>'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{{body_style}}}
h1{{font-size:1.5rem;margin-bottom:0.5rem}}
.subtitle{{color:#64748b;margin-bottom:2rem}}
table{{{table_style}}}
th{{background:#1e293b;color:white;padding:0.75rem 1rem;text-align:left;font-weight:500}}
td{{padding:0.75rem 1rem;border-bottom:1px solid #e2e8f0}}
tr:last-child td{{border-bottom:none}}
tr:hover{{background:#f1f5f9}}
.footer{{margin-top:2rem;color:#94a3b8;font-size:0.875rem}}
</style></head><body>
<h1>{title}</h1>
<p class="subtitle">Generated by KB Arena - retrieval strategy benchmark</p>
<table>
<thead>{thead_row}</thead>
<tbody>{table_html}</tbody>
</table>
<p class="footer">Benchmark results from {len(results)} strategies. \
Generated with {footer_link}.</p>
</body></html>"""


def generate_report(corpus: str = "all", output: str | None = None) -> None:
    """Load results JSON files and generate markdown report + summary.json."""
    results = _load_results(corpus)
    if not results:
        console.print(f"No results found in {settings.results_path}. Run benchmark first.")
        return

    markdown = _build_markdown(results)
    summary = _build_summary(results)

    results_dir = Path(settings.results_path)

    # Write markdown
    out_path = Path(output) if output else results_dir / "report.md"
    out_path.write_text(markdown)
    console.print(f"Report written to {out_path}")

    # Write summary JSON alongside the report
    summary_path = out_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.print(f"Summary written to {summary_path}")
