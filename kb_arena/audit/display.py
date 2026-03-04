"""Rich console display for audit and fix reports."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kb_arena.audit.analyzer import AuditReport

console = Console()


def display_audit_report(report: AuditReport, output: str | None = None) -> None:
    """Display audit results with Rich tables."""
    console.print()
    console.print(
        f"[bold]Audited {report.total_sections} sections, "
        f"{report.total_questions} questions — "
        f"overall accuracy: {report.overall_accuracy:.0%}[/bold]"
    )
    console.print()

    # Strong sections
    if report.strong:
        table = Table(title="Strong sections (>= 70%)", style="green")
        table.add_column("Section", style="bold")
        table.add_column("Doc")
        table.add_column("Accuracy", justify="right")
        table.add_column("Questions", justify="right")
        for s in sorted(report.strong, key=lambda x: x.avg_accuracy, reverse=True):
            table.add_row(
                s.section_title,
                s.doc_id,
                f"{s.avg_accuracy:.0%}",
                str(s.questions_tested),
            )
        console.print(table)
        console.print()

    # Weak sections
    if report.weak:
        table = Table(title="Weak sections (30-70%)", style="yellow")
        table.add_column("Section", style="bold")
        table.add_column("Doc")
        table.add_column("Accuracy", justify="right")
        table.add_column("Worst Question")
        for s in sorted(report.weak, key=lambda x: x.avg_accuracy):
            table.add_row(
                s.section_title,
                s.doc_id,
                f"{s.avg_accuracy:.0%}",
                s.worst_question[:80],
            )
        console.print(table)
        console.print()

    # Gap sections
    if report.gaps:
        table = Table(title="Gap sections (< 30%)", style="red")
        table.add_column("Section", style="bold")
        table.add_column("Doc")
        table.add_column("Accuracy", justify="right")
        table.add_column("Worst Question")
        for s in sorted(report.gaps, key=lambda x: x.avg_accuracy):
            table.add_row(
                s.section_title,
                s.doc_id,
                f"{s.avg_accuracy:.0%}",
                s.worst_question[:80],
            )
        console.print(table)
        console.print()

    # Uncovered sections
    if report.uncovered:
        console.print(f"[dim]Uncovered sections ({len(report.uncovered)}):[/dim]")
        for name in report.uncovered[:20]:
            console.print(f"  [dim]- {name}[/dim]")
        if len(report.uncovered) > 20:
            console.print(f"  [dim]... and {len(report.uncovered) - 20} more[/dim]")
        console.print()

    # Write JSON output
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus": report.corpus,
            "total_sections": report.total_sections,
            "total_questions": report.total_questions,
            "overall_accuracy": report.overall_accuracy,
            "strong": [
                {
                    "section_id": s.section_id,
                    "section_title": s.section_title,
                    "doc_id": s.doc_id,
                    "avg_accuracy": s.avg_accuracy,
                    "questions_tested": s.questions_tested,
                }
                for s in report.strong
            ],
            "weak": [
                {
                    "section_id": s.section_id,
                    "section_title": s.section_title,
                    "doc_id": s.doc_id,
                    "avg_accuracy": s.avg_accuracy,
                    "worst_question": s.worst_question,
                    "questions_tested": s.questions_tested,
                }
                for s in report.weak
            ],
            "gaps": [
                {
                    "section_id": s.section_id,
                    "section_title": s.section_title,
                    "doc_id": s.doc_id,
                    "avg_accuracy": s.avg_accuracy,
                    "worst_question": s.worst_question,
                    "questions_tested": s.questions_tested,
                }
                for s in report.gaps
            ],
            "uncovered": report.uncovered,
        }
        out_path.write_text(json.dumps(data, indent=2))
        console.print(f"[green]Report written to {out_path}[/green]")


def display_fix_report(report, output: str | None = None) -> None:
    """Display fix recommendations as numbered panels."""

    console.print()
    console.print(f"[bold]{report.total_fixes} fix recommendation(s)[/bold]")
    console.print()

    for rec in report.recommendations:
        priority_label = {1: "HIGH", 2: "HIGH", 3: "MEDIUM"}.get(rec.priority, "LOW")
        priority_color = {1: "red", 2: "red", 3: "yellow"}.get(rec.priority, "dim")

        content = (
            f"[bold]Doc:[/bold] {rec.doc_id}\n"
            f"[bold]Diagnosis:[/bold] {rec.diagnosis}\n\n"
            f"[bold]Suggested addition:[/bold]\n"
            f"  [italic]{rec.suggested_content}[/italic]\n\n"
            f"[bold]Add after:[/bold] {rec.placement}\n"
            f"[bold]Impact:[/bold] {rec.estimated_impact}\n"
            f"[bold]Current accuracy:[/bold] {rec.current_accuracy:.0%}"
        )

        if rec.failing_questions:
            content += "\n[bold]Failing questions:[/bold]"
            for q in rec.failing_questions[:3]:
                content += f"\n  - {q[:100]}"

        console.print(
            Panel(
                content,
                title=(
                    f"Fix #{rec.priority} ([{priority_color}]"
                    f"{priority_label}[/{priority_color}]) — {rec.section_title}"
                ),
                border_style=priority_color,
            )
        )
        console.print()

    # Write markdown output
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# Fix Recommendations for {report.corpus}\n"]
        for rec in report.recommendations:
            lines.append(f"## Fix #{rec.priority}: {rec.section_title}")
            lines.append(f"**Doc:** {rec.doc_id}")
            lines.append(f"**Current accuracy:** {rec.current_accuracy:.0%}")
            lines.append(f"**Diagnosis:** {rec.diagnosis}\n")
            lines.append(f"**Suggested addition:**\n> {rec.suggested_content}\n")
            lines.append(f"**Placement:** {rec.placement}")
            lines.append(f"**Impact:** {rec.estimated_impact}\n")
            if rec.failing_questions:
                lines.append("**Failing questions:**")
                for q in rec.failing_questions:
                    lines.append(f"- {q}")
            lines.append("")
        out_path.write_text("\n".join(lines))
        console.print(f"[green]Fixes written to {out_path}[/green]")
