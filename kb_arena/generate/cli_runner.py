"""CLI runner for standalone Q&A pair generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from kb_arena.generate.qna import generate_pairs_for_documents
from kb_arena.llm.client import LLMClient
from kb_arena.strategies import load_documents

log = logging.getLogger(__name__)
console = Console()


async def run_generate_qa(corpus: str, output: str | None = None) -> Path:
    """Generate Q&A pairs for all documents in a corpus.

    Returns path to the output JSONL file.
    """
    documents = load_documents(corpus)
    if not documents:
        console.print(f"[red]No documents found for corpus={corpus}[/red]")
        console.print("Run ingest first: [bold]kb-arena ingest <path> --corpus {corpus}[/bold]")
        raise SystemExit(1)

    total_sections = sum(1 for doc in documents for s in doc.sections if s.content.strip())
    console.print(
        f"[bold]Generating Q&A pairs[/bold] for {len(documents)} doc(s), "
        f"{total_sections} section(s)"
    )

    llm = LLMClient()
    doc_stats: dict[str, int] = {}

    def on_progress(doc_id: str, pair_count: int) -> None:
        doc_stats[doc_id] = pair_count
        progress.advance(task)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed}/{task.total} docs"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(documents))
        pairs = await generate_pairs_for_documents(documents, llm, on_progress=on_progress)

    # Determine output path
    if output:
        out_path = Path(output)
    else:
        out_path = Path(f"datasets/{corpus}/qa-pairs/qa_pairs.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    console.print()
    console.print(f"[green]Generated {len(pairs)} Q&A pairs[/green] -> {out_path}")
    console.print()

    # Per-doc summary
    for doc_id, count in sorted(doc_stats.items()):
        console.print(f"  {doc_id}: {count} pairs")

    sections_covered = sum(1 for _ in doc_stats.values() if _ > 0)
    console.print()
    console.print(
        f"  Sections covered: {sections_covered}/{total_sections}, "
        f"avg {len(pairs) / max(len(documents), 1):.1f} pairs/doc"
    )

    return out_path
