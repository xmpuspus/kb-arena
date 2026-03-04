"""Ingestion pipeline orchestrator.

Reads raw documents from a directory, selects the appropriate parser
per file extension (or uses the explicitly specified format), and writes
one Document JSON object per line to datasets/{corpus}/processed/documents.jsonl.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from kb_arena.ingest.parsers import PARSERS
from kb_arena.models.document import Document
from kb_arena.settings import settings

console = Console()
log = logging.getLogger(__name__)

# Extension → parser key mapping for auto-detect
_EXT_MAP: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "markdown",  # MarkdownParser handles RST
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "plaintext",
    ".text": "plaintext",
    ".csv": "csv",
    ".tsv": "csv",
}


def _detect_format(path: Path, corpus: str) -> str:
    return _EXT_MAP.get(path.suffix.lower(), "html")


def run_ingest(path: str, corpus: str = "custom", format: str = "auto") -> None:
    """Parse raw documents and write JSONL to datasets/{corpus}/processed/."""
    src = Path(path)
    if not src.exists():
        console.print(f"[red]Path does not exist: {src}[/red]")
        raise SystemExit(1)

    # Collect files — if path is a file, wrap it; otherwise glob recursively
    supported_exts = set(_EXT_MAP.keys())
    if src.is_file():
        if src.suffix.lower() not in supported_exts and format == "auto":
            console.print(f"[yellow]Unsupported file type: {src.suffix}[/yellow]")
            return
        files = [src]
    else:
        files = [f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in supported_exts]

    if not files:
        console.print(f"[yellow]No supported files found in {src}[/yellow]")
        return

    out_dir = Path(settings.datasets_path) / corpus / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "documents.jsonl"

    total_docs = 0
    total_sections = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Ingesting [bold]{corpus}[/bold]", total=len(files))

        with out_path.open("w", encoding="utf-8") as fout:
            for file in files:
                fmt = format if format != "auto" else _detect_format(file, corpus)
                parser_cls = PARSERS.get(fmt)

                if parser_cls is None:
                    log.warning("No parser for format %r, skipping %s", fmt, file)
                    progress.advance(task)
                    continue

                try:
                    parser = parser_cls()
                    docs: list[Document] = parser.parse(file, corpus)
                    for doc in docs:
                        fout.write(doc.model_dump_json())
                        fout.write("\n")
                        total_docs += 1
                        total_sections += len(doc.sections)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to parse %s: %s", file, exc)

                progress.advance(task)

    console.print(
        f"[green]Done.[/green] {total_docs} documents, {total_sections} sections "
        f"→ [bold]{out_path}[/bold]"
    )


def run_ingest_special(
    source: str,
    corpus: str = "custom",
    format: str = "web",
    max_depth: int = 3,
    max_pages: int = 50,
) -> None:
    """Ingest from URL or GitHub repo source."""
    out_dir = Path(settings.datasets_path) / corpus / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "documents.jsonl"

    parser_cls = PARSERS.get(format)
    if parser_cls is None:
        console.print(f"[red]No parser for format: {format}[/red]")
        raise SystemExit(1)

    console.print(f"Ingesting from [bold]{source}[/bold] as [bold]{format}[/bold]...")

    try:
        if format == "web":
            parser = parser_cls(max_depth=max_depth, max_pages=max_pages)
        else:
            parser = parser_cls()

        docs: list[Document] = parser.parse(Path(source), corpus)
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from None
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to ingest: {exc}[/red]")
        raise SystemExit(1) from None

    if not docs:
        console.print("[yellow]No documents extracted from source.[/yellow]")
        return

    total_sections = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for doc in docs:
            fout.write(doc.model_dump_json())
            fout.write("\n")
            total_sections += len(doc.sections)

    console.print(
        f"[green]Done.[/green] {len(docs)} documents, {total_sections} sections "
        f"→ [bold]{out_path}[/bold]"
    )
