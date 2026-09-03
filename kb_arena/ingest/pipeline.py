"""Ingestion pipeline orchestrator.

Reads raw documents from a directory, selects the appropriate parser
per file extension (or uses the explicitly specified format), and writes
one Document JSON object per line to datasets/{corpus}/processed/documents.jsonl.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from kb_arena.ingest.parsers import PARSERS
from kb_arena.models.document import Document
from kb_arena.settings import settings

console = Console()
log = logging.getLogger(__name__)

# Map extensions to parser keys for automatic detection.
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
SUPPORTED_EXTENSIONS = frozenset(_EXT_MAP)


def is_http_url(source: str) -> bool:
    """True for an http or https URL. The scheme compares case-insensitively,
    so `HTTPS://docs.example.com` routes to the web parser like the lowercase
    form, instead of falling through as a filesystem path.
    """
    return urlsplit(source).scheme.lower() in ("http", "https")


def _detect_format(path: Path, corpus: str) -> str:
    return _EXT_MAP.get(path.suffix.lower(), "html")


def run_ingest(path: str, corpus: str = "custom", format: str = "auto") -> int:
    """Parse raw documents and write JSONL to datasets/{corpus}/processed/."""
    src = Path(path)
    if not src.exists():
        console.print(f"[red]Path does not exist: {src}[/red]")
        raise SystemExit(1)

    # Wrap a single file or collect supported files recursively.
    if src.is_file():
        if src.suffix.lower() not in SUPPORTED_EXTENSIONS and format == "auto":
            console.print(f"[yellow]Unsupported file type: {src.suffix}[/yellow]")
            return 0
        files = [src]
    else:
        files = [
            f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    if not files:
        console.print(f"[yellow]No supported files found in {src}[/yellow]")
        return 0

    out_dir = Path(settings.datasets_path) / corpus / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "documents.jsonl"
    staged_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=out_dir,
        prefix=".documents.",
        suffix=".tmp",
        delete=False,
    )
    staged_path = Path(staged_file.name)

    total_docs = 0
    total_sections = 0
    failed_files: list[Path] = []

    published = False
    try:
        with (
            staged_file as fout,
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} files"),
                console=console,
            ) as progress,
        ):
            task = progress.add_task(f"Ingesting [bold]{corpus}[/bold]", total=len(files))
            for file in files:
                fmt = format if format != "auto" else _detect_format(file, corpus)
                parser_cls = PARSERS.get(fmt)

                if parser_cls is None:
                    log.warning("No parser for format %r: %s", fmt, file)
                    failed_files.append(file)
                    progress.advance(task)
                    continue

                try:
                    parser = parser_cls()
                    docs: list[Document] = parser.parse(file, corpus)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed to parse %s: %s", file, exc)
                    failed_files.append(file)
                    progress.advance(task)
                    continue

                if not docs:
                    log.warning("Parser returned no documents for %s", file)
                    failed_files.append(file)
                    progress.advance(task)
                    continue

                for doc in docs:
                    fout.write(doc.model_dump_json())
                    fout.write("\n")
                    total_docs += 1
                    total_sections += len(doc.sections)

                progress.advance(task)

        if failed_files:
            console.print(
                f"[red]Ingestion failed for {len(failed_files)} of {len(files)} files; "
                "the existing corpus was preserved.[/red]"
            )
            raise SystemExit(1)

        if total_docs > 0:
            staged_path.replace(out_path)
            published = True
    finally:
        if not published:
            staged_path.unlink(missing_ok=True)

    console.print(
        f"[green]Done.[/green] {total_docs} documents, {total_sections} sections "
        f"-> [bold]{out_path}[/bold]"
    )
    return total_docs


def run_ingest_special(
    source: str,
    corpus: str = "custom",
    format: str = "web",
    max_depth: int = 3,
    max_pages: int = 50,
) -> int:
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

        # Path() collapses "https://" to "https:/", and the web parser then
        # treats the result as a filename and returns nothing. A URL has to
        # reach it as the string the user typed.
        is_url = format == "web" and is_http_url(source)
        docs: list[Document] = parser.parse(source if is_url else Path(source), corpus)
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from None
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to ingest: {exc}[/red]")
        raise SystemExit(1) from None

    if not docs:
        console.print("[yellow]No documents extracted from source.[/yellow]")
        return 0

    staged_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=out_dir,
        prefix=".documents.",
        suffix=".tmp",
        delete=False,
    )
    staged_path = Path(staged_file.name)
    total_sections = 0
    published = False
    try:
        with staged_file as fout:
            for doc in docs:
                fout.write(doc.model_dump_json())
                fout.write("\n")
                total_sections += len(doc.sections)
        staged_path.replace(out_path)
        published = True
    finally:
        if not published:
            staged_path.unlink(missing_ok=True)

    console.print(
        f"[green]Done.[/green] {len(docs)} documents, {total_sections} sections "
        f"-> [bold]{out_path}[/bold]"
    )
    return len(docs)
