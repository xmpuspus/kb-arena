"""GitHub repository parser — clones and ingests supported doc files."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from kb_arena.ingest.parsers.utils import slugify
from kb_arena.models.document import Document

log = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "venv",
    ".venv",
    "dist",
    "build",
    ".eggs",
}
_SUPPORTED_EXTS = {".md", ".markdown", ".rst", ".txt", ".html", ".htm"}


def _clone_repo(repo_spec: str, target_dir: Path) -> Path:
    if repo_spec.startswith(("http://", "https://", "git@")):
        url = repo_spec
    else:
        url = f"https://github.com/{repo_spec}.git"

    if not (url.startswith("https://") or url.startswith("git@")):
        raise ValueError(f"Unsupported repository URL scheme: {url}")

    log.info("Cloning %s into %s", url, target_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--single-branch", "--", url, str(target_dir)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return target_dir


def _collect_files(repo_dir: Path) -> list[Path]:
    files = []
    for f in repo_dir.rglob("*"):
        if any(skip in f.parts for skip in _SKIP_DIRS):
            continue
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS:
            files.append(f)
    return sorted(files)


class GitHubParser:
    def parse(self, path: Path, corpus: str) -> list[Document]:
        repo_spec = str(path)

        if repo_spec.startswith("github:"):
            repo_spec = repo_spec[7:]
            return self._parse_remote(repo_spec, corpus)

        if path.is_dir() and (path / ".git").exists():
            return self._parse_local(path, corpus)

        try:
            repo_spec = path.read_text().strip()
            if "/" in repo_spec:
                return self._parse_remote(repo_spec, corpus)
        except Exception:  # noqa: BLE001
            pass

        return []

    def _parse_remote(self, repo_spec: str, corpus: str) -> list[Document]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="kb-arena-github-"))
        try:
            _clone_repo(repo_spec, tmp_dir)
            return self._parse_local(tmp_dir, corpus, source_prefix=f"github:{repo_spec}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _parse_local(
        self, repo_dir: Path, corpus: str, source_prefix: str | None = None
    ) -> list[Document]:
        from kb_arena.ingest.parsers.html import HtmlParser
        from kb_arena.ingest.parsers.markdown import MarkdownParser
        from kb_arena.ingest.parsers.plaintext import PlaintextParser

        files = _collect_files(repo_dir)
        if not files:
            log.warning("No supported files found in %s", repo_dir)
            return []

        md_parser = MarkdownParser()
        html_parser = HtmlParser()
        txt_parser = PlaintextParser()

        docs: list[Document] = []
        for f in files:
            ext = f.suffix.lower()
            try:
                if ext in (".md", ".markdown", ".rst"):
                    parsed = md_parser.parse(f, corpus)
                elif ext in (".html", ".htm"):
                    parsed = html_parser.parse(f, corpus)
                elif ext in (".txt",):
                    parsed = txt_parser.parse(f, corpus)
                else:
                    continue

                for doc in parsed:
                    rel_path = f.relative_to(repo_dir)
                    if source_prefix:
                        doc.source = f"{source_prefix}/{rel_path}"
                    else:
                        doc.source = str(rel_path)
                    doc.id = slugify(str(rel_path))

                docs.extend(parsed)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to parse %s: %s", f, exc)

        return docs
