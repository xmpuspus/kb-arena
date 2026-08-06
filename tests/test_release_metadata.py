"""Release metadata must identify the same published version."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGET_VERSION = "0.10.0"
TARGET_DATE = "2026-08-05"


def test_release_metadata_is_aligned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    changelog = (ROOT / "CHANGELOG.md").read_text()

    assert project["project"]["version"] == TARGET_VERSION
    assert citation["version"] == TARGET_VERSION
    assert str(citation["date-released"]) == TARGET_DATE
    assert f"## [{TARGET_VERSION}] - {TARGET_DATE}" in changelog
    security = (ROOT / "SECURITY.md").read_text()
    assert "| 0.10.x | Active fixes |" in security


def test_readme_links_resolve_from_package_indexes() -> None:
    readme = (ROOT / "README.md").read_text()
    relative_targets = []

    for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", readme):
        target = match.group(1).strip()
        if not target.startswith(("https://", "http://", "mailto:", "#")):
            relative_targets.append(target)

    assert relative_targets == []


def test_sdist_excludes_local_and_generated_work() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    excluded = set(project["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert {"tmp/", ".venv/", "results/run_*/", "uv.lock"} <= excluded


def test_generated_run_directories_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text()

    assert "results/run_*/" in gitignore


def test_frontend_ci_runs_lint_before_build() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    lint = workflow.index("- run: npm run lint")
    build = workflow.index("- run: npx next build")
    assert lint < build


def test_supported_python_versions_match_dependencies_and_ci() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert project["requires-python"] == ">=3.11,<3.14"
    assert "Programming Language :: Python :: 3.13" in project["classifiers"]
    assert "tiktoken==0.13.0" in project["dependencies"]
    assert "click==8.4.2" in project["dependencies"]
    assert "typer==0.27.1" in project["dependencies"]
    assert "fastapi==0.141.1" in project["dependencies"]
    assert "starlette==1.4.1" in project["dependencies"]
    assert 'python-version: ["3.11", "3.12", "3.13"]' in workflow


def test_browser_auth_covers_protected_requests() -> None:
    auth = (ROOT / "web" / "lib" / "auth.ts").read_text()
    api = (ROOT / "web" / "lib" / "api.ts").read_text()
    tools = (ROOT / "web" / "lib" / "tools-api.ts").read_text()
    arena = (ROOT / "web" / "app" / "arena" / "page.tsx").read_text()
    nav = (ROOT / "web" / "components" / "Nav.tsx").read_text()

    assert "sessionStorage" in auth
    assert 'headers.set("Authorization", `Bearer ${token}`)' in auth
    assert "apiFetch(`${API_URL}/api/graph/build`" in api
    assert "apiFetch(`${API_URL}/chat/stream`" in api
    assert tools.count("apiFetch(") == 3
    assert "apiFetch(`${API}/api/arena/match`" in arena
    assert "apiFetch(`${API}/api/arena/vote`" in arena
    assert 'event.key !== "Tab"' in nav
    assert "tokenTrigger?.focus()" in nav


def test_graph_build_ui_isolated_from_corpus_changes() -> None:
    page = (ROOT / "web" / "app" / "graph" / "page.tsx").read_text()

    assert 'disabled={buildStatus === "building"}' in page
    assert "abortRef.current?.abort()" in page
    assert "buildEpochRef.current" in page
    assert "await fetchGraphData(buildCorpus)" in page
    assert "setConnected(data.connected)" in page
    assert "setNodes(apiToGraphNodes(data.nodes))" in page


def test_graph_build_client_streams_with_server_build_id() -> None:
    api = (ROOT / "web" / "lib" / "api.ts").read_text()
    page = (ROOT / "web" / "app" / "graph" / "page.tsx").read_text()

    assert "build_id: string" in api
    assert "/api/graph/build/stream/${buildId}" in api
    assert "streamGraphBuild(build.build_id" in page
