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


def test_source_version_fallback_matches_project_metadata(monkeypatch) -> None:
    from importlib.metadata import PackageNotFoundError

    import kb_arena

    monkeypatch.setattr(
        kb_arena,
        "_distribution_version",
        lambda name: (_ for _ in ()).throw(PackageNotFoundError(name)),
    )

    assert kb_arena._resolve_version() == TARGET_VERSION


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
    assert "sentence-transformers>=5.0,<6" in project["optional-dependencies"]["rerank"]
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


def test_release_tools_are_pinned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    build_requires = project["build-system"]["requires"]
    assert any(
        re.fullmatch(r"hatchling==\d+(\.\d+)*", req) for req in build_requires
    ), f"build-system.requires must pin an exact hatchling version, got {build_requires}"

    publish_workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
    install_line = next(
        line for line in publish_workflow.splitlines() if "pip install" in line and "build" in line
    )
    for tool in ("build", "twine", "cyclonedx-bom"):
        assert re.search(
            rf"{tool}==\d+(\.\d+)*", install_line
        ), f"publish.yml must pin an exact {tool} version, got: {install_line.strip()}"


def test_publish_workflow_grants_attest_build_provenance_its_required_permissions() -> None:
    # actions/attest-build-provenance (a wrapper on actions/attest as of v4)
    # needs all three grants: id-token to mint the Sigstore OIDC token,
    # attestations to persist the attestation, and artifact-metadata to
    # create the artifact storage record. Miss one and the step fails.
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "publish.yml").read_text())
    permissions = workflow["jobs"]["publish"]["permissions"]

    for key in ("id-token", "attestations", "artifact-metadata"):
        assert (
            permissions.get(key) == "write"
        ), f"publish job permissions must grant '{key}: write', got {permissions}"


def test_the_publish_workflow_can_run_without_uploading() -> None:
    """The first real publish must not be the first time these steps ever ran.

    Every step before the upload, which is the build, the twine check, the
    SBOM and the attestation, is exercised by a dry run against main. Without
    a dry run the only way to learn that one of them is broken is a release.
    """
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "publish.yml").read_text())
    # PyYAML reads the bare key `on` as the boolean True.
    triggers = workflow.get("on") or workflow.get(True)
    inputs = triggers["workflow_dispatch"]["inputs"]

    assert "dry_run" in inputs, "the workflow must offer a run that uploads nothing"
    assert inputs["dry_run"]["default"] is True, "a dispatch must not publish by accident"

    steps = workflow["jobs"]["publish"]["steps"]
    upload = next(s for s in steps if s.get("name") == "Publish to PyPI")
    assert upload["if"] == "${{ !inputs.dry_run }}", "the upload must be the only guarded step"

    # The steps that prove the path must run either way.
    always = {"Build package", "Twine check", "Generate SBOM"}
    for step in steps:
        if step.get("name") in always:
            assert "if" not in step, f"{step['name']} must run in a dry run too"


def test_a_dispatch_cannot_publish_an_arbitrary_branch() -> None:
    """`workflow_dispatch` builds whatever ref the caller selected."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "publish.yml").read_text())
    steps = workflow["jobs"]["publish"]["steps"]
    guard = next(s for s in steps if "Refuse a real publish" in (s.get("name") or ""))

    assert guard["if"] == "${{ !inputs.dry_run }}"
    assert "refs/tags/v*" in guard["run"]
    assert "refs/heads/main" in guard["run"]
    # The guard must sit before the build, or it protects nothing.
    names = [s.get("name") or s.get("uses", "") for s in steps]
    assert names.index(guard["name"]) < names.index("Build package")


def test_the_workflow_never_reports_success_for_a_version_already_on_pypi() -> None:
    """`--skip-existing` made a run that published nothing look like a release."""
    text = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
    workflow = yaml.safe_load(text)
    steps = workflow["jobs"]["publish"]["steps"]
    upload = next(s for s in steps if s.get("name") == "Publish to PyPI")

    assert (
        "--skip-existing" not in upload["run"]
    ), "a version already on PyPI must fail the run, not pass it in silence"
    check = next(s for s in steps if "Refuse to republish" in (s.get("name") or ""))
    assert "pypi.org/pypi/kb-arena" in check["run"]


def test_the_dry_run_summary_fails_on_an_empty_sbom() -> None:
    """A count printed inside an echo hides every failure behind a zero exit."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "publish.yml").read_text())
    steps = workflow["jobs"]["publish"]["steps"]
    report = next(s for s in steps if s.get("name") == "Report the dry run")

    assert "set -euo pipefail" in report["run"]
    assert "sys.exit" in report["run"], "an empty or wrong SBOM must fail the step"
    # The dry run persists an attestation, and it says so rather than implying
    # that nothing outward happened.
    assert "DID persist a build attestation" in report["run"]
