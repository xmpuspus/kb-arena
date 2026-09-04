"""A route that returns corpus content is gated, and the demo still works."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from kb_arena.chatbot.auth import require_read_auth
from kb_arena.settings import settings


def _request(host: str = "127.0.0.1"):
    return SimpleNamespace(client=SimpleNamespace(host=host, port=1), headers={}, url="/x")


def test_a_local_reader_needs_no_token(monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "api_token", "")

    assert require_read_auth(_request(), None) is None


def test_a_remote_reader_without_a_token_is_refused(monkeypatch):
    """A laptop bound to every interface would otherwise serve its documents."""
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "api_token", "")

    with pytest.raises(HTTPException) as refused:
        require_read_auth(_request("203.0.113.7"), None)

    assert refused.value.status_code == 401
    assert refused.value.detail["code"] == "api_token_required_for_remote_access"


def test_a_token_is_required_once_one_is_set(monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "api_token", "s3cret")

    with pytest.raises(HTTPException) as refused:
        require_read_auth(_request("203.0.113.7"), None)
    assert refused.value.status_code == 401

    with pytest.raises(HTTPException):
        require_read_auth(_request("203.0.113.7"), "Bearer wrong")

    assert require_read_auth(_request("203.0.113.7"), "Bearer s3cret") is None


def test_the_public_demo_still_serves_its_reads(monkeypatch):
    """`require_auth` answers 503 in demo mode, and that is why this is separate.

    A hosted demo exists to serve exactly these reads. Gating them with the
    write dependency would turn the product's shop window off.
    """
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "demo_mode_auto", False)
    monkeypatch.setattr(settings, "api_token", "")

    assert require_read_auth(_request("203.0.113.7"), None) is None


def test_the_write_gate_still_refuses_a_demo(monkeypatch):
    """The two gates must stay different, or this slice removed a protection."""
    from kb_arena.chatbot.auth import require_auth

    monkeypatch.setattr(settings, "demo_mode", True)

    with pytest.raises(HTTPException) as refused:
        require_auth(_request(), None)

    assert refused.value.status_code == 503


@pytest.mark.parametrize(
    ("module", "path"),
    [
        ("kb_arena.chatbot.api", "/api/retriever-lab/{run_id}"),
        ("kb_arena.chatbot.api", "/api/benchmark/results"),
        ("kb_arena.chatbot.tools_api", "/qa-pairs"),
    ],
)
def test_every_content_route_carries_the_read_gate(module, path):
    """Named one by one, so a new content route is a deliberate decision."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))
    marker = f'"{path}", dependencies=[Depends(require_read_auth)]'
    assert marker in source, f"{path} must carry the content read gate"


def test_the_demo_command_serves_this_machine_by_default():
    """The demo bound to every interface while serve bound to loopback."""
    import inspect

    from kb_arena import cli

    source = inspect.getsource(cli.demo)
    assert '"0.0.0.0"' not in source, "the default must not serve the network"
    assert '"127.0.0.1"' in source


def test_demo_mode_never_removes_a_token_an_operator_set(monkeypatch):
    """Demo mode says who may read WITHOUT a token. It does not delete one."""
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "demo_mode_auto", False)
    monkeypatch.setattr(settings, "api_token", "s3cret")

    with pytest.raises(HTTPException) as refused:
        require_read_auth(_request("203.0.113.7"), None)
    assert refused.value.status_code == 401

    assert require_read_auth(_request("203.0.113.7"), "Bearer s3cret") is None


def test_the_graph_data_route_carries_the_read_gate():
    """It returns entities extracted from the documents, so it is content."""
    import inspect

    from kb_arena.chatbot import api

    source = inspect.getsource(api)
    assert '"/api/graph/data", dependencies=[Depends(require_read_auth)]' in source


def test_a_refused_read_is_never_reported_as_an_empty_corpus():
    """An empty list reads as a claim about the corpus, and a 401 is not one."""
    from pathlib import Path

    client = Path("web/lib/tools-api.ts").read_text()
    assert "QA_PAIRS_UNAUTHORIZED" in client
    assert "res.status === 401" in client


@pytest.mark.parametrize(
    "path",
    ["/api/graph/data", "/graph/stats", "/api/compare"],
)
def test_every_document_derived_route_carries_the_read_gate(path):
    """Entity names and per-question records are both corpus content."""
    import inspect

    from kb_arena.chatbot import api

    source = inspect.getsource(api)
    assert f'"{path}", dependencies=[Depends(require_read_auth)]' in source


def test_every_gated_route_has_a_client_that_carries_the_token():
    """Gating a route without updating its client breaks the page."""
    from pathlib import Path

    api_ts = Path("web/lib/api.ts").read_text()
    for call in (
        "apiFetch(`${API_URL}/api/graph/data",
        "apiFetch(`${API_URL}/api/benchmark/results",
    ):
        assert call in api_ts, f"{call} must carry the token"
    assert "fetch(`${API_URL}/api/graph/data" not in api_ts.replace("apiFetch(", "")


def test_a_refused_benchmark_read_is_never_shown_as_sample_numbers():
    """Sample rows under a real corpus name read as that corpus's results."""
    from pathlib import Path

    assert "BENCHMARK_UNAUTHORIZED" in Path("web/lib/api.ts").read_text()
    page = Path("web/app/benchmark/page.tsx").read_text()
    assert '"refused"' in page
    assert "setRows([])" in page


def test_the_generate_tab_handles_a_refused_read():
    """The client now rejects, and an unhandled rejection breaks the page."""
    from pathlib import Path

    tab = Path("web/components/tools/GenerateTab.tsx").read_text()
    assert ".catch(" in tab
    assert 'setState("error")' in tab


def test_demo_mode_the_app_turned_on_itself_never_widens_reads(monkeypatch):
    """A laptop with no API key auto-enables demo mode.

    Without this distinction the read gate would have allowed every remote
    reader on exactly the setup a first-time user has: no key, no token.
    """
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "demo_mode_auto", True)
    monkeypatch.setattr(settings, "api_token", "")

    with pytest.raises(HTTPException) as refused:
        require_read_auth(_request("203.0.113.7"), None)
    assert refused.value.status_code == 401

    # An operator who asked for a demo still gets one.
    monkeypatch.setattr(settings, "demo_mode_auto", False)
    assert require_read_auth(_request("203.0.113.7"), None) is None


def test_the_app_marks_demo_mode_it_enabled_for_itself():
    """The flag has to be set where demo mode turns itself on, or it is a lie."""
    import inspect

    from kb_arena.chatbot import api

    source = inspect.getsource(api)
    assert "settings.demo_mode_auto = True" in source


def test_a_refused_graph_read_is_never_shown_as_a_database_outage():
    """`connected: false` is a claim about the deployment, and a 401 is not one."""
    from pathlib import Path

    assert "GRAPH_UNAUTHORIZED" in Path("web/lib/api.ts").read_text()
    page = Path("web/app/graph/page.tsx").read_text()
    assert "readError" in page
    # The retry now lives in one shared hook, checked by its own test below.
    assert "useTokenEpoch()" in page, "a saved token must retry the refused read"


def test_saving_a_token_announces_itself():
    """A page that already got 401 shows the refusal until something reads again."""
    from pathlib import Path

    nav = Path("web/components/Nav.tsx").read_text()
    assert (
        nav.count('new Event("kb-arena-token-changed")') == 2
    ), "both saving and removing a token change what a page may read"


def test_a_failed_run_read_never_leaves_the_previous_run_on_screen():
    """One run's numbers under another run's name is the worst kind of wrong."""
    from pathlib import Path

    page = Path("web/app/retriever-lab/page.tsx").read_text()
    assert "setData(null)" in page


def test_the_leaderboard_and_the_variance_command_agree_on_a_build():
    """Two copies of this rule would drift, and the two surfaces would disagree."""
    import inspect

    from kb_arena.benchmark import variance
    from kb_arena.benchmark.manifest import build_identity

    assert "build_identity(run)" in inspect.getsource(variance._code_version)
    assert build_identity({"manifest": {"code_version": "1.0", "git_sha": "abc"}}) == "1.0@abc"
    assert build_identity({"manifest": {}}) == "unrecorded"


def test_the_leaderboard_never_averages_two_builds_into_one_row(tmp_path, monkeypatch):
    """The leaderboard is the surface a reader cites, and it blended commits."""
    import asyncio
    import json

    from kb_arena.chatbot import api
    from kb_arena.settings import settings as live

    monkeypatch.setattr(live, "results_path", str(tmp_path))
    manifest = {"question_set_fingerprint": "f1", "judge": {"model": "j"}, "top_k": 5}
    for run_id, accuracy, sha in (("r1", 0.2, "aaa"), ("r2", 0.8, "bbb")):
        run_dir = tmp_path / f"run_{run_id}"
        run_dir.mkdir()
        (run_dir / "c_bm25.json").write_text(
            json.dumps(
                {
                    "corpus": "c",
                    "strategy": "bm25",
                    "run_id": run_id,
                    "accuracy_by_tier": {"1": accuracy},
                    "records": [{}],
                    "manifest": {**manifest, "code_version": "0.11.0", "git_sha": sha},
                }
            )
        )

    board = asyncio.run(api.leaderboard(None, corpus="c"))["leaderboard"]

    assert len(board) == 2, "two commits are two builds, not one measurement repeated"
    assert {row["build"] for row in board} == {"0.11.0@aaa", "0.11.0@bbb"}
    assert all(row["runs"] == 1 for row in board)


def test_every_protected_page_retries_when_a_token_is_saved():
    """Retrying one page and not the others reads as the token half working."""
    from pathlib import Path

    hook = Path("web/lib/useTokenEpoch.ts")
    assert hook.is_file(), "one hook, so the pages cannot drift apart"
    assert "kb-arena-token-changed" in hook.read_text()

    for name in (
        "web/app/graph/page.tsx",
        "web/app/benchmark/page.tsx",
        "web/app/retriever-lab/page.tsx",
    ):
        page = Path(name).read_text()
        assert "useTokenEpoch()" in page, f"{name} must retry on a token change"
        assert "tokenEpoch]" in page, f"{name} must read tokenEpoch in its effect"


@pytest.mark.parametrize(
    "name",
    ["GRAPH_UNAVAILABLE", "BENCHMARK_UNAVAILABLE"],
)
def test_a_rate_limit_is_never_a_domain_answer(name):
    """A 429 or a 500 is a failure to read, not a result and not an outage."""
    from pathlib import Path

    client = Path("web/lib/api.ts").read_text()
    assert name in client
    assert "res.status === 429" in client
    assert "res.status >= 500" in client
