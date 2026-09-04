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
