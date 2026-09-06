"""Every page names its state, and no page dresses a failed read as data.

A merged slice found the same defect four times: a failure reached the reader
as an answer. Sample rows arrived under a real corpus name, a refused graph
read arrived as a database outage, and a failed run read left the previous
run's numbers under the new run's name.

These tests hold the two halves of the fix. The API reports both demo-mode
flags, because one of them alone calls a laptop a hosted demo. The pages name
the state they are in, and drop the data when a read fails.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kb_arena.chatbot.api import health
from kb_arena.settings import settings

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def _request():
    state = SimpleNamespace(
        neo4j=None, neo4j_error="", llm=None, arena=None, arena_error="", strategies={}
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.mark.parametrize(
    ("demo_mode", "demo_mode_auto"),
    [(False, False), (True, False), (True, True)],
)
async def test_health_reports_both_demo_flags(monkeypatch, demo_mode, demo_mode_auto):
    """The dashboard names one of three states, and needs both flags to do it."""
    monkeypatch.setattr(settings, "demo_mode", demo_mode)
    monkeypatch.setattr(settings, "demo_mode_auto", demo_mode_auto)

    body = await health(_request())

    assert body["demo_mode"] is demo_mode
    assert body["demo_mode_auto"] is demo_mode_auto


async def test_health_tells_a_published_demo_from_a_keyless_laptop(monkeypatch):
    """Demo mode the app turned on for itself is not a hosted deployment.

    Reading `demo_mode` alone put a laptop with no model key on screen as a
    hosted read-only demo, which is a claim about somebody else's deployment.
    """
    monkeypatch.setattr(settings, "demo_mode", True)

    monkeypatch.setattr(settings, "demo_mode_auto", False)
    published = await health(_request())
    monkeypatch.setattr(settings, "demo_mode_auto", True)
    keyless = await health(_request())

    assert published["demo_mode"] == keyless["demo_mode"] is True
    assert published["demo_mode_auto"] is False
    assert keyless["demo_mode_auto"] is True


def test_the_banner_reads_both_flags_from_the_api():
    """A hardcoded state would name a state the server is not in."""
    provider = (WEB / "components" / "ServerStateProvider.tsx").read_text()

    assert "fetchServerStatus" in provider, "the state comes from the API, not a constant"
    assert "status.demoMode && !status.demoModeAuto" in provider
    for state in ("unreachable", "hosted-read-only", "live-local"):
        assert f'"{state}"' in provider

    api = (WEB / "lib" / "api.ts").read_text()
    assert "demo_mode_auto" in api, "the client must read the flag the API reports"


@pytest.mark.parametrize(
    "route", ["page.tsx", "demo/page.tsx", "benchmark/page.tsx", "graph/page.tsx"]
)
def test_the_route_names_the_state_it_is_in(route):
    page = (WEB / "app" / route).read_text()

    assert "StateBanner" in page, f"{route} shows no state to its reader"


def test_no_sample_rows_stand_in_for_a_failed_benchmark_read():
    """Sample numbers under a real corpus name read as that corpus's results."""
    api = (WEB / "lib" / "api.ts").read_text()
    page = (WEB / "app" / "benchmark" / "page.tsx").read_text()

    assert "MOCK_BENCHMARK_DATA" not in api, "the invented rows are gone"
    assert "MOCK_BENCHMARK_DATA" not in page
    assert "BENCHMARK_UNAVAILABLE" in api, "a failed read throws instead"
    assert "FetchError" in page, "and the page says so, with a way to try again"


def test_a_failed_graph_read_shows_no_example_map():
    """A refused read is not an outage, and the example map is not this corpus."""
    api = (WEB / "lib" / "api.ts").read_text()
    page = (WEB / "app" / "graph" / "page.tsx").read_text()

    assert "connected: false }" not in api, "a failed read must throw, not answer"
    # The example map is reached from one place: a read that succeeded and
    # reported no graph database behind it.
    assert page.count("setNodes(SAMPLE_NODES)") == 1
    assert "{!readError && (" in page, "a failed read hides the map and the counts"
    assert "FetchError" in page
