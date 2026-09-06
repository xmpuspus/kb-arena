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


ROUTES = [
    "page.tsx",
    "demo/page.tsx",
    "benchmark/page.tsx",
    "graph/page.tsx",
    "arena/page.tsx",
    "leaderboard/page.tsx",
    "retriever-lab/page.tsx",
    "tools/page.tsx",
]


@pytest.mark.parametrize("route", ROUTES)
def test_the_route_names_the_state_it_is_in(route):
    page = (WEB / "app" / route).read_text()

    assert "StateBanner" in page, f"{route} shows no state to its reader"


def test_every_route_sits_inside_the_one_state_provider():
    """Two providers would fetch twice and could name two different states."""
    layout = (WEB / "app" / "layout.tsx").read_text()

    assert "<ServerStateProvider>" in layout


@pytest.mark.parametrize("route", ROUTES[1:])
def test_a_failed_read_offers_a_way_back(route):
    """A read that fails says what failed and what the reader can do next.

    The home route reads only the two catalogs it can name from the built-in
    defaults, and its banner says so, so it carries no error block.
    """
    page = (WEB / "app" / route).read_text()

    assert "FetchError" in page or "Try again" in page, f"{route} offers no retry"


def test_no_sample_rows_stand_in_for_a_failed_benchmark_read():
    """Sample numbers under a real corpus name read as that corpus's results."""
    api = (WEB / "lib" / "api.ts").read_text()
    page = (WEB / "app" / "benchmark" / "page.tsx").read_text()

    assert "MOCK_BENCHMARK_DATA" not in api, "the invented rows are gone"
    assert "MOCK_BENCHMARK_DATA" not in page
    assert "BENCHMARK_UNAVAILABLE" in api, "a failed read throws instead"
    assert "FetchError" in page, "and the page says so, with a way to try again"


def test_a_failed_leaderboard_read_drops_the_rows_it_had():
    """Rows from the last filter under a new corpus name read as that corpus."""
    page = (WEB / "app" / "leaderboard" / "page.tsx").read_text()

    assert "setData(null);" in page
    assert "FetchError" in page


def test_a_failed_run_list_never_reads_as_a_lab_nobody_ran():
    """An empty list in place of a failed read is a claim about the deployment."""
    page = (WEB / "app" / "retriever-lab" / "page.tsx").read_text()

    assert "if (!r.ok) throw new Error" in page, "the run list must check the status"
    assert "setRuns([]);" in page, "and drop the runs it cannot vouch for"
    assert "!loading && !error && !corpusSummary" in page, "no runs yet is not a failed read"
    # The Run control sat under the error and read "No runs yet", which is the
    # claim the error above it says the page does not make.
    assert 'error ? "Run list unavailable" : "No runs yet"' in page
    assert "disabled={Boolean(error)}" in page, "a failed read leaves nothing to pick"


def test_a_failed_arena_read_retries_the_read_that_failed():
    """A vote retry that starts a new match loses the vote the reader cast."""
    page = (WEB / "app" / "arena" / "page.tsx").read_text()

    assert "errorKind" in page
    assert "vote(lastWinner.current) : createMatch()" in page


def test_the_tools_never_act_on_a_corpus_from_a_failed_read():
    """The built-in list would send an audit at a corpus nobody holds."""
    api = (WEB / "lib" / "api.ts").read_text()
    page = (WEB / "app" / "tools" / "page.tsx").read_text()

    assert "fetchCorporaResult" in api, "the client must report the failed read"
    assert "fetchCorporaResult()" in page
    assert "{corpus && !failed && (" in page, "the tabs stay off after a failed read"
    # The three tab buttons and the corpus select stayed live beside an empty
    # list, so a reader could start an audit with no corpus.
    assert page.count("disabled={failed}") == 2
    assert "Corpus list unavailable" in page


def test_no_route_prints_the_browser_string_for_a_dead_api():
    """A dead API reached the reader as the TypeError message from `fetch`.

    The browser writes that message, and it names no action. Four surfaces
    printed it: the arena, the leaderboard, the retriever lab and the demo
    panels.
    """
    api = (WEB / "lib" / "api.ts").read_text()

    assert "The browser could not reach the API." in api
    assert "err instanceof TypeError" in api, "a network failure is a TypeError"

    for name in (
        "app/arena/page.tsx",
        "app/leaderboard/page.tsx",
        "app/retriever-lab/page.tsx",
        "components/ChatPanel.tsx",
    ):
        assert "readFailureMessage" in (WEB / name).read_text(), f"{name} shows the raw string"


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
