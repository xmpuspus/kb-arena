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

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from kb_arena.chatbot.api import health
from kb_arena.settings import settings

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def _request(host: str = "127.0.0.1"):
    state = SimpleNamespace(
        neo4j=None, neo4j_error="", llm=None, arena=None, arena_error="", strategies={}
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=state),
        client=SimpleNamespace(host=host, port=1),
        headers={},
    )


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
    assert "status.demoMode && status.demoModeAuto === false" in provider
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
    assert "!error && !corpusSummary" in page, "no runs yet is not a failed read"
    # The Run control sat under the error and read "No runs yet", which is the
    # claim the error above it says the page does not make.
    assert '"Run list unavailable"' in page
    assert "disabled={Boolean(error) || listPending}" in page, "nothing to pick, so nothing picks"


def test_a_failed_arena_read_retries_the_read_that_failed():
    """A vote retry that starts a new match loses the vote the reader cast."""
    page = (WEB / "app" / "arena" / "page.tsx").read_text()

    assert "errorKind" in page
    assert "vote(lastWinner.current) : createMatch()" in page


def test_one_match_carries_one_vote_and_a_retry_says_so():
    """The engine refuses the second vote, so a retry cannot double count.

    A lost response leaves the vote recorded. The page then read the refusal
    as a failed vote, which is the wrong claim about a vote that landed.
    """
    from kb_arena.arena.engine import ArenaEngine

    source = inspect.getsource(ArenaEngine.vote)
    assert "if match.winner is not None:" in source
    assert '"error": "Match already voted on"' in source

    page = (WEB / "app" / "arena" / "page.tsx").read_text()
    assert 'message.toLowerCase().includes("already voted")' in page
    assert "setVoteNotice(" in page


def test_a_lost_vote_answer_never_claims_the_ratings_held_still():
    """The request can fail after the server records the vote.

    The page said the ELO ratings stayed as they were, which the client cannot
    read from a transport failure. It now says the outcome is unknown and
    re-reads the board.
    """
    page = (WEB / "app" / "arena" / "page.tsx").read_text()

    assert "The ELO ratings stay as they were" not in page
    assert "the outcome is unknown" in page
    assert "fetchLeaderboard(corpus);\n    } finally {" in page, "a failed vote re-reads"


async def test_health_reports_whether_the_caller_is_local(monkeypatch):
    """Neither demo flag says where the server runs, and the banner claimed it.

    Only the server knows which address the caller arrived on. The read gate
    already decides it the same way.
    """
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "demo_mode_auto", False)
    monkeypatch.setattr(settings, "trusted_proxy_header", "")

    local = await health(_request(host="127.0.0.1"))
    remote = await health(_request(host="203.0.113.7"))

    assert local["caller_is_local"] is True
    assert remote["caller_is_local"] is False


def test_the_banner_names_a_remote_live_server_as_one():
    """Every reachable deployment read as the reader's own machine."""
    provider = (WEB / "components" / "ServerStateProvider.tsx").read_text()
    banner = (WEB / "components" / "StateBanner.tsx").read_text()

    assert "status.callerIsLocal" in provider
    assert '"live-remote"' in provider
    assert '"live-remote": { label: "Live server"' in banner
    assert "it runs on another machine" in banner


def test_a_health_answer_without_the_flags_claims_nothing():
    """`Boolean(undefined)` turned a missing flag into a positive answer.

    An older build that reports no locality read as a server on another
    machine, and a body that is not a health answer read as a live one.
    """
    api = (WEB / "lib" / "api.ts").read_text()
    provider = (WEB / "components" / "ServerStateProvider.tsx").read_text()
    banner = (WEB / "components" / "StateBanner.tsx").read_text()

    assert 'if (typeof data?.demo_mode !== "boolean") return null;' in api
    assert "function reportedFlag(value: unknown): boolean | null" in api
    assert "callerIsLocal: boolean | null" in api

    assert "status.callerIsLocal === true" in provider
    assert "status.callerIsLocal === false" in provider
    assert '"live-unknown"' in provider
    assert "It did not say whether it runs on your machine." in banner


def test_the_arena_match_goes_with_the_board_on_a_corpus_change():
    """A vote lands on the match's own corpus, whatever the picker now says."""
    page = (WEB / "app" / "arena" / "page.tsx").read_text()
    reset = page[page.index("useScopeReset(corpus") : page.index("async function fetchLeaderboard")]

    assert "setMatch(null);" in reset, "the old corpus's answers stay votable otherwise"
    assert "setVoteResult(null);" in reset


def test_the_demo_answers_go_with_the_corpus_that_produced_them():
    """The panels held answers from the corpus the picker named a moment ago."""
    page = (WEB / "app" / "demo" / "page.tsx").read_text()

    assert "useScopeReset(corpus, () => {" in page
    # The panel key carries the trigger, so this remount aborts a running read.
    assert "setTrigger(0);" in page
    assert "key={`${s}-${trigger}`}" in page


def test_a_failed_run_read_says_which_read_failed():
    """One error state said the run list failed when one run alone had."""
    page = (WEB / "app" / "retriever-lab" / "page.tsx").read_text()

    assert "detailError" in page
    assert "The retriever-lab run list did not load" in page
    assert "did not load`}" in page, "the run error names the run"
    assert "setDetailAttempt" in page, "and retries that run, not the list"


@pytest.mark.parametrize(
    ("route", "scope"),
    [
        ("arena/page.tsx", "corpus"),
        ("graph/page.tsx", "corpus"),
        ("leaderboard/page.tsx", "filter"),
        ("demo/page.tsx", "corpus"),
    ],
)
def test_a_scope_change_drops_the_data_the_last_scope_owned(route, scope):
    """The last corpus's rows under the new corpus name read as the new one.

    Ordering the responses does not close the gap, because nothing cleared
    when the read started.
    """
    page = (WEB / "app" / route).read_text()

    assert "useScopeReset" in page
    assert f"useScopeReset({scope}, () => {{" in page


def test_the_scope_reset_lives_in_one_place():
    """Three copies of the same clear drift, and two of them already had."""
    hook = WEB / "lib" / "useScopeReset.ts"

    assert hook.is_file()
    assert "export function useScopeReset" in hook.read_text()


def test_an_empty_corpus_answer_is_not_the_built_in_list():
    """A server that holds no corpus read as one holding the built-in set."""
    api = (WEB / "lib" / "api.ts").read_text()

    assert "return { corpora: data.corpora ?? [], failed: false };" in api
    assert "corpora: [], failed: true" in api


def test_the_run_list_says_nothing_definite_while_it_is_pending():
    """The page answered "No runs yet" before the first run-list read landed."""
    page = (WEB / "app" / "retriever-lab" / "page.tsx").read_text()

    assert "const [listPending, setListPending] = useState(true);" in page
    assert "setListPending(false)" in page
    assert "!loading && !listPending && !error" in page, "the empty state waits for an answer"


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
    assert "{!readError && !loading && (" in page, "a failed read hides the map and the counts"
    assert "FetchError" in page
