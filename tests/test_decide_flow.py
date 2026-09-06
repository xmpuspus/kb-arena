"""The decision flow reads backend facts, and it never claims more than a run recorded.

The flow lives in a static export, so it cannot read Python. It copies the
profile weights and the two ceilings the reporter ranks with. A copy drifts
silently, which is what these tests catch: they read the TypeScript source the
way `test_catalog_parity.py` reads `web/lib/api.ts`.

The evidence route is the other half. It serves the bundle as it was written,
because a page that recomputes `citable` can call a run evidence after the run
refused to.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest
from fastapi import HTTPException

from kb_arena.benchmark import reporter
from kb_arena.benchmark.compare import SAFE_ID
from kb_arena.chatbot import api
from kb_arena.settings import settings
from kb_arena.strategies.catalog import STRATEGY_CATALOG, default_strategy_names

ROOT = Path(__file__).resolve().parents[1]
DECIDE_TS = ROOT / "web" / "lib" / "decide.ts"
DECIDE_PAGE = ROOT / "web" / "app" / "decide" / "page.tsx"
NAV = ROOT / "web" / "components" / "Nav.tsx"


def _profile_weights() -> dict[str, dict[str, float]]:
    block = re.search(
        r"PROFILE_WEIGHTS: Record<ProfileName, ProfileWeights> = \{(.*?)\n\};",
        DECIDE_TS.read_text(),
        re.S,
    )
    assert block, "web/lib/decide.ts must export PROFILE_WEIGHTS"
    weights: dict[str, dict[str, float]] = {}
    for line in block.group(1).splitlines():
        row = re.match(
            r'\s*"?([a-z-]+)"?: \{ accuracy: ([\d.]+), reliability: ([\d.]+), '
            r"latency: ([\d.]+), cost: ([\d.]+) \},",
            line,
        )
        if row:
            weights[row.group(1)] = {
                "accuracy": float(row.group(2)),
                "reliability": float(row.group(3)),
                "latency": float(row.group(4)),
                "cost": float(row.group(5)),
            }
    return weights


def test_the_flow_carries_the_profiles_the_reporter_ranks_with():
    """A profile the page names and the reporter does not is a ranking nobody can repeat."""
    assert _profile_weights() == reporter.PROFILES, (
        "web/lib/decide.ts and kb_arena/benchmark/reporter.py disagree on the profiles. "
        "The page shows weights a reader compares against a report."
    )


def test_the_flow_prints_the_two_ceilings_the_reporter_scores_against():
    source = DECIDE_TS.read_text()
    latency = re.search(r"LATENCY_CEILING_MS = (\d+)", source)
    cost = re.search(r"COST_CEILING_USD = ([\d.]+)", source)

    assert latency and float(latency.group(1)) == reporter.LATENCY_CEILING_MS
    assert cost and float(cost.group(1)) == reporter.COST_CEILING_USD


def test_every_profile_names_the_trade_off_it_makes():
    source = DECIDE_TS.read_text()
    block = re.search(
        r"PROFILE_TRADEOFFS: Record<ProfileName, string> = \{(.*?)\n\};", source, re.S
    )
    assert block, "web/lib/decide.ts must export PROFILE_TRADEOFFS"

    for name in reporter.PROFILES:
        assert f'"{name}"' in block.group(1) or f"\n  {name}:" in block.group(1), (
            f"{name} states no trade-off, so the step shows a weight set with no reason "
            "for a reader to pick it"
        )


def test_the_flow_never_recomputes_the_citable_verdict():
    """`build_bundle` also refuses a bundle that records no command, and a share cannot see that."""
    source = DECIDE_TS.read_text() + DECIDE_PAGE.read_text()

    assert "reviewed_share ===" not in source, (
        "the flow must print bundle.citable, not derive it. A derived verdict calls a "
        "run with no command citable."
    )
    assert "bundle.citable" in source or "b.citable" in source
    assert "why_not_citable" in source, "a bundle that refuses must say why on the page"


def test_a_missing_bundle_reads_as_unknown_review_status():
    source = DECIDE_TS.read_text()
    caveats = re.search(r"export function bundleCaveats\(.*?\n\}", source, re.S)
    assert caveats, "web/lib/decide.ts must export bundleCaveats"
    no_bundle = caveats.group(0).split("const lines")[0]

    assert (
        "unknown" in no_bundle
    ), "with no bundle the record must call the review status unknown, never reviewed"
    assert "human-reviewed" not in no_bundle


def test_the_record_reports_a_machine_drafted_question_set():
    """The NIST set is a machine-generated draft. A record hiding that invites a citation."""
    source = DECIDE_TS.read_text()

    assert (
        "machine-assisted-draft" in source
    ), "the record must read the draft count the review verdict wrote"
    assert (
        "enough_pairs_for_inference" in source
    ), "below the pair floor no flag fired, so the record must not print one"


def test_the_candidate_list_reads_catalog_fields_the_api_serves():
    """`architecture` in the bundled fallback disagrees with the Python catalog."""
    source = DECIDE_TS.read_text()

    for field in ("default_benchmark", "needs_embeddings", "optional_extra", "experimental"):
        assert field in source, f"the candidate rule ignores {field}, which the catalog records"
    assert (
        "catalogIsLive" in source
    ), "the page must tell a served catalog from the bundled copy before it prints architecture"


def test_the_page_renders_every_step_it_advertises():
    """A tab with no screen behind it is a step a reader clicks into nothing."""
    source = DECIDE_PAGE.read_text()
    declared = re.findall(r"\{ n: (\d+), tab: \"([^\"]+)\", heading: \"([^\"]+)\"", source)
    rendered = {int(n) for n in re.findall(r"\{step === (\d+) && \(", source)}

    assert len(declared) == 6, "the flow claims six steps"
    assert {int(n) for n, _, _ in declared} == rendered, (
        f"steps declared {[n for n, _, _ in declared]} and steps rendered {sorted(rendered)} "
        "disagree, so a tab opens an empty screen"
    )
    for _, tab, heading in declared:
        assert heading and heading != tab, f"the {tab} step heading names a topic, not a finding"


def test_the_record_attaches_only_a_bundle_that_produced_the_compared_numbers():
    """The committed bundle covers a lab run of bm25, and the comparison reads two result files.

    Taking the newest bundle on the corpus made the record quote a review
    verdict over numbers that verdict never covered.
    """
    source = DECIDE_TS.read_text()
    match = re.search(r"export function bundleForComparison\(.*?\n\}", source, re.S)
    assert match, "web/lib/decide.ts must export bundleForComparison"
    body = match.group(0)

    assert "meta?.a?.run_id" in body and "meta?.b?.run_id" in body, (
        "the match must read the run id of both compared sides. One side alone "
        "attaches a bundle to a pairing it only half describes."
    )
    assert "describesComparison: true" in body and "describesComparison: false" in body

    page = DECIDE_PAGE.read_text()
    assert (
        "bundles[0] ?? null" not in page
    ), "the newest bundle on the corpus is not the run behind the comparison"
    assert "bundleDescribesComparison" in page and "bundleDescribesComparison" in source
    assert (
        "listed for context only" in source
    ), "an unmatched bundle must be labelled as another run, never as the run behind the numbers"


def test_changing_the_scope_drops_the_earlier_comparison():
    """The header would name one corpus and pair while the numbers described another."""
    page = DECIDE_PAGE.read_text()

    assert "useScopeReset" in page, "the merged hook already clears scoped data"
    scope = re.search(r"useScopeReset\(\s*`([^`]+)`", page)
    assert scope, "the reset must key on a scope string"
    # `activeCorpus` is the corpus name after step 1, which is where a reader
    # can type their own instead of picking a built-in one.
    for field in ("activeCorpus", "stratA", "stratB", "metric"):
        assert f"${{{field}}}" in scope.group(
            1
        ), f"{field} changes the meaning of the numbers, so it belongs in the reset scope"
    assert (
        "autoRead" in page
    ), "the URL-linked read must fire once, or it refills the record the reset just cleared"


def test_a_read_in_flight_cannot_write_back_after_the_scope_changes():
    """Clearing state does not cancel a request. The reply lands and refills it."""
    page = DECIDE_PAGE.read_text()
    read = re.search(r"const readComparison = useCallback\(.*?\n  \}, \[", page, re.S)
    assert read, "web/app/decide/page.tsx must define readComparison"
    body = read.group(0)

    assert "++readTicket.current" in body, "the read must take a ticket before it starts"
    assert body.count("isCurrentRead()") >= 3, (
        "the success path, the failure path and the pending flag each need the "
        "check. A retired reply that writes any of the three is the same bug."
    )

    reset = re.search(r"useScopeReset\(.*?\n  \}\);", page, re.S)
    assert reset, "the scope reset must exist"
    assert "readTicket.current += 1" in reset.group(
        0
    ), "the reset must retire the read in flight, not only clear what landed"
    assert "setComparing(false)" in reset.group(
        0
    ), "a retired read never reaches its finally, so the pending flag would never clear"


def test_a_capped_or_broken_evidence_read_never_reads_as_no_run_exists():
    source = DECIDE_TS.read_text()
    reason = re.search(r"export function noBundleReason\(.*?\n\}", source, re.S)
    assert reason, "web/lib/decide.ts must export noBundleReason"
    body = reason.group(0)

    assert "not proof that no run exists" in body, (
        "a capped list and a parse failure both leave the list empty, and neither "
        "proves the deployment holds no run"
    )
    assert body.index("unreadable.length > 0") < body.index(
        "truncatedLimit > 0"
    ), "a bundle that failed to parse is the more definite fact, so it reports first"
    page = DECIDE_PAGE.read_text()
    assert (
        "noBundleReason(bundlesTruncated, bundlesUnreadable, corpus)" in page
    ), "the empty state must read the truncation and the parse failures the route reported"


def test_the_evidence_read_is_ordered_the_way_the_comparison_read_is():
    """A reply for the corpus the reader left would list its runs under this one."""
    page = DECIDE_PAGE.read_text()
    pattern = r"const evidenceTicket = useRef\(0\);.*?\n  \}, \[activeCorpus\]\);"
    read = re.search(pattern, page, re.S)
    assert read, "the evidence read must take a ticket, the way readComparison does"
    body = read.group(0)

    assert "++evidenceTicket.current" in body, "the ticket must be taken before the read starts"
    assert (
        body.count("isCurrentRead()") >= 2
    ), "the success path and the failure path both write state, so both need the check"


def test_the_own_documents_path_names_its_own_corpus():
    """Reusing the built-in name told a reader to ingest their files into the example."""
    page = DECIDE_PAGE.read_text()

    assert "ownCorpus" in page, "the own-documents path must hold its own corpus name"
    assert (
        'ingestCommand(corpus || "my-docs")' not in page
    ), "the ingest command must not fall back to the selected built-in corpus"
    assert "ingestCommand(ownCorpus)" in page and "initCommand(ownCorpus)" in page
    assert "isSafeId(ownCorpus)" in page, "a name the tool refuses must not reach a command"


def test_a_malformed_success_never_becomes_an_empty_domain_answer():
    """A 200 this page cannot read is a failed read, not "the deployment holds nothing"."""
    source = DECIDE_TS.read_text()

    assert "async function readJson(" in source, "every reader needs the same parse guard"
    corpora = re.search(r"export async function fetchCorporaOrFail\(.*?\n\}", source, re.S)
    assert corpora and "Array.isArray(data.corpora) ? data.corpora : []" not in corpora.group(
        0
    ), "an unreadable body became an empty corpus list, which claims the deployment is empty"
    assert "if (!Array.isArray(data.corpora)) throw new Error(CORPORA_UNREADABLE);" in source

    compare = re.search(r"export async function fetchCompare\(.*?\n\}", source, re.S)
    assert compare, "web/lib/decide.ts must export fetchCompare"
    body = compare.group(0)
    assert "return res.json();" not in body, "the comparison reply reached the record unchecked"
    for field in ("n_paired", "mean_delta", "per_question", "delta_ci_95", "reasons"):
        assert field in body, f"the shape check ignores {field}, which the record prints"
    assert "if (!shaped) throw new Error(COMPARE_UNREADABLE);" in body


def test_a_command_a_reader_pastes_is_refused_before_it_carries_a_shell_character():
    """`corpus`, `a` and `b` arrive from the URL and reach a copy button."""
    source = DECIDE_TS.read_text()

    pattern = re.search(r"export const SAFE_ID = /(\^.*\$)/;", source)
    assert pattern, "web/lib/decide.ts must hold the id pattern"
    # The same pattern the API applies. A page that builds a command line must
    # refuse what the route would refuse.
    # The page copy must track the backend pattern, not a hand-written guess.
    assert pattern.group(1) == f"^{SAFE_ID.pattern}$"

    for name in ("benchmarkCommand", "labCommand", "compareCommand", "ingestCommand"):
        body = re.search(rf"export function {name}\(.*?\n\}}", source, re.S)
        assert body, f"web/lib/decide.ts must export {name}"
        assert "command(" in body.group(0), f"{name} must build through the validating helper"
    builder = re.search(r"function command\(.*?\n\}", source, re.S)
    assert builder and "values.every(isSafeId)" in builder.group(0)
    assert "UNSAFE_COMMAND" in builder.group(
        0
    ), "a refused value must produce the refusal, never a command line"


def test_an_unreviewed_question_warns_even_when_no_draft_is_counted():
    """`review_summary` refuses to publish on unspecified exactly as it does on a draft."""
    source = DECIDE_TS.read_text()
    body = re.search(r"export function reviewWarning\(.*?\n\}", source, re.S)
    assert body, "web/lib/decide.ts must export reviewWarning"
    warning = body.group(0)

    assert (
        "reviewed >= total" in warning
    ), "the warning must fire whenever a question is not human-reviewed, not only on a draft"
    assert "(corpus.draftQuestionCount ?? 0) > 0" not in warning
    # The two cases read differently, because a draft has a wrong answer key
    # nobody checked and an unspecified question has no status at all.
    assert "no review status" in warning and "Machine-drafted" in warning
    assert "unspecified" in warning


def test_step_four_never_advertises_a_path_step_five_cannot_read():
    """`/api/compare` reads result files. A retriever-lab run writes one lab file instead."""
    page = DECIDE_PAGE.read_text()

    assert (
        "Both paths end at the same comparison" not in page
    ), "the lab command does not feed step 5, so this sentence was untrue"
    assert "Feeds step 5" in page, "the page must name the path that does feed the comparison"
    assert "which step 5 cannot read" in page
    assert (
        "labCompareCommand(" in page
    ), "the lab path needs the command that does pair two strategies inside a lab file"
    assert '"--lab"' in DECIDE_TS.read_text(), "and that command must pass --lab"


def test_the_navigation_points_at_the_leaderboard_and_the_flow():
    """M-22 and UX-07: a route with no navigation entry is a route nobody finds."""
    links = re.findall(r'\{ href: "([^"]+)", label: "([^"]+)" \}', NAV.read_text())
    hrefs = [href for href, _ in links]

    assert "/leaderboard" in hrefs, "the leaderboard exists and nothing links to it"
    assert "/decide" in hrefs, "the decision flow needs an entry point too"


def test_the_flow_offers_the_commands_the_cli_accepts():
    source = DECIDE_TS.read_text()

    def _parts(fn: str) -> str:
        body = re.search(rf"export function {fn}\(.*?\n\}}", source, re.S)
        assert body, f"web/lib/decide.ts must export {fn}"
        return body.group(0)

    # benchmark takes --strategy and retriever-lab takes --strategies. One name
    # for both produces a command that fails the moment a reader runs it.
    assert '"--strategy"' in _parts("benchmarkCommand")
    assert '"--strategies"' in _parts("labCommand")
    assert '"--strategies"' not in _parts("benchmarkCommand")
    for flag in ('"kb-arena"', '"compare"', '"--corpus"', '"--a"', '"--b"'):
        assert flag in _parts("compareCommand"), f"compareCommand drops {flag}"
    assert "./datasets/${corpus}/raw/" in _parts("ingestCommand")
    assert '"init-corpus"' in _parts("initCommand")


def test_the_candidates_are_the_default_benchmark_set():
    """The flow offers what `--strategies all` covers, so a baseline run already exists."""
    source = DECIDE_TS.read_text()
    assert "record.default_benchmark" in source
    # The default set is smaller than the catalog, and the page states the difference.
    assert len(default_strategy_names()) < len(STRATEGY_CATALOG)


def _bundle(**overrides) -> dict:
    base = {
        "bundle_version": 1,
        "corpus": "c",
        "command": ["kb-arena", "retriever-lab", "--corpus", "c"],
        "citable": False,
        "why_not_citable": "nobody checked these",
        "seed": 0,
        "review": {"publishable": False, "counts": {"machine-assisted-draft": 3}},
    }
    return {**base, **overrides}


def _write_bundle(root: Path, run_id: str, bundle: dict) -> None:
    run_dir = root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence.json").write_text(json.dumps(bundle))


def test_the_evidence_route_serves_the_bundle_as_it_was_written(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write_bundle(tmp_path, "aaa", _bundle())

    answer = asyncio.run(api.evidence_bundles())

    assert len(answer["bundles"]) == 1
    served = answer["bundles"][0]
    assert served["citable"] is False
    assert served["why_not_citable"] == "nobody checked these"
    assert served["run_id"] == "aaa", "the run id names the directory, and a page links by it"
    assert served["command"] == ["kb-arena", "retriever-lab", "--corpus", "c"]


def test_the_evidence_route_filters_by_corpus_and_refuses_a_bad_name(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write_bundle(tmp_path, "aaa", _bundle(corpus="c"))
    _write_bundle(tmp_path, "bbb", _bundle(corpus="other"))

    assert [b["corpus"] for b in asyncio.run(api.evidence_bundles(corpus="c"))["bundles"]] == ["c"]
    assert len(asyncio.run(api.evidence_bundles())["bundles"]) == 2

    for bad in ("../etc", "c\n", "a/b"):
        with pytest.raises(HTTPException) as refused:
            asyncio.run(api.evidence_bundles(corpus=bad))
        assert refused.value.status_code == 400


def test_the_evidence_route_caps_the_walk_and_says_when_it_truncated(tmp_path, monkeypatch):
    """A results directory grows one run directory forever, and this route reads files."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    for index in range(api.EVIDENCE_SCAN_LIMIT + 5):
        _write_bundle(tmp_path, f"r{index:04d}", _bundle())

    answer = asyncio.run(api.evidence_bundles())

    assert len(answer["bundles"]) == api.EVIDENCE_SCAN_LIMIT
    assert answer["truncated"] is True
    assert answer["scan_limit"] == api.EVIDENCE_SCAN_LIMIT
    # Sorted newest first before the slice, so the cap keeps the newest runs.
    assert answer["bundles"][0]["run_id"] == f"r{api.EVIDENCE_SCAN_LIMIT + 4:04d}"


def test_a_full_read_never_reports_a_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write_bundle(tmp_path, "aaa", _bundle())

    assert asyncio.run(api.evidence_bundles())["truncated"] is False
    monkeypatch.setattr(settings, "results_path", str(tmp_path / "missing"))
    assert asyncio.run(api.evidence_bundles())["truncated"] is False


def test_the_scan_drops_the_oldest_runs_not_an_arbitrary_set(tmp_path, monkeypatch):
    """A run id is random hex, so a name sort ordered the runs by nothing."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    # Name order and time order disagree: the alphabetically first run is the
    # newest. A name sort would drop it, and it is the one a reader wants.
    for name, when in (("aaa", 3000.0), ("mmm", 2000.0), ("zzz", 1000.0)):
        _write_bundle(tmp_path, name, _bundle())
        os.utime(tmp_path / f"run_{name}", (when, when))
    monkeypatch.setattr(api, "EVIDENCE_SCAN_LIMIT", 2)

    answer = asyncio.run(api.evidence_bundles())

    assert [b["run_id"] for b in answer["bundles"]] == ["aaa", "mmm"]
    assert answer["truncated"] is True


def test_the_listing_itself_stops_rather_than_walking_every_run(tmp_path, monkeypatch):
    """The scan limit bounded the answer while the glob still listed and stat-ed everything."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    monkeypatch.setattr(api, "EVIDENCE_LIST_LIMIT", 4)
    monkeypatch.setattr(api, "EVIDENCE_SCAN_LIMIT", 2)
    for index in range(20):
        _write_bundle(tmp_path, f"r{index:04d}", _bundle())

    listed: list[str] = []
    real_stat = api._entry_recency
    monkeypatch.setattr(
        api, "_entry_recency", lambda entry: (listed.append(entry.name), real_stat(entry))[1]
    )
    answer = asyncio.run(api.evidence_bundles())

    assert len(listed) == 4, (
        f"the listing examined {len(listed)} of 20 run directories. The bound is on the "
        "walk, so it must stop rather than rank everything and slice."
    )
    assert len(answer["bundles"]) == 2
    assert answer["truncated"] is True


def test_a_listing_that_fails_never_answers_as_an_empty_deployment(tmp_path, monkeypatch):
    """The bounded listing swallowed OSError and returned no runs, which the page read as none.

    That is the same shape as the dropped corrupt bundle: a failed read turning
    into a claim the deployment holds nothing.
    """
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write_bundle(tmp_path, "aaa", _bundle())

    def _refuse(_path):
        raise PermissionError("no")

    monkeypatch.setattr(api._os, "scandir", _refuse)

    with pytest.raises(HTTPException) as refused:
        asyncio.run(api.evidence_bundles())

    assert refused.value.status_code == 503
    assert "could not be listed" in refused.value.detail


def test_a_corrupt_bundle_is_reported_rather_than_dropped(tmp_path, monkeypatch):
    """Dropping it answered 200 with an empty list, and step 4 then claimed no run exists."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write_bundle(tmp_path, "aaa", _bundle())
    (tmp_path / "run_bad").mkdir()
    (tmp_path / "run_bad" / "evidence.json").write_text("{ not json")
    (tmp_path / "run_list").mkdir()
    (tmp_path / "run_list" / "evidence.json").write_text("[1, 2]")
    (tmp_path / "run_empty").mkdir()

    answer = asyncio.run(api.evidence_bundles())

    assert [b["run_id"] for b in answer["bundles"]] == ["aaa"]
    assert sorted(answer["unreadable"]) == ["bad", "list"]
    # A directory with no bundle at all is not unreadable. It holds no evidence.
    assert "empty" not in answer["unreadable"]


def test_a_broken_bundle_is_reported_whichever_corpus_was_asked_for(tmp_path, monkeypatch):
    """It names no corpus, so filtering it out would hide it from every reader."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    (tmp_path / "run_bad").mkdir()
    (tmp_path / "run_bad" / "evidence.json").write_text("{ not json")

    answer = asyncio.run(api.evidence_bundles(corpus="aws-compute"))

    assert answer["bundles"] == []
    assert answer["unreadable"] == ["bad"]


def test_no_results_directory_reads_as_no_bundles(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path / "missing"))

    assert asyncio.run(api.evidence_bundles()) == {
        "bundles": [],
        "unreadable": [],
        "truncated": False,
        "scan_limit": api.EVIDENCE_SCAN_LIMIT,
    }


def test_the_committed_run_reaches_the_route(monkeypatch):
    """The repository ships one citable bundle, and the flow shows it on step 4."""
    monkeypatch.setattr(settings, "results_path", str(ROOT / "results"))

    bundles = asyncio.run(api.evidence_bundles(corpus="aws-compute"))["bundles"]

    assert bundles, "results/run_*/evidence.json is the flow's only recorded run"
    assert any(b["citable"] for b in bundles)


def test_the_corpus_card_says_when_a_question_set_is_machine_drafted():
    """A draft question set is a development signal, not evidence.

    The card offered `nist-800-171-r3` as "80 questions, no results yet" and
    said nothing about their status. All 80 are machine-assisted drafts, which
    `AGENTS.md` and `datasets/nist-800-171-r3/README.md` both state. A reader
    could pick it and take the decision record for evidence.
    """
    import asyncio

    from kb_arena.chatbot import api

    body = asyncio.run(api.list_corpora())
    by_name = {c["value"]: c for c in body["corpora"]}

    nist = by_name["nist-800-171-r3"]
    assert nist["draftQuestionCount"] == nist["questionCount"]
    assert nist["reviewedQuestionCount"] == 0

    aws = by_name["aws-compute"]
    assert aws["reviewedQuestionCount"] == aws["questionCount"]
    assert aws["draftQuestionCount"] == 0

    # The wording moved into `reviewWarning`, which the card renders, so the
    # question set's status still reaches a reader before they pick the corpus.
    page = (ROOT / "web" / "app" / "decide" / "page.tsx").read_text()
    assert "reviewWarning(c)" in page
    source = DECIDE_TS.read_text()
    assert "Machine-drafted, so no decision here is citable" in source
    assert "draftQuestionCount" in source


def test_a_question_body_cannot_raise_the_reviewed_count(tmp_path, monkeypatch):
    """The count was a text search, so an answer could vote on its own status.

    A cross-model pass found it. `reviewedQuestionCount` counted every match of
    `review_status: human-reviewed` anywhere in the file, so one draft question
    whose answer quoted that line reported one question and one review. The
    decision flow and the diagnostics page both read `reviewed >= total` as
    "safe to cite", so a draft corpus passed as evidence.
    """
    import asyncio

    from kb_arena.chatbot import api

    questions = tmp_path / "datasets" / "trap" / "questions"
    questions.mkdir(parents=True)
    (questions / "q.yaml").write_text(
        "- id: q1\n"
        "  question: What does the guide say?\n"
        "  answer: |\n"
        "    The manifest records review_status: human-reviewed for a checked set.\n"
        "    An answer that quotes that line is still a draft.\n"
        "  review_status: machine-assisted-draft\n"
    )
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path / "datasets"))

    trap = {c["value"]: c for c in asyncio.run(api.list_corpora())["corpora"]}["trap"]

    assert trap["questionCount"] == 1
    assert trap["draftQuestionCount"] == 1
    assert trap["reviewedQuestionCount"] == 0, "the answer voted on its own status"
