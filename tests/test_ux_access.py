"""Every control on the dashboard has a name, a keyboard path and a visible edge.

A reviewer measured the theme and found the interactive border token at 1.23
against the card, where the requirement is 3. The same pass found form controls
whose label sat beside them with nothing joining the two, and table headers that
sorted on a click and did nothing on a key.

Each of those is a list that a person closes once and reopens the next time
somebody adds a control. These tests hold the list at zero instead.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
CSS = WEB / "app" / "globals.css"

CONTROL = re.compile(r"<(input|select|textarea)\b")
CLICKABLE = re.compile(r"<(th|tr|td|li)\b")


def _pages() -> list[Path]:
    return sorted(list((WEB / "app").rglob("*.tsx")) + list((WEB / "components").rglob("*.tsx")))


def _opening_tag(lines: list[str], index: int) -> str:
    """The text of one JSX opening tag, which can run over several lines.

    Splitting on the first ">" cuts the tag at an arrow function, so a style
    written after an onClick handler stayed invisible to these checks. The scan
    below steps over "=>" and stops at the real tag end.
    """
    text = " ".join(lines[index : index + 14])
    position = 0
    while position < len(text):
        found = text.find(">", position)
        if found == -1:
            return text
        if found > 0 and text[found - 1] == "=":
            position = found + 1
            continue
        return text[: found + 1]
    return text


def _token(name: str) -> str:
    match = re.search(rf"^\s*--{name}:\s*(#[0-9a-fA-F]{{6}});", CSS.read_text(), re.MULTILINE)
    assert match, f"the theme declares no --{name}"
    return match.group(1)


def _relative_luminance(colour: str) -> float:
    channels = []
    for part in (colour[1:3], colour[3:5], colour[5:7]):
        value = int(part, 16) / 255
        channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(foreground: str, background: str) -> float:
    first, second = _relative_luminance(foreground), _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def test_every_form_control_carries_a_name_a_reader_can_hear():
    """A label beside a control is not a label on it.

    A screen reader reads the accessible name, so a visual label with nothing
    joining it to the control leaves the control unnamed. Either the label
    carries htmlFor and the control carries the matching id, or the control
    carries its own aria-label, or the control sits inside the label element.
    """
    unnamed = []
    for path in _pages():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            if not CONTROL.search(line):
                continue
            tag = _opening_tag(lines, index)
            if "aria-label" in tag or "aria-labelledby" in tag or " id=" in tag:
                continue
            before = " ".join(lines[max(0, index - 3) : index])
            if "<label" in before and "</label>" not in before:
                continue
            unnamed.append(f"{path.relative_to(ROOT)}:{index + 1}")
    assert not unnamed, "these controls carry no accessible name: " + ", ".join(unnamed)


def test_a_table_header_that_sorts_on_a_click_also_sorts_on_a_key():
    """A th is not a control, so the click handler needs a real button inside."""
    unreachable = []
    for path in _pages():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            if not CLICKABLE.search(line):
                continue
            tag = _opening_tag(lines, index)
            if "onClick" not in tag:
                continue
            if "tabIndex" in tag or "onKeyDown" in tag or "role=" in tag:
                continue
            # The element itself takes no key. A button inside it does the same
            # work, so the reader reaches the behaviour either way.
            body = " ".join(lines[index : index + 24])
            if "<button" in body:
                continue
            unreachable.append(f"{path.relative_to(ROOT)}:{index + 1}")
    assert not unreachable, "these click targets have no keyboard path: " + ", ".join(unreachable)


def test_the_interactive_border_token_meets_the_three_to_one_bar():
    """The plain border token draws separators. Controls need the strong one."""
    card, background = _token("card"), _token("background")
    strong = _token("border-strong")
    assert contrast(strong, card) >= 3.0
    assert contrast(strong, background) >= 3.0
    # The plain token is below the bar on purpose, which is why a control must
    # not use it. This assertion documents that, and fails if somebody raises
    # the plain token and leaves this file saying otherwise.
    assert contrast(_token("border"), card) < 3.0


def test_no_control_draws_its_edge_with_the_weak_border_token():
    weak = []
    for path in _pages():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            if not re.search(r"<(input|select|textarea|button)\b", line):
                continue
            tag = _opening_tag(lines, index)
            if "borderColor" not in tag:
                continue
            if 'borderColor: "var(--border)"' in tag:
                weak.append(f"{path.relative_to(ROOT)}:{index + 1}")
    assert not weak, "these controls draw a 1.23 edge: " + ", ".join(weak)


def test_no_shared_style_or_active_state_hands_a_control_the_weak_token():
    """The tag scan misses a border set through a variable or a ternary.

    A reviewer found four decision-flow controls and the shared select style
    still on the divider token, none of which the scan above can see: one hides
    the token in a `const ...Style` object, the other behind `active ? a : b`.
    """
    offenders = []
    for path in _pages():
        for index, line in enumerate(path.read_text().splitlines()):
            # The active-state idiom marks a control, so its inactive edge is a
            # control edge too.
            if 'var(--accent)" : "var(--border)"' in line:
                offenders.append(f"{path.relative_to(ROOT)}:{index + 1}")
    for path in _pages():
        source = path.read_text()
        for match in re.finditer(r"const (\w*Style) = \{(.*?)\};", source, re.S):
            if 'borderColor: "var(--border)"' in match.group(2):
                line = source[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line} ({match.group(1)})")
    assert not offenders, "these hand a control the 1.23 edge: " + ", ".join(offenders)


def test_the_diagnostics_route_is_reachable_from_the_navigation():
    """A route only a URL reaches is a route an operator never finds."""
    nav = (WEB / "components" / "Nav.tsx").read_text()
    assert '"/diagnostics"' in nav
    assert (WEB / "app" / "diagnostics" / "page.tsx").is_file()


def test_every_metric_note_reaches_a_page():
    """A note nobody renders explains nothing."""
    source = (WEB / "components" / "MetricNote.tsx").read_text()
    keys = re.findall(r"^  (\w+): \{$", source, re.MULTILINE)
    assert len(keys) >= 6, f"expected the full metric set, found {keys}"
    rendered = " ".join(
        p.read_text() for p in _pages() if p.name not in {"MetricNote.tsx", "InfoTip.tsx"}
    )
    # A page names the note directly, or hands it to a header that renders it.
    missing = [k for k in keys if f'metric="{k}"' not in rendered and f'note="{k}"' not in rendered]
    assert not missing, f"these metric notes render nowhere: {missing}"


def test_the_theme_answers_a_reduced_motion_request():
    css = CSS.read_text()
    assert "prefers-reduced-motion" in css


def test_an_empty_strategy_pick_refuses_instead_of_running_everything():
    """A page that says 0 of 19 must not hand over a command that runs nine.

    The panel used to fall back to `--strategy all` on an empty pick, so the
    count on screen and the command below it said different things.
    """
    panel = (WEB / "components" / "StrategyRunPanel.tsx").read_text()
    assert 'selectedNames.join(",")' in panel
    assert (
        'selectedNames.join(",") : "all"' not in panel
    ), "an empty pick must not fall back to every strategy"
    assert "NOTHING_PICKED" in panel
    assert panel.count("if (!strategyArg) return NOTHING_PICKED;") == 2
    assert "disabled={nothingPicked || corpora.length === 0}" in panel


def test_the_run_panel_refuses_a_value_the_tool_would_refuse():
    """The panel builds a line a reader pastes into a shell, so it validates.

    decide.ts closed this for the values that arrive from the URL. The run panel
    takes its corpus from the API and its strategy names from the catalog, and
    both reached the command with no check.
    """
    panel = (WEB / "components" / "StrategyRunPanel.tsx").read_text()
    assert 'from "@/lib/decide"' in panel, "reuse the one validator, do not copy the pattern"
    guard = "if (![corpus, ...selectedNames].every(isSafeId)) return UNSAFE_COMMAND;"
    assert panel.count(guard) == 2
    # An empty corpus list left the built-in default in place, so the panel
    # offered a command for a corpus the deployment never reported.
    assert panel.count("if (corpora.length === 0) return NO_CORPUS;") == 2


def test_the_decision_flow_carries_one_corpus_name_from_step_one():
    """A reader who names their own corpus must not read another one's evidence.

    Step 1 stored the typed name and every later step used the built-in pick, so
    the evidence read, the comparison and the downloaded record all named the
    example corpus.
    """
    page = (WEB / "app" / "decide" / "page.tsx").read_text()
    assert "const activeCorpus = ownDocs ? ownCorpus : corpus;" in page
    for call in (
        "fetchEvidenceBundles(activeCorpus)",
        "fetchCompare(activeCorpus, stratA, stratB, metric)",
        'corpus: activeCorpus || "none chosen",',
        'benchmarkCommand(activeCorpus || "my-docs", picked)',
        'labCommand(activeCorpus || "my-docs", picked)',
        'compareCommand(activeCorpus || "my-docs", stratA || "a", stratB || "b", metric)',
    ):
        assert call in page, f"this step still reads the built-in pick: {call}"


def test_a_reader_that_cannot_parse_a_body_says_so_instead_of_thinning_it():
    """A dropped row is a failed read, not a smaller answer.

    `parseStrategies` skipped entries it could not read and still answered ok,
    so a catalog nobody can parse showed as "0 of 0 strategies loaded". The
    comparison guard checked that per_question was an array and nothing about
    its rows, so a 200 carrying [null] crashed the table.
    """
    diagnostics = (WEB / "lib" / "diagnostics.ts").read_text()
    parser = diagnostics[diagnostics.index("export function parseStrategies") :]
    parser = parser[: parser.index("export async function readStrategies")]
    assert "continue;" not in parser, "a skipped entry hides an unreadable body"
    assert parser.count("return { ok: false, reason: WRONG_SHAPE };") >= 4

    decide = (WEB / "lib" / "decide.ts").read_text()
    assert "const rowsAreShaped =" in decide
    assert 'typeof cells.question_id === "string"' in decide


def test_the_diagnostics_page_calls_a_missing_flag_unknown():
    """A field the server never sent is not a yes.

    The model branch handled false and let null fall through to the success
    answer, which named an ability nobody measured.
    """
    page = (WEB / "app" / "diagnostics" / "page.tsx").read_text()
    assert "if (health.llmAvailable === null) {" in page
    # The null branch must return the unknown wording, not fall through to the
    # success answer below it.
    branch = page.index("if (health.llmAvailable === null) {")
    following = page[branch : branch + 400]
    assert "NOT_REPORTED" in following
    assert following.index("NOT_REPORTED") < following.index("can call the model")


def test_the_decision_flow_refuses_an_empty_strategy_pick_the_way_the_panel_does():
    """Two builders for the same command must not disagree on the empty case."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    assert 'return names.join(",");' in lib
    assert 'names.length ? names.join(",") : "all"' not in lib
    assert "return names.length ? command(parts, values) : NOTHING_PICKED;" in lib


def test_an_empty_corpus_drops_the_evidence_the_last_one_left():
    """Returning early kept the old bundles under the heading "no corpus"."""
    page = (WEB / "app" / "decide" / "page.tsx").read_text()
    branch = page.index("if (!activeCorpus) {")
    body = page[branch : page.index("const ticket = ++evidenceTicket.current;", branch)]
    for cleared in ("evidenceTicket.current += 1;", "setBundles([]);", "setBundlesUnreadable([]);"):
        assert cleared in body, f"the empty branch leaves this behind: {cleared}"


def test_the_comparison_guard_covers_every_field_the_type_declares():
    """Three rounds each asked for one more field, so this checks the whole type."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    declared = lib[lib.index("export interface CompareResult {") :]
    declared = declared[: declared.index("\n}")]
    fields = re.findall(r"^  (\w+):", declared, re.MULTILINE)
    start = lib.index("const ci = data.delta_ci_95;")
    guard = lib[start : lib.index("throw new Error(COMPARE_UNREADABLE)", start)]
    missing = [f for f in fields if f not in guard and f != "delta_ci_95"]
    assert not missing, f"the guard never checks these declared fields: {missing}"


def test_an_unreadable_question_file_blocks_the_citable_claim():
    """Dropping a failed file from both counts made a partial read look complete.

    The counts skipped a question file nobody could parse, so a corpus with one
    reviewed question and one broken file reported one of one reviewed. Both the
    decision flow and the diagnostics page read that as evidence.
    """
    api = (ROOT / "kb_arena" / "chatbot" / "api.py").read_text()
    assert '"unreadableQuestionFiles": unreadable_question_files,' in api
    # Three ways a question file fails to describe its corpus: it will not
    # read, it is not a list, and it holds an entry the loader rejects.
    assert api.count("unreadable_question_files += 1") == 3

    lib = (WEB / "lib" / "decide.ts").read_text()
    warning = lib[lib.index("export function reviewWarning") :]
    warning = warning[: warning.index("\n}")]
    assert "corpus.unreadableQuestionFiles" in warning
    # The unread branch has to run before the early return that reads a full
    # count as a full review.
    early_return = "if (total === 0 || reviewed >= total) return null;"
    assert warning.index("const unread = corpus.unreadableQuestionFiles") < warning.index(
        early_return
    )

    page = (WEB / "app" / "diagnostics" / "page.tsx").read_text()
    line = page[page.index("function corpusLine") :]
    line = line[: line.index("\n}")]
    assert "unreadableQuestionFiles" in line


def test_the_evidence_reader_checks_its_elements():
    """`bundles: [null]` passed a two-array check and crashed the step-4 table."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    assert "const bundlesAreShaped =" in lib
    assert "const unreadableIsShaped =" in lib
    assert 'data.unreadable.every((name) => typeof name === "string")' in lib


def test_the_truncation_notice_describes_the_scan_that_ran():
    """The scan takes whichever entries the filesystem lists first, not the newest."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    reason = lib[lib.index("export function noBundleReason") :]
    reason = reason[: reason.index("\n}")]
    assert "sits in the newest" not in reason, "the scan does not pick the newest entries"
    assert "Older directories went unread" not in reason
    assert "whichever directories the filesystem lists first" in reason


def test_the_run_panel_keeps_top_k_inside_what_retrieval_accepts():
    """A command the panel offers must be a command the tool will run.

    `kb_arena/strategies/base.py` refuses a top-k outside 1 to 1000, and the
    controls carried a minimum only.
    """
    base = (ROOT / "kb_arena" / "strategies" / "base.py").read_text()
    limit = re.search(r"^MAX_RETRIEVAL_CANDIDATES = (\d+)$", base, re.MULTILINE)
    assert limit, "kb_arena/strategies/base.py must declare the retrieval bound"

    panel = (WEB / "components" / "StrategyRunPanel.tsx").read_text()
    declared = re.search(r"^const MAX_RETRIEVAL_CANDIDATES = (\d+);$", panel, re.MULTILINE)
    assert declared, "the panel must declare the same bound"
    assert declared.group(1) == limit.group(1), "the panel copy drifted from the tool"
    assert panel.count("if (!kValuesFit) return OUT_OF_RANGE;") == 2
    assert panel.count("max={MAX_RETRIEVAL_CANDIDATES}") == 2


def test_neither_truncation_notice_claims_the_newest_runs():
    """The claim lived in two places, and one fix reached only one of them."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    page = (WEB / "app" / "decide" / "page.tsx").read_text()
    for name, source in (("decide.ts", lib), ("decide/page.tsx", page)):
        assert (
            "newest" not in source or "the newest run can be one it never opened" in source
        ), f"{name} still describes the scanned set as the newest"
    assert "holds the newest {bundlesTruncated}" not in page


def test_the_corpus_reader_checks_its_entries():
    """`corpora: [null]` reached the page and crashed on c.questionCount."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    reader = lib[lib.index("export async function fetchCorporaOrFail") :]
    reader = reader[: reader.index("\n}")]
    assert 'typeof fields.value !== "string"' in reader
    assert "questionCount" in reader


def test_the_evidence_reader_checks_the_types_the_table_reads():
    """`command: "..."` passed an object check and crashed on join()."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    assert "const listsAreLists =" in lib
    assert '["command", "results"]' in lib


def test_the_evidence_guard_covers_every_field_the_bundle_declares():
    """`citable: "false"` is truthy, so a corrupt bundle read as citable evidence."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    declared = lib[lib.index("export interface EvidenceBundle {") :]
    declared = declared[: declared.index("\n}")]
    fields = re.findall(r"^  (\w+)\?:", declared, re.MULTILINE)
    guard = lib[lib.index("const bundlesAreShaped =") : lib.index("const unreadableIsShaped =")]
    missing = [f for f in fields if f not in guard]
    assert not missing, f"the guard never checks these declared fields: {missing}"
    assert "const flagsAreFlags =" in guard


def test_an_absent_citable_verdict_is_not_read_as_a_negative_one():
    """A bundle that records no verdict said nothing, so the page must not.

    `bundleCaveats` and the step-4 badge both read every falsy value as an
    explicit "development signal", which put a conclusion in a bundle that
    never carried one.
    """
    lib = (WEB / "lib" / "decide.ts").read_text()
    assert "if (bundle.citable === undefined) {" in lib
    caveats = lib[lib.index("export function bundleCaveats") :]
    caveats = caveats[: caveats.index("\n}")]
    assert caveats.index("citable === undefined") < caveats.index(
        "calls this run a development signal"
    )

    page = (WEB / "app" / "decide" / "page.tsx").read_text()
    assert '"no verdict recorded"' in page
    assert page.count("b.citable === undefined") == 3


def test_a_malformed_question_entry_marks_the_corpus_unreadable():
    """`load_questions` raises on an entry with no id, so the corpus cannot run."""
    api = (ROOT / "kb_arena" / "chatbot" / "api.py").read_text()
    assert "malformed = True" in api
    assert "if malformed:\n                        unreadable_question_files += 1" in api


def test_the_evidence_scan_counts_every_entry_it_looks_at():
    """The cap counted kept directories, so an unrelated file never stopped it."""
    api = (ROOT / "kb_arena" / "chatbot" / "api.py").read_text()
    scan = api[api.index("entries: list[tuple[float, str, _Path]] = []") :]
    scan = scan[: scan.index("entries.sort(reverse=True)")]
    assert "if examined >= EVIDENCE_LIST_LIMIT:" in scan
    assert scan.index("examined += 1") < scan.index('entry.name.startswith("run_")')


def test_an_unreachable_results_directory_is_not_an_empty_deployment():
    """`Path.exists()` answers False on a permission error, which is not absence."""
    api = (ROOT / "kb_arena" / "chatbot" / "api.py").read_text()
    route = api[api.index("base = _Path(settings.results_path)") :]
    route = route[: route.index("scanned, truncated = _recent_run_dirs(base)")]
    assert "_os.stat(base)" in route
    assert "except FileNotFoundError:" in route
    assert "status_code=503" in route


def test_the_truncation_fields_are_checked_rather_than_coerced():
    """`Boolean("false")` is true, and a bad scan limit silently became zero."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    assert 'if (typeof data.truncated !== "boolean") throw new Error(EVIDENCE_UNREADABLE);' in lib
    assert "truncated: Boolean(data.truncated)" not in lib


def test_the_evidence_scan_does_not_follow_a_symlink_out_of_the_results_directory():
    """`/api/evidence` needs no token, so a planted symlink read any JSON file."""
    api = (ROOT / "kb_arena" / "chatbot" / "api.py").read_text()
    assert "follow_symlinks=False" in api
    scan = api[api.index("entries: list[tuple[float, str, _Path]] = []") :]
    scan = scan[: scan.index("entries.sort(reverse=True)")]
    assert "entry.is_dir()" not in scan, "is_dir follows a symlink by default"


def test_a_symlinked_bundle_file_is_refused_the_way_a_symlinked_directory_is():
    """Skipping the directory left the file, and the route still needs no token."""
    api = (ROOT / "kb_arena" / "chatbot" / "api.py").read_text()
    start = api.index('path = run_dir / "evidence.json"')
    end = api.index("bundles.append(bundle)", start)
    read = api[start:end]
    # One open refuses the symlink, so nothing can swap the name between a
    # check and a read.
    assert "_os.O_NOFOLLOW" in read
    assert "if path.is_symlink():" not in read, "a separate check leaves a race"
    assert "path.read_text()" not in read


def test_a_review_count_it_cannot_read_drops_the_whole_review_block():
    """Dropping one count left `publishable` behind it, which is the citable claim."""
    lib = (WEB / "lib" / "api.ts").read_text()
    parser = lib[lib.index("function reviewOf(") :]
    parser = parser[: parser.index("\n}")]
    assert 'if (typeof count !== "number" || !Number.isFinite(count)) return undefined;' in parser
    thinning = 'if (typeof count === "number" && Number.isFinite(count)) counts[status]'
    assert thinning not in parser


def test_the_evidence_route_bounds_the_bytes_it_reads():
    """Counting directories bounded the files opened, not the bytes read."""
    api = (ROOT / "kb_arena" / "chatbot" / "api.py").read_text()
    assert re.search(r"^EVIDENCE_MAX_BYTES = [\d_]+$", api, re.MULTILINE)
    # The read itself carries the bound. An fstat and then an unbounded read
    # is a check-then-act on a file another process can extend.
    assert "raw = handle.read(EVIDENCE_MAX_BYTES + 1)" in api
    assert "if len(raw) > EVIDENCE_MAX_BYTES:" in api
    assert "_os.fstat(handle.fileno())" not in api
    largest = max((p.stat().st_size for p in (ROOT / "results").rglob("evidence.json")), default=0)
    assert largest < 1_000_000, f"a bundle in this repository is {largest} bytes"


def test_the_record_claims_no_winner_when_the_runs_are_not_comparable():
    """A significance flag on incomparable runs is a winner from two question sets."""
    lib = (WEB / "lib" / "decide.ts").read_text()
    verdict = lib[lib.index('lines.push("### What the pairing supports");') :]
    verdict = verdict[: verdict.index("lines.push(c.note);")]
    assert verdict.index("!c.meta.comparable") < verdict.index("c.significant")
    assert "claims no winner and no significance" in verdict
