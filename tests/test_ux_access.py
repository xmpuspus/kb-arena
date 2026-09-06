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
