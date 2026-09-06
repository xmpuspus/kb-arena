"""A static Space serves exact file paths, so its copy links to them.

The live Space answered `/benchmark/` with a redirect to huggingface.co, while
`/benchmark/index.html` sat in the same Space answering 200. The export cannot
write those links, because the packaged bundle behind the API is served by
StaticFiles, which does resolve a directory. So the rewrite runs on the copy the
deploy makes, and these tests hold both halves of that.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "hf-space-static" / "link_exact_paths.py"


def _module():
    spec = importlib.util.spec_from_file_location("link_exact_paths", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<a href="/benchmark/">B</a>', '<a href="/benchmark/index.html">B</a>'),
        ('<a href="/retriever-lab/">L</a>', '<a href="/retriever-lab/index.html">L</a>'),
        # The root is a path the Space serves, so it stays.
        ('<a href="/">Home</a>', '<a href="/">Home</a>'),
        # An absolute URL belongs to another host.
        (
            '<a href="https://github.com/xmpuspus/kb-arena/">gh</a>',
            '<a href="https://github.com/xmpuspus/kb-arena/">gh</a>',
        ),
        # An asset already names a file.
        ('<link href="/_next/static/chunks/a.css"/>', '<link href="/_next/static/chunks/a.css"/>'),
        (
            '<a href="/favicon.ico?favicon.x.ico">f</a>',
            '<a href="/favicon.ico?favicon.x.ico">f</a>',
        ),
    ],
)
def test_only_a_route_link_gains_a_file_name(html, expected):
    known = {"/benchmark/", "/retriever-lab/"}
    assert _module().rewrite(html, known)[0] == expected


def test_the_rewrite_refuses_the_packaged_bundle(capsys):
    """The API resolves a directory, so the packaged bundle keeps its links."""
    assert _module().main(ROOT / "kb_arena" / "static") == 1
    assert "Refusing to rewrite the packaged bundle" in capsys.readouterr().err


def test_the_packaged_bundle_still_links_to_directories():
    """A guard on the other half. StaticFiles serves these, the Space does not."""
    index = (ROOT / "kb_arena" / "static" / "index.html").read_text()
    assert 'href="/benchmark/"' in index
    assert 'href="/benchmark/index.html"' not in index


def test_the_deploy_runs_the_rewrite_on_the_copy():
    push = (ROOT / "deploy" / "hf-space-static" / "push.sh").read_text()
    assert 'python3 "$SOURCE_DIR/link_exact_paths.py" "$WORK_DIR/space"' in push
    # It has to run after the copy and before the commit, or it rewrites nothing.
    assert push.index('cp -R "$BUNDLE"') < push.index("link_exact_paths.py")
    assert push.index("link_exact_paths.py") < push.index("git add -u")


def test_the_deploy_fails_when_a_route_survives_anywhere_it_is_served(tmp_path, capsys):
    """The anchors are not the only place a route can sit.

    A reviewer asked whether the hydration payload keeps the directory route,
    where the client router would read it after the page loads. Rather than
    reason about the bundler, the deploy reads every file the browser can fetch
    and refuses when one holds a route it just rewrote.
    """
    space = tmp_path / "space"
    (space / "benchmark").mkdir(parents=True)
    (space / "index.html").write_text('<a href="/benchmark/">B</a>')
    (space / "benchmark" / "index.html").write_text("<p>b</p>")
    # A payload file, which the rewrite does not touch. The check is the
    # backstop for every file type the rewrite leaves alone.
    payload = space / "index.txt"
    payload.write_text('1:{"href":"/benchmark/"}')

    module = _module()
    assert module.main(space) == 1
    assert "still written as a directory" in capsys.readouterr().err

    payload.write_text('1:{"href":"/benchmark/index.html"}')
    assert module.main(space) == 0


def test_the_check_reads_the_routes_off_the_build():
    """A hand-kept route list drifts. The directories holding an index name them."""
    module = _module()
    found = module.routes(ROOT / "kb_arena" / "static")
    assert "/benchmark/" in found
    assert "/decide/" in found
    assert "/diagnostics/" in found


def test_a_link_to_a_page_the_build_never_wrote_stops_the_deploy(tmp_path, capsys):
    """`/missing/index.html` is a 404 wearing the right shape."""
    space = tmp_path / "space"
    space.mkdir()
    (space / "index.html").write_text('<a href="/missing/">x</a>')

    module = _module()
    assert module.main(space) == 1
    assert "never wrote" in capsys.readouterr().err
    # The link is left alone, so the failure names the real problem.
    assert 'href="/missing/"' in (space / "index.html").read_text()


def test_the_packaged_bundle_guard_resolves_the_path_first(capsys):
    """`kb_arena/static/../static` reaches the same directory."""
    module = _module()
    assert module.main(ROOT / "kb_arena" / "static" / ".." / "static") == 1
    assert "Refusing to rewrite the packaged bundle" in capsys.readouterr().err


def test_the_navigation_uses_plain_anchors():
    """A next/link intercepts the click and asks for a payload that is not there.

    The hosted Space serves exact file paths, so the deploy rewrites each route
    link to its index.html. A next/link then requests a route payload under that
    name, gets a 404, and the page never moves. The click has to reach the
    browser.
    """
    for name in ("web/components/Nav.tsx", "web/app/page.tsx"):
        source = (ROOT / name).read_text()
        assert 'import Link from "next/link"' not in source, f"{name} intercepts its own links"
        assert "<a" in source, f"{name} must hand the click to the browser"


def test_a_route_inside_a_chunk_is_rewritten_too(tmp_path):
    """React redraws the anchor from the chunk, over the rewritten HTML.

    The navigation list is a data array, so the bundler writes each route into a
    chunk. Rewriting the HTML alone put the directory path back at hydration,
    which the deploy check caught on a real push.
    """
    space = tmp_path / "space"
    (space / "benchmark").mkdir(parents=True)
    (space / "index.html").write_text('<a href="/benchmark/">B</a>')
    (space / "benchmark" / "index.html").write_text("<p>b</p>")
    chunk = space / "_next" / "static" / "chunks"
    chunk.mkdir(parents=True)
    (chunk / "nav.js").write_text('const links=[{href:"/benchmark/",label:"Benchmark"}];')

    assert _module().main(space) == 0
    assert '"/benchmark/index.html"' in (chunk / "nav.js").read_text()


def test_a_route_that_is_not_a_route_stays_untouched_in_a_chunk(tmp_path):
    """Only a path the build wrote as a page is rewritten."""
    space = tmp_path / "space"
    (space / "benchmark").mkdir(parents=True)
    (space / "index.html").write_text("<p>home</p>")
    (space / "benchmark" / "index.html").write_text("<p>b</p>")
    chunk = space / "app.js"
    chunk.write_text('fetch("/api/corpora");const q="/other/";')

    assert _module().main(space) == 0
    body = chunk.read_text()
    assert '"/api/corpora"' in body
    assert '"/other/"' in body
