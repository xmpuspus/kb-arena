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
    chunk = space / "app.js"
    chunk.write_text('const nav = [{href:"/benchmark/"}];')

    module = _module()
    assert module.main(space) == 1
    assert "still written as a directory" in capsys.readouterr().err

    chunk.write_text('const nav = [{href:"/benchmark/index.html"}];')
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
