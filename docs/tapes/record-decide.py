"""Record the six-step decision flow as a GIF.

Drives Playwright against a running dashboard and walks `/decide/` the way a
reader does: pick the corpus, pick the objective, read the candidates, read the
commands, read the comparison, read the record. Every number on screen comes
from the deployment behind it, so run this against a server that holds a
recorded run.

Usage, with the dashboard on http://127.0.0.1:9911:

    python3 docs/tapes/record-decide.py

Set KB_ARENA_DEMO_BASE to record against another address.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

BASE = os.environ.get("KB_ARENA_DEMO_BASE", "http://127.0.0.1:9911")
OUT_DIR = Path(__file__).resolve().parents[1]  # docs/

# Each beat is the step the reader lands on and how long it holds. The pauses
# are long enough to read the heading and the first lines under it.
BEATS = [
    ("1. Corpus", 3000),
    ("2. Objective", 3000),
    ("3. Candidates", 3800),
    ("4. Run", 3800),
    ("5. Compare", 4200),
    ("6. Record", 4200),
]


def _target(name: str) -> Path:
    """The output path, inside docs/ and nowhere else.

    ffmpeg takes -y, so a name carrying .. or an absolute path would overwrite
    whatever it reached. This script writes one GIF beside the others.
    """
    out = (OUT_DIR / name).resolve()
    if out.parent != OUT_DIR or not out.name.endswith(".gif"):
        raise SystemExit(f"Write a .gif name inside {OUT_DIR}, not {name}.")
    return out


async def main(target_gif: str, width: int = 1400, height: int = 900) -> None:
    out = _target(target_gif)
    # A fresh directory per run. One shared name let a second recording delete
    # the first one's video while it was still being written.
    tmp_dir = Path(tempfile.mkdtemp(prefix="kbarena-decide-"))

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(tmp_dir),
            record_video_size={"width": width, "height": height},
        )
        page = await context.new_page()

        await page.goto(f"{BASE}/decide/", wait_until="networkidle")
        await page.wait_for_timeout(BEATS[0][1])

        for label, hold in BEATS[1:]:
            # The step tabs are buttons, so a click moves the flow the way a
            # reader moves it. `Next step` would work too and shows less.
            await page.get_by_role("button", name=label).click()
            if label.endswith("Compare"):
                # Without this the record reads "the comparison step did not
                # run", which is honest and shows nothing. A reader compares.
                await page.get_by_role("button", name="Read the comparison").click()
                await page.wait_for_timeout(1500)
            await page.wait_for_timeout(hold)

        await context.close()
        await browser.close()

    webm_files = sorted(tmp_dir.glob("*.webm"))
    if not webm_files:
        print("ERROR: no .webm captured", file=sys.stderr)
        sys.exit(1)
    webm = webm_files[-1]

    palette = tmp_dir / "palette.png"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(webm),
            "-vf",
            "fps=6,scale=900:-1:flags=lanczos,palettegen=stats_mode=diff:max_colors=96",
            str(palette),
        ]
    )
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(webm),
            "-i",
            str(palette),
            "-lavfi",
            "fps=6,scale=900:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=5",
            str(out),
        ]
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "demo-decide.gif"
    asyncio.run(main(target))
