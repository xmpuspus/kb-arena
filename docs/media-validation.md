# Canonical media validation

The README references only the three assets below. A reviewer validated the own-documents recording
and the benchmark visual on 2026-08-05, and the hero recording on 2026-09-06.

## The 2026-08-05 captures used the 0.10.0 release candidate

This block covers the own-documents recording and the benchmark visual. The hero recording lists
its own capture environment below.

- KB Arena 0.10.0 release candidate from the repository virtual environment
- VHS 0.11.0
- FFmpeg 8.1
- Node.js 25.2.1
- Google Chrome 150.0.7871.189

## The hero recording runs the published 0.11.0 wheel with no API key

- Asset: `docs/demo.gif`
- Source: `docs/tapes/hero-demo.tape`
- Duration: 44.16 seconds, 1104 frames
- Size: 479216 bytes
- SHA-256: `eab7447801b6b67839ab002f3690980aad876a9f999090e8455231f5c98da229`
- Command: `vhs docs/tapes/hero-demo.tape`
- Capture environment: the published `kb-arena==0.11.0` wheel in a clean virtualenv at
  `/tmp/release-verify`, vhs 0.11.0, FFmpeg 9.0.1, Python 3.12.4, macOS 26.5
- Numerical source: `results/run_9429b154`, the run the recording itself produced
- Reviewed frames: 001, 004, 010, 018, 024, 030, 036, and 043, extracted with
  `ffmpeg -i docs/demo.gif -vf fps=1 /tmp/hero-frame-%03d.png`

Each reviewed frame shows this:

- 001: the typed `kb-arena --version` command, above an empty prompt
- 004: the reply `kb-arena 0.11.0`, and the next comment line starting clean
- 010: the dataset adapter table, 8 rows, each with a licence and a pinned revision, no scroll
- 018: the typed `kb-arena ingest datasets/aws-compute/raw --corpus aws-compute` command
- 024: ingest at 3 of 3 files, then `Done. Built 1 vector index(es) from 3 documents`
- 030: the Retriever Lab table for run `9429b154`, Recall@5 0.275, P@5 0.171, Hit@5 0.440
- 036: the retrieval-ceiling note, then `Run 9429b154 written to results/run_9429b154/`
- 043: `results/run_422209dd/evidence.json: complete`, and the closing comment line

No frame shows an error, a truncated line, or a stale strategy count. The run directory
`results/run_9429b154/` stays out of the repository, because `.gitignore` excludes `results/run_*/`.
The tracked run `results/run_422209dd/retriever_lab.md` scored the same command and reports the
same numbers: Recall@5 0.275, P@5 0.171, Hit@5 0.440, MRR 0.352, NDCG@5 0.278, n 75. A reader can
check the numbers in the recording against that tracked report.

## Own-documents recording

- Asset: `docs/demo-own-docs.gif`
- Source: `docs/tapes/own-docs.tape`
- Duration: 26.24 seconds
- SHA-256: `9fa37e80a7e9286ee2326c7a3a98e57f6a8a3f95e2b35b33f0826fea219ee22a`
- Command: `vhs docs/tapes/own-docs.tape`
- Reviewed frames: 3.0, 12.0, 19.0, and 25.0 seconds

The tape uses a temporary directory, creates an `Operations Handbook` Markdown file, and runs the
real `init-corpus` and `ingest` commands. The final frame confirms one document and two sections.

## Historical benchmark visual

- Asset: `docs/benchmark-evidence.png`
- Source: `scripts/render_benchmark_evidence.mjs`
- Dimensions: 1200 by 675 pixels
- SHA-256: `aefc2c94c1400e6bc7e11ab94007dff3219910acebf76cde8cc30d104260b55a`
- Data source: `results/run_855aac4e/retriever_lab.json`

Regenerate the HTML with `node scripts/render_benchmark_evidence.mjs`, then capture it with a
1200 by 675 headless browser viewport. The full-resolution PNG was inspected after capture. Its six
rows and three metrics match the JSON to three decimal places, and the interpretation limit remains
visible at README size.

## The social preview is one frame of the hero recording

`docs/social-preview.png` is 1280x640, the size GitHub asks for. It is frame 36
of `docs/demo.gif`, cropped and scaled, so it shows the same real keyless run
and invents nothing.

GitHub has no REST endpoint for the social preview. Upload it by hand:
Settings, then General, then Social preview, then Edit, then Upload an image.
Check the result by opening `https://opengraph.githubassets.com/1/xmpuspus/kb-arena`
in a browser. GitHub caches that image, so the new one does not appear at once.

Rebuild the file after a new hero recording:

```
ffmpeg -i docs/demo.gif -vf fps=1/6 /tmp/hero-%02d.png
ffmpeg -y -i /tmp/hero-06.png -vf "crop=1200:600:0:0,scale=1280:640:flags=lanczos" docs/social-preview.png
```

## The MCP recording shows a live server, not a printed transcript

- Asset: `docs/demo-mcp.gif`
- Source: `docs/tapes/mcp.tape` driving `scripts/mcp_stdio_demo.py`
- Size: 220331 bytes, 1280x800, about 24 seconds
- SHA-256: `55a234986c10aba7d1a0a0e511fce43dd94dea50d829b768027dddcd9dc89f59`
- Capture environment: the `mcp` extra in a virtualenv at `/tmp/mcp-demo`, vhs 0.11.0

The driver starts `python3 -m kb_arena.mcp.server` as a subprocess and speaks
newline-delimited JSON-RPC on its stdin and stdout. Every line on screen comes
from the running server.

Four frames read back at one every six seconds:

- Frame 1: the `(mcp-demo)` prompt and the typed annotation.
- Frame 2: `-> initialize` sent and the reply `kb-arena, protocol 2025-11-25`.
- Frame 3: `tools/list` naming all eight tools, then the `list_corpora` table.
- Frame 4: the `compare` answer with `significant True` and `comparable False`,
  and the reason that neither file carries a manifest.

Rebuild it with:

```
python3 -m venv /tmp/mcp-demo && /tmp/mcp-demo/bin/pip install -e '.[mcp]'
vhs docs/tapes/mcp.tape
```
