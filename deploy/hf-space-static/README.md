---
title: KB Arena
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
short_description: Static tour of the KB Arena dashboard, with sample data
---

## This Space is a static tour, and no API runs behind it

The Space serves the built KB Arena dashboard as files. A static Space is free
on Hugging Face, and it runs no process, so every `/api/...` call from these
pages answers 404.

Read [the project repository](https://github.com/xmpuspus/kb-arena) to run the
lab on your own documents, with the API and the recorded results.

## The numbers on these pages are sample data

The benchmark page falls back to the sample rows built into the dashboard, and
it marks that state: "Checked sample run. Use kb-arena benchmark to evaluate
your corpus." Those rows are not this project's recorded run. The sample BM25
row reads 0.2600 US dollars a query, and the recorded run reads 0.0035.

The table footer below those rows still reads "Results from your benchmark
runs". No run stands behind this Space, so read that line as sample data too.

The graph page draws a sample graph, and it says "Showing sample data because
Neo4j is not connected." This Space runs no database and no API, so that line
names the wrong cause. The graph you see is sample data.

## Open each page by its file path

The host answers exact file paths. The home page opens at `/`, which redirects
to `/index.html`. A second page needs its own file, for example
`/benchmark/index.html`. A request for `/benchmark/` leaves this host and
lands on `huggingface.co`, so the links in the top navigation do not move the
page. Open the file path instead.

## The Space holds no corpus and no key

- No corpus document travels with this deploy. The Space carries the dashboard
  files only.
- The `nist-800-171-r3` corpus stays out of this Space. That corpus holds a
  machine-generated draft question set, and it waits on a qualified reviewer.
- The Space carries no model key and no API token. It has no server to use one.

## Redeploy the dashboard

Run `deploy/hf-space-static/push.sh` from the repository. It copies
`kb_arena/static/`, the same dashboard build that the Python package ships.
Run `python3 scripts/sync_frontend_bundle.py` first when you change the web
sources.
