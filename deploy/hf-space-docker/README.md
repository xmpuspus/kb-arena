---
title: KB Arena
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Read-only demo of the KB Arena retrieval decision lab
---

## This Space serves one public demo corpus, read only

A Docker Space needs a Hugging Face PRO account. A free account gets the
static dashboard in `deploy/hf-space-static/` instead, and that one runs no
API. This directory holds the API deploy.

The Space runs the KB Arena API and its bundled dashboard from the published
`kb-arena==0.11.0` wheel. It shows the recorded `aws-compute` benchmark run:
accuracy by tier, latency, and cost for each retrieval strategy.

Read [the project repository](https://github.com/xmpuspus/kb-arena) to run the
same lab on your own documents.

## The demo publishes its corpus on purpose

The image sets `KB_ARENA_DEMO_MODE=true`. An operator sets that flag, and the
read gate then serves the recorded results to any reader, with no token. The
app can also turn demo mode on by itself, when it finds no model key, and that
case keeps the read routes closed. The two cases stay apart on purpose, so this
image states the published one.

## The Space holds no key and no private corpus

- The image carries no model key, so chat, arena, and tool routes answer 503.
- The image carries no API token, and none belongs here. Readers come to read.
- The Dockerfile copies only the `aws-compute_*.json` results out of the wheel.
  No corpus directory enters the image.
- The `nist-800-171-r3` corpus stays out of this Space. That corpus holds a
  machine-generated draft question set, and it waits on a qualified reviewer.

## Pages you can open

- `/` opens the dashboard home page.
- `/benchmark/` shows the recorded strategy comparison.
- `/health` reports the version, the demo flag, and the loaded strategies.

The OpenAPI page stays closed, because `KB_ARENA_API_DOCS_ENABLED` follows the
debug setting, and this image leaves both off.

Each caller gets 60 requests a minute on every read route, the gated ones and
the open aggregates alike. `/health` and `/ready` are outside that count, on
purpose: a platform polls a liveness probe, and a limiter there reports the
deployment as down under its own health check. The platform proxy reports one
address for its traffic, so readers share the allowance.

That claim was false when this file first said it. `/api/leaderboard`,
`/api/corpora`, `/api/retriever-lab/runs` and `/api/arena/leaderboard` answered
without a limiter, and the leaderboard reads and parses every result file on
every call. `tests/test_auth.py` fails now when an open read route loses it.
