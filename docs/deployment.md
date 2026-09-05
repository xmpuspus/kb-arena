# Deployment

This page covers running the API and the dashboard, what each environment
variable controls, which services you can skip, and the failures a first
deploy actually hits.

Read [the environment reference](reference-environment.md) for the full,
generated list of settings. This page names only the ones that change
deployment behavior. Read [getting started](getting-started.md) for corpus
setup and provider choices, and [the HTTP reference](reference-http.md) for
every route and its auth rule.

## Run the API and the dashboard locally

```bash
pip install kb-arena
kb-arena serve --host 127.0.0.1 --port 8000
```

`kb-arena serve` runs `uvicorn kb_arena.chatbot.api:app` under the hood. Pass
`--reload` during development. `kb-arena demo` does the same, then opens the
bundled dashboard against the packaged `aws-compute` results, with no key
and no Docker service needed.

The Next.js dashboard runs on its own too.

```bash
cd web
npm run dev
```

`npm run build` produces a static export in `web/out`, because
`next.config.mjs` sets `output: "export"`. A static export has no server to
read environment variables from at request time, so `NEXT_PUBLIC_API_URL`
Set it before you run the build. Setting it afterward has no effect on the
static files already exported.

## Run the full stack with Docker Compose

```bash
export KB_ARENA_NEO4J_PASSWORD=choose-a-password
docker compose up -d
```

`docker-compose.yml` defines three services. `neo4j` stores the graph.
`api` builds from the repo `Dockerfile` and serves `kb_arena.chatbot.api:app`
on port 8000. `web` builds the dashboard's static export and serves it on
port 3000. Compose passes it `NEXT_PUBLIC_API_URL` only as a build argument,
because the export has no server left to read a runtime variable from.

Compose refuses to start without `KB_ARENA_NEO4J_PASSWORD` set. This command
checks that, with nothing started yet.

```bash
docker compose config
```

## Environment variables that change deployment behavior

The full list lives in [the environment reference](reference-environment.md),
generated from `kb_arena/settings.py`. The variables below decide whether the
process starts, what it can reach, and who may call it.

| Variable | What it controls |
|---|---|
| `KB_ARENA_HOST`, `KB_ARENA_PORT` | The bind address and port for the API |
| `KB_ARENA_API_TOKEN` | Bearer token required on content and LLM routes once set |
| `KB_ARENA_DEMO_MODE` | Turns `/chat`, arena, and tool routes into a 503, for a public read-only demo |
| `KB_ARENA_CORS_ORIGINS` | Origins the browser may call the API from, beyond `localhost:3000` and `3001` |
| `KB_ARENA_TRUSTED_PROXY_HEADER` | The forwarded-for header a loopback proxy may set for the real client address |
| `KB_ARENA_LLM_PROVIDER`, `KB_ARENA_ANTHROPIC_API_KEY`, `KB_ARENA_OPENAI_API_KEY` | Which model answers a question, and its key |
| `KB_ARENA_EMBEDDING_PROVIDER` and its matching key | Which service embeds chunks and queries |
| `KB_ARENA_NEO4J_URI`, `KB_ARENA_NEO4J_USER`, `KB_ARENA_NEO4J_PASSWORD`, `KB_ARENA_NEO4J_DATABASE` | Where the graph strategies connect |
| `KB_ARENA_CHROMA_PATH`, `KB_ARENA_DATASETS_PATH`, `KB_ARENA_RESULTS_PATH` | Where indexes, corpora, and run artifacts live on disk |
| `KB_ARENA_BENCHMARK_COST_CAP_USD` | Stops a benchmark run once observed spend passes this amount, `0` disables it |

## What each service costs to skip

**Neo4j.** Needed only when the loaded strategy set includes
`knowledge_graph`, `lightrag`, or `hybrid`. Without it, those strategies fall
back to mock graph data with a logged warning instead of failing outright.
`/ready` fails outside demo mode when the app loads one of the three and
Neo4j does not answer.

**A generation model key.** Without `KB_ARENA_ANTHROPIC_API_KEY`,
`KB_ARENA_OPENAI_API_KEY`, or an Ollama setup, the app sets
`KB_ARENA_DEMO_MODE=true` on its own at startup. Chat, arena, and tool routes
then return 503. The static benchmark, leaderboard, and corpus list routes
stay open.

**An embedding provider key.** Needed for every strategy except BM25 and
SPLADE, which read no vector index. An embedding-backed strategy run with no
key still starts the server, then fails at the embedding call.

**`KB_ARENA_API_TOKEN`.** Without it, a route that returns corpus content
refuses a caller that is not on the loopback address, unless an operator set
`KB_ARENA_DEMO_MODE=true` on purpose. A demo mode the app enabled on its own,
for lack of a generation key, does not count as that choice.

## One process, one worker

Run one `uvicorn` worker and one API replica. Graph-build progress and other
live tool jobs use process-local state, so a second worker or replica would
lose track of jobs a caller started against the first one. Add an external
job and event store before you scale the API past one process.

## Health and readiness

`GET /health` is open and always returns 200. It reports the app version,
whether Neo4j and the LLM client connected, and the loaded strategy names.
Use it for a liveness dashboard.

`GET /ready` returns 503 outside demo mode until Neo4j answers, when the app
loads a graph strategy, and until you set a generation key. Both the
Dockerfile and `docker-compose.yml` poll `/ready` for their health checks.
`scripts/docker-smoke.sh` builds the image, starts it in demo mode, and polls
`/ready` and `/health` the same way CI does.

## Troubleshooting a first deploy

**`docker compose up` exits with a missing-variable error.**
`KB_ARENA_NEO4J_PASSWORD` has no default. Set it before you start compose.

**The container never reports healthy, though the process stays up.**
Outside demo mode, `/ready` returns 503 until a generation key is
configured. Set `KB_ARENA_ANTHROPIC_API_KEY` or `KB_ARENA_OPENAI_API_KEY`,
or set `KB_ARENA_DEMO_MODE=true` if a read-only demo is what you want.

**The dashboard loads, but every API call goes to the wrong host, or
fails.** `NEXT_PUBLIC_API_URL` is baked into the static export at build
time. Setting it as a compose `environment:` entry or an OS variable at
container start has no effect. Rebuild the `web` image with the correct
`NEXT_PUBLIC_API_URL` build argument.

**The browser blocks API calls with a CORS error.**
`KB_ARENA_CORS_ORIGINS` defaults to `http://localhost:3000` and `:3001`
only. Set it to the dashboard's real origin as a JSON array, for example
`KB_ARENA_CORS_ORIGINS='["https://dashboard.example.com"]'`.

**A remote read request returns 401 `api_token_required_for_remote_access`.**
The API accepts every caller on the loopback address by default, but refuses
a remote one until you set `KB_ARENA_API_TOKEN`, or set
`KB_ARENA_DEMO_MODE=true` to publish the corpus on purpose.

**Chat or tool routes return 503 `demo_mode`, though a key looks set.**
Check that the key variable matches `KB_ARENA_LLM_PROVIDER`. An Anthropic
key with `KB_ARENA_LLM_PROVIDER=openai` still leaves generation unconfigured,
and demo mode turns itself on.

**Knowledge Graph, Hybrid, or LightRAG answers with a mock-data warning.**
Neo4j is unreachable. Check `KB_ARENA_NEO4J_URI`, `KB_ARENA_NEO4J_USER`, and
`KB_ARENA_NEO4J_PASSWORD` against the running `neo4j` service, and confirm
its health check passes.
