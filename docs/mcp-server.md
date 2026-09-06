# MCP server

KB Arena ships a Model Context Protocol server. It exposes eight tools for
corpus, strategy, and benchmark work over stdio, so an editor or agent can
call kb-arena the same way the CLI does, without shelling out.

![A stdio client reads the eight tools and three answers from the running server](demo-mcp.gif)

The recording shows a client that starts the server, lists its eight tools,
and then reads the answers to `list_corpora`, `validate_corpus`, and
`compare`. Repeat it with `python3 scripts/mcp_stdio_demo.py` after you
install the extra, or record it again with `vhs docs/tapes/mcp.tape`.

The server calls a plain kb-arena function for every tool, never a CLI
command, so no tool result depends on a Typer wrapper. No tool returns an
empty result to mean the call failed. A failure raises instead, so a caller
can tell "nothing here" from "could not check".

Every raised exception reaches the client as a `ToolError`, carrying the
original message. Below, "raises" always means the client sees a
`ToolError` with the stated text, never a generic crash message.

## Install the extra

```bash
pip install 'kb-arena[mcp]'
```

The extra installs `mcp==2.1.1`, which needs `sse-starlette>=3.0.0`. The core
package already pins `sse-starlette==3.4.10` in its own dependencies, for
this reason, so the two requirements resolve to one version.

## Start the server

```bash
python3 -m kb_arena.mcp.server
```

The server speaks JSON-RPC over stdin and stdout. An MCP client starts this
command as a subprocess and sends requests on stdin. It does not open a
network port and does not take command-line flags.

The server writes every log line to stderr, because stdout carries only the
protocol. A benchmark run started through `start_benchmark` prints its
progress through the same redirect, so a client parsing stdout never sees
progress text mixed into a JSON-RPC message.

The server reads the same `KB_ARENA_*` settings as the CLI, for example
`KB_ARENA_DATASETS_PATH` and `KB_ARENA_RESULTS_PATH`. Set them before you
start the server, the same way you would for `kb-arena` on the command line.

## The eight tools

### `list_corpora`

No arguments.

Lists every corpus under the configured datasets root, with its pipeline
status. Returns `{"corpora": [...]}`, where each entry carries `name`,
`has_processed`, `question_count`, and `has_results`. Raises if the
configured datasets root itself does not exist.

### `list_strategies`

No arguments.

Lists every built-in strategy and its runtime status. Returns
`{"strategies": [...], "catalog": [...]}`. The server builds no strategy
runtime of its own, so an entry with a missing optional dependency still
reports its install hint instead of disappearing from the list.

### `validate_corpus`

| Argument | Type | Required | Default |
|---|---|---|---|
| `corpus` | string | yes | |

Checks whether a corpus exists under the configured root and is buildable.
A corpus that does not exist is a normal, reported outcome. The tool returns
`{"corpus": ..., "valid": false, "reason": "corpus directory not found"}`.
A found corpus reports `has_processed`, `has_questions`, `question_count`,
and an `errors` list for a question file that did not load. An unsafe
corpus name, for example one with a path-traversal segment, raises instead
of returning a result.

### `start_benchmark`

| Argument | Type | Required | Default |
|---|---|---|---|
| `corpus` | string | no | `"all"` |
| `strategy` | string | no | `"all"` |
| `tier` | integer | no | `0` |
| `split` | string | no | `""` |
| `top_k` | integer | no | `5` |

Starts a benchmark run in the background and returns
`{"job_id": ..., "status": "queued"}` right away, because a benchmark can run
for minutes. Poll `job_status` with the returned job ID for the outcome.
Raises for an unknown strategy name, for a `top_k` outside the strategies'
allowed range, or when four benchmark jobs are already running, which is the
server's concurrency cap.

### `job_status`

| Argument | Type | Required | Default |
|---|---|---|---|
| `job_id` | string | yes | |

Returns the current state of a job started by `start_benchmark`. The fields
are `job_id`, `corpus`, `strategy`, `status` (`queued`, `running`,
`completed`, or `failed`), `created_at`, `run_id`, `error`, and
`finished_at`. Raises for an unknown job ID rather than returning a default
status.

### `compare`

| Argument | Type | Required | Default |
|---|---|---|---|
| `corpus` | string | yes | |
| `a` | string | yes | |
| `b` | string | yes | |
| `run_a` | string | no | `""` |
| `run_b` | string | no | `""` |
| `metric` | string | no | `"accuracy"` |

Pairs two strategies question by question on the same corpus, from their
stored result files. The reported delta is `b` minus `a`. Raises with the
message `no result file at <path>` when a named result file is missing.

### `get_manifest`

| Argument | Type | Required | Default |
|---|---|---|---|
| `corpus` | string | yes | |
| `strategy` | string | yes | |
| `run_id` | string | no | `""` |

Returns the manifest stored with one result file. The fields are `corpus`,
`strategy`, `run_id`, `manifest`, and `summary`. A result file written before manifests
existed returns an empty manifest and an empty summary, never a fabricated
one. Raises when the named result file is missing.

### `export_evidence`

| Argument | Type | Required | Default |
|---|---|---|---|
| `corpus` | string | yes | |
| `run_id` | string | yes | |

Writes the evidence bundle for a completed run, mirroring `kb-arena
evidence`. A run whose questions are not fully human-reviewed does not get
written as citable evidence. It gets `{"written": false, "problems": [...]}`
instead, naming each reason the bundle was refused. A run that passes every
check returns `{"written": true, "path": ..., "citable": ..., "bundle": ...}`.
Raises when the named run directory does not exist or when the run ID is not
a valid ID.

## Client configuration

### Claude Code

Add the server with the `claude mcp add` command, or add this block to
`.mcp.json` in the project root.

```json
{
  "mcpServers": {
    "kb-arena": {
      "command": "python3",
      "args": ["-m", "kb_arena.mcp.server"]
    }
  }
}
```

### Cursor

Add this block to `.cursor/mcp.json` in the project root, or to Cursor's
global MCP settings.

```json
{
  "mcpServers": {
    "kb-arena": {
      "command": "python3",
      "args": ["-m", "kb_arena.mcp.server"]
    }
  }
}
```

### A generic stdio client

Any client that speaks MCP over stdio starts the same command and talks
JSON-RPC over its stdin and stdout.

```json
{
  "command": "python3",
  "args": ["-m", "kb_arena.mcp.server"]
}
```

Point the client's working directory, or its `KB_ARENA_DATASETS_PATH` and
`KB_ARENA_RESULTS_PATH` settings, at the checkout or corpus root you want the
tools to read.
