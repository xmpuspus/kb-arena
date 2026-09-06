#!/usr/bin/env python3
"""Drive the KB Arena MCP server over stdio, the way an MCP client does.

Used by `docs/tapes/mcp.tape`, so the recording shows the real server
answering real JSON-RPC requests rather than a printed transcript.

Repeat it with the `mcp` extra installed:

    python3 -m venv /tmp/mcp-demo
    /tmp/mcp-demo/bin/pip install -e '.[mcp]'
    /tmp/mcp-demo/bin/python3 scripts/mcp_stdio_demo.py

The script starts `python3 -m kb_arena.mcp.server` as a subprocess and speaks
newline-delimited JSON-RPC on its stdin and stdout: `initialize`, then
`notifications/initialized`, then `tools/list`, then one `tools/call` for each
entry in CALLS.
"""

from __future__ import annotations

import json
import subprocess
import sys

# The newest revision the initialize handshake reaches. LATEST_PROTOCOL_VERSION
# is newer, but it names the stateless per-request era, which this handshake
# cannot negotiate.
from mcp_types.version import LATEST_HANDSHAKE_VERSION

# nist-800-171-r3 is the one corpus whose processed documents are committed, so
# a reader who clones the repository gets the same validation answer.
CALLS: list[tuple[str, dict]] = [
    ("list_corpora", {}),
    ("validate_corpus", {"corpus": "nist-800-171-r3"}),
    ("compare", {"corpus": "aws-compute", "a": "bm25", "b": "naive_vector"}),
]

# compare pairs 75 questions and returns a per-question list far longer than one
# screen. These are the fields that decide whether the pairing may be read.
COMPARE_FIELDS = (
    "a",
    "b",
    "metric",
    "n_paired",
    "mean_a",
    "mean_b",
    "mean_delta",
    "delta_ci_95",
    "wilcoxon_p",
    "significant",
)


def send(proc: subprocess.Popen, message: dict) -> None:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def read_reply(proc: subprocess.Popen, request_id: int) -> dict:
    """The reply to one request id.

    A line that does not parse as JSON is skipped rather than raised on: a
    dependency that prints at import time lands on this stream too, and it is
    not a protocol message.
    """
    for line in proc.stdout:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("id") == request_id:
            return message
    raise RuntimeError(f"the server closed stdout before it answered request {request_id}")


def result(reply: dict) -> dict:
    """The JSON-RPC result, or a raised error carrying the server's own text."""
    if "error" in reply:
        raise RuntimeError(reply["error"].get("message", "the server returned an error"))
    return reply["result"]


def payload(reply: dict) -> dict:
    """The tool payload inside a `tools/call` result.

    A tool typed to return a concrete model carries `structuredContent`. One
    typed `-> dict` does not, so this falls back to the text block that every
    tool call carries either way.
    """
    call = result(reply)
    if call.get("isError"):
        raise RuntimeError(call["content"][0]["text"])
    if call.get("structuredContent") is not None:
        return call["structuredContent"]
    return json.loads(call["content"][0]["text"])


def number(value: object) -> str:
    """A float at four places, so a screen of results stays readable."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return "[" + ", ".join(number(item) for item in value) + "]"
    return str(value)


def show(name: str, tool_result: dict) -> None:
    """One tool payload, sized for a terminal recording."""
    if name == "list_corpora":
        for entry in tool_result["corpora"]:
            print(
                f"   {entry['name']:<22} processed={entry['has_processed']!s:<6} "
                f"questions={entry['question_count']:<4} results={entry['has_results']}"
            )
        return
    if name == "compare":
        for field in COMPARE_FIELDS:
            print(f"   {field:<13} {number(tool_result[field])}")
        meta = tool_result["meta"]
        print(f"   {'comparable':<13} {meta['comparable']}")
        for reason in meta["reasons"]:
            print(f"   {'why not':<13} {reason}")
        return
    print(json.dumps(tool_result, indent=2))


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "kb_arena.mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # The server logs to stderr by design. A pipe nobody drains fills and
        # deadlocks the run, so this goes to the terminal instead.
        text=True,
        bufsize=1,
    )
    try:
        print("-> initialize")
        send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": LATEST_HANDSHAKE_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "kb-arena-demo", "version": "1"},
                },
            },
        )
        info = result(read_reply(proc, 1))
        server_name = info["serverInfo"]["name"]
        print(f"<- {server_name}, protocol {info['protocolVersion']}\n")
        send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        print("-> tools/list")
        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = result(read_reply(proc, 2))["tools"]
        print(f"<- {len(tools)} tools: {', '.join(sorted(tool['name'] for tool in tools))}\n")

        for index, (name, arguments) in enumerate(CALLS, start=3):
            print(f"-> tools/call {name} {json.dumps(arguments)}")
            send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": index,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            print("<-")
            show(name, payload(read_reply(proc, index)))
            print()
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
