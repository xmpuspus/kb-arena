# The MCP server answering a real client over stdio. Regenerate from the
# repository root with:
#   python3 -m venv /tmp/mcp-demo && /tmp/mcp-demo/bin/pip install -e '.[mcp]'
#   vhs docs/tapes/mcp.tape
# Output: docs/demo-mcp.gif
#
# scripts/mcp_stdio_demo.py starts `python3 -m kb_arena.mcp.server` as a
# subprocess and speaks JSON-RPC on its stdin and stdout, so every answer in
# this recording comes from the running server, not from a printed transcript.
#
# The tape carries no `cd` and no worktree path, because a hard-coded path goes
# stale the moment that worktree is removed. The venv falls back to `.venv`,
# for a checkout that already has the `mcp` extra installed there.

Output docs/demo-mcp.gif

Set FontSize 14
Set Width 1280
Set Height 800
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 24ms
Set Shell zsh

Hide
Type "source /tmp/mcp-demo/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null; setopt interactive_comments; clear"
Enter
Sleep 1000ms
Show

Type "# An editor or an agent calls kb-arena over MCP, with no shell in between."
Enter
Sleep 1500ms

Type "python3 scripts/mcp_stdio_demo.py"
Enter
Sleep 12000ms

Type "# The server names its eight tools, then answers three of them."
Enter
Sleep 2000ms

Type "# The last answer pairs 75 questions and says the pairing is not comparable, and why."
Enter
Sleep 4000ms
