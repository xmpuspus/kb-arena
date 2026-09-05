"""MCP (Model Context Protocol) server for KB Arena.

Exposes corpus, strategy, benchmark, and evidence tools over stdio so an
MCP client (Claude Code, Codex, Cursor, or a generic client) can drive a
retrieval comparison without shelling out to the CLI.

This package needs the ``mcp`` optional extra: ``pip install 'kb-arena[mcp]'``.
Nothing in the core install imports this package, so a plain ``kb-arena``
install never needs the MCP dependency.
"""
