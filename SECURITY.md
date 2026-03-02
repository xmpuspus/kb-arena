# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email security concerns to xavier@xmpuspus.dev
3. Include a description of the vulnerability, steps to reproduce, and potential impact
4. You'll receive a response within 48 hours

## Security Model

### API Keys

- All API keys are loaded from environment variables via `pydantic-settings`
- Keys are never logged, serialized to disk, or included in error messages
- The `KB_ARENA_DEBUG=false` default ensures production error responses are generic

### Network

- The chatbot API includes per-IP rate limiting (60 requests/minute)
- CORS is configured with explicit allowed origins — never `*` in production
- Neo4j is on an internal Docker network, not exposed to the frontend network

### Database

- All Neo4j queries use parameterized Cypher — no string interpolation
- The chatbot API uses read-only query patterns (`MATCH`/`RETURN`)
- Graph mutations only occur during the `build-graph` CLI command

### Dependencies

All 14 direct dependencies are pinned to exact versions. We do not use `>=`, `^`, or `~` version specifiers.

### Input Validation

- All API request bodies are validated by Pydantic v2 with strict type checking
- Strategy names are validated against the registry — unknown strategies return a structured error
- Question YAML files are validated against the `Question` Pydantic model at load time

## Known Limitations

- The in-memory rate limiter resets on process restart. For production deployments behind a load balancer, use an external rate limiter (e.g., Redis-based)
- The chatbot API binds to `0.0.0.0` by default. In production, bind to `127.0.0.1` and use a reverse proxy
- LLM responses are not sanitized for XSS before rendering in the frontend. The Next.js frontend uses React's built-in escaping, but custom integrations should sanitize output
