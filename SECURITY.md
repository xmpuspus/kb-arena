# Security Policy

## Supported versions

| Version | Support |
|---|---|
| 0.9.x | Active fixes |
| 0.8.x | Critical fixes on a best-effort basis |
| 0.7.x and earlier | Unsupported |

Fixes land on the latest release in the active minor line.

## Report a vulnerability

Do not open a public issue for a suspected vulnerability. Email `xavier@xmpuspus.dev` with:

- the affected version;
- a clear description and impact;
- steps or a minimal example that reproduce the issue;
- any known mitigation.

The project aims to acknowledge a report within 48 hours. That target is not a service-level
agreement.

## Security boundaries

### API access

- LLM-triggering endpoints use `Depends(require_auth)`.
- When you set `KB_ARENA_API_TOKEN`, clients send `Authorization: Bearer <token>`.
- Token comparison uses `hmac.compare_digest`.
- `KB_ARENA_DEMO_MODE=true` makes every LLM-triggering endpoint return 503.
- Read-only result, corpus, health, readiness, and Retriever Lab endpoints do not need a token.

The default open API mode is intended for local development. Set a token and a narrow CORS list
before you expose an instance to a network.

### Input and query handling

- Pydantic validates request bodies and caps questions at 4,000 characters.
- Corpus and strategy names use allow-listed patterns or the strategy registry.
- YAML readers use `yaml.safe_load`.
- The code does not use `pickle`, `eval`, or `exec` on user input.
- LLM-generated Cypher passes a write-operation block list.
- Neo4j read paths use `neo4j.READ_ACCESS` at the driver level.
- Production graph extraction uses parameterized Cypher.

### URL ingestion

- URL ingestion accepts HTTP and HTTPS only.
- The URL validator checks DNS results against private, loopback, link-local, multicast, and reserved ranges.
- The URL validator blocks known cloud metadata hosts by name.
- The parser checks redirects one hop at a time, with a limit of five.
- GitHub ingestion uses a shallow, single-branch clone with a 120-second timeout.

### Spend and availability

- `KB_ARENA_BENCHMARK_COST_CAP_USD` defaults to `10.0`.
- Benchmark concurrency, query timeouts, and retries use fixed defaults and environment controls.
- Demo mode prevents a no-key instance from making LLM calls.
- The in-memory rate limiter allows 60 requests per minute for each client and caps cold keys.

### Network and container defaults

- `KB_ARENA_CORS_ORIGINS` controls allowed browser origins and does not default to `*`.
- `docker-compose.yml` binds Neo4j to `127.0.0.1` and needs an explicit password.
- The container runs as the non-root `kbarena` user.
- The container health check polls `/health`.

### Dependencies

`pyproject.toml` pins most direct dependencies and sets a minimum SciPy version. The repository does
not currently publish a tracked transitive lock file. Review resolved dependencies during release
validation and use an isolated environment for untrusted corpora.

## Known limits

- The in-memory rate limiter resets when the process restarts and does not coordinate across workers.
- A reverse proxy must remove untrusted forwarding headers before you set
  `KB_ARENA_TRUSTED_PROXY_HEADER`.
- React escapes normal text output, but custom integrations must sanitize content before rendering
  it as HTML.
- `/api/debug/explain` is available only when `KB_ARENA_DEBUG=true`; do not enable it on a public
  deployment.
- Document ingestion processes untrusted text. Run parsers with the minimum file and network access
  needed for the corpus.
