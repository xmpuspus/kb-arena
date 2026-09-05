"""Application settings via pydantic-settings. All config from environment."""

import math

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "KB_ARENA_", "env_file": ".env", "extra": "ignore"}

    # LLM — Anthropic (latest models)
    anthropic_api_key: str = ""
    generate_model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5-20251001"
    # Use a different model family for evaluation to avoid self-evaluation bias
    judge_model: str = "claude-opus-4-6"

    # LLM provider selection
    llm_provider: str = "anthropic"  # anthropic | openai | ollama
    # The judge can run on a different provider than generation, so the
    # model that answers never grades itself. Empty follows llm_provider.
    judge_provider: str = ""  # "" | anthropic | openai | ollama
    llm_api_key: str = ""  # generic key, falls back to provider-specific

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI generation model names (when provider=openai)
    openai_generate_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"
    openai_judge_model: str = "gpt-4o"

    # Ollama model names (when provider=ollama)
    ollama_generate_model: str = "llama3.1:8b"
    ollama_fast_model: str = "llama3.1:8b"
    ollama_judge_model: str = "llama3.1:8b"

    # LLM — OpenAI (for embeddings)
    openai_api_key: str = ""

    # A named arena voter is a claim about who judged. Only a caller that
    # sends this key in x-kb-arena-reviewer-key can make one.
    arena_reviewer_key: str = ""
    # Each (corpus, rubric) pair makes a persistent rating table. A free-text
    # rubric would otherwise grow the state file and every leaderboard reply.
    arena_max_scopes: int = Field(default=50, ge=1)
    # Graph analysis budgets. Exact betweenness is O(n*m) and loads the whole
    # graph into one process. Above these the analyzer loads a bounded slice
    # and samples the centrality instead of hanging the API.
    graph_node_budget: int = Field(default=5000, ge=1)
    graph_edge_budget: int = Field(default=50000, ge=1)
    graph_centrality_exact_max_nodes: int = Field(default=1000, ge=1)
    graph_centrality_samples: int = Field(default=200, ge=1)

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""  # set KB_ARENA_NEO4J_PASSWORD or NEO4J_AUTH in docker-compose
    neo4j_database: str = "neo4j"

    # ChromaDB
    chroma_path: str = "./chroma_data"

    # Embeddings — provider-agnostic. Pick via KB_ARENA_EMBEDDING_PROVIDER:
    # openai (default), voyage, cohere, bge (local), ollama (local), gemini.
    embedding_provider: str = "openai"
    # One SQLite file in front of every embedding provider. Empty path means
    # <chroma_path>/embedding_cache.sqlite.
    embedding_cache_enabled: bool = True
    embedding_cache_path: str = ""
    # Part of every cache key. Change it when a model changed under the same
    # tag, so vectors from the old revision are never read again.
    embedding_cache_salt: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    ollama_embedding_model: str = "nomic-embed-text"
    voyage_api_key: str = ""
    cohere_api_key: str = ""
    gemini_api_key: str = ""

    # Reranker — used by the rerank_vector strategy (#9). Backends: bge | cohere | voyage.
    reranker_backend: str = "bge"
    reranker_model: str = ""  # blank = backend default

    # LLM call resilience. The retry predicate is provider-aware (see llm/client.py).
    llm_call_timeout_s: float = 60.0
    llm_max_attempts: int = 3
    # Streaming deadlines: time to the first token, then the longest silence between tokens.
    llm_stream_first_token_timeout_s: float = 30.0
    llm_stream_idle_timeout_s: float = 60.0

    # Logging. KB_ARENA_LOG_LEVEL applies to the CLI and the API process; --verbose wins.
    log_level: str = "INFO"
    log_format: str = "text"  # text | json

    # OpenTelemetry tracing (needs the `otel` extra). Off by default, so the
    # core install and every code path work with the extra absent. See
    # kb_arena/telemetry.py for what a span may and may not carry.
    otel_enabled: bool = False

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    cors_origins: list[str] = []  # Override via KB_ARENA_CORS_ORIGINS='["http://myapp:3000"]'
    session_ttl_minutes: int = 30
    # Serve /docs, /redoc, and /openapi.json. Unset follows `debug`: closed in
    # production, open under KB_ARENA_DEBUG=true. Set explicitly to override.
    api_docs_enabled: bool | None = None

    # API auth — when set, requests must include `Authorization: Bearer <token>`.
    # When unset, the API runs in open mode (only safe for localhost dev).
    api_token: str = ""
    # Demo mode: when true, /chat, /chat/stream, /api/arena/*, /api/tools/*,
    # /api/graph/build, /api/debug/explain return 503. Used by the hosted public demo.
    demo_mode: bool = False
    # True when the app turned demo mode on itself because no LLM key was
    # configured. That is not the same as an operator publishing a demo, and
    # only the operator's choice widens who may read corpus content.
    demo_mode_auto: bool = False
    # Reverse-proxy client header, honored only when the socket peer is loopback.
    trusted_proxy_header: str = ""

    # Benchmark
    benchmark_temperature: float = 0.0
    benchmark_max_concurrent: int = 5
    benchmark_query_timeout_s: int = 120
    benchmark_max_retries: int = 2
    # Default budget guard: 10 USD. Set to 0 to disable. Halts run when cumulative cost exceeds.
    benchmark_cost_cap_usd: float = 10.0
    benchmark_enable_ragas: bool = False  # enable RAGAS metrics (adds 4 LLM calls per question)

    # Chunking — consumed by every token-chunking strategy (naive_vector,
    # contextual_vector, raptor). Exposed as settings so `kb-arena optimize`
    # can sweep them per strategy.
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # PageIndex
    pageindex_beam_width: int = 3
    pageindex_max_depth: int = 4

    # Quantum strategies (#10 qiss, #11 sqr) — both rerank naive_vector candidates.
    # sqr needs the optional [quantum] extra (qiskit, qiskit-aer, scikit-learn).
    qiss_fanout: int = 4  # coarse-retrieve top_k * fanout before the fidelity rerank
    qiss_decompose: bool = False  # gate multi-query superposition fusion (LLM sub-query split)
    qiss_max_subqueries: int = 3
    sqr_fanout: int = 4
    sqr_n_qubits: int = 4  # amplitude-encode into 2^n_qubits dims (4 -> 16)
    sqr_shots: int = 0  # 0 = exact statevector (benchmark default); >0 = sampled SWAP test

    # Paths
    # The seed a run records and sets. Two runs that differ only by seed are
    # the same experiment, so the seed stays out of the compatibility key.
    # scipy's random_state rejects anything outside the 32-bit range, and it
    # does so after a sweep finishes, so the bound belongs here.
    run_seed: int = Field(default=0, ge=0, le=2**32 - 1)

    datasets_path: str = "./datasets"
    results_path: str = "./results"

    @field_validator("benchmark_cost_cap_usd")
    @classmethod
    def _validate_benchmark_cost_cap(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("benchmark cost cap must be a finite non-negative number")
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        return normalized

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"text", "json"}:
            raise ValueError("log format must be text or json")
        return normalized

    @field_validator("llm_max_attempts")
    @classmethod
    def _validate_llm_max_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("llm max attempts must be at least 1")
        return value

    @field_validator("reranker_backend")
    @classmethod
    def _validate_reranker_backend(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"bge", "cohere", "voyage"}:
            raise ValueError("reranker backend must be one of: bge, cohere, voyage")
        return normalized

    @model_validator(mode="after")
    def _validate_chunk_window(self) -> "Settings":
        if (
            self.chunk_tokens < 1
            or self.chunk_overlap_tokens < 0
            or self.chunk_overlap_tokens >= self.chunk_tokens
        ):
            raise ValueError("chunk overlap must satisfy 0 <= overlap_tokens < chunk_tokens")
        return self

    @field_validator("judge_provider")
    @classmethod
    def _judge_provider_known(cls, value: str) -> str:
        if value not in ("", "anthropic", "openai", "ollama"):
            raise ValueError(
                "KB_ARENA_JUDGE_PROVIDER must be empty, anthropic, openai, or ollama, "
                f"not {value!r}"
            )
        return value


settings = Settings()
