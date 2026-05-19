"""Application settings via pydantic-settings. All config from environment."""

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

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""  # set KB_ARENA_NEO4J_PASSWORD or NEO4J_AUTH in docker-compose

    # ChromaDB
    chroma_path: str = "./chroma_data"

    # Embeddings — provider-agnostic. Pick via KB_ARENA_EMBEDDING_PROVIDER:
    # openai (default), voyage, cohere, bge (local), ollama (local), gemini.
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    ollama_embedding_model: str = "nomic-embed-text"
    voyage_api_key: str = ""
    cohere_api_key: str = ""
    gemini_api_key: str = ""

    # Reranker — used by the rerank_vector strategy (#9). Backends: bge | cohere | voyage.
    reranker_backend: str = "bge"
    reranker_model: str = ""  # blank = backend default

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list[str] = []  # Override via KB_ARENA_CORS_ORIGINS='["http://myapp:3000"]'
    session_ttl_minutes: int = 30

    # API auth — when set, requests must include `Authorization: Bearer <token>`.
    # When unset, the API runs in open mode (only safe for localhost dev).
    api_token: str = ""
    # Demo mode: when true, /chat, /chat/stream, /api/arena/*, /api/tools/*,
    # /api/graph/build, /api/debug/explain return 503. Used by the hosted public demo.
    demo_mode: bool = False
    # Trusted reverse-proxy header for client IP rate limiting (e.g. "x-forwarded-for").
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

    # Paths
    datasets_path: str = "./datasets"
    results_path: str = "./results"


settings = Settings()
