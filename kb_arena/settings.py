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

    # Embeddings — text-embedding-3-large is the latest/best OpenAI embedding model
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Benchmark
    benchmark_temperature: float = 0.0
    benchmark_max_concurrent: int = 5
    benchmark_query_timeout_s: int = 120
    benchmark_max_retries: int = 2

    # PageIndex
    pageindex_beam_width: int = 3
    pageindex_max_depth: int = 4

    # Paths
    datasets_path: str = "./datasets"
    results_path: str = "./results"


settings = Settings()
