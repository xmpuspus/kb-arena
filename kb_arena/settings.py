"""Application settings via pydantic-settings. All config from environment."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "KB_ARENA_", "env_file": ".env", "extra": "ignore"}

    # LLM
    anthropic_api_key: str = ""
    generate_model: str = "claude-sonnet-4-6"
    fast_model: str = "claude-haiku-4-5-20251001"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "kbarena"

    # ChromaDB
    chroma_path: str = "./chroma_data"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Benchmark
    benchmark_temperature: float = 0.0
    benchmark_max_concurrent: int = 5

    # Paths
    datasets_path: str = "./datasets"
    results_path: str = "./results"


settings = Settings()
