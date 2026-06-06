"""Configuration for Fractal Graph MCP server."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Storage paths
    db_path: Path = Path(__file__).parent / "data" / "fractal.db"
    chroma_path: Path = Path(__file__).parent / "data" / "chroma"

    # Ollama embedding service (Lappy via Tailscale)
    ollama_url: str = "http://100.84.161.63:11434"
    embed_model: str = "nomic-embed-text"

    # SearXNG search (Lappy)
    searxng_url: str = "http://100.84.161.63:8888"

    # LLM for classification (optional, Ollama on Lappy)
    llm_url: str = "http://100.84.161.63:11434"
    llm_model: str = "qwen3.5:2b"

    # Mother model for intelligent seeding (larger, slower, smarter)
    mother_url: str = "http://100.84.161.63:11434"
    mother_model: str = "granite4.1:8b"

    # Server
    server_port: int = 8018

    # Token budgets for 2B reasoning pipeline
    max_context_tokens: int = 2048
    max_answer_tokens: int = 768

    # Autonomous expansion
    auto_expand_threshold: float = 0.4  # confidence below this triggers expansion
    max_gap_fill_nodes: int = 5          # max nodes per gap-fill round
    max_enrich_nodes: int = 4            # max nodes per enrichment round
    curiosity_max_expansions: int = 3     # max expansions per curiosity scan
    max_expansion_rounds: int = 2         # max re-answer rounds (prevents infinite loop)

    model_config = {"env_prefix": "FRACTAL_"}


settings = Settings()
