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
    mother_model: str = "qwen3.5:4b"

    # Server
    server_port: int = 8018

    model_config = {"env_prefix": "FRACTAL_"}


settings = Settings()
