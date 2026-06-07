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
    mother_model: str = "lfm2.5:gpu3"

    # Server
    server_port: int = 8018

    # Token budgets for 2B reasoning pipeline
    max_context_tokens: int = 6144
    max_answer_tokens: int = 1024

    # Autonomous expansion
    auto_expand_threshold: float = 0.4  # confidence below this triggers expansion
    max_gap_fill_nodes: int = 5          # max nodes per gap-fill round
    max_enrich_nodes: int = 4            # max nodes per enrichment round
    curiosity_max_expansions: int = 3     # max expansions per curiosity scan
    max_expansion_rounds: int = 2         # max re-answer rounds (prevents infinite loop)

    # Search trigger layer
    search_trigger_enabled: bool = True
    search_max_urls: int = 3

    # z.ai GLM5.1 API (cloud mother/grandmother model)
    # WARNING: GLM Coding Plan is restricted to supported tools only (Claude Code, Cline, etc).
    # Using via MCP server / direct API may trigger account restrictions per ToS.
    zai_api_key: str = "b7143b5694e443eaa6858550fd2bcf2e.Jl2Q0bsQWWKrBdbd"
    zai_base_url: str = "https://api.z.ai/api/coding/paas/v4"
    zai_model: str = "GLM-5.1"

    # Grandmother model — cloud escalation when mother fails
    grandmother_model: str = "glm-5.1"
    grandmother_enabled: bool = True  # auto-escalation toggle (on, z.ai default)
    grandmother_max_retries_before_escalate: int = 2  # retry mother N times first

    # Monte Carlo decision enhancement
    mc_simulations: int = 5         # simulations per decide_mc call
    mc_context_pool: int = 10       # candidate nodes per level to draw from
    mc_context_subset: int = 3       # nodes per simulation (sampled from pool)

    # Behavior distillation
    behavior_probes: int = 10
    behavior_simulations_per: int = 3

    # Meeseeks system — task-scoped ephemeral souls
    meeseeks_max_concurrent: int = 5
    meeseeks_max_steps: int = 20
    meeseeks_suffering_threshold: float = 0.3
    meeseeks_suffering_window: int = 3

    model_config = {"env_prefix": "FRACTAL_"}


settings = Settings()
