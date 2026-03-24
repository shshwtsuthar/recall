"""
Recall configuration — single source of truth for all settings.

Reads from environment variables (prefixed RECALL_) or .env file.
All components import from here instead of hardcoding values.

Usage:
    from recall.config.settings import get_settings
    settings = get_settings()
    print(settings.qdrant_url)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # -- Qdrant --
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "trajectories"
    qdrant_api_key: str | None = None

    # -- Embeddings --
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # -- Paths --
    data_dir: str = "data"
    adp_dir: str = "data/adp"
    raw_dir: str = "data/raw"

    # -- OpenHands --
    openhands_conversations_dir: str = "~/.openhands/conversations"

    model_config = {
        "env_prefix": "RECALL_",
        "env_file": ".env",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
