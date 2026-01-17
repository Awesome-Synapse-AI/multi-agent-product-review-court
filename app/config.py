from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for the Product Review Court backend."""

    ollama_model: str = Field(default="qwen3:0.6b", env="OLLAMA_MODEL")
    ollama_temperature: float = Field(default=0.35, env="OLLAMA_TEMPERATURE")
    ollama_num_ctx: int = Field(default=4096, env="OLLAMA_NUM_CTX")
    ollama_base_url: str = Field(
        default="http://ollama:11434", env="OLLAMA_BASE_URL"
    )
    judge_temperature: Optional[float] = Field(default=None, env="JUDGE_TEMPERATURE")
    judge_max_output_tokens: Optional[int] = Field(
        default=None, env="JUDGE_MAX_OUTPUT_TOKENS"
    )
    langsmith_tracing: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langsmith_project: str = Field(
        default="product-review-court", env="LANGCHAIN_PROJECT"
    )
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")

    # Google Custom Search Engine (PSE)
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    google_cse_id: str = Field(..., env="GOOGLE_CSE_ID")

    # Persistence
    database_url: str = Field(
        default="postgresql://localhost:5432/reviews", env="DATABASE_URL"
    )
    memory_path: str = Field(
        default=os.path.join("data", "global_memory.json"), env="MEMORY_PATH"
    )
    chat_memory_path: str = Field(
        default=os.path.join("data", "chat_memory.sqlite"), env="CHAT_MEMORY_PATH"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
