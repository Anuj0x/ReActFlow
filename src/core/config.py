"""Configuration management using modern Pydantic settings."""

from typing import Optional
import asyncio
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class AgentConfig(BaseSettings):
    """Modern configuration management for ReAct agents."""

    # OpenAI settings
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=1000, gt=0)
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")

    # Agent behavior
    max_interactions: int = Field(default=10, gt=0)
    token_limit: int = Field(default=5000, gt=0)
    parallel_tool_calls: bool = Field(default=True)
    tool_choice: Optional[str] = Field(default=None)

    # Execution settings
    max_concurrent_requests: int = Field(default=5, gt=0)
    request_timeout: float = Field(default=30.0, gt=0)

    # Caching (optional)
    enable_caching: bool = Field(default=False)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_structured_logging: bool = Field(default=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @computed_field
    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Async semaphore for controlling concurrent requests."""
        return asyncio.Semaphore(self.max_concurrent_requests)

    def get_openai_client_kwargs(self) -> dict:
        """Get OpenAI client initialization kwargs."""
        kwargs = {
            "api_key": self.openai_api_key,
        }
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
        return kwargs
