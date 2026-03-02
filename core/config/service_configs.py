"""
Service-specific configurations for CognitionOS.

Unified Pydantic v2 configs for tool-runner, api-gateway, auth, etc.
Merges shared.libs.config patterns with core config.
"""

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ToolRunnerConfig(BaseSettings):
    """Configuration for Tool Runner service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "tool-runner"
    service_version: str = "0.1.0"
    port: int = Field(default=8006, description="Service port")
    host: str = Field(default="0.0.0.0", description="Service host")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")

    database_url: str = Field(default="postgresql://cognition:changeme@localhost:5432/cognitionos")
    redis_url: str = Field(default="redis://localhost:6379/0")
    log_level: str = Field(default="INFO", description="Log level")

    sandbox_enabled: bool = Field(default=True, description="Enable sandbox")
    sandbox_network_enabled: bool = Field(default=False, description="Allow network in sandbox")
    sandbox_memory_limit: str = Field(default="512m", description="Sandbox memory limit")
    sandbox_cpu_limit: float = Field(default=1.0, description="Sandbox CPU limit")
    default_tool_timeout: int = Field(default=30, description="Default tool timeout seconds")
    max_concurrent_executions: int = Field(default=10, description="Max concurrent executions")


class APIGatewayConfig(BaseSettings):
    """Configuration for API Gateway service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "api-gateway"
    service_version: str = "0.1.0"
    port: int = Field(default=8000)
    host: str = Field(default="0.0.0.0")
    database_url: str = Field(default="postgresql://cognition:changeme@localhost:5432/cognitionos")
    redis_url: str = Field(default="redis://localhost:6379/0")
    log_level: str = Field(default="INFO")
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")

    rate_limit_enabled: bool = Field(default=True)
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_burst: int = Field(default=10)
    auth_service_url: str = Field(default="http://localhost:8001")
    task_service_url: str = Field(default="http://localhost:8002")


class AuthServiceConfig(BaseSettings):
    """Configuration for Auth service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "auth-service"
    service_version: str = "0.1.0"
    port: int = Field(default=8001)
    host: str = Field(default="0.0.0.0")
    database_url: str = Field(default="postgresql://cognition:changeme@localhost:5432/cognitionos")
    redis_url: str = Field(default="redis://localhost:6379/0")
    log_level: str = Field(default="INFO")
    rate_limit_per_minute: int = Field(default=60)

    secret_key: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=15)
    jwt_refresh_token_expire_days: int = Field(default=7)
    bcrypt_rounds: int = Field(default=12)


class AgentOrchestratorConfig(BaseSettings):
    """Configuration for Agent Orchestrator service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "agent-orchestrator"
    service_version: str = "0.1.0"
    port: int = Field(default=8003)
    database_url: str = Field(default="postgresql://cognition:changeme@localhost:5432/cognitionos")
    redis_url: str = Field(default="redis://localhost:6379/0")
    log_level: str = Field(default="INFO")

    max_agents: int = Field(default=100)
    agent_timeout_seconds: int = Field(default=300)
    max_retries: int = Field(default=3)


class AIRuntimeConfig(BaseSettings):
    """Configuration for AI Runtime service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "ai-runtime"
    service_version: str = "0.1.0"
    port: int = Field(default=8005)
    database_url: str = Field(default="postgresql://cognition:changeme@localhost:5432/cognitionos")
    redis_url: str = Field(default="redis://localhost:6379/0")
    log_level: str = Field(default="INFO")

    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    default_planner_model: str = Field(default="gpt-4")
    default_executor_model: str = Field(default="gpt-3.5-turbo")
    default_critic_model: str = Field(default="gpt-4")
    default_max_tokens: int = Field(default=2000)
    cache_enabled: bool = Field(default=True)


class MemoryServiceConfig(BaseSettings):
    """Configuration for Memory service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "memory-service"
    service_version: str = "0.1.0"
    port: int = Field(default=8004)
    database_url: str = Field(default="postgresql://cognition:changeme@localhost:5432/cognitionos")
    redis_url: str = Field(default="redis://localhost:6379/0")
    log_level: str = Field(default="INFO")

    vector_dimension: int = Field(default=1536)
    embedding_model: str = Field(default="text-embedding-ada-002")
    max_memories_per_user: int = Field(default=10000)
    default_retrieval_k: int = Field(default=5)


class BaseConfig(BaseSettings):
    """Base configuration for services using flat env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "cognitionos-service"
    service_version: str = "0.1.0"
    port: int = Field(default=8000)
    host: str = Field(default="0.0.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    database_url: str = Field(default="postgresql://cognition:changeme@localhost:5432/cognitionos")
    redis_url: str = Field(default="redis://localhost:6379/0")
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
