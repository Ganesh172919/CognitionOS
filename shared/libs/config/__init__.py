"""
Configuration management for CognitionOS services.

Loads configuration from environment variables with validation.
"""

import os
from typing import Any, Dict, Optional
from pydantic import BaseSettings, Field, validator


class BaseConfig(BaseSettings):
    """Base configuration with common settings."""

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Service
    service_name: str
    service_version: str = "0.1.0"
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")

    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_pool_size: int = Field(default=10, env="REDIS_POOL_SIZE")

    # Message Queue
    message_queue_url: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        env="MESSAGE_QUEUE_URL"
    )

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text

    # Tracing
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    trace_sample_rate: float = Field(default=0.1, env="TRACE_SAMPLE_RATE")

    # Security
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    secret_key: str = Field(..., env="SECRET_KEY")

    @validator("cors_origins")
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",")]

    @validator("log_level")
    def validate_log_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class APIGatewayConfig(BaseConfig):
    """Configuration for API Gateway service."""

    service_name: str = "api-gateway"

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=10, env="RATE_LIMIT_BURST")

    # Timeouts
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    connection_timeout: int = Field(default=10, env="CONNECTION_TIMEOUT")

    # Circuit breaker
    circuit_breaker_enabled: bool = Field(default=True, env="CIRCUIT_BREAKER_ENABLED")
    circuit_breaker_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_THRESHOLD")
    circuit_breaker_timeout: int = Field(default=60, env="CIRCUIT_BREAKER_TIMEOUT")

    # Service URLs
    auth_service_url: str = Field(..., env="AUTH_SERVICE_URL")
    task_service_url: str = Field(..., env="TASK_SERVICE_URL")


class AuthServiceConfig(BaseConfig):
    """Configuration for Auth Service."""

    service_name: str = "auth-service"

    # JWT
    jwt_secret: str = Field(..., env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=15,
        env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7,
        env="JWT_REFRESH_TOKEN_EXPIRE_DAYS"
    )

    # Password hashing
    bcrypt_rounds: int = Field(default=12, env="BCRYPT_ROUNDS")

    # Session
    session_ttl_hours: int = Field(default=24, env="SESSION_TTL_HOURS")


class AgentOrchestratorConfig(BaseConfig):
    """Configuration for Agent Orchestrator."""

    service_name: str = "agent-orchestrator"

    # Agent pool
    max_agents: int = Field(default=100, env="MAX_AGENTS")
    agent_timeout_seconds: int = Field(default=300, env="AGENT_TIMEOUT_SECONDS")
    agent_idle_timeout_seconds: int = Field(
        default=600,
        env="AGENT_IDLE_TIMEOUT_SECONDS"
    )

    # Task queue
    task_queue_name: str = Field(default="tasks", env="TASK_QUEUE_NAME")
    task_prefetch_count: int = Field(default=1, env="TASK_PREFETCH_COUNT")

    # Retry policy
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_backoff_base: int = Field(default=2, env="RETRY_BACKOFF_BASE")


class AIRuntimeConfig(BaseConfig):
    """Configuration for AI Runtime."""

    service_name: str = "ai-runtime"

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")

    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")

    # Model defaults
    default_planner_model: str = Field(
        default="gpt-4",
        env="DEFAULT_PLANNER_MODEL"
    )
    default_executor_model: str = Field(
        default="gpt-3.5-turbo",
        env="DEFAULT_EXECUTOR_MODEL"
    )
    default_critic_model: str = Field(
        default="gpt-4",
        env="DEFAULT_CRITIC_MODEL"
    )

    # Budget
    default_max_tokens: int = Field(default=2000, env="DEFAULT_MAX_TOKENS")
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")

    # Caching
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")


class MemoryServiceConfig(BaseConfig):
    """Configuration for Memory Service."""

    service_name: str = "memory-service"

    # Vector database
    vector_db_url: Optional[str] = Field(default=None, env="VECTOR_DB_URL")
    vector_dimension: int = Field(default=1536, env="VECTOR_DIMENSION")  # text-embedding-ada-002
    vector_index_type: str = Field(default="ivfflat", env="VECTOR_INDEX_TYPE")

    # Embeddings
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        env="EMBEDDING_MODEL"
    )
    embedding_batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")

    # Memory limits
    max_memories_per_user: int = Field(default=10000, env="MAX_MEMORIES_PER_USER")
    memory_retention_days: int = Field(default=365, env="MEMORY_RETENTION_DAYS")

    # Retrieval
    default_retrieval_k: int = Field(default=5, env="DEFAULT_RETRIEVAL_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")


class ToolRunnerConfig(BaseConfig):
    """Configuration for Tool Runner."""

    service_name: str = "tool-runner"

    # Execution
    execution_mode: str = Field(default="docker", env="EXECUTION_MODE")  # docker or process
    max_concurrent_executions: int = Field(
        default=10,
        env="MAX_CONCURRENT_EXECUTIONS"
    )

    # Sandboxing
    sandbox_enabled: bool = Field(default=True, env="SANDBOX_ENABLED")
    sandbox_network_enabled: bool = Field(default=False, env="SANDBOX_NETWORK_ENABLED")
    sandbox_memory_limit: str = Field(default="512m", env="SANDBOX_MEMORY_LIMIT")
    sandbox_cpu_limit: float = Field(default=1.0, env="SANDBOX_CPU_LIMIT")

    # Timeouts
    default_tool_timeout: int = Field(default=30, env="DEFAULT_TOOL_TIMEOUT")
    max_tool_timeout: int = Field(default=300, env="MAX_TOOL_TIMEOUT")

    # Docker
    docker_base_image: str = Field(
        default="python:3.11-slim",
        env="DOCKER_BASE_IMAGE"
    )


def load_config(config_class: type = BaseConfig) -> BaseConfig:
    """
    Load configuration from environment.

    Args:
        config_class: Configuration class to instantiate

    Returns:
        Configuration instance

    Raises:
        ValueError: If required environment variables are missing
    """
    try:
        return config_class()
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")


def get_config_dict(config: BaseConfig) -> Dict[str, Any]:
    """
    Convert config to dictionary.

    Args:
        config: Configuration instance

    Returns:
        Dictionary representation
    """
    return config.dict()
