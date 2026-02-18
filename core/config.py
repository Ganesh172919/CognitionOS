"""
CognitionOS V3 - Centralized Configuration Management

Provides configuration management for all services and components.
Uses Pydantic v2 for validation and settings management.
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    model_config = SettingsConfigDict(env_prefix='DB_', case_sensitive=False)
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="cognitionos", description="Database name")
    username: str = Field(default="cognition", description="Database username")
    password: str = Field(default="changeme", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max connections overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseSettings):
    """Redis configuration"""
    model_config = SettingsConfigDict(env_prefix='REDIS_', case_sensitive=False)
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    max_connections: int = Field(default=50, description="Max connections in pool")
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class RabbitMQConfig(BaseSettings):
    """RabbitMQ configuration"""
    model_config = SettingsConfigDict(env_prefix='RABBITMQ_', case_sensitive=False)
    
    host: str = Field(default="localhost", description="RabbitMQ host")
    port: int = Field(default=5672, description="RabbitMQ port")
    username: str = Field(default="guest", description="RabbitMQ username")
    password: str = Field(default="guest", description="RabbitMQ password")
    virtual_host: str = Field(default="/", description="Virtual host")
    
    @property
    def url(self) -> str:
        """Get RabbitMQ URL"""
        return f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/{self.virtual_host}"


class LLMConfig(BaseSettings):
    """LLM provider configuration"""
    model_config = SettingsConfigDict(env_prefix='LLM_', case_sensitive=False)
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    default_provider: str = Field(default="openai", description="Default LLM provider")
    default_model: str = Field(default="gpt-4", description="Default model name")
    max_retries: int = Field(default=3, description="Max retry attempts")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_tokens: int = Field(default=4096, description="Max tokens per request")


class CeleryConfig(BaseSettings):
    """Celery task queue configuration"""
    model_config = SettingsConfigDict(env_prefix='CELERY_', case_sensitive=False)
    
    broker_url: str = Field(default="redis://localhost:6379/0", description="Message broker URL")
    result_backend: str = Field(default="redis://localhost:6379/0", description="Result backend URL")
    task_serializer: str = Field(default="json", description="Task serializer")
    result_serializer: str = Field(default="json", description="Result serializer")
    accept_content: List[str] = Field(default=["json"], description="Accepted content types")
    timezone: str = Field(default="UTC", description="Timezone")
    enable_utc: bool = Field(default=True, description="Enable UTC")
    task_track_started: bool = Field(default=True, description="Track task started")
    task_time_limit: int = Field(default=3600, description="Task time limit in seconds")
    task_soft_time_limit: int = Field(default=3300, description="Task soft time limit in seconds")


class APIConfig(BaseSettings):
    """API service configuration"""
    model_config = SettingsConfigDict(env_prefix='API_', case_sensitive=False)
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8100, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    workers: int = Field(default=4, description="Number of workers")
    log_level: str = Field(default="info", description="Logging level")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    max_request_size: int = Field(default=10485760, description="Max request size in bytes (10MB)")
    shutdown_timeout_seconds: int = Field(default=30, description="Graceful shutdown timeout in seconds")


class SecurityConfig(BaseSettings):
    """Security configuration"""
    model_config = SettingsConfigDict(env_prefix='SECURITY_', case_sensitive=False)
    
    secret_key: str = Field(default="change-me-in-production", description="Secret key for JWT")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration")
    
    # API key settings
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    
    # Encryption
    encryption_algorithm: str = Field(default="AES-256-GCM", description="Encryption algorithm")


class ObservabilityConfig(BaseSettings):
    """Observability configuration"""
    model_config = SettingsConfigDict(env_prefix='OBSERVABILITY_', case_sensitive=False)
    
    # Logging
    log_level: str = Field(default="info", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    
    # Tracing
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    jaeger_host: str = Field(default="localhost", description="Jaeger host")
    jaeger_port: int = Field(default=6831, description="Jaeger port")
    
    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics port")


class CognitionOSConfig(BaseSettings):
    """Main CognitionOS configuration"""
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False)
    
    # Service metadata
    service_name: str = Field(default="cognitionos-api", description="Service name")
    service_version: str = Field(default="3.0.0", description="Service version")
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    rabbitmq: RabbitMQConfig = Field(default_factory=RabbitMQConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"


# Singleton configuration instance
_config: Optional[CognitionOSConfig] = None


def get_config() -> CognitionOSConfig:
    """Get or create configuration singleton"""
    global _config
    if _config is None:
        _config = CognitionOSConfig()
    return _config


def reload_config() -> CognitionOSConfig:
    """Reload configuration (useful for testing)"""
    global _config
    _config = CognitionOSConfig()
    return _config
