"""
Unified configuration loader for CognitionOS.

Loads configuration from environment with env-based overrides.
Merges core.config and shared.libs.config patterns.
"""

from typing import Any, Optional, Type, TypeVar, Union

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from core.config.main import CognitionOSConfig, get_config as _get_config

T = TypeVar("T", bound=Union[BaseSettings, BaseModel])


def load_config(
    config_class: Optional[Type[T]] = None,
    env_file: Optional[str] = ".env",
    env_file_encoding: str = "utf-8",
    **kwargs: Any,
) -> Union[CognitionOSConfig, T]:
    """
    Load configuration from environment.

    Args:
        config_class: Configuration class to instantiate. If None, returns
            CognitionOSConfig.
        env_file: Path to .env file.
        env_file_encoding: Encoding for env file.
        **kwargs: Additional kwargs passed to config class constructor.

    Returns:
        Configuration instance.
    """
    if config_class is None:
        return _get_config()
    try:
        if issubclass(config_class, BaseSettings):
            return config_class(_env_file=env_file, _env_file_encoding=env_file_encoding, **kwargs)
        return config_class(**kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}") from e


def get_config() -> CognitionOSConfig:
    """Get or create main CognitionOS configuration singleton."""
    return _get_config()


def reload_config() -> CognitionOSConfig:
    """Reload main configuration (useful for testing)."""
    from core.config.main import reload_config as _reload

    return _reload()


def get_unified_config() -> CognitionOSConfig:
    """Alias for get_config - returns unified CognitionOS config."""
    return _get_config()
