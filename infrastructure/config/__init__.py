"""Config package"""
from .config_manager import (
    ConfigChangeEvent,
    ConfigEntry,
    ConfigImmutableError,
    ConfigManager,
    ConfigSchema as ConfigNamespace,
    ConfigType,
    get_config_manager,
)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""


__all__ = [
    "ConfigManager",
    "ConfigEntry",
    "ConfigType",
    "ConfigNamespace",
    "ConfigChangeEvent",
    "ConfigValidationError",
    "ConfigImmutableError",
    "get_config_manager",
]
