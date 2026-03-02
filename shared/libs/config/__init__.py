"""
Configuration management for CognitionOS services.

Re-exports from core.config for backward compatibility.
Unified config: core.config.loader, core.config.service_configs.
"""

from typing import Any, Dict

from core.config import (
    BaseConfig,
    ToolRunnerConfig,
    APIGatewayConfig,
    AuthServiceConfig,
    AgentOrchestratorConfig,
    AIRuntimeConfig,
    MemoryServiceConfig,
    load_config,
)


def get_config_dict(config: BaseConfig) -> Dict[str, Any]:
    """
    Convert config to dictionary.

    Args:
        config: Configuration instance

    Returns:
        Dictionary representation
    """
    return config.model_dump() if hasattr(config, "model_dump") else config.dict()


__all__ = [
    "BaseConfig",
    "ToolRunnerConfig",
    "APIGatewayConfig",
    "AuthServiceConfig",
    "AgentOrchestratorConfig",
    "AIRuntimeConfig",
    "MemoryServiceConfig",
    "load_config",
    "get_config_dict",
]
