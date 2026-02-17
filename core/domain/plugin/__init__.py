"""
Plugin Domain Module

Comprehensive plugin system for CognitionOS.
Provides entities, repositories, and services for plugin management.
"""

from .entities import (
    # Enums
    PluginStatus,
    PluginRuntime,
    ExecutionStatus,
    TrustLevel,
    
    # Value Objects
    PluginId,
    VersionInfo,
    Permission,
    SandboxConfig,
    TrustFactor,
    
    # Entities
    PluginManifest,
    PluginTrustScore,
    Plugin,
    PluginExecution,
    PluginInstallation,
)

from .repositories import (
    PluginRepository,
    PluginExecutionRepository,
    PluginInstallationRepository,
)

from .services import (
    PluginRegistryService,
    PluginTrustScoringService,
    PluginExecutionService,
    PluginMarketplaceService,
)

__all__ = [
    # Enums
    "PluginStatus",
    "PluginRuntime",
    "ExecutionStatus",
    "TrustLevel",
    
    # Value Objects
    "PluginId",
    "VersionInfo",
    "Permission",
    "SandboxConfig",
    "TrustFactor",
    
    # Entities
    "PluginManifest",
    "PluginTrustScore",
    "Plugin",
    "PluginExecution",
    "PluginInstallation",
    
    # Repositories
    "PluginRepository",
    "PluginExecutionRepository",
    "PluginInstallationRepository",
    
    # Services
    "PluginRegistryService",
    "PluginTrustScoringService",
    "PluginExecutionService",
    "PluginMarketplaceService",
]
