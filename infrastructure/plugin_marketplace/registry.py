"""
Plugin Registry - Plugin metadata storage and lookup.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass
class PluginMetadata:
    """Plugin metadata for registry."""

    id: UUID
    name: str
    version: str
    description: str
    developer_id: UUID
    category: str
    status: str
    created_at: datetime
    updated_at: datetime
    manifest: Dict[str, Any]


class PluginRegistry:
    """
    Registry for plugin metadata.
    Delegates to plugin repository for persistence.
    """

    def __init__(self, plugin_repository, install_repository=None):
        self._repo = plugin_repository
        self._install_repo = install_repository

    async def get_plugin(self, plugin_id: UUID) -> Optional[PluginMetadata]:
        """Get plugin metadata by ID."""
        from core.domain.plugin.entities import PluginId
        plugin = await self._repo.get_by_id(PluginId(value=plugin_id))
        if plugin is None:
            return None
        return self._plugin_to_metadata(plugin)

    async def list_by_tenant(self, tenant_id: UUID, limit: int = 50) -> List[PluginMetadata]:
        """List plugins installed for tenant. Requires install_repo in constructor."""
        install_repo = getattr(self, "_install_repo", None)
        if install_repo is None:
            return []
        installations = await install_repo.list_by_tenant(tenant_id)
        result = []
        for inst in installations[:limit]:
            plugin = await self._repo.get_by_id(inst.plugin_id)
            if plugin:
                result.append(self._plugin_to_metadata(plugin))
        return result

    def _plugin_to_metadata(self, plugin) -> PluginMetadata:
        id_val = plugin.id.value if hasattr(plugin.id, "value") else plugin.id
        return PluginMetadata(
            id=id_val,
            name=plugin.name,
            version=str(getattr(plugin, "version", "0.1.0")),
            description=getattr(plugin, "description", "") or "",
            developer_id=getattr(plugin, "developer_id", id_val),
            category=getattr(plugin, "category", "general") or "general",
            status=str(getattr(plugin, "status", "active")),
            created_at=getattr(plugin, "created_at", None),
            updated_at=getattr(plugin, "updated_at", None),
            manifest=getattr(plugin, "manifest", {}) or {},
        )
