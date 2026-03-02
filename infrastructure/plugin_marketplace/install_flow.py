"""
Plugin Install Flow - Install and enable plugins for tenants.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class PluginInstallFlow:
    """
    Handles plugin installation flow for tenants.
    """

    def __init__(self, plugin_installation_repository, plugin_repository):
        self._install_repo = plugin_installation_repository
        self._plugin_repo = plugin_repository

    async def install(
        self,
        tenant_id: UUID,
        plugin_id: UUID,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Install a plugin for a tenant.
        Returns True on success.
        """
        try:
            from core.domain.plugin.entities import PluginId, PluginInstallation

            pid = PluginId(value=plugin_id)
            plugin = await self._plugin_repo.get_by_id(pid)
            if plugin is None:
                logger.warning("Plugin not found: %s", plugin_id)
                return False
            if str(plugin.status).lower() != "active":
                logger.warning("Plugin not active: %s", plugin_id)
                return False
            existing = await self._install_repo.get_by_plugin_and_tenant(pid, tenant_id)
            if existing:
                logger.info("Plugin already installed: %s for tenant %s", plugin_id, tenant_id)
                return True
            installation = PluginInstallation.create(
                plugin_id=pid,
                tenant_id=tenant_id,
                configuration=config or {},
            )
            await self._install_repo.save(installation)
            return True
        except Exception as e:
            logger.exception("Plugin install failed: %s", e)
            return False

    async def uninstall(self, tenant_id: UUID, plugin_id: UUID) -> bool:
        """Uninstall a plugin for a tenant."""
        try:
            from core.domain.plugin.entities import PluginId

            pid = PluginId(value=plugin_id)
            existing = await self._install_repo.get_by_plugin_and_tenant(pid, tenant_id)
            if existing is None:
                logger.info("Plugin not installed: %s for tenant %s", plugin_id, tenant_id)
                return True
            await self._install_repo.delete(existing.id)
            return True
        except Exception as e:
            logger.exception("Plugin uninstall failed: %s", e)
            return False
