"""
Plugin Marketplace Infrastructure

Registry, catalog, and install flow for plugins.
"""

from infrastructure.plugin_marketplace.registry import PluginRegistry
from infrastructure.plugin_marketplace.catalog import PluginCatalog
from infrastructure.plugin_marketplace.install_flow import PluginInstallFlow

__all__ = ["PluginRegistry", "PluginCatalog", "PluginInstallFlow"]
