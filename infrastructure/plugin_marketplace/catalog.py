"""
Plugin Catalog - Browsable plugin marketplace catalog.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass
class PluginCatalogEntry:
    """Catalog entry for a plugin."""

    id: UUID
    name: str
    version: str
    description: str
    category: str
    avg_rating: float
    install_count: int
    status: str


class PluginCatalog:
    """
    Catalog for browsing and searching plugins.
    """

    def __init__(self, session):
        self._session = session

    async def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[PluginCatalogEntry]:
        """Search plugins in catalog."""
        from infrastructure.marketplace.plugin_marketplace import PluginMarketplace, PluginCategory

        marketplace = PluginMarketplace(self._session)
        cat = PluginCategory(category) if category else None
        results = await marketplace.search_plugins(
            query=query or "",
            category=cat,
            limit=limit,
            offset=offset,
        )
        return [
            PluginCatalogEntry(
                id=r["id"],
                name=r["name"],
                version=r.get("version", "0.1.0"),
                description=r.get("description", ""),
                category=r.get("category", "general"),
                avg_rating=r.get("avg_rating", 0.0),
                install_count=r.get("install_count", 0),
                status=r.get("status", "active"),
            )
            for r in results
        ]
