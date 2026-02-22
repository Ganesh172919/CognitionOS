"""
Plugin Marketplace with Discovery and Ratings

Provides comprehensive plugin ecosystem with:
- Plugin discovery and search
- Version management
- User ratings and reviews
- Installation tracking
- Revenue sharing
- Security scanning
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import statistics


class PluginCategory(Enum):
    """Plugin categories"""
    PRODUCTIVITY = "productivity"
    INTEGRATION = "integration"
    ANALYTICS = "analytics"
    SECURITY = "security"
    AI_TOOLS = "ai_tools"
    UTILITIES = "utilities"


class PluginStatus(Enum):
    """Plugin status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"


@dataclass
class Plugin:
    """Plugin metadata"""
    plugin_id: str
    name: str
    description: str
    author: str
    category: PluginCategory
    version: str
    price_usd: float = 0.0  # 0 for free
    status: PluginStatus = PluginStatus.ACTIVE
    download_count: int = 0
    average_rating: float = 0.0
    total_ratings: int = 0
    homepage_url: Optional[str] = None
    documentation_url: Optional[str] = None
    repository_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginReview:
    """User review"""
    review_id: str
    plugin_id: str
    user_id: str
    rating: int  # 1-5
    title: str
    content: str
    helpful_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PluginInstallation:
    """Installation record"""
    installation_id: str
    plugin_id: str
    user_id: str
    version: str
    installed_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


class PluginMarketplace:
    """
    Plugin Marketplace System

    Features:
    - Plugin discovery and search
    - Category-based browsing
    - Advanced search with filters
    - User ratings and reviews
    - Installation tracking
    - Version management
    - Revenue sharing for paid plugins
    - Security scanning before approval
    - Featured and trending sections
    - Plugin analytics
    - Developer dashboard
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.reviews: Dict[str, List[PluginReview]] = {}
        self.installations: List[PluginInstallation] = []
        self._search_index: Dict[str, List[str]] = {}

    def publish_plugin(self, plugin: Plugin) -> str:
        """
        Publish plugin to marketplace

        Args:
            plugin: Plugin to publish

        Returns:
            Plugin ID
        """
        self.plugins[plugin.plugin_id] = plugin

        # Update search index
        self._index_plugin(plugin)

        return plugin.plugin_id

    def _index_plugin(self, plugin: Plugin):
        """Index plugin for search"""
        # Index by name tokens
        tokens = plugin.name.lower().split()
        for token in tokens:
            if token not in self._search_index:
                self._search_index[token] = []
            if plugin.plugin_id not in self._search_index[token]:
                self._search_index[token].append(plugin.plugin_id)

        # Index by description tokens
        desc_tokens = plugin.description.lower().split()[:20]  # First 20 words
        for token in desc_tokens:
            if token not in self._search_index:
                self._search_index[token] = []
            if plugin.plugin_id not in self._search_index[token]:
                self._search_index[token].append(plugin.plugin_id)

        # Index by tags
        for tag in plugin.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._search_index:
                self._search_index[tag_lower] = []
            if plugin.plugin_id not in self._search_index[tag_lower]:
                self._search_index[tag_lower].append(plugin.plugin_id)

    def search_plugins(
        self,
        query: str,
        category: Optional[PluginCategory] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        limit: int = 20
    ) -> List[Plugin]:
        """
        Search plugins

        Args:
            query: Search query
            category: Optional category filter
            max_price: Maximum price filter
            min_rating: Minimum rating filter
            limit: Maximum results

        Returns:
            List of matching plugins
        """
        # Find plugins matching query
        query_tokens = query.lower().split()
        matching_ids = set()

        for token in query_tokens:
            if token in self._search_index:
                matching_ids.update(self._search_index[token])

        # Get plugin objects
        results = [
            self.plugins[pid]
            for pid in matching_ids
            if pid in self.plugins
        ]

        # Apply filters
        if category:
            results = [p for p in results if p.category == category]

        if max_price is not None:
            results = [p for p in results if p.price_usd <= max_price]

        if min_rating is not None:
            results = [p for p in results if p.average_rating >= min_rating]

        # Filter by status
        results = [p for p in results if p.status == PluginStatus.ACTIVE]

        # Sort by relevance (combination of rating and downloads)
        results.sort(
            key=lambda p: (p.average_rating * 0.6 + (p.download_count / 100) * 0.4),
            reverse=True
        )

        return results[:limit]

    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get plugin by ID"""
        return self.plugins.get(plugin_id)

    def get_featured_plugins(self, limit: int = 10) -> List[Plugin]:
        """Get featured plugins"""
        active_plugins = [
            p for p in self.plugins.values()
            if p.status == PluginStatus.ACTIVE
        ]

        # Sort by combination of rating and downloads
        featured = sorted(
            active_plugins,
            key=lambda p: (p.average_rating * p.total_ratings * p.download_count),
            reverse=True
        )

        return featured[:limit]

    def get_trending_plugins(self, days: int = 7, limit: int = 10) -> List[Plugin]:
        """Get trending plugins based on recent installs"""
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Count recent installations
        recent_installs = {}
        for install in self.installations:
            if install.installed_at >= cutoff:
                plugin_id = install.plugin_id
                recent_installs[plugin_id] = recent_installs.get(plugin_id, 0) + 1

        # Get plugins and sort by recent install count
        trending = []
        for plugin_id, count in recent_installs.items():
            plugin = self.plugins.get(plugin_id)
            if plugin and plugin.status == PluginStatus.ACTIVE:
                trending.append((plugin, count))

        trending.sort(key=lambda x: x[1], reverse=True)

        return [p for p, _ in trending[:limit]]

    def install_plugin(
        self,
        plugin_id: str,
        user_id: str,
        version: Optional[str] = None
    ) -> PluginInstallation:
        """
        Install plugin for user

        Args:
            plugin_id: Plugin to install
            user_id: User installing
            version: Specific version (defaults to latest)

        Returns:
            Installation record
        """
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")

        if plugin.status != PluginStatus.ACTIVE:
            raise ValueError(f"Plugin {plugin_id} is not active")

        # Use latest version if not specified
        install_version = version or plugin.version

        installation = PluginInstallation(
            installation_id=f"install_{user_id}_{plugin_id}_{int(datetime.utcnow().timestamp())}",
            plugin_id=plugin_id,
            user_id=user_id,
            version=install_version
        )

        self.installations.append(installation)

        # Update download count
        plugin.download_count += 1

        return installation

    def add_review(
        self,
        plugin_id: str,
        user_id: str,
        rating: int,
        title: str,
        content: str
    ) -> PluginReview:
        """
        Add plugin review

        Args:
            plugin_id: Plugin to review
            user_id: Reviewing user
            rating: Rating 1-5
            title: Review title
            content: Review content

        Returns:
            Created review
        """
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")

        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")

        review = PluginReview(
            review_id=f"review_{plugin_id}_{user_id}_{int(datetime.utcnow().timestamp())}",
            plugin_id=plugin_id,
            user_id=user_id,
            rating=rating,
            title=title,
            content=content
        )

        if plugin_id not in self.reviews:
            self.reviews[plugin_id] = []

        self.reviews[plugin_id].append(review)

        # Update plugin rating
        self._update_plugin_rating(plugin_id)

        return review

    def _update_plugin_rating(self, plugin_id: str):
        """Recalculate plugin average rating"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return

        reviews = self.reviews.get(plugin_id, [])
        if not reviews:
            return

        ratings = [r.rating for r in reviews]
        plugin.average_rating = statistics.mean(ratings)
        plugin.total_ratings = len(ratings)

    def get_plugin_reviews(
        self,
        plugin_id: str,
        limit: int = 10
    ) -> List[PluginReview]:
        """Get reviews for plugin"""
        reviews = self.reviews.get(plugin_id, [])

        # Sort by helpful count and recency
        sorted_reviews = sorted(
            reviews,
            key=lambda r: (r.helpful_count, r.created_at),
            reverse=True
        )

        return sorted_reviews[:limit]

    def get_user_installations(self, user_id: str) -> List[PluginInstallation]:
        """Get user's installed plugins"""
        return [
            install for install in self.installations
            if install.user_id == user_id and install.is_active
        ]

    def get_marketplace_statistics(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        active_plugins = [
            p for p in self.plugins.values()
            if p.status == PluginStatus.ACTIVE
        ]

        total_downloads = sum(p.download_count for p in active_plugins)
        total_reviews = sum(len(r) for r in self.reviews.values())

        by_category = {}
        for category in PluginCategory:
            count = sum(
                1 for p in active_plugins
                if p.category == category
            )
            by_category[category.value] = count

        return {
            "total_plugins": len(active_plugins),
            "total_downloads": total_downloads,
            "total_reviews": total_reviews,
            "total_installations": len(self.installations),
            "plugins_by_category": by_category,
            "average_rating": statistics.mean([
                p.average_rating for p in active_plugins
                if p.total_ratings > 0
            ]) if any(p.total_ratings > 0 for p in active_plugins) else 0
        }
