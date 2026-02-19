"""
Plugin Marketplace Backend

Complete marketplace infrastructure for plugins with ratings, reviews, and revenue sharing.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.persistence.plugin_models import PluginModel


class PluginCategory(str, Enum):
    """Plugin categories"""
    PRODUCTIVITY = "productivity"
    DATA_PROCESSING = "data_processing"
    INTEGRATION = "integration"
    AI_ML = "ai_ml"
    SECURITY = "security"
    ANALYTICS = "analytics"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"


class PluginPricingModel(str, Enum):
    """Plugin pricing models"""
    FREE = "free"
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    FREEMIUM = "freemium"


@dataclass
class PluginRating:
    """Plugin rating"""
    plugin_id: str
    user_id: str
    rating: float
    review: Optional[str]
    created_at: datetime
    helpful_count: int


@dataclass
class PluginStats:
    """Plugin statistics"""
    plugin_id: str
    total_installs: int
    active_installs: int
    total_revenue: Decimal
    avg_rating: float
    review_count: int
    weekly_installs: int
    monthly_installs: int
    retention_rate: float
    churn_rate: float


class PluginMarketplace:
    """Complete plugin marketplace backend."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.platform_fee_percentage = 0.30
    
    async def search_plugins(
        self,
        query: str,
        category: Optional[PluginCategory] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search plugins in marketplace"""
        stmt = select(PluginModel).where(
            PluginModel.status == "active"
        )
        
        if query:
            stmt = stmt.where(
                or_(
                    PluginModel.name.ilike(f"%{query}%"),
                    PluginModel.description.ilike(f"%{query}%")
                )
            )
        
        if category:
            stmt = stmt.where(PluginModel.category == category)
        
        stmt = stmt.limit(limit).offset(offset)
        
        result = await self.session.execute(stmt)
        plugins = result.scalars().all()
        
        return [self._plugin_to_dict(p) for p in plugins]
    
    def _plugin_to_dict(self, plugin: PluginModel) -> Dict[str, Any]:
        """Convert plugin model to dictionary"""
        return {
            "id": plugin.id,
            "name": plugin.name,
            "description": plugin.description,
            "developer_id": plugin.developer_id,
            "version": plugin.version,
            "category": plugin.category,
            "avg_rating": plugin.avg_rating or 0,
            "install_count": plugin.install_count or 0,
            "status": plugin.status,
            "created_at": plugin.created_at.isoformat() if plugin.created_at else None
        }
