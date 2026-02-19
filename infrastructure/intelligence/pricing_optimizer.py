"""
AI-Powered Pricing Recommendation System

Uses machine learning to optimize pricing and tier recommendations.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.billing.entities import SubscriptionTier
from infrastructure.persistence.billing_models import SubscriptionModel, UsageRecordModel


class PricingStrategy(str, Enum):
    """Pricing optimization strategies"""
    VALUE_BASED = "value_based"
    COMPETITIVE = "competitive"
    PENETRATION = "penetration"
    PREMIUM = "premium"
    DYNAMIC = "dynamic"


class TierRecommendation(str, Enum):
    """Tier change recommendations"""
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    STAY = "stay"
    CUSTOM = "custom"


@dataclass
class PricingRecommendation:
    """Pricing recommendation result"""
    tenant_id: str
    current_tier: SubscriptionTier
    recommended_tier: SubscriptionTier
    recommendation_type: TierRecommendation
    confidence_score: float  # 0-1
    estimated_savings_monthly: float
    estimated_revenue_impact: float
    reasoning: str
    factors: Dict[str, float]
    effective_date: datetime


@dataclass
class PricingElasticity:
    """Price elasticity analysis"""
    resource_type: str
    current_price: float
    elasticity_coefficient: float  # < -1 elastic, > -1 inelastic
    optimal_price: float
    expected_revenue_lift: float
    confidence: float


@dataclass
class CustomerValue:
    """Customer lifetime value analysis"""
    tenant_id: str
    ltv: float  # Lifetime value
    cac: float  # Customer acquisition cost
    ltv_cac_ratio: float
    churn_probability: float
    engagement_score: float  # 0-100
    value_segment: str  # "high", "medium", "low"
    retention_priority: str  # "critical", "high", "medium", "low"


class PricingOptimizer:
    """
    AI-powered pricing optimization and recommendation engine.
    
    Features:
    - Usage-based tier recommendations
    - Price elasticity analysis
    - Customer lifetime value calculation
    - Churn risk assessment
    - Dynamic pricing optimization
    - Revenue impact forecasting
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        
        # Pricing configuration
        self.tier_prices = {
            SubscriptionTier.FREE: 0,
            SubscriptionTier.PRO: 49,
            SubscriptionTier.TEAM: 199,
            SubscriptionTier.ENTERPRISE: 999
        }
        
        self.tier_limits = {
            SubscriptionTier.FREE: {"api_calls": 1000, "llm_tokens": 100000},
            SubscriptionTier.PRO: {"api_calls": 10000, "llm_tokens": 1000000},
            SubscriptionTier.TEAM: {"api_calls": 100000, "llm_tokens": 10000000},
            SubscriptionTier.ENTERPRISE: {"api_calls": float('inf'), "llm_tokens": float('inf')}
        }
    
    async def get_tier_recommendation(
        self,
        tenant_id: str,
        analysis_days: int = 30
    ) -> PricingRecommendation:
        """Get tier recommendation for a tenant"""
        # Get current subscription
        sub_query = select(SubscriptionModel).where(
            SubscriptionModel.tenant_id == tenant_id,
            SubscriptionModel.status == "active"
        )
        sub_result = await self.session.execute(sub_query)
        subscription = sub_result.scalar_one_or_none()
        
        if not subscription:
            current_tier = SubscriptionTier.FREE
        else:
            current_tier = subscription.tier
        
        # Analyze usage patterns
        usage_data = await self._get_usage_data(tenant_id, analysis_days)
        
        # Calculate utilization
        utilization = self._calculate_utilization(usage_data, current_tier)
        
        # Get growth rate
        growth_rate = await self._calculate_growth_rate(tenant_id, analysis_days)
        
        # Calculate engagement
        engagement = await self._calculate_engagement_score(tenant_id, analysis_days)
        
        # Determine optimal tier
        recommended_tier, reasoning, factors = self._determine_optimal_tier(
            current_tier,
            utilization,
            growth_rate,
            engagement,
            usage_data
        )
        
        # Calculate financial impact
        savings, revenue_impact = self._calculate_financial_impact(
            current_tier,
            recommended_tier,
            usage_data
        )
        
        # Calculate confidence
        confidence = self._calculate_recommendation_confidence(factors)
        
        # Determine recommendation type
        if recommended_tier == current_tier:
            rec_type = TierRecommendation.STAY
        elif self._tier_level(recommended_tier) > self._tier_level(current_tier):
            rec_type = TierRecommendation.UPGRADE
        else:
            rec_type = TierRecommendation.DOWNGRADE
        
        return PricingRecommendation(
            tenant_id=tenant_id,
            current_tier=current_tier,
            recommended_tier=recommended_tier,
            recommendation_type=rec_type,
            confidence_score=confidence,
            estimated_savings_monthly=savings,
            estimated_revenue_impact=revenue_impact,
            reasoning=reasoning,
            factors=factors,
            effective_date=datetime.utcnow() + timedelta(days=7)  # Week lead time
        )
    
    async def analyze_price_elasticity(
        self,
        resource_type: str,
        price_points: List[float],
        historical_days: int = 90
    ) -> PricingElasticity:
        """Analyze price elasticity for a resource type"""
        # Get historical demand at different price points
        # In production, would use A/B test data or historical pricing changes
        
        # Simulate demand curve (would be real data in production)
        demands = []
        for price in price_points:
            # Simple demand model: demand = base / (1 + price_sensitivity * price)
            base_demand = 10000
            price_sensitivity = 0.1
            demand = base_demand / (1 + price_sensitivity * price)
            demands.append(demand)
        
        # Calculate elasticity
        if len(price_points) >= 2:
            # Point elasticity at current price
            current_price = price_points[0]
            current_demand = demands[0]
            
            # Use second point for elasticity calculation
            price_change = (price_points[1] - price_points[0]) / price_points[0]
            demand_change = (demands[1] - demands[0]) / demands[0]
            
            if price_change != 0:
                elasticity = demand_change / price_change
            else:
                elasticity = 0
        else:
            elasticity = -1.0  # Default assumption
        
        # Find optimal price (maximize revenue)
        revenues = [p * d for p, d in zip(price_points, demands)]
        optimal_idx = revenues.index(max(revenues))
        optimal_price = price_points[optimal_idx]
        
        # Calculate revenue lift
        current_revenue = price_points[0] * demands[0]
        optimal_revenue = price_points[optimal_idx] * demands[optimal_idx]
        revenue_lift = ((optimal_revenue - current_revenue) / current_revenue) * 100
        
        return PricingElasticity(
            resource_type=resource_type,
            current_price=price_points[0],
            elasticity_coefficient=elasticity,
            optimal_price=optimal_price,
            expected_revenue_lift=revenue_lift,
            confidence=0.75  # Would be calculated from data quality
        )
    
    async def calculate_customer_value(
        self,
        tenant_id: str
    ) -> CustomerValue:
        """Calculate customer lifetime value"""
        # Get subscription history
        sub_query = select(SubscriptionModel).where(
            SubscriptionModel.tenant_id == tenant_id
        ).order_by(SubscriptionModel.created_at)
        
        sub_result = await self.session.execute(sub_query)
        subscriptions = sub_result.scalars().all()
        
        if not subscriptions:
            return CustomerValue(
                tenant_id=tenant_id,
                ltv=0,
                cac=50,  # Default CAC
                ltv_cac_ratio=0,
                churn_probability=0.5,
                engagement_score=0,
                value_segment="low",
                retention_priority="low"
            )
        
        # Calculate revenue to date
        total_revenue = sum(
            self.tier_prices[sub.tier] * 
            self._months_active(sub)
            for sub in subscriptions
        )
        
        # Estimate lifetime (simplified)
        account_age_months = (
            datetime.utcnow() - subscriptions[0].created_at
        ).days / 30
        
        # Estimate monthly revenue
        current_sub = subscriptions[-1]
        monthly_revenue = self.tier_prices[current_sub.tier]
        
        # Calculate churn probability
        churn_prob = await self._calculate_churn_probability(tenant_id)
        
        # Calculate expected lifetime
        if churn_prob > 0:
            expected_lifetime_months = 1 / churn_prob
        else:
            expected_lifetime_months = 36  # 3 years default
        
        # Calculate LTV
        ltv = monthly_revenue * expected_lifetime_months
        
        # Estimate CAC (simplified - would come from marketing data)
        cac = 50 + (self.tier_prices[current_sub.tier] * 0.5)
        
        # Calculate engagement
        engagement = await self._calculate_engagement_score(tenant_id, 30)
        
        # Determine value segment
        ltv_cac_ratio = ltv / cac if cac > 0 else 0
        
        if ltv_cac_ratio >= 3:
            value_segment = "high"
            retention_priority = "critical"
        elif ltv_cac_ratio >= 1.5:
            value_segment = "medium"
            retention_priority = "high"
        else:
            value_segment = "low"
            retention_priority = "medium" if churn_prob < 0.5 else "low"
        
        return CustomerValue(
            tenant_id=tenant_id,
            ltv=ltv,
            cac=cac,
            ltv_cac_ratio=ltv_cac_ratio,
            churn_probability=churn_prob,
            engagement_score=engagement,
            value_segment=value_segment,
            retention_priority=retention_priority
        )
    
    async def optimize_dynamic_pricing(
        self,
        resource_type: str,
        target_revenue: float,
        constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Optimize dynamic pricing to meet revenue targets"""
        constraints = constraints or {}
        
        # Get current demand
        current_price = 0.01  # $0.01 per unit default
        
        # Get usage statistics
        query = select(
            func.count(UsageRecordModel.id).label('count'),
            func.sum(UsageRecordModel.quantity).label('total'),
            func.avg(UsageRecordModel.quantity).label('avg')
        ).where(
            UsageRecordModel.resource_type == resource_type,
            UsageRecordModel.timestamp >= datetime.utcnow() - timedelta(days=30)
        )
        
        result = await self.session.execute(query)
        stats = result.one()
        
        current_demand = stats.total or 0
        current_revenue = current_demand * current_price
        
        # Calculate required price to meet target
        if current_demand > 0:
            required_price = target_revenue / current_demand
        else:
            required_price = current_price
        
        # Apply constraints
        min_price = constraints.get('min_price', current_price * 0.5)
        max_price = constraints.get('max_price', current_price * 2.0)
        
        optimized_price = np.clip(required_price, min_price, max_price)
        
        # Estimate demand at new price (simplified elasticity)
        elasticity = -1.2  # Slightly elastic
        price_change = (optimized_price - current_price) / current_price
        demand_change = elasticity * price_change
        new_demand = current_demand * (1 + demand_change)
        
        projected_revenue = optimized_price * new_demand
        
        return {
            "resource_type": resource_type,
            "current_price": current_price,
            "optimized_price": optimized_price,
            "current_demand": current_demand,
            "projected_demand": new_demand,
            "current_revenue": current_revenue,
            "projected_revenue": projected_revenue,
            "revenue_lift": ((projected_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0,
            "price_change_pct": ((optimized_price - current_price) / current_price * 100)
        }
    
    async def get_cohort_analysis(
        self,
        cohort_date: datetime,
        months: int = 12
    ) -> Dict[str, any]:
        """Analyze customer cohort retention and revenue"""
        # Get tenants that signed up in cohort month
        cohort_start = cohort_date.replace(day=1)
        cohort_end = (cohort_start + timedelta(days=32)).replace(day=1)
        
        # Query subscriptions from cohort
        query = select(SubscriptionModel).where(
            SubscriptionModel.created_at >= cohort_start,
            SubscriptionModel.created_at < cohort_end
        )
        
        result = await self.session.execute(query)
        cohort_subs = result.scalars().all()
        
        cohort_size = len(cohort_subs)
        if cohort_size == 0:
            return {
                "cohort_date": cohort_date.isoformat(),
                "cohort_size": 0,
                "retention_rates": [],
                "revenue_by_month": [],
                "ltv": 0
            }
        
        # Calculate retention and revenue by month
        retention_rates = []
        revenue_by_month = []
        
        for month_offset in range(months):
            check_date = cohort_start + timedelta(days=30 * month_offset)
            
            # Count active subscriptions
            active_query = select(func.count(SubscriptionModel.id)).where(
                SubscriptionModel.tenant_id.in_([s.tenant_id for s in cohort_subs]),
                SubscriptionModel.status == "active",
                SubscriptionModel.created_at <= check_date
            )
            
            active_result = await self.session.execute(active_query)
            active_count = active_result.scalar() or 0
            
            retention_rate = (active_count / cohort_size) * 100
            retention_rates.append(retention_rate)
            
            # Calculate revenue
            # Simplified - would track actual charges in production
            month_revenue = active_count * 49  # Average PRO tier price
            revenue_by_month.append(month_revenue)
        
        # Calculate cohort LTV
        total_revenue = sum(revenue_by_month)
        ltv = total_revenue / cohort_size if cohort_size > 0 else 0
        
        return {
            "cohort_date": cohort_date.isoformat(),
            "cohort_size": cohort_size,
            "retention_rates": retention_rates,
            "revenue_by_month": revenue_by_month,
            "ltv": ltv,
            "months_analyzed": months
        }
    
    async def _get_usage_data(
        self,
        tenant_id: str,
        days: int
    ) -> Dict[str, float]:
        """Get usage data for a tenant"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            UsageRecordModel.resource_type,
            func.sum(UsageRecordModel.quantity).label('total')
        ).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.timestamp >= start_date
        ).group_by(UsageRecordModel.resource_type)
        
        result = await self.session.execute(query)
        
        usage = {}
        for row in result:
            usage[row.resource_type] = row.total
        
        return usage
    
    def _calculate_utilization(
        self,
        usage_data: Dict[str, float],
        tier: SubscriptionTier
    ) -> float:
        """Calculate utilization percentage"""
        limits = self.tier_limits[tier]
        
        utilizations = []
        for resource_type, usage in usage_data.items():
            if resource_type in limits:
                limit = limits[resource_type]
                if limit != float('inf'):
                    utilization = (usage / limit) * 100
                    utilizations.append(utilization)
        
        return np.mean(utilizations) if utilizations else 0
    
    async def _calculate_growth_rate(
        self,
        tenant_id: str,
        days: int
    ) -> float:
        """Calculate usage growth rate"""
        mid_point = datetime.utcnow() - timedelta(days=days // 2)
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # First half
        first_query = select(func.sum(UsageRecordModel.quantity)).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.timestamp >= start_date,
            UsageRecordModel.timestamp < mid_point
        )
        first_result = await self.session.execute(first_query)
        first_half = first_result.scalar() or 0
        
        # Second half
        second_query = select(func.sum(UsageRecordModel.quantity)).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.timestamp >= mid_point
        )
        second_result = await self.session.execute(second_query)
        second_half = second_result.scalar() or 0
        
        if first_half > 0:
            growth_rate = ((second_half - first_half) / first_half) * 100
        else:
            growth_rate = 0
        
        return growth_rate
    
    async def _calculate_engagement_score(
        self,
        tenant_id: str,
        days: int
    ) -> float:
        """Calculate engagement score (0-100)"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Count active days
        active_days_query = select(
            func.count(func.distinct(func.date(UsageRecordModel.timestamp)))
        ).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.timestamp >= start_date
        )
        
        result = await self.session.execute(active_days_query)
        active_days = result.scalar() or 0
        
        # Calculate engagement score
        engagement = (active_days / days) * 100
        
        return min(100, engagement)
    
    def _determine_optimal_tier(
        self,
        current_tier: SubscriptionTier,
        utilization: float,
        growth_rate: float,
        engagement: float,
        usage_data: Dict[str, float]
    ) -> Tuple[SubscriptionTier, str, Dict[str, float]]:
        """Determine optimal tier based on multiple factors"""
        factors = {
            "utilization": utilization,
            "growth_rate": growth_rate,
            "engagement": engagement
        }
        
        # Decision logic
        if utilization > 80 and growth_rate > 20:
            # High utilization and growing - recommend upgrade
            recommended = self._next_tier_up(current_tier)
            reasoning = f"High utilization ({utilization:.1f}%) and strong growth ({growth_rate:.1f}%) indicate need for higher tier"
        
        elif utilization < 30 and growth_rate < -10:
            # Low utilization and declining - recommend downgrade
            recommended = self._next_tier_down(current_tier)
            reasoning = f"Low utilization ({utilization:.1f}%) and declining usage ({growth_rate:.1f}%) suggest downgrade opportunity"
        
        elif utilization > 90:
            # Near limit - urgent upgrade needed
            recommended = self._next_tier_up(current_tier)
            reasoning = f"Critical: Utilization at {utilization:.1f}%, upgrade needed to avoid service disruption"
        
        elif engagement < 20 and current_tier != SubscriptionTier.FREE:
            # Low engagement - might downgrade
            recommended = self._next_tier_down(current_tier)
            reasoning = f"Low engagement ({engagement:.1f}%) suggests features may not be fully utilized"
        
        else:
            # Stay on current tier
            recommended = current_tier
            reasoning = f"Current tier is appropriate: utilization {utilization:.1f}%, growth {growth_rate:.1f}%"
        
        return recommended, reasoning, factors
    
    async def _calculate_churn_probability(
        self,
        tenant_id: str
    ) -> float:
        """Calculate probability of churn (0-1)"""
        # Get recent usage trend
        growth_rate = await self._calculate_growth_rate(tenant_id, 30)
        engagement = await self._calculate_engagement_score(tenant_id, 30)
        
        # Simple churn model (would be ML model in production)
        base_churn = 0.15  # 15% base monthly churn
        
        # Adjust based on engagement
        engagement_factor = (100 - engagement) / 100
        
        # Adjust based on growth
        growth_factor = 1.0
        if growth_rate < -20:
            growth_factor = 1.5  # Declining usage increases churn
        elif growth_rate > 20:
            growth_factor = 0.5  # Growing usage decreases churn
        
        churn_prob = base_churn * engagement_factor * growth_factor
        
        return min(0.95, max(0.05, churn_prob))
    
    def _months_active(self, subscription: SubscriptionModel) -> int:
        """Calculate months subscription has been active"""
        if subscription.status == "active":
            end_date = datetime.utcnow()
        else:
            end_date = subscription.updated_at
        
        months = (end_date - subscription.created_at).days / 30
        return max(1, int(months))
    
    def _calculate_financial_impact(
        self,
        current_tier: SubscriptionTier,
        recommended_tier: SubscriptionTier,
        usage_data: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate financial impact of tier change"""
        current_price = self.tier_prices[current_tier]
        recommended_price = self.tier_prices[recommended_tier]
        
        # Monthly savings (from customer perspective)
        savings = max(0, current_price - recommended_price)
        
        # Revenue impact (from business perspective)
        revenue_impact = recommended_price - current_price
        
        return savings, revenue_impact
    
    def _calculate_recommendation_confidence(
        self,
        factors: Dict[str, float]
    ) -> float:
        """Calculate confidence in recommendation"""
        # Based on how clear the signals are
        utilization = factors.get("utilization", 50)
        growth_rate = abs(factors.get("growth_rate", 0))
        engagement = factors.get("engagement", 50)
        
        # Strong signals increase confidence
        if utilization > 80 or utilization < 20:
            util_confidence = 0.9
        elif utilization > 60 or utilization < 40:
            util_confidence = 0.7
        else:
            util_confidence = 0.5
        
        growth_confidence = min(1.0, growth_rate / 100)
        engagement_confidence = engagement / 100
        
        # Weighted average
        confidence = (
            util_confidence * 0.4 +
            growth_confidence * 0.3 +
            engagement_confidence * 0.3
        )
        
        return confidence
    
    def _tier_level(self, tier: SubscriptionTier) -> int:
        """Get numeric level of tier"""
        levels = {
            SubscriptionTier.FREE: 0,
            SubscriptionTier.PRO: 1,
            SubscriptionTier.TEAM: 2,
            SubscriptionTier.ENTERPRISE: 3
        }
        return levels[tier]
    
    def _next_tier_up(self, current_tier: SubscriptionTier) -> SubscriptionTier:
        """Get next tier up"""
        tiers = [
            SubscriptionTier.FREE,
            SubscriptionTier.PRO,
            SubscriptionTier.TEAM,
            SubscriptionTier.ENTERPRISE
        ]
        
        try:
            current_idx = tiers.index(current_tier)
            if current_idx < len(tiers) - 1:
                return tiers[current_idx + 1]
        except ValueError:
            pass
        
        return current_tier
    
    def _next_tier_down(self, current_tier: SubscriptionTier) -> SubscriptionTier:
        """Get next tier down"""
        tiers = [
            SubscriptionTier.FREE,
            SubscriptionTier.PRO,
            SubscriptionTier.TEAM,
            SubscriptionTier.ENTERPRISE
        ]
        
        try:
            current_idx = tiers.index(current_tier)
            if current_idx > 0:
                return tiers[current_idx - 1]
        except ValueError:
            pass
        
        return current_tier
