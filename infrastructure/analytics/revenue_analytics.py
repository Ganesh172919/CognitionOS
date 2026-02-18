"""
Revenue Analytics Engine

Advanced analytics for revenue metrics, churn analysis, and forecasting.
Provides real-time insights into business performance and growth trends.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from enum import Enum

from sqlalchemy import select, func, and_, or_, extract
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.billing.entities import (
    Subscription,
    SubscriptionStatus,
    SubscriptionTier,
    Invoice,
    InvoiceStatus,
)
from infrastructure.persistence.billing_models import (
    SubscriptionModel,
    InvoiceModel,
    UsageRecordModel,
)

logger = logging.getLogger(__name__)


class RevenueMetricType(str, Enum):
    """Types of revenue metrics."""
    MRR = "mrr"  # Monthly Recurring Revenue
    ARR = "arr"  # Annual Recurring Revenue
    ARPU = "arpu"  # Average Revenue Per User
    LTV = "ltv"  # Lifetime Value
    CAC = "cac"  # Customer Acquisition Cost
    CHURN_RATE = "churn_rate"
    EXPANSION_RATE = "expansion_rate"


class CohortPeriod(str, Enum):
    """Cohort analysis periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class RevenueAnalyticsEngine:
    """
    Revenue analytics engine for business intelligence.
    
    Features:
    - MRR/ARR tracking with trends
    - Cohort analysis for retention
    - Churn prediction
    - Revenue forecasting
    - Customer segmentation
    - Growth rate analysis
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize revenue analytics engine.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def calculate_mrr(
        self,
        as_of_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Monthly Recurring Revenue (MRR).
        
        MRR = Sum of all active monthly subscription values
        Annual subscriptions are normalized to monthly values (annual_amount / 12)
        
        Args:
            as_of_date: Calculate MRR as of this date (defaults to now)
            
        Returns:
            MRR breakdown by tier and totals
        """
        as_of = as_of_date or datetime.utcnow()
        
        # Get all active subscriptions as of date
        result = await self.session.execute(
            select(
                SubscriptionModel.tier,
                func.count(SubscriptionModel.id).label("count"),
                func.sum(
                    func.case(
                        (SubscriptionModel.billing_cycle == "monthly", SubscriptionModel.amount),
                        (SubscriptionModel.billing_cycle == "yearly", SubscriptionModel.amount / 12),
                        else_=0
                    )
                ).label("mrr")
            )
            .where(
                and_(
                    SubscriptionModel.status == SubscriptionStatus.ACTIVE,
                    SubscriptionModel.created_at <= as_of,
                    or_(
                        SubscriptionModel.cancelled_at.is_(None),
                        SubscriptionModel.cancelled_at > as_of,
                    ),
                )
            )
            .group_by(SubscriptionModel.tier)
        )
        
        by_tier = {}
        total_mrr = Decimal(0)
        total_count = 0
        
        for row in result:
            tier = row.tier
            count = row.count
            mrr = Decimal(row.mrr or 0)
            
            by_tier[tier] = {
                "subscriber_count": count,
                "mrr": float(mrr),
            }
            total_mrr += mrr
            total_count += count
        
        return {
            "as_of_date": as_of.isoformat(),
            "total_mrr": float(total_mrr),
            "total_subscribers": total_count,
            "by_tier": by_tier,
        }
    
    async def calculate_arr(
        self,
        as_of_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Annual Recurring Revenue (ARR).
        
        ARR = MRR * 12
        
        Args:
            as_of_date: Calculate ARR as of this date
            
        Returns:
            ARR metrics
        """
        mrr_data = await self.calculate_mrr(as_of_date)
        
        return {
            "as_of_date": mrr_data["as_of_date"],
            "total_arr": mrr_data["total_mrr"] * 12,
            "total_subscribers": mrr_data["total_subscribers"],
            "by_tier": {
                tier: {
                    "subscriber_count": data["subscriber_count"],
                    "arr": data["mrr"] * 12,
                }
                for tier, data in mrr_data["by_tier"].items()
            },
        }
    
    async def calculate_churn_rate(
        self,
        period_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Calculate customer churn rate.
        
        Churn Rate = (Customers Lost / Total Customers at Start) * 100
        
        Args:
            period_days: Period to calculate churn over
            
        Returns:
            Churn rate metrics
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Count customers at start of period
        result_start = await self.session.execute(
            select(func.count(SubscriptionModel.id))
            .where(
                and_(
                    SubscriptionModel.status == SubscriptionStatus.ACTIVE,
                    SubscriptionModel.created_at <= start_date,
                    or_(
                        SubscriptionModel.cancelled_at.is_(None),
                        SubscriptionModel.cancelled_at > start_date,
                    ),
                )
            )
        )
        customers_start = result_start.scalar() or 0
        
        # Count customers who churned during period
        result_churned = await self.session.execute(
            select(func.count(SubscriptionModel.id))
            .where(
                and_(
                    SubscriptionModel.cancelled_at.isnot(None),
                    SubscriptionModel.cancelled_at >= start_date,
                    SubscriptionModel.cancelled_at <= end_date,
                )
            )
        )
        customers_churned = result_churned.scalar() or 0
        
        # Calculate churn rate
        churn_rate = (customers_churned / customers_start * 100) if customers_start > 0 else 0
        
        # Calculate revenue churn
        result_revenue = await self.session.execute(
            select(
                func.sum(
                    func.case(
                        (SubscriptionModel.billing_cycle == "monthly", SubscriptionModel.amount),
                        (SubscriptionModel.billing_cycle == "yearly", SubscriptionModel.amount / 12),
                        else_=0
                    )
                )
            )
            .where(
                and_(
                    SubscriptionModel.cancelled_at.isnot(None),
                    SubscriptionModel.cancelled_at >= start_date,
                    SubscriptionModel.cancelled_at <= end_date,
                )
            )
        )
        revenue_churned = Decimal(result_revenue.scalar() or 0)
        
        return {
            "period_days": period_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "customers_at_start": customers_start,
            "customers_churned": customers_churned,
            "customer_churn_rate": round(churn_rate, 2),
            "revenue_churned_mrr": float(revenue_churned),
        }
    
    async def calculate_arpu(
        self,
        as_of_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Average Revenue Per User (ARPU).
        
        ARPU = Total MRR / Total Active Customers
        
        Args:
            as_of_date: Calculate ARPU as of this date
            
        Returns:
            ARPU metrics by tier
        """
        mrr_data = await self.calculate_mrr(as_of_date)
        
        by_tier = {}
        for tier, data in mrr_data["by_tier"].items():
            arpu = data["mrr"] / data["subscriber_count"] if data["subscriber_count"] > 0 else 0
            by_tier[tier] = {
                "arpu": round(arpu, 2),
                "subscriber_count": data["subscriber_count"],
            }
        
        overall_arpu = (
            mrr_data["total_mrr"] / mrr_data["total_subscribers"]
            if mrr_data["total_subscribers"] > 0
            else 0
        )
        
        return {
            "as_of_date": mrr_data["as_of_date"],
            "overall_arpu": round(overall_arpu, 2),
            "by_tier": by_tier,
        }
    
    async def calculate_ltv(
        self,
        average_customer_lifetime_months: int = 24,
    ) -> Dict[str, Any]:
        """
        Calculate Customer Lifetime Value (LTV).
        
        LTV = ARPU * Average Customer Lifetime (months)
        
        Args:
            average_customer_lifetime_months: Average months a customer stays
            
        Returns:
            LTV metrics
        """
        arpu_data = await self.calculate_arpu()
        
        by_tier = {}
        for tier, data in arpu_data["by_tier"].items():
            ltv = data["arpu"] * average_customer_lifetime_months
            by_tier[tier] = {
                "ltv": round(ltv, 2),
                "arpu": data["arpu"],
                "lifetime_months": average_customer_lifetime_months,
            }
        
        overall_ltv = arpu_data["overall_arpu"] * average_customer_lifetime_months
        
        return {
            "as_of_date": arpu_data["as_of_date"],
            "overall_ltv": round(overall_ltv, 2),
            "lifetime_months": average_customer_lifetime_months,
            "by_tier": by_tier,
        }
    
    async def forecast_revenue(
        self,
        months_ahead: int = 12,
        growth_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Forecast future revenue based on current trends.
        
        Uses simple growth rate model or calculates from historical data.
        
        Args:
            months_ahead: Number of months to forecast
            growth_rate: Monthly growth rate (e.g., 0.05 for 5%). If None, calculated from data.
            
        Returns:
            Revenue forecast
        """
        # Get current MRR
        current_mrr_data = await self.calculate_mrr()
        current_mrr = current_mrr_data["total_mrr"]
        
        # Calculate growth rate if not provided
        if growth_rate is None:
            # Calculate from last 3 months
            three_months_ago = datetime.utcnow() - timedelta(days=90)
            past_mrr_data = await self.calculate_mrr(three_months_ago)
            past_mrr = past_mrr_data["total_mrr"]
            
            if past_mrr > 0:
                # Monthly compound growth rate
                growth_rate = ((current_mrr / past_mrr) ** (1/3) - 1)
            else:
                growth_rate = 0.05  # Default 5% if no historical data
        
        # Generate forecast
        forecast = []
        projected_mrr = current_mrr
        
        for month in range(1, months_ahead + 1):
            projected_mrr *= (1 + growth_rate)
            forecast.append({
                "month": month,
                "date": (datetime.utcnow() + timedelta(days=30 * month)).strftime("%Y-%m"),
                "projected_mrr": round(projected_mrr, 2),
                "projected_arr": round(projected_mrr * 12, 2),
            })
        
        return {
            "current_mrr": current_mrr,
            "growth_rate": round(growth_rate * 100, 2),
            "months_ahead": months_ahead,
            "forecast": forecast,
        }
    
    async def get_revenue_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive revenue summary.
        
        Returns all key metrics in a single response.
        
        Returns:
            Complete revenue summary
        """
        mrr = await self.calculate_mrr()
        arr = await self.calculate_arr()
        churn = await self.calculate_churn_rate()
        arpu = await self.calculate_arpu()
        ltv = await self.calculate_ltv()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "mrr": mrr,
            "arr": arr,
            "churn": churn,
            "arpu": arpu,
            "ltv": ltv,
        }
