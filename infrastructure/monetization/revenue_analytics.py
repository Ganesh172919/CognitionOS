"""
Revenue Analytics and Forecasting Engine

Real-time revenue tracking, MRR/ARR calculation, churn analysis,
and ML-powered forecasting.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RevenueMetricType(str, Enum):
    """Types of revenue metrics"""
    MRR = "mrr"  # Monthly Recurring Revenue
    ARR = "arr"  # Annual Recurring Revenue
    ARPU = "arpu"  # Average Revenue Per User
    LTV = "ltv"  # Lifetime Value
    CAC = "cac"  # Customer Acquisition Cost
    CHURN_RATE = "churn_rate"
    EXPANSION_MRR = "expansion_mrr"
    CONTRACTION_MRR = "contraction_mrr"


@dataclass
class RevenueMetrics:
    """Revenue metrics snapshot"""
    period_start: datetime
    period_end: datetime

    # Core metrics
    mrr: float = 0.0
    arr: float = 0.0
    arpu: float = 0.0

    # Growth metrics
    new_mrr: float = 0.0
    expansion_mrr: float = 0.0
    contraction_mrr: float = 0.0
    churned_mrr: float = 0.0
    net_new_mrr: float = 0.0

    # Customer metrics
    total_customers: int = 0
    new_customers: int = 0
    churned_customers: int = 0
    active_customers: int = 0

    # Efficiency metrics
    ltv: float = 0.0
    cac: float = 0.0
    ltv_cac_ratio: float = 0.0

    # Rates
    churn_rate: float = 0.0
    growth_rate: float = 0.0
    net_retention: float = 0.0

    # Breakdown by tier
    mrr_by_tier: Dict[str, float] = field(default_factory=dict)
    customers_by_tier: Dict[str, int] = field(default_factory=dict)


@dataclass
class RevenueforecastResult:
    """Revenue forecast"""
    forecast_date: datetime
    forecasted_mrr: float
    confidence_low: float
    confidence_high: float
    factors: Dict[str, Any] = field(default_factory=dict)


class RevenueAnalytics:
    """
    Comprehensive revenue analytics system

    Tracks all revenue metrics, cohort analysis, and provides
    actionable insights.
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self._subscriptions: List[Dict[str, Any]] = []
        self._payments: List[Dict[str, Any]] = []
        self._historical_metrics: List[RevenueMetrics] = []

    async def calculate_current_metrics(self) -> RevenueMetrics:
        """Calculate current revenue metrics"""

        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = now

        metrics = RevenueMetrics(
            period_start=period_start,
            period_end=period_end
        )

        # Calculate MRR from active subscriptions
        active_subs = self._get_active_subscriptions()
        metrics.total_customers = len(active_subs)

        mrr_by_tier = defaultdict(float)
        customers_by_tier = defaultdict(int)

        for sub in active_subs:
            tier = sub.get("tier", "unknown")
            monthly_value = sub.get("monthly_value", 0.0)

            metrics.mrr += monthly_value
            mrr_by_tier[tier] += monthly_value
            customers_by_tier[tier] += 1

        metrics.mrr_by_tier = dict(mrr_by_tier)
        metrics.customers_by_tier = dict(customers_by_tier)

        # Calculate ARR
        metrics.arr = metrics.mrr * 12

        # Calculate ARPU
        if metrics.total_customers > 0:
            metrics.arpu = metrics.mrr / metrics.total_customers

        # Calculate growth metrics
        await self._calculate_growth_metrics(metrics, period_start)

        # Calculate efficiency metrics
        await self._calculate_efficiency_metrics(metrics)

        # Store for historical tracking
        self._historical_metrics.append(metrics)

        return metrics

    def _get_active_subscriptions(self) -> List[Dict[str, Any]]:
        """Get currently active subscriptions"""
        # Mock data - would query from database
        return [
            {"tenant_id": f"tenant_{i}", "tier": "pro", "monthly_value": 99.0}
            for i in range(500)
        ] + [
            {"tenant_id": f"tenant_ent_{i}", "tier": "enterprise", "monthly_value": 499.0}
            for i in range(100)
        ] + [
            {"tenant_id": f"tenant_starter_{i}", "tier": "starter", "monthly_value": 29.0}
            for i in range(1000)
        ]

    async def _calculate_growth_metrics(
        self,
        metrics: RevenueMetrics,
        period_start: datetime
    ):
        """Calculate MRR growth components"""

        # Get previous period for comparison
        prev_period_start = period_start - timedelta(days=30)

        # New MRR from new customers
        new_subs = self._get_new_subscriptions(period_start)
        metrics.new_mrr = sum(s.get("monthly_value", 0.0) for s in new_subs)
        metrics.new_customers = len(new_subs)

        # Expansion MRR from upgrades
        upgrades = self._get_upgrades(period_start)
        metrics.expansion_mrr = sum(u.get("mrr_increase", 0.0) for u in upgrades)

        # Contraction MRR from downgrades
        downgrades = self._get_downgrades(period_start)
        metrics.contraction_mrr = sum(d.get("mrr_decrease", 0.0) for d in downgrades)

        # Churned MRR from cancellations
        churned_subs = self._get_churned_subscriptions(period_start)
        metrics.churned_mrr = sum(s.get("monthly_value", 0.0) for s in churned_subs)
        metrics.churned_customers = len(churned_subs)

        # Net New MRR
        metrics.net_new_mrr = (
            metrics.new_mrr +
            metrics.expansion_mrr -
            metrics.contraction_mrr -
            metrics.churned_mrr
        )

        # Calculate rates
        if metrics.total_customers > 0:
            metrics.churn_rate = metrics.churned_customers / metrics.total_customers

        # Net retention = (Starting MRR + Expansion - Contraction - Churn) / Starting MRR
        prev_metrics = self._get_metrics_for_period(prev_period_start)
        if prev_metrics and prev_metrics.mrr > 0:
            metrics.net_retention = (
                (prev_metrics.mrr + metrics.expansion_mrr - metrics.contraction_mrr - metrics.churned_mrr) /
                prev_metrics.mrr
            )
            metrics.growth_rate = (metrics.mrr - prev_metrics.mrr) / prev_metrics.mrr
        else:
            metrics.net_retention = 1.0
            metrics.growth_rate = 0.0

        metrics.active_customers = metrics.total_customers

    def _get_new_subscriptions(self, since: datetime) -> List[Dict[str, Any]]:
        """Get new subscriptions since date"""
        # Mock - would query database
        return [
            {"tenant_id": f"new_{i}", "monthly_value": 99.0}
            for i in range(50)
        ]

    def _get_upgrades(self, since: datetime) -> List[Dict[str, Any]]:
        """Get upgrades since date"""
        return [
            {"tenant_id": f"upgrade_{i}", "mrr_increase": 70.0}  # Starter to Pro
            for i in range(20)
        ]

    def _get_downgrades(self, since: datetime) -> List[Dict[str, Any]]:
        """Get downgrades since date"""
        return [
            {"tenant_id": f"downgrade_{i}", "mrr_decrease": 70.0}
            for i in range(5)
        ]

    def _get_churned_subscriptions(self, since: datetime) -> List[Dict[str, Any]]:
        """Get churned subscriptions since date"""
        return [
            {"tenant_id": f"churned_{i}", "monthly_value": 99.0}
            for i in range(10)
        ]

    def _get_metrics_for_period(self, period_start: datetime) -> Optional[RevenueMetrics]:
        """Get historical metrics for period"""
        for metrics in reversed(self._historical_metrics):
            if metrics.period_start == period_start:
                return metrics
        return None

    async def _calculate_efficiency_metrics(self, metrics: RevenueMetrics):
        """Calculate LTV, CAC, and related metrics"""

        # LTV = ARPU × Customer Lifetime
        # Customer Lifetime = 1 / Churn Rate
        if metrics.churn_rate > 0:
            customer_lifetime_months = 1 / metrics.churn_rate
            metrics.ltv = metrics.arpu * customer_lifetime_months
        else:
            metrics.ltv = metrics.arpu * 36  # Assume 3 years if no churn

        # CAC - would calculate from marketing spend
        # Mock value
        metrics.cac = 150.0

        # LTV:CAC ratio
        if metrics.cac > 0:
            metrics.ltv_cac_ratio = metrics.ltv / metrics.cac

    def get_historical_metrics(
        self,
        months: int = 12
    ) -> List[RevenueMetrics]:
        """Get historical metrics"""
        return self._historical_metrics[-months:]

    def analyze_cohorts(
        self,
        cohort_month: datetime
    ) -> Dict[str, Any]:
        """Analyze customer cohort"""

        # Get customers who started in cohort_month
        cohort_customers = self._get_cohort_customers(cohort_month)

        # Track retention over time
        retention_by_month = {}
        for months_since in range(12):
            check_date = cohort_month + timedelta(days=30 * months_since)
            still_active = self._get_active_from_cohort(cohort_customers, check_date)
            retention_by_month[months_since] = len(still_active) / len(cohort_customers) if cohort_customers else 0

        return {
            "cohort_month": cohort_month.strftime("%Y-%m"),
            "cohort_size": len(cohort_customers),
            "retention_by_month": retention_by_month
        }

    def _get_cohort_customers(self, cohort_month: datetime) -> List[str]:
        """Get customers who joined in specific month"""
        # Mock
        return [f"customer_{i}" for i in range(100)]

    def _get_active_from_cohort(
        self,
        cohort_customers: List[str],
        check_date: datetime
    ) -> List[str]:
        """Get customers from cohort still active at date"""
        # Mock - assume 90% retention
        return cohort_customers[:int(len(cohort_customers) * 0.9)]

    def get_revenue_breakdown(self) -> Dict[str, Any]:
        """Get detailed revenue breakdown"""

        metrics = self._historical_metrics[-1] if self._historical_metrics else None
        if not metrics:
            return {}

        return {
            "mrr": {
                "total": metrics.mrr,
                "by_tier": metrics.mrr_by_tier,
                "new": metrics.new_mrr,
                "expansion": metrics.expansion_mrr,
                "contraction": metrics.contraction_mrr,
                "churned": metrics.churned_mrr,
                "net_new": metrics.net_new_mrr
            },
            "customers": {
                "total": metrics.total_customers,
                "by_tier": metrics.customers_by_tier,
                "new": metrics.new_customers,
                "churned": metrics.churned_customers,
                "active": metrics.active_customers
            },
            "metrics": {
                "arr": metrics.arr,
                "arpu": metrics.arpu,
                "ltv": metrics.ltv,
                "cac": metrics.cac,
                "ltv_cac_ratio": metrics.ltv_cac_ratio,
                "churn_rate": metrics.churn_rate,
                "growth_rate": metrics.growth_rate,
                "net_retention": metrics.net_retention
            }
        }


class ForecastEngine:
    """
    ML-powered revenue forecasting

    Predicts future revenue based on historical trends and growth factors.
    """

    def __init__(self, analytics: RevenueAnalytics):
        self.analytics = analytics

    async def forecast_mrr(
        self,
        months_ahead: int = 12
    ) -> List[RevenueforecastResult]:
        """
        Forecast MRR for future months

        Args:
            months_ahead: Number of months to forecast

        Returns:
            List of monthly forecasts
        """
        logger.info(f"Forecasting MRR for next {months_ahead} months")

        # Get historical data
        historical = self.analytics.get_historical_metrics(12)

        if len(historical) < 3:
            logger.warning("Insufficient data for forecasting")
            return []

        # Calculate growth trend
        recent_growth_rates = [
            h.growth_rate for h in historical[-6:]
            if h.growth_rate is not None
        ]

        avg_growth_rate = sum(recent_growth_rates) / len(recent_growth_rates) if recent_growth_rates else 0.0

        # Current MRR
        current_mrr = historical[-1].mrr

        # Generate forecasts
        forecasts = []
        for month in range(1, months_ahead + 1):
            forecast_date = datetime.utcnow() + timedelta(days=30 * month)

            # Simple compound growth model
            forecasted_mrr = current_mrr * ((1 + avg_growth_rate) ** month)

            # Confidence interval (mock - would use statistical methods)
            confidence_range = forecasted_mrr * 0.15
            confidence_low = forecasted_mrr - confidence_range
            confidence_high = forecasted_mrr + confidence_range

            forecast = RevenueforecastResult(
                forecast_date=forecast_date,
                forecasted_mrr=forecasted_mrr,
                confidence_low=confidence_low,
                confidence_high=confidence_high,
                factors={
                    "avg_growth_rate": avg_growth_rate,
                    "months_ahead": month,
                    "model": "compound_growth"
                }
            )

            forecasts.append(forecast)

        return forecasts

    async def predict_churn_risk(
        self,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Predict churn risk for tenant

        Args:
            tenant_id: Tenant to analyze

        Returns:
            Churn risk prediction
        """
        # Mock implementation - would use ML model
        # Features: usage patterns, support tickets, payment issues, etc.

        risk_score = 0.0

        # Check usage decline
        # Check support tickets
        # Check payment failures
        # Check engagement

        risk_score = 0.25  # Mock value

        risk_level = "low"
        if risk_score > 0.7:
            risk_level = "high"
        elif risk_score > 0.4:
            risk_level = "medium"

        return {
            "tenant_id": tenant_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "contributing_factors": [
                "Declining API usage (-30% last month)",
                "No activity in last 7 days"
            ],
            "recommended_actions": [
                "Send re-engagement email",
                "Offer personalized onboarding",
                "Check for technical issues"
            ]
        }

    async def calculate_expansion_opportunities(self) -> List[Dict[str, Any]]:
        """Identify expansion revenue opportunities"""

        opportunities = []

        # Mock - would analyze actual usage patterns
        # Look for:
        # - Customers hitting tier limits
        # - Heavy feature users on lower tiers
        # - Power users without advanced features

        opportunities.append({
            "tenant_id": "tenant_123",
            "current_tier": "starter",
            "recommended_tier": "pro",
            "reason": "Hitting API rate limits frequently",
            "potential_mrr_increase": 70.0,
            "confidence": 0.85
        })

        return opportunities
