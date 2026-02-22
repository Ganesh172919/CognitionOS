"""
Customer Lifetime Value (LTV) Prediction Engine

Implements advanced LTV prediction with:
- ML-based churn prediction
- Revenue forecasting per customer
- Cohort analysis
- Customer health scoring
- Retention optimization recommendations
- Expansion revenue opportunities
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from decimal import Decimal
import math


class CustomerTier(Enum):
    """Customer tier classifications"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class ChurnRisk(Enum):
    """Churn risk levels"""
    LOW = "low"          # <10% probability
    MEDIUM = "medium"    # 10-30% probability
    HIGH = "high"        # 30-60% probability
    CRITICAL = "critical"  # >60% probability


class HealthScore(Enum):
    """Customer health status"""
    EXCELLENT = "excellent"  # 80-100
    GOOD = "good"           # 60-79
    FAIR = "fair"           # 40-59
    POOR = "poor"           # 20-39
    CRITICAL = "critical"   # 0-19


@dataclass
class CustomerProfile:
    """Comprehensive customer profile"""
    customer_id: str
    tier: CustomerTier
    signup_date: datetime
    total_revenue: Decimal
    monthly_recurring_revenue: Decimal
    contracts_count: int
    last_activity_date: datetime
    usage_metrics: Dict[str, float]
    engagement_score: float  # 0-100
    support_tickets: int
    nps_score: Optional[int]  # -100 to 100
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LTVPrediction:
    """Customer lifetime value prediction"""
    customer_id: str
    predicted_ltv: Decimal
    confidence_interval: Tuple[Decimal, Decimal]
    confidence_score: float  # 0-1
    time_horizon_months: int
    predicted_monthly_revenue: Decimal
    predicted_retention_months: int
    churn_probability: float
    expansion_potential: Decimal
    factors: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChurnPrediction:
    """Churn risk prediction for customer"""
    customer_id: str
    churn_probability: float  # 0-1
    risk_level: ChurnRisk
    days_to_churn: Optional[int]
    primary_risk_factors: List[Dict[str, Any]]
    recommended_actions: List[str]
    prevention_value: Decimal  # Revenue at risk
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CohortAnalysis:
    """Cohort-based analysis"""
    cohort_name: str
    cohort_start_date: datetime
    customer_count: int
    total_revenue: Decimal
    average_ltv: Decimal
    retention_rates: Dict[int, float]  # month -> retention %
    churn_rates: Dict[int, float]      # month -> churn %
    revenue_by_month: Dict[int, Decimal]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMetrics:
    """Customer health metrics"""
    customer_id: str
    overall_score: float  # 0-100
    health_status: HealthScore
    component_scores: Dict[str, float]
    trend: str  # "improving", "stable", "declining"
    alerts: List[str]
    recommendations: List[str]
    last_updated: datetime = field(default_factory=datetime.utcnow)


class LTVPredictionEngine:
    """
    Advanced Customer Lifetime Value (LTV) prediction engine.

    Features:
    - ML-based LTV forecasting
    - Churn probability prediction
    - Customer health scoring
    - Cohort analysis
    - Retention optimization
    - Expansion opportunity identification
    """

    def __init__(
        self,
        default_churn_rate: float = 0.05,  # 5% monthly churn
        discount_rate: float = 0.01,        # 1% monthly discount for NPV
        min_ltv_prediction_months: int = 12
    ):
        self.default_churn_rate = default_churn_rate
        self.discount_rate = discount_rate
        self.min_ltv_prediction_months = min_ltv_prediction_months

        # Historical data storage
        self.customer_profiles: Dict[str, CustomerProfile] = {}
        self.ltv_predictions: Dict[str, LTVPrediction] = {}
        self.churn_predictions: Dict[str, ChurnPrediction] = {}
        self.cohorts: Dict[str, CohortAnalysis] = {}
        self.health_scores: Dict[str, HealthMetrics] = {}

        # Model parameters (simplified - would be trained ML models)
        self.ltv_model_weights = {
            "mrr": 0.35,
            "tenure": 0.20,
            "engagement": 0.15,
            "tier": 0.15,
            "usage": 0.10,
            "support": 0.05
        }

        self.churn_model_weights = {
            "activity_decay": 0.25,
            "usage_decline": 0.20,
            "engagement_drop": 0.20,
            "support_spike": 0.15,
            "payment_issues": 0.10,
            "feature_adoption": 0.10
        }

    def predict_ltv(
        self,
        customer_id: str,
        time_horizon_months: int = 36
    ) -> LTVPrediction:
        """
        Predict customer lifetime value.

        Uses multiple factors:
        - Current MRR
        - Historical retention
        - Engagement patterns
        - Usage trends
        - Customer tier
        - Support interactions

        Args:
            customer_id: Customer identifier
            time_horizon_months: Prediction time horizon

        Returns:
            LTVPrediction with detailed breakdown
        """
        if customer_id not in self.customer_profiles:
            raise ValueError(f"Customer {customer_id} not found")

        profile = self.customer_profiles[customer_id]

        # Get churn prediction
        churn_pred = self.predict_churn(customer_id)

        # Calculate expected lifetime in months
        if churn_pred.churn_probability > 0:
            expected_lifetime = 1.0 / churn_pred.churn_probability
        else:
            expected_lifetime = time_horizon_months

        expected_lifetime = min(expected_lifetime, time_horizon_months)

        # Calculate base LTV using discounted cash flow
        base_ltv = self._calculate_discounted_revenue(
            monthly_revenue=profile.monthly_recurring_revenue,
            months=int(expected_lifetime),
            churn_rate=churn_pred.churn_probability
        )

        # Add expansion potential
        expansion_potential = self._calculate_expansion_potential(profile)

        # Total predicted LTV
        predicted_ltv = base_ltv + expansion_potential

        # Calculate confidence interval (simplified)
        variance = self._calculate_ltv_variance(profile, churn_pred)
        std_dev = math.sqrt(variance)
        confidence_interval = (
            Decimal(str(max(0, float(predicted_ltv) - 1.96 * std_dev))),
            Decimal(str(float(predicted_ltv) + 1.96 * std_dev))
        )

        # Calculate confidence score
        confidence_score = self._calculate_prediction_confidence(
            profile, churn_pred, expected_lifetime
        )

        prediction = LTVPrediction(
            customer_id=customer_id,
            predicted_ltv=predicted_ltv,
            confidence_interval=confidence_interval,
            confidence_score=confidence_score,
            time_horizon_months=time_horizon_months,
            predicted_monthly_revenue=profile.monthly_recurring_revenue,
            predicted_retention_months=int(expected_lifetime),
            churn_probability=churn_pred.churn_probability,
            expansion_potential=expansion_potential,
            factors={
                "base_ltv": float(base_ltv),
                "expansion_ltv": float(expansion_potential),
                "current_mrr": float(profile.monthly_recurring_revenue),
                "tenure_months": (datetime.utcnow() - profile.signup_date).days // 30,
                "engagement_score": profile.engagement_score,
                "tier": profile.tier.value,
                "discount_rate": self.discount_rate
            }
        )

        self.ltv_predictions[customer_id] = prediction
        return prediction

    def predict_churn(self, customer_id: str) -> ChurnPrediction:
        """
        Predict customer churn probability.

        Analyzes multiple risk factors:
        - Activity patterns
        - Usage trends
        - Engagement metrics
        - Support interactions
        - Payment history

        Args:
            customer_id: Customer identifier

        Returns:
            ChurnPrediction with risk factors and recommendations
        """
        if customer_id not in self.customer_profiles:
            raise ValueError(f"Customer {customer_id} not found")

        profile = self.customer_profiles[customer_id]

        # Calculate risk factors
        risk_factors = self._calculate_churn_risk_factors(profile)

        # Calculate weighted churn probability
        churn_probability = sum(
            risk_factors.get(factor, 0.0) * weight
            for factor, weight in self.churn_model_weights.items()
        )

        # Clip to 0-1 range
        churn_probability = max(0.0, min(1.0, churn_probability))

        # Determine risk level
        if churn_probability < 0.1:
            risk_level = ChurnRisk.LOW
        elif churn_probability < 0.3:
            risk_level = ChurnRisk.MEDIUM
        elif churn_probability < 0.6:
            risk_level = ChurnRisk.HIGH
        else:
            risk_level = ChurnRisk.CRITICAL

        # Estimate days to churn
        days_to_churn = None
        if churn_probability > 0.1:
            # Simplified: inverse of churn probability * 30
            days_to_churn = int((1.0 / churn_probability) * 30)

        # Identify primary risk factors
        primary_risks = sorted(
            [{"factor": k, "score": v} for k, v in risk_factors.items()],
            key=lambda x: x["score"],
            reverse=True
        )[:3]

        # Generate recommendations
        recommendations = self._generate_retention_recommendations(
            risk_factors, risk_level
        )

        # Calculate prevention value (potential revenue loss)
        prevention_value = profile.monthly_recurring_revenue * Decimal(
            str(12)  # Assume 1 year of revenue at risk
        )

        prediction = ChurnPrediction(
            customer_id=customer_id,
            churn_probability=churn_probability,
            risk_level=risk_level,
            days_to_churn=days_to_churn,
            primary_risk_factors=primary_risks,
            recommended_actions=recommendations,
            prevention_value=prevention_value
        )

        self.churn_predictions[customer_id] = prediction
        return prediction

    def calculate_health_score(self, customer_id: str) -> HealthMetrics:
        """
        Calculate comprehensive customer health score.

        Components:
        - Usage health (40%)
        - Engagement health (25%)
        - Financial health (20%)
        - Support health (10%)
        - Adoption health (5%)

        Args:
            customer_id: Customer identifier

        Returns:
            HealthMetrics with detailed breakdown
        """
        if customer_id not in self.customer_profiles:
            raise ValueError(f"Customer {customer_id} not found")

        profile = self.customer_profiles[customer_id]

        # Calculate component scores
        component_scores = {
            "usage": self._calculate_usage_health(profile),
            "engagement": self._calculate_engagement_health(profile),
            "financial": self._calculate_financial_health(profile),
            "support": self._calculate_support_health(profile),
            "adoption": self._calculate_adoption_health(profile)
        }

        # Calculate weighted overall score
        weights = {
            "usage": 0.40,
            "engagement": 0.25,
            "financial": 0.20,
            "support": 0.10,
            "adoption": 0.05
        }

        overall_score = sum(
            component_scores[component] * weight
            for component, weight in weights.items()
        )

        # Determine health status
        if overall_score >= 80:
            health_status = HealthScore.EXCELLENT
        elif overall_score >= 60:
            health_status = HealthScore.GOOD
        elif overall_score >= 40:
            health_status = HealthScore.FAIR
        elif overall_score >= 20:
            health_status = HealthScore.POOR
        else:
            health_status = HealthScore.CRITICAL

        # Calculate trend
        trend = self._calculate_health_trend(customer_id, overall_score)

        # Generate alerts
        alerts = self._generate_health_alerts(component_scores, health_status)

        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            component_scores, health_status
        )

        metrics = HealthMetrics(
            customer_id=customer_id,
            overall_score=overall_score,
            health_status=health_status,
            component_scores=component_scores,
            trend=trend,
            alerts=alerts,
            recommendations=recommendations
        )

        self.health_scores[customer_id] = metrics
        return metrics

    def analyze_cohort(
        self,
        cohort_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> CohortAnalysis:
        """
        Perform cohort analysis for customers.

        Args:
            cohort_name: Name of cohort (e.g., "Q4_2025")
            start_date: Cohort start date
            end_date: Cohort end date

        Returns:
            CohortAnalysis with retention and revenue metrics
        """
        # Filter customers in cohort
        cohort_customers = [
            profile for profile in self.customer_profiles.values()
            if start_date <= profile.signup_date < end_date
        ]

        if not cohort_customers:
            raise ValueError(f"No customers found in cohort {cohort_name}")

        # Calculate metrics
        customer_count = len(cohort_customers)
        total_revenue = sum(c.total_revenue for c in cohort_customers)
        average_ltv = total_revenue / customer_count

        # Calculate retention and churn rates by month
        max_months = 24  # Analyze up to 24 months
        retention_rates = {}
        churn_rates = {}
        revenue_by_month = {}

        for month in range(max_months):
            month_date = start_date + timedelta(days=30 * month)
            if month_date > datetime.utcnow():
                break

            # Count active customers in this month
            active_count = sum(
                1 for c in cohort_customers
                if c.last_activity_date >= month_date
            )

            retention_rate = (active_count / customer_count) * 100
            churn_rate = 100 - retention_rate

            retention_rates[month] = retention_rate
            churn_rates[month] = churn_rate

            # Calculate revenue for this month
            month_revenue = sum(
                c.monthly_recurring_revenue for c in cohort_customers
                if c.last_activity_date >= month_date
            )
            revenue_by_month[month] = month_revenue

        analysis = CohortAnalysis(
            cohort_name=cohort_name,
            cohort_start_date=start_date,
            customer_count=customer_count,
            total_revenue=total_revenue,
            average_ltv=average_ltv,
            retention_rates=retention_rates,
            churn_rates=churn_rates,
            revenue_by_month=revenue_by_month,
            metadata={
                "cohort_end_date": end_date.isoformat(),
                "analysis_date": datetime.utcnow().isoformat(),
                "months_analyzed": len(retention_rates)
            }
        )

        self.cohorts[cohort_name] = analysis
        return analysis

    def register_customer(self, profile: CustomerProfile) -> None:
        """Register or update customer profile"""
        self.customer_profiles[profile.customer_id] = profile

    # Helper methods

    def _calculate_discounted_revenue(
        self,
        monthly_revenue: Decimal,
        months: int,
        churn_rate: float
    ) -> Decimal:
        """Calculate NPV of future revenue stream"""
        total = Decimal("0")
        retention_probability = 1.0

        for month in range(months):
            # Apply churn and discount
            discounted_value = (
                monthly_revenue *
                Decimal(str(retention_probability)) *
                Decimal(str(1 / (1 + self.discount_rate) ** month))
            )
            total += discounted_value

            # Update retention probability
            retention_probability *= (1 - churn_rate)

        return total

    def _calculate_expansion_potential(self, profile: CustomerProfile) -> Decimal:
        """Calculate potential expansion revenue"""
        # Simplified: based on current tier and usage
        tier_multipliers = {
            CustomerTier.FREE: 5.0,
            CustomerTier.STARTER: 3.0,
            CustomerTier.PROFESSIONAL: 2.0,
            CustomerTier.BUSINESS: 1.5,
            CustomerTier.ENTERPRISE: 1.2
        }

        multiplier = tier_multipliers.get(profile.tier, 1.0)

        # High usage indicates expansion potential
        usage_factor = profile.usage_metrics.get("utilization", 0.5)
        if usage_factor > 0.8:
            multiplier *= 1.5

        expansion = profile.monthly_recurring_revenue * Decimal(str(multiplier - 1.0))
        return max(Decimal("0"), expansion)

    def _calculate_ltv_variance(
        self,
        profile: CustomerProfile,
        churn_pred: ChurnPrediction
    ) -> float:
        """Calculate variance in LTV prediction"""
        # Simplified variance calculation
        base_variance = float(profile.monthly_recurring_revenue) ** 2
        churn_variance = churn_pred.churn_probability * (1 - churn_pred.churn_probability)

        return base_variance * churn_variance * 12  # Annual variance

    def _calculate_prediction_confidence(
        self,
        profile: CustomerProfile,
        churn_pred: ChurnPrediction,
        expected_lifetime: float
    ) -> float:
        """Calculate confidence in LTV prediction"""
        # Factors affecting confidence
        tenure_months = (datetime.utcnow() - profile.signup_date).days // 30

        factors = {
            "tenure": min(1.0, tenure_months / 12),  # More history = higher confidence
            "churn_confidence": 1.0 - churn_pred.churn_probability,
            "engagement": profile.engagement_score / 100,
            "data_quality": 0.8  # Would be calculated from data completeness
        }

        return sum(factors.values()) / len(factors)

    def _calculate_churn_risk_factors(
        self,
        profile: CustomerProfile
    ) -> Dict[str, float]:
        """Calculate individual churn risk factors"""
        now = datetime.utcnow()
        days_inactive = (now - profile.last_activity_date).days

        return {
            "activity_decay": min(1.0, days_inactive / 30),  # Normalize to 0-1
            "usage_decline": max(0.0, 1.0 - profile.usage_metrics.get("trend", 0.5)),
            "engagement_drop": max(0.0, 1.0 - profile.engagement_score / 100),
            "support_spike": min(1.0, profile.support_tickets / 10),
            "payment_issues": profile.metadata.get("payment_failures", 0) * 0.2,
            "feature_adoption": max(0.0, 1.0 - profile.usage_metrics.get("feature_adoption", 0.5))
        }

    def _generate_retention_recommendations(
        self,
        risk_factors: Dict[str, float],
        risk_level: ChurnRisk
    ) -> List[str]:
        """Generate retention recommendations based on risk factors"""
        recommendations = []

        if risk_factors.get("activity_decay", 0) > 0.5:
            recommendations.append("Schedule check-in call to understand reduced activity")
            recommendations.append("Send product update emails highlighting new features")

        if risk_factors.get("usage_decline", 0) > 0.5:
            recommendations.append("Offer training session or demo of underutilized features")
            recommendations.append("Provide usage optimization consultation")

        if risk_factors.get("engagement_drop", 0) > 0.5:
            recommendations.append("Trigger re-engagement campaign")
            recommendations.append("Offer personalized content based on use case")

        if risk_factors.get("support_spike", 0) > 0.5:
            recommendations.append("Assign dedicated support manager")
            recommendations.append("Review and address recurring support issues")

        if risk_level == ChurnRisk.CRITICAL:
            recommendations.append("⚠️ URGENT: Executive-level outreach required")
            recommendations.append("Consider retention incentive or discount")

        return recommendations or ["Continue monitoring - no immediate action needed"]

    def _calculate_usage_health(self, profile: CustomerProfile) -> float:
        """Calculate usage health score (0-100)"""
        utilization = profile.usage_metrics.get("utilization", 0.5)
        frequency = profile.usage_metrics.get("frequency", 0.5)
        return (utilization * 60 + frequency * 40)

    def _calculate_engagement_health(self, profile: CustomerProfile) -> float:
        """Calculate engagement health score (0-100)"""
        return profile.engagement_score

    def _calculate_financial_health(self, profile: CustomerProfile) -> float:
        """Calculate financial health score (0-100)"""
        payment_success = 1.0 - min(1.0, profile.metadata.get("payment_failures", 0) * 0.2)
        mrr_growth = profile.metadata.get("mrr_growth", 0.0)

        return (payment_success * 70 + min(100, (1 + mrr_growth) * 30))

    def _calculate_support_health(self, profile: CustomerProfile) -> float:
        """Calculate support health score (0-100)"""
        # Lower support tickets = better health
        ticket_penalty = min(100, profile.support_tickets * 10)
        return max(0, 100 - ticket_penalty)

    def _calculate_adoption_health(self, profile: CustomerProfile) -> float:
        """Calculate feature adoption health score (0-100)"""
        return profile.usage_metrics.get("feature_adoption", 0.5) * 100

    def _calculate_health_trend(self, customer_id: str, current_score: float) -> str:
        """Calculate health score trend"""
        # Would compare with historical scores
        previous_score = self.health_scores.get(customer_id)
        if previous_score:
            diff = current_score - previous_score.overall_score
            if diff > 5:
                return "improving"
            elif diff < -5:
                return "declining"
        return "stable"

    def _generate_health_alerts(
        self,
        component_scores: Dict[str, float],
        health_status: HealthScore
    ) -> List[str]:
        """Generate health alerts"""
        alerts = []

        if health_status in [HealthScore.POOR, HealthScore.CRITICAL]:
            alerts.append("⚠️ Overall health is poor - immediate attention required")

        for component, score in component_scores.items():
            if score < 30:
                alerts.append(f"❌ {component.title()} health critical: {score:.1f}/100")
            elif score < 50:
                alerts.append(f"⚠️ {component.title()} health needs attention: {score:.1f}/100")

        return alerts

    def _generate_health_recommendations(
        self,
        component_scores: Dict[str, float],
        health_status: HealthScore
    ) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        if component_scores.get("usage", 100) < 60:
            recommendations.append("Increase usage through training and feature discovery")

        if component_scores.get("engagement", 100) < 60:
            recommendations.append("Boost engagement with personalized content and campaigns")

        if component_scores.get("financial", 100) < 60:
            recommendations.append("Address payment issues and optimize billing")

        if component_scores.get("support", 100) < 60:
            recommendations.append("Reduce support burden through better onboarding and documentation")

        return recommendations or ["Continue current strategy - health is good"]
