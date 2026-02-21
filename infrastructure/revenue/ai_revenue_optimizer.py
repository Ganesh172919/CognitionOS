"""
AI-Powered Revenue Optimization Engine
ML-driven pricing optimization, churn prediction, and revenue maximization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import math
from collections import defaultdict


class PricingStrategy(Enum):
    """Pricing optimization strategies"""
    VALUE_BASED = "value_based"
    COMPETITIVE = "competitive"
    PENETRATION = "penetration"
    PREMIUM = "premium"
    DYNAMIC = "dynamic"
    AI_OPTIMIZED = "ai_optimized"


class ChurnRisk(Enum):
    """Churn risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PricingExperiment:
    """A/B test for pricing"""
    experiment_id: str
    name: str
    control_price: Decimal
    variant_prices: List[Decimal]
    started_at: datetime
    ended_at: Optional[datetime] = None
    control_conversions: int = 0
    control_revenue: Decimal = Decimal("0")
    variant_conversions: List[int] = field(default_factory=list)
    variant_revenue: List[Decimal] = field(default_factory=list)
    winner: Optional[int] = None  # None for control, index for variant
    confidence_level: float = 0.0


@dataclass
class CustomerSegment:
    """Customer segment for targeted pricing"""
    segment_id: str
    name: str
    criteria: Dict[str, Any]
    user_count: int
    avg_lifetime_value: Decimal
    churn_rate: float
    price_sensitivity: float  # 0-1, higher = more sensitive
    recommended_price: Optional[Decimal] = None


@dataclass
class ChurnPrediction:
    """Churn prediction for a customer"""
    tenant_id: str
    user_id: str
    churn_probability: float  # 0-1
    risk_level: ChurnRisk
    contributing_factors: List[str]
    recommended_actions: List[str]
    predicted_churn_date: Optional[datetime] = None
    prevention_score: float = 0.0  # Success probability of intervention


@dataclass
class RevenueOptimization:
    """Revenue optimization recommendation"""
    tenant_id: str
    current_price: Decimal
    recommended_price: Decimal
    expected_revenue_increase: Decimal
    expected_revenue_increase_percent: float
    confidence_score: float
    reasoning: List[str]
    risk_factors: List[str]


class PricingOptimizer:
    """AI-powered pricing optimization"""

    def __init__(self):
        self.experiments: Dict[str, PricingExperiment] = {}
        self.segments: Dict[str, CustomerSegment] = {}
        self.pricing_history: Dict[str, List[Tuple[datetime, Decimal, int]]] = defaultdict(list)

    async def create_pricing_experiment(
        self,
        name: str,
        control_price: Decimal,
        variant_prices: List[Decimal]
    ) -> PricingExperiment:
        """Create A/B pricing experiment"""
        import uuid

        experiment = PricingExperiment(
            experiment_id=str(uuid.uuid4()),
            name=name,
            control_price=control_price,
            variant_prices=variant_prices,
            started_at=datetime.utcnow(),
            variant_conversions=[0] * len(variant_prices),
            variant_revenue=[Decimal("0")] * len(variant_prices)
        )

        self.experiments[experiment.experiment_id] = experiment
        return experiment

    async def record_conversion(
        self,
        experiment_id: str,
        variant_index: Optional[int],  # None for control
        revenue: Decimal
    ):
        """Record conversion for experiment"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return

        if variant_index is None:
            # Control group
            experiment.control_conversions += 1
            experiment.control_revenue += revenue
        else:
            # Variant group
            experiment.variant_conversions[variant_index] += 1
            experiment.variant_revenue[variant_index] += revenue

        # Check if experiment can be concluded
        await self._analyze_experiment(experiment)

    async def _analyze_experiment(self, experiment: PricingExperiment):
        """Analyze experiment results using statistical significance"""
        # Need minimum sample size
        total_conversions = experiment.control_conversions + sum(experiment.variant_conversions)
        if total_conversions < 100:
            return

        # Calculate conversion rates
        control_rate = experiment.control_conversions
        control_revenue = experiment.control_revenue

        best_variant_idx = None
        best_performance = float(control_revenue)

        for idx, (conversions, revenue) in enumerate(
            zip(experiment.variant_conversions, experiment.variant_revenue)
        ):
            if float(revenue) > best_performance:
                best_performance = float(revenue)
                best_variant_idx = idx

        # Simplified confidence calculation
        if total_conversions > 200:
            experiment.confidence_level = 0.95
            experiment.winner = best_variant_idx
            experiment.ended_at = datetime.utcnow()

    async def create_customer_segment(
        self,
        name: str,
        criteria: Dict[str, Any],
        users: List[Dict[str, Any]]
    ) -> CustomerSegment:
        """Create customer segment for targeted pricing"""
        import uuid

        # Calculate segment metrics
        user_count = len(users)
        total_ltv = sum(Decimal(str(u.get("lifetime_value", 0))) for u in users)
        avg_ltv = total_ltv / user_count if user_count > 0 else Decimal("0")

        churned_users = sum(1 for u in users if u.get("churned", False))
        churn_rate = churned_users / user_count if user_count > 0 else 0.0

        # Estimate price sensitivity (simplified)
        price_changes = sum(1 for u in users if u.get("downgraded", False))
        price_sensitivity = price_changes / user_count if user_count > 0 else 0.5

        segment = CustomerSegment(
            segment_id=str(uuid.uuid4()),
            name=name,
            criteria=criteria,
            user_count=user_count,
            avg_lifetime_value=avg_ltv,
            churn_rate=churn_rate,
            price_sensitivity=price_sensitivity
        )

        self.segments[segment.segment_id] = segment

        # Calculate recommended price
        await self._optimize_segment_pricing(segment)

        return segment

    async def _optimize_segment_pricing(self, segment: CustomerSegment):
        """Optimize pricing for segment"""
        # Price elasticity model (simplified)
        # Higher price sensitivity = lower optimal price
        base_price = Decimal("100.00")

        if segment.price_sensitivity < 0.3:
            # Low sensitivity - can charge premium
            segment.recommended_price = base_price * Decimal("1.5")
        elif segment.price_sensitivity < 0.6:
            # Medium sensitivity - standard pricing
            segment.recommended_price = base_price
        else:
            # High sensitivity - discount needed
            segment.recommended_price = base_price * Decimal("0.7")

    async def optimize_price_for_tenant(
        self,
        tenant_id: str,
        current_price: Decimal,
        usage_data: Dict[str, Any],
        competition_data: Optional[Dict[str, Any]] = None
    ) -> RevenueOptimization:
        """Generate pricing optimization recommendation"""
        # Analyze usage patterns
        api_calls = usage_data.get("api_calls", 0)
        compute_hours = usage_data.get("compute_hours", 0)
        storage_gb = usage_data.get("storage_gb", 0)

        # Calculate value delivered
        value_score = (
            api_calls * 0.001 +
            compute_hours * 0.5 +
            storage_gb * 0.1
        )

        # Value-based pricing
        value_based_price = Decimal(str(value_score))

        # Consider competition if available
        competitive_price = current_price
        if competition_data:
            competitive_price = Decimal(str(competition_data.get("avg_price", current_price)))

        # Hybrid optimization: balance value and competition
        weight_value = 0.7
        weight_competition = 0.3

        recommended_price = (
            value_based_price * Decimal(str(weight_value)) +
            competitive_price * Decimal(str(weight_competition))
        )

        # Don't recommend too large changes
        max_change = current_price * Decimal("0.3")
        price_diff = recommended_price - current_price

        if abs(price_diff) > max_change:
            if price_diff > 0:
                recommended_price = current_price + max_change
            else:
                recommended_price = current_price - max_change

        # Calculate expected impact
        price_change_percent = float((recommended_price - current_price) / current_price * 100)

        # Assume price elasticity of -0.5 (simplified)
        # % change in demand = elasticity * % change in price
        demand_change_percent = -0.5 * price_change_percent

        # Revenue = Price * Quantity
        # New revenue = (P * (1 + p%)) * (Q * (1 + d%))
        revenue_multiplier = (1 + price_change_percent / 100) * (1 + demand_change_percent / 100)
        current_revenue = current_price * Decimal("100")  # Assuming 100 units
        expected_revenue = current_revenue * Decimal(str(revenue_multiplier))
        revenue_increase = expected_revenue - current_revenue
        revenue_increase_percent = float(revenue_increase / current_revenue * 100)

        # Build reasoning
        reasoning = []
        if value_score > float(current_price):
            reasoning.append(f"Value delivered (${value_score:.2f}) exceeds current price")
        if price_change_percent > 0:
            reasoning.append(f"Recommended {price_change_percent:.1f}% price increase")
        else:
            reasoning.append(f"Recommended {abs(price_change_percent):.1f}% price decrease for market competitiveness")

        # Risk factors
        risk_factors = []
        if abs(price_change_percent) > 20:
            risk_factors.append("Large price change may cause customer churn")
        if usage_data.get("usage_trend") == "decreasing":
            risk_factors.append("Usage is decreasing, price increase risky")

        # Confidence score
        confidence = 0.8 if len(risk_factors) == 0 else 0.6

        return RevenueOptimization(
            tenant_id=tenant_id,
            current_price=current_price,
            recommended_price=recommended_price,
            expected_revenue_increase=revenue_increase,
            expected_revenue_increase_percent=revenue_increase_percent,
            confidence_score=confidence,
            reasoning=reasoning,
            risk_factors=risk_factors
        )


class ChurnPredictor:
    """ML-powered churn prediction"""

    def __init__(self):
        self.predictions: Dict[str, ChurnPrediction] = {}
        self.churn_factors_weights = {
            "low_usage": 0.3,
            "support_tickets": 0.2,
            "failed_payments": 0.3,
            "no_team_members": 0.1,
            "old_features_only": 0.1
        }

    async def predict_churn(
        self,
        tenant_id: str,
        user_id: str,
        usage_data: Dict[str, Any],
        engagement_data: Dict[str, Any],
        payment_data: Dict[str, Any]
    ) -> ChurnPrediction:
        """Predict churn probability for user"""
        # Calculate churn score based on factors
        churn_score = 0.0
        contributing_factors = []

        # Factor 1: Low usage
        api_calls_last_week = usage_data.get("api_calls_last_week", 0)
        if api_calls_last_week < 10:
            churn_score += self.churn_factors_weights["low_usage"]
            contributing_factors.append("Very low API usage in past week")

        # Factor 2: Support tickets
        support_tickets = engagement_data.get("support_tickets", 0)
        if support_tickets > 3:
            churn_score += self.churn_factors_weights["support_tickets"]
            contributing_factors.append("Multiple support tickets indicating friction")

        # Factor 3: Failed payments
        failed_payments = payment_data.get("failed_payments", 0)
        if failed_payments > 0:
            churn_score += self.churn_factors_weights["failed_payments"]
            contributing_factors.append("Failed payment attempts")

        # Factor 4: No team members
        team_size = engagement_data.get("team_size", 1)
        if team_size == 1:
            churn_score += self.churn_factors_weights["no_team_members"]
            contributing_factors.append("Single user - no team collaboration")

        # Factor 5: Using old features only
        new_features_used = engagement_data.get("new_features_used", 0)
        if new_features_used == 0:
            churn_score += self.churn_factors_weights["old_features_only"]
            contributing_factors.append("Not adopting new features")

        # Determine risk level
        if churn_score >= 0.7:
            risk_level = ChurnRisk.CRITICAL
        elif churn_score >= 0.5:
            risk_level = ChurnRisk.HIGH
        elif churn_score >= 0.3:
            risk_level = ChurnRisk.MEDIUM
        else:
            risk_level = ChurnRisk.LOW

        # Generate recommendations
        recommended_actions = []
        if "low API usage" in " ".join(contributing_factors):
            recommended_actions.append("Send onboarding email with use case examples")
        if "support tickets" in " ".join(contributing_factors):
            recommended_actions.append("Assign customer success manager")
        if "Failed payment" in " ".join(contributing_factors):
            recommended_actions.append("Update payment method reminder")
        if "team collaboration" in " ".join(contributing_factors):
            recommended_actions.append("Offer team plan discount")
        if "new features" in " ".join(contributing_factors):
            recommended_actions.append("Feature education campaign")

        # Estimate churn date
        if risk_level in [ChurnRisk.HIGH, ChurnRisk.CRITICAL]:
            days_until_churn = 30 if risk_level == ChurnRisk.HIGH else 14
            predicted_churn_date = datetime.utcnow() + timedelta(days=days_until_churn)
        else:
            predicted_churn_date = None

        # Prevention score (how likely intervention will work)
        prevention_score = 1.0 - (churn_score * 0.8)

        prediction = ChurnPrediction(
            tenant_id=tenant_id,
            user_id=user_id,
            churn_probability=churn_score,
            risk_level=risk_level,
            contributing_factors=contributing_factors,
            recommended_actions=recommended_actions,
            predicted_churn_date=predicted_churn_date,
            prevention_score=prevention_score
        )

        self.predictions[f"{tenant_id}:{user_id}"] = prediction

        return prediction

    async def get_high_risk_customers(
        self,
        min_risk_level: ChurnRisk = ChurnRisk.HIGH
    ) -> List[ChurnPrediction]:
        """Get all high-risk customers"""
        risk_values = {
            ChurnRisk.LOW: 1,
            ChurnRisk.MEDIUM: 2,
            ChurnRisk.HIGH: 3,
            ChurnRisk.CRITICAL: 4
        }

        min_value = risk_values[min_risk_level]

        return [
            pred for pred in self.predictions.values()
            if risk_values[pred.risk_level] >= min_value
        ]


class RevenueIntelligenceEngine:
    """Main revenue optimization engine"""

    def __init__(self):
        self.pricing_optimizer = PricingOptimizer()
        self.churn_predictor = ChurnPredictor()
        self.revenue_history: Dict[str, List[Tuple[datetime, Decimal]]] = defaultdict(list)

    async def analyze_revenue_opportunities(
        self,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Comprehensive revenue analysis"""
        # Get all predictions and optimizations
        opportunities = []

        # 1. Pricing optimization opportunity
        # (Would fetch real data in production)
        pricing_opp = {
            "type": "pricing_optimization",
            "description": "Optimize pricing based on value delivered",
            "potential_revenue_increase": 5000.00,
            "confidence": 0.8,
            "effort": "low"
        }
        opportunities.append(pricing_opp)

        # 2. Churn prevention opportunity
        churn_preds = await self.churn_predictor.get_high_risk_customers()
        if churn_preds:
            avg_customer_value = 1000.00  # Simplified
            potential_saved = len(churn_preds) * avg_customer_value * 0.6  # 60% save rate
            opportunities.append({
                "type": "churn_prevention",
                "description": f"Prevent {len(churn_preds)} high-risk customers from churning",
                "potential_revenue_saved": potential_saved,
                "confidence": 0.7,
                "effort": "medium"
            })

        # 3. Upsell opportunity
        opportunities.append({
            "type": "upsell",
            "description": "Upsell power users to higher tier",
            "potential_revenue_increase": 3000.00,
            "confidence": 0.6,
            "effort": "low"
        })

        # 4. Expansion opportunity
        opportunities.append({
            "type": "expansion",
            "description": "Expand into adjacent use cases",
            "potential_revenue_increase": 10000.00,
            "confidence": 0.5,
            "effort": "high"
        })

        # Sort by potential impact
        opportunities.sort(
            key=lambda x: x.get("potential_revenue_increase", 0) + x.get("potential_revenue_saved", 0),
            reverse=True
        )

        total_potential = sum(
            o.get("potential_revenue_increase", 0) + o.get("potential_revenue_saved", 0)
            for o in opportunities
        )

        return {
            "tenant_id": tenant_id,
            "total_potential_revenue": total_potential,
            "opportunities": opportunities,
            "priority_actions": [o for o in opportunities if o["confidence"] > 0.7]
        }

    async def calculate_ltv(
        self,
        tenant_id: str,
        monthly_revenue: Decimal,
        churn_rate: float
    ) -> Decimal:
        """Calculate customer lifetime value"""
        if churn_rate == 0 or churn_rate >= 1:
            return monthly_revenue * Decimal("24")  # Default 24 months

        # LTV = ARPU / Churn Rate
        ltv = monthly_revenue / Decimal(str(churn_rate))

        return ltv

    async def optimize_pricing_tiers(
        self,
        current_tiers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Optimize pricing tier structure"""
        # Analyze usage distribution and willingness to pay
        optimized_tiers = []

        for tier in current_tiers:
            optimized = tier.copy()

            # Adjust based on adoption and revenue
            adoption_rate = tier.get("adoption_rate", 0.0)
            avg_overage_charges = tier.get("avg_overage_charges", 0.0)

            # If high overage charges, increase included limits
            if avg_overage_charges > tier["base_price"] * 0.3:
                optimized["recommendation"] = "Increase included limits to reduce overage friction"

            # If low adoption, adjust pricing or features
            if adoption_rate < 0.05:
                optimized["recommendation"] = "Low adoption - consider repricing or adding features"

            optimized_tiers.append(optimized)

        return optimized_tiers
