"""
Dynamic Pricing Engine - Advanced Revenue Optimization System

Implements ML-based dynamic pricing with:
- Real-time demand forecasting
- Competitive pricing analysis
- Customer segmentation-based pricing
- Revenue maximization algorithms
- A/B testing for pricing strategies
- Price elasticity modeling
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import math
from decimal import Decimal


class PricingStrategy(Enum):
    """Pricing strategy types"""
    VALUE_BASED = "value_based"
    COMPETITIVE = "competitive"
    COST_PLUS = "cost_plus"
    DYNAMIC = "dynamic"
    PENETRATION = "penetration"
    PREMIUM = "premium"
    FREEMIUM = "freemium"


class CustomerSegment(Enum):
    """Customer segmentation categories"""
    STARTUP = "startup"
    SMB = "smb"
    ENTERPRISE = "enterprise"
    STRATEGIC = "strategic"
    HIGH_VOLUME = "high_volume"
    TRIAL = "trial"


@dataclass
class PricePoint:
    """Individual price point with metadata"""
    base_price: Decimal
    discount_percentage: Decimal
    final_price: Decimal
    currency: str
    effective_from: datetime
    effective_until: Optional[datetime]
    strategy: PricingStrategy
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemandForecast:
    """Demand forecasting result"""
    period_start: datetime
    period_end: datetime
    predicted_demand: float
    lower_bound: float
    upper_bound: float
    confidence_interval: float
    trend: str  # "increasing", "stable", "decreasing"
    seasonality_factor: float
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class PriceElasticity:
    """Price elasticity analysis"""
    product_id: str
    current_price: Decimal
    price_change_percentage: float
    demand_change_percentage: float
    elasticity_coefficient: float
    is_elastic: bool  # True if |elasticity| > 1
    optimal_price: Decimal
    revenue_impact: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class CompetitivePricing:
    """Competitive pricing analysis"""
    competitor_id: str
    product_name: str
    price: Decimal
    features: List[str]
    market_position: str
    last_updated: datetime
    price_difference: Decimal
    recommendation: str


class DynamicPricingEngine:
    """
    Advanced dynamic pricing engine with ML-based optimization.

    Features:
    - Real-time price adjustments based on demand
    - Competitor price monitoring
    - Customer segment-based pricing
    - Revenue maximization algorithms
    - Price elasticity modeling
    - A/B testing framework
    """

    def __init__(
        self,
        base_prices: Dict[str, Decimal],
        min_margin_percentage: float = 20.0,
        max_discount_percentage: float = 50.0,
        price_update_frequency: int = 3600  # seconds
    ):
        self.base_prices = base_prices
        self.min_margin_percentage = min_margin_percentage
        self.max_discount_percentage = max_discount_percentage
        self.price_update_frequency = price_update_frequency

        # Price history and analytics
        self.price_history: Dict[str, List[PricePoint]] = {}
        self.demand_forecasts: Dict[str, DemandForecast] = {}
        self.elasticity_models: Dict[str, PriceElasticity] = {}
        self.competitor_prices: Dict[str, List[CompetitivePricing]] = {}

        # A/B testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}

        # Revenue metrics
        self.revenue_metrics: Dict[str, Dict[str, Any]] = {
            "daily": {},
            "weekly": {},
            "monthly": {}
        }

    def calculate_dynamic_price(
        self,
        product_id: str,
        customer_segment: CustomerSegment,
        current_demand: float,
        time_of_day: Optional[datetime] = None
    ) -> PricePoint:
        """
        Calculate dynamic price based on multiple factors.

        Args:
            product_id: Product identifier
            customer_segment: Customer segment category
            current_demand: Current demand level (0.0-1.0)
            time_of_day: Time for time-based pricing

        Returns:
            PricePoint with calculated price and metadata
        """
        if product_id not in self.base_prices:
            raise ValueError(f"Product {product_id} not found in base prices")

        base_price = self.base_prices[product_id]
        time_of_day = time_of_day or datetime.utcnow()

        # Factor 1: Demand-based pricing
        demand_multiplier = self._calculate_demand_multiplier(current_demand)

        # Factor 2: Segment-based pricing
        segment_multiplier = self._get_segment_multiplier(customer_segment)

        # Factor 3: Time-based pricing
        time_multiplier = self._calculate_time_multiplier(time_of_day)

        # Factor 4: Competitive positioning
        competitive_adjustment = self._get_competitive_adjustment(product_id)

        # Factor 5: Inventory/capacity-based pricing
        capacity_multiplier = self._calculate_capacity_multiplier(product_id)

        # Combine all factors
        total_multiplier = (
            demand_multiplier *
            segment_multiplier *
            time_multiplier *
            capacity_multiplier
        ) + competitive_adjustment

        # Calculate final price with constraints
        adjusted_price = base_price * Decimal(str(total_multiplier))

        # Apply minimum margin constraint
        min_price = base_price * Decimal(str(1.0 - self.max_discount_percentage / 100))
        final_price = max(adjusted_price, min_price)

        # Calculate discount
        discount = ((base_price - final_price) / base_price) * 100

        price_point = PricePoint(
            base_price=base_price,
            discount_percentage=Decimal(str(abs(discount))),
            final_price=final_price,
            currency="USD",
            effective_from=time_of_day,
            effective_until=time_of_day + timedelta(seconds=self.price_update_frequency),
            strategy=PricingStrategy.DYNAMIC,
            confidence_score=self._calculate_confidence_score(
                demand_multiplier, segment_multiplier, time_multiplier
            ),
            metadata={
                "demand_multiplier": demand_multiplier,
                "segment_multiplier": segment_multiplier,
                "time_multiplier": time_multiplier,
                "competitive_adjustment": competitive_adjustment,
                "capacity_multiplier": capacity_multiplier,
                "customer_segment": customer_segment.value
            }
        )

        # Store in history
        if product_id not in self.price_history:
            self.price_history[product_id] = []
        self.price_history[product_id].append(price_point)

        return price_point

    def forecast_demand(
        self,
        product_id: str,
        lookback_days: int = 30,
        forecast_days: int = 7
    ) -> DemandForecast:
        """
        Forecast future demand using historical patterns.

        Implements simplified time series forecasting with:
        - Trend analysis
        - Seasonality detection
        - Moving averages
        """
        # Get historical data
        history = self.price_history.get(product_id, [])
        if len(history) < 7:
            # Not enough data, return neutral forecast
            now = datetime.utcnow()
            return DemandForecast(
                period_start=now,
                period_end=now + timedelta(days=forecast_days),
                predicted_demand=0.5,
                lower_bound=0.3,
                upper_bound=0.7,
                confidence_interval=0.4,
                trend="stable",
                seasonality_factor=1.0,
                factors={"insufficient_data": True}
            )

        # Calculate trend
        recent_prices = [float(p.final_price) for p in history[-lookback_days:]]
        trend = self._calculate_trend(recent_prices)

        # Calculate seasonality
        seasonality_factor = self._calculate_seasonality(history)

        # Simple forecasting: weighted moving average
        weights = [i + 1 for i in range(min(7, len(recent_prices)))]
        weighted_avg = sum(
            p * w for p, w in zip(recent_prices[-7:], weights)
        ) / sum(weights)

        # Normalize to 0-1 range for demand
        demand_index = min(1.0, weighted_avg / float(self.base_prices.get(product_id, Decimal("100"))))

        # Calculate confidence interval
        variance = sum((p - weighted_avg) ** 2 for p in recent_prices) / len(recent_prices)
        std_dev = math.sqrt(variance)
        confidence_interval = 1.96 * std_dev / math.sqrt(len(recent_prices))

        now = datetime.utcnow()
        forecast = DemandForecast(
            period_start=now,
            period_end=now + timedelta(days=forecast_days),
            predicted_demand=demand_index,
            lower_bound=max(0.0, demand_index - confidence_interval),
            upper_bound=min(1.0, demand_index + confidence_interval),
            confidence_interval=confidence_interval,
            trend=trend,
            seasonality_factor=seasonality_factor,
            factors={
                "historical_points": len(history),
                "weighted_average": weighted_avg,
                "variance": variance
            }
        )

        self.demand_forecasts[product_id] = forecast
        return forecast

    def analyze_price_elasticity(
        self,
        product_id: str,
        price_changes: List[Tuple[Decimal, float]],  # (price, demand)
    ) -> PriceElasticity:
        """
        Analyze price elasticity and find optimal price point.

        Args:
            product_id: Product identifier
            price_changes: List of (price, demand) tuples from historical data

        Returns:
            PriceElasticity analysis with optimal price recommendation
        """
        if len(price_changes) < 2:
            raise ValueError("Need at least 2 price-demand pairs for elasticity analysis")

        # Sort by price
        sorted_changes = sorted(price_changes, key=lambda x: x[0])

        # Calculate elasticity using midpoint method
        elasticities = []
        for i in range(len(sorted_changes) - 1):
            p1, d1 = sorted_changes[i]
            p2, d2 = sorted_changes[i + 1]

            price_change = ((float(p2) - float(p1)) / ((float(p2) + float(p1)) / 2)) * 100
            demand_change = ((d2 - d1) / ((d2 + d1) / 2)) * 100

            if price_change != 0:
                elasticity = demand_change / price_change
                elasticities.append(elasticity)

        avg_elasticity = sum(elasticities) / len(elasticities) if elasticities else 0.0

        # Find optimal price (maximize revenue = price * demand)
        revenues = [float(price) * demand for price, demand in price_changes]
        optimal_idx = revenues.index(max(revenues))
        optimal_price = price_changes[optimal_idx][0]

        current_price = self.base_prices.get(product_id, sorted_changes[0][0])

        analysis = PriceElasticity(
            product_id=product_id,
            current_price=current_price,
            price_change_percentage=((float(optimal_price) - float(current_price)) / float(current_price)) * 100,
            demand_change_percentage=0.0,  # Would be calculated from actual data
            elasticity_coefficient=avg_elasticity,
            is_elastic=abs(avg_elasticity) > 1.0,
            optimal_price=optimal_price,
            revenue_impact={
                "current_revenue": Decimal(str(revenues[0])),
                "optimal_revenue": Decimal(str(max(revenues))),
                "improvement_percentage": Decimal(str(((max(revenues) - revenues[0]) / revenues[0]) * 100))
            }
        )

        self.elasticity_models[product_id] = analysis
        return analysis

    def monitor_competitor_pricing(
        self,
        competitor_id: str,
        product_name: str,
        price: Decimal,
        features: List[str]
    ) -> CompetitivePricing:
        """
        Monitor and analyze competitor pricing.

        Args:
            competitor_id: Competitor identifier
            product_name: Competitor's product name
            price: Competitor's price
            features: List of features offered

        Returns:
            CompetitivePricing analysis with recommendations
        """
        # Find our equivalent product
        our_price = None
        our_product_id = None
        for pid, base_price in self.base_prices.items():
            if product_name.lower() in pid.lower():
                our_price = base_price
                our_product_id = pid
                break

        if our_price is None:
            our_price = list(self.base_prices.values())[0]  # Default to first product

        price_difference = our_price - price
        percentage_diff = (float(price_difference) / float(price)) * 100

        # Generate recommendation
        if percentage_diff > 20:
            recommendation = "Consider price reduction - significantly higher than competitor"
        elif percentage_diff > 10:
            recommendation = "Monitor closely - moderately higher pricing"
        elif percentage_diff > -10:
            recommendation = "Competitive parity - maintain current pricing"
        elif percentage_diff > -20:
            recommendation = "Premium positioning - justified by features"
        else:
            recommendation = "Consider price increase - underpriced vs competitor"

        competitive_analysis = CompetitivePricing(
            competitor_id=competitor_id,
            product_name=product_name,
            price=price,
            features=features,
            market_position="competitor",
            last_updated=datetime.utcnow(),
            price_difference=price_difference,
            recommendation=recommendation
        )

        if competitor_id not in self.competitor_prices:
            self.competitor_prices[competitor_id] = []
        self.competitor_prices[competitor_id].append(competitive_analysis)

        return competitive_analysis

    def create_ab_test(
        self,
        test_name: str,
        product_id: str,
        variant_a_price: Decimal,
        variant_b_price: Decimal,
        traffic_split: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create A/B test for price comparison.

        Args:
            test_name: Name of the test
            product_id: Product to test
            variant_a_price: Price for variant A (control)
            variant_b_price: Price for variant B (test)
            traffic_split: Percentage of traffic for variant B (0.0-1.0)

        Returns:
            A/B test configuration
        """
        test_config = {
            "test_id": f"ab_{test_name}_{datetime.utcnow().timestamp()}",
            "product_id": product_id,
            "variant_a": {
                "price": variant_a_price,
                "traffic": 1.0 - traffic_split,
                "conversions": 0,
                "revenue": Decimal("0"),
                "impressions": 0
            },
            "variant_b": {
                "price": variant_b_price,
                "traffic": traffic_split,
                "conversions": 0,
                "revenue": Decimal("0"),
                "impressions": 0
            },
            "start_time": datetime.utcnow(),
            "end_time": None,
            "status": "active",
            "winner": None,
            "confidence_level": 0.0
        }

        self.ab_tests[test_name] = test_config
        return test_config

    def get_revenue_forecast(
        self,
        time_period: str = "monthly",
        months_ahead: int = 12
    ) -> Dict[str, Any]:
        """
        Generate revenue forecast based on pricing and demand.

        Args:
            time_period: "daily", "weekly", or "monthly"
            months_ahead: Number of months to forecast

        Returns:
            Revenue forecast with breakdown
        """
        forecasts = {}
        total_revenue = Decimal("0")

        for product_id, base_price in self.base_prices.items():
            # Get demand forecast
            demand_forecast = self.forecast_demand(product_id, forecast_days=30 * months_ahead)

            # Estimate monthly revenue
            avg_daily_units = demand_forecast.predicted_demand * 100  # Assuming 100 units at full demand
            monthly_units = avg_daily_units * 30
            monthly_revenue = base_price * Decimal(str(monthly_units))

            forecasts[product_id] = {
                "base_price": float(base_price),
                "predicted_monthly_units": monthly_units,
                "predicted_monthly_revenue": float(monthly_revenue),
                "demand_trend": demand_forecast.trend,
                "confidence_interval": demand_forecast.confidence_interval
            }

            total_revenue += monthly_revenue * months_ahead

        return {
            "time_period": time_period,
            "forecast_months": months_ahead,
            "product_forecasts": forecasts,
            "total_forecast_revenue": float(total_revenue),
            "currency": "USD",
            "generated_at": datetime.utcnow().isoformat(),
            "confidence_level": "medium"  # Would be calculated from variance
        }

    # Helper methods

    def _calculate_demand_multiplier(self, demand: float) -> float:
        """Calculate price multiplier based on demand (0.0-1.0)"""
        # Sigmoid curve for smooth price adjustment
        # High demand (>0.7) -> increase price up to 30%
        # Low demand (<0.3) -> decrease price up to 30%
        if demand > 0.7:
            return 1.0 + (demand - 0.7) * 1.0  # Up to 30% increase
        elif demand < 0.3:
            return 1.0 - (0.3 - demand) * 1.0  # Up to 30% decrease
        return 1.0

    def _get_segment_multiplier(self, segment: CustomerSegment) -> float:
        """Get pricing multiplier for customer segment"""
        multipliers = {
            CustomerSegment.STARTUP: 0.7,      # 30% discount
            CustomerSegment.SMB: 0.85,          # 15% discount
            CustomerSegment.ENTERPRISE: 1.2,    # 20% premium
            CustomerSegment.STRATEGIC: 1.5,     # 50% premium
            CustomerSegment.HIGH_VOLUME: 0.8,   # 20% discount (bulk)
            CustomerSegment.TRIAL: 0.5          # 50% discount
        }
        return multipliers.get(segment, 1.0)

    def _calculate_time_multiplier(self, time: datetime) -> float:
        """Calculate time-based pricing multiplier"""
        # Example: peak hours (9 AM - 5 PM) vs off-peak
        hour = time.hour
        if 9 <= hour < 17:
            return 1.1  # 10% premium during business hours
        return 1.0

    def _get_competitive_adjustment(self, product_id: str) -> float:
        """Get competitive pricing adjustment"""
        # Simplified: adjust based on competitor prices
        competitor_data = self.competitor_prices.get(product_id, [])
        if not competitor_data:
            return 0.0

        # Average competitive positioning
        avg_diff = sum(
            float(c.price_difference) / float(c.price)
            for c in competitor_data
        ) / len(competitor_data)

        # If we're significantly higher, reduce price
        return -min(0.1, max(-0.1, avg_diff * 0.5))

    def _calculate_capacity_multiplier(self, product_id: str) -> float:
        """Calculate capacity-based pricing multiplier"""
        # Simplified: would integrate with actual capacity metrics
        # For now, return neutral multiplier
        return 1.0

    def _calculate_confidence_score(
        self,
        demand_mult: float,
        segment_mult: float,
        time_mult: float
    ) -> float:
        """Calculate confidence score for pricing decision"""
        # Simple heuristic: closer to 1.0 = higher confidence
        variance = sum([
            abs(demand_mult - 1.0),
            abs(segment_mult - 1.0),
            abs(time_mult - 1.0)
        ]) / 3.0

        return max(0.5, 1.0 - variance)

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from historical values"""
        if len(values) < 2:
            return "stable"

        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        return "stable"

    def _calculate_seasonality(self, history: List[PricePoint]) -> float:
        """Calculate seasonality factor from price history"""
        # Simplified: would use proper time series decomposition
        if len(history) < 7:
            return 1.0

        # Calculate day-of-week patterns
        weekly_avg = sum(float(p.final_price) for p in history[-7:]) / 7
        overall_avg = sum(float(p.final_price) for p in history) / len(history)

        return weekly_avg / overall_avg if overall_avg > 0 else 1.0
