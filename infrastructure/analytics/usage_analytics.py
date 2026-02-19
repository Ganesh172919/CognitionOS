"""
Advanced Usage Analytics and Forecasting Engine

Provides deep insights into usage patterns, forecasting, and anomaly detection.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.billing.entities import UsageRecord
from infrastructure.persistence.billing_models import UsageRecordModel


class ForecastMethod(str, Enum):
    """Forecasting methods"""
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_REGRESSION = "linear_regression"
    SEASONAL = "seasonal"


class AnomalyType(str, Enum):
    """Types of usage anomalies"""
    SPIKE = "spike"
    DROP = "drop"
    UNUSUAL_PATTERN = "unusual_pattern"
    FRAUD_SUSPECTED = "fraud_suspected"


@dataclass
class UsageForecast:
    """Usage forecast result"""
    tenant_id: str
    resource_type: str
    forecast_date: datetime
    predicted_quantity: float
    confidence_lower: float
    confidence_upper: float
    method: ForecastMethod
    accuracy_score: Optional[float] = None


@dataclass
class UsageAnomaly:
    """Usage anomaly detection result"""
    tenant_id: str
    resource_type: str
    detected_at: datetime
    anomaly_type: AnomalyType
    severity: float  # 0-1 scale
    expected_value: float
    actual_value: float
    description: str
    suggested_action: str


@dataclass
class UsageAnalytics:
    """Comprehensive usage analytics"""
    tenant_id: str
    resource_type: str
    period_start: datetime
    period_end: datetime
    total_usage: float
    avg_daily_usage: float
    peak_usage: float
    peak_time: datetime
    trend: str  # "increasing", "decreasing", "stable"
    growth_rate: float  # Percentage
    percentile_rank: float  # Among all tenants
    cost_efficiency_score: float  # 0-100


class UsageAnalyticsEngine:
    """
    Advanced usage analytics and forecasting engine.
    
    Features:
    - Real-time usage aggregation
    - Time-series forecasting
    - Anomaly detection
    - Trend analysis
    - Cost optimization recommendations
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        
    async def get_usage_analytics(
        self,
        tenant_id: str,
        resource_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> UsageAnalytics:
        """Get comprehensive usage analytics for a tenant"""
        # Query usage data
        query = select(
            func.sum(UsageRecordModel.quantity).label('total'),
            func.avg(UsageRecordModel.quantity).label('avg'),
            func.max(UsageRecordModel.quantity).label('peak'),
            func.count(UsageRecordModel.id).label('count')
        ).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.resource_type == resource_type,
            UsageRecordModel.timestamp >= start_date,
            UsageRecordModel.timestamp <= end_date
        )
        
        result = await self.session.execute(query)
        stats = result.one()
        
        # Get peak time
        peak_query = select(UsageRecordModel).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.resource_type == resource_type,
            UsageRecordModel.timestamp >= start_date,
            UsageRecordModel.timestamp <= end_date
        ).order_by(UsageRecordModel.quantity.desc()).limit(1)
        
        peak_result = await self.session.execute(peak_query)
        peak_record = peak_result.scalar_one_or_none()
        
        # Calculate trend
        days = (end_date - start_date).days or 1
        avg_daily = stats.total / days if stats.total else 0
        
        # Get historical data for trend calculation
        trend, growth_rate = await self._calculate_trend(
            tenant_id, resource_type, start_date, end_date
        )
        
        # Calculate percentile rank
        percentile = await self._calculate_percentile_rank(
            tenant_id, resource_type, stats.total
        )
        
        # Calculate cost efficiency
        efficiency_score = await self._calculate_cost_efficiency(
            tenant_id, resource_type, stats.total, days
        )
        
        return UsageAnalytics(
            tenant_id=tenant_id,
            resource_type=resource_type,
            period_start=start_date,
            period_end=end_date,
            total_usage=stats.total or 0,
            avg_daily_usage=avg_daily,
            peak_usage=stats.peak or 0,
            peak_time=peak_record.timestamp if peak_record else start_date,
            trend=trend,
            growth_rate=growth_rate,
            percentile_rank=percentile,
            cost_efficiency_score=efficiency_score
        )
    
    async def forecast_usage(
        self,
        tenant_id: str,
        resource_type: str,
        days_ahead: int = 30,
        method: ForecastMethod = ForecastMethod.EXPONENTIAL_SMOOTHING
    ) -> List[UsageForecast]:
        """Forecast future usage"""
        # Get historical data (last 90 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=90)
        
        query = select(UsageRecordModel).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.resource_type == resource_type,
            UsageRecordModel.timestamp >= start_date,
            UsageRecordModel.timestamp <= end_date
        ).order_by(UsageRecordModel.timestamp)
        
        result = await self.session.execute(query)
        records = result.scalars().all()
        
        if len(records) < 7:  # Need at least a week of data
            return []
        
        # Aggregate by day
        daily_usage = self._aggregate_daily(records)
        
        # Apply forecasting method
        forecasts = []
        if method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            forecasts = self._exponential_smoothing_forecast(
                daily_usage, days_ahead, tenant_id, resource_type
            )
        elif method == ForecastMethod.MOVING_AVERAGE:
            forecasts = self._moving_average_forecast(
                daily_usage, days_ahead, tenant_id, resource_type
            )
        elif method == ForecastMethod.LINEAR_REGRESSION:
            forecasts = self._linear_regression_forecast(
                daily_usage, days_ahead, tenant_id, resource_type
            )
        elif method == ForecastMethod.SEASONAL:
            forecasts = self._seasonal_forecast(
                daily_usage, days_ahead, tenant_id, resource_type
            )
        
        return forecasts
    
    async def detect_anomalies(
        self,
        tenant_id: str,
        resource_type: str,
        lookback_days: int = 7
    ) -> List[UsageAnomaly]:
        """Detect usage anomalies"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get recent usage
        query = select(UsageRecordModel).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.resource_type == resource_type,
            UsageRecordModel.timestamp >= start_date,
            UsageRecordModel.timestamp <= end_date
        ).order_by(UsageRecordModel.timestamp)
        
        result = await self.session.execute(query)
        records = result.scalars().all()
        
        if len(records) < 3:
            return []
        
        # Calculate statistics
        quantities = [r.quantity for r in records]
        mean = np.mean(quantities)
        std = np.std(quantities)
        
        anomalies = []
        
        # Detect spikes and drops
        for record in records[-3:]:  # Check last 3 records
            z_score = abs((record.quantity - mean) / std) if std > 0 else 0
            
            if z_score > 3:  # 3 standard deviations
                anomaly_type = AnomalyType.SPIKE if record.quantity > mean else AnomalyType.DROP
                severity = min(z_score / 5, 1.0)  # Normalize to 0-1
                
                anomalies.append(UsageAnomaly(
                    tenant_id=tenant_id,
                    resource_type=resource_type,
                    detected_at=record.timestamp,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    expected_value=mean,
                    actual_value=record.quantity,
                    description=f"Detected {anomaly_type.value}: {record.quantity:.2f} vs expected {mean:.2f}",
                    suggested_action=self._get_anomaly_action(anomaly_type, severity)
                ))
        
        # Detect unusual patterns
        if len(records) >= 7:
            pattern_anomalies = await self._detect_pattern_anomalies(
                tenant_id, resource_type, records
            )
            anomalies.extend(pattern_anomalies)
        
        return anomalies
    
    async def get_cost_optimization_recommendations(
        self,
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Analyze all resource types
        resource_types = ["api_calls", "llm_tokens", "storage", "compute"]
        
        for resource_type in resource_types:
            # Get recent usage
            analytics = await self.get_usage_analytics(
                tenant_id,
                resource_type,
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow()
            )
            
            # Check for optimization opportunities
            if analytics.growth_rate > 50:  # High growth
                recommendations.append({
                    "resource_type": resource_type,
                    "type": "upgrade_tier",
                    "priority": "high",
                    "potential_savings": self._calculate_tier_savings(analytics),
                    "description": f"Consider upgrading tier due to {analytics.growth_rate:.1f}% growth",
                    "action": "Review usage-based pricing tiers"
                })
            
            if analytics.cost_efficiency_score < 50:  # Low efficiency
                recommendations.append({
                    "resource_type": resource_type,
                    "type": "optimize_usage",
                    "priority": "medium",
                    "potential_savings": analytics.total_usage * 0.2,  # Estimated 20% savings
                    "description": f"Low cost efficiency score: {analytics.cost_efficiency_score:.1f}",
                    "action": "Enable caching or optimize query patterns"
                })
            
            # Detect underutilization
            if analytics.percentile_rank < 25 and analytics.total_usage > 0:
                recommendations.append({
                    "resource_type": resource_type,
                    "type": "downgrade_tier",
                    "priority": "low",
                    "potential_savings": analytics.total_usage * 0.3,
                    "description": "Usage in bottom 25% of users",
                    "action": "Consider downgrading to a lower tier"
                })
        
        return sorted(recommendations, key=lambda x: x["priority"], reverse=True)
    
    def _aggregate_daily(self, records: List[UsageRecordModel]) -> Dict[str, float]:
        """Aggregate usage by day"""
        daily = {}
        for record in records:
            day_key = record.timestamp.date().isoformat()
            daily[day_key] = daily.get(day_key, 0) + record.quantity
        return daily
    
    def _exponential_smoothing_forecast(
        self,
        daily_usage: Dict[str, float],
        days_ahead: int,
        tenant_id: str,
        resource_type: str,
        alpha: float = 0.3
    ) -> List[UsageForecast]:
        """Exponential smoothing forecast"""
        values = list(daily_usage.values())
        if not values:
            return []
        
        # Simple exponential smoothing
        forecasts = []
        smoothed = values[0]
        
        # Calculate smoothed values
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Forecast future values
        last_date = max(datetime.fromisoformat(d) for d in daily_usage.keys())
        
        for day in range(1, days_ahead + 1):
            forecast_date = last_date + timedelta(days=day)
            
            # Simple forecast with increasing uncertainty
            std_dev = np.std(values) * (1 + day * 0.1)
            
            forecasts.append(UsageForecast(
                tenant_id=tenant_id,
                resource_type=resource_type,
                forecast_date=forecast_date,
                predicted_quantity=smoothed,
                confidence_lower=max(0, smoothed - 1.96 * std_dev),
                confidence_upper=smoothed + 1.96 * std_dev,
                method=ForecastMethod.EXPONENTIAL_SMOOTHING
            ))
        
        return forecasts
    
    def _moving_average_forecast(
        self,
        daily_usage: Dict[str, float],
        days_ahead: int,
        tenant_id: str,
        resource_type: str,
        window: int = 7
    ) -> List[UsageForecast]:
        """Moving average forecast"""
        values = list(daily_usage.values())
        if len(values) < window:
            return []
        
        # Calculate moving average
        ma = np.mean(values[-window:])
        std_dev = np.std(values[-window:])
        
        forecasts = []
        last_date = max(datetime.fromisoformat(d) for d in daily_usage.keys())
        
        for day in range(1, days_ahead + 1):
            forecast_date = last_date + timedelta(days=day)
            uncertainty = std_dev * (1 + day * 0.05)
            
            forecasts.append(UsageForecast(
                tenant_id=tenant_id,
                resource_type=resource_type,
                forecast_date=forecast_date,
                predicted_quantity=ma,
                confidence_lower=max(0, ma - 1.96 * uncertainty),
                confidence_upper=ma + 1.96 * uncertainty,
                method=ForecastMethod.MOVING_AVERAGE
            ))
        
        return forecasts
    
    def _linear_regression_forecast(
        self,
        daily_usage: Dict[str, float],
        days_ahead: int,
        tenant_id: str,
        resource_type: str
    ) -> List[UsageForecast]:
        """Linear regression forecast"""
        if len(daily_usage) < 2:
            return []
        
        # Prepare data
        x = np.arange(len(daily_usage))
        y = np.array(list(daily_usage.values()))
        
        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        # Calculate residuals for confidence interval
        fitted = slope * x + intercept
        residuals = y - fitted
        std_dev = np.std(residuals)
        
        forecasts = []
        last_date = max(datetime.fromisoformat(d) for d in daily_usage.keys())
        
        for day in range(1, days_ahead + 1):
            forecast_date = last_date + timedelta(days=day)
            x_forecast = len(daily_usage) + day - 1
            predicted = slope * x_forecast + intercept
            
            # Increasing uncertainty with distance
            uncertainty = std_dev * np.sqrt(1 + day / len(daily_usage))
            
            forecasts.append(UsageForecast(
                tenant_id=tenant_id,
                resource_type=resource_type,
                forecast_date=forecast_date,
                predicted_quantity=max(0, predicted),
                confidence_lower=max(0, predicted - 1.96 * uncertainty),
                confidence_upper=max(0, predicted + 1.96 * uncertainty),
                method=ForecastMethod.LINEAR_REGRESSION
            ))
        
        return forecasts
    
    def _seasonal_forecast(
        self,
        daily_usage: Dict[str, float],
        days_ahead: int,
        tenant_id: str,
        resource_type: str
    ) -> List[UsageForecast]:
        """Seasonal forecast (weekly patterns)"""
        if len(daily_usage) < 14:  # Need at least 2 weeks
            return []
        
        # Calculate day-of-week patterns
        dow_patterns = [[] for _ in range(7)]
        
        dates = sorted(daily_usage.keys())
        for date_str in dates:
            date = datetime.fromisoformat(date_str)
            dow = date.weekday()
            dow_patterns[dow].append(daily_usage[date_str])
        
        # Average by day of week
        dow_averages = [np.mean(pattern) if pattern else 0 for pattern in dow_patterns]
        dow_stds = [np.std(pattern) if len(pattern) > 1 else 0 for pattern in dow_patterns]
        
        forecasts = []
        last_date = max(datetime.fromisoformat(d) for d in daily_usage.keys())
        
        for day in range(1, days_ahead + 1):
            forecast_date = last_date + timedelta(days=day)
            dow = forecast_date.weekday()
            
            predicted = dow_averages[dow]
            std_dev = dow_stds[dow]
            
            forecasts.append(UsageForecast(
                tenant_id=tenant_id,
                resource_type=resource_type,
                forecast_date=forecast_date,
                predicted_quantity=predicted,
                confidence_lower=max(0, predicted - 1.96 * std_dev),
                confidence_upper=predicted + 1.96 * std_dev,
                method=ForecastMethod.SEASONAL
            ))
        
        return forecasts
    
    async def _calculate_trend(
        self,
        tenant_id: str,
        resource_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> tuple[str, float]:
        """Calculate usage trend"""
        mid_date = start_date + (end_date - start_date) / 2
        
        # First half usage
        first_query = select(func.sum(UsageRecordModel.quantity)).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.resource_type == resource_type,
            UsageRecordModel.timestamp >= start_date,
            UsageRecordModel.timestamp < mid_date
        )
        first_result = await self.session.execute(first_query)
        first_half = first_result.scalar() or 0
        
        # Second half usage
        second_query = select(func.sum(UsageRecordModel.quantity)).where(
            UsageRecordModel.tenant_id == tenant_id,
            UsageRecordModel.resource_type == resource_type,
            UsageRecordModel.timestamp >= mid_date,
            UsageRecordModel.timestamp <= end_date
        )
        second_result = await self.session.execute(second_query)
        second_half = second_result.scalar() or 0
        
        # Calculate growth rate
        if first_half > 0:
            growth_rate = ((second_half - first_half) / first_half) * 100
        else:
            growth_rate = 0
        
        # Determine trend
        if abs(growth_rate) < 5:
            trend = "stable"
        elif growth_rate > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return trend, growth_rate
    
    async def _calculate_percentile_rank(
        self,
        tenant_id: str,
        resource_type: str,
        total_usage: float
    ) -> float:
        """Calculate percentile rank among all tenants"""
        # Get all tenant totals
        query = select(
            UsageRecordModel.tenant_id,
            func.sum(UsageRecordModel.quantity).label('total')
        ).where(
            UsageRecordModel.resource_type == resource_type,
            UsageRecordModel.timestamp >= datetime.utcnow() - timedelta(days=30)
        ).group_by(UsageRecordModel.tenant_id)
        
        result = await self.session.execute(query)
        totals = [row.total for row in result]
        
        if not totals:
            return 50.0
        
        # Calculate percentile
        below = sum(1 for t in totals if t < total_usage)
        percentile = (below / len(totals)) * 100
        
        return percentile
    
    async def _calculate_cost_efficiency(
        self,
        tenant_id: str,
        resource_type: str,
        total_usage: float,
        days: int
    ) -> float:
        """Calculate cost efficiency score (0-100)"""
        # This is a simplified calculation
        # In production, would compare actual cost vs optimal cost
        
        avg_daily = total_usage / days if days > 0 else 0
        
        # Get tier information (simplified)
        # In production, would query subscription tier
        optimal_daily = 1000  # Example threshold
        
        if avg_daily > optimal_daily:
            # Over-usage penalty
            efficiency = max(0, 100 - (avg_daily - optimal_daily) / optimal_daily * 50)
        else:
            # Under-usage is less efficient
            efficiency = 50 + (avg_daily / optimal_daily) * 50
        
        return min(100, max(0, efficiency))
    
    async def _detect_pattern_anomalies(
        self,
        tenant_id: str,
        resource_type: str,
        records: List[UsageRecordModel]
    ) -> List[UsageAnomaly]:
        """Detect unusual usage patterns"""
        anomalies = []
        
        # Check for rapid sustained changes
        if len(records) >= 7:
            recent_avg = np.mean([r.quantity for r in records[-3:]])
            older_avg = np.mean([r.quantity for r in records[:4]])
            
            if older_avg > 0:
                change_ratio = recent_avg / older_avg
                
                if change_ratio > 3 or change_ratio < 0.33:
                    anomalies.append(UsageAnomaly(
                        tenant_id=tenant_id,
                        resource_type=resource_type,
                        detected_at=records[-1].timestamp,
                        anomaly_type=AnomalyType.UNUSUAL_PATTERN,
                        severity=min(abs(change_ratio - 1) / 3, 1.0),
                        expected_value=older_avg,
                        actual_value=recent_avg,
                        description=f"Sustained usage pattern change: {change_ratio:.1f}x",
                        suggested_action="Review recent application changes"
                    ))
        
        # Check for potential fraud (very high sudden usage)
        if records:
            max_quantity = max(r.quantity for r in records)
            avg_quantity = np.mean([r.quantity for r in records])
            
            if avg_quantity > 0 and max_quantity / avg_quantity > 10:
                anomalies.append(UsageAnomaly(
                    tenant_id=tenant_id,
                    resource_type=resource_type,
                    detected_at=records[-1].timestamp,
                    anomaly_type=AnomalyType.FRAUD_SUSPECTED,
                    severity=0.8,
                    expected_value=avg_quantity,
                    actual_value=max_quantity,
                    description="Unusually high usage spike detected",
                    suggested_action="Investigate API key usage and enable rate limiting"
                ))
        
        return anomalies
    
    def _get_anomaly_action(self, anomaly_type: AnomalyType, severity: float) -> str:
        """Get suggested action for anomaly"""
        actions = {
            AnomalyType.SPIKE: "Check for traffic spikes or review recent deployments",
            AnomalyType.DROP: "Investigate service health or user activity changes",
            AnomalyType.UNUSUAL_PATTERN: "Review application behavior and usage patterns",
            AnomalyType.FRAUD_SUSPECTED: "Enable additional security controls and investigate API usage"
        }
        
        action = actions.get(anomaly_type, "Monitor usage closely")
        
        if severity > 0.7:
            action = f"HIGH PRIORITY: {action}"
        
        return action
    
    def _calculate_tier_savings(self, analytics: UsageAnalytics) -> float:
        """Calculate potential savings from tier upgrade"""
        # Simplified calculation
        # In production, would use actual pricing tiers
        current_cost = analytics.total_usage * 0.01  # $0.01 per unit
        upgraded_cost = analytics.total_usage * 0.008  # $0.008 per unit with volume discount
        
        return max(0, current_cost - upgraded_cost)
