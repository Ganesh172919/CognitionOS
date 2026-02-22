"""
Cost Optimization Engine with AI Recommendations

Provides intelligent cost optimization with:
- Resource usage analysis
- Cost anomaly detection
- Optimization recommendations
- Auto-scaling suggestions
- Budget forecasting
- Cost allocation tracking
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict


class ResourceType(Enum):
    """Resource types"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    AI_API = "ai_api"
    CDN = "cdn"


class OptimizationPriority(Enum):
    """Optimization priority"""
    CRITICAL = "critical"  # >50% savings
    HIGH = "high"          # 20-50% savings
    MEDIUM = "medium"      # 10-20% savings
    LOW = "low"            # <10% savings


@dataclass
class ResourceUsage:
    """Resource usage data"""
    resource_id: str
    resource_type: ResourceType
    cost_usd: float
    usage_amount: float
    usage_unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    recommendation_id: str
    title: str
    description: str
    resource_type: ResourceType
    priority: OptimizationPriority
    estimated_savings_usd: float
    estimated_savings_percentage: float
    implementation_effort: str  # "low", "medium", "high"
    action_items: List[str] = field(default_factory=list)
    risk_level: str = "low"
    created_at: datetime = field(default_factory=datetime.utcnow)


class CostOptimizationEngine:
    """
    Cost Optimization Engine with AI Recommendations

    Features:
    - Real-time cost monitoring
    - Usage pattern analysis
    - Anomaly detection in spending
    - Automated optimization recommendations
    - Budget forecasting
    - Cost allocation by service/team
    - Right-sizing recommendations
    - Reserved instance suggestions
    - Idle resource detection
    - Auto-scaling optimization
    - Cost trend analysis
    """

    def __init__(self):
        self._usage_data: List[ResourceUsage] = []
        self._recommendations: List[OptimizationRecommendation] = []
        self._cost_by_resource: Dict[str, float] = defaultdict(float)
        self._budgets: Dict[str, float] = {}
        self._alerts: List[Dict[str, Any]] = []

    def track_usage(self, usage: ResourceUsage):
        """Track resource usage"""
        self._usage_data.append(usage)
        self._cost_by_resource[usage.resource_id] += usage.cost_usd

        # Check for anomalies
        self._detect_cost_anomalies(usage)

    def _detect_cost_anomalies(self, usage: ResourceUsage):
        """Detect cost anomalies"""
        # Get historical average
        historical = [
            u for u in self._usage_data
            if u.resource_id == usage.resource_id
            and u.timestamp < usage.timestamp
        ]

        if len(historical) < 10:
            return

        # Calculate average cost
        avg_cost = sum(u.cost_usd for u in historical[-30:]) / len(historical[-30:])

        # Check if current cost is significantly higher
        if usage.cost_usd > avg_cost * 2:  # 2x threshold
            self._alerts.append({
                "type": "cost_anomaly",
                "resource_id": usage.resource_id,
                "current_cost": usage.cost_usd,
                "average_cost": avg_cost,
                "increase_percentage": ((usage.cost_usd - avg_cost) / avg_cost) * 100,
                "timestamp": usage.timestamp.isoformat()
            })

    def analyze_and_recommend(
        self,
        time_window_days: int = 30
    ) -> List[OptimizationRecommendation]:
        """
        Analyze usage and generate recommendations

        Args:
            time_window_days: Days to analyze

        Returns:
            List of optimization recommendations
        """
        recommendations = []
        cutoff = datetime.utcnow() - timedelta(days=time_window_days)

        # Filter recent usage
        recent_usage = [
            u for u in self._usage_data
            if u.timestamp >= cutoff
        ]

        if not recent_usage:
            return recommendations

        # Analyze by resource type
        by_type = defaultdict(list)
        for usage in recent_usage:
            by_type[usage.resource_type].append(usage)

        # Check compute resources
        if ResourceType.COMPUTE in by_type:
            recommendations.extend(
                self._analyze_compute(by_type[ResourceType.COMPUTE])
            )

        # Check AI API usage
        if ResourceType.AI_API in by_type:
            recommendations.extend(
                self._analyze_ai_api(by_type[ResourceType.AI_API])
            )

        # Check storage
        if ResourceType.STORAGE in by_type:
            recommendations.extend(
                self._analyze_storage(by_type[ResourceType.STORAGE])
            )

        # Check for idle resources
        recommendations.extend(self._detect_idle_resources(recent_usage))

        # Sort by estimated savings
        recommendations.sort(
            key=lambda r: r.estimated_savings_usd,
            reverse=True
        )

        self._recommendations.extend(recommendations)
        return recommendations

    def _analyze_compute(
        self,
        compute_usage: List[ResourceUsage]
    ) -> List[OptimizationRecommendation]:
        """Analyze compute resource usage"""
        recommendations = []

        # Group by resource
        by_resource = defaultdict(list)
        for usage in compute_usage:
            by_resource[usage.resource_id].append(usage)

        for resource_id, usages in by_resource.items():
            total_cost = sum(u.cost_usd for u in usages)

            # Check utilization patterns
            avg_usage = sum(u.usage_amount for u in usages) / len(usages)

            # Low utilization (<30%)
            if avg_usage < 30:
                savings = total_cost * 0.4  # 40% savings potential

                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"compute_downsize_{resource_id}",
                    title=f"Downsize under-utilized compute resource {resource_id}",
                    description=f"Resource running at {avg_usage:.1f}% average utilization. Consider downsizing to save costs.",
                    resource_type=ResourceType.COMPUTE,
                    priority=OptimizationPriority.HIGH if savings > total_cost * 0.2 else OptimizationPriority.MEDIUM,
                    estimated_savings_usd=savings,
                    estimated_savings_percentage=40.0,
                    implementation_effort="medium",
                    action_items=[
                        "Review application requirements",
                        "Test with smaller instance type",
                        "Monitor performance after downsize",
                        "Update infrastructure configuration"
                    ]
                ))

            # Check for reserved instance opportunity
            if total_cost > 100:  # Significant cost
                savings = total_cost * 0.3  # 30% with reserved

                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"compute_reserved_{resource_id}",
                    title=f"Consider reserved instances for {resource_id}",
                    description=f"Consistent usage pattern detected. Reserved instances could save 30-40%.",
                    resource_type=ResourceType.COMPUTE,
                    priority=OptimizationPriority.MEDIUM,
                    estimated_savings_usd=savings,
                    estimated_savings_percentage=30.0,
                    implementation_effort="low",
                    action_items=[
                        "Analyze usage consistency",
                        "Compare on-demand vs reserved pricing",
                        "Purchase reserved instances",
                        "Monitor commitment utilization"
                    ]
                ))

        return recommendations

    def _analyze_ai_api(
        self,
        ai_usage: List[ResourceUsage]
    ) -> List[OptimizationRecommendation]:
        """Analyze AI API usage"""
        recommendations = []

        total_cost = sum(u.cost_usd for u in ai_usage)

        if total_cost < 10:
            return recommendations

        # Check for expensive model usage
        by_model = defaultdict(list)
        for usage in ai_usage:
            model = usage.metadata.get("model", "unknown")
            by_model[model].append(usage)

        # Identify opportunities to use cheaper models
        for model, usages in by_model.items():
            model_cost = sum(u.cost_usd for u in usages)

            if model_cost > total_cost * 0.3:  # >30% of AI costs
                savings = model_cost * 0.5  # 50% potential savings

                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"ai_model_optimize_{model}",
                    title=f"Optimize {model} usage",
                    description=f"Model {model} accounts for ${model_cost:.2f}. Consider cheaper alternatives for non-critical tasks.",
                    resource_type=ResourceType.AI_API,
                    priority=OptimizationPriority.HIGH if savings > 100 else OptimizationPriority.MEDIUM,
                    estimated_savings_usd=savings,
                    estimated_savings_percentage=50.0,
                    implementation_effort="medium",
                    action_items=[
                        "Identify critical vs non-critical API calls",
                        "Test cheaper models for non-critical tasks",
                        "Implement model selection logic",
                        "Add prompt caching"
                    ],
                    risk_level="low"
                ))

        # Check for caching opportunities
        if len(ai_usage) > 100:
            cache_savings = total_cost * 0.2  # 20% through caching

            recommendations.append(OptimizationRecommendation(
                recommendation_id="ai_caching",
                title="Implement AI response caching",
                description="High API call volume detected. Caching could reduce costs by 20-30%.",
                resource_type=ResourceType.AI_API,
                priority=OptimizationPriority.MEDIUM,
                estimated_savings_usd=cache_savings,
                estimated_savings_percentage=20.0,
                implementation_effort="medium",
                action_items=[
                    "Implement prompt caching layer",
                    "Set appropriate TTLs",
                    "Monitor cache hit rates",
                    "Adjust caching strategy"
                ]
            ))

        return recommendations

    def _analyze_storage(
        self,
        storage_usage: List[ResourceUsage]
    ) -> List[OptimizationRecommendation]:
        """Analyze storage usage"""
        recommendations = []

        total_cost = sum(u.cost_usd for u in storage_usage)

        if total_cost < 5:
            return recommendations

        # Check for old data that could be archived
        recommendations.append(OptimizationRecommendation(
            recommendation_id="storage_lifecycle",
            title="Implement storage lifecycle policies",
            description="Move old data to cheaper storage tiers",
            resource_type=ResourceType.STORAGE,
            priority=OptimizationPriority.LOW,
            estimated_savings_usd=total_cost * 0.3,
            estimated_savings_percentage=30.0,
            implementation_effort="low",
            action_items=[
                "Define data retention policies",
                "Configure lifecycle rules",
                "Archive data older than 90 days",
                "Monitor storage tier distribution"
            ]
        ))

        return recommendations

    def _detect_idle_resources(
        self,
        recent_usage: List[ResourceUsage]
    ) -> List[OptimizationRecommendation]:
        """Detect idle or underutilized resources"""
        recommendations = []

        # Group by resource
        by_resource = defaultdict(list)
        for usage in recent_usage:
            by_resource[usage.resource_id].append(usage)

        for resource_id, usages in by_resource.items():
            # Check if resource has zero usage recently
            recent = usages[-7:]  # Last 7 data points
            if all(u.usage_amount == 0 for u in recent):
                total_cost = sum(u.cost_usd for u in usages)

                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"idle_{resource_id}",
                    title=f"Idle resource detected: {resource_id}",
                    description=f"Resource has no activity for extended period. Consider decommissioning.",
                    resource_type=usages[0].resource_type,
                    priority=OptimizationPriority.CRITICAL,
                    estimated_savings_usd=total_cost,
                    estimated_savings_percentage=100.0,
                    implementation_effort="low",
                    action_items=[
                        "Verify resource is truly unused",
                        "Check for dependencies",
                        "Create backup if needed",
                        "Decommission resource"
                    ],
                    risk_level="medium"
                ))

        return recommendations

    def forecast_costs(
        self,
        days_ahead: int = 30
    ) -> Dict[str, Any]:
        """
        Forecast costs for next period

        Args:
            days_ahead: Days to forecast

        Returns:
            Cost forecast
        """
        # Get historical data (last 30 days)
        cutoff = datetime.utcnow() - timedelta(days=30)
        historical = [
            u for u in self._usage_data
            if u.timestamp >= cutoff
        ]

        if not historical:
            return {"error": "Insufficient data"}

        # Calculate daily average
        total_cost = sum(u.cost_usd for u in historical)
        daily_avg = total_cost / 30

        # Simple linear forecast
        forecasted_cost = daily_avg * days_ahead

        # Apply growth factor (5% monthly growth assumption)
        growth_factor = 1.05
        forecasted_cost *= growth_factor

        return {
            "forecasted_cost_usd": round(forecasted_cost, 2),
            "daily_average_usd": round(daily_avg, 2),
            "confidence": 0.7,
            "days_ahead": days_ahead
        }

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get cost breakdown by resource type"""
        by_type = defaultdict(float)
        for usage in self._usage_data:
            by_type[usage.resource_type.value] += usage.cost_usd

        total = sum(by_type.values())

        return {
            "total_cost_usd": round(total, 2),
            "by_type": {
                k: {
                    "cost_usd": round(v, 2),
                    "percentage": round((v / total * 100), 2) if total > 0 else 0
                }
                for k, v in by_type.items()
            }
        }

    def get_top_recommendations(self, limit: int = 5) -> List[OptimizationRecommendation]:
        """Get top optimization recommendations by savings"""
        sorted_recs = sorted(
            self._recommendations,
            key=lambda r: r.estimated_savings_usd,
            reverse=True
        )
        return sorted_recs[:limit]
