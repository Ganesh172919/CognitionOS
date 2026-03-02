"""
Cost Governance Engine — CognitionOS

Enterprise cost management and optimization with:
- Per-tenant cost tracking and budgets
- AI inference cost optimization
- Resource cost allocation
- Cost alerts and thresholds
- Budget enforcement
- Cost forecasting
- Cost anomaly detection
- Usage-based billing integration
- Cost reporting and dashboards
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CostCategory(str, Enum):
    AI_INFERENCE = "ai_inference"
    STORAGE = "storage"
    COMPUTE = "compute"
    BANDWIDTH = "bandwidth"
    API_CALLS = "api_calls"
    PLUGINS = "plugins"
    SUPPORT = "support"
    DATABASE = "database"
    SEARCH = "search"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CostEntry:
    entry_id: str
    tenant_id: str
    category: CostCategory
    amount: float
    description: str = ""
    resource: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Budget:
    budget_id: str
    tenant_id: str
    monthly_limit: float
    daily_limit: float = 0
    category_limits: Dict[str, float] = field(default_factory=dict)
    alert_thresholds: List[float] = field(default_factory=lambda: [50, 75, 90, 100])
    hard_limit: bool = False  # If True, block usage when limit reached
    created_at: float = field(default_factory=time.time)


@dataclass
class CostAlert:
    alert_id: str
    tenant_id: str
    severity: AlertSeverity
    message: str
    current_cost: float
    limit: float
    threshold_pct: float
    category: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id, "severity": self.severity.value,
            "message": self.message,
            "current": round(self.current_cost, 2),
            "limit": round(self.limit, 2),
            "threshold_pct": self.threshold_pct,
        }


@dataclass
class CostForecast:
    tenant_id: str
    current_month_cost: float
    projected_month_cost: float
    daily_avg: float
    days_remaining: int
    budget_limit: float
    will_exceed: bool
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": round(self.current_month_cost, 2),
            "projected": round(self.projected_month_cost, 2),
            "daily_avg": round(self.daily_avg, 2),
            "days_remaining": self.days_remaining,
            "budget": round(self.budget_limit, 2),
            "will_exceed": self.will_exceed,
        }


class CostGovernanceEngine:
    """
    Enterprise cost governance with budgets, alerts, forecasting,
    and optimization recommendations.
    """

    def __init__(self):
        self._entries: List[CostEntry] = []
        self._budgets: Dict[str, Budget] = {}  # tenant_id -> budget
        self._alerts: List[CostAlert] = []
        self._alert_callbacks: List[Callable[[CostAlert], None]] = []
        self._fired_thresholds: Dict[str, Set[float]] = defaultdict(set)
        self._optimizations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # ── Cost Recording ──

    def record_cost(self, tenant_id: str, category: CostCategory,
                      amount: float, *, description: str = "",
                      resource: str = "",
                      metadata: Optional[Dict] = None) -> CostEntry:
        entry = CostEntry(
            entry_id=uuid.uuid4().hex[:12],
            tenant_id=tenant_id, category=category,
            amount=amount, description=description,
            resource=resource, metadata=metadata or {},
        )
        self._entries.append(entry)

        # Check budgets
        self._check_budget(tenant_id)

        return entry

    # ── Budget Management ──

    def set_budget(self, tenant_id: str, monthly_limit: float, *,
                     daily_limit: float = 0,
                     category_limits: Optional[Dict[str, float]] = None,
                     hard_limit: bool = False) -> Budget:
        budget = Budget(
            budget_id=uuid.uuid4().hex[:12],
            tenant_id=tenant_id,
            monthly_limit=monthly_limit,
            daily_limit=daily_limit or monthly_limit / 30,
            category_limits=category_limits or {},
            hard_limit=hard_limit,
        )
        self._budgets[tenant_id] = budget
        return budget

    def check_allowed(self, tenant_id: str, estimated_cost: float) -> bool:
        """Check if a cost is within budget. Returns True if allowed."""
        budget = self._budgets.get(tenant_id)
        if not budget or not budget.hard_limit:
            return True

        monthly = self.get_monthly_cost(tenant_id)
        return (monthly + estimated_cost) <= budget.monthly_limit

    def _check_budget(self, tenant_id: str):
        budget = self._budgets.get(tenant_id)
        if not budget:
            return

        monthly_cost = self.get_monthly_cost(tenant_id)
        usage_pct = (monthly_cost / max(budget.monthly_limit, 0.01)) * 100

        for threshold in budget.alert_thresholds:
            if usage_pct >= threshold and threshold not in self._fired_thresholds[tenant_id]:
                self._fired_thresholds[tenant_id].add(threshold)
                severity = AlertSeverity.CRITICAL if threshold >= 90 else (
                    AlertSeverity.WARNING if threshold >= 75 else AlertSeverity.INFO
                )
                alert = CostAlert(
                    alert_id=uuid.uuid4().hex[:12],
                    tenant_id=tenant_id,
                    severity=severity,
                    message=f"Budget usage at {round(usage_pct, 1)}% "
                            f"(${round(monthly_cost, 2)} / ${round(budget.monthly_limit, 2)})",
                    current_cost=monthly_cost,
                    limit=budget.monthly_limit,
                    threshold_pct=threshold,
                )
                self._alerts.append(alert)
                for callback in self._alert_callbacks:
                    try:
                        callback(alert)
                    except Exception:
                        pass

    # ── Cost Queries ──

    def get_monthly_cost(self, tenant_id: str) -> float:
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0).timestamp()
        return sum(
            e.amount for e in self._entries
            if e.tenant_id == tenant_id and e.timestamp >= month_start
        )

    def get_daily_cost(self, tenant_id: str) -> float:
        day_start = time.time() - 86400
        return sum(
            e.amount for e in self._entries
            if e.tenant_id == tenant_id and e.timestamp >= day_start
        )

    def get_cost_breakdown(self, tenant_id: str, *,
                              days: int = 30) -> Dict[str, Any]:
        cutoff = time.time() - (days * 86400)
        entries = [
            e for e in self._entries
            if e.tenant_id == tenant_id and e.timestamp >= cutoff
        ]

        by_category: Dict[str, float] = defaultdict(float)
        by_day: Dict[str, float] = defaultdict(float)
        by_resource: Dict[str, float] = defaultdict(float)

        for e in entries:
            by_category[e.category.value] += e.amount
            day = datetime.fromtimestamp(e.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
            by_day[day] += e.amount
            if e.resource:
                by_resource[e.resource] += e.amount

        total = sum(e.amount for e in entries)

        return {
            "total": round(total, 2),
            "period_days": days,
            "daily_avg": round(total / max(days, 1), 2),
            "by_category": {k: round(v, 2) for k, v in sorted(
                by_category.items(), key=lambda x: -x[1]
            )},
            "daily_trend": [
                {"date": d, "cost": round(c, 2)}
                for d, c in sorted(by_day.items())
            ],
            "top_resources": sorted(
                [{"resource": r, "cost": round(c, 2)}
                 for r, c in by_resource.items()],
                key=lambda x: -x["cost"]
            )[:10],
        }

    # ── Forecasting ──

    def forecast(self, tenant_id: str) -> CostForecast:
        now = datetime.now(timezone.utc)
        day_of_month = now.day
        days_in_month = 30
        days_remaining = max(days_in_month - day_of_month, 1)

        monthly_cost = self.get_monthly_cost(tenant_id)
        daily_avg = monthly_cost / max(day_of_month, 1)
        projected = monthly_cost + (daily_avg * days_remaining)

        budget = self._budgets.get(tenant_id)
        budget_limit = budget.monthly_limit if budget else 0

        return CostForecast(
            tenant_id=tenant_id,
            current_month_cost=monthly_cost,
            projected_month_cost=projected,
            daily_avg=daily_avg,
            days_remaining=days_remaining,
            budget_limit=budget_limit,
            will_exceed=projected > budget_limit if budget_limit > 0 else False,
        )

    # ── Optimization Recommendations ──

    def get_optimizations(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations."""
        recommendations = []
        breakdown = self.get_cost_breakdown(tenant_id, days=30)

        # AI inference optimization
        ai_cost = breakdown["by_category"].get("ai_inference", 0)
        total = breakdown["total"]
        if total > 0 and ai_cost / total > 0.6:
            recommendations.append({
                "type": "model_optimization",
                "title": "Optimize AI Model Usage",
                "description": f"AI inference accounts for {round(ai_cost/total*100)}% of costs. "
                               "Consider using smaller models for simple tasks.",
                "potential_savings_pct": 20,
                "priority": "high",
            })

        # Storage optimization
        storage_cost = breakdown["by_category"].get("storage", 0)
        if storage_cost > 10:
            recommendations.append({
                "type": "storage_optimization",
                "title": "Optimize Storage Usage",
                "description": "Review stored data for cleanup opportunities.",
                "potential_savings_pct": 15,
                "priority": "medium",
            })

        # Caching recommendation
        api_cost = breakdown["by_category"].get("api_calls", 0)
        if api_cost > 5:
            recommendations.append({
                "type": "caching",
                "title": "Enable Response Caching",
                "description": "Caching frequently accessed data can reduce API costs.",
                "potential_savings_pct": 25,
                "priority": "medium",
            })

        # Tier recommendation
        forecast = self.forecast(tenant_id)
        budget = self._budgets.get(tenant_id)
        if budget and forecast.projected_month_cost > budget.monthly_limit * 0.9:
            recommendations.append({
                "type": "tier_upgrade",
                "title": "Consider Higher Tier",
                "description": "Projected costs approach your budget limit. "
                               "Higher tiers offer bulk discounts.",
                "priority": "high",
            })

        return recommendations

    # ── Alerts ──

    def add_alert_callback(self, callback: Callable[[CostAlert], None]):
        self._alert_callbacks.append(callback)

    def get_alerts(self, *, tenant_id: Optional[str] = None,
                     severity: Optional[AlertSeverity] = None,
                     limit: int = 50) -> List[Dict[str, Any]]:
        alerts = self._alerts
        if tenant_id:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return [a.to_dict() for a in alerts[-limit:]]

    def acknowledge_alert(self, alert_id: str):
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                break

    # ── Platform-wide Stats ──

    def get_platform_costs(self) -> Dict[str, Any]:
        total = sum(e.amount for e in self._entries)
        by_tenant: Dict[str, float] = defaultdict(float)
        by_category: Dict[str, float] = defaultdict(float)

        for e in self._entries:
            by_tenant[e.tenant_id] += e.amount
            by_category[e.category.value] += e.amount

        return {
            "total_cost": round(total, 2),
            "total_entries": len(self._entries),
            "tenants_tracked": len(by_tenant),
            "by_category": {k: round(v, 2) for k, v in sorted(
                by_category.items(), key=lambda x: -x[1]
            )},
            "top_tenants": sorted(
                [{"tenant_id": t, "cost": round(c, 2)}
                 for t, c in by_tenant.items()],
                key=lambda x: -x["cost"]
            )[:10],
            "active_budgets": len(self._budgets),
            "unacknowledged_alerts": sum(
                1 for a in self._alerts if not a.acknowledged
            ),
        }

    def cleanup(self, *, max_entries: int = 500000):
        if len(self._entries) > max_entries:
            self._entries = self._entries[-max_entries // 2:]


# ── Singleton ──
_engine: Optional[CostGovernanceEngine] = None


def get_cost_engine() -> CostGovernanceEngine:
    global _engine
    if not _engine:
        _engine = CostGovernanceEngine()
    return _engine
