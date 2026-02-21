"""
Advanced Billing Metering System

Tracks compute, tokens, storage, and API usage for billing purposes.
Supports:
- Real-time metering with sub-second precision
- Multiple billing dimensions (tokens, requests, compute-seconds, storage-GB)
- Tiered pricing with overage billing
- Per-tenant, per-user, and per-project cost attribution
- Invoice generation
- Cost forecasting based on trailing consumption
- Budget alerts
- Webhook notifications on threshold breach
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4


class BillingDimension(str, Enum):
    """Billable resource dimensions"""
    LLM_INPUT_TOKENS = "llm_input_tokens"
    LLM_OUTPUT_TOKENS = "llm_output_tokens"
    API_REQUESTS = "api_requests"
    COMPUTE_SECONDS = "compute_seconds"
    STORAGE_GB_HOURS = "storage_gb_hours"
    AGENT_EXECUTIONS = "agent_executions"
    WORKFLOW_EXECUTIONS = "workflow_executions"
    TOOL_CALLS = "tool_calls"
    DATA_TRANSFER_GB = "data_transfer_gb"
    PLUGIN_CALLS = "plugin_calls"


class SubscriptionTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class PricingRule:
    """Price per unit of a billing dimension"""
    dimension: BillingDimension
    unit_price: float         # USD per unit
    free_units: float = 0.0   # Free allowance per billing period
    overage_unit_price: float = 0.0  # Price after free tier exhausted
    tier_breaks: List[Tuple[float, float]] = field(default_factory=list)  # [(units, price/unit)]

    def calculate_cost(self, units: float) -> float:
        """Calculate cost for a given number of units (includes free tier and tiers)"""
        billable = max(0.0, units - self.free_units)
        if not self.tier_breaks:
            return billable * self.unit_price

        # Tiered pricing
        remaining = billable
        cost = 0.0
        prev_break = 0.0
        for break_units, price in sorted(self.tier_breaks, key=lambda x: x[0]):
            tier_units = min(remaining, break_units - prev_break)
            cost += tier_units * price
            remaining -= tier_units
            prev_break = break_units
            if remaining <= 0:
                break
        if remaining > 0:
            cost += remaining * self.overage_unit_price
        return cost


@dataclass
class UsageRecord:
    """Single usage event"""
    record_id: str
    tenant_id: str
    user_id: Optional[str]
    project_id: Optional[str]
    dimension: BillingDimension
    quantity: float
    unit_cost: float
    total_cost: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "dimension": self.dimension.value,
            "quantity": self.quantity,
            "unit_cost": round(self.unit_cost, 8),
            "total_cost": round(self.total_cost, 8),
            "timestamp": self.timestamp,
        }


@dataclass
class BillingPeriod:
    """Summary of a billing period"""
    period_id: str
    tenant_id: str
    start_ts: float
    end_ts: float
    dimension_totals: Dict[str, float] = field(default_factory=dict)
    dimension_costs: Dict[str, float] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    credits_applied: float = 0.0
    amount_due: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_id": self.period_id,
            "tenant_id": self.tenant_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "dimension_totals": self.dimension_totals,
            "dimension_costs": {k: round(v, 4) for k, v in self.dimension_costs.items()},
            "total_cost_usd": round(self.total_cost_usd, 4),
            "subscription_tier": self.subscription_tier.value,
            "credits_applied": round(self.credits_applied, 4),
            "amount_due": round(self.amount_due, 4),
        }


@dataclass
class BudgetAlert:
    """Budget threshold alert configuration"""
    alert_id: str
    tenant_id: str
    threshold_usd: float
    period_days: int = 30
    notify_at_pct: List[float] = field(default_factory=lambda: [50.0, 75.0, 90.0, 100.0])
    last_notified_pct: float = 0.0
    created_at: float = field(default_factory=time.time)


class BillingMeter:
    """
    Real-time billing metering engine.

    Usage::

        meter = BillingMeter()
        meter.set_pricing(SubscriptionTier.PRO, [
            PricingRule(BillingDimension.LLM_INPUT_TOKENS, 0.003 / 1000, free_units=10000),
            PricingRule(BillingDimension.API_REQUESTS, 0.0001, free_units=1000),
        ])

        meter.record(
            tenant_id="t-abc",
            dimension=BillingDimension.LLM_INPUT_TOKENS,
            quantity=1500,
            tier=SubscriptionTier.PRO,
        )
        print(meter.get_current_usage("t-abc"))
    """

    def __init__(self) -> None:
        self._records: List[UsageRecord] = []
        self._pricing: Dict[SubscriptionTier, Dict[BillingDimension, PricingRule]] = {}
        self._tenant_tiers: Dict[str, SubscriptionTier] = {}
        self._tenant_credits: Dict[str, float] = defaultdict(float)
        self._budget_alerts: Dict[str, BudgetAlert] = {}
        self._alert_callbacks: List[Callable[[str, float, float], None]] = []
        # In-memory aggregated counters per (tenant, dimension, period_day)
        self._agg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # ──────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────

    def set_pricing(
        self,
        tier: SubscriptionTier,
        rules: List[PricingRule],
    ) -> None:
        self._pricing[tier] = {r.dimension: r for r in rules}

    def set_tenant_tier(self, tenant_id: str, tier: SubscriptionTier) -> None:
        self._tenant_tiers[tenant_id] = tier

    def add_credits(self, tenant_id: str, amount_usd: float) -> None:
        self._tenant_credits[tenant_id] += amount_usd

    def set_budget_alert(self, alert: BudgetAlert) -> None:
        self._budget_alerts[alert.tenant_id] = alert

    def on_budget_alert(self, callback: Callable[[str, float, float], None]) -> None:
        """callback(tenant_id, current_spend_usd, budget_usd)"""
        self._alert_callbacks.append(callback)

    # ──────────────────────────────────────────────
    # Metering
    # ──────────────────────────────────────────────

    def record(
        self,
        tenant_id: str,
        dimension: BillingDimension,
        quantity: float,
        tier: Optional[SubscriptionTier] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Record a usage event and calculate its cost"""
        effective_tier = tier or self._tenant_tiers.get(tenant_id, SubscriptionTier.FREE)
        rule = self._pricing.get(effective_tier, {}).get(dimension)
        unit_cost = rule.unit_price if rule else 0.0
        total_cost = rule.calculate_cost(quantity) if rule else 0.0

        record = UsageRecord(
            record_id=str(uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            project_id=project_id,
            dimension=dimension,
            quantity=quantity,
            unit_cost=unit_cost,
            total_cost=total_cost,
            timestamp=time.time(),
            metadata=metadata or {},
            session_id=session_id,
            workflow_id=workflow_id,
        )
        self._records.append(record)

        # Update aggregates
        day_key = self._day_key(record.timestamp)
        agg_key = f"{tenant_id}:{dimension.value}:{day_key}"
        self._agg[tenant_id][agg_key] += quantity

        # Budget check
        self._check_budget(tenant_id)

        return record

    def record_llm_usage(
        self,
        tenant_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "unknown",
        **kwargs: Any,
    ) -> Tuple[UsageRecord, UsageRecord]:
        """Convenience method for recording LLM token usage"""
        meta = {"model": model}
        r_in = self.record(
            tenant_id, BillingDimension.LLM_INPUT_TOKENS, input_tokens,
            metadata=meta, **kwargs,
        )
        r_out = self.record(
            tenant_id, BillingDimension.LLM_OUTPUT_TOKENS, output_tokens,
            metadata=meta, **kwargs,
        )
        return r_in, r_out

    # ──────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────

    def get_current_usage(
        self,
        tenant_id: str,
        period_days: int = 30,
    ) -> Dict[str, Any]:
        """Return aggregated usage for the current billing period"""
        now = time.time()
        period_start = now - (period_days * 86400)
        records = [
            r for r in self._records
            if r.tenant_id == tenant_id and r.timestamp >= period_start
        ]

        dimension_totals: Dict[str, float] = defaultdict(float)
        dimension_costs: Dict[str, float] = defaultdict(float)
        for r in records:
            dimension_totals[r.dimension.value] += r.quantity
            dimension_costs[r.dimension.value] += r.total_cost

        total_cost = sum(dimension_costs.values())
        credits = min(self._tenant_credits.get(tenant_id, 0.0), total_cost)
        amount_due = max(0.0, total_cost - credits)

        return {
            "tenant_id": tenant_id,
            "period_start": period_start,
            "period_end": now,
            "dimension_totals": dict(dimension_totals),
            "dimension_costs": {k: round(v, 6) for k, v in dimension_costs.items()},
            "total_cost_usd": round(total_cost, 6),
            "credits_applied": round(credits, 6),
            "amount_due": round(amount_due, 6),
            "subscription_tier": self._tenant_tiers.get(tenant_id, SubscriptionTier.FREE).value,
        }

    def get_usage_history(
        self,
        tenant_id: str,
        days: int = 7,
        dimension: Optional[BillingDimension] = None,
    ) -> List[Dict[str, Any]]:
        """Return daily usage breakdown"""
        now = time.time()
        cutoff = now - (days * 86400)
        records = [
            r for r in self._records
            if r.tenant_id == tenant_id
            and r.timestamp >= cutoff
            and (dimension is None or r.dimension == dimension)
        ]

        # Group by day
        daily: Dict[str, Dict[str, Any]] = {}
        for r in records:
            day = self._day_key(r.timestamp)
            if day not in daily:
                daily[day] = {"date": day, "total_cost": 0.0, "dimensions": defaultdict(float)}
            daily[day]["total_cost"] += r.total_cost
            daily[day]["dimensions"][r.dimension.value] += r.quantity

        return [
            {
                "date": d["date"],
                "total_cost_usd": round(d["total_cost"], 6),
                "dimensions": dict(d["dimensions"]),
            }
            for d in sorted(daily.values(), key=lambda x: x["date"])
        ]

    def forecast_monthly_cost(self, tenant_id: str) -> Dict[str, Any]:
        """Forecast end-of-month cost based on trailing 7-day average"""
        trailing = self.get_current_usage(tenant_id, period_days=7)
        daily_avg = trailing["total_cost_usd"] / 7
        projected = daily_avg * 30
        current_month = self.get_current_usage(tenant_id, period_days=30)

        return {
            "daily_average_usd": round(daily_avg, 6),
            "projected_monthly_usd": round(projected, 6),
            "current_period_cost_usd": current_month["total_cost_usd"],
            "confidence": "medium",
        }

    def generate_invoice(self, tenant_id: str, period_days: int = 30) -> Dict[str, Any]:
        """Generate a human-readable invoice"""
        usage = self.get_current_usage(tenant_id, period_days)
        tier = self._tenant_tiers.get(tenant_id, SubscriptionTier.FREE)

        line_items = []
        for dim_str, cost in usage["dimension_costs"].items():
            qty = usage["dimension_totals"].get(dim_str, 0)
            line_items.append({
                "description": dim_str.replace("_", " ").title(),
                "quantity": qty,
                "unit": dim_str.split("_")[-1],
                "cost_usd": cost,
            })

        return {
            "invoice_id": str(uuid4()),
            "tenant_id": tenant_id,
            "subscription_tier": tier.value,
            "billing_period_days": period_days,
            "line_items": line_items,
            "subtotal_usd": usage["total_cost_usd"],
            "credits_applied": usage["credits_applied"],
            "total_due_usd": usage["amount_due"],
            "generated_at": time.time(),
            "status": "draft",
        }

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _check_budget(self, tenant_id: str) -> None:
        alert = self._budget_alerts.get(tenant_id)
        if not alert:
            return

        usage = self.get_current_usage(tenant_id, period_days=alert.period_days)
        current = usage["total_cost_usd"]
        pct = (current / alert.threshold_usd) * 100 if alert.threshold_usd > 0 else 0

        for notify_pct in sorted(alert.notify_at_pct, reverse=True):
            if pct >= notify_pct and alert.last_notified_pct < notify_pct:
                alert.last_notified_pct = notify_pct
                for cb in self._alert_callbacks:
                    try:
                        cb(tenant_id, current, alert.threshold_usd)
                    except Exception:  # noqa: BLE001
                        pass
                break

    @staticmethod
    def _day_key(ts: float) -> str:
        import datetime
        dt = datetime.datetime.utcfromtimestamp(ts)
        return dt.strftime("%Y-%m-%d")


# Default pricing catalog
def build_default_pricing() -> Dict[SubscriptionTier, List[PricingRule]]:
    return {
        SubscriptionTier.FREE: [
            PricingRule(BillingDimension.LLM_INPUT_TOKENS, 0.0, free_units=10000),
            PricingRule(BillingDimension.LLM_OUTPUT_TOKENS, 0.0, free_units=5000),
            PricingRule(BillingDimension.API_REQUESTS, 0.0, free_units=1000),
            PricingRule(BillingDimension.AGENT_EXECUTIONS, 0.0, free_units=5),
        ],
        SubscriptionTier.STARTER: [
            PricingRule(BillingDimension.LLM_INPUT_TOKENS, 0.003 / 1000, free_units=100000),
            PricingRule(BillingDimension.LLM_OUTPUT_TOKENS, 0.015 / 1000, free_units=50000),
            PricingRule(BillingDimension.API_REQUESTS, 0.0001, free_units=10000),
            PricingRule(BillingDimension.AGENT_EXECUTIONS, 0.05, free_units=100),
            PricingRule(BillingDimension.WORKFLOW_EXECUTIONS, 0.02, free_units=200),
        ],
        SubscriptionTier.PRO: [
            PricingRule(BillingDimension.LLM_INPUT_TOKENS, 0.002 / 1000, free_units=1000000),
            PricingRule(BillingDimension.LLM_OUTPUT_TOKENS, 0.010 / 1000, free_units=500000),
            PricingRule(BillingDimension.API_REQUESTS, 0.00005, free_units=100000),
            PricingRule(BillingDimension.AGENT_EXECUTIONS, 0.02, free_units=1000),
            PricingRule(BillingDimension.WORKFLOW_EXECUTIONS, 0.01, free_units=2000),
            PricingRule(BillingDimension.COMPUTE_SECONDS, 0.00001, free_units=100000),
        ],
        SubscriptionTier.ENTERPRISE: [
            PricingRule(BillingDimension.LLM_INPUT_TOKENS, 0.001 / 1000),
            PricingRule(BillingDimension.LLM_OUTPUT_TOKENS, 0.005 / 1000),
            PricingRule(BillingDimension.API_REQUESTS, 0.00002),
            PricingRule(BillingDimension.AGENT_EXECUTIONS, 0.01),
            PricingRule(BillingDimension.WORKFLOW_EXECUTIONS, 0.005),
            PricingRule(BillingDimension.COMPUTE_SECONDS, 0.000005),
            PricingRule(BillingDimension.STORAGE_GB_HOURS, 0.00003),
        ],
    }
