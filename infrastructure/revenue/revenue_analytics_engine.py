"""
Revenue Analytics Engine — CognitionOS

Complete revenue intelligence:
- MRR/ARR calculations
- Churn analysis and prediction
- Cohort analysis
- Revenue per user/tenant segmentation
- LTV projections
- Growth rate tracking
- Revenue breakdown by tier/feature
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RevenueEventType(str, Enum):
    SUBSCRIPTION_START = "subscription_start"
    SUBSCRIPTION_RENEWAL = "subscription_renewal"
    SUBSCRIPTION_UPGRADE = "subscription_upgrade"
    SUBSCRIPTION_DOWNGRADE = "subscription_downgrade"
    SUBSCRIPTION_CANCEL = "subscription_cancel"
    ONE_TIME_PAYMENT = "one_time_payment"
    USAGE_CHARGE = "usage_charge"
    REFUND = "refund"


@dataclass
class RevenueEvent:
    event_id: str
    tenant_id: str
    event_type: RevenueEventType
    amount_usd: Decimal
    tier: str = "free"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MRRSnapshot:
    date: str
    total_mrr: Decimal
    new_mrr: Decimal
    expansion_mrr: Decimal
    contraction_mrr: Decimal
    churn_mrr: Decimal
    net_new_mrr: Decimal
    active_subscriptions: int
    arpu: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {k: float(v) if isinstance(v, Decimal) else v
                for k, v in self.__dict__.items()}


@dataclass
class CohortMetrics:
    cohort_month: str
    initial_count: int
    retained: Dict[int, int] = field(default_factory=dict)
    revenue: Dict[int, Decimal] = field(default_factory=dict)

    def retention_rate(self, month: int) -> float:
        if self.initial_count == 0:
            return 0
        return self.retained.get(month, 0) / self.initial_count * 100


class RevenueAnalyticsEngine:
    """Revenue analytics with MRR tracking, churn analysis, and LTV projections."""

    def __init__(self) -> None:
        self._events: List[RevenueEvent] = []
        self._subscriptions: Dict[str, Dict[str, Any]] = {}  # tenant_id -> sub info
        self._mrr_snapshots: List[MRRSnapshot] = []
        self._cohorts: Dict[str, CohortMetrics] = {}

    # ---- event recording ----
    def record_event(self, event: RevenueEvent) -> None:
        self._events.append(event)
        self._update_subscription_state(event)

    def _update_subscription_state(self, event: RevenueEvent) -> None:
        tid = event.tenant_id
        if event.event_type == RevenueEventType.SUBSCRIPTION_START:
            self._subscriptions[tid] = {
                "tier": event.tier, "mrr": event.amount_usd,
                "started": event.timestamp, "active": True,
                "start_month": event.timestamp[:7]}
        elif event.event_type == RevenueEventType.SUBSCRIPTION_CANCEL:
            if tid in self._subscriptions:
                self._subscriptions[tid]["active"] = False
                self._subscriptions[tid]["cancelled_at"] = event.timestamp
        elif event.event_type == RevenueEventType.SUBSCRIPTION_UPGRADE:
            if tid in self._subscriptions:
                self._subscriptions[tid]["tier"] = event.tier
                self._subscriptions[tid]["mrr"] = event.amount_usd
        elif event.event_type == RevenueEventType.SUBSCRIPTION_DOWNGRADE:
            if tid in self._subscriptions:
                self._subscriptions[tid]["tier"] = event.tier
                self._subscriptions[tid]["mrr"] = event.amount_usd

    # ---- MRR calculations ----
    def calculate_mrr(self) -> MRRSnapshot:
        active = {k: v for k, v in self._subscriptions.items() if v.get("active")}
        total_mrr = sum(Decimal(str(s["mrr"])) for s in active.values())
        count = len(active)
        arpu = total_mrr / count if count > 0 else Decimal("0")

        # Net new MRR from recent events (simplified)
        now = datetime.now(timezone.utc)
        month_ago = (now - timedelta(days=30)).isoformat()
        recent = [e for e in self._events if e.timestamp >= month_ago]

        new_mrr = sum(Decimal(str(e.amount_usd)) for e in recent
                      if e.event_type == RevenueEventType.SUBSCRIPTION_START)
        expansion = sum(Decimal(str(e.amount_usd)) for e in recent
                        if e.event_type == RevenueEventType.SUBSCRIPTION_UPGRADE)
        contraction = sum(Decimal(str(e.amount_usd)) for e in recent
                          if e.event_type == RevenueEventType.SUBSCRIPTION_DOWNGRADE)
        churn = sum(Decimal(str(e.amount_usd)) for e in recent
                    if e.event_type == RevenueEventType.SUBSCRIPTION_CANCEL)

        snapshot = MRRSnapshot(
            date=now.strftime("%Y-%m-%d"), total_mrr=total_mrr,
            new_mrr=new_mrr, expansion_mrr=expansion,
            contraction_mrr=contraction, churn_mrr=churn,
            net_new_mrr=new_mrr + expansion - contraction - churn,
            active_subscriptions=count, arpu=arpu)
        self._mrr_snapshots.append(snapshot)
        return snapshot

    def get_arr(self) -> Decimal:
        if not self._mrr_snapshots:
            self.calculate_mrr()
        return self._mrr_snapshots[-1].total_mrr * 12

    # ---- churn analysis ----
    def calculate_churn_rate(self, *, days: int = 30) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=days)).isoformat()

        start_count = sum(1 for s in self._subscriptions.values()
                          if s.get("started", "") < cutoff)
        churned = sum(1 for e in self._events
                      if e.event_type == RevenueEventType.SUBSCRIPTION_CANCEL
                      and e.timestamp >= cutoff)
        rate = (churned / start_count * 100) if start_count > 0 else 0

        return {"period_days": days, "start_count": start_count,
                "churned": churned, "churn_rate_pct": round(rate, 2),
                "retention_rate_pct": round(100 - rate, 2)}

    # ---- LTV projection ----
    def estimate_ltv(self, tier: str = "pro") -> Dict[str, Any]:
        active = [s for s in self._subscriptions.values()
                  if s.get("tier") == tier and s.get("active")]
        if not active:
            return {"tier": tier, "ltv_usd": 0, "avg_mrr": 0, "avg_lifetime_months": 0}

        avg_mrr = sum(float(s["mrr"]) for s in active) / len(active)
        churn = self.calculate_churn_rate()
        monthly_churn = churn["churn_rate_pct"] / 100
        avg_lifetime = 1 / monthly_churn if monthly_churn > 0 else 36
        ltv = avg_mrr * avg_lifetime

        return {"tier": tier, "ltv_usd": round(ltv, 2),
                "avg_mrr": round(avg_mrr, 2),
                "avg_lifetime_months": round(avg_lifetime, 1)}

    # ---- revenue breakdown ----
    def revenue_by_tier(self) -> Dict[str, Any]:
        by_tier: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "mrr": Decimal("0")})
        for s in self._subscriptions.values():
            if s.get("active"):
                tier = s.get("tier", "unknown")
                by_tier[tier]["count"] += 1
                by_tier[tier]["mrr"] += Decimal(str(s["mrr"]))
        return {t: {"count": d["count"], "mrr": float(d["mrr"]),
                     "arr": float(d["mrr"] * 12)}
                for t, d in by_tier.items()}

    # ---- growth metrics ----
    def growth_rate(self) -> Dict[str, Any]:
        if len(self._mrr_snapshots) < 2:
            return {"monthly_growth_pct": 0, "snapshots": len(self._mrr_snapshots)}
        prev = float(self._mrr_snapshots[-2].total_mrr)
        curr = float(self._mrr_snapshots[-1].total_mrr)
        rate = ((curr - prev) / prev * 100) if prev > 0 else 0
        return {"monthly_growth_pct": round(rate, 2),
                "previous_mrr": prev, "current_mrr": curr}

    # ---- cohort analysis ----
    def build_cohorts(self) -> Dict[str, Dict[str, Any]]:
        by_month: Dict[str, List[str]] = defaultdict(list)
        for tid, sub in self._subscriptions.items():
            month = sub.get("start_month", "unknown")
            by_month[month].append(tid)

        cohorts: Dict[str, Dict[str, Any]] = {}
        for month, tids in sorted(by_month.items()):
            active_now = sum(1 for t in tids if self._subscriptions[t].get("active"))
            cohorts[month] = {
                "initial": len(tids), "active_now": active_now,
                "retention_pct": round(active_now / len(tids) * 100, 1)}
        return cohorts

    # ---- dashboard summary ----
    def get_dashboard(self) -> Dict[str, Any]:
        mrr = self.calculate_mrr()
        return {
            "mrr": mrr.to_dict(),
            "arr": float(self.get_arr()),
            "churn": self.calculate_churn_rate(),
            "by_tier": self.revenue_by_tier(),
            "growth": self.growth_rate(),
            "total_events": len(self._events),
            "total_subscriptions": len(self._subscriptions),
            "active_subscriptions": sum(1 for s in self._subscriptions.values() if s.get("active")),
        }


_engine: RevenueAnalyticsEngine | None = None

def get_revenue_analytics() -> RevenueAnalyticsEngine:
    global _engine
    if not _engine:
        _engine = RevenueAnalyticsEngine()
    return _engine
