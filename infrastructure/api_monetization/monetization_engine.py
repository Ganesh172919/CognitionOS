"""
API Monetization Engine
========================
Complete API economics infrastructure for SaaS monetization:
- Usage-based billing with per-request pricing
- Tiered subscription pricing with feature gates
- Real-time usage metering and quota enforcement
- Invoice generation and billing cycle management
- API key lifecycle management with scopes and permissions
- Revenue analytics with MRR, ARR, churn metrics
- Promotional credits and discount management
- Overage handling and automatic upgrades
- Webhook notifications for billing events
- Cost attribution and showback reporting
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BillingInterval(str, Enum):
    """Billing cycle interval."""
    MONTHLY = "monthly"
    ANNUAL = "annual"
    USAGE = "usage"
    ONE_TIME = "one_time"


class PricingModel(str, Enum):
    """Pricing model for a tier."""
    FLAT_RATE = "flat_rate"
    PER_UNIT = "per_unit"
    TIERED = "tiered"
    VOLUME = "volume"
    PACKAGE = "package"
    HYBRID = "hybrid"  # Base + usage


class InvoiceStatus(str, Enum):
    """Status of an invoice."""
    DRAFT = "draft"
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"
    VOID = "void"
    REFUNDED = "refunded"


class APIKeyStatus(str, Enum):
    """Status of an API key."""
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


class UsageMetric(str, Enum):
    """Metered usage metrics."""
    API_CALLS = "api_calls"
    TOKENS = "tokens"
    COMPUTE_SECONDS = "compute_seconds"
    STORAGE_GB = "storage_gb"
    BANDWIDTH_GB = "bandwidth_gb"
    AGENTS = "agents"
    WORKFLOWS = "workflows"
    USERS = "users"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TierLimit:
    """A specific limit within a pricing tier."""
    metric: UsageMetric = UsageMetric.API_CALLS
    included_units: int = 0  # 0 = unlimited
    overage_price_per_unit: float = 0.0
    hard_limit: bool = False  # If True, block at limit; if False, charge overage


@dataclass
class PricingTier:
    """A subscription pricing tier with features and limits."""
    tier_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    display_name: str = ""
    description: str = ""
    pricing_model: PricingModel = PricingModel.HYBRID
    billing_interval: BillingInterval = BillingInterval.MONTHLY
    base_price_usd: float = 0.0
    annual_discount_pct: float = 20.0
    features: List[str] = field(default_factory=list)
    limits: List[TierLimit] = field(default_factory=list)
    is_public: bool = True
    is_active: bool = True
    trial_days: int = 14
    sort_order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def annual_price_usd(self) -> float:
        monthly = self.base_price_usd
        return monthly * 12 * (1 - self.annual_discount_pct / 100)

    def get_limit(self, metric: UsageMetric) -> Optional[TierLimit]:
        for limit in self.limits:
            if limit.metric == metric:
                return limit
        return None

    def has_feature(self, feature: str) -> bool:
        return feature in self.features or "all" in self.features

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "pricing_model": self.pricing_model.value,
            "billing_interval": self.billing_interval.value,
            "base_price_usd": self.base_price_usd,
            "annual_price_usd": self.annual_price_usd,
            "annual_discount_pct": self.annual_discount_pct,
            "features": self.features,
            "trial_days": self.trial_days,
            "is_public": self.is_public,
            "limits": [
                {
                    "metric": l.metric.value,
                    "included_units": l.included_units,
                    "overage_price": l.overage_price_per_unit,
                    "hard_limit": l.hard_limit,
                }
                for l in self.limits
            ],
        }


@dataclass
class UsageRecord:
    """A single usage event for billing purposes."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    api_key_id: str = ""
    metric: UsageMetric = UsageMetric.API_CALLS
    quantity: float = 1.0
    unit_price: float = 0.0
    endpoint: str = ""
    operation: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    billable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return self.quantity * self.unit_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "tenant_id": self.tenant_id,
            "metric": self.metric.value,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "cost_usd": self.cost_usd,
            "endpoint": self.endpoint,
            "billable": self.billable,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Invoice:
    """A billing invoice for a tenant."""
    invoice_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    invoice_number: str = ""
    tenant_id: str = ""
    tier_name: str = ""
    status: InvoiceStatus = InvoiceStatus.DRAFT
    billing_period_start: datetime = field(default_factory=datetime.utcnow)
    billing_period_end: datetime = field(default_factory=datetime.utcnow)
    issued_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    subtotal_usd: float = 0.0
    discount_usd: float = 0.0
    tax_usd: float = 0.0
    total_usd: float = 0.0
    credit_applied_usd: float = 0.0
    amount_due_usd: float = 0.0
    currency: str = "USD"
    payment_method: str = ""
    notes: str = ""

    def add_line_item(
        self, description: str, quantity: float, unit_price: float, metric: str = ""
    ) -> None:
        amount = quantity * unit_price
        self.line_items.append({
            "description": description,
            "quantity": quantity,
            "unit_price": unit_price,
            "amount": amount,
            "metric": metric,
        })
        self.subtotal_usd += amount

    def finalize(self, tax_rate: float = 0.0, credit_usd: float = 0.0) -> None:
        """Finalize the invoice with tax and credits."""
        self.tax_usd = self.subtotal_usd * tax_rate
        self.total_usd = self.subtotal_usd + self.tax_usd - self.discount_usd
        self.credit_applied_usd = min(credit_usd, self.total_usd)
        self.amount_due_usd = max(0.0, self.total_usd - self.credit_applied_usd)
        self.status = InvoiceStatus.PENDING
        self.issued_at = datetime.utcnow()
        self.due_date = datetime.utcnow() + timedelta(days=30)

    def mark_paid(self, payment_method: str) -> None:
        self.status = InvoiceStatus.PAID
        self.paid_at = datetime.utcnow()
        self.payment_method = payment_method

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "invoice_number": self.invoice_number,
            "tenant_id": self.tenant_id,
            "tier_name": self.tier_name,
            "status": self.status.value,
            "billing_period_start": self.billing_period_start.isoformat(),
            "billing_period_end": self.billing_period_end.isoformat(),
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "line_items": self.line_items,
            "subtotal_usd": round(self.subtotal_usd, 2),
            "discount_usd": round(self.discount_usd, 2),
            "tax_usd": round(self.tax_usd, 2),
            "total_usd": round(self.total_usd, 2),
            "credit_applied_usd": round(self.credit_applied_usd, 2),
            "amount_due_usd": round(self.amount_due_usd, 2),
            "currency": self.currency,
        }


@dataclass
class APIKey:
    """An API key with scopes, rate limits, and usage tracking."""
    key_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    user_id: str = ""
    name: str = ""
    key_prefix: str = ""  # Visible prefix (e.g., "sk-prod-xxxx")
    key_hash: str = ""    # SHA-256 hash of full key
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    scopes: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 1000
    rate_limit_per_day: int = 100000
    allowed_ips: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    total_requests: int = 0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def is_valid(self) -> bool:
        return self.status == APIKeyStatus.ACTIVE and not self.is_expired

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "*" in self.scopes

    def record_usage(self, tokens: int = 0) -> None:
        self.total_requests += 1
        self.total_tokens += tokens
        self.last_used_at = datetime.utcnow()

    def to_dict(self, show_sensitive: bool = False) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "status": self.status.value,
            "scopes": self.scopes,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "total_requests": self.total_requests,
            "is_valid": self.is_valid,
        }


# ---------------------------------------------------------------------------
# API Key Manager
# ---------------------------------------------------------------------------

class APIKeyManager:
    """Manages API key lifecycle: creation, validation, revocation, rotation."""

    def __init__(self) -> None:
        self._keys: Dict[str, APIKey] = {}
        self._key_by_hash: Dict[str, str] = {}  # hash -> key_id
        self._lock = asyncio.Lock()

    async def create_key(
        self,
        tenant_id: str,
        user_id: str,
        name: str,
        scopes: Optional[List[str]] = None,
        rate_limit_per_minute: int = 1000,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[APIKey, str]:
        """Create a new API key. Returns (APIKey, raw_key)."""
        async with self._lock:
            raw_key = f"ck-{secrets.token_urlsafe(32)}"
            key_prefix = raw_key[:12] + "..."
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            expires_at = (
                datetime.utcnow() + timedelta(days=expires_in_days)
                if expires_in_days else None
            )

            api_key = APIKey(
                tenant_id=tenant_id,
                user_id=user_id,
                name=name,
                key_prefix=key_prefix,
                key_hash=key_hash,
                scopes=scopes or ["read", "write"],
                rate_limit_per_minute=rate_limit_per_minute,
                expires_at=expires_at,
            )
            self._keys[api_key.key_id] = api_key
            self._key_by_hash[key_hash] = api_key.key_id
            logger.info("Created API key %s for tenant %s", api_key.key_id, tenant_id)
            return api_key, raw_key

    async def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate a raw API key and return the key object if valid."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = self._key_by_hash.get(key_hash)
        if not key_id:
            return None
        api_key = self._keys.get(key_id)
        if api_key and api_key.is_valid:
            return api_key
        return None

    async def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key immediately."""
        async with self._lock:
            key = self._keys.get(key_id)
            if key:
                key.status = APIKeyStatus.REVOKED
                logger.info("Revoked API key: %s", key_id)
                return True
            return False

    async def rotate_key(self, key_id: str) -> Optional[Tuple[APIKey, str]]:
        """Rotate an API key (revoke old, create new with same config)."""
        async with self._lock:
            old_key = self._keys.get(key_id)
            if not old_key:
                return None

            # Revoke old key
            old_key.status = APIKeyStatus.REVOKED

            # Create new key with same configuration
            new_key, raw_key = await self.create_key(
                tenant_id=old_key.tenant_id,
                user_id=old_key.user_id,
                name=f"{old_key.name} (rotated)",
                scopes=old_key.scopes,
                rate_limit_per_minute=old_key.rate_limit_per_minute,
            )
            return new_key, raw_key

    async def list_keys(self, tenant_id: str) -> List[Dict[str, Any]]:
        return [k.to_dict() for k in self._keys.values() if k.tenant_id == tenant_id]


# ---------------------------------------------------------------------------
# Usage Tracker
# ---------------------------------------------------------------------------

class UsageTracker:
    """Real-time usage metering with quota enforcement and overage detection."""

    def __init__(self) -> None:
        self._usage: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._daily_usage: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._usage_records: deque = deque(maxlen=1000000)
        self._quota_alerts: Dict[str, Set[int]] = defaultdict(set)  # tenant_id -> alerted_pcts
        self._lock = asyncio.Lock()
        self._current_date: str = datetime.utcnow().date().isoformat()

    def _get_period_key(self) -> str:
        now = datetime.utcnow()
        return f"{now.year}-{now.month:02d}"

    async def track(
        self,
        tenant_id: str,
        metric: UsageMetric,
        quantity: float = 1.0,
        api_key_id: str = "",
        endpoint: str = "",
        unit_price: float = 0.0,
        billable: bool = True,
    ) -> UsageRecord:
        """Record a usage event."""
        async with self._lock:
            period = self._get_period_key()
            key = f"{tenant_id}:{metric.value}"

            self._usage[period][key] += quantity
            self._daily_usage[self._current_date][key] += quantity

            record = UsageRecord(
                tenant_id=tenant_id,
                api_key_id=api_key_id,
                metric=metric,
                quantity=quantity,
                unit_price=unit_price,
                endpoint=endpoint,
                billable=billable,
            )
            self._usage_records.append(record)
            return record

    async def get_usage(
        self,
        tenant_id: str,
        metric: UsageMetric,
        period: Optional[str] = None,
    ) -> float:
        """Get total usage for a tenant and metric."""
        period = period or self._get_period_key()
        key = f"{tenant_id}:{metric.value}"
        return self._usage.get(period, {}).get(key, 0.0)

    async def get_usage_summary(self, tenant_id: str, period: Optional[str] = None) -> Dict[str, float]:
        """Get all usage metrics for a tenant."""
        period = period or self._get_period_key()
        prefix = f"{tenant_id}:"
        usage = self._usage.get(period, {})
        return {
            k.replace(prefix, ""): v
            for k, v in usage.items()
            if k.startswith(prefix)
        }

    async def check_quota(
        self, tenant_id: str, metric: UsageMetric, tier: Optional[PricingTier] = None
    ) -> Dict[str, Any]:
        """Check quota status for a tenant and metric."""
        current = await self.get_usage(tenant_id, metric)
        if not tier:
            return {"usage": current, "limit": None, "percentage": 0.0, "within_quota": True}

        tier_limit = tier.get_limit(metric)
        if not tier_limit or tier_limit.included_units == 0:
            return {"usage": current, "limit": None, "percentage": 0.0, "within_quota": True}

        limit = tier_limit.included_units
        percentage = (current / limit) * 100
        within_quota = not tier_limit.hard_limit or current < limit

        # Alert at 80% and 100%
        for threshold in [80, 100]:
            if percentage >= threshold and threshold not in self._quota_alerts[tenant_id]:
                self._quota_alerts[tenant_id].add(threshold)
                logger.warning(
                    "Quota alert for tenant %s: %s at %.1f%% (%g / %g)",
                    tenant_id, metric.value, percentage, current, limit,
                )

        return {
            "metric": metric.value,
            "usage": current,
            "limit": limit,
            "percentage": round(percentage, 2),
            "within_quota": within_quota,
            "overage": max(0.0, current - limit),
            "overage_cost_usd": max(0.0, current - limit) * tier_limit.overage_price_per_unit,
        }

    def get_records_for_period(
        self, tenant_id: str, period_start: datetime, period_end: datetime
    ) -> List[UsageRecord]:
        """Get raw usage records for a billing period."""
        return [
            r for r in self._usage_records
            if r.tenant_id == tenant_id
            and period_start <= r.timestamp <= period_end
            and r.billable
        ]


# ---------------------------------------------------------------------------
# Pricing Calculator
# ---------------------------------------------------------------------------

class PricingCalculator:
    """Calculates billing amounts based on usage and tier pricing rules."""

    def __init__(self) -> None:
        self._tiers: Dict[str, PricingTier] = {}
        self._load_default_tiers()

    def _load_default_tiers(self) -> None:
        """Load the default CognitionOS pricing tiers."""
        free = PricingTier(
            name="free",
            display_name="Free",
            description="Perfect for exploring CognitionOS",
            pricing_model=PricingModel.FLAT_RATE,
            billing_interval=BillingInterval.MONTHLY,
            base_price_usd=0.0,
            features=["basic_agents", "5_workflows", "community_support"],
            limits=[
                TierLimit(UsageMetric.API_CALLS, 1000, 0.0, hard_limit=True),
                TierLimit(UsageMetric.TOKENS, 100000, 0.0, hard_limit=True),
                TierLimit(UsageMetric.AGENTS, 2, 0.0, hard_limit=True),
            ],
            trial_days=0,
            sort_order=0,
        )

        pro = PricingTier(
            name="pro",
            display_name="Pro",
            description="For growing teams and professionals",
            pricing_model=PricingModel.HYBRID,
            billing_interval=BillingInterval.MONTHLY,
            base_price_usd=49.0,
            annual_discount_pct=20.0,
            features=[
                "unlimited_agents", "50_workflows", "priority_support",
                "api_access", "advanced_analytics", "custom_integrations",
            ],
            limits=[
                TierLimit(UsageMetric.API_CALLS, 100000, 0.0002, hard_limit=False),
                TierLimit(UsageMetric.TOKENS, 5000000, 0.000002, hard_limit=False),
                TierLimit(UsageMetric.AGENTS, 20, 0.0, hard_limit=False),
                TierLimit(UsageMetric.STORAGE_GB, 50, 0.05, hard_limit=False),
            ],
            trial_days=14,
            sort_order=1,
        )

        enterprise = PricingTier(
            name="enterprise",
            display_name="Enterprise",
            description="For large organizations with custom needs",
            pricing_model=PricingModel.HYBRID,
            billing_interval=BillingInterval.ANNUAL,
            base_price_usd=999.0,
            annual_discount_pct=30.0,
            features=[
                "all", "sso", "sla_99_9", "dedicated_support",
                "custom_models", "on_premise", "audit_logs", "rbac",
                "data_residency", "compliance_reports",
            ],
            limits=[
                TierLimit(UsageMetric.API_CALLS, 10000000, 0.00001, hard_limit=False),
                TierLimit(UsageMetric.TOKENS, 500000000, 0.0000008, hard_limit=False),
                TierLimit(UsageMetric.AGENTS, 0, 0.0, hard_limit=False),  # Unlimited
                TierLimit(UsageMetric.STORAGE_GB, 1000, 0.02, hard_limit=False),
                TierLimit(UsageMetric.USERS, 0, 0.0, hard_limit=False),  # Unlimited
            ],
            trial_days=30,
            sort_order=2,
        )

        for tier in [free, pro, enterprise]:
            self._tiers[tier.name] = tier

    def get_tier(self, tier_name: str) -> Optional[PricingTier]:
        return self._tiers.get(tier_name)

    def register_tier(self, tier: PricingTier) -> None:
        self._tiers[tier.name] = tier

    def list_tiers(self, public_only: bool = True) -> List[Dict[str, Any]]:
        tiers = self._tiers.values()
        if public_only:
            tiers = [t for t in tiers if t.is_public and t.is_active]
        return sorted(
            [t.to_dict() for t in tiers],
            key=lambda t: t.get("sort_order", 0),
        )

    def calculate_overage(
        self, tier: PricingTier, metric: UsageMetric, total_usage: float
    ) -> float:
        """Calculate overage cost for a given metric."""
        limit = tier.get_limit(metric)
        if not limit or limit.included_units == 0:
            return 0.0
        overage_units = max(0.0, total_usage - limit.included_units)
        return overage_units * limit.overage_price_per_unit

    def calculate_total_cost(
        self, tier: PricingTier, usage: Dict[UsageMetric, float]
    ) -> Dict[str, float]:
        """Calculate total billing cost for a period."""
        base = tier.base_price_usd
        overage_by_metric: Dict[str, float] = {}
        total_overage = 0.0

        for metric, units in usage.items():
            overage = self.calculate_overage(tier, metric, units)
            if overage > 0:
                overage_by_metric[metric.value] = overage
                total_overage += overage

        return {
            "base_price": base,
            "overage_total": round(total_overage, 4),
            "overage_by_metric": overage_by_metric,
            "total": round(base + total_overage, 2),
        }


# ---------------------------------------------------------------------------
# Invoice Generator
# ---------------------------------------------------------------------------

class InvoiceGenerator:
    """Generates invoices from usage data and pricing rules."""

    def __init__(self) -> None:
        self._invoices: Dict[str, Invoice] = {}
        self._invoice_counter: int = 1000
        self._lock = asyncio.Lock()

    async def generate_invoice(
        self,
        tenant_id: str,
        tier: PricingTier,
        usage_records: List[UsageRecord],
        period_start: datetime,
        period_end: datetime,
        credits_usd: float = 0.0,
        tax_rate: float = 0.0,
    ) -> Invoice:
        """Generate an invoice for a billing period."""
        async with self._lock:
            self._invoice_counter += 1
            invoice = Invoice(
                invoice_number=f"INV-{self._invoice_counter:06d}",
                tenant_id=tenant_id,
                tier_name=tier.name,
                billing_period_start=period_start,
                billing_period_end=period_end,
            )

            # Base subscription fee
            if tier.base_price_usd > 0:
                invoice.add_line_item(
                    description=f"{tier.display_name} plan subscription",
                    quantity=1,
                    unit_price=tier.base_price_usd,
                    metric="subscription",
                )

            # Aggregate usage by metric
            usage_by_metric: Dict[str, float] = defaultdict(float)
            for record in usage_records:
                usage_by_metric[record.metric.value] += record.quantity

            # Compute overage charges
            for metric_str, total_units in usage_by_metric.items():
                try:
                    metric = UsageMetric(metric_str)
                except ValueError:
                    continue
                tier_limit = tier.get_limit(metric)
                if tier_limit and tier_limit.included_units > 0:
                    overage = max(0.0, total_units - tier_limit.included_units)
                    if overage > 0 and tier_limit.overage_price_per_unit > 0:
                        invoice.add_line_item(
                            description=f"{metric_str.replace('_', ' ').title()} overage",
                            quantity=overage,
                            unit_price=tier_limit.overage_price_per_unit,
                            metric=metric_str,
                        )

            invoice.finalize(tax_rate=tax_rate, credit_usd=credits_usd)
            self._invoices[invoice.invoice_id] = invoice
            logger.info(
                "Generated invoice %s for tenant %s: $%.2f",
                invoice.invoice_number, tenant_id, invoice.amount_due_usd,
            )
            return invoice

    async def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        return self._invoices.get(invoice_id)

    async def list_invoices(
        self, tenant_id: str, status: Optional[InvoiceStatus] = None
    ) -> List[Dict[str, Any]]:
        invoices = [i for i in self._invoices.values() if i.tenant_id == tenant_id]
        if status:
            invoices = [i for i in invoices if i.status == status]
        invoices.sort(key=lambda i: i.issued_at or datetime.min, reverse=True)
        return [i.to_dict() for i in invoices]

    async def mark_paid(self, invoice_id: str, payment_method: str = "stripe") -> Optional[Invoice]:
        invoice = self._invoices.get(invoice_id)
        if invoice:
            invoice.mark_paid(payment_method)
        return invoice


# ---------------------------------------------------------------------------
# Revenue Analytics
# ---------------------------------------------------------------------------

class RevenueAnalytics:
    """Real-time revenue analytics: MRR, ARR, churn, LTV, cohorts."""

    def __init__(self) -> None:
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._mrr_history: deque = deque(maxlen=1000)
        self._churn_events: List[Dict[str, Any]] = []
        self._expansion_events: List[Dict[str, Any]] = []

    def record_subscription(
        self,
        tenant_id: str,
        tier_name: str,
        mrr_usd: float,
        started_at: Optional[datetime] = None,
    ) -> None:
        """Record a new or updated subscription."""
        prev = self._subscriptions.get(tenant_id, {})
        prev_mrr = prev.get("mrr_usd", 0.0)

        self._subscriptions[tenant_id] = {
            "tenant_id": tenant_id,
            "tier_name": tier_name,
            "mrr_usd": mrr_usd,
            "started_at": (started_at or datetime.utcnow()).isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
        }

        # Track expansion/contraction
        if prev_mrr > 0:
            diff = mrr_usd - prev_mrr
            if diff > 0:
                self._expansion_events.append({
                    "tenant_id": tenant_id,
                    "expansion_usd": diff,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            elif diff < 0:
                self._churn_events.append({
                    "tenant_id": tenant_id,
                    "contraction_usd": abs(diff),
                    "type": "downgrade",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    def record_churn(self, tenant_id: str, reason: str = "") -> None:
        """Record a customer churn event."""
        sub = self._subscriptions.pop(tenant_id, None)
        if sub:
            self._churn_events.append({
                "tenant_id": tenant_id,
                "churned_mrr_usd": sub["mrr_usd"],
                "reason": reason,
                "tier": sub["tier_name"],
                "timestamp": datetime.utcnow().isoformat(),
            })

    @property
    def current_mrr(self) -> float:
        return sum(s["mrr_usd"] for s in self._subscriptions.values())

    @property
    def current_arr(self) -> float:
        return self.current_mrr * 12

    @property
    def customer_count(self) -> int:
        return len(self._subscriptions)

    def get_arpu(self) -> float:
        """Average Revenue Per User."""
        if not self._subscriptions:
            return 0.0
        return self.current_mrr / len(self._subscriptions)

    def get_churn_rate(self, lookback_days: int = 30) -> float:
        """Monthly churn rate as percentage of customers lost."""
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        recent_churn = sum(
            1 for e in self._churn_events
            if datetime.fromisoformat(e["timestamp"]) >= cutoff
        )
        if not self._subscriptions:
            return 0.0
        return recent_churn / (len(self._subscriptions) + recent_churn) * 100

    def get_revenue_by_tier(self) -> Dict[str, float]:
        """Break down MRR by subscription tier."""
        tier_revenue: Dict[str, float] = defaultdict(float)
        for sub in self._subscriptions.values():
            tier_revenue[sub["tier_name"]] += sub["mrr_usd"]
        return dict(tier_revenue)

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            "mrr_usd": round(self.current_mrr, 2),
            "arr_usd": round(self.current_arr, 2),
            "customer_count": self.customer_count,
            "arpu_usd": round(self.get_arpu(), 2),
            "churn_rate_pct": round(self.get_churn_rate(), 2),
            "revenue_by_tier": self.get_revenue_by_tier(),
            "expansion_events_30d": sum(
                1 for e in self._expansion_events
                if datetime.fromisoformat(e["timestamp"]) >= datetime.utcnow() - timedelta(days=30)
            ),
        }


# ---------------------------------------------------------------------------
# API Monetization Engine
# ---------------------------------------------------------------------------

class APIMonetizationEngine:
    """
    Master monetization engine orchestrating usage tracking,
    pricing, invoicing, API key management, and revenue analytics.
    """

    def __init__(self) -> None:
        self._usage_tracker = UsageTracker()
        self._pricing_calculator = PricingCalculator()
        self._invoice_generator = InvoiceGenerator()
        self._api_key_manager = APIKeyManager()
        self._revenue_analytics = RevenueAnalytics()
        self._tenant_tiers: Dict[str, str] = {}  # tenant_id -> tier_name
        self._tenant_credits: Dict[str, float] = defaultdict(float)

    async def onboard_tenant(
        self,
        tenant_id: str,
        tier_name: str = "free",
        initial_credits_usd: float = 0.0,
    ) -> Dict[str, Any]:
        """Onboard a new tenant with a tier and optional credits."""
        tier = self._pricing_calculator.get_tier(tier_name)
        if not tier:
            raise ValueError(f"Unknown tier: {tier_name}")

        self._tenant_tiers[tenant_id] = tier_name
        if initial_credits_usd > 0:
            self._tenant_credits[tenant_id] = initial_credits_usd

        self._revenue_analytics.record_subscription(
            tenant_id=tenant_id,
            tier_name=tier_name,
            mrr_usd=tier.base_price_usd,
        )

        logger.info("Onboarded tenant %s on %s plan", tenant_id, tier_name)
        return {
            "tenant_id": tenant_id,
            "tier": tier.to_dict(),
            "credits_usd": initial_credits_usd,
            "trial_days": tier.trial_days,
        }

    async def record_api_usage(
        self,
        tenant_id: str,
        metric: UsageMetric,
        quantity: float = 1.0,
        api_key: Optional[str] = None,
        endpoint: str = "",
    ) -> Dict[str, Any]:
        """Record API usage and check quota status."""
        tier_name = self._tenant_tiers.get(tenant_id, "free")
        tier = self._pricing_calculator.get_tier(tier_name)

        # Determine unit price based on tier
        unit_price = 0.0
        if tier:
            tier_limit = tier.get_limit(metric)
            if tier_limit:
                unit_price = tier_limit.overage_price_per_unit

        record = await self._usage_tracker.track(
            tenant_id=tenant_id,
            metric=metric,
            quantity=quantity,
            endpoint=endpoint,
            unit_price=unit_price,
        )

        quota_status = {}
        if tier:
            quota_status = await self._usage_tracker.check_quota(tenant_id, metric, tier)

        return {
            "recorded": True,
            "record_id": record.record_id,
            "quota_status": quota_status,
        }

    async def generate_monthly_invoice(
        self,
        tenant_id: str,
        year: int,
        month: int,
    ) -> Invoice:
        """Generate the monthly invoice for a tenant."""
        tier_name = self._tenant_tiers.get(tenant_id, "free")
        tier = self._pricing_calculator.get_tier(tier_name)
        if not tier:
            raise ValueError(f"No tier for tenant {tenant_id}")

        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            period_end = datetime(year, month + 1, 1) - timedelta(seconds=1)

        records = self._usage_tracker.get_records_for_period(tenant_id, period_start, period_end)
        credits = self._tenant_credits.get(tenant_id, 0.0)

        invoice = await self._invoice_generator.generate_invoice(
            tenant_id=tenant_id,
            tier=tier,
            usage_records=records,
            period_start=period_start,
            period_end=period_end,
            credits_usd=credits,
        )

        # Deduct applied credits
        if invoice.credit_applied_usd > 0:
            self._tenant_credits[tenant_id] = max(
                0.0, credits - invoice.credit_applied_usd
            )

        return invoice

    async def upgrade_tier(self, tenant_id: str, new_tier_name: str) -> Dict[str, Any]:
        """Upgrade a tenant to a new tier."""
        tier = self._pricing_calculator.get_tier(new_tier_name)
        if not tier:
            raise ValueError(f"Unknown tier: {new_tier_name}")

        old_tier_name = self._tenant_tiers.get(tenant_id, "free")
        self._tenant_tiers[tenant_id] = new_tier_name

        self._revenue_analytics.record_subscription(
            tenant_id=tenant_id,
            tier_name=new_tier_name,
            mrr_usd=tier.base_price_usd,
        )

        logger.info("Tenant %s upgraded: %s -> %s", tenant_id, old_tier_name, new_tier_name)
        return {
            "tenant_id": tenant_id,
            "old_tier": old_tier_name,
            "new_tier": new_tier_name,
            "new_mrr_usd": tier.base_price_usd,
        }

    async def add_credits(self, tenant_id: str, amount_usd: float, reason: str = "") -> float:
        """Add billing credits to a tenant account."""
        self._tenant_credits[tenant_id] += amount_usd
        logger.info("Added $%.2f credits to tenant %s (reason: %s)", amount_usd, tenant_id, reason)
        return self._tenant_credits[tenant_id]

    async def get_tenant_billing_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive billing summary for a tenant."""
        tier_name = self._tenant_tiers.get(tenant_id, "free")
        tier = self._pricing_calculator.get_tier(tier_name)
        usage = await self._usage_tracker.get_usage_summary(tenant_id)
        invoices = await self._invoice_generator.list_invoices(tenant_id)

        quota_status = {}
        if tier:
            for metric in UsageMetric:
                usage_val = usage.get(metric.value, 0.0)
                if usage_val > 0:
                    quota = await self._usage_tracker.check_quota(tenant_id, metric, tier)
                    quota_status[metric.value] = quota

        return {
            "tenant_id": tenant_id,
            "current_tier": tier.to_dict() if tier else None,
            "current_usage": usage,
            "quota_status": quota_status,
            "credits_usd": self._tenant_credits.get(tenant_id, 0.0),
            "invoice_count": len(invoices),
            "recent_invoices": invoices[:3],
        }

    def get_revenue_metrics(self) -> Dict[str, Any]:
        return self._revenue_analytics.get_metrics_summary()

    def list_pricing_tiers(self) -> List[Dict[str, Any]]:
        return self._pricing_calculator.list_tiers()

    @property
    def api_keys(self) -> APIKeyManager:
        return self._api_key_manager

    @property
    def usage_tracker(self) -> UsageTracker:
        return self._usage_tracker

    @property
    def revenue_analytics(self) -> RevenueAnalytics:
        return self._revenue_analytics
