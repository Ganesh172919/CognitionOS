"""
Advanced Usage-Based Billing Engine
Real-time metering and billing for SaaS platform with multiple pricing models.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class UsageMetricType(str, Enum):
    """Types of usage metrics"""
    API_CALLS = "api_calls"
    COMPUTE_MINUTES = "compute_minutes"
    STORAGE_GB = "storage_gb"
    BANDWIDTH_GB = "bandwidth_gb"
    LLM_TOKENS = "llm_tokens"
    WORKFLOW_EXECUTIONS = "workflow_executions"
    AGENT_HOURS = "agent_hours"
    DATABASE_QUERIES = "database_queries"


class PricingModel(str, Enum):
    """Pricing model types"""
    TIERED = "tiered"  # Price per unit decreases with volume
    VOLUME = "volume"  # All units priced at tier rate
    FLAT = "flat"  # Fixed price per unit
    GRADUATED = "graduated"  # Price changes at thresholds


class BillingPeriod(str, Enum):
    """Billing period frequencies"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    USAGE_BASED = "usage_based"


class UsageRecord(BaseModel):
    """Individual usage record"""
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str
    user_id: Optional[str] = None
    metric_type: UsageMetricType
    quantity: Decimal
    unit_cost: Decimal = Decimal("0.0")
    total_cost: Decimal = Decimal("0.0")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    billed: bool = False


class PricingTier(BaseModel):
    """Pricing tier configuration"""
    tier_name: str
    start_quantity: int
    end_quantity: Optional[int] = None  # None means unlimited
    price_per_unit: Decimal
    flat_fee: Decimal = Decimal("0.0")


class MetricPricing(BaseModel):
    """Pricing configuration for a metric"""
    metric_type: UsageMetricType
    pricing_model: PricingModel
    tiers: List[PricingTier]
    minimum_charge: Decimal = Decimal("0.0")
    maximum_charge: Optional[Decimal] = None


class UsageAggregation(BaseModel):
    """Aggregated usage for billing period"""
    tenant_id: str
    metric_type: UsageMetricType
    period_start: datetime
    period_end: datetime
    total_quantity: Decimal
    total_cost: Decimal
    records_count: int


class Invoice(BaseModel):
    """Customer invoice"""
    invoice_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str
    billing_period: BillingPeriod
    period_start: datetime
    period_end: datetime
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    subtotal: Decimal = Decimal("0.0")
    tax: Decimal = Decimal("0.0")
    total: Decimal = Decimal("0.0")
    status: str = "draft"  # draft, sent, paid, overdue
    created_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None


class UsageBasedBillingEngine:
    """
    Advanced billing engine supporting multiple pricing models
    and real-time usage metering.
    """

    def __init__(self):
        self.usage_records: Dict[str, List[UsageRecord]] = {}
        self.pricing_configs: Dict[UsageMetricType, MetricPricing] = {}
        self.invoices: Dict[str, Invoice] = {}
        self._initialize_default_pricing()

    def _initialize_default_pricing(self):
        """Initialize default pricing configurations"""

        # API Calls - Tiered pricing
        self.pricing_configs[UsageMetricType.API_CALLS] = MetricPricing(
            metric_type=UsageMetricType.API_CALLS,
            pricing_model=PricingModel.TIERED,
            tiers=[
                PricingTier(
                    tier_name="First 10K",
                    start_quantity=0,
                    end_quantity=10000,
                    price_per_unit=Decimal("0.001")
                ),
                PricingTier(
                    tier_name="Next 90K",
                    start_quantity=10001,
                    end_quantity=100000,
                    price_per_unit=Decimal("0.0008")
                ),
                PricingTier(
                    tier_name="Above 100K",
                    start_quantity=100001,
                    end_quantity=None,
                    price_per_unit=Decimal("0.0005")
                )
            ]
        )

        # LLM Tokens - Graduated pricing
        self.pricing_configs[UsageMetricType.LLM_TOKENS] = MetricPricing(
            metric_type=UsageMetricType.LLM_TOKENS,
            pricing_model=PricingModel.GRADUATED,
            tiers=[
                PricingTier(
                    tier_name="Free tier",
                    start_quantity=0,
                    end_quantity=100000,
                    price_per_unit=Decimal("0.0")
                ),
                PricingTier(
                    tier_name="Standard",
                    start_quantity=100001,
                    end_quantity=1000000,
                    price_per_unit=Decimal("0.00002")
                ),
                PricingTier(
                    tier_name="Volume",
                    start_quantity=1000001,
                    end_quantity=None,
                    price_per_unit=Decimal("0.00001")
                )
            ],
            minimum_charge=Decimal("5.0")
        )

        # Compute Minutes - Flat rate
        self.pricing_configs[UsageMetricType.COMPUTE_MINUTES] = MetricPricing(
            metric_type=UsageMetricType.COMPUTE_MINUTES,
            pricing_model=PricingModel.FLAT,
            tiers=[
                PricingTier(
                    tier_name="Standard",
                    start_quantity=0,
                    end_quantity=None,
                    price_per_unit=Decimal("0.10")
                )
            ]
        )

        # Workflow Executions - Volume pricing
        self.pricing_configs[UsageMetricType.WORKFLOW_EXECUTIONS] = MetricPricing(
            metric_type=UsageMetricType.WORKFLOW_EXECUTIONS,
            pricing_model=PricingModel.VOLUME,
            tiers=[
                PricingTier(
                    tier_name="Small",
                    start_quantity=0,
                    end_quantity=100,
                    price_per_unit=Decimal("0.50")
                ),
                PricingTier(
                    tier_name="Medium",
                    start_quantity=101,
                    end_quantity=1000,
                    price_per_unit=Decimal("0.40")
                ),
                PricingTier(
                    tier_name="Large",
                    start_quantity=1001,
                    end_quantity=None,
                    price_per_unit=Decimal("0.30")
                )
            ]
        )

    async def record_usage(
        self,
        tenant_id: str,
        metric_type: UsageMetricType,
        quantity: Decimal,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageRecord:
        """
        Record a usage event in real-time
        """
        # Calculate cost immediately
        unit_cost, total_cost = await self._calculate_incremental_cost(
            tenant_id,
            metric_type,
            quantity
        )

        record = UsageRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            metric_type=metric_type,
            quantity=quantity,
            unit_cost=unit_cost,
            total_cost=total_cost,
            metadata=metadata or {}
        )

        # Store record
        if tenant_id not in self.usage_records:
            self.usage_records[tenant_id] = []

        self.usage_records[tenant_id].append(record)

        return record

    async def _calculate_incremental_cost(
        self,
        tenant_id: str,
        metric_type: UsageMetricType,
        quantity: Decimal
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate incremental cost for new usage
        """
        if metric_type not in self.pricing_configs:
            return Decimal("0.0"), Decimal("0.0")

        pricing = self.pricing_configs[metric_type]

        # Get current usage for the period
        current_usage = await self._get_current_period_usage(tenant_id, metric_type)

        # Calculate cost based on pricing model
        if pricing.pricing_model == PricingModel.FLAT:
            unit_cost = pricing.tiers[0].price_per_unit
            total_cost = quantity * unit_cost

        elif pricing.pricing_model == PricingModel.TIERED:
            total_cost = self._calculate_tiered_cost(
                current_usage,
                quantity,
                pricing.tiers
            )
            unit_cost = total_cost / quantity if quantity > 0 else Decimal("0.0")

        elif pricing.pricing_model == PricingModel.VOLUME:
            # Volume pricing - all units at the tier rate
            new_total = current_usage + quantity
            tier = self._find_tier(new_total, pricing.tiers)
            unit_cost = tier.price_per_unit
            total_cost = quantity * unit_cost

        elif pricing.pricing_model == PricingModel.GRADUATED:
            total_cost = self._calculate_graduated_cost(
                current_usage,
                quantity,
                pricing.tiers
            )
            unit_cost = total_cost / quantity if quantity > 0 else Decimal("0.0")

        else:
            unit_cost = Decimal("0.0")
            total_cost = Decimal("0.0")

        return unit_cost, total_cost

    def _calculate_tiered_cost(
        self,
        current_usage: Decimal,
        new_quantity: Decimal,
        tiers: List[PricingTier]
    ) -> Decimal:
        """Calculate cost using tiered pricing"""
        total_cost = Decimal("0.0")
        remaining = new_quantity
        current_position = current_usage

        for tier in tiers:
            if remaining <= 0:
                break

            tier_start = Decimal(str(tier.start_quantity))
            tier_end = Decimal(str(tier.end_quantity)) if tier.end_quantity else Decimal("inf")

            # Skip if we're already past this tier
            if current_position >= tier_end:
                continue

            # Calculate how much of this tier we use
            tier_usage_start = max(current_position, tier_start)
            tier_usage_end = min(current_position + remaining, tier_end)
            tier_quantity = tier_usage_end - tier_usage_start

            if tier_quantity > 0:
                tier_cost = tier_quantity * tier.price_per_unit
                total_cost += tier_cost
                remaining -= tier_quantity
                current_position = tier_usage_end

        return total_cost

    def _calculate_graduated_cost(
        self,
        current_usage: Decimal,
        new_quantity: Decimal,
        tiers: List[PricingTier]
    ) -> Decimal:
        """Calculate cost using graduated pricing"""
        # Similar to tiered, but each unit within a tier has the same price
        return self._calculate_tiered_cost(current_usage, new_quantity, tiers)

    def _find_tier(self, quantity: Decimal, tiers: List[PricingTier]) -> PricingTier:
        """Find applicable tier for given quantity"""
        for tier in tiers:
            if tier.end_quantity is None:
                return tier
            if quantity <= tier.end_quantity:
                return tier
        return tiers[-1]

    async def _get_current_period_usage(
        self,
        tenant_id: str,
        metric_type: UsageMetricType
    ) -> Decimal:
        """Get current billing period usage"""
        if tenant_id not in self.usage_records:
            return Decimal("0.0")

        # Get current period start (simplified - use month start)
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Sum usage for current period
        total = Decimal("0.0")
        for record in self.usage_records[tenant_id]:
            if (
                record.metric_type == metric_type and
                record.timestamp >= period_start and
                not record.billed
            ):
                total += record.quantity

        return total

    async def get_usage_summary(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[UsageMetricType, UsageAggregation]:
        """
        Get aggregated usage summary for period
        """
        summary = {}

        if tenant_id not in self.usage_records:
            return summary

        # Group by metric type
        metric_records: Dict[UsageMetricType, List[UsageRecord]] = {}

        for record in self.usage_records[tenant_id]:
            if period_start <= record.timestamp < period_end:
                if record.metric_type not in metric_records:
                    metric_records[record.metric_type] = []
                metric_records[record.metric_type].append(record)

        # Aggregate each metric
        for metric_type, records in metric_records.items():
            total_quantity = sum(r.quantity for r in records)
            total_cost = sum(r.total_cost for r in records)

            summary[metric_type] = UsageAggregation(
                tenant_id=tenant_id,
                metric_type=metric_type,
                period_start=period_start,
                period_end=period_end,
                total_quantity=total_quantity,
                total_cost=total_cost,
                records_count=len(records)
            )

        return summary

    async def generate_invoice(
        self,
        tenant_id: str,
        billing_period: BillingPeriod,
        period_start: datetime,
        period_end: datetime
    ) -> Invoice:
        """
        Generate invoice for billing period
        """
        # Get usage summary
        usage_summary = await self.get_usage_summary(
            tenant_id,
            period_start,
            period_end
        )

        # Create invoice
        invoice = Invoice(
            tenant_id=tenant_id,
            billing_period=billing_period,
            period_start=period_start,
            period_end=period_end
        )

        # Add line items
        subtotal = Decimal("0.0")

        for metric_type, aggregation in usage_summary.items():
            pricing = self.pricing_configs.get(metric_type)

            # Apply minimum charge if configured
            cost = aggregation.total_cost
            if pricing and pricing.minimum_charge > cost:
                cost = pricing.minimum_charge

            # Apply maximum charge if configured
            if pricing and pricing.maximum_charge and cost > pricing.maximum_charge:
                cost = pricing.maximum_charge

            line_item = {
                "description": f"{metric_type.value.replace('_', ' ').title()}",
                "metric_type": metric_type.value,
                "quantity": float(aggregation.total_quantity),
                "unit_cost": float(aggregation.total_cost / aggregation.total_quantity)
                    if aggregation.total_quantity > 0 else 0.0,
                "amount": float(cost)
            }

            invoice.line_items.append(line_item)
            subtotal += cost

        # Calculate tax (simplified - 10%)
        tax = subtotal * Decimal("0.10")

        invoice.subtotal = subtotal
        invoice.tax = tax
        invoice.total = subtotal + tax
        invoice.status = "draft"
        invoice.due_date = period_end + timedelta(days=30)

        # Store invoice
        self.invoices[invoice.invoice_id] = invoice

        # Mark records as billed
        if tenant_id in self.usage_records:
            for record in self.usage_records[tenant_id]:
                if period_start <= record.timestamp < period_end:
                    record.billed = True

        return invoice

    async def get_tenant_invoices(
        self,
        tenant_id: str,
        limit: int = 10
    ) -> List[Invoice]:
        """Get invoices for tenant"""
        invoices = [
            inv for inv in self.invoices.values()
            if inv.tenant_id == tenant_id
        ]

        # Sort by created_at desc
        invoices.sort(key=lambda x: x.created_at, reverse=True)

        return invoices[:limit]

    async def get_current_usage_cost(
        self,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Get current month-to-date usage and estimated cost
        """
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        usage_summary = await self.get_usage_summary(
            tenant_id,
            period_start,
            now
        )

        total_cost = sum(
            agg.total_cost for agg in usage_summary.values()
        )

        return {
            "period_start": period_start.isoformat(),
            "current_date": now.isoformat(),
            "usage_by_metric": {
                metric.value: {
                    "quantity": float(agg.total_quantity),
                    "cost": float(agg.total_cost),
                    "records_count": agg.records_count
                }
                for metric, agg in usage_summary.items()
            },
            "estimated_total": float(total_cost)
        }

    def configure_pricing(
        self,
        metric_type: UsageMetricType,
        pricing_config: MetricPricing
    ) -> bool:
        """Configure pricing for a metric"""
        self.pricing_configs[metric_type] = pricing_config
        return True

    def get_pricing_config(
        self,
        metric_type: UsageMetricType
    ) -> Optional[MetricPricing]:
        """Get pricing configuration for metric"""
        return self.pricing_configs.get(metric_type)
