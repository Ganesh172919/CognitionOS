"""
Intelligent Usage Metering and Token Tracking System
Real-time usage tracking, cost calculation, and predictive analytics for AI operations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import json
from collections import defaultdict


class UsageEventType(Enum):
    """Types of usage events"""
    API_CALL = "api_call"
    LLM_COMPLETION = "llm_completion"
    EMBEDDING_GENERATION = "embedding_generation"
    CODE_EXECUTION = "code_execution"
    STORAGE_OPERATION = "storage_operation"
    NETWORK_TRANSFER = "network_transfer"
    COMPUTE_TIME = "compute_time"
    DATABASE_QUERY = "database_query"
    AGENT_EXECUTION = "agent_execution"
    WORKFLOW_EXECUTION = "workflow_execution"


class TokenProvider(Enum):
    """LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class TokenCost:
    """Token pricing per provider and model"""
    provider: TokenProvider
    model_name: str
    input_token_cost: Decimal  # Cost per 1M input tokens
    output_token_cost: Decimal  # Cost per 1M output tokens
    context_window: int
    supports_streaming: bool = True


@dataclass
class UsageEvent:
    """Individual usage event"""
    event_id: str
    tenant_id: str
    user_id: str
    event_type: UsageEventType
    timestamp: datetime
    resource_id: Optional[str] = None
    quantity: Decimal = Decimal("1")
    unit: str = "count"
    cost: Decimal = Decimal("0")
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Token-specific fields
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider: Optional[TokenProvider] = None
    model_name: Optional[str] = None

    # Performance metrics
    duration_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class TokenUsageSummary:
    """Summary of token usage"""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost: Decimal
    by_provider: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_user: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class UsageQuota:
    """Usage quotas and limits"""
    tenant_id: str
    quota_type: str  # tokens, api_calls, compute_hours, storage
    limit: int
    period: str  # hourly, daily, monthly
    current_usage: int = 0
    reset_at: Optional[datetime] = None
    soft_limit_percent: int = 80  # Alert at 80%
    hard_limit_enabled: bool = True


@dataclass
class UsageForecast:
    """Predictive usage forecast"""
    tenant_id: str
    forecast_period: str  # day, week, month
    predicted_tokens: int
    predicted_cost: Decimal
    confidence_score: float
    trend: str  # increasing, decreasing, stable
    anomaly_detected: bool = False
    anomaly_details: Optional[Dict[str, Any]] = None


class TokenPricingEngine:
    """Manages token pricing across providers"""

    def __init__(self):
        self.pricing_table = self._initialize_pricing()

    def _initialize_pricing(self) -> Dict[str, TokenCost]:
        """Initialize pricing for various models"""
        return {
            # OpenAI pricing (as of 2024)
            "openai_gpt-4": TokenCost(
                provider=TokenProvider.OPENAI,
                model_name="gpt-4",
                input_token_cost=Decimal("30.00"),  # $30 per 1M tokens
                output_token_cost=Decimal("60.00"),  # $60 per 1M tokens
                context_window=8192
            ),
            "openai_gpt-4-32k": TokenCost(
                provider=TokenProvider.OPENAI,
                model_name="gpt-4-32k",
                input_token_cost=Decimal("60.00"),
                output_token_cost=Decimal("120.00"),
                context_window=32768
            ),
            "openai_gpt-3.5-turbo": TokenCost(
                provider=TokenProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                input_token_cost=Decimal("0.50"),
                output_token_cost=Decimal("1.50"),
                context_window=16384
            ),
            "openai_gpt-4-turbo": TokenCost(
                provider=TokenProvider.OPENAI,
                model_name="gpt-4-turbo",
                input_token_cost=Decimal("10.00"),
                output_token_cost=Decimal("30.00"),
                context_window=128000
            ),
            # Anthropic pricing
            "anthropic_claude-3-opus": TokenCost(
                provider=TokenProvider.ANTHROPIC,
                model_name="claude-3-opus",
                input_token_cost=Decimal("15.00"),
                output_token_cost=Decimal("75.00"),
                context_window=200000
            ),
            "anthropic_claude-3-sonnet": TokenCost(
                provider=TokenProvider.ANTHROPIC,
                model_name="claude-3-sonnet",
                input_token_cost=Decimal("3.00"),
                output_token_cost=Decimal("15.00"),
                context_window=200000
            ),
            "anthropic_claude-3-haiku": TokenCost(
                provider=TokenProvider.ANTHROPIC,
                model_name="claude-3-haiku",
                input_token_cost=Decimal("0.25"),
                output_token_cost=Decimal("1.25"),
                context_window=200000
            ),
            # Cohere pricing
            "cohere_command": TokenCost(
                provider=TokenProvider.COHERE,
                model_name="command",
                input_token_cost=Decimal("1.00"),
                output_token_cost=Decimal("2.00"),
                context_window=4096
            ),
        }

    def get_token_cost(
        self,
        provider: TokenProvider,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> Decimal:
        """Calculate cost for token usage"""
        key = f"{provider.value}_{model_name}"
        pricing = self.pricing_table.get(key)

        if not pricing:
            # Use default pricing if model not found
            return Decimal("0")

        # Calculate cost per million tokens
        input_cost = (Decimal(input_tokens) / Decimal("1000000")) * pricing.input_token_cost
        output_cost = (Decimal(output_tokens) / Decimal("1000000")) * pricing.output_token_cost

        return input_cost + output_cost

    def add_custom_pricing(
        self,
        provider: TokenProvider,
        model_name: str,
        input_cost: Decimal,
        output_cost: Decimal,
        context_window: int
    ):
        """Add custom model pricing"""
        key = f"{provider.value}_{model_name}"
        self.pricing_table[key] = TokenCost(
            provider=provider,
            model_name=model_name,
            input_token_cost=input_cost,
            output_token_cost=output_cost,
            context_window=context_window
        )

    def get_cheapest_model(
        self,
        min_context_window: int,
        provider: Optional[TokenProvider] = None
    ) -> Tuple[str, TokenCost]:
        """Find cheapest model meeting requirements"""
        candidates = []

        for key, pricing in self.pricing_table.items():
            if pricing.context_window < min_context_window:
                continue

            if provider and pricing.provider != provider:
                continue

            # Calculate average cost
            avg_cost = (pricing.input_token_cost + pricing.output_token_cost) / Decimal("2")
            candidates.append((key, pricing, avg_cost))

        if not candidates:
            raise ValueError("No model found meeting requirements")

        # Sort by cost
        candidates.sort(key=lambda x: x[2])

        return candidates[0][0], candidates[0][1]


class UsageMeteringEngine:
    """Core usage metering engine"""

    def __init__(self):
        self.pricing_engine = TokenPricingEngine()
        self.usage_events: Dict[str, List[UsageEvent]] = defaultdict(list)
        self.quotas: Dict[str, List[UsageQuota]] = defaultdict(list)

    async def record_usage(
        self,
        tenant_id: str,
        user_id: str,
        event_type: UsageEventType,
        quantity: Decimal = Decimal("1"),
        unit: str = "count",
        metadata: Optional[Dict[str, Any]] = None,
        # Token-specific
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider: Optional[TokenProvider] = None,
        model_name: Optional[str] = None,
        # Performance
        duration_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> UsageEvent:
        """Record usage event"""
        import uuid

        event_id = str(uuid.uuid4())
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost = Decimal("0")
        if event_type == UsageEventType.LLM_COMPLETION and provider and model_name:
            cost = self.pricing_engine.get_token_cost(
                provider=provider,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        elif event_type == UsageEventType.API_CALL:
            cost = Decimal("0.001")  # $0.001 per API call
        elif event_type == UsageEventType.COMPUTE_TIME:
            cost = quantity * Decimal("0.50")  # $0.50 per compute hour
        elif event_type == UsageEventType.STORAGE_OPERATION:
            cost = quantity * Decimal("0.10")  # $0.10 per GB

        event = UsageEvent(
            event_id=event_id,
            tenant_id=tenant_id,
            user_id=user_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            quantity=quantity,
            unit=unit,
            cost=cost,
            metadata=metadata or {},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            provider=provider,
            model_name=model_name,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )

        self.usage_events[tenant_id].append(event)

        # Check quotas
        await self._check_quotas(tenant_id, event)

        return event

    async def _check_quotas(self, tenant_id: str, event: UsageEvent):
        """Check if event exceeds quotas"""
        quotas = self.quotas.get(tenant_id, [])

        for quota in quotas:
            # Check if quota applies to this event
            if quota.quota_type == "tokens" and event.total_tokens > 0:
                quota.current_usage += event.total_tokens
            elif quota.quota_type == "api_calls" and event.event_type == UsageEventType.API_CALL:
                quota.current_usage += 1
            elif quota.quota_type == "compute_hours" and event.event_type == UsageEventType.COMPUTE_TIME:
                quota.current_usage += int(event.quantity)

            # Check if quota exceeded
            if quota.current_usage >= quota.limit * quota.soft_limit_percent / 100:
                # Alert - approaching limit
                pass

            if quota.hard_limit_enabled and quota.current_usage >= quota.limit:
                # Hard limit reached - should block further usage
                raise Exception(f"Quota exceeded for {quota.quota_type}")

    async def get_usage_summary(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> TokenUsageSummary:
        """Get usage summary for period"""
        events = [
            e for e in self.usage_events.get(tenant_id, [])
            if period_start <= e.timestamp <= period_end
        ]

        if not events:
            return TokenUsageSummary(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=period_end,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_cost=Decimal("0")
            )

        # Aggregate totals
        total_input = sum(e.input_tokens for e in events)
        total_output = sum(e.output_tokens for e in events)
        total_tokens = sum(e.total_tokens for e in events)
        total_cost = sum(e.cost for e in events)

        # Aggregate by provider
        by_provider = defaultdict(lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": Decimal("0")
        })

        for event in events:
            if event.provider:
                by_provider[event.provider.value]["input_tokens"] += event.input_tokens
                by_provider[event.provider.value]["output_tokens"] += event.output_tokens
                by_provider[event.provider.value]["total_tokens"] += event.total_tokens
                by_provider[event.provider.value]["cost"] += event.cost

        # Aggregate by model
        by_model = defaultdict(lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": Decimal("0")
        })

        for event in events:
            if event.model_name:
                by_model[event.model_name]["input_tokens"] += event.input_tokens
                by_model[event.model_name]["output_tokens"] += event.output_tokens
                by_model[event.model_name]["total_tokens"] += event.total_tokens
                by_model[event.model_name]["cost"] += event.cost

        # Aggregate by user
        by_user = defaultdict(lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": Decimal("0")
        })

        for event in events:
            by_user[event.user_id]["input_tokens"] += event.input_tokens
            by_user[event.user_id]["output_tokens"] += event.output_tokens
            by_user[event.user_id]["total_tokens"] += event.total_tokens
            by_user[event.user_id]["cost"] += event.cost

        return TokenUsageSummary(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_cost=total_cost,
            by_provider=dict(by_provider),
            by_model=dict(by_model),
            by_user=dict(by_user)
        )

    async def set_quota(
        self,
        tenant_id: str,
        quota_type: str,
        limit: int,
        period: str,
        soft_limit_percent: int = 80,
        hard_limit_enabled: bool = True
    ) -> UsageQuota:
        """Set usage quota"""
        # Calculate reset time
        now = datetime.utcnow()
        if period == "hourly":
            reset_at = now + timedelta(hours=1)
        elif period == "daily":
            reset_at = now + timedelta(days=1)
        elif period == "monthly":
            reset_at = now + timedelta(days=30)
        else:
            reset_at = now + timedelta(hours=1)

        quota = UsageQuota(
            tenant_id=tenant_id,
            quota_type=quota_type,
            limit=limit,
            period=period,
            reset_at=reset_at,
            soft_limit_percent=soft_limit_percent,
            hard_limit_enabled=hard_limit_enabled
        )

        self.quotas[tenant_id].append(quota)

        return quota

    async def get_real_time_metrics(
        self,
        tenant_id: str,
        window_minutes: int = 5
    ) -> Dict[str, Any]:
        """Get real-time usage metrics"""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_events = [
            e for e in self.usage_events.get(tenant_id, [])
            if e.timestamp >= cutoff
        ]

        if not recent_events:
            return {
                "tenant_id": tenant_id,
                "window_minutes": window_minutes,
                "metrics": {}
            }

        # Calculate metrics
        total_events = len(recent_events)
        successful_events = sum(1 for e in recent_events if e.success)
        failed_events = total_events - successful_events
        success_rate = (successful_events / total_events * 100) if total_events > 0 else 0

        avg_duration = sum(e.duration_ms for e in recent_events if e.duration_ms) / len(recent_events) if recent_events else 0

        total_cost = sum(e.cost for e in recent_events)
        total_tokens = sum(e.total_tokens for e in recent_events)

        # Events per second
        time_span_seconds = window_minutes * 60
        events_per_second = total_events / time_span_seconds

        return {
            "tenant_id": tenant_id,
            "window_minutes": window_minutes,
            "metrics": {
                "total_events": total_events,
                "successful_events": successful_events,
                "failed_events": failed_events,
                "success_rate": round(success_rate, 2),
                "avg_duration_ms": round(avg_duration, 2),
                "total_cost": float(total_cost),
                "total_tokens": total_tokens,
                "events_per_second": round(events_per_second, 2)
            }
        }


class UsageForecastingEngine:
    """Predictive usage forecasting"""

    def __init__(self, metering_engine: UsageMeteringEngine):
        self.metering_engine = metering_engine

    async def forecast_usage(
        self,
        tenant_id: str,
        forecast_period: str = "day"
    ) -> UsageForecast:
        """Forecast future usage based on historical data"""
        # Get historical data (last 30 days)
        now = datetime.utcnow()
        historical_start = now - timedelta(days=30)

        summary = await self.metering_engine.get_usage_summary(
            tenant_id=tenant_id,
            period_start=historical_start,
            period_end=now
        )

        # Simple linear forecasting
        days_of_data = 30
        avg_tokens_per_day = summary.total_tokens / days_of_data
        avg_cost_per_day = summary.total_cost / days_of_data

        # Forecast based on period
        if forecast_period == "day":
            predicted_tokens = int(avg_tokens_per_day)
            predicted_cost = avg_cost_per_day
        elif forecast_period == "week":
            predicted_tokens = int(avg_tokens_per_day * 7)
            predicted_cost = avg_cost_per_day * 7
        elif forecast_period == "month":
            predicted_tokens = int(avg_tokens_per_day * 30)
            predicted_cost = avg_cost_per_day * 30
        else:
            predicted_tokens = int(avg_tokens_per_day)
            predicted_cost = avg_cost_per_day

        # Detect trend
        recent_events = self.metering_engine.usage_events.get(tenant_id, [])
        if len(recent_events) >= 2:
            recent_week = [
                e for e in recent_events
                if e.timestamp >= now - timedelta(days=7)
            ]
            previous_week = [
                e for e in recent_events
                if now - timedelta(days=14) <= e.timestamp < now - timedelta(days=7)
            ]

            recent_tokens = sum(e.total_tokens for e in recent_week)
            previous_tokens = sum(e.total_tokens for e in previous_week)

            if recent_tokens > previous_tokens * 1.2:
                trend = "increasing"
            elif recent_tokens < previous_tokens * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Anomaly detection (simple threshold-based)
        anomaly_detected = False
        anomaly_details = None

        if predicted_tokens > avg_tokens_per_day * 2:
            anomaly_detected = True
            anomaly_details = {
                "type": "spike",
                "message": "Predicted usage significantly higher than average"
            }

        # Confidence score (simplified)
        confidence_score = 0.7 if days_of_data >= 30 else 0.5

        return UsageForecast(
            tenant_id=tenant_id,
            forecast_period=forecast_period,
            predicted_tokens=predicted_tokens,
            predicted_cost=predicted_cost,
            confidence_score=confidence_score,
            trend=trend,
            anomaly_detected=anomaly_detected,
            anomaly_details=anomaly_details
        )

    async def get_cost_optimization_suggestions(
        self,
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        """Get cost optimization suggestions"""
        suggestions = []

        # Get recent usage
        now = datetime.utcnow()
        summary = await self.metering_engine.get_usage_summary(
            tenant_id=tenant_id,
            period_start=now - timedelta(days=7),
            period_end=now
        )

        # Analyze model usage
        for model_name, usage in summary.by_model.items():
            if "gpt-4" in model_name and usage["total_tokens"] > 100000:
                # Suggest using cheaper model
                suggestions.append({
                    "type": "model_optimization",
                    "current_model": model_name,
                    "suggestion": "Consider using gpt-3.5-turbo for non-critical tasks",
                    "potential_savings": float(usage["cost"] * Decimal("0.7"))
                })

        # Check for inefficient usage patterns
        events = [
            e for e in self.metering_engine.usage_events.get(tenant_id, [])
            if e.timestamp >= now - timedelta(days=7)
        ]

        failed_events = [e for e in events if not e.success]
        if len(failed_events) > len(events) * 0.1:  # More than 10% failures
            suggestions.append({
                "type": "reliability",
                "message": "High failure rate detected",
                "suggestion": "Implement retry logic and error handling",
                "potential_savings": float(sum(e.cost for e in failed_events))
            })

        return suggestions
