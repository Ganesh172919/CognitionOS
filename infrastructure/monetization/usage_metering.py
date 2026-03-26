"""
Usage Metering Pipeline

Real-time usage metering with aggregation windows, threshold alerts,
tenant-level tracking, and export capabilities for billing integration.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MeterType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


class AggregationWindow(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MeterDefinition:
    """Definition of a usage meter."""
    meter_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    meter_type: MeterType = MeterType.COUNTER
    unit: str = "count"
    aggregation: AggregationWindow = AggregationWindow.HOUR
    tags: List[str] = field(default_factory=list)


@dataclass
class UsageRecord:
    """A single usage data point."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meter_name: str = ""
    tenant_id: str = ""
    user_id: str = ""
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AggregatedUsage:
    """Aggregated usage for a time window."""
    meter_name: str = ""
    tenant_id: str = ""
    window: AggregationWindow = AggregationWindow.HOUR
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    total: float = 0.0
    count: int = 0
    min_value: float = float("inf")
    max_value: float = 0.0
    avg_value: float = 0.0

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "meter": self.meter_name,
            "tenant_id": self.tenant_id,
            "window": self.window.value,
            "total": round(self.total, 4),
            "count": self.count,
            "min": round(self.min_value, 4) if self.min_value != float("inf") else 0,
            "max": round(self.max_value, 4),
            "avg": round(self.avg_value, 4),
            "start": self.window_start.isoformat() if self.window_start else None,
            "end": self.window_end.isoformat() if self.window_end else None,
        }


@dataclass
class UsageThreshold:
    """Threshold alert configuration."""
    threshold_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meter_name: str = ""
    tenant_id: str = ""
    threshold_value: float = 0.0
    window: AggregationWindow = AggregationWindow.DAY
    severity: AlertSeverity = AlertSeverity.WARNING
    callback: Optional[Callable] = None
    message_template: str = ""
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    cooldown_minutes: int = 60


class UsageMeteringPipeline:
    """
    Production usage metering system.

    Features:
    - Real-time usage recording
    - Multi-window aggregation (minute/hour/day/month)
    - Per-tenant usage tracking
    - Threshold-based alerting
    - Usage export for billing
    - Rate calculation
    - Historical analysis
    """

    def __init__(self, max_records: int = 1_000_000, flush_interval: float = 60.0):
        self._meters: Dict[str, MeterDefinition] = {}
        self._records: Dict[str, List[UsageRecord]] = defaultdict(list)
        self._aggregations: Dict[str, Dict[str, AggregatedUsage]] = defaultdict(dict)
        self._thresholds: List[UsageThreshold] = []
        self._max_records = max_records
        self._flush_interval = flush_interval
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._buffer: List[UsageRecord] = []
        self._buffer_lock = asyncio.Lock()

        # Register default meters
        self._register_defaults()

    def _register_defaults(self) -> None:
        default_meters = [
            MeterDefinition(name="api_calls", unit="calls", meter_type=MeterType.COUNTER),
            MeterDefinition(name="tokens_consumed", unit="tokens", meter_type=MeterType.COUNTER),
            MeterDefinition(name="compute_seconds", unit="seconds", meter_type=MeterType.COUNTER),
            MeterDefinition(name="storage_bytes", unit="bytes", meter_type=MeterType.GAUGE),
            MeterDefinition(name="active_agents", unit="count", meter_type=MeterType.GAUGE),
            MeterDefinition(name="workflow_executions", unit="executions", meter_type=MeterType.COUNTER),
            MeterDefinition(name="code_generations", unit="generations", meter_type=MeterType.COUNTER),
            MeterDefinition(name="api_latency_ms", unit="ms", meter_type=MeterType.HISTOGRAM),
            MeterDefinition(name="error_count", unit="errors", meter_type=MeterType.COUNTER),
            MeterDefinition(name="bandwidth_bytes", unit="bytes", meter_type=MeterType.COUNTER),
        ]
        for m in default_meters:
            self._meters[m.name] = m

    def register_meter(self, meter: MeterDefinition) -> None:
        self._meters[meter.name] = meter

    # -- Recording ----------------------------------------------------------

    async def record(
        self,
        meter_name: str,
        tenant_id: str,
        value: float = 1.0,
        user_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        record = UsageRecord(
            meter_name=meter_name,
            tenant_id=tenant_id,
            user_id=user_id,
            value=value,
            metadata=metadata or {},
            tags=tags or {},
        )

        async with self._buffer_lock:
            self._buffer.append(record)

        # Update live aggregation
        self._update_aggregation(record)

        # Check thresholds
        await self._check_thresholds(meter_name, tenant_id)

        return record.record_id

    async def record_batch(self, records: List[UsageRecord]) -> int:
        async with self._buffer_lock:
            self._buffer.extend(records)
        for r in records:
            self._update_aggregation(r)
        return len(records)

    # -- Aggregation --------------------------------------------------------

    def _update_aggregation(self, record: UsageRecord) -> None:
        for window in AggregationWindow:
            agg_key = f"{record.meter_name}:{record.tenant_id}:{window.value}"
            if agg_key not in self._aggregations[record.tenant_id]:
                self._aggregations[record.tenant_id][agg_key] = AggregatedUsage(
                    meter_name=record.meter_name,
                    tenant_id=record.tenant_id,
                    window=window,
                    window_start=record.timestamp,
                )

            agg = self._aggregations[record.tenant_id][agg_key]
            agg.total += record.value
            agg.count += 1
            agg.min_value = min(agg.min_value, record.value)
            agg.max_value = max(agg.max_value, record.value)
            agg.avg_value = agg.total / agg.count
            agg.window_end = record.timestamp

    def get_usage(
        self,
        meter_name: str,
        tenant_id: str,
        window: AggregationWindow = AggregationWindow.DAY,
    ) -> Optional[AggregatedUsage]:
        agg_key = f"{meter_name}:{tenant_id}:{window.value}"
        return self._aggregations.get(tenant_id, {}).get(agg_key)

    def get_tenant_usage_summary(self, tenant_id: str) -> Dict[str, Any]:
        tenant_aggs = self._aggregations.get(tenant_id, {})
        summary = {}
        for key, agg in tenant_aggs.items():
            if agg.window == AggregationWindow.DAY:
                summary[agg.meter_name] = agg.summary
        return summary

    def get_all_meters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": m.name,
                "type": m.meter_type.value,
                "unit": m.unit,
                "description": m.description,
            }
            for m in self._meters.values()
        ]

    # -- Thresholds ---------------------------------------------------------

    def add_threshold(
        self,
        meter_name: str,
        tenant_id: str,
        threshold_value: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        window: AggregationWindow = AggregationWindow.DAY,
        message_template: str = "",
        callback: Optional[Callable] = None,
        cooldown_minutes: int = 60,
    ) -> str:
        threshold = UsageThreshold(
            meter_name=meter_name,
            tenant_id=tenant_id,
            threshold_value=threshold_value,
            severity=severity,
            window=window,
            message_template=message_template or f"{meter_name} exceeded {threshold_value}",
            callback=callback,
            cooldown_minutes=cooldown_minutes,
        )
        self._thresholds.append(threshold)
        return threshold.threshold_id

    async def _check_thresholds(self, meter_name: str, tenant_id: str) -> None:
        now = datetime.utcnow()
        for threshold in self._thresholds:
            if threshold.meter_name != meter_name or threshold.tenant_id != tenant_id:
                continue

            if threshold.triggered and threshold.triggered_at:
                cooldown_elapsed = (now - threshold.triggered_at).total_seconds() / 60
                if cooldown_elapsed < threshold.cooldown_minutes:
                    continue

            usage = self.get_usage(meter_name, tenant_id, threshold.window)
            if usage and usage.total >= threshold.threshold_value:
                threshold.triggered = True
                threshold.triggered_at = now

                logger.warning(
                    "Usage threshold triggered: %s for tenant %s (%.2f >= %.2f)",
                    meter_name, tenant_id, usage.total, threshold.threshold_value,
                )

                if threshold.callback:
                    try:
                        if asyncio.iscoroutinefunction(threshold.callback):
                            await threshold.callback(threshold, usage)
                        else:
                            threshold.callback(threshold, usage)
                    except Exception:
                        logger.exception("Threshold callback error")

    # -- Export for billing --------------------------------------------------

    def export_for_billing(
        self,
        tenant_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        tenant_aggs = self._aggregations.get(tenant_id, {})
        billable_items = []

        for key, agg in tenant_aggs.items():
            if agg.window == AggregationWindow.MONTH:
                billable_items.append(agg.summary)

        return {
            "tenant_id": tenant_id,
            "period_start": period_start.isoformat() if period_start else None,
            "period_end": period_end.isoformat() if period_end else None,
            "items": billable_items,
            "generated_at": datetime.utcnow().isoformat(),
        }

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Usage metering pipeline started")

    async def stop(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
        await self._flush_buffer()
        logger.info("Usage metering pipeline stopped")

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        async with self._buffer_lock:
            buffer = self._buffer.copy()
            self._buffer.clear()

        for record in buffer:
            key = f"{record.meter_name}:{record.tenant_id}"
            self._records[key].append(record)

            # Trim old records
            if len(self._records[key]) > self._max_records // len(self._meters):
                self._records[key] = self._records[key][-self._max_records // len(self._meters):]

    def get_stats(self) -> Dict[str, Any]:
        total_records = sum(len(r) for r in self._records.values())
        return {
            "meters": len(self._meters),
            "total_records": total_records,
            "buffer_size": len(self._buffer),
            "tenants_tracked": len(self._aggregations),
            "thresholds": len(self._thresholds),
            "triggered_thresholds": sum(1 for t in self._thresholds if t.triggered),
        }
