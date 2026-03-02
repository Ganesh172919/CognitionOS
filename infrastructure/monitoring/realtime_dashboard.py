"""
Real-Time Monitoring Dashboard Service — CognitionOS

Collects and aggregates system metrics for dashboard display:
- System health (CPU, memory, disk)
- API performance (latency, throughput, errors)
- Agent execution metrics
- Revenue / billing metrics
- Queue depth and processing rates
- Database connection pool stats
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class MetricSeries:
    name: str
    points: Deque[MetricPoint] = field(default_factory=lambda: deque(maxlen=1000))
    unit: str = ""

    @property
    def latest(self) -> float:
        return self.points[-1].value if self.points else 0.0

    @property
    def avg(self) -> float:
        if not self.points:
            return 0.0
        return sum(p.value for p in self.points) / len(self.points)

    @property
    def max_val(self) -> float:
        return max((p.value for p in self.points), default=0.0)

    @property
    def min_val(self) -> float:
        return min((p.value for p in self.points), default=0.0)


class MonitoringDashboard:
    """Collects, aggregates, and serves system metrics."""

    def __init__(self, *, history_size: int = 1000, collection_interval: float = 5.0) -> None:
        self._series: Dict[str, MetricSeries] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=history_size))
        self._alerts: List[Dict[str, Any]] = []
        self._alert_rules: List[Dict[str, Any]] = []
        self._collection_interval = collection_interval
        self._running = False
        self._collector_task: Optional[asyncio.Task] = None

    # ---- lifecycle ----
    async def start(self) -> None:
        self._running = True
        self._collector_task = asyncio.create_task(self._collect_loop())
        logger.info("Monitoring dashboard started")

    async def stop(self) -> None:
        self._running = False
        if self._collector_task:
            self._collector_task.cancel()

    # ---- record metrics ----
    def record(self, name: str, value: float, *, tags: Dict[str, str] | None = None, unit: str = "") -> None:
        if name not in self._series:
            self._series[name] = MetricSeries(name=name, unit=unit)
        self._series[name].points.append(MetricPoint(name=name, value=value, tags=tags or {}, unit=unit))
        self._check_alerts(name, value)

    def increment(self, name: str, amount: int = 1) -> None:
        self._counters[name] += amount

    def gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def histogram(self, name: str, value: float) -> None:
        self._histograms[name].append(value)

    # ---- alerts ----
    def add_alert_rule(self, name: str, metric: str, *, threshold: float,
                       comparison: str = "gt", message: str = "") -> None:
        self._alert_rules.append({
            "name": name, "metric": metric, "threshold": threshold,
            "comparison": comparison, "message": message})

    def _check_alerts(self, name: str, value: float) -> None:
        for rule in self._alert_rules:
            if rule["metric"] != name:
                continue
            triggered = False
            if rule["comparison"] == "gt" and value > rule["threshold"]:
                triggered = True
            elif rule["comparison"] == "lt" and value < rule["threshold"]:
                triggered = True
            if triggered:
                alert = {
                    "rule": rule["name"], "metric": name, "value": value,
                    "threshold": rule["threshold"], "message": rule["message"],
                    "timestamp": datetime.now(timezone.utc).isoformat()}
                self._alerts.append(alert)
                if len(self._alerts) > 500:
                    self._alerts = self._alerts[-500:]
                logger.warning("Alert triggered: %s — %s = %.2f (threshold: %.2f)",
                               rule["name"], name, value, rule["threshold"])

    # ---- system metrics collection ----
    async def _collect_loop(self) -> None:
        while self._running:
            try:
                self._collect_system_metrics()
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Metrics collection error")
                await asyncio.sleep(self._collection_interval)

    def _collect_system_metrics(self) -> None:
        if not HAS_PSUTIL:
            return
        self.record("system.cpu_percent", psutil.cpu_percent(), unit="%")
        mem = psutil.virtual_memory()
        self.record("system.memory_percent", mem.percent, unit="%")
        self.record("system.memory_used_gb", mem.used / (1024**3), unit="GB")
        disk = psutil.disk_usage("/")
        self.record("system.disk_percent", disk.percent, unit="%")
        self.record("system.process_count", len(psutil.pids()))

    # ---- dashboard API ----
    def get_dashboard(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": self._get_system_section(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "series_summary": {
                n: {"latest": s.latest, "avg": round(s.avg, 3),
                    "min": round(s.min_val, 3), "max": round(s.max_val, 3),
                    "count": len(s.points)}
                for n, s in self._series.items()},
            "recent_alerts": self._alerts[-20:],
        }

    def _get_system_section(self) -> Dict[str, Any]:
        section: Dict[str, Any] = {}
        for key in ["system.cpu_percent", "system.memory_percent", "system.disk_percent"]:
            if key in self._series:
                s = self._series[key]
                section[key.split(".")[-1]] = {"current": s.latest, "avg": round(s.avg, 2)}
        return section

    def get_series(self, name: str, *, limit: int = 100) -> List[Dict[str, Any]]:
        s = self._series.get(name)
        if not s:
            return []
        return [{"value": p.value, "timestamp": p.timestamp, "tags": p.tags}
                for p in list(s.points)[-limit:]]

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        h = self._histograms.get(name)
        if not h or len(h) == 0:
            return {}
        data = sorted(h)
        n = len(data)
        return {
            "count": n, "avg": sum(data) / n,
            "min": data[0], "max": data[-1],
            "p50": data[int(n * 0.5)], "p95": data[int(n * 0.95)],
            "p99": data[min(int(n * 0.99), n - 1)],
        }

    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._alerts[-limit:]

    def clear_alerts(self) -> int:
        c = len(self._alerts)
        self._alerts.clear()
        return c


_dashboard: MonitoringDashboard | None = None

def get_monitoring_dashboard() -> MonitoringDashboard:
    global _dashboard
    if not _dashboard:
        _dashboard = MonitoringDashboard()
    return _dashboard
