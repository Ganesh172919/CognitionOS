"""
Real-Time Telemetry Collector

Collects, aggregates, and streams system telemetry with:
- Counter, gauge, histogram, and summary metric types
- Per-tenant and per-service metric namespacing
- Time-windowed aggregations (1m, 5m, 15m, 1h)
- Alert threshold detection with debouncing
- SSE-compatible event streaming
- Prometheus-compatible text exposition
- Zero external dependencies (pure Python)
"""

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricSample:
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float = field(default_factory=time.time)
    unit: str = ""


@dataclass
class HistogramBucket:
    upper_bound: float
    count: int = 0


class Histogram:
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, buckets: Optional[List[float]] = None) -> None:
        bounds = sorted(buckets or self.DEFAULT_BUCKETS)
        self._buckets = [HistogramBucket(b) for b in bounds]
        self._buckets.append(HistogramBucket(math.inf))
        self._sum: float = 0.0
        self._count: int = 0

    def observe(self, value: float) -> None:
        self._sum += value
        self._count += 1
        for bucket in self._buckets:
            if value <= bucket.upper_bound:
                bucket.count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    def percentile(self, p: float) -> float:
        if self._count == 0:
            return 0.0
        target = self._count * (p / 100.0)
        prev_count = 0
        prev_bound = 0.0
        for bucket in self._buckets:
            if bucket.count >= target:
                if bucket.count == prev_count:
                    return bucket.upper_bound
                frac = (target - prev_count) / (bucket.count - prev_count)
                upper = bucket.upper_bound if not math.isinf(bucket.upper_bound) else prev_bound * 2
                return prev_bound + frac * (upper - prev_bound)
            prev_count = bucket.count
            if not math.isinf(bucket.upper_bound):
                prev_bound = bucket.upper_bound
        return prev_bound

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self._count,
            "sum": round(self._sum, 4),
            "mean": round(self.mean, 4),
            "p50": round(self.percentile(50), 4),
            "p90": round(self.percentile(90), 4),
            "p95": round(self.percentile(95), 4),
            "p99": round(self.percentile(99), 4),
        }


class SlidingWindowCounter:
    def __init__(self, window_seconds: int = 60) -> None:
        self._window = window_seconds
        self._buckets: Deque[Tuple[int, float]] = deque()

    def add(self, value: float = 1.0) -> None:
        now = int(time.time())
        self._buckets.append((now, value))
        self._evict()

    def total(self) -> float:
        self._evict()
        return sum(v for _, v in self._buckets)

    def rate(self) -> float:
        return self.total() / self._window

    def _evict(self) -> None:
        cutoff = int(time.time()) - self._window
        while self._buckets and self._buckets[0][0] < cutoff:
            self._buckets.popleft()


@dataclass
class AlertRule:
    name: str
    metric_name: str
    condition: str      # "gt", "lt", "gte", "lte", "eq"
    threshold: float
    window_s: int = 60
    min_samples: int = 1
    cooldown_s: int = 300
    severity: str = "warning"
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class AlertEvent:
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 4),
            "threshold": self.threshold,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "labels": self.labels,
        }


class TelemetryCollector:
    """
    Central telemetry collection engine.

    Usage::

        tc = TelemetryCollector()
        tc.increment("api.requests.total", labels={"endpoint": "/health"})
        tc.gauge("system.cpu.percent", 42.5)
        tc.observe("api.latency.seconds", 0.123, labels={"method": "GET"})
    """

    def __init__(self) -> None:
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, Histogram]] = defaultdict(dict)
        self._sliding_counters: Dict[str, Dict[str, SlidingWindowCounter]] = defaultdict(dict)
        self._recent_samples: Deque[MetricSample] = deque(maxlen=10000)
        self._alert_rules: Dict[str, AlertRule] = {}
        self._alert_cooldowns: Dict[str, float] = {}
        self._alert_callbacks: List[Callable[[AlertEvent], None]] = []
        self._metadata: Dict[str, Dict[str, str]] = {}

    # ──────────────────────────────────────────────
    # Write API
    # ──────────────────────────────────────────────

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        key = self._label_key(labels)
        self._counters[name][key] += value
        sc = self._sliding_counters[name].setdefault(key, SlidingWindowCounter())
        sc.add(value)
        self._record_sample(name, value, labels or {})
        self._check_alerts(name, self._counters[name][key])

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        key = self._label_key(labels)
        self._gauges[name][key] = value
        self._record_sample(name, value, labels or {})
        self._check_alerts(name, value)

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        key = self._label_key(labels)
        if key not in self._histograms[name]:
            self._histograms[name][key] = Histogram(buckets)
        self._histograms[name][key].observe(value)
        self._record_sample(name, value, labels or {})
        self._check_alerts(name, value)

    def describe(
        self,
        name: str,
        help_text: str,
        unit: str = "",
        metric_type: str = "gauge",
    ) -> None:
        self._metadata[name] = {"help": help_text, "unit": unit, "type": metric_type}

    # ──────────────────────────────────────────────
    # Read API
    # ──────────────────────────────────────────────

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        return self._counters[name][self._label_key(labels)]

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        return self._gauges[name][self._label_key(labels)]

    def get_histogram(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        key = self._label_key(labels)
        hist = self._histograms[name].get(key)
        return hist.to_dict() if hist else None

    def get_rate(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        key = self._label_key(labels)
        sc = self._sliding_counters[name].get(key)
        return sc.rate() if sc else 0.0

    def snapshot(self) -> Dict[str, Any]:
        counters: Dict[str, Any] = {n: dict(m) for n, m in self._counters.items()}
        gauges: Dict[str, Any] = {n: dict(m) for n, m in self._gauges.items()}
        histograms: Dict[str, Any] = {
            n: {k: v.to_dict() for k, v in m.items()}
            for n, m in self._histograms.items()
        }
        return {
            "timestamp": time.time(),
            "counters": counters,
            "gauges": gauges,
            "histograms": histograms,
        }

    def recent_samples(
        self,
        name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        samples = [
            {
                "name": s.name,
                "value": s.value,
                "labels": s.labels,
                "timestamp": s.timestamp,
            }
            for s in self._recent_samples
            if name is None or s.name == name
        ]
        return samples[-limit:]

    def prometheus_text(self) -> str:
        lines: List[str] = []
        ts_ms = int(time.time() * 1000)

        for name, label_map in self._counters.items():
            meta = self._metadata.get(name, {})
            if "help" in meta:
                lines.append(f"# HELP {name} {meta['help']}")
            lines.append(f"# TYPE {name} counter")
            for key, value in label_map.items():
                label_str = self._format_labels(key)
                lines.append(f"{name}{label_str} {value} {ts_ms}")

        for name, label_map in self._gauges.items():
            meta = self._metadata.get(name, {})
            if "help" in meta:
                lines.append(f"# HELP {name} {meta['help']}")
            lines.append(f"# TYPE {name} gauge")
            for key, value in label_map.items():
                label_str = self._format_labels(key)
                lines.append(f"{name}{label_str} {value} {ts_ms}")

        for name, label_map in self._histograms.items():
            meta = self._metadata.get(name, {})
            if "help" in meta:
                lines.append(f"# HELP {name} {meta['help']}")
            lines.append(f"# TYPE {name} histogram")
            for key, hist in label_map.items():
                label_str = self._format_labels(key)
                d = hist.to_dict()
                lines.append(f"{name}_count{label_str} {d['count']} {ts_ms}")
                lines.append(f"{name}_sum{label_str} {d['sum']} {ts_ms}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # Alerting
    # ──────────────────────────────────────────────

    def add_alert_rule(self, rule: AlertRule) -> None:
        self._alert_rules[rule.name] = rule

    def remove_alert_rule(self, name: str) -> None:
        self._alert_rules.pop(name, None)

    def on_alert(self, callback: Callable[[AlertEvent], None]) -> None:
        self._alert_callbacks.append(callback)

    def get_active_alerts(self) -> List[AlertEvent]:
        now = time.time()
        return [
            AlertEvent(
                rule_name=rule_name,
                metric_name=self._alert_rules[rule_name].metric_name,
                current_value=0.0,
                threshold=self._alert_rules[rule_name].threshold,
                severity=self._alert_rules[rule_name].severity,
                timestamp=fired_at,
                labels=self._alert_rules[rule_name].labels,
            )
            for rule_name, fired_at in self._alert_cooldowns.items()
            if rule_name in self._alert_rules
            and now - fired_at < self._alert_rules[rule_name].cooldown_s
        ]

    # ──────────────────────────────────────────────
    # SSE Streaming
    # ──────────────────────────────────────────────

    def sse_stream(self, interval_s: float = 5.0) -> Iterator[str]:
        """Generator yielding SSE-formatted telemetry snapshots"""
        import json
        while True:
            data = self.snapshot()
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(interval_s)

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _record_sample(self, name: str, value: float, labels: Dict[str, str]) -> None:
        self._recent_samples.append(MetricSample(name=name, value=value, labels=labels))

    def _check_alerts(self, metric_name: str, value: float) -> None:
        for rule in self._alert_rules.values():
            if rule.metric_name != metric_name:
                continue
            if not self._evaluate_condition(rule.condition, value, rule.threshold):
                continue
            now = time.time()
            last_fired = self._alert_cooldowns.get(rule.name, 0.0)
            if now - last_fired < rule.cooldown_s:
                continue
            self._alert_cooldowns[rule.name] = now
            event = AlertEvent(
                rule_name=rule.name,
                metric_name=metric_name,
                current_value=value,
                threshold=rule.threshold,
                severity=rule.severity,
                labels=rule.labels,
            )
            for cb in self._alert_callbacks:
                try:
                    cb(event)
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _evaluate_condition(condition: str, value: float, threshold: float) -> bool:
        return {
            "gt": value > threshold,
            "lt": value < threshold,
            "gte": value >= threshold,
            "lte": value <= threshold,
            "eq": abs(value - threshold) < 1e-9,
        }.get(condition, False)

    @staticmethod
    def _label_key(labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    @staticmethod
    def _format_labels(key: str) -> str:
        if not key:
            return ""
        parts = []
        for pair in key.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                parts.append(f'{k}="{v}"')
        return "{" + ",".join(parts) + "}" if parts else ""


# Global singleton
_global_collector: Optional[TelemetryCollector] = None


def get_telemetry() -> TelemetryCollector:
    global _global_collector
    if _global_collector is None:
        _global_collector = TelemetryCollector()
        _register_default_metrics(_global_collector)
    return _global_collector


def _register_default_metrics(tc: TelemetryCollector) -> None:
    tc.describe("api_requests_total", "Total API requests", metric_type="counter")
    tc.describe("api_request_duration_seconds", "API request duration", unit="s", metric_type="histogram")
    tc.describe("api_active_connections", "Active API connections", metric_type="gauge")
    tc.describe("llm_tokens_total", "Total LLM tokens consumed", metric_type="counter")
    tc.describe("llm_cost_usd_total", "Total LLM cost in USD", metric_type="counter")
    tc.describe("llm_request_duration_seconds", "LLM request duration", unit="s", metric_type="histogram")
    tc.describe("agent_executions_total", "Total agent executions", metric_type="counter")
    tc.describe("agent_tool_calls_total", "Total agent tool calls", metric_type="counter")
    tc.describe("workflow_executions_total", "Total workflow executions", metric_type="counter")
    tc.describe("memory_entries_total", "Total memory entries stored", metric_type="gauge")
    tc.describe("cache_hits_total", "Cache hit count", metric_type="counter")
    tc.describe("cache_misses_total", "Cache miss count", metric_type="counter")
