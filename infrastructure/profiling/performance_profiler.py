"""
Advanced Performance Profiling System — Call graph analysis, hotspot detection,
memory profiling, latency tracking, SLO monitoring, and optimization recommendations.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Tuple


# ─────────────────────────────────── Enums ───────────────────────────────────


class ProfilerMode(str, Enum):
    SAMPLING = "sampling"
    TRACING = "tracing"
    STATISTICAL = "statistical"


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class SLOStatus(str, Enum):
    MET = "met"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class OptimizationCategory(str, Enum):
    DATABASE = "database"
    CACHING = "caching"
    ASYNC = "async"
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    ALGORITHM = "algorithm"
    ARCHITECTURE = "architecture"


# ────────────────────────────── Data structures ──────────────────────────────


@dataclass
class ProfileSpan:
    span_id: str
    name: str
    parent_id: Optional[str]
    start_time: float
    end_time: Optional[float]
    tags: Dict[str, Any]
    logs: List[Dict[str, Any]]
    error: Optional[str]

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def finish(self, error: Optional[str] = None) -> None:
        self.end_time = time.monotonic()
        if error:
            self.error = error

    def log(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.logs.append(
            {
                "event": event,
                "data": data or {},
                "ts": time.monotonic() - self.start_time,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 3),
            "tags": self.tags,
            "logs": self.logs,
            "error": self.error,
        }


@dataclass
class MetricSeries:
    name: str
    metric_type: MetricType
    unit: str
    values: Deque
    labels: Dict[str, str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def record(self, value: float) -> None:
        self.values.append((time.monotonic(), value))

    def get_statistics(self) -> Dict[str, Any]:
        vals = [v for _, v in self.values]
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "stddev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "p95": self._percentile(vals, 95),
            "p99": self._percentile(vals, 99),
        }

    @staticmethod
    def _percentile(data: List[float], pct: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * pct / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]


@dataclass
class CallGraphNode:
    function_name: str
    module: str
    call_count: int
    total_time_ms: float
    self_time_ms: float
    children: List[str]
    callers: List[str]

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.call_count, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "module": self.module,
            "call_count": self.call_count,
            "total_time_ms": round(self.total_time_ms, 3),
            "self_time_ms": round(self.self_time_ms, 3),
            "avg_time_ms": round(self.avg_time_ms, 3),
            "children": self.children,
            "callers": self.callers,
        }


@dataclass
class SLODefinition:
    slo_id: str
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison: str
    window_seconds: int
    error_budget_pct: float
    owner: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slo_id": self.slo_id,
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "window_seconds": self.window_seconds,
            "error_budget_pct": self.error_budget_pct,
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class OptimizationRecommendation:
    rec_id: str
    category: OptimizationCategory
    title: str
    description: str
    impact: str
    effort: str
    estimated_improvement_pct: float
    affected_functions: List[str]
    code_suggestions: List[str]
    priority_score: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rec_id": self.rec_id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "impact": self.impact,
            "effort": self.effort,
            "estimated_improvement_pct": self.estimated_improvement_pct,
            "affected_functions": self.affected_functions,
            "code_suggestions": self.code_suggestions,
            "priority_score": self.priority_score,
            "created_at": self.created_at.isoformat(),
        }


# ──────────────────────── Trace Collector ───────────────────────────────────


class TraceCollector:
    """
    Collects distributed traces with span hierarchy, sampling, and
    latency percentile computation.
    """

    def __init__(self, sample_rate: float = 1.0, max_traces: int = 1000):
        self.sample_rate = sample_rate
        self.max_traces = max_traces
        self._traces: Dict[str, Dict[str, ProfileSpan]] = {}
        self._completed: Deque[Dict[str, Any]] = deque(maxlen=max_traces)
        self._latency_histogram: Deque[float] = deque(maxlen=5000)
        self._span_counts: Dict[str, int] = defaultdict(int)

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ProfileSpan]:
        trace_id = trace_id or str(uuid.uuid4())
        span = ProfileSpan(
            span_id=str(uuid.uuid4()),
            name=name,
            parent_id=parent_id,
            start_time=time.monotonic(),
            end_time=None,
            tags=tags or {},
            logs=[],
            error=None,
        )
        if trace_id not in self._traces:
            self._traces[trace_id] = {}
        self._traces[trace_id][span.span_id] = span
        self._span_counts[name] += 1
        return trace_id, span

    def finish_span(
        self,
        trace_id: str,
        span: ProfileSpan,
        error: Optional[str] = None,
    ) -> float:
        span.finish(error)
        latency = span.duration_ms
        self._latency_histogram.append(latency)
        return latency

    def finish_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        spans = self._traces.pop(trace_id, {})
        if not spans:
            return None
        root_spans = [s for s in spans.values() if s.parent_id is None]
        total_duration = sum(s.duration_ms for s in root_spans)
        trace_summary = {
            "trace_id": trace_id,
            "total_duration_ms": round(total_duration, 3),
            "span_count": len(spans),
            "spans": [s.to_dict() for s in spans.values()],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._completed.append(trace_summary)
        return trace_summary

    @contextmanager
    def span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[str, ProfileSpan], None, None]:
        tid, span_obj = self.start_span(name, trace_id, tags=tags)
        try:
            yield tid, span_obj
            self.finish_span(tid, span_obj)
        except Exception as exc:
            self.finish_span(tid, span_obj, error=str(exc))
            raise
        finally:
            self.finish_trace(tid)

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        tid, span_obj = self.start_span(name, trace_id, tags=tags)
        try:
            yield tid, span_obj
            self.finish_span(tid, span_obj)
        except Exception as exc:
            self.finish_span(tid, span_obj, error=str(exc))
            raise
        finally:
            self.finish_trace(tid)

    def get_latency_stats(self) -> Dict[str, Any]:
        lats = list(self._latency_histogram)
        if not lats:
            return {"count": 0}
        return {
            "count": len(lats),
            "min_ms": round(min(lats), 3),
            "max_ms": round(max(lats), 3),
            "mean_ms": round(statistics.mean(lats), 3),
            "p50_ms": round(MetricSeries._percentile(lats, 50), 3),
            "p95_ms": round(MetricSeries._percentile(lats, 95), 3),
            "p99_ms": round(MetricSeries._percentile(lats, 99), 3),
        }

    def get_hot_spans(self, top_k: int = 10) -> List[Dict[str, Any]]:
        return sorted(
            [{"name": k, "call_count": v} for k, v in self._span_counts.items()],
            key=lambda x: x["call_count"],
            reverse=True,
        )[:top_k]

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        traces = list(self._completed)
        return list(reversed(traces[-limit:]))


# ─────────────────────── Metrics Aggregator ─────────────────────────────────


class MetricsAggregator:
    """
    Aggregates application metrics with time-series storage, label filtering,
    and multi-dimensional analysis.
    """

    def __init__(self):
        self._metrics: Dict[str, MetricSeries] = {}

    def register(
        self,
        name: str,
        metric_type: MetricType,
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
        maxlen: int = 10000,
    ) -> MetricSeries:
        series = MetricSeries(
            name=name,
            metric_type=metric_type,
            unit=unit,
            values=deque(maxlen=maxlen),
            labels=labels or {},
        )
        self._metrics[name] = series
        return series

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        if name not in self._metrics:
            self.register(name, MetricType.GAUGE)
        self._metrics[name].record(value)

    def increment(self, name: str, delta: float = 1.0) -> None:
        if name not in self._metrics:
            self.register(name, MetricType.COUNTER)
        series = self._metrics[name]
        last_val = series.values[-1][1] if series.values else 0.0
        series.record(last_val + delta)

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        series = self._metrics.get(name)
        if series is None:
            return None
        stats = series.get_statistics()
        return {
            "name": name,
            "type": series.metric_type.value,
            "unit": series.unit,
            "labels": series.labels,
            **stats,
        }

    def list_metrics(self) -> List[str]:
        return list(self._metrics.keys())

    def get_all_metrics(self) -> Dict[str, Any]:
        return {name: self.get_metric(name) for name in self._metrics}


# ─────────────────────── Call Graph Analyzer ────────────────────────────────


class CallGraphAnalyzer:
    """
    Builds and analyzes function call graphs to identify performance
    bottlenecks, hot paths, and optimization opportunities.
    """

    def __init__(self):
        self._nodes: Dict[str, CallGraphNode] = {}
        self._call_stacks: List[List[str]] = []

    def record_call(
        self,
        function_name: str,
        module: str,
        duration_ms: float,
        caller: Optional[str] = None,
    ) -> None:
        key = f"{module}.{function_name}"
        if key not in self._nodes:
            self._nodes[key] = CallGraphNode(
                function_name=function_name,
                module=module,
                call_count=0,
                total_time_ms=0.0,
                self_time_ms=0.0,
                children=[],
                callers=[],
            )
        node = self._nodes[key]
        node.call_count += 1
        node.total_time_ms += duration_ms
        if caller:
            if caller not in node.callers:
                node.callers.append(caller)
            caller_node = self._nodes.get(caller)
            if caller_node and key not in caller_node.children:
                caller_node.children.append(key)

    def compute_self_times(self) -> None:
        for key, node in self._nodes.items():
            child_time = sum(
                self._nodes[child].total_time_ms
                for child in node.children
                if child in self._nodes
            )
            node.self_time_ms = max(0.0, node.total_time_ms - child_time)

    def get_hotspots(self, top_k: int = 10) -> List[Dict[str, Any]]:
        self.compute_self_times()
        return sorted(
            [n.to_dict() for n in self._nodes.values()],
            key=lambda n: n["self_time_ms"],
            reverse=True,
        )[:top_k]

    def get_critical_path(self, entry: Optional[str] = None) -> List[str]:
        if not self._nodes:
            return []
        root = entry or max(
            self._nodes.keys(),
            key=lambda k: self._nodes[k].total_time_ms,
        )
        path = [root]
        visited = {root}
        current = root
        for _ in range(50):  # Max depth
            node = self._nodes.get(current)
            if node is None or not node.children:
                break
            best_child = max(
                (c for c in node.children if c not in visited),
                key=lambda c: self._nodes.get(c, CallGraphNode("", "", 0, 0, 0, [], [])).self_time_ms,
                default=None,
            )
            if best_child is None:
                break
            path.append(best_child)
            visited.add(best_child)
            current = best_child
        return path

    def get_graph_stats(self) -> Dict[str, Any]:
        nodes = list(self._nodes.values())
        return {
            "total_functions": len(nodes),
            "total_calls": sum(n.call_count for n in nodes),
            "total_time_ms": round(sum(n.self_time_ms for n in nodes), 3),
            "avg_call_time_ms": round(
                sum(n.avg_time_ms for n in nodes) / max(len(nodes), 1), 3
            ),
        }


# ─────────────────────── SLO Monitor ────────────────────────────────────────


class SLOMonitor:
    """
    Monitors Service Level Objectives with error budget tracking,
    burn rate alerting, and status reporting.
    """

    def __init__(self):
        self._slos: Dict[str, SLODefinition] = {}
        self._violation_log: List[Dict[str, Any]] = []

    def define_slo(
        self,
        name: str,
        description: str,
        metric_name: str,
        threshold: float,
        comparison: str = "lt",
        window_seconds: int = 3600,
        error_budget_pct: float = 0.1,
        owner: str = "platform",
    ) -> SLODefinition:
        slo = SLODefinition(
            slo_id=str(uuid.uuid4()),
            name=name,
            description=description,
            metric_name=metric_name,
            threshold=threshold,
            comparison=comparison,
            window_seconds=window_seconds,
            error_budget_pct=error_budget_pct,
            owner=owner,
        )
        self._slos[slo.slo_id] = slo
        return slo

    def evaluate_slo(
        self, slo_id: str, metrics: MetricsAggregator
    ) -> Dict[str, Any]:
        slo = self._slos.get(slo_id)
        if slo is None:
            return {"error": "SLO not found"}
        metric = metrics.get_metric(slo.metric_name)
        if metric is None:
            return {"slo_id": slo_id, "status": SLOStatus.UNKNOWN.value}

        current_value = metric.get("mean", 0.0)
        slo_met = self._check_threshold(current_value, slo.threshold, slo.comparison)
        status = SLOStatus.MET if slo_met else SLOStatus.BREACHED

        # Compute error budget
        p95 = metric.get("p95", current_value)
        error_budget_used = min(1.0, p95 / max(slo.threshold, 0.001))

        if error_budget_used > 0.8:
            status = SLOStatus.AT_RISK if slo_met else SLOStatus.BREACHED

        if status == SLOStatus.BREACHED:
            self._violation_log.append(
                {
                    "slo_id": slo_id,
                    "slo_name": slo.name,
                    "current_value": current_value,
                    "threshold": slo.threshold,
                    "violated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        return {
            "slo_id": slo_id,
            "name": slo.name,
            "status": status.value,
            "current_value": round(current_value, 4),
            "threshold": slo.threshold,
            "error_budget_used_pct": round(error_budget_used * 100, 2),
            "metric_stats": metric,
        }

    def evaluate_all(self, metrics: MetricsAggregator) -> List[Dict[str, Any]]:
        return [self.evaluate_slo(slo_id, metrics) for slo_id in self._slos]

    def get_violation_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._violation_log[-limit:]

    @staticmethod
    def _check_threshold(value: float, threshold: float, comparison: str) -> bool:
        if comparison == "lt":
            return value < threshold
        if comparison == "lte":
            return value <= threshold
        if comparison == "gt":
            return value > threshold
        if comparison == "gte":
            return value >= threshold
        if comparison == "eq":
            return abs(value - threshold) < 0.001
        return True


# ─────────────────────── Optimization Advisor ───────────────────────────────


class OptimizationAdvisor:
    """
    Analyzes profiling data to generate actionable optimization recommendations
    prioritized by impact and effort.
    """

    def generate_recommendations(
        self,
        call_graph: CallGraphAnalyzer,
        trace_collector: TraceCollector,
        metrics: MetricsAggregator,
        slo_results: List[Dict[str, Any]],
    ) -> List[OptimizationRecommendation]:
        recommendations: List[OptimizationRecommendation] = []

        # Check for hot functions
        hotspots = call_graph.get_hotspots(5)
        for hotspot in hotspots:
            if hotspot["self_time_ms"] > 50:
                recommendations.append(
                    OptimizationRecommendation(
                        rec_id=str(uuid.uuid4()),
                        category=OptimizationCategory.CPU,
                        title=f"Optimize hot function: {hotspot['function_name']}",
                        description=(
                            f"Function {hotspot['function_name']} in {hotspot['module']} "
                            f"accounts for {hotspot['self_time_ms']:.1f}ms total self-time "
                            f"across {hotspot['call_count']} calls."
                        ),
                        impact="high",
                        effort="medium",
                        estimated_improvement_pct=20.0,
                        affected_functions=[f"{hotspot['module']}.{hotspot['function_name']}"],
                        code_suggestions=[
                            "Consider memoization for pure functions",
                            "Profile inner loops for algorithmic improvements",
                            "Check for redundant computations",
                        ],
                        priority_score=hotspot["self_time_ms"] / 100,
                    )
                )

        # Latency recommendations
        latency_stats = trace_collector.get_latency_stats()
        p99 = latency_stats.get("p99_ms", 0)
        if p99 > 500:
            recommendations.append(
                OptimizationRecommendation(
                    rec_id=str(uuid.uuid4()),
                    category=OptimizationCategory.ASYNC,
                    title="High p99 latency detected",
                    description=(
                        f"p99 latency is {p99:.1f}ms, which may impact user experience."
                    ),
                    impact="high",
                    effort="high",
                    estimated_improvement_pct=40.0,
                    affected_functions=[],
                    code_suggestions=[
                        "Convert synchronous I/O operations to async/await",
                        "Implement request parallelism where safe",
                        "Add response caching for frequently accessed data",
                        "Profile database queries for N+1 issues",
                    ],
                    priority_score=min(1.0, p99 / 1000),
                )
            )

        # SLO breach recommendations
        breached_slos = [r for r in slo_results if r.get("status") == SLOStatus.BREACHED.value]
        for slo_result in breached_slos:
            recommendations.append(
                OptimizationRecommendation(
                    rec_id=str(uuid.uuid4()),
                    category=OptimizationCategory.ARCHITECTURE,
                    title=f"SLO Breach: {slo_result.get('name', 'Unknown')}",
                    description=(
                        f"SLO '{slo_result.get('name')}' is breached. "
                        f"Current: {slo_result.get('current_value')}, "
                        f"Threshold: {slo_result.get('threshold')}"
                    ),
                    impact="critical",
                    effort="high",
                    estimated_improvement_pct=35.0,
                    affected_functions=[],
                    code_suggestions=[
                        "Investigate root cause of SLO violation",
                        "Consider horizontal scaling if CPU-bound",
                        "Review and optimize critical code paths",
                        "Implement circuit breakers to prevent cascading failures",
                    ],
                    priority_score=1.0,
                )
            )

        # Caching recommendation if high call counts
        hot_spans = trace_collector.get_hot_spans(5)
        for span in hot_spans:
            if span["call_count"] > 1000:
                recommendations.append(
                    OptimizationRecommendation(
                        rec_id=str(uuid.uuid4()),
                        category=OptimizationCategory.CACHING,
                        title=f"Cache high-frequency span: {span['name']}",
                        description=(
                            f"Span '{span['name']}' was called {span['call_count']} times. "
                            f"Caching its results could significantly reduce load."
                        ),
                        impact="medium",
                        effort="low",
                        estimated_improvement_pct=30.0,
                        affected_functions=[span["name"]],
                        code_suggestions=[
                            "Implement Redis-based result caching with appropriate TTL",
                            "Add cache invalidation on data mutations",
                            "Consider in-memory LRU cache for hot paths",
                        ],
                        priority_score=min(1.0, span["call_count"] / 10000),
                    )
                )

        return sorted(recommendations, key=lambda r: r.priority_score, reverse=True)


# ────────────────────── Performance Profiler ────────────────────────────────


class PerformanceProfiler:
    """
    Master performance profiler integrating trace collection, call graph analysis,
    metrics aggregation, SLO monitoring, and optimization recommendations.
    """

    def __init__(self):
        self.tracer = TraceCollector()
        self.metrics = MetricsAggregator()
        self.call_graph = CallGraphAnalyzer()
        self.slo_monitor = SLOMonitor()
        self.advisor = OptimizationAdvisor()
        self._profiling_enabled = True
        self._setup_default_metrics()
        self._setup_default_slos()

    def _setup_default_metrics(self) -> None:
        self.metrics.register("request_latency_ms", MetricType.HISTOGRAM, "ms")
        self.metrics.register("request_count", MetricType.COUNTER)
        self.metrics.register("error_count", MetricType.COUNTER)
        self.metrics.register("memory_usage_mb", MetricType.GAUGE, "MB")
        self.metrics.register("cpu_usage_pct", MetricType.GAUGE, "%")
        self.metrics.register("active_connections", MetricType.GAUGE)
        self.metrics.register("cache_hit_rate", MetricType.GAUGE, "%")
        self.metrics.register("db_query_time_ms", MetricType.HISTOGRAM, "ms")

    def _setup_default_slos(self) -> None:
        self.slo_monitor.define_slo(
            name="API Latency p99 < 500ms",
            description="99th percentile API response time must be under 500ms",
            metric_name="request_latency_ms",
            threshold=500.0,
            comparison="lt",
            window_seconds=3600,
            error_budget_pct=0.01,
        )
        self.slo_monitor.define_slo(
            name="Error Rate < 1%",
            description="Error rate should not exceed 1%",
            metric_name="error_count",
            threshold=100.0,
            comparison="lt",
            window_seconds=3600,
            error_budget_pct=0.05,
        )
        self.slo_monitor.define_slo(
            name="Cache Hit Rate > 80%",
            description="Cache hit rate should be at least 80%",
            metric_name="cache_hit_rate",
            threshold=80.0,
            comparison="gte",
            window_seconds=3600,
            error_budget_pct=0.1,
        )

    def record_request(
        self,
        endpoint: str,
        method: str,
        latency_ms: float,
        status_code: int,
        user_id: Optional[str] = None,
    ) -> None:
        if not self._profiling_enabled:
            return
        self.metrics.record("request_latency_ms", latency_ms)
        self.metrics.increment("request_count")
        if status_code >= 500:
            self.metrics.increment("error_count")
        self.call_graph.record_call(
            function_name=f"{method}:{endpoint}",
            module="api",
            duration_ms=latency_ms,
        )

    def record_db_query(
        self,
        query_type: str,
        table: str,
        duration_ms: float,
        rows_affected: int = 0,
    ) -> None:
        if not self._profiling_enabled:
            return
        self.metrics.record("db_query_time_ms", duration_ms)
        self.call_graph.record_call(
            function_name=f"{query_type}:{table}",
            module="database",
            duration_ms=duration_ms,
        )

    def update_system_metrics(
        self,
        memory_mb: float,
        cpu_pct: float,
        active_connections: int,
        cache_hit_rate: float,
    ) -> None:
        self.metrics.record("memory_usage_mb", memory_mb)
        self.metrics.record("cpu_usage_pct", cpu_pct)
        self.metrics.record("active_connections", float(active_connections))
        self.metrics.record("cache_hit_rate", cache_hit_rate)

    def get_performance_report(self) -> Dict[str, Any]:
        slo_results = self.slo_monitor.evaluate_all(self.metrics)
        recommendations = self.advisor.generate_recommendations(
            self.call_graph,
            self.tracer,
            self.metrics,
            slo_results,
        )
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_stats": self.tracer.get_latency_stats(),
            "metrics_summary": self.metrics.get_all_metrics(),
            "call_graph_stats": self.call_graph.get_graph_stats(),
            "hotspots": self.call_graph.get_hotspots(10),
            "slo_status": slo_results,
            "recommendations": [r.to_dict() for r in recommendations[:10]],
            "recent_traces": self.tracer.get_recent_traces(5),
            "slo_violations": self.slo_monitor.get_violation_log(10),
        }

    def get_dashboard_summary(self) -> Dict[str, Any]:
        latency = self.tracer.get_latency_stats()
        slo_results = self.slo_monitor.evaluate_all(self.metrics)
        breached = sum(1 for r in slo_results if r.get("status") == SLOStatus.BREACHED.value)
        return {
            "p99_latency_ms": latency.get("p99_ms", 0),
            "mean_latency_ms": latency.get("mean_ms", 0),
            "total_request_count": (
                self.metrics.get_metric("request_count") or {}
            ).get("count", 0),
            "error_count": (
                self.metrics.get_metric("error_count") or {}
            ).get("count", 0),
            "slos_defined": len(self.slo_monitor._slos),
            "slos_breached": breached,
            "hotspot_count": len(self.call_graph._nodes),
            "profiling_enabled": self._profiling_enabled,
        }

    def enable(self) -> None:
        self._profiling_enabled = True

    def disable(self) -> None:
        self._profiling_enabled = False
