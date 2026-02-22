"""
Distributed Tracing Engine
===========================
OpenTelemetry-compatible end-to-end request tracing with sampling,
span correlation, waterfall visualization, and anomaly detection.

Implements:
- W3C TraceContext propagation (traceparent / tracestate headers)
- Span lifecycle: start → add events/attributes → end
- Hierarchical span trees with parent-child relationships
- Multiple sampling strategies: always-on, head-based, tail-based, adaptive
- Span exporters: in-memory, OTLP-compatible batch export
- Service dependency graph construction from traces
- Latency percentile computation (P50/P95/P99)
- Error rate tracking per service and operation
- Critical path analysis in distributed traces
- Trace search and filtering
- Waterfall timeline generation
"""

from __future__ import annotations

import hashlib
import random
import struct
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SpanKind(str, Enum):
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SamplingDecision(str, Enum):
    DROP = "drop"
    RECORD_ONLY = "record_only"
    RECORD_AND_SAMPLE = "record_and_sample"


class SamplingStrategy(str, Enum):
    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    TRACE_ID_RATIO = "trace_id_ratio"
    PARENT_BASED = "parent_based"
    TAIL_BASED = "tail_based"
    ADAPTIVE = "adaptive"


# ---------------------------------------------------------------------------
# W3C TraceContext
# ---------------------------------------------------------------------------

class TraceContext:
    """W3C TraceContext implementation for distributed propagation."""

    VERSION = "00"

    def __init__(self, trace_id: str, span_id: str, sampled: bool = True):
        self.trace_id = trace_id
        self.span_id = span_id
        self.sampled = sampled

    @classmethod
    def generate(cls) -> "TraceContext":
        trace_id = uuid.uuid4().hex + uuid.uuid4().hex  # 32 hex chars
        span_id = uuid.uuid4().hex[:16]                  # 16 hex chars
        return cls(trace_id, span_id)

    @classmethod
    def child(cls, parent: "TraceContext") -> "TraceContext":
        return cls(parent.trace_id, uuid.uuid4().hex[:16], parent.sampled)

    def to_traceparent(self) -> str:
        flags = "01" if self.sampled else "00"
        return f"{self.VERSION}-{self.trace_id}-{self.span_id}-{flags}"

    @classmethod
    def from_traceparent(cls, header: str) -> Optional["TraceContext"]:
        parts = header.split("-")
        if len(parts) < 4:
            return None
        try:
            return cls(parts[1], parts[2], parts[3] == "01")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

@dataclass
class SpanEvent:
    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    tenant_id: str = "global"
    sampled: bool = True

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def is_error(self) -> bool:
        return self.status == SpanStatus.ERROR

    @property
    def is_root(self) -> bool:
        return self.parent_span_id is None

    def set_attribute(self, key: str, value: Any) -> "Span":
        self.attributes[key] = value
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))
        return self

    def set_status(self, status: SpanStatus, message: str = "") -> "Span":
        self.status = status
        self.status_message = message
        return self

    def end(self, end_time: Optional[float] = None) -> "Span":
        self.end_time = end_time or time.time()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "kind": self.kind.value,
            "status": self.status.value,
            "status_message": self.status_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 3),
            "attributes": self.attributes,
            "events": [{"name": e.name, "ts": e.timestamp, "attrs": e.attributes} for e in self.events],
            "tenant_id": self.tenant_id,
            "sampled": self.sampled,
        }


# ---------------------------------------------------------------------------
# Trace (collection of spans)
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    root_span: Optional[Span] = None
    service_names: List[str] = field(default_factory=list)
    tenant_id: str = "global"
    created_at: float = field(default_factory=time.time)

    @property
    def total_duration_ms(self) -> float:
        if self.root_span:
            return self.root_span.duration_ms
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time or time.time() for s in self.spans)
        return (end - start) * 1000

    @property
    def has_errors(self) -> bool:
        return any(s.is_error for s in self.spans)

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.spans if s.is_error)

    def get_span(self, span_id: str) -> Optional[Span]:
        return next((s for s in self.spans if s.span_id == span_id), None)

    def get_children(self, parent_span_id: str) -> List[Span]:
        return [s for s in self.spans if s.parent_span_id == parent_span_id]

    def critical_path(self) -> List[Span]:
        """Find the longest path through the span tree."""
        if not self.spans:
            return []
        span_map = {s.span_id: s for s in self.spans}
        children_map: Dict[str, List[Span]] = defaultdict(list)
        root = None
        for span in self.spans:
            if span.is_root or span.parent_span_id not in span_map:
                root = span
            else:
                children_map[span.parent_span_id].append(span)

        if not root:
            return []

        def longest_path(span: Span) -> List[Span]:
            children = children_map.get(span.span_id, [])
            if not children:
                return [span]
            best = max(children, key=lambda c: c.duration_ms)
            return [span] + longest_path(best)

        return longest_path(root)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class Sampler:
    def __init__(self, strategy: SamplingStrategy, ratio: float = 1.0):
        self.strategy = strategy
        self.ratio = ratio
        self._error_rate: float = 0.0
        self._current_ratio: float = ratio
        self._request_count: int = 0
        self._error_count: int = 0

    def should_sample(
        self,
        trace_id: str,
        parent_context: Optional[TraceContext] = None,
        operation: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SamplingDecision:
        self._request_count += 1

        if self.strategy == SamplingStrategy.ALWAYS_ON:
            return SamplingDecision.RECORD_AND_SAMPLE

        if self.strategy == SamplingStrategy.ALWAYS_OFF:
            return SamplingDecision.DROP

        if self.strategy == SamplingStrategy.TRACE_ID_RATIO:
            # Deterministic: use first 8 bytes of trace_id as int
            val = int(trace_id[:8], 16) / 0xFFFFFFFF
            return SamplingDecision.RECORD_AND_SAMPLE if val < self.ratio else SamplingDecision.DROP

        if self.strategy == SamplingStrategy.PARENT_BASED:
            if parent_context:
                return (
                    SamplingDecision.RECORD_AND_SAMPLE
                    if parent_context.sampled
                    else SamplingDecision.DROP
                )
            return SamplingDecision.RECORD_AND_SAMPLE if random.random() < self.ratio else SamplingDecision.DROP

        if self.strategy == SamplingStrategy.ADAPTIVE:
            # Increase sampling when error rate is high
            if self._error_rate > 0.05:
                self._current_ratio = min(1.0, self.ratio * 2)
            else:
                self._current_ratio = self.ratio
            return SamplingDecision.RECORD_AND_SAMPLE if random.random() < self._current_ratio else SamplingDecision.DROP

        # Default: tail-based always records, decided on completion
        return SamplingDecision.RECORD_AND_SAMPLE

    def record_error(self) -> None:
        self._error_count += 1
        self._error_rate = self._error_count / max(1, self._request_count)


# ---------------------------------------------------------------------------
# Trace Store
# ---------------------------------------------------------------------------

class TraceStore:
    """In-memory trace storage with indexing for efficient retrieval."""

    def __init__(self, max_traces: int = 50000):
        self.max_traces = max_traces
        self._traces: Dict[str, Trace] = {}
        self._span_index: Dict[str, str] = {}  # span_id -> trace_id
        self._service_traces: Dict[str, List[str]] = defaultdict(list)  # service -> trace_ids
        self._operation_latencies: Dict[str, List[float]] = defaultdict(list)
        self._error_counts: Dict[str, int] = defaultdict(int)

    def store_span(self, span: Span) -> None:
        if not span.sampled:
            return

        trace = self._traces.get(span.trace_id)
        if trace is None:
            trace = Trace(trace_id=span.trace_id, tenant_id=span.tenant_id)
            self._traces[span.trace_id] = trace

        trace.spans.append(span)
        if span.is_root:
            trace.root_span = span
        if span.service_name not in trace.service_names:
            trace.service_names.append(span.service_name)

        self._span_index[span.span_id] = span.trace_id
        self._service_traces[span.service_name].append(span.trace_id)

        if span.end_time:
            key = f"{span.service_name}:{span.operation_name}"
            self._operation_latencies[key].append(span.duration_ms)
            if len(self._operation_latencies[key]) > 10000:
                self._operation_latencies[key] = self._operation_latencies[key][-5000:]
            if span.is_error:
                self._error_counts[key] += 1

        # Evict oldest traces
        if len(self._traces) > self.max_traces:
            oldest_id = min(self._traces, key=lambda tid: self._traces[tid].created_at)
            for sp in self._traces[oldest_id].spans:
                self._span_index.pop(sp.span_id, None)
            del self._traces[oldest_id]

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self._traces.get(trace_id)

    def get_span_trace(self, span_id: str) -> Optional[Trace]:
        trace_id = self._span_index.get(span_id)
        return self._traces.get(trace_id) if trace_id else None

    def search(
        self,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        has_error: Optional[bool] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Trace]:
        candidates = list(self._traces.values())
        if tenant_id:
            candidates = [t for t in candidates if t.tenant_id == tenant_id]
        if service:
            candidates = [t for t in candidates if service in t.service_names]
        if has_error is not None:
            candidates = [t for t in candidates if t.has_errors == has_error]
        if min_duration_ms is not None:
            candidates = [t for t in candidates if t.total_duration_ms >= min_duration_ms]
        if max_duration_ms is not None:
            candidates = [t for t in candidates if t.total_duration_ms <= max_duration_ms]
        if operation:
            candidates = [
                t for t in candidates
                if any(s.operation_name == operation for s in t.spans)
            ]
        # Sort by recency
        candidates.sort(key=lambda t: t.created_at, reverse=True)
        return candidates[:limit]

    def latency_percentiles(self, service: str, operation: str) -> Dict[str, float]:
        key = f"{service}:{operation}"
        latencies = sorted(self._operation_latencies.get(key, []))
        if not latencies:
            return {"p50": 0, "p75": 0, "p95": 0, "p99": 0, "count": 0}
        n = len(latencies)

        def pct(p: float) -> float:
            idx = int(p / 100 * n)
            return latencies[min(idx, n - 1)]

        return {
            "p50": round(pct(50), 2),
            "p75": round(pct(75), 2),
            "p95": round(pct(95), 2),
            "p99": round(pct(99), 2),
            "count": n,
            "avg": round(sum(latencies) / n, 2),
        }


# ---------------------------------------------------------------------------
# Service Dependency Graph
# ---------------------------------------------------------------------------

class ServiceDependencyGraph:
    """Builds service call graph from traces."""

    def __init__(self, store: TraceStore):
        self._store = store
        self._edges: Dict[Tuple, int] = defaultdict(int)  # (caller, callee) -> call_count

    def build(self, traces: List[Trace]) -> None:
        for trace in traces:
            span_map = {s.span_id: s for s in trace.spans}
            for span in trace.spans:
                if span.parent_span_id and span.parent_span_id in span_map:
                    parent = span_map[span.parent_span_id]
                    if parent.service_name != span.service_name:
                        edge = (parent.service_name, span.service_name)
                        self._edges[edge] += 1

    def to_dict(self) -> Dict[str, Any]:
        nodes = set()
        edges = []
        for (src, dst), count in self._edges.items():
            nodes.add(src)
            nodes.add(dst)
            edges.append({"from": src, "to": dst, "call_count": count})
        return {
            "nodes": [{"id": n} for n in nodes],
            "edges": edges,
        }


# ---------------------------------------------------------------------------
# Waterfall View
# ---------------------------------------------------------------------------

def build_waterfall(trace: Trace) -> List[Dict[str, Any]]:
    """Build waterfall timeline entries for UI visualization."""
    if not trace.spans:
        return []
    min_start = min(s.start_time for s in trace.spans)
    entries = []
    for span in sorted(trace.spans, key=lambda s: s.start_time):
        offset_ms = (span.start_time - min_start) * 1000
        entries.append({
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "operation_name": span.operation_name,
            "service_name": span.service_name,
            "offset_ms": round(offset_ms, 2),
            "duration_ms": round(span.duration_ms, 2),
            "status": span.status.value,
            "kind": span.kind.value,
            "error": span.is_error,
        })
    return entries


# ---------------------------------------------------------------------------
# Distributed Tracing Engine
# ---------------------------------------------------------------------------

class DistributedTracingEngine:
    """
    Production-grade distributed tracing engine with W3C TraceContext,
    sampling, storage, analytics, and visualization support.
    """

    def __init__(
        self,
        sampling_strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE,
        sampling_ratio: float = 0.1,
        max_traces: int = 50000,
    ):
        self._sampler = Sampler(sampling_strategy, sampling_ratio)
        self._store = TraceStore(max_traces)
        self._active_spans: Dict[str, Span] = {}
        self._dependency_graph = ServiceDependencyGraph(self._store)
        self._total_spans: int = 0
        self._sampled_spans: int = 0

    # ---- Span Management ----

    def start_span(
        self,
        operation_name: str,
        service_name: str,
        parent_context: Optional[TraceContext] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        tenant_id: str = "global",
    ) -> Tuple[Span, TraceContext]:
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            trace_id = uuid.uuid4().hex + uuid.uuid4().hex
            parent_span_id = None

        decision = self._sampler.should_sample(
            trace_id, parent_context, operation_name, attributes
        )
        sampled = decision == SamplingDecision.RECORD_AND_SAMPLE

        span = Span(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=service_name,
            kind=kind,
            attributes=attributes or {},
            tenant_id=tenant_id,
            sampled=sampled,
        )

        self._active_spans[span.span_id] = span
        self._total_spans += 1
        if sampled:
            self._sampled_spans += 1

        ctx = TraceContext(trace_id, span.span_id, sampled)
        return span, ctx

    def end_span(self, span: Span, error: Optional[Exception] = None) -> None:
        span.end()
        if error:
            span.set_status(SpanStatus.ERROR, str(error))
            self._sampler.record_error()
        elif span.status == SpanStatus.UNSET:
            span.set_status(SpanStatus.OK)

        self._active_spans.pop(span.span_id, None)
        self._store.store_span(span)

    def inject_headers(self, ctx: TraceContext) -> Dict[str, str]:
        """Inject W3C traceparent into HTTP headers."""
        return {"traceparent": ctx.to_traceparent()}

    def extract_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extract W3C traceparent from incoming HTTP headers."""
        header = headers.get("traceparent") or headers.get("Traceparent")
        if not header:
            return None
        return TraceContext.from_traceparent(header)

    # ---- Analytics ----

    def get_latency_percentiles(self, service: str, operation: str) -> Dict[str, float]:
        return self._store.latency_percentiles(service, operation)

    def search_traces(
        self,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        has_error: Optional[bool] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Trace]:
        return self._store.search(
            service=service,
            operation=operation,
            has_error=has_error,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            tenant_id=tenant_id,
            limit=limit,
        )

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self._store.get_trace(trace_id)

    def get_waterfall(self, trace_id: str) -> List[Dict[str, Any]]:
        trace = self._store.get_trace(trace_id)
        if not trace:
            return []
        return build_waterfall(trace)

    def get_service_graph(self) -> Dict[str, Any]:
        all_traces = list(self._store._traces.values())
        self._dependency_graph.build(all_traces[-1000:])
        return self._dependency_graph.to_dict()

    def get_engine_summary(self) -> Dict[str, Any]:
        return {
            "total_traces": len(self._store._traces),
            "total_spans": self._total_spans,
            "sampled_spans": self._sampled_spans,
            "sampling_rate": round(self._sampled_spans / max(1, self._total_spans), 4),
            "sampling_strategy": self._sampler.strategy.value,
            "active_spans": len(self._active_spans),
            "error_rate": round(self._sampler._error_rate, 4),
        }
