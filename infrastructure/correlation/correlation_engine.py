"""
Request Correlation Engine — CognitionOS Production Infrastructure

End-to-end request tracing with correlation ID propagation across
service boundaries, event bus, task queue, and database queries.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Context vars for propagation
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_span_id: ContextVar[str] = ContextVar("span_id", default="")
_parent_span_id: ContextVar[str] = ContextVar("parent_span_id", default="")


class SpanKind(str, Enum):
    SERVER = "server"
    CLIENT = "client"
    INTERNAL = "internal"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class Span:
    """A single unit of work within a trace."""
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    operation: str = ""
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.OK
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    duration_ms: float = 0
    tags: Dict[str, str] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def finish(self, *, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.events.append({
            "name": name, "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id, "span_id": self.span_id,
            "parent_span_id": self.parent_span_id, "operation": self.operation,
            "kind": self.kind.value, "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "tags": self.tags, "events": self.events, "error": self.error,
        }


@dataclass
class Trace:
    """A complete trace with all its spans."""
    trace_id: str
    root_operation: str = ""
    spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    tenant_id: str = ""
    user_id: str = ""

    @property
    def duration_ms(self) -> float:
        if not self.spans:
            return 0
        return sum(s.duration_ms for s in self.spans if s.parent_span_id == "")

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def has_errors(self) -> bool:
        return any(s.status == SpanStatus.ERROR for s in self.spans)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id, "root_operation": self.root_operation,
            "duration_ms": round(self.duration_ms, 2),
            "span_count": self.span_count, "has_errors": self.has_errors,
            "tenant_id": self.tenant_id, "spans": [s.to_dict() for s in self.spans],
        }


class CorrelationEngine:
    """
    Request correlation and distributed tracing engine.

    Generates trace/span IDs at API gateway level and propagates
    them through all service boundaries for end-to-end observability.
    """

    def __init__(self, *, service_name: str = "cognitionos",
                 max_traces: int = 10000, sample_rate: float = 1.0):
        self._service_name = service_name
        self._max_traces = max_traces
        self._sample_rate = sample_rate
        self._traces: Dict[str, Trace] = {}
        self._active_spans: Dict[str, Span] = {}
        self._total_traces = 0
        self._total_spans = 0
        self._error_spans = 0
        self._latency_histogram: List[float] = []
        logger.info("CorrelationEngine initialized (service=%s)", service_name)

    def start_trace(self, operation: str, *, tenant_id: str = "",
                    user_id: str = "", trace_id: str = "") -> Span:
        """Start a new trace with a root span."""
        tid = trace_id or uuid.uuid4().hex[:16]
        sid = uuid.uuid4().hex[:12]

        _trace_id.set(tid)
        _span_id.set(sid)
        _parent_span_id.set("")

        trace = Trace(trace_id=tid, root_operation=operation,
                      tenant_id=tenant_id, user_id=user_id)
        span = Span(trace_id=tid, span_id=sid, operation=operation,
                    kind=SpanKind.SERVER, tags={"service": self._service_name})

        trace.spans.append(span)
        self._traces[tid] = trace
        self._active_spans[sid] = span
        self._total_traces += 1
        self._total_spans += 1

        # Evict old traces if at capacity
        if len(self._traces) > self._max_traces:
            oldest = sorted(self._traces.keys(),
                            key=lambda k: self._traces[k].start_time)[:100]
            for old_tid in oldest:
                del self._traces[old_tid]

        return span

    def start_span(self, operation: str, *, kind: SpanKind = SpanKind.INTERNAL,
                   tags: Optional[Dict[str, str]] = None) -> Span:
        """Start a child span within the current trace."""
        tid = _trace_id.get("")
        if not tid:
            return self.start_trace(operation)

        parent_sid = _span_id.get("")
        sid = uuid.uuid4().hex[:12]

        _span_id.set(sid)
        _parent_span_id.set(parent_sid)

        span = Span(trace_id=tid, span_id=sid, parent_span_id=parent_sid,
                    operation=operation, kind=kind,
                    tags={**(tags or {}), "service": self._service_name})

        trace = self._traces.get(tid)
        if trace:
            trace.spans.append(span)
        self._active_spans[sid] = span
        self._total_spans += 1

        return span

    def finish_span(self, span: Span, *, status: SpanStatus = SpanStatus.OK,
                    error: Optional[str] = None):
        """Finish a span and record its duration."""
        span.finish(status=status, error=error)
        self._active_spans.pop(span.span_id, None)

        if status == SpanStatus.ERROR:
            self._error_spans += 1

        self._latency_histogram.append(span.duration_ms)
        if len(self._latency_histogram) > 10000:
            self._latency_histogram = self._latency_histogram[-10000:]

        # Restore parent span
        if span.parent_span_id:
            _span_id.set(span.parent_span_id)
            _parent_span_id.set("")

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self._traces.get(trace_id)

    def get_current_trace_id(self) -> str:
        return _trace_id.get("")

    def get_current_span_id(self) -> str:
        return _span_id.get("")

    def inject_headers(self) -> Dict[str, str]:
        """Get headers for propagating trace context to downstream services."""
        return {
            "X-Trace-Id": _trace_id.get(""),
            "X-Span-Id": _span_id.get(""),
            "X-Parent-Span-Id": _parent_span_id.get(""),
        }

    def extract_headers(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract trace context from incoming request headers."""
        trace_id = headers.get("X-Trace-Id", "")
        if trace_id:
            _trace_id.set(trace_id)
            parent = headers.get("X-Span-Id", "")
            if parent:
                _parent_span_id.set(parent)
        return trace_id or None

    class SpanContext:
        """Context manager for automatic span lifecycle."""
        def __init__(self, engine: "CorrelationEngine", operation: str,
                     kind: SpanKind, tags: Optional[Dict[str, str]]):
            self._engine = engine
            self._operation = operation
            self._kind = kind
            self._tags = tags
            self._span: Optional[Span] = None

        def __enter__(self) -> Span:
            self._span = self._engine.start_span(
                self._operation, kind=self._kind, tags=self._tags
            )
            return self._span

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._span:
                status = SpanStatus.ERROR if exc_type else SpanStatus.OK
                error = str(exc_val) if exc_val else None
                self._engine.finish_span(self._span, status=status, error=error)
            return False

    def span(self, operation: str, *, kind: SpanKind = SpanKind.INTERNAL,
             tags: Optional[Dict[str, str]] = None) -> SpanContext:
        """Create a span context manager for automatic lifecycle management."""
        return self.SpanContext(self, operation, kind, tags)

    def get_stats(self) -> Dict[str, Any]:
        latencies = sorted(self._latency_histogram) if self._latency_histogram else [0]
        p50_idx = int(len(latencies) * 0.5)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)
        return {
            "service_name": self._service_name,
            "total_traces": self._total_traces,
            "total_spans": self._total_spans,
            "active_spans": len(self._active_spans),
            "stored_traces": len(self._traces),
            "error_spans": self._error_spans,
            "error_rate": round(self._error_spans / max(self._total_spans, 1), 4),
            "latency_p50_ms": round(latencies[p50_idx], 2),
            "latency_p95_ms": round(latencies[min(p95_idx, len(latencies) - 1)], 2),
            "latency_p99_ms": round(latencies[min(p99_idx, len(latencies) - 1)], 2),
        }
