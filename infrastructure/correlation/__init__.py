"""Request Correlation Module — CognitionOS Production Infrastructure."""

from .correlation_engine import (
    CorrelationEngine,
    Span,
    Trace,
    SpanKind,
    SpanStatus,
)

__all__ = [
    "CorrelationEngine",
    "Span",
    "Trace",
    "SpanKind",
    "SpanStatus",
]
