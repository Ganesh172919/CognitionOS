"""
Observability Infrastructure
"""

from .logging import (
    setup_structured_logging,
    set_trace_id,
    get_trace_id,
    clear_trace_id,
    get_logger,
    LoggerAdapter,
)

from .tracing import (
    setup_tracing,
    instrument_fastapi,
    instrument_sqlalchemy,
    instrument_redis,
    instrument_http_clients,
    get_tracer,
    trace_operation,
    add_span_attributes,
    add_span_event,
)

from .metrics import (
    PrometheusMiddleware,
    create_metrics_app,
    track_workflow_execution,
    track_llm_request,
    track_event_published,
    track_event_consumed,
    # Metrics objects
    http_requests_total,
    http_request_duration_seconds,
    workflows_created_total,
    workflow_executions_total,
    llm_requests_total,
    auth_requests_total,
)

__all__ = [
    # Logging
    "setup_structured_logging",
    "set_trace_id",
    "get_trace_id",
    "clear_trace_id",
    "get_logger",
    "LoggerAdapter",
    # Tracing
    "setup_tracing",
    "instrument_fastapi",
    "instrument_sqlalchemy",
    "instrument_redis",
    "instrument_http_clients",
    "get_tracer",
    "trace_operation",
    "add_span_attributes",
    "add_span_event",
    # Metrics
    "PrometheusMiddleware",
    "create_metrics_app",
    "track_workflow_execution",
    "track_llm_request",
    "track_event_published",
    "track_event_consumed",
    "http_requests_total",
    "http_request_duration_seconds",
    "workflows_created_total",
    "workflow_executions_total",
    "llm_requests_total",
    "auth_requests_total",
]
