"""
Structured Logger - JSON logs with trace and tenant IDs.

Extends infrastructure.observability.logging with tenant context.
All log records include trace_id and tenant_id when available.
"""

import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger


trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
tenant_id_var: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)


def set_trace_id(trace_id: str) -> None:
    """Set trace ID in context for correlation across services."""
    trace_id_var.set(trace_id)


def get_trace_id() -> Optional[str]:
    """Get trace ID from context."""
    return trace_id_var.get()


def clear_trace_id() -> None:
    """Clear trace ID from context."""
    try:
        trace_id_var.set(None)
    except LookupError:
        pass


def set_tenant_id(tenant_id: Optional[str]) -> None:
    """Set tenant ID in context for multi-tenant log correlation."""
    tenant_id_var.set(tenant_id)


def get_tenant_id() -> Optional[str]:
    """Get tenant ID from context."""
    return tenant_id_var.get()


def clear_tenant_id() -> None:
    """Clear tenant ID from context."""
    try:
        tenant_id_var.set(None)
    except LookupError:
        pass


def set_log_context(trace_id: Optional[str] = None, tenant_id: Optional[str] = None) -> None:
    """Set both trace and tenant context."""
    if trace_id is not None:
        trace_id_var.set(trace_id)
    if tenant_id is not None:
        tenant_id_var.set(tenant_id)


def clear_log_context() -> None:
    """Clear trace and tenant context."""
    clear_trace_id()
    clear_tenant_id()


class StructuredLogFormatter(jsonlogger.JsonFormatter):
    """
    JSON formatter that adds trace_id and tenant_id to every log record.
    """

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        trace_id = trace_id_var.get()
        if trace_id:
            log_record["trace_id"] = trace_id
        tenant_id = tenant_id_var.get()
        if tenant_id:
            log_record["tenant_id"] = tenant_id
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that ensures trace_id and tenant_id in extra."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        extra = kwargs.get("extra", {})
        trace_id = trace_id_var.get()
        if trace_id and "trace_id" not in extra:
            extra["trace_id"] = trace_id
        tenant_id = tenant_id_var.get()
        if tenant_id and "tenant_id" not in extra:
            extra["tenant_id"] = tenant_id
        kwargs["extra"] = extra
        return msg, kwargs


def get_structured_logger(name: str, **context: Any) -> StructuredLoggerAdapter:
    """
    Get a structured logger with automatic trace/tenant context.
    """
    logger = logging.getLogger(name)
    return StructuredLoggerAdapter(logger, context)


def setup_structured_logger(
    name: str = "cognitionos",
    level: str = "INFO",
    json_format: bool = True,
) -> logging.Logger:
    """
    Configure root or named logger with structured JSON format.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    logger.addHandler(handler)
    return logger
