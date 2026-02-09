"""
Structured logging for CognitionOS.

Provides JSON-formatted logs with trace IDs for distributed tracing.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


# Context variable for trace ID (propagates through async code)
trace_id_ctx: ContextVar[Optional[UUID]] = ContextVar('trace_id', default=None)


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": str(trace_id_ctx.get()) if trace_id_ctx.get() else None,
        }

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location
        log_data["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: str = "INFO",
    json_format: bool = True
) -> logging.Logger:
    """
    Set up a logger with structured output.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, output JSON; if False, human-readable

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds extra fields to all log messages.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add extra fields to log record."""
        extra = kwargs.get("extra", {})

        # Add adapter's extra fields
        if hasattr(self, "extra"):
            extra.update(self.extra)

        # Add trace ID if available
        if trace_id_ctx.get():
            extra["trace_id"] = str(trace_id_ctx.get())

        kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs


def get_contextual_logger(
    name: str,
    **extra_fields: Any
) -> LoggerAdapter:
    """
    Get a logger that automatically includes extra fields.

    Args:
        name: Logger name
        **extra_fields: Additional fields to include in every log

    Returns:
        Logger adapter with extra fields

    Example:
        logger = get_contextual_logger(__name__, user_id=user.id, service="api-gateway")
        logger.info("Processing request")  # Will include user_id and service
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, extra_fields)


def set_trace_id(trace_id: Optional[UUID] = None) -> UUID:
    """
    Set the trace ID for current context.

    Args:
        trace_id: Trace ID to set. If None, generates a new one.

    Returns:
        The trace ID that was set
    """
    if trace_id is None:
        trace_id = uuid4()
    trace_id_ctx.set(trace_id)
    return trace_id


def get_trace_id() -> Optional[UUID]:
    """Get the current trace ID."""
    return trace_id_ctx.get()


def clear_trace_id():
    """Clear the trace ID from current context."""
    trace_id_ctx.set(None)


# Example usage
if __name__ == "__main__":
    # Basic logger
    logger = setup_logger(__name__, level="DEBUG")
    logger.info("This is an info message")
    logger.warning("This is a warning", extra={"user_id": "123"})

    # Contextual logger
    set_trace_id()
    ctx_logger = get_contextual_logger(
        __name__,
        service="test",
        environment="dev"
    )
    ctx_logger.info("Request started")
    ctx_logger.error("Request failed", extra={"error_code": "AUTH_001"})
