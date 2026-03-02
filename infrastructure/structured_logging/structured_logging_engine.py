"""
Structured Logging Engine — CognitionOS Production Infrastructure

Production-grade structured logging with:
- JSON-formatted log output
- Automatic correlation ID injection
- Configurable log levels per module
- Log sampling for high-volume paths
- Performance metrics in log entries
- Export compatibility (ELK, Datadog, CloudWatch)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Context variable for correlation ID propagation
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_tenant_id: ContextVar[str] = ContextVar("tenant_id", default="")
_user_id: ContextVar[str] = ContextVar("user_id", default="")
_request_id: ContextVar[str] = ContextVar("request_id", default="")

logger = logging.getLogger(__name__)


def get_correlation_id() -> str:
    return _correlation_id.get("")

def set_correlation_id(cid: str):
    _correlation_id.set(cid)

def set_request_context(*, correlation_id: str = "", tenant_id: str = "",
                        user_id: str = "", request_id: str = ""):
    if correlation_id:
        _correlation_id.set(correlation_id)
    if tenant_id:
        _tenant_id.set(tenant_id)
    if user_id:
        _user_id.set(user_id)
    if request_id:
        _request_id.set(request_id)


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SamplingConfig:
    """Configuration for log sampling on high-volume paths."""
    path_patterns: Dict[str, float] = field(default_factory=dict)  # pattern -> sample rate (0-1)
    default_rate: float = 1.0  # 1.0 = log everything

    def should_log(self, path: str) -> bool:
        import random
        for pattern, rate in self.path_patterns.items():
            if pattern in path:
                return random.random() < rate
        return random.random() < self.default_rate


class StructuredJsonFormatter(logging.Formatter):
    """JSON log formatter with automatic context injection."""

    def __init__(self, *, service_name: str = "cognitionos",
                 environment: str = "development",
                 include_stacktrace: bool = True):
        super().__init__()
        self._service_name = service_name
        self._environment = environment
        self._include_stacktrace = include_stacktrace

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self._service_name,
            "environment": self._environment,
        }

        # Inject context vars
        cid = _correlation_id.get("")
        if cid:
            log_entry["correlation_id"] = cid
        tid = _tenant_id.get("")
        if tid:
            log_entry["tenant_id"] = tid
        uid = _user_id.get("")
        if uid:
            log_entry["user_id"] = uid
        rid = _request_id.get("")
        if rid:
            log_entry["request_id"] = rid

        # Source location
        log_entry["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Process info
        log_entry["process"] = {
            "pid": record.process,
            "thread": record.thread,
        }

        # Extra fields
        if hasattr(record, "extra_fields"):
            log_entry["extra"] = record.extra_fields

        # Duration if present
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        # Exception info
        if record.exc_info and self._include_stacktrace:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, default=str)


class PerformanceLogger:
    """Context manager for logging operation performance."""

    def __init__(self, operation: str, logger_instance: logging.Logger,
                 *, level: int = logging.INFO, extra: Optional[Dict[str, Any]] = None):
        self._operation = operation
        self._logger = logger_instance
        self._level = level
        self._extra = extra or {}
        self._start: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._start) * 1000
        record_extra = {"duration_ms": duration_ms, "operation": self._operation}
        record_extra.update(self._extra)

        if exc_type:
            self._logger.error(
                "Operation '%s' failed after %.1fms: %s",
                self._operation, duration_ms, exc_val,
                extra={"extra_fields": record_extra},
            )
        else:
            self._logger.log(
                self._level,
                "Operation '%s' completed in %.1fms",
                self._operation, duration_ms,
                extra={"extra_fields": record_extra},
            )
        return False


class StructuredLoggingEngine:
    """
    Central logging engine that configures structured logging across the platform.
    """

    def __init__(self, *, service_name: str = "cognitionos",
                 environment: str = "", default_level: str = "INFO",
                 json_output: bool = True, sampling: Optional[SamplingConfig] = None):
        self._service_name = service_name
        self._environment = environment or os.getenv("ENVIRONMENT", "development")
        self._default_level = getattr(logging, default_level.upper(), logging.INFO)
        self._json_output = json_output
        self._sampling = sampling or SamplingConfig()
        self._module_levels: Dict[str, int] = {}
        self._configured = False

    def configure(self):
        """Configure the root logger with structured formatting."""
        if self._configured:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(self._default_level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self._default_level)

        if self._json_output:
            formatter = StructuredJsonFormatter(
                service_name=self._service_name,
                environment=self._environment,
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s — %(message)s"
            )

        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        # Apply per-module levels
        for module, level in self._module_levels.items():
            logging.getLogger(module).setLevel(level)

        self._configured = True
        logger.info("Structured logging configured (service=%s, env=%s)",
                     self._service_name, self._environment)

    def set_module_level(self, module: str, level: str):
        """Set log level for a specific module."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        self._module_levels[module] = log_level
        logging.getLogger(module).setLevel(log_level)

    def create_logger(self, name: str) -> logging.Logger:
        """Create a logger with the configured settings."""
        if not self._configured:
            self.configure()
        return logging.getLogger(name)

    def timed(self, operation: str, logger_instance: Optional[logging.Logger] = None,
              **kwargs) -> PerformanceLogger:
        """Create a performance logging context manager."""
        return PerformanceLogger(
            operation, logger_instance or logger, **kwargs
        )

    @staticmethod
    def generate_correlation_id() -> str:
        return uuid.uuid4().hex[:16]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "service_name": self._service_name,
            "environment": self._environment,
            "default_level": logging.getLevelName(self._default_level),
            "json_output": self._json_output,
            "module_overrides": {
                m: logging.getLevelName(l) for m, l in self._module_levels.items()
            },
            "configured": self._configured,
        }
