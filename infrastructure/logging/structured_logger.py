"""
Structured Logging Framework — CognitionOS

Features:
- JSON-structured log entries
- Correlation / request ID propagation
- Log level management per module
- Sampling for high-volume endpoints
- Buffered async sinks (stdout, file, remote)
- Sensitive field masking
- Performance metrics integration
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import threading
import time
import traceback
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, TextIO

# ---------------------------------------------------------------------------
# Context vars for request-scoped data
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_tenant_id: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


def get_correlation_id() -> Optional[str]:
    return _correlation_id.get()


def set_request_context(*, request_id: str = "", tenant_id: str = "", user_id: str = "") -> None:
    if request_id:
        _request_id.set(request_id)
    if tenant_id:
        _tenant_id.set(tenant_id)
    if user_id:
        _user_id.set(user_id)


# ---------------------------------------------------------------------------
# Log level enum
# ---------------------------------------------------------------------------


class LogLevel(IntEnum):
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# ---------------------------------------------------------------------------
# Structured log entry
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    timestamp: str
    level: str
    logger_name: str
    message: str
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    line_no: Optional[int] = None
    exception: Optional[str] = None
    duration_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timestamp": self.timestamp,
            "level": self.level,
            "logger": self.logger_name,
            "message": self.message,
        }
        if self.correlation_id:
            d["correlation_id"] = self.correlation_id
        if self.request_id:
            d["request_id"] = self.request_id
        if self.tenant_id:
            d["tenant_id"] = self.tenant_id
        if self.user_id:
            d["user_id"] = self.user_id
        if self.module:
            d["module"] = self.module
        if self.function:
            d["function"] = self.function
        if self.line_no:
            d["line_no"] = self.line_no
        if self.exception:
            d["exception"] = self.exception
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        if self.extra:
            d["extra"] = self.extra
        if self.tags:
            d["tags"] = self.tags
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Sensitive field masking
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS = {
    "password", "secret", "token", "api_key", "authorization",
    "credit_card", "ssn", "private_key", "credentials", "access_key",
}


def mask_sensitive(data: Dict[str, Any]) -> Dict[str, Any]:
    masked = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(pattern in key_lower for pattern in _SENSITIVE_PATTERNS):
            masked[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked[key] = mask_sensitive(value)
        else:
            masked[key] = value
    return masked


# ---------------------------------------------------------------------------
# Log sinks
# ---------------------------------------------------------------------------


class LogSink:
    """Base class for log sinks."""

    def write(self, entry: LogEntry) -> None:
        raise NotImplementedError

    async def flush(self) -> None:
        pass

    async def close(self) -> None:
        pass


class ConsoleSink(LogSink):
    """Writes JSON logs to stdout/stderr."""

    def __init__(self, stream: TextIO = sys.stdout, *, colorize: bool = True) -> None:
        self._stream = stream
        self._colorize = colorize
        self._colors = {
            "TRACE": "\033[90m",
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
            "CRITICAL": "\033[35m",
        }
        self._reset = "\033[0m"

    def write(self, entry: LogEntry) -> None:
        output = entry.to_json()
        if self._colorize and self._stream.isatty():
            color = self._colors.get(entry.level, "")
            output = f"{color}{output}{self._reset}"
        self._stream.write(output + "\n")
        self._stream.flush()


class FileSink(LogSink):
    """Writes JSON logs to a rotating file."""

    def __init__(
        self,
        path: str,
        *,
        max_size_mb: int = 100,
        max_files: int = 10,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_size_mb * 1024 * 1024
        self._max_files = max_files
        self._current_size = 0
        self._lock = threading.Lock()
        self._file: Optional[TextIO] = None
        self._open_file()

    def _open_file(self) -> None:
        if self._file:
            self._file.close()
        self._file = self._path.open("a", encoding="utf-8")
        self._current_size = self._path.stat().st_size if self._path.exists() else 0

    def write(self, entry: LogEntry) -> None:
        line = entry.to_json() + "\n"
        with self._lock:
            if self._current_size + len(line) > self._max_bytes:
                self._rotate()
            if self._file:
                self._file.write(line)
                self._file.flush()
                self._current_size += len(line)

    def _rotate(self) -> None:
        if self._file:
            self._file.close()
        for i in range(self._max_files - 1, 0, -1):
            src = self._path.with_suffix(f".{i}.log")
            dst = self._path.with_suffix(f".{i + 1}.log")
            if src.exists():
                src.rename(dst)
        if self._path.exists():
            self._path.rename(self._path.with_suffix(".1.log"))
        self._open_file()

    async def close(self) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class BufferedSink(LogSink):
    """Buffers log entries and flushes in batches."""

    def __init__(self, inner: LogSink, *, buffer_size: int = 100, flush_interval_seconds: float = 5.0) -> None:
        self._inner = inner
        self._buffer: Deque[LogEntry] = deque(maxlen=buffer_size * 2)
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def write(self, entry: LogEntry) -> None:
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._buffer_size:
                self._flush_sync()

    def _flush_sync(self) -> None:
        while self._buffer:
            e = self._buffer.popleft()
            self._inner.write(e)
        self._last_flush = time.monotonic()

    async def flush(self) -> None:
        with self._lock:
            self._flush_sync()

    async def close(self) -> None:
        await self.flush()
        await self._inner.close()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass
class SamplingRule:
    """Sample a percentage of logs matching a pattern."""

    logger_pattern: str
    level: LogLevel
    sample_rate: float  # 0.0 to 1.0

    def should_sample(self, logger_name: str, level: int) -> bool:
        if level < self.level:
            return False
        if self.logger_pattern != "*" and self.logger_pattern not in logger_name:
            return False
        return random.random() < self.sample_rate


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------


class StructuredLogger:
    """Main structured logging interface."""

    def __init__(
        self,
        name: str,
        *,
        level: LogLevel = LogLevel.INFO,
        sinks: Optional[List[LogSink]] = None,
        sampling_rules: Optional[List[SamplingRule]] = None,
        default_tags: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.level = level
        self._sinks = sinks or [ConsoleSink()]
        self._sampling_rules = sampling_rules or []
        self._default_tags = default_tags or []

    def _should_log(self, level: LogLevel) -> bool:
        if level < self.level:
            return False
        for rule in self._sampling_rules:
            if not rule.should_sample(self.name, level):
                return False
        return True

    def _build_entry(
        self,
        level: LogLevel,
        message: str,
        *,
        exc_info: Optional[Exception] = None,
        duration_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> LogEntry:
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.name,
            logger_name=self.name,
            message=message,
            correlation_id=_correlation_id.get(),
            request_id=_request_id.get(),
            tenant_id=_tenant_id.get(),
            user_id=_user_id.get(),
            duration_ms=duration_ms,
            extra=mask_sensitive(extra) if extra else {},
            tags=list(set(self._default_tags + (tags or []))),
        )
        if exc_info:
            entry.exception = "".join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
        return entry

    def _emit(self, entry: LogEntry) -> None:
        for sink in self._sinks:
            try:
                sink.write(entry)
            except Exception:
                pass  # Logging must never crash the app

    def log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        if not self._should_log(level):
            return
        entry = self._build_entry(level, message, **kwargs)
        self._emit(entry)

    def trace(self, message: str, **kwargs: Any) -> None:
        self.log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self.log(LogLevel.CRITICAL, message, **kwargs)

    def exception(self, message: str, exc: Exception, **kwargs: Any) -> None:
        self.log(LogLevel.ERROR, message, exc_info=exc, **kwargs)

    def timed(self, message: str) -> "_TimedContext":
        return _TimedContext(self, message)

    def child(self, name: str) -> "StructuredLogger":
        return StructuredLogger(
            f"{self.name}.{name}",
            level=self.level,
            sinks=self._sinks,
            sampling_rules=self._sampling_rules,
            default_tags=self._default_tags,
        )


class _TimedContext:
    """Context manager for timing operations."""

    def __init__(self, logger: StructuredLogger, message: str) -> None:
        self._logger = logger
        self._message = message
        self._start: float = 0

    def __enter__(self) -> "_TimedContext":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = (time.monotonic() - self._start) * 1000
        self._logger.info(self._message, duration_ms=round(elapsed, 2))


# ---------------------------------------------------------------------------
# Global logger factory
# ---------------------------------------------------------------------------

_root_sinks: List[LogSink] = []
_root_level = LogLevel.INFO


def configure_logging(
    *,
    level: LogLevel = LogLevel.INFO,
    sinks: Optional[List[LogSink]] = None,
    log_file: Optional[str] = None,
    json_stdout: bool = True,
) -> None:
    global _root_sinks, _root_level
    _root_level = level
    _root_sinks = sinks or []
    if json_stdout:
        _root_sinks.append(ConsoleSink())
    if log_file:
        _root_sinks.append(FileSink(log_file))


def get_logger(name: str, **kwargs: Any) -> StructuredLogger:
    return StructuredLogger(name, level=_root_level, sinks=_root_sinks or [ConsoleSink()], **kwargs)
