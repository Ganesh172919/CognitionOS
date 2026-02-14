"""
Structured Logging for CognitionOS V3

Provides JSON-formatted structured logging with correlation IDs.
"""

import sys
import os
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger


# Context variable for trace ID
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to log record"""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add trace ID from context
        trace_id = trace_id_var.get()
        if trace_id:
            log_record['trace_id'] = trace_id
        
        # Add service information
        log_record['service'] = 'cognitionos-v3'
        log_record['environment'] = os.getenv('ENVIRONMENT', 'development')
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add source location
        log_record['source'] = {
            'file': record.filename,
            'line': record.lineno,
            'function': record.funcName,
        }


def setup_structured_logging(
    name: str,
    level: str = "INFO",
    format_type: str = "json",
) -> logging.Logger:
    """
    Setup structured logging.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (json or text)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    if format_type == "json":
        # JSON formatter
        formatter = StructuredFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        # Text formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def set_trace_id(trace_id: str) -> None:
    """Set trace ID in context"""
    trace_id_var.set(trace_id)


def get_trace_id() -> Optional[str]:
    """Get trace ID from context"""
    return trace_id_var.get()


def clear_trace_id() -> None:
    """Clear trace ID from context"""
    trace_id_var.set(None)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context to all log messages"""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add context to log message"""
        extra = kwargs.get('extra', {})
        
        # Add trace ID if available
        trace_id = trace_id_var.get()
        if trace_id:
            extra['trace_id'] = trace_id
        
        kwargs['extra'] = extra
        return msg, kwargs


def get_logger(name: str, **context) -> LoggerAdapter:
    """
    Get a logger with context.
    
    Args:
        name: Logger name
        **context: Additional context to include in all log messages
    
    Returns:
        Logger adapter with context
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)
