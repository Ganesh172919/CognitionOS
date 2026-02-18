"""
OpenTelemetry Tracing Integration

Provides distributed tracing for CognitionOS V3 with automatic instrumentation.
"""

import os
from typing import Optional
from contextlib import contextmanager
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from core.config import get_config
from infrastructure.observability import get_logger


logger = get_logger(__name__)
config = get_config()


# Global tracer instance
_tracer: Optional[trace.Tracer] = None


def setup_tracing(service_name: Optional[str] = None) -> trace.Tracer:
    """
    Setup OpenTelemetry tracing.
    
    Args:
        service_name: Optional service name override
        
    Returns:
        Configured tracer instance
    """
    global _tracer
    
    if not config.observability.enable_tracing:
        logger.info("Tracing is disabled")
        return trace.get_tracer(__name__)
    
    # Create resource with service information
    resource = Resource.create({
        "service.name": service_name or config.service_name,
        "service.version": config.service_version,
        "deployment.environment": config.environment,
    })
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=config.observability.jaeger_host,
        agent_port=config.observability.jaeger_port,
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Get tracer instance
    _tracer = trace.get_tracer(__name__)
    
    logger.info(
        "OpenTelemetry tracing initialized",
        extra={
            "service_name": service_name or config.service_name,
            "jaeger_host": config.observability.jaeger_host,
            "jaeger_port": config.observability.jaeger_port,
        }
    )
    
    return _tracer


def instrument_fastapi(app):
    """
    Instrument FastAPI application with OpenTelemetry.
    
    Args:
        app: FastAPI application instance
    """
    if not config.observability.enable_tracing:
        return
    
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation enabled")
    except Exception as e:
        logger.error("Failed to instrument FastAPI", extra={"error": str(e)})


def instrument_sqlalchemy(engine):
    """
    Instrument SQLAlchemy engine with OpenTelemetry.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    if not config.observability.enable_tracing:
        return
    
    try:
        SQLAlchemyInstrumentor().instrument(engine=engine)
        logger.info("SQLAlchemy instrumentation enabled")
    except Exception as e:
        logger.error("Failed to instrument SQLAlchemy", extra={"error": str(e)})


def instrument_redis():
    """Instrument Redis with OpenTelemetry."""
    if not config.observability.enable_tracing:
        return
    
    try:
        RedisInstrumentor().instrument()
        logger.info("Redis instrumentation enabled")
    except Exception as e:
        logger.error("Failed to instrument Redis", extra={"error": str(e)})


def instrument_http_clients():
    """Instrument HTTP clients with OpenTelemetry."""
    if not config.observability.enable_tracing:
        return
    
    try:
        RequestsInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTP client instrumentation enabled")
    except Exception as e:
        logger.error("Failed to instrument HTTP clients", extra={"error": str(e)})


def get_tracer() -> trace.Tracer:
    """Get or create tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = setup_tracing()
    return _tracer


@contextmanager
def trace_operation(name: str, attributes: Optional[dict] = None):
    """
    Context manager for tracing an operation.
    
    Args:
        name: Operation name
        attributes: Optional attributes to add to span
        
    Example:
        with trace_operation("create_workflow", {"workflow_id": "wf-123"}):
            # ... operation code ...
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            raise


def add_span_attributes(**attributes):
    """
    Add attributes to the current span.
    
    Args:
        **attributes: Key-value pairs to add as attributes
    """
    current_span = trace.get_current_span()
    if current_span:
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[dict] = None):
    """
    Add an event to the current span.
    
    Args:
        name: Event name
        attributes: Optional event attributes
    """
    current_span = trace.get_current_span()
    if current_span:
        current_span.add_event(name, attributes or {})
