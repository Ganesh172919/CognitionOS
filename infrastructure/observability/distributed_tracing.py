"""
OpenTelemetry Distributed Tracing Configuration

Provides comprehensive distributed tracing across all services.
"""

from typing import Optional
import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

logger = logging.getLogger(__name__)


class DistributedTracing:
    """Manages OpenTelemetry distributed tracing configuration."""
    
    def __init__(
        self,
        service_name: str = "cognitionos-api",
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
        enabled: bool = True,
    ):
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.enabled = enabled
        self._tracer_provider: Optional[TracerProvider] = None
    
    def setup(self) -> TracerProvider:
        """
        Set up OpenTelemetry tracing with Jaeger exporter.
        
        Returns:
            TracerProvider: Configured tracer provider
        """
        if not self.enabled:
            logger.info("Distributed tracing is disabled")
            return None
        
        # Create resource with service name
        resource = Resource(attributes={
            SERVICE_NAME: self.service_name
        })
        
        # Create tracer provider
        self._tracer_provider = TracerProvider(resource=resource)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.jaeger_host,
            agent_port=self.jaeger_port,
        )
        
        # Add batch span processor for better performance
        span_processor = BatchSpanProcessor(jaeger_exporter)
        self._tracer_provider.add_span_processor(span_processor)
        
        # Set as global tracer provider
        trace.set_tracer_provider(self._tracer_provider)
        
        logger.info(
            f"Distributed tracing initialized for {self.service_name} "
            f"(Jaeger: {self.jaeger_host}:{self.jaeger_port})"
        )
        
        return self._tracer_provider
    
    def instrument_fastapi(self, app):
        """
        Instrument FastAPI application for automatic tracing.
        
        Args:
            app: FastAPI application instance
        """
        if not self.enabled:
            return
        
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented for distributed tracing")
    
    def instrument_sqlalchemy(self, engine):
        """
        Instrument SQLAlchemy engine for database query tracing.
        
        Args:
            engine: SQLAlchemy engine
        """
        if not self.enabled:
            return
        
        SQLAlchemyInstrumentor().instrument(
            engine=engine,
            service=f"{self.service_name}-db"
        )
        logger.info("SQLAlchemy instrumented for distributed tracing")
    
    def instrument_redis(self):
        """Instrument Redis client for cache operation tracing."""
        if not self.enabled:
            return
        
        RedisInstrumentor().instrument()
        logger.info("Redis instrumented for distributed tracing")
    
    def instrument_httpx(self):
        """Instrument HTTPX client for HTTP request tracing."""
        if not self.enabled:
            return
        
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX instrumented for distributed tracing")
    
    def get_tracer(self, name: str = __name__):
        """
        Get a tracer instance for manual instrumentation.
        
        Args:
            name: Name for the tracer
            
        Returns:
            Tracer instance
        """
        return trace.get_tracer(name)
    
    def shutdown(self):
        """Shutdown tracer provider and flush remaining spans."""
        if self._tracer_provider:
            self._tracer_provider.shutdown()
            logger.info("Distributed tracing shutdown complete")


# Global tracing instance
_tracing: Optional[DistributedTracing] = None


def setup_distributed_tracing(
    service_name: str = "cognitionos-api",
    jaeger_host: str = "localhost",
    jaeger_port: int = 6831,
    enabled: bool = True,
) -> DistributedTracing:
    """
    Setup and configure distributed tracing.
    
    Args:
        service_name: Name of the service
        jaeger_host: Jaeger agent hostname
        jaeger_port: Jaeger agent port
        enabled: Whether tracing is enabled
        
    Returns:
        DistributedTracing instance
    """
    global _tracing
    
    _tracing = DistributedTracing(
        service_name=service_name,
        jaeger_host=jaeger_host,
        jaeger_port=jaeger_port,
        enabled=enabled,
    )
    
    _tracing.setup()
    return _tracing


def get_tracing() -> Optional[DistributedTracing]:
    """Get the global tracing instance."""
    return _tracing


def instrument_all(app, engine=None):
    """
    Instrument all supported libraries for automatic tracing.
    
    Args:
        app: FastAPI application
        engine: SQLAlchemy engine (optional)
    """
    if _tracing is None:
        logger.warning("Tracing not initialized, call setup_distributed_tracing() first")
        return
    
    _tracing.instrument_fastapi(app)
    _tracing.instrument_httpx()
    _tracing.instrument_redis()
    
    if engine:
        _tracing.instrument_sqlalchemy(engine)
