"""
Prometheus Metrics Integration

Provides metrics collection and exposure for CognitionOS V3.
"""

import os
from typing import Optional
import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    REGISTRY,
    CollectorRegistry,
)
from prometheus_client.exposition import make_asgi_app

from core.config import get_config
from infrastructure.observability import get_logger


logger = get_logger(__name__)
config = get_config()


# ==================== HTTP Metrics ====================

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'HTTP requests currently in progress',
    ['method', 'endpoint']
)


# ==================== Database Metrics ====================

db_queries_total = Counter(
    'db_queries_total',
    'Total database queries',
    ['operation', 'table']
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

db_connections_active = Gauge(
    'db_connections_active',
    'Active database connections'
)

db_connections_idle = Gauge(
    'db_connections_idle',
    'Idle database connections'
)


# ==================== Workflow Metrics ====================

workflows_created_total = Counter(
    'workflows_created_total',
    'Total workflows created',
    ['workflow_id']
)

workflow_executions_total = Counter(
    'workflow_executions_total',
    'Total workflow executions',
    ['workflow_id', 'status']
)

workflow_execution_duration_seconds = Histogram(
    'workflow_execution_duration_seconds',
    'Workflow execution duration in seconds',
    ['workflow_id'],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0)
)

workflows_in_progress = Gauge(
    'workflows_in_progress',
    'Workflows currently executing',
    ['workflow_id']
)

workflow_steps_executed_total = Counter(
    'workflow_steps_executed_total',
    'Total workflow steps executed',
    ['workflow_id', 'step_id', 'status']
)


# ==================== LLM Metrics ====================

llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['provider', 'model', 'status']
)

llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['provider', 'model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

llm_tokens_used_total = Counter(
    'llm_tokens_used_total',
    'Total LLM tokens used',
    ['provider', 'model', 'token_type']
)

llm_cost_usd_total = Counter(
    'llm_cost_usd_total',
    'Total LLM cost in USD',
    ['provider', 'model']
)


# ==================== Event Bus Metrics ====================

events_published_total = Counter(
    'events_published_total',
    'Total events published',
    ['event_type']
)

events_consumed_total = Counter(
    'events_consumed_total',
    'Total events consumed',
    ['event_type', 'handler']
)

events_failed_total = Counter(
    'events_failed_total',
    'Total event processing failures',
    ['event_type', 'handler']
)

event_processing_duration_seconds = Histogram(
    'event_processing_duration_seconds',
    'Event processing duration in seconds',
    ['event_type', 'handler'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0)
)


# ==================== Authentication Metrics ====================

auth_requests_total = Counter(
    'auth_requests_total',
    'Total authentication requests',
    ['endpoint', 'status']
)

auth_tokens_created_total = Counter(
    'auth_tokens_created_total',
    'Total authentication tokens created',
    ['token_type']
)

auth_failures_total = Counter(
    'auth_failures_total',
    'Total authentication failures',
    ['reason']
)


# ==================== System Metrics ====================

service_info = Info(
    'cognitionos_service',
    'CognitionOS service information'
)

# Set service info
service_info.info({
    'version': config.service_version,
    'environment': config.environment,
    'service_name': config.service_name,
})


# ==================== Middleware ====================

class PrometheusMiddleware:
    """
    Middleware to collect HTTP metrics.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        method = scope["method"]
        path = scope["path"]
        
        # Skip metrics endpoint
        if path == "/metrics":
            await self.app(scope, receive, send)
            return
        
        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=path).inc()
        
        start_time = time.time()
        status_code = 500
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            http_requests_in_progress.labels(
                method=method,
                endpoint=path
            ).dec()


# ==================== Metrics Endpoint ====================

def create_metrics_app():
    """Create ASGI app for metrics endpoint."""
    if not config.observability.enable_metrics:
        # Return dummy app if metrics disabled
        async def dummy_app(scope, receive, send):
            await send({
                'type': 'http.response.start',
                'status': 503,
                'headers': [[b'content-type', b'text/plain']],
            })
            await send({
                'type': 'http.response.body',
                'body': b'Metrics disabled',
            })
        return dummy_app
    
    return make_asgi_app()


# ==================== Helper Functions ====================

def track_workflow_execution(workflow_id: str, duration: float, status: str):
    """Track workflow execution metrics."""
    workflow_executions_total.labels(
        workflow_id=workflow_id,
        status=status
    ).inc()
    
    workflow_execution_duration_seconds.labels(
        workflow_id=workflow_id
    ).observe(duration)


def track_llm_request(provider: str, model: str, duration: float, 
                     tokens_used: dict, cost_usd: float, status: str):
    """Track LLM request metrics."""
    llm_requests_total.labels(
        provider=provider,
        model=model,
        status=status
    ).inc()
    
    llm_request_duration_seconds.labels(
        provider=provider,
        model=model
    ).observe(duration)
    
    if tokens_used:
        for token_type, count in tokens_used.items():
            llm_tokens_used_total.labels(
                provider=provider,
                model=model,
                token_type=token_type
            ).inc(count)
    
    if cost_usd:
        llm_cost_usd_total.labels(
            provider=provider,
            model=model
        ).inc(cost_usd)


def track_event_published(event_type: str):
    """Track event publication."""
    events_published_total.labels(event_type=event_type).inc()


def track_event_consumed(event_type: str, handler: str, duration: float, success: bool):
    """Track event consumption."""
    events_consumed_total.labels(
        event_type=event_type,
        handler=handler
    ).inc()
    
    event_processing_duration_seconds.labels(
        event_type=event_type,
        handler=handler
    ).observe(duration)
    
    if not success:
        events_failed_total.labels(
            event_type=event_type,
            handler=handler
        ).inc()
