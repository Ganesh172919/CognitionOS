"""
Observability Service.

Centralized monitoring, metrics collection, tracing, and alerting for CognitionOS.
Provides real-time insights into system health, performance, and failures.
"""

import sys
import os

# Add shared libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
from enum import Enum
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Float, Integer, JSON, Text, Index, Boolean
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from shared.libs.config import BaseConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import ErrorResponse
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)


# Configuration
class ObservabilityConfig(BaseConfig):
    """Configuration for Observability service."""

    service_name: str = "observability"
    port: int = Field(default=8009, env="PORT")

    # Metrics
    metrics_retention_hours: int = Field(default=168, env="METRICS_RETENTION_HOURS")  # 7 days
    metrics_aggregation_interval_seconds: int = Field(default=60, env="METRICS_AGGREGATION_INTERVAL")

    # Alerting
    enable_alerting: bool = Field(default=True, env="ENABLE_ALERTING")
    error_rate_threshold: float = Field(default=0.1, env="ERROR_RATE_THRESHOLD")  # 10%
    latency_threshold_ms: int = Field(default=5000, env="LATENCY_THRESHOLD_MS")


config = load_config(ObservabilityConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Observability",
    version=config.service_version,
    description="Monitoring, metrics, and alerting service"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)


# ============================================================================
# Database Models
# ============================================================================

Base = declarative_base()


class MetricDataPoint(Base):
    """
    Time-series metric data point.
    """
    __tablename__ = "metrics"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Metric identification
    metric_name = Column(String(200), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram, summary
    service_name = Column(String(100), nullable=False, index=True)

    # Labels/Tags for filtering
    labels = Column(JSON, nullable=True)

    # Value
    value = Column(Float, nullable=False)
    count = Column(Integer, default=1)  # For aggregated metrics

    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Context
    trace_id = Column(String(100), nullable=True, index=True)
    span_id = Column(String(100), nullable=True)

    __table_args__ = (
        Index('idx_metrics_name_time', 'metric_name', 'timestamp'),
        Index('idx_metrics_service_time', 'service_name', 'timestamp'),
    )


class TraceSpan(Base):
    """
    Distributed tracing span.
    """
    __tablename__ = "traces"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Trace context
    trace_id = Column(String(100), nullable=False, index=True)
    span_id = Column(String(100), nullable=False, unique=True, index=True)
    parent_span_id = Column(String(100), nullable=True, index=True)

    # Service
    service_name = Column(String(100), nullable=False, index=True)
    operation_name = Column(String(200), nullable=False)

    # Timing
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Status
    status_code = Column(String(50), nullable=False)  # ok, error
    error = Column(Boolean, default=False, index=True)
    error_message = Column(Text, nullable=True)

    # Tags and logs
    tags = Column(JSON, nullable=True)
    logs = Column(JSON, nullable=True)

    # Metadata
    user_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    task_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)

    __table_args__ = (
        Index('idx_traces_trace_id_time', 'trace_id', 'start_time'),
        Index('idx_traces_error', 'error', 'start_time'),
    )


class Alert(Base):
    """
    System alert.
    """
    __tablename__ = "alerts"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Alert details
    alert_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(50), nullable=False, index=True)  # info, warning, critical
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)

    # Context
    service_name = Column(String(100), nullable=True)
    metric_name = Column(String(200), nullable=True)
    threshold_value = Column(Float, nullable=True)
    current_value = Column(Float, nullable=True)

    # Status
    status = Column(String(50), nullable=False, default="active")  # active, acknowledged, resolved
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Timing
    triggered_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Metadata
    metadata = Column(JSON, nullable=True)

    __table_args__ = (
        Index('idx_alerts_status_time', 'status', 'triggered_at'),
        Index('idx_alerts_severity', 'severity', 'triggered_at'),
    )


# ============================================================================
# Enums and Models
# ============================================================================

class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Can go up or down
    HISTOGRAM = "histogram"  # Distribution
    SUMMARY = "summary"  # Quantiles


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RecordMetricRequest(BaseModel):
    """Request to record a metric."""
    metric_name: str
    metric_type: MetricType
    service_name: str
    value: float
    labels: Optional[Dict[str, str]] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class RecordTraceRequest(BaseModel):
    """Request to record a trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    service_name: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status_code: str = "ok"
    error: bool = False
    error_message: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    logs: Optional[List[Dict[str, Any]]] = None
    user_id: Optional[UUID] = None
    task_id: Optional[UUID] = None


class MetricQueryRequest(BaseModel):
    """Request to query metrics."""
    metric_name: str
    service_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    labels: Optional[Dict[str, str]] = None
    aggregation: Optional[str] = "avg"  # avg, sum, min, max, count


class DashboardData(BaseModel):
    """Dashboard data model."""
    service_metrics: Dict[str, Any]
    error_rates: Dict[str, float]
    latency_percentiles: Dict[str, Dict[str, float]]
    active_alerts: List[Dict[str, Any]]
    recent_failures: List[Dict[str, Any]]
    system_health: str


# ============================================================================
# Database
# ============================================================================

engine = create_async_engine(
    config.database_url,
    pool_size=config.database_pool_size,
    max_overflow=config.database_max_overflow,
    echo=config.debug
)

async_session_maker = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncSession:
    """Get database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


# ============================================================================
# Observability Engine
# ============================================================================

class ObservabilityEngine:
    """
    Manages metrics collection, tracing, and alerting.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="ObservabilityEngine")

        # In-memory metric buffers for real-time aggregation
        self.metric_buffer = defaultdict(lambda: deque(maxlen=1000))

        # Alert state tracking
        self.alert_states = {}

    async def record_metric(
        self,
        request: RecordMetricRequest,
        db: AsyncSession
    ):
        """Record a metric data point."""
        # Create metric entry
        metric = MetricDataPoint(
            metric_name=request.metric_name,
            metric_type=request.metric_type.value,
            service_name=request.service_name,
            labels=request.labels,
            value=request.value,
            timestamp=request.timestamp or datetime.utcnow(),
            trace_id=request.trace_id,
            span_id=request.span_id
        )

        db.add(metric)
        await db.commit()

        # Add to in-memory buffer for real-time analysis
        buffer_key = f"{request.service_name}:{request.metric_name}"
        self.metric_buffer[buffer_key].append({
            "value": request.value,
            "timestamp": metric.timestamp
        })

        # Check for alert conditions
        if config.enable_alerting:
            await self._check_alert_conditions(request, db)

    async def _check_alert_conditions(
        self,
        metric_request: RecordMetricRequest,
        db: AsyncSession
    ):
        """Check if metric violates alert thresholds."""
        # Example: Check error rate
        if "error" in metric_request.metric_name.lower():
            buffer_key = f"{metric_request.service_name}:{metric_request.metric_name}"
            recent_values = list(self.metric_buffer[buffer_key])

            if len(recent_values) >= 10:
                error_count = sum(1 for v in recent_values[-10:] if v["value"] > 0)
                error_rate = error_count / 10

                if error_rate > config.error_rate_threshold:
                    # Create alert
                    await self._create_alert(
                        alert_type="high_error_rate",
                        severity=AlertSeverity.CRITICAL,
                        title=f"High error rate in {metric_request.service_name}",
                        description=f"Error rate {error_rate:.1%} exceeds threshold {config.error_rate_threshold:.1%}",
                        service_name=metric_request.service_name,
                        metric_name=metric_request.metric_name,
                        threshold_value=config.error_rate_threshold,
                        current_value=error_rate,
                        db=db
                    )

    async def _create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        description: str,
        service_name: Optional[str],
        metric_name: Optional[str],
        threshold_value: Optional[float],
        current_value: Optional[float],
        db: AsyncSession
    ):
        """Create an alert."""
        # Check if similar alert already exists
        alert_key = f"{alert_type}:{service_name}:{metric_name}"

        if alert_key in self.alert_states:
            last_alert_time = self.alert_states[alert_key]
            # Don't create duplicate alerts within 5 minutes
            if (datetime.utcnow() - last_alert_time).total_seconds() < 300:
                return

        alert = Alert(
            alert_type=alert_type,
            severity=severity.value,
            title=title,
            description=description,
            service_name=service_name,
            metric_name=metric_name,
            threshold_value=threshold_value,
            current_value=current_value,
            status="active"
        )

        db.add(alert)
        await db.commit()

        self.alert_states[alert_key] = datetime.utcnow()

        self.logger.warning(
            f"Alert triggered: {title}",
            extra={
                "alert_type": alert_type,
                "severity": severity.value,
                "service": service_name
            }
        )

    async def get_dashboard_data(self, db: AsyncSession) -> DashboardData:
        """Generate dashboard data."""
        from sqlalchemy import select, func, and_

        # Calculate time ranges
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)

        # Get service metrics (request counts)
        service_query = select(
            MetricDataPoint.service_name,
            func.count(MetricDataPoint.id).label('count'),
            func.avg(MetricDataPoint.value).label('avg_value')
        ).where(
            MetricDataPoint.timestamp >= one_hour_ago
        ).group_by(MetricDataPoint.service_name)

        result = await db.execute(service_query)
        service_metrics = {
            row.service_name: {
                "count": row.count,
                "avg_value": float(row.avg_value or 0)
            }
            for row in result
        }

        # Calculate error rates
        error_rates = {}
        for service in service_metrics.keys():
            # Get error count
            error_query = select(func.count(TraceSpan.id)).where(
                and_(
                    TraceSpan.service_name == service,
                    TraceSpan.error == True,
                    TraceSpan.start_time >= one_hour_ago
                )
            )
            error_result = await db.execute(error_query)
            error_count = error_result.scalar() or 0

            # Get total count
            total_query = select(func.count(TraceSpan.id)).where(
                and_(
                    TraceSpan.service_name == service,
                    TraceSpan.start_time >= one_hour_ago
                )
            )
            total_result = await db.execute(total_query)
            total_count = total_result.scalar() or 0

            error_rates[service] = error_count / total_count if total_count > 0 else 0.0

        # Calculate latency percentiles
        latency_percentiles = {}
        for service in service_metrics.keys():
            # Get latencies
            latency_query = select(TraceSpan.duration_ms).where(
                and_(
                    TraceSpan.service_name == service,
                    TraceSpan.start_time >= one_hour_ago,
                    TraceSpan.duration_ms.isnot(None)
                )
            ).order_by(TraceSpan.duration_ms)

            latency_result = await db.execute(latency_query)
            latencies = [row[0] for row in latency_result]

            if latencies:
                latency_percentiles[service] = {
                    "p50": latencies[int(len(latencies) * 0.5)],
                    "p95": latencies[int(len(latencies) * 0.95)],
                    "p99": latencies[int(len(latencies) * 0.99)] if len(latencies) > 100 else latencies[-1]
                }

        # Get active alerts
        active_alerts_query = select(Alert).where(
            Alert.status == "active"
        ).order_by(Alert.triggered_at.desc()).limit(10)

        alerts_result = await db.execute(active_alerts_query)
        active_alerts = [
            {
                "id": str(alert.id),
                "severity": alert.severity,
                "title": alert.title,
                "service": alert.service_name,
                "triggered_at": alert.triggered_at.isoformat()
            }
            for alert in alerts_result.scalars().all()
        ]

        # Get recent failures
        failures_query = select(TraceSpan).where(
            and_(
                TraceSpan.error == True,
                TraceSpan.start_time >= one_day_ago
            )
        ).order_by(TraceSpan.start_time.desc()).limit(20)

        failures_result = await db.execute(failures_query)
        recent_failures = [
            {
                "trace_id": span.trace_id,
                "service": span.service_name,
                "operation": span.operation_name,
                "error": span.error_message,
                "timestamp": span.start_time.isoformat()
            }
            for span in failures_result.scalars().all()
        ]

        # Determine overall system health
        avg_error_rate = sum(error_rates.values()) / len(error_rates) if error_rates else 0
        critical_alerts = len([a for a in active_alerts if a["severity"] == "critical"])

        if critical_alerts > 0 or avg_error_rate > 0.1:
            system_health = "degraded"
        elif avg_error_rate > 0.05 or len(active_alerts) > 5:
            system_health = "warning"
        else:
            system_health = "healthy"

        return DashboardData(
            service_metrics=service_metrics,
            error_rates=error_rates,
            latency_percentiles=latency_percentiles,
            active_alerts=active_alerts,
            recent_failures=recent_failures,
            system_health=system_health
        )


# Initialize engine
observability_engine = ObservabilityEngine()


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/metrics/record")
async def record_metric(
    request: RecordMetricRequest,
    db: AsyncSession = Depends(get_db)
):
    """Record a metric data point."""
    await observability_engine.record_metric(request, db)
    return {"recorded": True}


@app.post("/traces/record")
async def record_trace(
    request: RecordTraceRequest,
    db: AsyncSession = Depends(get_db)
):
    """Record a distributed trace span."""
    span = TraceSpan(
        trace_id=request.trace_id,
        span_id=request.span_id,
        parent_span_id=request.parent_span_id,
        service_name=request.service_name,
        operation_name=request.operation_name,
        start_time=request.start_time,
        end_time=request.end_time,
        duration_ms=request.duration_ms,
        status_code=request.status_code,
        error=request.error,
        error_message=request.error_message,
        tags=request.tags,
        logs=request.logs,
        user_id=request.user_id,
        task_id=request.task_id
    )

    db.add(span)
    await db.commit()

    return {"recorded": True, "span_id": span.span_id}


@app.post("/metrics/query")
async def query_metrics(
    request: MetricQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Query metrics with filters and aggregation."""
    from sqlalchemy import select, func, and_

    # Build query
    filters = [MetricDataPoint.metric_name == request.metric_name]

    if request.service_name:
        filters.append(MetricDataPoint.service_name == request.service_name)

    if request.start_time:
        filters.append(MetricDataPoint.timestamp >= request.start_time)

    if request.end_time:
        filters.append(MetricDataPoint.timestamp <= request.end_time)

    # Select aggregation function
    if request.aggregation == "sum":
        agg_func = func.sum(MetricDataPoint.value)
    elif request.aggregation == "min":
        agg_func = func.min(MetricDataPoint.value)
    elif request.aggregation == "max":
        agg_func = func.max(MetricDataPoint.value)
    elif request.aggregation == "count":
        agg_func = func.count(MetricDataPoint.id)
    else:  # avg
        agg_func = func.avg(MetricDataPoint.value)

    query = select(agg_func).where(and_(*filters))

    result = await db.execute(query)
    value = result.scalar()

    return {
        "metric_name": request.metric_name,
        "aggregation": request.aggregation,
        "value": float(value or 0)
    }


@app.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all spans for a trace."""
    from sqlalchemy import select

    query = select(TraceSpan).where(
        TraceSpan.trace_id == trace_id
    ).order_by(TraceSpan.start_time)

    result = await db.execute(query)
    spans = result.scalars().all()

    return {
        "trace_id": trace_id,
        "span_count": len(spans),
        "spans": [
            {
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "service": span.service_name,
                "operation": span.operation_name,
                "start_time": span.start_time.isoformat(),
                "duration_ms": span.duration_ms,
                "status": span.status_code,
                "error": span.error,
                "tags": span.tags
            }
            for span in spans
        ]
    }


@app.get("/dashboard", response_model=DashboardData)
async def get_dashboard(
    db: AsyncSession = Depends(get_db)
):
    """
    Get dashboard data with system health metrics.

    Provides real-time view of:
    - Service health and request rates
    - Error rates per service
    - Latency percentiles
    - Active alerts
    - Recent failures
    """
    return await observability_engine.get_dashboard_data(db)


@app.get("/alerts")
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by status"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, le=500),
    db: AsyncSession = Depends(get_db)
):
    """Get alerts with optional filters."""
    from sqlalchemy import select, and_

    filters = []
    if status:
        filters.append(Alert.status == status)
    if severity:
        filters.append(Alert.severity == severity)

    query = select(Alert)
    if filters:
        query = query.where(and_(*filters))

    query = query.order_by(Alert.triggered_at.desc()).limit(limit)

    result = await db.execute(query)
    alerts = result.scalars().all()

    return {
        "count": len(alerts),
        "alerts": [
            {
                "id": str(alert.id),
                "type": alert.alert_type,
                "severity": alert.severity,
                "title": alert.title,
                "description": alert.description,
                "service": alert.service_name,
                "status": alert.status,
                "triggered_at": alert.triggered_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in alerts
        ]
    }


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Acknowledge an alert."""
    from sqlalchemy import select

    query = select(Alert).where(Alert.id == alert_id)
    result = await db.execute(query)
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )

    alert.status = "acknowledged"
    alert.acknowledged_at = datetime.utcnow()

    await db.commit()

    return {"acknowledged": True}


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Resolve an alert."""
    from sqlalchemy import select

    query = select(Alert).where(Alert.id == alert_id)
    result = await db.execute(query)
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )

    alert.status = "resolved"
    alert.resolved_at = datetime.utcnow()

    await db.commit()

    return {"resolved": True}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "observability",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat(),
        "alerting_enabled": config.enable_alerting
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    logger.info(
        "Observability service starting",
        extra={
            "version": config.service_version,
            "alerting_enabled": config.enable_alerting
        }
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Observability service ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Observability service shutting down")
    await engine.dispose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )
