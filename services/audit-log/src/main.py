"""
Audit Log Service.

Comprehensive audit logging for all tool executions, API calls, and security events.
Provides tamper-evident logging with integrity checks and forensic analysis.
"""

import os

# Add shared libs to path

import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
from enum import Enum

from fastapi import FastAPI, HTTPException, status, Query, Depends
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Integer, JSON, Text, Index, Boolean
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
class AuditLogConfig(BaseConfig):
    """Configuration for Audit Log service."""

    service_name: str = "audit-log"
    port: int = Field(default=8007, env="PORT")

    # Audit settings
    retention_days: int = Field(default=365, env="AUDIT_RETENTION_DAYS")
    enable_integrity_checks: bool = Field(default=True, env="ENABLE_INTEGRITY_CHECKS")
    compression_enabled: bool = Field(default=True, env="COMPRESSION_ENABLED")

    # Alert thresholds
    alert_on_critical_events: bool = Field(default=True, env="ALERT_ON_CRITICAL_EVENTS")
    suspicious_pattern_threshold: int = Field(default=10, env="SUSPICIOUS_PATTERN_THRESHOLD")


config = load_config(AuditLogConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Audit Log",
    version=config.service_version,
    description="Tamper-evident audit logging and forensics"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)


# ============================================================================
# Database Models
# ============================================================================

Base = declarative_base()


class AuditLogEntry(Base):
    """
    Tamper-evident audit log entry.

    Uses chain hashing for integrity verification.
    """
    __tablename__ = "audit_logs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    sequence_number = Column(Integer, nullable=False, unique=True, index=True)

    # Event details
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)

    # Context
    user_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    agent_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    task_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)

    # Actor information
    actor_type = Column(String(50), nullable=False)  # user, agent, system
    actor_id = Column(String(200), nullable=False)
    source_ip = Column(String(45), nullable=True)  # IPv6 compatible

    # Event data
    action = Column(String(200), nullable=False)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(200), nullable=True)
    outcome = Column(String(50), nullable=False)  # success, failure, error

    # Details
    details = Column(JSON, nullable=True)
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # Security
    permission_required = Column(String(200), nullable=True)
    permission_granted = Column(Boolean, default=False)

    # Integrity
    previous_hash = Column(String(64), nullable=True)
    entry_hash = Column(String(64), nullable=False)

    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    duration_ms = Column(Integer, nullable=True)

    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_audit_user_time', 'user_id', 'timestamp'),
        Index('idx_audit_type_time', 'event_type', 'timestamp'),
        Index('idx_audit_severity', 'severity', 'timestamp'),
        Index('idx_audit_outcome', 'outcome', 'timestamp'),
    )


# ============================================================================
# Enums and Models
# ============================================================================

class EventCategory(str, Enum):
    """Categories of auditable events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    TOOL_EXECUTION = "tool_execution"
    API_CALL = "api_call"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    ADMIN = "admin"
    SYSTEM = "system"


class EventSeverity(str, Enum):
    """Severity levels for events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventOutcome(str, Enum):
    """Outcome of event."""
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    DENIED = "denied"


class AuditLogRequest(BaseModel):
    """Request to log an audit event."""
    event_type: str = Field(..., description="Specific type of event")
    event_category: EventCategory
    severity: EventSeverity

    # Actor
    actor_type: str = Field(..., description="Type of actor (user, agent, system)")
    actor_id: str = Field(..., description="ID of actor")
    source_ip: Optional[str] = None

    # Context
    user_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    task_id: Optional[UUID] = None
    session_id: Optional[str] = None

    # Action
    action: str = Field(..., description="Action performed")
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    outcome: EventOutcome

    # Data
    details: Optional[Dict[str, Any]] = None
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # Security
    permission_required: Optional[str] = None
    permission_granted: bool = False

    # Timing
    duration_ms: Optional[int] = None


class AuditLogResponse(BaseModel):
    """Response with audit log entry."""
    id: UUID
    sequence_number: int
    event_type: str
    event_category: str
    severity: str
    timestamp: datetime
    actor_id: str
    action: str
    outcome: str
    entry_hash: str


class AuditSearchRequest(BaseModel):
    """Search criteria for audit logs."""
    user_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    task_id: Optional[UUID] = None
    event_category: Optional[EventCategory] = None
    event_type: Optional[str] = None
    severity: Optional[EventSeverity] = None
    outcome: Optional[EventOutcome] = None
    actor_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class IntegrityCheckResult(BaseModel):
    """Result of integrity check."""
    is_valid: bool
    total_entries: int
    checked_entries: int
    invalid_entries: List[int] = Field(default_factory=list)
    first_invalid_sequence: Optional[int] = None
    error_message: Optional[str] = None


# ============================================================================
# Database
# ============================================================================

# Create async engine
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
# Audit Logger
# ============================================================================

class AuditLogger:
    """
    Manages tamper-evident audit logging with chain hashing.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="AuditLogger")
        self._last_hash = None
        self._sequence = 0

    async def log_event(
        self,
        request: AuditLogRequest,
        db: AsyncSession
    ) -> AuditLogEntry:
        """
        Log an audit event with integrity hash.

        Args:
            request: Audit log request
            db: Database session

        Returns:
            Created audit log entry
        """
        # Get next sequence number
        self._sequence += 1
        sequence_number = self._sequence

        # Build entry data for hashing
        entry_data = {
            "sequence": sequence_number,
            "event_type": request.event_type,
            "event_category": request.event_category.value,
            "actor_id": request.actor_id,
            "action": request.action,
            "outcome": request.outcome.value,
            "timestamp": datetime.utcnow().isoformat(),
            "previous_hash": self._last_hash or "GENESIS"
        }

        # Calculate entry hash
        entry_hash = self._calculate_hash(entry_data)

        # Create entry
        entry = AuditLogEntry(
            sequence_number=sequence_number,
            event_type=request.event_type,
            event_category=request.event_category.value,
            severity=request.severity.value,
            user_id=request.user_id,
            agent_id=request.agent_id,
            task_id=request.task_id,
            session_id=request.session_id,
            actor_type=request.actor_type,
            actor_id=request.actor_id,
            source_ip=request.source_ip,
            action=request.action,
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            outcome=request.outcome.value,
            details=request.details,
            request_data=request.request_data,
            response_data=request.response_data,
            error_message=request.error_message,
            permission_required=request.permission_required,
            permission_granted=request.permission_granted,
            previous_hash=self._last_hash,
            entry_hash=entry_hash,
            duration_ms=request.duration_ms
        )

        db.add(entry)
        await db.commit()
        await db.refresh(entry)

        # Update last hash
        self._last_hash = entry_hash

        self.logger.info(
            "Audit event logged",
            extra={
                "sequence": sequence_number,
                "event_type": request.event_type,
                "actor": request.actor_id,
                "outcome": request.outcome.value
            }
        )

        return entry

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of entry data."""
        # Convert to deterministic JSON string
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def verify_integrity(
        self,
        db: AsyncSession,
        start_sequence: Optional[int] = None,
        end_sequence: Optional[int] = None
    ) -> IntegrityCheckResult:
        """
        Verify integrity of audit log chain.

        Args:
            db: Database session
            start_sequence: Start of range to check (optional)
            end_sequence: End of range to check (optional)

        Returns:
            Integrity check result
        """
        from sqlalchemy import select, func

        # Get total count
        count_query = select(func.count(AuditLogEntry.id))
        result = await db.execute(count_query)
        total_entries = result.scalar()

        # Build query for entries to check
        query = select(AuditLogEntry).order_by(AuditLogEntry.sequence_number)

        if start_sequence is not None:
            query = query.where(AuditLogEntry.sequence_number >= start_sequence)
        if end_sequence is not None:
            query = query.where(AuditLogEntry.sequence_number <= end_sequence)

        result = await db.execute(query)
        entries = result.scalars().all()

        invalid_entries = []
        previous_hash = None

        for entry in entries:
            # Reconstruct entry data
            entry_data = {
                "sequence": entry.sequence_number,
                "event_type": entry.event_type,
                "event_category": entry.event_category,
                "actor_id": entry.actor_id,
                "action": entry.action,
                "outcome": entry.outcome,
                "timestamp": entry.timestamp.isoformat(),
                "previous_hash": entry.previous_hash or "GENESIS"
            }

            # Calculate expected hash
            expected_hash = self._calculate_hash(entry_data)

            # Verify hash matches
            if expected_hash != entry.entry_hash:
                invalid_entries.append(entry.sequence_number)

            # Verify chain
            if previous_hash and entry.previous_hash != previous_hash:
                invalid_entries.append(entry.sequence_number)

            previous_hash = entry.entry_hash

        is_valid = len(invalid_entries) == 0

        return IntegrityCheckResult(
            is_valid=is_valid,
            total_entries=total_entries,
            checked_entries=len(entries),
            invalid_entries=invalid_entries,
            first_invalid_sequence=invalid_entries[0] if invalid_entries else None,
            error_message=f"Found {len(invalid_entries)} invalid entries" if not is_valid else None
        )


# Initialize audit logger
audit_logger = AuditLogger()


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/audit", response_model=AuditLogResponse, status_code=status.HTTP_201_CREATED)
async def log_audit_event(
    request: AuditLogRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Log an audit event.

    Creates a tamper-evident audit log entry with integrity hash.
    """
    log = get_contextual_logger(
        __name__,
        action="log_audit_event",
        event_type=request.event_type,
        actor=request.actor_id
    )

    try:
        entry = await audit_logger.log_event(request, db)

        return AuditLogResponse(
            id=entry.id,
            sequence_number=entry.sequence_number,
            event_type=entry.event_type,
            event_category=entry.event_category,
            severity=entry.severity,
            timestamp=entry.timestamp,
            actor_id=entry.actor_id,
            action=entry.action,
            outcome=entry.outcome,
            entry_hash=entry.entry_hash
        )

    except Exception as e:
        log.error(f"Failed to log audit event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log audit event: {str(e)}"
        )


@app.post("/audit/search")
async def search_audit_logs(
    search: AuditSearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Search audit logs.

    Supports filtering by user, agent, task, time range, and more.
    """
    from sqlalchemy import select, and_

    # Build query
    query = select(AuditLogEntry)

    filters = []
    if search.user_id:
        filters.append(AuditLogEntry.user_id == search.user_id)
    if search.agent_id:
        filters.append(AuditLogEntry.agent_id == search.agent_id)
    if search.task_id:
        filters.append(AuditLogEntry.task_id == search.task_id)
    if search.event_category:
        filters.append(AuditLogEntry.event_category == search.event_category.value)
    if search.event_type:
        filters.append(AuditLogEntry.event_type == search.event_type)
    if search.severity:
        filters.append(AuditLogEntry.severity == search.severity.value)
    if search.outcome:
        filters.append(AuditLogEntry.outcome == search.outcome.value)
    if search.actor_id:
        filters.append(AuditLogEntry.actor_id == search.actor_id)
    if search.start_time:
        filters.append(AuditLogEntry.timestamp >= search.start_time)
    if search.end_time:
        filters.append(AuditLogEntry.timestamp <= search.end_time)

    if filters:
        query = query.where(and_(*filters))

    # Order by timestamp descending (most recent first)
    query = query.order_by(AuditLogEntry.timestamp.desc())

    # Apply pagination
    query = query.offset(search.offset).limit(search.limit)

    # Execute query
    result = await db.execute(query)
    entries = result.scalars().all()

    return {
        "count": len(entries),
        "offset": search.offset,
        "limit": search.limit,
        "entries": [
            {
                "id": str(entry.id),
                "sequence_number": entry.sequence_number,
                "event_type": entry.event_type,
                "event_category": entry.event_category,
                "severity": entry.severity,
                "timestamp": entry.timestamp.isoformat(),
                "actor_type": entry.actor_type,
                "actor_id": entry.actor_id,
                "action": entry.action,
                "resource_type": entry.resource_type,
                "resource_id": entry.resource_id,
                "outcome": entry.outcome,
                "details": entry.details,
                "entry_hash": entry.entry_hash
            }
            for entry in entries
        ]
    }


@app.get("/audit/{entry_id}")
async def get_audit_entry(
    entry_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed audit log entry by ID."""
    from sqlalchemy import select

    query = select(AuditLogEntry).where(AuditLogEntry.id == entry_id)
    result = await db.execute(query)
    entry = result.scalar_one_or_none()

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit entry {entry_id} not found"
        )

    return {
        "id": str(entry.id),
        "sequence_number": entry.sequence_number,
        "event_type": entry.event_type,
        "event_category": entry.event_category,
        "severity": entry.severity,
        "timestamp": entry.timestamp.isoformat(),
        "actor_type": entry.actor_type,
        "actor_id": entry.actor_id,
        "source_ip": entry.source_ip,
        "user_id": str(entry.user_id) if entry.user_id else None,
        "agent_id": str(entry.agent_id) if entry.agent_id else None,
        "task_id": str(entry.task_id) if entry.task_id else None,
        "session_id": entry.session_id,
        "action": entry.action,
        "resource_type": entry.resource_type,
        "resource_id": entry.resource_id,
        "outcome": entry.outcome,
        "details": entry.details,
        "request_data": entry.request_data,
        "response_data": entry.response_data,
        "error_message": entry.error_message,
        "permission_required": entry.permission_required,
        "permission_granted": entry.permission_granted,
        "previous_hash": entry.previous_hash,
        "entry_hash": entry.entry_hash,
        "duration_ms": entry.duration_ms
    }


@app.post("/audit/verify", response_model=IntegrityCheckResult)
async def verify_audit_integrity(
    start_sequence: Optional[int] = Query(None, description="Start sequence to check"),
    end_sequence: Optional[int] = Query(None, description="End sequence to check"),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify integrity of audit log chain.

    Checks hash chain integrity to detect tampering.
    """
    log = get_contextual_logger(
        __name__,
        action="verify_integrity",
        start_seq=start_sequence,
        end_seq=end_sequence
    )

    try:
        result = await audit_logger.verify_integrity(db, start_sequence, end_sequence)

        if not result.is_valid:
            log.warning(
                "Audit log integrity violation detected",
                extra={"invalid_count": len(result.invalid_entries)}
            )

        return result

    except Exception as e:
        log.error(f"Integrity check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Integrity check failed: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "audit-log",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat(),
        "retention_days": config.retention_days,
        "integrity_checks_enabled": config.enable_integrity_checks
    }


@app.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db)
):
    """Get audit log statistics."""
    from sqlalchemy import select, func

    # Total entries
    total_query = select(func.count(AuditLogEntry.id))
    total_result = await db.execute(total_query)
    total_entries = total_result.scalar()

    # Entries by severity
    severity_query = select(
        AuditLogEntry.severity,
        func.count(AuditLogEntry.id)
    ).group_by(AuditLogEntry.severity)
    severity_result = await db.execute(severity_query)
    by_severity = {row[0]: row[1] for row in severity_result}

    # Entries by category
    category_query = select(
        AuditLogEntry.event_category,
        func.count(AuditLogEntry.id)
    ).group_by(AuditLogEntry.event_category)
    category_result = await db.execute(category_query)
    by_category = {row[0]: row[1] for row in category_result}

    # Recent critical events
    critical_query = select(AuditLogEntry).where(
        AuditLogEntry.severity == EventSeverity.CRITICAL.value
    ).order_by(AuditLogEntry.timestamp.desc()).limit(10)
    critical_result = await db.execute(critical_query)
    recent_critical = critical_result.scalars().all()

    return {
        "total_entries": total_entries,
        "by_severity": by_severity,
        "by_category": by_category,
        "recent_critical_count": len(recent_critical),
        "recent_critical": [
            {
                "sequence": entry.sequence_number,
                "event_type": entry.event_type,
                "timestamp": entry.timestamp.isoformat(),
                "actor": entry.actor_id,
                "action": entry.action
            }
            for entry in recent_critical
        ]
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    logger.info(
        "Audit Log service starting",
        extra={
            "version": config.service_version,
            "retention_days": config.retention_days,
            "integrity_checks": config.enable_integrity_checks
        }
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Audit Log service ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Audit Log service shutting down")
    await engine.dispose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )
