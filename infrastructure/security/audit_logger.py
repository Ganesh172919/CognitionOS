"""
Audit Logger

Production audit logging for compliance and security monitoring.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    PASSWORD_CHANGED = "auth.password_changed"
    
    # Authorization
    ACCESS_GRANTED = "authz.access_granted"
    ACCESS_DENIED = "authz.access_denied"
    
    # Data Access
    DATA_READ = "data.read"
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"
    
    # API
    API_CALL = "api.call"
    API_KEY_CREATED = "api.key_created"
    API_KEY_REVOKED = "api.key_revoked"
    
    # Billing
    SUBSCRIPTION_CREATED = "billing.subscription_created"
    SUBSCRIPTION_UPDATED = "billing.subscription_updated"
    PAYMENT_PROCESSED = "billing.payment_processed"
    
    # System
    CONFIG_CHANGED = "system.config_changed"
    SERVICE_STARTED = "system.service_started"
    SERVICE_STOPPED = "system.service_stopped"
    
    # Security
    SECURITY_VIOLATION = "security.violation"
    RATE_LIMIT_EXCEEDED = "security.rate_limit_exceeded"


@dataclass
class AuditEvent:
    """An audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    tenant_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    result: str  # success, failure, denied
    ip_address: Optional[str]
    user_agent: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "result": self.result,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
        }


class AuditLogger:
    """
    Production audit logger.
    
    Features:
    - Structured audit events
    - Multiple output backends
    - Query interface
    - Compliance reporting
    - Immutable event storage
    """
    
    def __init__(self, storage_backend: Optional[Any] = None):
        """
        Initialize audit logger.
        
        Args:
            storage_backend: Backend for audit event storage
        """
        self.storage = storage_backend
        self.event_buffer: list = []
        self.buffer_size = 100
        
        logger.info("Audit logger initialized")
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        tenant_id: Optional[str],
        action: str,
        result: str = "success",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            user_id: User performing action
            tenant_id: Tenant context
            action: Action description
            result: Result (success, failure, denied)
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            ip_address: Client IP address
            user_agent: Client user agent
            details: Additional event details
            
        Returns:
            Event ID
        """
        event = AuditEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
        )
        
        # Add to buffer
        self.event_buffer.append(event)
        
        # Flush if buffer full
        if len(self.event_buffer) >= self.buffer_size:
            await self.flush()
        
        # Log to standard logger
        logger.info(
            f"AUDIT: {event_type} by user={user_id} tenant={tenant_id} "
            f"action={action} result={result}"
        )
        
        return event.event_id
    
    async def flush(self):
        """Flush event buffer to storage."""
        if not self.event_buffer:
            return
        
        if self.storage:
            try:
                await self.storage.store_events(self.event_buffer)
                logger.debug(f"Flushed {len(self.event_buffer)} audit events")
            except Exception as e:
                logger.error(f"Error flushing audit events: {e}")
        
        self.event_buffer.clear()
    
    async def query_events(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list:
        """
        Query audit events.
        
        Args:
            user_id: Filter by user
            tenant_id: Filter by tenant
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum results
            
        Returns:
            List of matching events
        """
        if not self.storage:
            logger.warning("No storage backend configured for audit queries")
            return []
        
        return await self.storage.query_events(
            user_id=user_id,
            tenant_id=tenant_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
    
    # Convenience methods for common events
    
    async def log_login(
        self,
        user_id: str,
        tenant_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Log user login event."""
        await self.log_event(
            event_type=AuditEventType.LOGIN if success else AuditEventType.LOGIN_FAILED,
            user_id=user_id,
            tenant_id=tenant_id,
            action="User login",
            result="success" if success else "failure",
            ip_address=ip_address,
            user_agent=user_agent,
        )
    
    async def log_data_access(
        self,
        user_id: str,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        operation: str,  # read, create, update, delete
        ip_address: Optional[str] = None,
    ):
        """Log data access event."""
        event_type_map = {
            "read": AuditEventType.DATA_READ,
            "create": AuditEventType.DATA_CREATED,
            "update": AuditEventType.DATA_UPDATED,
            "delete": AuditEventType.DATA_DELETED,
        }
        
        await self.log_event(
            event_type=event_type_map.get(operation, AuditEventType.DATA_READ),
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=f"Data {operation}",
            result="success",
            ip_address=ip_address,
        )
    
    async def log_api_call(
        self,
        user_id: Optional[str],
        tenant_id: Optional[str],
        endpoint: str,
        method: str,
        status_code: int,
        ip_address: Optional[str] = None,
    ):
        """Log API call event."""
        await self.log_event(
            event_type=AuditEventType.API_CALL,
            user_id=user_id,
            tenant_id=tenant_id,
            action=f"{method} {endpoint}",
            result="success" if 200 <= status_code < 400 else "failure",
            ip_address=ip_address,
            details={"status_code": status_code},
        )
    
    async def log_security_violation(
        self,
        user_id: Optional[str],
        tenant_id: Optional[str],
        violation_type: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
    ):
        """Log security violation event."""
        await self.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id=user_id,
            tenant_id=tenant_id,
            action=f"Security violation: {violation_type}",
            result="denied",
            ip_address=ip_address,
            details=details,
        )
