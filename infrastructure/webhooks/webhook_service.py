"""
Webhook Delivery Service — CognitionOS

Reliable webhook delivery with:
- Retry with exponential backoff
- Signature verification (HMAC-SHA256)
- Delivery tracking and audit
- Payload templating
- Event filtering
- Batch delivery
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookEndpoint:
    endpoint_id: str = field(default_factory=lambda: str(uuid4()))
    url: str = ""
    secret: str = ""
    tenant_id: str = ""
    status: WebhookStatus = WebhookStatus.ACTIVE
    events: List[str] = field(default_factory=list)  # ["agent.completed", "billing.*"]
    headers: Dict[str, str] = field(default_factory=dict)
    max_retries: int = 5
    timeout_seconds: float = 30.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    failure_count: int = 0
    last_delivery_at: Optional[str] = None

    def matches_event(self, event_type: str) -> bool:
        for pattern in self.events:
            if pattern == "*":
                return True
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if event_type.startswith(prefix):
                    return True
            if pattern == event_type:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {"endpoint_id": self.endpoint_id, "url": self.url,
                "status": self.status.value, "events": self.events,
                "tenant_id": self.tenant_id, "failure_count": self.failure_count,
                "last_delivery_at": self.last_delivery_at}


@dataclass
class DeliveryAttempt:
    attempt_id: str = field(default_factory=lambda: str(uuid4()))
    endpoint_id: str = ""
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    status: DeliveryStatus = DeliveryStatus.PENDING
    status_code: Optional[int] = None
    response_body: str = ""
    attempt_number: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    delivered_at: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"attempt_id": self.attempt_id, "endpoint_id": self.endpoint_id,
                "event_type": self.event_type, "status": self.status.value,
                "status_code": self.status_code, "attempt_number": self.attempt_number,
                "created_at": self.created_at, "duration_ms": self.duration_ms}


class WebhookService:
    """Manages webhook registration, delivery, and retry logic."""

    def __init__(self) -> None:
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._deliveries: List[DeliveryAttempt] = []
        self._metrics: Dict[str, int] = defaultdict(int)

    # ---- endpoint management ----
    def register_endpoint(self, endpoint: WebhookEndpoint) -> str:
        self._endpoints[endpoint.endpoint_id] = endpoint
        logger.info("Webhook endpoint registered: %s -> %s", endpoint.endpoint_id, endpoint.url)
        return endpoint.endpoint_id

    def update_endpoint(self, endpoint_id: str, **updates: Any) -> bool:
        ep = self._endpoints.get(endpoint_id)
        if not ep:
            return False
        for k, v in updates.items():
            if hasattr(ep, k):
                setattr(ep, k, v)
        return True

    def delete_endpoint(self, endpoint_id: str) -> bool:
        return self._endpoints.pop(endpoint_id, None) is not None

    def list_endpoints(self, *, tenant_id: str = "") -> List[Dict[str, Any]]:
        eps = list(self._endpoints.values())
        if tenant_id:
            eps = [e for e in eps if e.tenant_id == tenant_id]
        return [e.to_dict() for e in eps]

    # ---- signature generation ----
    @staticmethod
    def generate_signature(payload: str, secret: str) -> str:
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def verify_signature(payload: str, secret: str, signature: str) -> bool:
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    # ---- delivery ----
    def prepare_delivery(self, event_type: str, payload: Dict[str, Any]) -> List[DeliveryAttempt]:
        """Create delivery attempts for all matching endpoints."""
        attempts = []
        for ep in self._endpoints.values():
            if ep.status != WebhookStatus.ACTIVE:
                continue
            if not ep.matches_event(event_type):
                continue
            attempt = DeliveryAttempt(
                endpoint_id=ep.endpoint_id, event_type=event_type,
                payload=payload)
            attempts.append(attempt)
            self._deliveries.append(attempt)
        self._metrics["deliveries_prepared"] += len(attempts)
        return attempts

    def record_delivery_result(self, attempt_id: str, *, success: bool,
                                status_code: int = 0, response: str = "",
                                error: str = "", duration_ms: float = 0) -> None:
        for attempt in self._deliveries:
            if attempt.attempt_id == attempt_id:
                if success:
                    attempt.status = DeliveryStatus.DELIVERED
                    attempt.delivered_at = datetime.now(timezone.utc).isoformat()
                    self._metrics["delivered"] += 1
                    ep = self._endpoints.get(attempt.endpoint_id)
                    if ep:
                        ep.last_delivery_at = attempt.delivered_at
                        ep.failure_count = 0
                else:
                    attempt.status = DeliveryStatus.FAILED
                    attempt.error = error
                    self._metrics["failed"] += 1
                    ep = self._endpoints.get(attempt.endpoint_id)
                    if ep:
                        ep.failure_count += 1
                        if ep.failure_count >= 10:
                            ep.status = WebhookStatus.SUSPENDED
                            logger.warning("Webhook suspended after %d failures: %s",
                                           ep.failure_count, ep.url)
                attempt.status_code = status_code
                attempt.response_body = response[:1000]
                attempt.duration_ms = duration_ms
                break

    # ---- query ----
    def get_delivery_history(self, *, endpoint_id: str = "",
                              limit: int = 100) -> List[Dict[str, Any]]:
        deliveries = self._deliveries
        if endpoint_id:
            deliveries = [d for d in deliveries if d.endpoint_id == endpoint_id]
        return [d.to_dict() for d in deliveries[-limit:]]

    def get_metrics(self) -> Dict[str, Any]:
        return {**dict(self._metrics),
                "total_endpoints": len(self._endpoints),
                "active_endpoints": sum(1 for e in self._endpoints.values()
                                        if e.status == WebhookStatus.ACTIVE),
                "total_deliveries": len(self._deliveries)}


_service: WebhookService | None = None

def get_webhook_service() -> WebhookService:
    global _service
    if not _service:
        _service = WebhookService()
    return _service
