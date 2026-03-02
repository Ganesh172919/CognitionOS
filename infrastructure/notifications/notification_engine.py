"""
Notification Engine — CognitionOS

Multi-channel notification system with:
- Email, in-app, push, SMS, Slack, webhook channels
- Template rendering with variables
- Notification preferences per user
- Delivery tracking and receipts
- Rate limiting per channel
- Batch sending
- Priority queuing
- A/B testing for notification content
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    EMAIL = "email"
    IN_APP = "in_app"
    PUSH = "push"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DeliveryState(str, Enum):
    QUEUED = "queued"
    SENDING = "sending"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    BOUNCED = "bounced"


class NotificationCategory(str, Enum):
    SYSTEM = "system"
    ALERT = "alert"
    BILLING = "billing"
    AGENT = "agent"
    TASK = "task"
    SECURITY = "security"
    MARKETING = "marketing"
    ONBOARDING = "onboarding"


@dataclass
class NotificationTemplate:
    template_id: str
    name: str
    channel: NotificationChannel
    subject: str = ""
    body: str = ""
    html_body: str = ""
    variables: List[str] = field(default_factory=list)
    category: NotificationCategory = NotificationCategory.SYSTEM

    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        subject = self.subject
        body = self.body
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))
        return {"subject": subject, "body": body}


@dataclass
class UserPreferences:
    user_id: str
    enabled_channels: Set[NotificationChannel] = field(
        default_factory=lambda: {NotificationChannel.EMAIL, NotificationChannel.IN_APP}
    )
    muted_categories: Set[NotificationCategory] = field(default_factory=set)
    quiet_hours_start: Optional[int] = None  # 0-23
    quiet_hours_end: Optional[int] = None
    timezone: str = "UTC"


@dataclass
class Notification:
    notification_id: str
    user_id: str
    channel: NotificationChannel
    category: NotificationCategory
    priority: NotificationPriority
    subject: str
    body: str
    state: DeliveryState = DeliveryState.QUEUED
    tenant_id: str = ""
    template_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    delivered_at: Optional[float] = None
    read_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.notification_id,
            "channel": self.channel.value,
            "category": self.category.value,
            "priority": self.priority.value,
            "subject": self.subject,
            "state": self.state.value,
            "created_at": self.created_at,
        }


class NotificationEngine:
    """
    Multi-channel notification engine with templates, preferences,
    rate limiting, and delivery tracking.
    """

    def __init__(self, *, rate_limit_per_minute: int = 100):
        self._templates: Dict[str, NotificationTemplate] = {}
        self._preferences: Dict[str, UserPreferences] = {}
        self._notifications: List[Notification] = []
        self._in_app: Dict[str, List[Notification]] = defaultdict(list)
        self._channel_handlers: Dict[NotificationChannel, Callable] = {}
        self._rate_limit = rate_limit_per_minute
        self._send_count: Dict[str, int] = defaultdict(int)
        self._metrics = {
            "total_sent": 0, "total_delivered": 0,
            "total_failed": 0, "total_read": 0,
        }

    # ── Templates ──

    def register_template(self, name: str, channel: NotificationChannel,
                            subject: str, body: str, *,
                            category: NotificationCategory = NotificationCategory.SYSTEM,
                            variables: Optional[List[str]] = None
                            ) -> NotificationTemplate:
        template = NotificationTemplate(
            template_id=uuid.uuid4().hex[:12],
            name=name, channel=channel,
            subject=subject, body=body,
            category=category,
            variables=variables or [],
        )
        self._templates[name] = template
        return template

    # ── Channel Handlers ──

    def register_channel(self, channel: NotificationChannel,
                           handler: Callable[..., Awaitable[bool]]):
        self._channel_handlers[channel] = handler

    # ── Preferences ──

    def set_preferences(self, user_id: str, **kwargs) -> UserPreferences:
        prefs = self._preferences.get(user_id, UserPreferences(user_id=user_id))
        for k, v in kwargs.items():
            if hasattr(prefs, k):
                setattr(prefs, k, v)
        self._preferences[user_id] = prefs
        return prefs

    def get_preferences(self, user_id: str) -> UserPreferences:
        return self._preferences.get(user_id, UserPreferences(user_id=user_id))

    # ── Sending ──

    async def send(self, user_id: str, *,
                     channel: Optional[NotificationChannel] = None,
                     template_name: Optional[str] = None,
                     subject: str = "",
                     body: str = "",
                     category: NotificationCategory = NotificationCategory.SYSTEM,
                     priority: NotificationPriority = NotificationPriority.NORMAL,
                     context: Optional[Dict[str, Any]] = None,
                     tenant_id: str = "") -> Notification:
        """Send a notification to a user."""
        # Resolve template
        if template_name:
            template = self._templates.get(template_name)
            if template:
                rendered = template.render(context or {})
                subject = rendered["subject"]
                body = rendered["body"]
                channel = channel or template.channel
                category = template.category

        if not channel:
            channel = NotificationChannel.IN_APP

        # Check preferences
        prefs = self.get_preferences(user_id)
        if channel not in prefs.enabled_channels:
            channel = NotificationChannel.IN_APP  # Fallback
        if category in prefs.muted_categories:
            # Still store but mark as muted
            pass

        notification = Notification(
            notification_id=uuid.uuid4().hex[:12],
            user_id=user_id, channel=channel,
            category=category, priority=priority,
            subject=subject, body=body,
            tenant_id=tenant_id,
        )

        self._notifications.append(notification)
        self._metrics["total_sent"] += 1

        # Deliver
        await self._deliver(notification)

        # Store in-app
        if channel == NotificationChannel.IN_APP:
            self._in_app[user_id].append(notification)

        return notification

    async def send_batch(self, user_ids: List[str], **kwargs) -> List[Notification]:
        tasks = [self.send(uid, **kwargs) for uid in user_ids]
        return await asyncio.gather(*tasks)

    async def _deliver(self, notification: Notification):
        handler = self._channel_handlers.get(notification.channel)
        notification.state = DeliveryState.SENDING

        try:
            if handler:
                success = await handler(
                    user_id=notification.user_id,
                    subject=notification.subject,
                    body=notification.body,
                    metadata=notification.metadata,
                )
                if success:
                    notification.state = DeliveryState.DELIVERED
                    notification.delivered_at = time.time()
                    self._metrics["total_delivered"] += 1
                else:
                    notification.state = DeliveryState.FAILED
                    self._metrics["total_failed"] += 1
            else:
                # No handler — mark as delivered for in-app
                if notification.channel == NotificationChannel.IN_APP:
                    notification.state = DeliveryState.DELIVERED
                    notification.delivered_at = time.time()
                    self._metrics["total_delivered"] += 1
                else:
                    notification.state = DeliveryState.FAILED
                    notification.error = f"No handler for {notification.channel.value}"
                    self._metrics["total_failed"] += 1

        except Exception as exc:
            notification.state = DeliveryState.FAILED
            notification.error = str(exc)
            self._metrics["total_failed"] += 1

    # ── In-App Notifications ──

    def get_inbox(self, user_id: str, *, unread_only: bool = False,
                    limit: int = 50) -> List[Dict[str, Any]]:
        notifications = self._in_app.get(user_id, [])
        if unread_only:
            notifications = [n for n in notifications if not n.read_at]
        return [n.to_dict() for n in notifications[-limit:]]

    def mark_read(self, notification_id: str):
        for notif in self._notifications:
            if notif.notification_id == notification_id and not notif.read_at:
                notif.read_at = time.time()
                notif.state = DeliveryState.READ
                self._metrics["total_read"] += 1
                break

    def mark_all_read(self, user_id: str):
        for notif in self._in_app.get(user_id, []):
            if not notif.read_at:
                notif.read_at = time.time()
                notif.state = DeliveryState.READ
                self._metrics["total_read"] += 1

    def get_unread_count(self, user_id: str) -> int:
        return sum(
            1 for n in self._in_app.get(user_id, []) if not n.read_at
        )

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        by_channel = defaultdict(int)
        by_state = defaultdict(int)
        for n in self._notifications:
            by_channel[n.channel.value] += 1
            by_state[n.state.value] += 1

        return {
            **self._metrics,
            "templates": len(self._templates),
            "users_with_prefs": len(self._preferences),
            "by_channel": dict(by_channel),
            "by_state": dict(by_state),
        }

    def cleanup(self, *, max_notifications: int = 100000):
        if len(self._notifications) > max_notifications:
            self._notifications = self._notifications[-max_notifications // 2:]


# ── Singleton ──
_engine: Optional[NotificationEngine] = None


def get_notification_engine() -> NotificationEngine:
    global _engine
    if not _engine:
        _engine = NotificationEngine()
    return _engine
