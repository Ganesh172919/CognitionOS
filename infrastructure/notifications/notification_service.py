"""
Notification Service — CognitionOS

Multi-channel notification delivery:
- In-app notifications
- Email templates
- Push notification dispatching
- Notification preferences
- Read/unread tracking
- Batching and digest
- Priority levels
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"
    WEBHOOK = "webhook"
    SMS = "sms"


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationType(str, Enum):
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_STATUS = "agent_status"
    BILLING_ALERT = "billing_alert"
    SECURITY_ALERT = "security_alert"
    SYSTEM_UPDATE = "system_update"
    TEAM_INVITE = "team_invite"
    QUOTA_WARNING = "quota_warning"
    ACHIEVEMENT = "achievement"
    CUSTOM = "custom"


@dataclass
class Notification:
    notification_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    tenant_id: str = ""
    notification_type: NotificationType = NotificationType.CUSTOM
    title: str = ""
    message: str = ""
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.IN_APP])
    is_read: bool = False
    is_delivered: bool = False
    action_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    read_at: Optional[str] = None
    delivered_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "type": self.notification_type.value,
            "title": self.title, "message": self.message,
            "priority": self.priority.value, "is_read": self.is_read,
            "action_url": self.action_url, "created_at": self.created_at}


@dataclass
class NotificationPreferences:
    user_id: str
    enabled_channels: Dict[NotificationType, List[NotificationChannel]] = field(
        default_factory=dict)
    muted_types: Set[NotificationType] = field(default_factory=set)
    digest_enabled: bool = False
    digest_frequency: str = "daily"  # daily, weekly
    quiet_hours_start: Optional[int] = None  # hour 0-23
    quiet_hours_end: Optional[int] = None


class NotificationService:
    """Multi-channel notification delivery with preferences and tracking."""

    def __init__(self) -> None:
        self._notifications: Dict[str, List[Notification]] = defaultdict(list)  # user_id -> notifs
        self._preferences: Dict[str, NotificationPreferences] = {}
        self._templates: Dict[str, Dict[str, str]] = {}  # type -> {title_template, body_template}
        self._metrics: Dict[str, int] = defaultdict(int)

    # ---- send ----
    def send(self, notification: Notification) -> str:
        # Check preferences
        prefs = self._preferences.get(notification.user_id)
        if prefs and notification.notification_type in prefs.muted_types:
            self._metrics["muted"] += 1
            return ""

        # Apply channel preferences
        if prefs and notification.notification_type in prefs.enabled_channels:
            notification.channels = prefs.enabled_channels[notification.notification_type]

        self._notifications[notification.user_id].append(notification)
        notification.is_delivered = True
        notification.delivered_at = datetime.now(timezone.utc).isoformat()
        self._metrics["sent"] += 1
        self._metrics[f"sent_{notification.notification_type.value}"] += 1

        logger.info("Notification sent: %s → %s [%s]",
                     notification.notification_type.value,
                     notification.user_id, notification.title)
        return notification.notification_id

    def send_bulk(self, user_ids: List[str], *,
                  notification_type: NotificationType,
                  title: str, message: str,
                  priority: NotificationPriority = NotificationPriority.NORMAL) -> int:
        count = 0
        for uid in user_ids:
            notif = Notification(
                user_id=uid, notification_type=notification_type,
                title=title, message=message, priority=priority)
            if self.send(notif):
                count += 1
        return count

    # ---- read/unread ----
    def mark_read(self, user_id: str, notification_id: str) -> bool:
        for notif in self._notifications.get(user_id, []):
            if notif.notification_id == notification_id:
                notif.is_read = True
                notif.read_at = datetime.now(timezone.utc).isoformat()
                return True
        return False

    def mark_all_read(self, user_id: str) -> int:
        count = 0
        now = datetime.now(timezone.utc).isoformat()
        for notif in self._notifications.get(user_id, []):
            if not notif.is_read:
                notif.is_read = True
                notif.read_at = now
                count += 1
        return count

    # ---- query ----
    def get_notifications(self, user_id: str, *, unread_only: bool = False,
                           limit: int = 50) -> List[Dict[str, Any]]:
        notifs = self._notifications.get(user_id, [])
        if unread_only:
            notifs = [n for n in notifs if not n.is_read]
        return [n.to_dict() for n in sorted(notifs, key=lambda x: x.created_at, reverse=True)[:limit]]

    def get_unread_count(self, user_id: str) -> int:
        return sum(1 for n in self._notifications.get(user_id, []) if not n.is_read)

    # ---- preferences ----
    def set_preferences(self, prefs: NotificationPreferences) -> None:
        self._preferences[prefs.user_id] = prefs

    def get_preferences(self, user_id: str) -> Optional[NotificationPreferences]:
        return self._preferences.get(user_id)

    # ---- templates ----
    def register_template(self, ntype: NotificationType,
                           title_template: str, body_template: str) -> None:
        self._templates[ntype.value] = {
            "title": title_template, "body": body_template}

    def send_from_template(self, user_id: str, ntype: NotificationType,
                            variables: Dict[str, str], **kwargs: Any) -> str:
        template = self._templates.get(ntype.value)
        if not template:
            return ""
        title = template["title"]
        body = template["body"]
        for k, v in variables.items():
            title = title.replace(f"{{{k}}}", v)
            body = body.replace(f"{{{k}}}", v)
        notif = Notification(user_id=user_id, notification_type=ntype,
                              title=title, message=body, **kwargs)
        return self.send(notif)

    # ---- cleanup ----
    def delete_old(self, *, days: int = 90) -> int:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        total = 0
        for uid in list(self._notifications):
            before = len(self._notifications[uid])
            self._notifications[uid] = [
                n for n in self._notifications[uid] if n.created_at >= cutoff]
            total += before - len(self._notifications[uid])
        return total

    def get_metrics(self) -> Dict[str, Any]:
        total = sum(len(v) for v in self._notifications.values())
        unread = sum(self.get_unread_count(uid) for uid in self._notifications)
        return {**dict(self._metrics), "total_notifications": total,
                "total_unread": unread, "total_users": len(self._notifications)}


_service: NotificationService | None = None

def get_notification_service() -> NotificationService:
    global _service
    if not _service:
        _service = NotificationService()
    return _service
