"""
Advanced Multi-Channel Notification System

Production-grade notification system with:
- Multi-channel delivery (email, SMS, webhook, push)
- Template engine with variables
- Delivery tracking and retries
- User preference management
- Notification batching
- Rate limiting per channel
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
import json
from uuid import uuid4
import re

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    PUSH = "push"
    SLACK = "slack"
    TEAMS = "teams"


class NotificationPriority(int, Enum):
    """Notification priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class NotificationStatus(str, Enum):
    """Notification delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class NotificationTemplate:
    """Notification template with variable substitution."""
    template_id: str
    name: str
    channel: NotificationChannel
    subject: Optional[str] = None
    body: str = ""
    variables: List[str] = field(default_factory=list)
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Render template with context variables."""
        rendered = {
            "body": self.body,
        }
        
        if self.subject:
            rendered["subject"] = self.subject
            
        # Replace variables
        for key, value in context.items():
            for field in rendered:
                rendered[field] = rendered[field].replace(f"{{{{{key}}}}}", str(value))
                
        return rendered


@dataclass
class UserPreferences:
    """User notification preferences."""
    user_id: str
    enabled_channels: Set[NotificationChannel] = field(
        default_factory=lambda: {NotificationChannel.EMAIL}
    )
    quiet_hours_start: Optional[int] = None  # Hour 0-23
    quiet_hours_end: Optional[int] = None
    frequency_limit: Dict[NotificationChannel, int] = field(default_factory=dict)  # Per day
    
    def can_send(self, channel: NotificationChannel) -> bool:
        """Check if notification can be sent."""
        if channel not in self.enabled_channels:
            return False
            
        # Check quiet hours
        if self.quiet_hours_start and self.quiet_hours_end:
            current_hour = datetime.utcnow().hour
            if self.quiet_hours_start <= current_hour < self.quiet_hours_end:
                return False
                
        return True


@dataclass
class Notification:
    """Notification message."""
    notification_id: str
    user_id: str
    channel: NotificationChannel
    priority: NotificationPriority
    subject: Optional[str]
    body: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Delivery tracking
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class DeliveryResult:
    """Notification delivery result."""
    success: bool
    message: str
    delivered_at: Optional[datetime] = None
    error: Optional[str] = None


class NotificationProvider:
    """Base notification provider interface."""
    
    async def send(
        self,
        recipient: str,
        subject: Optional[str],
        body: str,
        metadata: Dict[str, Any],
    ) -> DeliveryResult:
        """Send notification."""
        raise NotImplementedError


class EmailProvider(NotificationProvider):
    """Email notification provider."""
    
    def __init__(self, smtp_host: str, smtp_port: int, from_address: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_address = from_address
        
    async def send(
        self,
        recipient: str,
        subject: Optional[str],
        body: str,
        metadata: Dict[str, Any],
    ) -> DeliveryResult:
        """Send email."""
        try:
            # Simulate email sending (replace with actual SMTP in production)
            await asyncio.sleep(0.1)
            
            logger.info(f"Email sent to {recipient}: {subject}")
            return DeliveryResult(
                success=True,
                message=f"Email sent to {recipient}",
                delivered_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Email failed: {e}")
            return DeliveryResult(
                success=False,
                message="Email delivery failed",
                error=str(e),
            )


class SMSProvider(NotificationProvider):
    """SMS notification provider."""
    
    def __init__(self, api_key: str, from_number: str):
        self.api_key = api_key
        self.from_number = from_number
        
    async def send(
        self,
        recipient: str,
        subject: Optional[str],
        body: str,
        metadata: Dict[str, Any],
    ) -> DeliveryResult:
        """Send SMS."""
        try:
            # Simulate SMS sending (replace with Twilio/etc in production)
            await asyncio.sleep(0.1)
            
            logger.info(f"SMS sent to {recipient}")
            return DeliveryResult(
                success=True,
                message=f"SMS sent to {recipient}",
                delivered_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"SMS failed: {e}")
            return DeliveryResult(
                success=False,
                message="SMS delivery failed",
                error=str(e),
            )


class WebhookProvider(NotificationProvider):
    """Webhook notification provider."""
    
    def __init__(self, signing_secret: str):
        self.signing_secret = signing_secret
        
    async def send(
        self,
        recipient: str,  # webhook URL
        subject: Optional[str],
        body: str,
        metadata: Dict[str, Any],
    ) -> DeliveryResult:
        """Send webhook."""
        try:
            # Simulate HTTP POST (replace with aiohttp in production)
            await asyncio.sleep(0.1)
            
            logger.info(f"Webhook sent to {recipient}")
            return DeliveryResult(
                success=True,
                message=f"Webhook sent to {recipient}",
                delivered_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Webhook failed: {e}")
            return DeliveryResult(
                success=False,
                message="Webhook delivery failed",
                error=str(e),
            )


class NotificationSystem:
    """
    Advanced multi-channel notification system.
    
    Features:
    - Multiple delivery channels (email, SMS, webhook, push)
    - Template engine with variable substitution
    - Delivery tracking and retries
    - User preference management
    - Notification batching
    - Rate limiting per channel
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: int = 300,  # seconds
        batch_size: int = 100,
        rate_limit_per_minute: int = 60,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.rate_limit_per_minute = rate_limit_per_minute
        
        self.templates: Dict[str, NotificationTemplate] = {}
        self.preferences: Dict[str, UserPreferences] = {}
        self.notifications: Dict[str, Notification] = {}
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        
        # Rate limiting
        self.sent_counts: Dict[NotificationChannel, List[datetime]] = {}
        
        # Background tasks
        self.is_running = False
        
    async def start(self):
        """Start notification system."""
        self.is_running = True
        logger.info("Starting notification system")
        
        await asyncio.gather(
            self._delivery_loop(),
            self._retry_loop(),
        )
        
    async def stop(self):
        """Stop notification system."""
        logger.info("Stopping notification system")
        self.is_running = False
        
    def register_provider(
        self,
        channel: NotificationChannel,
        provider: NotificationProvider,
    ):
        """Register notification provider."""
        self.providers[channel] = provider
        logger.info(f"Registered provider for {channel.value}")
        
    def register_template(self, template: NotificationTemplate):
        """Register notification template."""
        self.templates[template.template_id] = template
        logger.info(f"Registered template: {template.name}")
        
    def set_user_preferences(self, preferences: UserPreferences):
        """Set user notification preferences."""
        self.preferences[preferences.user_id] = preferences
        
    async def send_notification(
        self,
        user_id: str,
        channel: NotificationChannel,
        subject: Optional[str],
        body: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Send notification."""
        metadata = metadata or {}
        
        # Check user preferences
        prefs = self.preferences.get(user_id, UserPreferences(user_id=user_id))
        if not prefs.can_send(channel):
            logger.info(f"Notification blocked by preferences: {user_id}/{channel.value}")
            return ""
            
        # Create notification
        notification = Notification(
            notification_id=str(uuid4()),
            user_id=user_id,
            channel=channel,
            priority=priority,
            subject=subject,
            body=body,
            metadata=metadata,
        )
        
        self.notifications[notification.notification_id] = notification
        logger.info(f"Queued notification: {notification.notification_id}")
        
        return notification.notification_id
        
    async def send_from_template(
        self,
        user_id: str,
        template_id: str,
        context: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> str:
        """Send notification from template."""
        if template_id not in self.templates:
            logger.error(f"Template not found: {template_id}")
            return ""
            
        template = self.templates[template_id]
        rendered = template.render(context)
        
        return await self.send_notification(
            user_id=user_id,
            channel=template.channel,
            subject=rendered.get("subject"),
            body=rendered["body"],
            priority=priority,
            metadata={"template_id": template_id},
        )
        
    async def send_batch(
        self,
        user_ids: List[str],
        channel: NotificationChannel,
        subject: Optional[str],
        body: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> List[str]:
        """Send notification to multiple users."""
        notification_ids = []
        
        for user_id in user_ids:
            notification_id = await self.send_notification(
                user_id=user_id,
                channel=channel,
                subject=subject,
                body=body,
                priority=priority,
            )
            if notification_id:
                notification_ids.append(notification_id)
                
        logger.info(f"Queued {len(notification_ids)} batch notifications")
        return notification_ids
        
    async def _delivery_loop(self):
        """Main delivery loop."""
        while self.is_running:
            try:
                # Get pending notifications sorted by priority
                pending = [
                    n for n in self.notifications.values()
                    if n.status == NotificationStatus.PENDING
                ]
                pending.sort(key=lambda n: n.priority.value)
                
                # Deliver up to batch_size notifications
                for notification in pending[:self.batch_size]:
                    if not self._check_rate_limit(notification.channel):
                        continue
                        
                    await self._deliver_notification(notification)
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in delivery loop: {e}")
                await asyncio.sleep(1)
                
    async def _retry_loop(self):
        """Retry failed notifications."""
        while self.is_running:
            try:
                # Get failed notifications eligible for retry
                now = datetime.utcnow()
                
                for notification in self.notifications.values():
                    if notification.status == NotificationStatus.RETRY:
                        if notification.failed_at:
                            elapsed = (now - notification.failed_at).total_seconds()
                            if elapsed >= self.retry_delay:
                                await self._deliver_notification(notification)
                                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in retry loop: {e}")
                await asyncio.sleep(60)
                
    async def _deliver_notification(self, notification: Notification):
        """Deliver single notification."""
        if notification.channel not in self.providers:
            logger.error(f"No provider for channel: {notification.channel.value}")
            notification.status = NotificationStatus.FAILED
            notification.error = "No provider configured"
            return
            
        provider = self.providers[notification.channel]
        
        try:
            notification.status = NotificationStatus.PENDING
            notification.sent_at = datetime.utcnow()
            
            # Get recipient from metadata (simplified - would be from user profile)
            recipient = notification.metadata.get("recipient", notification.user_id)
            
            result = await provider.send(
                recipient=recipient,
                subject=notification.subject,
                body=notification.body,
                metadata=notification.metadata,
            )
            
            if result.success:
                notification.status = NotificationStatus.DELIVERED
                notification.delivered_at = result.delivered_at
                self._record_sent(notification.channel)
                
                logger.info(f"Delivered notification: {notification.notification_id}")
            else:
                raise Exception(result.error or "Delivery failed")
                
        except Exception as e:
            notification.failed_at = datetime.utcnow()
            notification.error = str(e)
            notification.retry_count += 1
            
            if notification.retry_count < self.max_retries:
                notification.status = NotificationStatus.RETRY
                logger.warning(
                    f"Notification failed, will retry: {notification.notification_id} "
                    f"(attempt {notification.retry_count}/{self.max_retries})"
                )
            else:
                notification.status = NotificationStatus.FAILED
                logger.error(f"Notification failed permanently: {notification.notification_id}")
                
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check rate limit for channel."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        
        # Clean old entries
        if channel not in self.sent_counts:
            self.sent_counts[channel] = []
            
        self.sent_counts[channel] = [
            t for t in self.sent_counts[channel] if t > cutoff
        ]
        
        # Check limit
        return len(self.sent_counts[channel]) < self.rate_limit_per_minute
        
    def _record_sent(self, channel: NotificationChannel):
        """Record sent notification for rate limiting."""
        if channel not in self.sent_counts:
            self.sent_counts[channel] = []
        self.sent_counts[channel].append(datetime.utcnow())
        
    def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get notification status."""
        if notification_id not in self.notifications:
            return None
            
        notification = self.notifications[notification_id]
        return {
            "notification_id": notification.notification_id,
            "user_id": notification.user_id,
            "channel": notification.channel.value,
            "status": notification.status.value,
            "created_at": notification.created_at.isoformat(),
            "sent_at": notification.sent_at.isoformat() if notification.sent_at else None,
            "delivered_at": notification.delivered_at.isoformat() if notification.delivered_at else None,
            "retry_count": notification.retry_count,
            "error": notification.error,
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get notification metrics."""
        return {
            "total": len(self.notifications),
            "pending": len([n for n in self.notifications.values() if n.status == NotificationStatus.PENDING]),
            "sent": len([n for n in self.notifications.values() if n.status == NotificationStatus.SENT]),
            "delivered": len([n for n in self.notifications.values() if n.status == NotificationStatus.DELIVERED]),
            "failed": len([n for n in self.notifications.values() if n.status == NotificationStatus.FAILED]),
            "retry": len([n for n in self.notifications.values() if n.status == NotificationStatus.RETRY]),
        }
