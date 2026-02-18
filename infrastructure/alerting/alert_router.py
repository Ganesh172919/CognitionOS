"""
Alert Router

Multi-channel alert routing and notification delivery system.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AlertChannel(str, Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class ChannelConfig:
    """Configuration for an alert channel."""
    channel: AlertChannel
    enabled: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class AlertChannelBase(ABC):
    """Base class for alert channels."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize channel with configuration."""
        self.config = config
    
    @abstractmethod
    async def send_alert(self, alert: Any) -> bool:
        """
        Send alert through this channel.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if successful
        """
        pass


class EmailChannel(AlertChannelBase):
    """Email alert channel."""
    
    async def send_alert(self, alert: Any) -> bool:
        """Send alert via email."""
        try:
            # Get email config
            to_addresses = self.config.get("to_addresses", [])
            from_address = self.config.get("from_address", "alerts@cognitionos.ai")
            
            if not to_addresses:
                logger.warning("No email addresses configured")
                return False
            
            # Format email
            subject = f"[{alert.severity.upper()}] {alert.rule_name}"
            body = self._format_email_body(alert)
            
            # Send email (would integrate with actual email service)
            logger.info(f"Sending email alert to {to_addresses}: {subject}")
            
            # Placeholder for actual email sending
            # await email_service.send(to=to_addresses, subject=subject, body=body)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False
    
    def _format_email_body(self, alert: Any) -> str:
        """Format email body."""
        return f"""
Alert: {alert.rule_name}
Severity: {alert.severity}
State: {alert.state}
Message: {alert.message}

Started: {alert.starts_at}
Labels: {alert.labels}
Annotations: {alert.annotations}

Alert ID: {alert.alert_id}
"""


class SlackChannel(AlertChannelBase):
    """Slack alert channel."""
    
    async def send_alert(self, alert: Any) -> bool:
        """Send alert to Slack."""
        try:
            webhook_url = self.config.get("webhook_url")
            channel = self.config.get("channel", "#alerts")
            
            if not webhook_url:
                logger.warning("No Slack webhook URL configured")
                return False
            
            # Format Slack message
            message = self._format_slack_message(alert)
            
            # Send to Slack (would use actual Slack API)
            logger.info(f"Sending Slack alert to {channel}: {alert.rule_name}")
            
            # Placeholder for actual Slack sending
            # await slack_client.post_message(webhook_url, message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False
    
    def _format_slack_message(self, alert: Any) -> Dict[str, Any]:
        """Format Slack message payload."""
        # Color based on severity
        color_map = {
            "critical": "danger",
            "high": "warning",
            "medium": "warning",
            "low": "#439FE0",
            "info": "good",
        }
        
        color = color_map.get(alert.severity, "#cccccc")
        
        return {
            "attachments": [
                {
                    "color": color,
                    "title": f"{alert.severity.upper()}: {alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "State", "value": alert.state, "short": True},
                        {"title": "Started", "value": str(alert.starts_at), "short": True},
                    ],
                    "footer": f"Alert ID: {alert.alert_id}",
                }
            ]
        }


class PagerDutyChannel(AlertChannelBase):
    """PagerDuty alert channel."""
    
    async def send_alert(self, alert: Any) -> bool:
        """Send alert to PagerDuty."""
        try:
            integration_key = self.config.get("integration_key")
            
            if not integration_key:
                logger.warning("No PagerDuty integration key configured")
                return False
            
            # Only send critical/high severity to PagerDuty
            if alert.severity not in ["critical", "high"]:
                logger.debug(f"Skipping PagerDuty for {alert.severity} alert")
                return True
            
            # Format PagerDuty event
            event = self._format_pagerduty_event(alert)
            
            logger.info(f"Sending PagerDuty alert: {alert.rule_name}")
            
            # Placeholder for actual PagerDuty API call
            # await pagerduty_client.send_event(integration_key, event)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending PagerDuty alert: {e}")
            return False
    
    def _format_pagerduty_event(self, alert: Any) -> Dict[str, Any]:
        """Format PagerDuty event payload."""
        return {
            "routing_key": self.config.get("integration_key"),
            "event_action": "trigger" if alert.state == "firing" else "resolve",
            "dedup_key": alert.alert_id,
            "payload": {
                "summary": f"{alert.rule_name}: {alert.message}",
                "severity": alert.severity,
                "source": "CognitionOS",
                "timestamp": alert.starts_at.isoformat(),
                "custom_details": {
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                },
            },
        }


class WebhookChannel(AlertChannelBase):
    """Generic webhook alert channel."""
    
    async def send_alert(self, alert: Any) -> bool:
        """Send alert to webhook."""
        try:
            url = self.config.get("url")
            
            if not url:
                logger.warning("No webhook URL configured")
                return False
            
            # Format payload
            payload = {
                "alert_id": alert.alert_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity,
                "state": alert.state,
                "message": alert.message,
                "starts_at": alert.starts_at.isoformat(),
                "labels": alert.labels,
                "annotations": alert.annotations,
            }
            
            logger.info(f"Sending webhook alert to {url}")
            
            # Placeholder for actual HTTP POST
            # async with httpx.AsyncClient() as client:
            #     await client.post(url, json=payload)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            return False


class AlertRouter:
    """
    Alert routing system for multi-channel notification delivery.
    
    Features:
    - Multiple channel support (email, Slack, PagerDuty, webhooks)
    - Severity-based routing
    - Channel-specific formatting
    - Retry logic for failed deliveries
    - Rate limiting to prevent spam
    """
    
    def __init__(self):
        """Initialize alert router."""
        self.channels: Dict[AlertChannel, AlertChannelBase] = {}
        self.routing_rules: List[Dict[str, Any]] = []
        
        # Rate limiting
        self.alert_counts: Dict[str, int] = {}
        self.rate_limit_window = 300  # 5 minutes
        self.max_alerts_per_window = 10
        
        logger.info("Alert router initialized")
    
    def register_channel(self, channel_type: AlertChannel, config: Dict[str, Any]):
        """
        Register an alert channel.
        
        Args:
            channel_type: Type of channel
            config: Channel configuration
        """
        # Create channel instance
        if channel_type == AlertChannel.EMAIL:
            channel = EmailChannel(config)
        elif channel_type == AlertChannel.SLACK:
            channel = SlackChannel(config)
        elif channel_type == AlertChannel.PAGERDUTY:
            channel = PagerDutyChannel(config)
        elif channel_type == AlertChannel.WEBHOOK:
            channel = WebhookChannel(config)
        else:
            logger.warning(f"Unknown channel type: {channel_type}")
            return
        
        self.channels[channel_type] = channel
        logger.info(f"Registered alert channel: {channel_type}")
    
    def add_routing_rule(
        self,
        name: str,
        channels: List[AlertChannel],
        severity_filter: Optional[List[str]] = None,
        label_filter: Optional[Dict[str, str]] = None,
    ):
        """
        Add routing rule.
        
        Args:
            name: Rule name
            channels: Channels to route to
            severity_filter: Only route these severities
            label_filter: Only route alerts with these labels
        """
        rule = {
            "name": name,
            "channels": channels,
            "severity_filter": severity_filter,
            "label_filter": label_filter or {},
        }
        
        self.routing_rules.append(rule)
        logger.info(f"Added routing rule: {name}")
    
    async def route_alert(self, alert: Any):
        """
        Route alert to appropriate channels.
        
        Args:
            alert: Alert to route
        """
        # Check rate limiting
        if self._is_rate_limited(alert):
            logger.warning(f"Alert rate limited: {alert.alert_id}")
            return
        
        # Find matching routing rules
        matching_rules = self._find_matching_rules(alert)
        
        if not matching_rules:
            logger.debug(f"No routing rules matched for alert: {alert.alert_id}")
            return
        
        # Collect unique channels
        channels_to_notify = set()
        for rule in matching_rules:
            channels_to_notify.update(rule["channels"])
        
        # Send to each channel
        tasks = []
        for channel_type in channels_to_notify:
            if channel_type in self.channels:
                task = self._send_with_retry(channel_type, alert)
                tasks.append(task)
        
        # Wait for all sends to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if r is True)
            logger.info(f"Alert routed to {success_count}/{len(tasks)} channels")
    
    async def _send_with_retry(
        self,
        channel_type: AlertChannel,
        alert: Any,
        max_retries: int = 3,
    ) -> bool:
        """Send alert with retry logic."""
        channel = self.channels[channel_type]
        
        for attempt in range(max_retries):
            try:
                success = await channel.send_alert(alert)
                if success:
                    return True
                
                # Wait before retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Error sending alert via {channel_type} (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        logger.error(f"Failed to send alert via {channel_type} after {max_retries} attempts")
        return False
    
    def _find_matching_rules(self, alert: Any) -> List[Dict[str, Any]]:
        """Find routing rules that match the alert."""
        matching = []
        
        for rule in self.routing_rules:
            # Check severity filter
            if rule["severity_filter"]:
                if alert.severity not in rule["severity_filter"]:
                    continue
            
            # Check label filter
            if rule["label_filter"]:
                if not all(
                    alert.labels.get(k) == v
                    for k, v in rule["label_filter"].items()
                ):
                    continue
            
            matching.append(rule)
        
        return matching
    
    def _is_rate_limited(self, alert: Any) -> bool:
        """Check if alert should be rate limited."""
        key = f"{alert.rule_name}:{alert.severity}"
        
        # Get current count
        current_count = self.alert_counts.get(key, 0)
        
        # Increment
        self.alert_counts[key] = current_count + 1
        
        # Check limit
        if current_count >= self.max_alerts_per_window:
            return True
        
        # Reset counts periodically (simplified)
        # In production, would use time-based sliding window
        if current_count > self.max_alerts_per_window * 2:
            self.alert_counts[key] = 0
        
        return False
