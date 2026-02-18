"""
WebSocket Support for Real-Time Updates

Provides WebSocket endpoints for streaming real-time workflow status,
event notifications, and live updates to clients.
"""

import os
import json
import asyncio
from typing import Set, Dict, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from fastapi.responses import HTMLResponse

from services.api.src.auth import get_current_user, CurrentUser
from infrastructure.observability import get_logger


logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.
    
    Handles connection lifecycle, authentication, and message broadcasting.
    """
    
    def __init__(self):
        # Active connections by user_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
        # Connections by workflow_id for targeted updates
        self.workflow_subscriptions: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        
        logger.info(
            "WebSocket connection established",
            extra={"user_id": user_id, "total_connections": self._count_connections()}
        )
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove a WebSocket connection"""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            
            # Clean up empty sets
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Remove from workflow subscriptions
        for workflow_id, connections in list(self.workflow_subscriptions.items()):
            connections.discard(websocket)
            if not connections:
                del self.workflow_subscriptions[workflow_id]
        
        logger.info(
            "WebSocket connection closed",
            extra={"user_id": user_id, "total_connections": self._count_connections()}
        )
    
    async def subscribe_to_workflow(self, websocket: WebSocket, workflow_id: str):
        """Subscribe a connection to workflow-specific updates"""
        if workflow_id not in self.workflow_subscriptions:
            self.workflow_subscriptions[workflow_id] = set()
        
        self.workflow_subscriptions[workflow_id].add(websocket)
        
        logger.info(
            "Subscribed to workflow updates",
            extra={"workflow_id": workflow_id}
        )
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def send_to_user(self, message: dict, user_id: str):
        """Send a message to all connections of a specific user"""
        if user_id not in self.active_connections:
            return
        
        disconnected = set()
        
        for connection in self.active_connections[user_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to user {user_id}: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection, user_id)
    
    async def broadcast_to_workflow(self, message: dict, workflow_id: str):
        """Broadcast a message to all subscribers of a workflow"""
        if workflow_id not in self.workflow_subscriptions:
            return
        
        disconnected = set()
        
        for connection in self.workflow_subscriptions[workflow_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to workflow {workflow_id}: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            # Find user_id for cleanup (inefficient, but works)
            for user_id, connections in self.active_connections.items():
                if connection in connections:
                    self.disconnect(connection, user_id)
                    break
    
    async def broadcast_all(self, message: dict):
        """Broadcast a message to all active connections"""
        total_sent = 0
        
        for user_id, connections in list(self.active_connections.items()):
            await self.send_to_user(message, user_id)
            total_sent += len(connections)
        
        logger.info(f"Broadcasted to {total_sent} connections")
    
    def _count_connections(self) -> int:
        """Count total active connections"""
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": self._count_connections(),
            "total_users": len(self.active_connections),
            "workflow_subscriptions": len(self.workflow_subscriptions),
            "workflows": list(self.workflow_subscriptions.keys()),
        }


# Global connection manager
manager = ConnectionManager()


def create_message(
    message_type: str,
    data: Any,
    workflow_id: str = None,
    execution_id: str = None,
) -> dict:
    """
    Create a standardized WebSocket message.
    
    Args:
        message_type: Type of message (workflow_status, event, notification, etc.)
        data: Message payload
        workflow_id: Optional workflow ID
        execution_id: Optional execution ID
    
    Returns:
        Formatted message dict
    """
    message = {
        "type": message_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if workflow_id:
        message["workflow_id"] = workflow_id
    
    if execution_id:
        message["execution_id"] = execution_id
    
    return message


async def send_workflow_status_update(
    workflow_id: str,
    execution_id: str,
    status: str,
    progress: int = None,
    current_step: str = None,
    message: str = None,
):
    """
    Send workflow status update to all subscribers.
    
    Args:
        workflow_id: Workflow ID
        execution_id: Execution ID
        status: Current status (running, completed, failed, etc.)
        progress: Optional progress percentage (0-100)
        current_step: Optional current step ID
        message: Optional status message
    """
    data = {
        "status": status,
        "progress": progress,
        "current_step": current_step,
        "message": message,
    }
    
    msg = create_message(
        "workflow_status",
        data,
        workflow_id=workflow_id,
        execution_id=execution_id
    )
    
    await manager.broadcast_to_workflow(msg, workflow_id)


async def send_event_notification(
    event_type: str,
    event_data: dict,
    user_id: str = None,
):
    """
    Send event notification.
    
    Args:
        event_type: Type of event
        event_data: Event payload
        user_id: Optional user to send to (otherwise broadcast)
    """
    msg = create_message("event", {
        "event_type": event_type,
        "data": event_data,
    })
    
    if user_id:
        await manager.send_to_user(msg, user_id)
    else:
        await manager.broadcast_all(msg)


async def send_system_notification(
    notification_type: str,
    title: str,
    message: str,
    user_id: str = None,
):
    """
    Send system notification.
    
    Args:
        notification_type: Type (info, warning, error, success)
        title: Notification title
        message: Notification message
        user_id: Optional user to send to
    """
    msg = create_message("notification", {
        "notification_type": notification_type,
        "title": title,
        "message": message,
    })
    
    if user_id:
        await manager.send_to_user(msg, user_id)
    else:
        await manager.broadcast_all(msg)


# Export connection manager and utilities
__all__ = [
    "ConnectionManager",
    "manager",
    "create_message",
    "send_workflow_status_update",
    "send_event_notification",
    "send_system_notification",
]
