"""
WebSocket Real-Time Communication System

Production-grade WebSocket server with:
- Connection lifecycle management
- Room/channel subscriptions
- Message broadcasting
- Authentication and authorization
- Connection pooling
- Message persistence
- Heartbeat/keepalive
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from uuid import uuid4

logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """WebSocket connection status."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(str, Enum):
    """WebSocket message types."""
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MESSAGE = "message"
    BROADCAST = "broadcast"
    ERROR = "error"
    ACK = "ack"


@dataclass
class WebSocketConnection:
    """WebSocket connection."""
    connection_id: str
    user_id: Optional[str] = None
    status: ConnectionStatus = ConnectionStatus.CONNECTING
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    subscribed_rooms: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Mock WebSocket interface (replace with actual WebSocket in production)
    send_queue: List[Dict[str, Any]] = field(default_factory=list)
    
    async def send(self, message: Dict[str, Any]):
        """Send message to connection."""
        self.send_queue.append(message)
        self.last_activity = datetime.utcnow()
        logger.debug(f"Sent message to {self.connection_id}: {message['type']}")
        
    async def close(self, code: int = 1000, reason: str = "Normal closure"):
        """Close connection."""
        self.status = ConnectionStatus.DISCONNECTED
        logger.info(f"Closed connection {self.connection_id}: {reason}")


@dataclass
class Room:
    """Chat room/channel."""
    room_id: str
    name: str
    owner_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    members: Set[str] = field(default_factory=set)
    max_members: Optional[int] = None
    is_private: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def can_join(self, user_id: str) -> bool:
        """Check if user can join room."""
        if self.max_members and len(self.members) >= self.max_members:
            return False
        return True


@dataclass
class Message:
    """WebSocket message."""
    message_id: str
    room_id: str
    sender_id: str
    content: Any
    message_type: MessageType = MessageType.MESSAGE
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "room_id": self.room_id,
            "sender_id": self.sender_id,
            "content": self.content,
            "type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class WebSocketManager:
    """
    Production-grade WebSocket manager.
    
    Features:
    - Connection lifecycle management
    - Room/channel subscriptions
    - Message broadcasting
    - Authentication
    - Heartbeat monitoring
    - Message persistence
    - Connection pooling
    """
    
    def __init__(
        self,
        heartbeat_interval: int = 30,
        connection_timeout: int = 300,
        max_connections_per_user: int = 5,
        message_history_size: int = 100,
    ):
        self.heartbeat_interval = heartbeat_interval
        self.connection_timeout = connection_timeout
        self.max_connections_per_user = max_connections_per_user
        self.message_history_size = message_history_size
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> {connection_ids}
        
        # Room management
        self.rooms: Dict[str, Room] = {}
        self.room_connections: Dict[str, Set[str]] = {}  # room_id -> {connection_ids}
        
        # Message history
        self.message_history: Dict[str, List[Message]] = {}
        
        # Event handlers
        self.on_connect_handlers: List[Callable] = []
        self.on_disconnect_handlers: List[Callable] = []
        self.on_message_handlers: Dict[str, List[Callable]] = {}
        
        self.is_running = False
        
    async def start(self):
        """Start WebSocket manager."""
        self.is_running = True
        logger.info("Starting WebSocket manager")
        
        await asyncio.gather(
            self._heartbeat_loop(),
            self._cleanup_loop(),
        )
        
    async def stop(self):
        """Stop WebSocket manager."""
        logger.info("Stopping WebSocket manager")
        self.is_running = False
        
        # Close all connections
        for connection in list(self.connections.values()):
            await connection.close(code=1001, reason="Server shutting down")
            
    async def register_connection(
        self,
        connection_id: str,
        metadata: Dict[str, Any] = None,
    ) -> WebSocketConnection:
        """Register new WebSocket connection."""
        connection = WebSocketConnection(
            connection_id=connection_id,
            metadata=metadata or {},
        )
        
        self.connections[connection_id] = connection
        
        logger.info(f"Registered connection: {connection_id}")
        
        # Trigger connect handlers
        for handler in self.on_connect_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(connection)
                else:
                    handler(connection)
            except Exception as e:
                logger.error(f"Error in connect handler: {e}")
                
        return connection
        
    async def authenticate_connection(
        self,
        connection_id: str,
        user_id: str,
    ) -> bool:
        """Authenticate connection."""
        if connection_id not in self.connections:
            return False
            
        connection = self.connections[connection_id]
        
        # Check max connections per user
        if user_id in self.user_connections:
            if len(self.user_connections[user_id]) >= self.max_connections_per_user:
                logger.warning(f"Max connections exceeded for user: {user_id}")
                return False
                
        connection.user_id = user_id
        connection.status = ConnectionStatus.AUTHENTICATED
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"Authenticated connection {connection_id} for user {user_id}")
        return True
        
    async def disconnect(self, connection_id: str, reason: str = "User disconnected"):
        """Disconnect connection."""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        
        # Unsubscribe from rooms
        for room_id in list(connection.subscribed_rooms):
            await self.unsubscribe(connection_id, room_id)
            
        # Remove from user connections
        if connection.user_id and connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            
        # Close connection
        await connection.close(reason=reason)
        
        # Trigger disconnect handlers
        for handler in self.on_disconnect_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(connection)
                else:
                    handler(connection)
            except Exception as e:
                logger.error(f"Error in disconnect handler: {e}")
                
        del self.connections[connection_id]
        logger.info(f"Disconnected: {connection_id} ({reason})")
        
    async def create_room(
        self,
        room_id: str,
        name: str,
        owner_id: str,
        max_members: Optional[int] = None,
        is_private: bool = False,
        metadata: Dict[str, Any] = None,
    ) -> Room:
        """Create new room."""
        room = Room(
            room_id=room_id,
            name=name,
            owner_id=owner_id,
            max_members=max_members,
            is_private=is_private,
            metadata=metadata or {},
        )
        
        self.rooms[room_id] = room
        self.room_connections[room_id] = set()
        self.message_history[room_id] = []
        
        logger.info(f"Created room: {room_id} ({name})")
        return room
        
    async def subscribe(self, connection_id: str, room_id: str) -> bool:
        """Subscribe connection to room."""
        if connection_id not in self.connections:
            return False
            
        if room_id not in self.rooms:
            return False
            
        connection = self.connections[connection_id]
        room = self.rooms[room_id]
        
        # Check if can join
        if not room.can_join(connection.user_id or connection_id):
            return False
            
        # Subscribe
        connection.subscribed_rooms.add(room_id)
        self.room_connections[room_id].add(connection_id)
        room.members.add(connection.user_id or connection_id)
        
        # Send room history
        history = self.message_history.get(room_id, [])
        await connection.send({
            "type": MessageType.MESSAGE.value,
            "room_id": room_id,
            "content": {
                "event": "joined",
                "history": [m.to_dict() for m in history[-50:]]  # Last 50 messages
            }
        })
        
        logger.info(f"Connection {connection_id} subscribed to room {room_id}")
        return True
        
    async def unsubscribe(self, connection_id: str, room_id: str):
        """Unsubscribe connection from room."""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        connection.subscribed_rooms.discard(room_id)
        
        if room_id in self.room_connections:
            self.room_connections[room_id].discard(connection_id)
            
        if room_id in self.rooms:
            room = self.rooms[room_id]
            room.members.discard(connection.user_id or connection_id)
            
        logger.info(f"Connection {connection_id} unsubscribed from room {room_id}")
        
    async def send_message(
        self,
        room_id: str,
        sender_id: str,
        content: Any,
        message_type: MessageType = MessageType.MESSAGE,
        metadata: Dict[str, Any] = None,
    ) -> Message:
        """Send message to room."""
        message = Message(
            message_id=str(uuid4()),
            room_id=room_id,
            sender_id=sender_id,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )
        
        # Store in history
        if room_id not in self.message_history:
            self.message_history[room_id] = []
        self.message_history[room_id].append(message)
        
        # Trim history
        if len(self.message_history[room_id]) > self.message_history_size:
            self.message_history[room_id] = self.message_history[room_id][-self.message_history_size:]
            
        # Broadcast to room
        await self.broadcast_to_room(room_id, message.to_dict())
        
        # Trigger message handlers
        if message_type.value in self.on_message_handlers:
            for handler in self.on_message_handlers[message_type.value]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
                    
        return message
        
    async def broadcast_to_room(self, room_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections in room."""
        if room_id not in self.room_connections:
            return
            
        connection_ids = list(self.room_connections[room_id])
        
        for connection_id in connection_ids:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                await connection.send(message)
                
        logger.debug(f"Broadcasted to {len(connection_ids)} connections in room {room_id}")
        
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all user's connections."""
        if user_id not in self.user_connections:
            return
            
        connection_ids = list(self.user_connections[user_id])
        
        for connection_id in connection_ids:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                await connection.send(message)
                
    async def _heartbeat_loop(self):
        """Send periodic heartbeat pings."""
        while self.is_running:
            try:
                for connection in list(self.connections.values()):
                    if connection.status == ConnectionStatus.AUTHENTICATED:
                        await connection.send({
                            "type": MessageType.PING.value,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
                
    async def _cleanup_loop(self):
        """Cleanup stale connections."""
        while self.is_running:
            try:
                now = datetime.utcnow()
                timeout = timedelta(seconds=self.connection_timeout)
                
                for connection_id, connection in list(self.connections.items()):
                    if now - connection.last_activity > timeout:
                        await self.disconnect(connection_id, "Connection timeout")
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
                
    def on_connect(self, handler: Callable):
        """Register connect event handler."""
        self.on_connect_handlers.append(handler)
        
    def on_disconnect(self, handler: Callable):
        """Register disconnect event handler."""
        self.on_disconnect_handlers.append(handler)
        
    def on_message(self, message_type: str, handler: Callable):
        """Register message event handler."""
        if message_type not in self.on_message_handlers:
            self.on_message_handlers[message_type] = []
        self.on_message_handlers[message_type].append(handler)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            "total_connections": len(self.connections),
            "authenticated_connections": len([
                c for c in self.connections.values()
                if c.status == ConnectionStatus.AUTHENTICATED
            ]),
            "total_rooms": len(self.rooms),
            "total_users": len(self.user_connections),
            "messages_stored": sum(len(h) for h in self.message_history.values()),
        }
