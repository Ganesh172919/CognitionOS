"""
Real-Time WebSocket Manager

Production-grade WebSocket connection management with rooms,
broadcasts, presence tracking, rate limiting, and tenant isolation.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


class MessageType(str, Enum):
    EVENT = "event"
    BROADCAST = "broadcast"
    DIRECT = "direct"
    SYSTEM = "system"
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PRESENCE = "presence"
    ERROR = "error"


@dataclass
class WSMessage:
    """WebSocket message container."""

    type: MessageType
    channel: str = ""
    event: str = ""
    data: Any = None
    sender_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "channel": self.channel,
            "event": self.event,
            "data": self.data,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
        })


@dataclass
class WSConnection:
    """Represents a single WebSocket connection."""

    connection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    state: ConnectionState = ConnectionState.CONNECTING
    websocket: Any = None  # The actual WebSocket object
    channels: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_ping: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0
    rate_limit_window_start: float = field(default_factory=time.monotonic)
    rate_limit_count: int = 0

    @property
    def is_authenticated(self) -> bool:
        return self.state == ConnectionState.AUTHENTICATED and self.user_id is not None


class RateLimiter:
    """Per-connection rate limiting."""

    def __init__(self, max_per_second: int = 50, max_per_minute: int = 500):
        self.max_per_second = max_per_second
        self.max_per_minute = max_per_minute
        self._minute_counts: Dict[str, List[float]] = defaultdict(list)

    def check(self, connection: WSConnection) -> bool:
        now = time.monotonic()

        # Per-second check
        elapsed = now - connection.rate_limit_window_start
        if elapsed < 1.0:
            if connection.rate_limit_count >= self.max_per_second:
                return False
            connection.rate_limit_count += 1
        else:
            connection.rate_limit_window_start = now
            connection.rate_limit_count = 1

        # Per-minute check
        history = self._minute_counts[connection.connection_id]
        history.append(now)
        # Prune older than 60s
        cutoff = now - 60
        self._minute_counts[connection.connection_id] = [t for t in history if t > cutoff]
        if len(self._minute_counts[connection.connection_id]) > self.max_per_minute:
            return False

        return True


@dataclass
class Channel:
    """A pub/sub channel (room)."""

    name: str
    tenant_id: Optional[str] = None
    is_private: bool = False
    max_members: int = 10_000
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    members: Set[str] = field(default_factory=set)  # connection_ids

    @property
    def member_count(self) -> int:
        return len(self.members)


class PresenceTracker:
    """Tracks user presence across connections."""

    def __init__(self):
        self._user_connections: Dict[str, Set[str]] = defaultdict(set)
        self._connection_users: Dict[str, str] = {}
        self._user_status: Dict[str, str] = {}

    def track(self, connection_id: str, user_id: str) -> None:
        self._user_connections[user_id].add(connection_id)
        self._connection_users[connection_id] = user_id
        self._user_status[user_id] = "online"

    def untrack(self, connection_id: str) -> Optional[str]:
        user_id = self._connection_users.pop(connection_id, None)
        if user_id:
            self._user_connections[user_id].discard(connection_id)
            if not self._user_connections[user_id]:
                self._user_status[user_id] = "offline"
                del self._user_connections[user_id]
        return user_id

    def get_online_users(self, user_ids: Optional[List[str]] = None) -> Dict[str, str]:
        if user_ids:
            return {uid: self._user_status.get(uid, "offline") for uid in user_ids}
        return dict(self._user_status)

    def is_online(self, user_id: str) -> bool:
        return self._user_status.get(user_id) == "online"

    def get_connection_count(self, user_id: str) -> int:
        return len(self._user_connections.get(user_id, set()))


class WebSocketManager:
    """
    Production WebSocket connection manager.

    Features:
    - Connection lifecycle management
    - Channel-based pub/sub (rooms)
    - Presence tracking
    - Rate limiting
    - Tenant isolation
    - Broadcast / direct messaging
    - Heartbeat monitoring
    - Message history per channel
    """

    def __init__(
        self,
        heartbeat_interval: float = 30.0,
        connection_timeout: float = 120.0,
        max_connections: int = 50_000,
        rate_limit_per_second: int = 50,
        rate_limit_per_minute: int = 500,
        message_history_size: int = 100,
    ):
        self._connections: Dict[str, WSConnection] = {}
        self._channels: Dict[str, Channel] = {}
        self._presence = PresenceTracker()
        self._rate_limiter = RateLimiter(rate_limit_per_second, rate_limit_per_minute)

        self._heartbeat_interval = heartbeat_interval
        self._connection_timeout = connection_timeout
        self._max_connections = max_connections
        self._message_history_size = message_history_size
        self._message_history: Dict[str, List[WSMessage]] = defaultdict(list)

        # Event hooks
        self._on_connect_hooks: List[Callable] = []
        self._on_disconnect_hooks: List[Callable] = []
        self._on_message_hooks: List[Callable] = []
        self._on_subscribe_hooks: List[Callable] = []

        # Metrics
        self._total_messages_sent = 0
        self._total_messages_received = 0
        self._total_connections = 0
        self._total_disconnections = 0
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket manager started")

    async def stop(self) -> None:
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        # Close all connections
        for conn in list(self._connections.values()):
            await self.disconnect(conn.connection_id)
        logger.info("WebSocket manager stopped")

    # -- Connection management ----------------------------------------------

    async def connect(
        self,
        websocket: Any,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WSConnection:
        if len(self._connections) >= self._max_connections:
            raise RuntimeError("Maximum connections reached")

        conn = WSConnection(
            websocket=websocket,
            user_id=user_id,
            tenant_id=tenant_id,
            state=ConnectionState.CONNECTED if not user_id else ConnectionState.AUTHENTICATED,
            metadata=metadata or {},
        )
        self._connections[conn.connection_id] = conn
        self._total_connections += 1

        if user_id:
            self._presence.track(conn.connection_id, user_id)

        # Run hooks
        for hook in self._on_connect_hooks:
            try:
                await hook(conn)
            except Exception:
                logger.exception("Connect hook error")

        # Send welcome message
        await self._send_to_connection(conn, WSMessage(
            type=MessageType.SYSTEM,
            event="connected",
            data={
                "connection_id": conn.connection_id,
                "server_time": datetime.utcnow().isoformat(),
            },
        ))

        logger.debug("Client connected: %s (user=%s, tenant=%s)",
                      conn.connection_id, user_id, tenant_id)
        return conn

    async def disconnect(self, connection_id: str) -> None:
        conn = self._connections.pop(connection_id, None)
        if not conn:
            return

        conn.state = ConnectionState.DISCONNECTED
        self._total_disconnections += 1

        # Remove from all channels
        for channel_name in list(conn.channels):
            await self.unsubscribe(connection_id, channel_name)

        # Untrack presence
        user_id = self._presence.untrack(connection_id)

        # Broadcast presence update
        if user_id and not self._presence.is_online(user_id):
            await self._broadcast_presence(user_id, "offline")

        # Run hooks
        for hook in self._on_disconnect_hooks:
            try:
                await hook(conn)
            except Exception:
                logger.exception("Disconnect hook error")

        # Close WebSocket
        if conn.websocket:
            try:
                await conn.websocket.close()
            except Exception:
                pass

    async def authenticate(self, connection_id: str, user_id: str, tenant_id: Optional[str] = None) -> bool:
        conn = self._connections.get(connection_id)
        if not conn:
            return False

        conn.user_id = user_id
        conn.tenant_id = tenant_id
        conn.state = ConnectionState.AUTHENTICATED
        self._presence.track(connection_id, user_id)

        await self._broadcast_presence(user_id, "online")
        return True

    # -- Channel / room management ------------------------------------------

    def create_channel(
        self,
        name: str,
        tenant_id: Optional[str] = None,
        is_private: bool = False,
        max_members: int = 10_000,
    ) -> Channel:
        if name not in self._channels:
            self._channels[name] = Channel(
                name=name,
                tenant_id=tenant_id,
                is_private=is_private,
                max_members=max_members,
            )
        return self._channels[name]

    async def subscribe(self, connection_id: str, channel_name: str) -> bool:
        conn = self._connections.get(connection_id)
        if not conn:
            return False

        channel = self._channels.get(channel_name)
        if not channel:
            channel = self.create_channel(channel_name, tenant_id=conn.tenant_id)

        if channel.member_count >= channel.max_members:
            return False

        # Tenant isolation
        if channel.tenant_id and conn.tenant_id and channel.tenant_id != conn.tenant_id:
            logger.warning("Tenant mismatch for channel %s", channel_name)
            return False

        conn.channels.add(channel_name)
        channel.members.add(connection_id)

        for hook in self._on_subscribe_hooks:
            try:
                await hook(conn, channel_name)
            except Exception:
                logger.exception("Subscribe hook error")

        # Notify channel
        await self.broadcast_to_channel(
            channel_name,
            WSMessage(
                type=MessageType.SYSTEM,
                channel=channel_name,
                event="member_joined",
                data={
                    "user_id": conn.user_id,
                    "member_count": channel.member_count,
                },
            ),
            exclude={connection_id},
        )
        return True

    async def unsubscribe(self, connection_id: str, channel_name: str) -> bool:
        conn = self._connections.get(connection_id)
        if not conn:
            return False

        conn.channels.discard(channel_name)
        channel = self._channels.get(channel_name)
        if channel:
            channel.members.discard(connection_id)

            await self.broadcast_to_channel(
                channel_name,
                WSMessage(
                    type=MessageType.SYSTEM,
                    channel=channel_name,
                    event="member_left",
                    data={
                        "user_id": conn.user_id,
                        "member_count": channel.member_count,
                    },
                ),
                exclude={connection_id},
            )

            # Cleanup empty channels
            if channel.member_count == 0:
                del self._channels[channel_name]

        return True

    # -- Messaging ----------------------------------------------------------

    async def send_to_user(self, user_id: str, message: WSMessage) -> int:
        sent = 0
        for conn in self._connections.values():
            if conn.user_id == user_id and conn.state == ConnectionState.AUTHENTICATED:
                await self._send_to_connection(conn, message)
                sent += 1
        return sent

    async def send_to_connection(self, connection_id: str, message: WSMessage) -> bool:
        conn = self._connections.get(connection_id)
        if not conn:
            return False
        await self._send_to_connection(conn, message)
        return True

    async def broadcast_to_channel(
        self,
        channel_name: str,
        message: WSMessage,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        channel = self._channels.get(channel_name)
        if not channel:
            return 0

        message.channel = channel_name
        self._store_message(channel_name, message)

        sent = 0
        exclude = exclude or set()
        for conn_id in channel.members:
            if conn_id in exclude:
                continue
            conn = self._connections.get(conn_id)
            if conn and conn.state in (ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED):
                await self._send_to_connection(conn, message)
                sent += 1
        return sent

    async def broadcast_to_tenant(
        self, tenant_id: str, message: WSMessage, exclude: Optional[Set[str]] = None
    ) -> int:
        sent = 0
        exclude = exclude or set()
        for conn in self._connections.values():
            if conn.tenant_id == tenant_id and conn.connection_id not in exclude:
                await self._send_to_connection(conn, message)
                sent += 1
        return sent

    async def broadcast_all(
        self, message: WSMessage, exclude: Optional[Set[str]] = None
    ) -> int:
        sent = 0
        exclude = exclude or set()
        for conn in self._connections.values():
            if conn.connection_id not in exclude:
                await self._send_to_connection(conn, message)
                sent += 1
        return sent

    async def handle_message(self, connection_id: str, raw_data: str) -> None:
        conn = self._connections.get(connection_id)
        if not conn:
            return

        if not self._rate_limiter.check(conn):
            await self._send_to_connection(conn, WSMessage(
                type=MessageType.ERROR,
                event="rate_limited",
                data={"message": "Rate limit exceeded"},
            ))
            return

        conn.message_count += 1
        self._total_messages_received += 1

        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            await self._send_to_connection(conn, WSMessage(
                type=MessageType.ERROR,
                event="invalid_json",
                data={"message": "Invalid JSON"},
            ))
            return

        msg_type = data.get("type", "event")

        if msg_type == "ping":
            conn.last_ping = datetime.utcnow()
            await self._send_to_connection(conn, WSMessage(type=MessageType.PONG))
        elif msg_type == "subscribe":
            await self.subscribe(connection_id, data.get("channel", ""))
        elif msg_type == "unsubscribe":
            await self.unsubscribe(connection_id, data.get("channel", ""))
        elif msg_type == "event":
            message = WSMessage(
                type=MessageType.EVENT,
                channel=data.get("channel", ""),
                event=data.get("event", ""),
                data=data.get("data"),
                sender_id=conn.user_id,
            )
            if message.channel:
                await self.broadcast_to_channel(
                    message.channel, message, exclude={connection_id}
                )

        for hook in self._on_message_hooks:
            try:
                await hook(conn, data)
            except Exception:
                logger.exception("Message hook error")

    # -- Internal -----------------------------------------------------------

    async def _send_to_connection(self, conn: WSConnection, message: WSMessage) -> None:
        if conn.websocket and conn.state != ConnectionState.DISCONNECTED:
            try:
                await conn.websocket.send_text(message.to_json())
                self._total_messages_sent += 1
            except Exception:
                await self.disconnect(conn.connection_id)

    async def _broadcast_presence(self, user_id: str, status: str) -> None:
        message = WSMessage(
            type=MessageType.PRESENCE,
            event="presence_change",
            data={"user_id": user_id, "status": status},
        )
        # Send to all channels where the user had connections
        for channel in self._channels.values():
            has_user = any(
                self._connections.get(cid, WSConnection()).user_id == user_id
                for cid in channel.members
            )
            if has_user or status == "offline":
                await self.broadcast_to_channel(channel.name, message)

    def _store_message(self, channel_name: str, message: WSMessage) -> None:
        history = self._message_history[channel_name]
        history.append(message)
        if len(history) > self._message_history_size:
            self._message_history[channel_name] = history[-self._message_history_size:]

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            now = datetime.utcnow()
            stale_connections = []

            for conn_id, conn in self._connections.items():
                elapsed = (now - conn.last_ping).total_seconds()
                if elapsed > self._connection_timeout:
                    stale_connections.append(conn_id)
                elif elapsed > self._heartbeat_interval:
                    # Send ping
                    await self._send_to_connection(conn, WSMessage(type=MessageType.PING))

            for conn_id in stale_connections:
                logger.info("Disconnecting stale connection: %s", conn_id)
                await self.disconnect(conn_id)

    # -- Hook registration --------------------------------------------------

    def on_connect(self, fn: Callable) -> Callable:
        self._on_connect_hooks.append(fn)
        return fn

    def on_disconnect(self, fn: Callable) -> Callable:
        self._on_disconnect_hooks.append(fn)
        return fn

    def on_message(self, fn: Callable) -> Callable:
        self._on_message_hooks.append(fn)
        return fn

    def on_subscribe(self, fn: Callable) -> Callable:
        self._on_subscribe_hooks.append(fn)
        return fn

    # -- Queries ------------------------------------------------------------

    def get_channel_history(
        self, channel_name: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        msgs = self._message_history.get(channel_name, [])
        return [
            {
                "message_id": m.message_id,
                "type": m.type.value,
                "event": m.event,
                "data": m.data,
                "sender_id": m.sender_id,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in msgs[-limit:]
        ]

    def get_channel_members(self, channel_name: str) -> List[Dict[str, Any]]:
        channel = self._channels.get(channel_name)
        if not channel:
            return []
        result = []
        for conn_id in channel.members:
            conn = self._connections.get(conn_id)
            if conn:
                result.append({
                    "connection_id": conn_id,
                    "user_id": conn.user_id,
                    "connected_at": conn.connected_at.isoformat(),
                })
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_connections": len(self._connections),
            "authenticated_connections": sum(
                1 for c in self._connections.values() if c.is_authenticated
            ),
            "total_connections": self._total_connections,
            "total_disconnections": self._total_disconnections,
            "channels": len(self._channels),
            "messages_sent": self._total_messages_sent,
            "messages_received": self._total_messages_received,
            "online_users": len(self._presence.get_online_users()),
        }

    def get_connection(self, connection_id: str) -> Optional[WSConnection]:
        return self._connections.get(connection_id)
