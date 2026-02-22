"""
Real-Time Event Streaming System

WebSocket & SSE event distribution with:
- Multi-protocol support (WebSocket, SSE, long-polling)
- Topic-based pub/sub
- Connection pooling and management
- Message filtering and routing
- Backpressure handling
- Connection recovery
- Event replay and persistence
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
import uuid
import json
import asyncio
from collections import deque


class EventProtocol(Enum):
    """Event streaming protocols"""
    WEBSOCKET = "websocket"
    SSE = "sse"
    LONG_POLLING = "long_polling"


class ConnectionState(Enum):
    """Connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class StreamEvent:
    """Event message"""
    event_id: str
    event_type: str
    topic: str
    payload: Dict[str, Any]
    priority: EventPriority
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: Optional[int] = None


@dataclass
class Connection:
    """Client connection"""
    connection_id: str
    client_id: str
    protocol: EventProtocol
    state: ConnectionState
    subscribed_topics: Set[str]
    connected_at: datetime
    last_activity: datetime
    events_sent: int = 0
    events_received: int = 0
    message_queue: deque = field(default_factory=deque)
    max_queue_size: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Topic:
    """Event topic"""
    topic_name: str
    subscribers: Set[str]  # connection_ids
    total_events: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamStatistics:
    """Streaming statistics"""
    total_connections: int = 0
    active_connections: int = 0
    total_events: int = 0
    events_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    dropped_events: int = 0


class RealTimeEventStream:
    """
    Real-time event streaming system.

    Features:
    - Multi-protocol support
    - Topic-based pub/sub
    - Connection management
    - Event filtering
    - Backpressure handling
    - Message persistence
    """

    def __init__(
        self,
        max_connections: int = 10000,
        max_queue_size: int = 1000,
        event_ttl: int = 3600
    ):
        self.max_connections = max_connections
        self.max_queue_size = max_queue_size
        self.event_ttl = event_ttl

        # Connection management
        self.connections: Dict[str, Connection] = {}
        self.client_connections: Dict[str, Set[str]] = {}  # client_id -> connection_ids

        # Topic management
        self.topics: Dict[str, Topic] = {}

        # Event storage
        self.event_history: deque = deque(maxlen=10000)
        self.event_filters: Dict[str, Callable] = {}

        # Statistics
        self.stats = StreamStatistics()

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None

    async def connect(
        self,
        client_id: str,
        protocol: EventProtocol,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Establish new connection.

        Args:
            client_id: Client identifier
            protocol: Streaming protocol
            metadata: Connection metadata

        Returns:
            Connection ID
        """
        if len(self.connections) >= self.max_connections:
            raise ValueError("Maximum connections reached")

        connection_id = str(uuid.uuid4())

        connection = Connection(
            connection_id=connection_id,
            client_id=client_id,
            protocol=protocol,
            state=ConnectionState.CONNECTED,
            subscribed_topics=set(),
            connected_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            max_queue_size=self.max_queue_size,
            metadata=metadata or {}
        )

        self.connections[connection_id] = connection

        # Track by client
        if client_id not in self.client_connections:
            self.client_connections[client_id] = set()
        self.client_connections[client_id].add(connection_id)

        self.stats.total_connections += 1
        self.stats.active_connections += 1

        return connection_id

    async def disconnect(
        self,
        connection_id: str
    ) -> bool:
        """Disconnect connection"""
        connection = self.connections.get(connection_id)
        if not connection:
            return False

        connection.state = ConnectionState.DISCONNECTED

        # Unsubscribe from all topics
        for topic_name in list(connection.subscribed_topics):
            await self.unsubscribe(connection_id, topic_name)

        # Remove from client tracking
        client_id = connection.client_id
        if client_id in self.client_connections:
            self.client_connections[client_id].discard(connection_id)
            if not self.client_connections[client_id]:
                del self.client_connections[client_id]

        del self.connections[connection_id]
        self.stats.active_connections -= 1

        return True

    async def subscribe(
        self,
        connection_id: str,
        topic: str
    ) -> bool:
        """
        Subscribe connection to topic.

        Args:
            connection_id: Connection to subscribe
            topic: Topic name

        Returns:
            True if successful
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return False

        # Create topic if doesn't exist
        if topic not in self.topics:
            self.topics[topic] = Topic(
                topic_name=topic,
                subscribers=set()
            )

        # Subscribe
        self.topics[topic].subscribers.add(connection_id)
        connection.subscribed_topics.add(topic)

        return True

    async def unsubscribe(
        self,
        connection_id: str,
        topic: str
    ) -> bool:
        """Unsubscribe from topic"""
        connection = self.connections.get(connection_id)
        if not connection:
            return False

        if topic in self.topics:
            self.topics[topic].subscribers.discard(connection_id)

            # Remove empty topics
            if not self.topics[topic].subscribers:
                del self.topics[topic]

        connection.subscribed_topics.discard(topic)

        return True

    async def publish(
        self,
        topic: str,
        event_type: str,
        payload: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        ttl_seconds: Optional[int] = None
    ) -> str:
        """
        Publish event to topic.

        Args:
            topic: Topic name
            event_type: Event type
            payload: Event data
            priority: Event priority
            ttl_seconds: Time to live

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())

        event = StreamEvent(
            event_id=event_id,
            event_type=event_type,
            topic=topic,
            payload=payload,
            priority=priority,
            timestamp=datetime.utcnow(),
            ttl_seconds=ttl_seconds or self.event_ttl
        )

        # Store in history
        self.event_history.append(event)
        self.stats.total_events += 1

        # Update topic stats
        if topic in self.topics:
            self.topics[topic].total_events += 1

        # Deliver to subscribers
        await self._deliver_event(event)

        return event_id

    async def send_to_client(
        self,
        client_id: str,
        event_type: str,
        payload: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL
    ) -> int:
        """
        Send event directly to client (all connections).

        Args:
            client_id: Target client
            event_type: Event type
            payload: Event data
            priority: Event priority

        Returns:
            Number of connections delivered to
        """
        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            topic="_direct_",
            payload=payload,
            priority=priority,
            timestamp=datetime.utcnow()
        )

        delivered = 0

        connection_ids = self.client_connections.get(client_id, set())
        for connection_id in connection_ids:
            if await self._enqueue_event(connection_id, event):
                delivered += 1

        return delivered

    async def get_events(
        self,
        connection_id: str,
        timeout: float = 30.0
    ) -> List[StreamEvent]:
        """
        Get pending events for connection (long-polling).

        Args:
            connection_id: Connection ID
            timeout: Wait timeout in seconds

        Returns:
            List of events
        """
        connection = self.connections.get(connection_id)
        if not connection:
            return []

        connection.last_activity = datetime.utcnow()

        # Wait for events with timeout
        start_time = asyncio.get_event_loop().time()

        while True:
            if connection.message_queue:
                events = list(connection.message_queue)
                connection.message_queue.clear()
                return events

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return []

            await asyncio.sleep(0.1)

    async def replay_events(
        self,
        connection_id: str,
        topic: str,
        since: datetime
    ) -> int:
        """
        Replay historical events to connection.

        Args:
            connection_id: Target connection
            topic: Topic to replay
            since: Replay events since this time

        Returns:
            Number of events replayed
        """
        replayed = 0

        for event in self.event_history:
            if event.topic == topic and event.timestamp >= since:
                if await self._enqueue_event(connection_id, event):
                    replayed += 1

        return replayed

    def register_filter(
        self,
        filter_name: str,
        filter_func: Callable[[StreamEvent], bool]
    ) -> None:
        """Register event filter function"""
        self.event_filters[filter_name] = filter_func

    def get_connection_info(
        self,
        connection_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get connection information"""
        connection = self.connections.get(connection_id)
        if not connection:
            return None

        return {
            "connection_id": connection.connection_id,
            "client_id": connection.client_id,
            "protocol": connection.protocol.value,
            "state": connection.state.value,
            "subscribed_topics": list(connection.subscribed_topics),
            "connected_at": connection.connected_at.isoformat(),
            "last_activity": connection.last_activity.isoformat(),
            "events_sent": connection.events_sent,
            "queue_size": len(connection.message_queue),
            "uptime_seconds": (datetime.utcnow() - connection.connected_at).total_seconds()
        }

    def get_topic_info(
        self,
        topic: str
    ) -> Optional[Dict[str, Any]]:
        """Get topic information"""
        topic_obj = self.topics.get(topic)
        if not topic_obj:
            return None

        return {
            "topic_name": topic_obj.topic_name,
            "subscriber_count": len(topic_obj.subscribers),
            "total_events": topic_obj.total_events,
            "created_at": topic_obj.created_at.isoformat()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            "total_connections": self.stats.total_connections,
            "active_connections": self.stats.active_connections,
            "total_events": self.stats.total_events,
            "events_per_second": self.stats.events_per_second,
            "avg_latency_ms": self.stats.avg_latency_ms,
            "dropped_events": self.stats.dropped_events,
            "total_topics": len(self.topics),
            "active_topics": len([t for t in self.topics.values() if t.subscribers])
        }

    # Private helper methods

    async def _deliver_event(
        self,
        event: StreamEvent
    ) -> None:
        """Deliver event to all subscribers"""
        topic = self.topics.get(event.topic)
        if not topic:
            return

        for connection_id in topic.subscribers:
            await self._enqueue_event(connection_id, event)

    async def _enqueue_event(
        self,
        connection_id: str,
        event: StreamEvent
    ) -> bool:
        """Enqueue event to connection"""
        connection = self.connections.get(connection_id)
        if not connection or connection.state != ConnectionState.CONNECTED:
            return False

        # Apply filters
        for filter_func in self.event_filters.values():
            if not filter_func(event):
                return False

        # Check backpressure
        if len(connection.message_queue) >= connection.max_queue_size:
            # Drop lowest priority message
            dropped = self._drop_lowest_priority(connection)
            if not dropped:
                self.stats.dropped_events += 1
                return False

        # Enqueue
        connection.message_queue.append(event)
        connection.events_sent += 1

        return True

    def _drop_lowest_priority(
        self,
        connection: Connection
    ) -> bool:
        """Drop lowest priority message from queue"""
        if not connection.message_queue:
            return False

        # Find lowest priority
        lowest_priority = EventPriority.CRITICAL.value
        lowest_idx = -1

        for i, event in enumerate(connection.message_queue):
            if event.priority.value > lowest_priority:
                lowest_priority = event.priority.value
                lowest_idx = i

        if lowest_idx >= 0:
            del connection.message_queue[lowest_idx]
            return True

        return False

    async def cleanup_stale_connections(self) -> int:
        """Clean up stale connections"""
        stale_timeout = timedelta(minutes=5)
        now = datetime.utcnow()
        stale_connections = []

        for connection_id, connection in self.connections.items():
            if connection.state == ConnectionState.CONNECTED:
                if now - connection.last_activity > stale_timeout:
                    stale_connections.append(connection_id)

        for connection_id in stale_connections:
            await self.disconnect(connection_id)

        return len(stale_connections)

    async def cleanup_old_events(self) -> int:
        """Clean up expired events from history"""
        now = datetime.utcnow()
        cleaned = 0

        # Remove expired events
        while self.event_history:
            event = self.event_history[0]
            if event.ttl_seconds:
                age = (now - event.timestamp).total_seconds()
                if age > event.ttl_seconds:
                    self.event_history.popleft()
                    cleaned += 1
                else:
                    break
            else:
                break

        return cleaned
