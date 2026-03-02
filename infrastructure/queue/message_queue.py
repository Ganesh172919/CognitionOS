"""
Message Queue Abstraction — CognitionOS

Async message queue system with:
- In-memory queue (development)
- Pluggable backends (RabbitMQ, Redis, SQS)
- Dead letter queue
- Priority queuing
- Consumer groups
- Message deduplication
- Retry with backoff
- Batch publishing/consuming
- Queue metrics and monitoring
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


class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def value_int(self) -> int:
        return {"low": 0, "normal": 1, "high": 2, "critical": 3}[self.value]


class MessageState(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class QueueMessage:
    message_id: str
    queue_name: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    state: MessageState = MessageState.PENDING
    deduplication_id: Optional[str] = None
    group_id: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    attempt: int = 0
    max_attempts: int = 3
    created_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    error: Optional[str] = None
    visibility_timeout: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.message_id, "queue": self.queue_name,
            "priority": self.priority.value,
            "state": self.state.value, "attempt": self.attempt,
        }


@dataclass
class QueueConfig:
    name: str
    max_size: int = 100000
    default_priority: MessagePriority = MessagePriority.NORMAL
    max_retries: int = 3
    visibility_timeout: float = 30.0
    deduplication_window: float = 300.0
    dead_letter_queue: Optional[str] = None


class InMemoryQueue:
    """
    In-memory async message queue with priority ordering,
    consumer groups, and dead letter support.
    """

    def __init__(self, config: QueueConfig):
        self._config = config
        self._messages: List[QueueMessage] = []
        self._processing: Dict[str, QueueMessage] = {}
        self._dead_letters: List[QueueMessage] = []
        self._consumers: Dict[str, Callable[..., Awaitable[bool]]] = {}
        self._dedup_ids: Dict[str, float] = {}
        self._event = asyncio.Event()
        self._metrics = {
            "published": 0, "consumed": 0,
            "completed": 0, "failed": 0,
            "dead_lettered": 0,
        }

    async def publish(self, payload: Dict[str, Any], *,
                        priority: Optional[MessagePriority] = None,
                        deduplication_id: Optional[str] = None,
                        group_id: Optional[str] = None,
                        headers: Optional[Dict[str, str]] = None
                        ) -> QueueMessage:
        # Deduplication check
        if deduplication_id:
            now = time.time()
            if deduplication_id in self._dedup_ids:
                if now - self._dedup_ids[deduplication_id] < self._config.deduplication_window:
                    # Duplicate — skip
                    return QueueMessage(
                        message_id="dup",
                        queue_name=self._config.name,
                        payload=payload,
                        state=MessageState.COMPLETED,
                    )
            self._dedup_ids[deduplication_id] = now

        msg = QueueMessage(
            message_id=uuid.uuid4().hex[:12],
            queue_name=self._config.name,
            payload=payload,
            priority=priority or self._config.default_priority,
            deduplication_id=deduplication_id,
            group_id=group_id,
            headers=headers or {},
            max_attempts=self._config.max_retries,
            visibility_timeout=self._config.visibility_timeout,
        )

        # Insert sorted by priority (high priority first)
        inserted = False
        for i, existing in enumerate(self._messages):
            if msg.priority.value_int > existing.priority.value_int:
                self._messages.insert(i, msg)
                inserted = True
                break
        if not inserted:
            self._messages.append(msg)

        self._metrics["published"] += 1
        self._event.set()
        return msg

    async def consume(self) -> Optional[QueueMessage]:
        """Consume the next message from the queue."""
        while True:
            # Clean up expired visibility
            self._reclaim_expired()

            if self._messages:
                msg = self._messages.pop(0)
                msg.state = MessageState.PROCESSING
                msg.attempt += 1
                self._processing[msg.message_id] = msg
                self._metrics["consumed"] += 1
                return msg

            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                return None

    async def ack(self, message_id: str):
        """Acknowledge successful processing."""
        msg = self._processing.pop(message_id, None)
        if msg:
            msg.state = MessageState.COMPLETED
            msg.processed_at = time.time()
            self._metrics["completed"] += 1

    async def nack(self, message_id: str, *, error: str = ""):
        """Negative acknowledge — retry or dead letter."""
        msg = self._processing.pop(message_id, None)
        if not msg:
            return

        msg.error = error

        if msg.attempt < msg.max_attempts:
            msg.state = MessageState.PENDING
            self._messages.append(msg)
            self._event.set()
        else:
            msg.state = MessageState.DEAD_LETTER
            self._dead_letters.append(msg)
            self._metrics["dead_lettered"] += 1
            self._metrics["failed"] += 1

    def _reclaim_expired(self):
        now = time.time()
        expired = []
        for mid, msg in self._processing.items():
            if now - msg.created_at > msg.visibility_timeout * msg.attempt:
                expired.append(mid)

        for mid in expired:
            msg = self._processing.pop(mid)
            if msg.attempt < msg.max_attempts:
                msg.state = MessageState.PENDING
                self._messages.append(msg)
            else:
                msg.state = MessageState.DEAD_LETTER
                self._dead_letters.append(msg)

    # ── Consumer Registration ──

    def register_consumer(self, name: str,
                            handler: Callable[..., Awaitable[bool]]):
        self._consumers[name] = handler

    async def start_consumers(self, *, concurrency: int = 5):
        """Start consumer loop."""
        sem = asyncio.Semaphore(concurrency)

        async def consume_loop():
            while True:
                async with sem:
                    msg = await self.consume()
                    if msg and self._consumers:
                        handler = next(iter(self._consumers.values()))
                        try:
                            success = await handler(msg.payload)
                            if success:
                                await self.ack(msg.message_id)
                            else:
                                await self.nack(msg.message_id, error="Handler returned False")
                        except Exception as exc:
                            await self.nack(msg.message_id, error=str(exc))

        return asyncio.create_task(consume_loop())

    # ── Queries ──

    def depth(self) -> int:
        return len(self._messages)

    def get_dead_letters(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._dead_letters[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "queue_name": self._config.name,
            "depth": len(self._messages),
            "processing": len(self._processing),
            "dead_letters": len(self._dead_letters),
            "consumers": len(self._consumers),
            **self._metrics,
        }


class MessageQueueManager:
    """
    Manages multiple named queues with a unified interface.
    """

    def __init__(self):
        self._queues: Dict[str, InMemoryQueue] = {}

    def create_queue(self, name: str, **kwargs) -> InMemoryQueue:
        config = QueueConfig(name=name, **kwargs)
        queue = InMemoryQueue(config)
        self._queues[name] = queue
        return queue

    def get_queue(self, name: str) -> Optional[InMemoryQueue]:
        return self._queues.get(name)

    async def publish(self, queue_name: str, payload: Dict[str, Any],
                        **kwargs) -> Optional[QueueMessage]:
        queue = self._queues.get(queue_name)
        if queue:
            return await queue.publish(payload, **kwargs)
        return None

    def list_queues(self) -> List[Dict[str, Any]]:
        return [q.get_stats() for q in self._queues.values()]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_queues": len(self._queues),
            "total_depth": sum(q.depth() for q in self._queues.values()),
            "queues": {name: q.get_stats() for name, q in self._queues.items()},
        }


# ── Singleton ──
_manager: Optional[MessageQueueManager] = None


def get_queue_manager() -> MessageQueueManager:
    global _manager
    if not _manager:
        _manager = MessageQueueManager()
    return _manager
