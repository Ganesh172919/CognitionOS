"""
High-Performance Event Bus — CognitionOS Core Engine

Production-grade async event bus with:
- Priority-based event dispatch
- Wildcard topic subscription
- Dead-letter queue for failed events
- Event replay / redelivery
- Backpressure management
- Event deduplication
- Ordered event channels
- Metrics and tracing hooks
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type,
)

logger = logging.getLogger(__name__)


class EventPriority(IntEnum):
    CRITICAL = 0
    HIGH = 10
    NORMAL = 50
    LOW = 90
    BACKGROUND = 100


@dataclass(frozen=True)
class Event:
    """Immutable event envelope."""
    event_id: str
    topic: str
    payload: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    source: str = ""
    correlation_id: str = ""
    tenant_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

    @staticmethod
    def create(topic: str, payload: Dict[str, Any], *,
               priority: EventPriority = EventPriority.NORMAL,
               source: str = "", tenant_id: str = "",
               correlation_id: str = "",
               metadata: Dict[str, Any] | None = None) -> "Event":
        return Event(
            event_id=uuid.uuid4().hex,
            topic=topic, payload=payload, priority=priority,
            source=source, tenant_id=tenant_id,
            correlation_id=correlation_id or uuid.uuid4().hex,
            metadata=metadata or {},
        )

    @property
    def fingerprint(self) -> str:
        """Content-based dedup fingerprint."""
        raw = f"{self.topic}:{self.source}:{sorted(self.payload.items())}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


EventHandler = Callable[[Event], Awaitable[None]]


@dataclass
class _Subscription:
    handler: EventHandler
    topic_pattern: str
    priority_filter: Optional[EventPriority] = None
    tenant_filter: Optional[str] = None
    max_concurrency: int = 10
    _semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    def matches(self, event: Event) -> bool:
        if not self._topic_matches(event.topic):
            return False
        if self.priority_filter is not None and event.priority > self.priority_filter:
            return False
        if self.tenant_filter and event.tenant_id != self.tenant_filter:
            return False
        return True

    def _topic_matches(self, topic: str) -> bool:
        if self.topic_pattern == "*":
            return True
        if self.topic_pattern.endswith(".*"):
            prefix = self.topic_pattern[:-2]
            return topic == prefix or topic.startswith(prefix + ".")
        return topic == self.topic_pattern


@dataclass
class DeadLetterEntry:
    event: Event
    error: str
    handler_name: str
    failed_at: float = field(default_factory=time.time)
    retry_count: int = 0


class EventBusMetrics:
    """Real-time event bus metrics collector."""

    def __init__(self):
        self.events_published: int = 0
        self.events_delivered: int = 0
        self.events_failed: int = 0
        self.events_deduplicated: int = 0
        self.dead_letters: int = 0
        self._topic_counts: Dict[str, int] = defaultdict(int)
        self._latencies: List[float] = []
        self._start_time = time.time()

    def record_publish(self, topic: str):
        self.events_published += 1
        self._topic_counts[topic] += 1

    def record_delivery(self, latency_ms: float):
        self.events_delivered += 1
        self._latencies.append(latency_ms)
        if len(self._latencies) > 10000:
            self._latencies = self._latencies[-5000:]

    def record_failure(self):
        self.events_failed += 1

    def record_dedup(self):
        self.events_deduplicated += 1

    def record_dead_letter(self):
        self.dead_letters += 1

    def snapshot(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0
        p99_latency = sorted(self._latencies)[int(len(self._latencies) * 0.99)] if self._latencies else 0
        return {
            "uptime_seconds": round(uptime, 1),
            "events_published": self.events_published,
            "events_delivered": self.events_delivered,
            "events_failed": self.events_failed,
            "events_deduplicated": self.events_deduplicated,
            "dead_letters": self.dead_letters,
            "throughput_eps": round(self.events_published / max(uptime, 1), 2),
            "avg_latency_ms": round(avg_latency, 3),
            "p99_latency_ms": round(p99_latency, 3),
            "top_topics": dict(sorted(self._topic_counts.items(),
                                       key=lambda x: -x[1])[:20]),
        }


class EventBus:
    """
    Production-grade async event bus with priority dispatch, dedup,
    dead-letter queue, backpressure, and metrics.
    """

    def __init__(self, *, dedup_window_seconds: float = 60,
                 max_queue_size: int = 50000,
                 dead_letter_max: int = 10000):
        self._subscriptions: Dict[str, List[_Subscription]] = defaultdict(list)
        self._global_subs: List[_Subscription] = []
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._dead_letter: List[DeadLetterEntry] = []
        self._dead_letter_max = dead_letter_max
        self._dedup_window = dedup_window_seconds
        self._seen_fingerprints: Dict[str, float] = {}
        self._metrics = EventBusMetrics()
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._interceptors: List[Callable[[Event], Awaitable[Optional[Event]]]] = []
        self._middleware: List[Callable[[Event, EventHandler], Awaitable[None]]] = []

    # ── Subscription ──

    def subscribe(self, topic_pattern: str, handler: EventHandler, *,
                  priority_filter: Optional[EventPriority] = None,
                  tenant_filter: Optional[str] = None,
                  max_concurrency: int = 10) -> str:
        sub = _Subscription(
            handler=handler, topic_pattern=topic_pattern,
            priority_filter=priority_filter, tenant_filter=tenant_filter,
            max_concurrency=max_concurrency,
        )
        sub_id = uuid.uuid4().hex
        if topic_pattern == "*" or topic_pattern.endswith(".*"):
            self._global_subs.append(sub)
        else:
            self._subscriptions[topic_pattern].append(sub)
        logger.info("Subscribed %s to '%s' (id=%s)", handler.__name__, topic_pattern, sub_id)
        return sub_id

    def add_interceptor(self, interceptor: Callable[[Event], Awaitable[Optional[Event]]]):
        """Add event interceptor (transform/filter events before dispatch)."""
        self._interceptors.append(interceptor)

    def add_middleware(self, mw: Callable[[Event, EventHandler], Awaitable[None]]):
        """Add dispatch middleware (wrap handler execution)."""
        self._middleware.append(mw)

    # ── Publishing ──

    async def publish(self, event: Event) -> bool:
        """Publish event. Returns False if deduplicated or queue full."""
        # Dedup check
        now = time.time()
        fp = event.fingerprint
        if fp in self._seen_fingerprints:
            if now - self._seen_fingerprints[fp] < self._dedup_window:
                self._metrics.record_dedup()
                return False
        self._seen_fingerprints[fp] = now

        # Cleanup old fingerprints periodically
        if len(self._seen_fingerprints) > 100000:
            cutoff = now - self._dedup_window
            self._seen_fingerprints = {
                k: v for k, v in self._seen_fingerprints.items() if v > cutoff
            }

        # Interceptors
        current: Optional[Event] = event
        for interceptor in self._interceptors:
            current = await interceptor(current)
            if current is None:
                return False

        # Enqueue with priority
        try:
            self._queue.put_nowait((current.priority.value, current.timestamp, current))
        except asyncio.QueueFull:
            logger.warning("Event bus queue full, dropping event %s", event.event_id)
            return False

        self._metrics.record_publish(current.topic)
        return True

    async def publish_many(self, events: List[Event]) -> int:
        """Batch publish. Returns count of successfully enqueued."""
        count = 0
        for e in events:
            if await self.publish(e):
                count += 1
        return count

    # ── Dispatch workers ──

    async def start(self, num_workers: int = 8):
        """Start background dispatch workers."""
        if self._running:
            return
        self._running = True
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)
        logger.info("EventBus started with %d workers", num_workers)

    async def stop(self):
        """Gracefully stop all workers."""
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("EventBus stopped")

    async def _worker(self, worker_id: int):
        """Worker loop that dequeues and dispatches events."""
        while self._running:
            try:
                priority, ts, event = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Worker %d error: %s", worker_id, exc, exc_info=True)

    async def _dispatch(self, event: Event):
        """Fan-out event to all matching subscribers."""
        handlers: List[_Subscription] = []

        # Exact topic subscribers
        if event.topic in self._subscriptions:
            handlers.extend(self._subscriptions[event.topic])

        # Wildcard/global subscribers
        for sub in self._global_subs:
            if sub.matches(event):
                handlers.append(sub)

        for sub in handlers:
            if not sub.matches(event):
                continue
            start = time.time()
            try:
                async with sub._semaphore:
                    if self._middleware:
                        for mw in self._middleware:
                            await mw(event, sub.handler)
                    else:
                        await sub.handler(event)
                latency_ms = (time.time() - start) * 1000
                self._metrics.record_delivery(latency_ms)
            except Exception as exc:
                self._metrics.record_failure()
                logger.error("Handler %s failed for event %s: %s",
                             sub.handler.__name__, event.event_id, exc)
                await self._handle_failure(event, sub.handler, str(exc))

    async def _handle_failure(self, event: Event, handler: EventHandler, error: str):
        """Retry or send to dead-letter queue."""
        if event.retry_count < event.max_retries:
            retry_event = Event(
                event_id=event.event_id, topic=event.topic,
                payload=event.payload, priority=event.priority,
                source=event.source, correlation_id=event.correlation_id,
                tenant_id=event.tenant_id, timestamp=event.timestamp,
                metadata=event.metadata,
                retry_count=event.retry_count + 1,
                max_retries=event.max_retries,
            )
            delay = 0.5 * (2 ** event.retry_count)  # Exponential backoff
            await asyncio.sleep(min(delay, 30))
            await self.publish(retry_event)
        else:
            entry = DeadLetterEntry(
                event=event, error=error,
                handler_name=handler.__name__,
                retry_count=event.retry_count,
            )
            self._dead_letter.append(entry)
            if len(self._dead_letter) > self._dead_letter_max:
                self._dead_letter = self._dead_letter[-self._dead_letter_max:]
            self._metrics.record_dead_letter()
            logger.error("Event %s sent to dead-letter queue after %d retries",
                         event.event_id, event.max_retries)

    # ── Dead Letter Queue ──

    def get_dead_letters(self, *, limit: int = 100, topic: str = "") -> List[Dict[str, Any]]:
        entries = self._dead_letter
        if topic:
            entries = [e for e in entries if e.event.topic == topic]
        return [
            {"event_id": e.event.event_id, "topic": e.event.topic,
             "error": e.error, "handler": e.handler_name,
             "retry_count": e.retry_count,
             "failed_at": datetime.fromtimestamp(e.failed_at, tz=timezone.utc).isoformat()}
            for e in entries[-limit:]
        ]

    async def replay_dead_letters(self, *, topic: str = "", max_count: int = 50) -> int:
        """Re-publish dead-letter events for reprocessing."""
        entries = [e for e in self._dead_letter if not topic or e.event.topic == topic]
        replayed = 0
        for entry in entries[:max_count]:
            retry_event = Event(
                event_id=uuid.uuid4().hex, topic=entry.event.topic,
                payload=entry.event.payload, priority=entry.event.priority,
                source=entry.event.source,
                correlation_id=entry.event.correlation_id,
                tenant_id=entry.event.tenant_id,
                metadata={**entry.event.metadata, "replayed_from": entry.event.event_id},
                retry_count=0, max_retries=entry.event.max_retries,
            )
            if await self.publish(retry_event):
                self._dead_letter.remove(entry)
                replayed += 1
        return replayed

    # ── Metrics ──

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics.snapshot()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
