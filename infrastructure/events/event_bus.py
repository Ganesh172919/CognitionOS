"""
Production Event Bus — CognitionOS Core Engine Layer

Async publish/subscribe with:
- Topic-based routing with wildcard support
- Dead-letter queue for failed handlers
- Event replay / sourcing hooks
- Middleware chain (rate-limit, dedup, audit)
- Backpressure via bounded internal queues
- JSON-serialisable event envelopes
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event primitives
# ---------------------------------------------------------------------------


class EventPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """Immutable event envelope."""

    topic: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "system"
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "metadata": self.metadata,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        data = dict(data)
        data["priority"] = EventPriority(data.get("priority", "normal"))
        return cls(**data)

    @property
    def fingerprint(self) -> str:
        raw = json.dumps({"topic": self.topic, "payload": self.payload}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class DeadLetterEntry:
    event: Event
    handler_name: str
    error: str
    traceback_str: str
    failed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    retry_count: int = 0


# ---------------------------------------------------------------------------
# Middleware protocol
# ---------------------------------------------------------------------------

EventHandler = Callable[[Event], Awaitable[None]]
MiddlewareNext = Callable[[Event], Awaitable[None]]
EventMiddleware = Callable[[Event, MiddlewareNext], Awaitable[None]]


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------


@dataclass
class Subscription:
    handler: EventHandler
    topic_pattern: str
    handler_name: str
    priority: int = 0  # higher = runs first
    max_retries: int = 2
    timeout_seconds: float = 30.0
    active: bool = True

    def matches(self, topic: str) -> bool:
        """Support exact match + single-level wildcard (``*``)."""
        pattern_parts = self.topic_pattern.split(".")
        topic_parts = topic.split(".")
        if len(pattern_parts) != len(topic_parts):
            if pattern_parts[-1] == "#":
                return topic.startswith(".".join(pattern_parts[:-1]))
            return False
        for p, t in zip(pattern_parts, topic_parts):
            if p == "*":
                continue
            if p != t:
                return False
        return True


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------


class EventBus:
    """Async in-process event bus with optional persistence hooks."""

    def __init__(
        self,
        *,
        max_queue_size: int = 10_000,
        max_dead_letter_size: int = 5_000,
        enable_dedup: bool = True,
        dedup_window_seconds: int = 60,
    ) -> None:
        self._subscriptions: List[Subscription] = []
        self._middleware: List[EventMiddleware] = []
        self._dead_letter: List[DeadLetterEntry] = []
        self._event_store: List[Event] = []  # in-memory sourcing
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._max_dead_letter = max_dead_letter_size
        self._enable_dedup = enable_dedup
        self._dedup_window = dedup_window_seconds
        self._seen_fingerprints: Dict[str, float] = {}
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None
        self._metrics: Dict[str, int] = defaultdict(int)
        self._event_hooks: Dict[str, List[Callable]] = defaultdict(list)

    # ----- lifecycle -----

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop())
        logger.info("EventBus started (queue_max=%d)", self._queue.maxsize)

    async def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await asyncio.wait_for(self._consumer_task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        logger.info("EventBus stopped. Published=%d Delivered=%d Failed=%d",
                     self._metrics["published"], self._metrics["delivered"], self._metrics["failed"])

    # ----- subscribe -----

    def subscribe(
        self,
        topic_pattern: str,
        handler: EventHandler,
        *,
        handler_name: Optional[str] = None,
        priority: int = 0,
        max_retries: int = 2,
        timeout_seconds: float = 30.0,
    ) -> Subscription:
        sub = Subscription(
            handler=handler,
            topic_pattern=topic_pattern,
            handler_name=handler_name or handler.__qualname__,
            priority=priority,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        self._subscriptions.append(sub)
        self._subscriptions.sort(key=lambda s: -s.priority)
        logger.debug("Subscribed %s to %s", sub.handler_name, topic_pattern)
        return sub

    def unsubscribe(self, subscription: Subscription) -> None:
        self._subscriptions = [s for s in self._subscriptions if s is not subscription]

    # ----- middleware -----

    def use(self, middleware: EventMiddleware) -> None:
        self._middleware.append(middleware)

    # ----- publish -----

    async def publish(self, event: Event) -> None:
        if self._enable_dedup:
            fp = event.fingerprint
            now = time.monotonic()
            if fp in self._seen_fingerprints and (now - self._seen_fingerprints[fp]) < self._dedup_window:
                self._metrics["deduplicated"] += 1
                logger.debug("Deduplicated event %s", event.event_id)
                return
            self._seen_fingerprints[fp] = now
            self._cleanup_dedup()

        self._event_store.append(event)
        self._metrics["published"] += 1

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._metrics["backpressure_drops"] += 1
            logger.warning("EventBus queue full — dropping event %s", event.event_id)

    async def publish_many(self, events: List[Event]) -> None:
        for e in events:
            await self.publish(e)

    # ----- convenience factory -----

    async def emit(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        source: str = "system",
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> Event:
        event = Event(topic=topic, payload=payload, source=source, priority=priority, correlation_id=correlation_id)
        await self.publish(event)
        return event

    # ----- consumer loop -----

    async def _consume_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            await self._dispatch(event)

    async def _dispatch(self, event: Event) -> None:
        matched = [s for s in self._subscriptions if s.active and s.matches(event.topic)]
        if not matched:
            self._metrics["unmatched"] += 1
            return

        for sub in matched:
            await self._run_with_middleware(event, sub)

    async def _run_with_middleware(self, event: Event, sub: Subscription) -> None:
        async def final_handler(ev: Event) -> None:
            await self._execute_handler(ev, sub)

        chain = final_handler
        for mw in reversed(self._middleware):
            prev = chain

            async def wrapped(ev: Event, _mw=mw, _prev=prev) -> None:
                await _mw(ev, _prev)

            chain = wrapped

        await chain(event)

    async def _execute_handler(self, event: Event, sub: Subscription) -> None:
        for attempt in range(sub.max_retries + 1):
            try:
                await asyncio.wait_for(sub.handler(event), timeout=sub.timeout_seconds)
                self._metrics["delivered"] += 1
                return
            except asyncio.TimeoutError:
                logger.warning("Handler %s timed out on event %s (attempt %d)", sub.handler_name, event.event_id, attempt + 1)
            except Exception as exc:
                logger.error("Handler %s failed on event %s (attempt %d): %s", sub.handler_name, event.event_id, attempt + 1, exc)
                if attempt == sub.max_retries:
                    self._send_to_dead_letter(event, sub.handler_name, exc)

        self._metrics["failed"] += 1

    # ----- dead-letter queue -----

    def _send_to_dead_letter(self, event: Event, handler_name: str, exc: Exception) -> None:
        entry = DeadLetterEntry(
            event=event,
            handler_name=handler_name,
            error=str(exc),
            traceback_str=traceback.format_exc(),
        )
        self._dead_letter.append(entry)
        if len(self._dead_letter) > self._max_dead_letter:
            self._dead_letter = self._dead_letter[-self._max_dead_letter:]
        self._metrics["dead_lettered"] += 1

    def get_dead_letters(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [
            {
                "event": dl.event.to_dict(),
                "handler": dl.handler_name,
                "error": dl.error,
                "failed_at": dl.failed_at,
                "retry_count": dl.retry_count,
            }
            for dl in self._dead_letter[-limit:]
        ]

    async def retry_dead_letter(self, index: int) -> bool:
        if 0 <= index < len(self._dead_letter):
            entry = self._dead_letter.pop(index)
            entry.retry_count += 1
            await self.publish(entry.event)
            return True
        return False

    # ----- replay / event sourcing -----

    async def replay(self, topic_prefix: str, *, since: Optional[str] = None, handler: Optional[EventHandler] = None) -> int:
        replayed = 0
        for event in self._event_store:
            if not event.topic.startswith(topic_prefix):
                continue
            if since and event.timestamp < since:
                continue
            if handler:
                await handler(event)
            else:
                await self._dispatch(event)
            replayed += 1
        return replayed

    # ----- metrics -----

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **dict(self._metrics),
            "queue_size": self._queue.qsize(),
            "subscription_count": len(self._subscriptions),
            "dead_letter_count": len(self._dead_letter),
            "event_store_count": len(self._event_store),
        }

    # ----- cleanup -----

    def _cleanup_dedup(self) -> None:
        now = time.monotonic()
        expired = [fp for fp, ts in self._seen_fingerprints.items() if now - ts > self._dedup_window]
        for fp in expired:
            del self._seen_fingerprints[fp]

    def clear_event_store(self) -> int:
        count = len(self._event_store)
        self._event_store.clear()
        return count


# ---------------------------------------------------------------------------
# Built-in middleware factories
# ---------------------------------------------------------------------------


def audit_middleware(audit_log: List[Dict[str, Any]]) -> EventMiddleware:
    """Log every event passing through for audit trail."""

    async def middleware(event: Event, next_fn: MiddlewareNext) -> None:
        audit_log.append({
            "event_id": event.event_id,
            "topic": event.topic,
            "source": event.source,
            "timestamp": event.timestamp,
        })
        await next_fn(event)

    return middleware


def rate_limit_middleware(*, max_per_second: int = 100) -> EventMiddleware:
    """Simple token-bucket rate limiter."""

    tokens = max_per_second
    last_refill = time.monotonic()

    async def middleware(event: Event, next_fn: MiddlewareNext) -> None:
        nonlocal tokens, last_refill
        now = time.monotonic()
        elapsed = now - last_refill
        tokens = min(max_per_second, tokens + int(elapsed * max_per_second))
        last_refill = now
        if tokens <= 0:
            logger.warning("Rate limit hit — dropping event %s", event.event_id)
            return
        tokens -= 1
        await next_fn(event)

    return middleware


def filter_middleware(*, allowed_topics: Optional[Set[str]] = None, blocked_topics: Optional[Set[str]] = None) -> EventMiddleware:
    """Allow/block based on topic."""

    async def middleware(event: Event, next_fn: MiddlewareNext) -> None:
        if blocked_topics and event.topic in blocked_topics:
            return
        if allowed_topics and event.topic not in allowed_topics:
            return
        await next_fn(event)

    return middleware


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_default_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


async def init_event_bus(**kwargs: Any) -> EventBus:
    global _default_bus
    _default_bus = EventBus(**kwargs)
    await _default_bus.start()
    return _default_bus
