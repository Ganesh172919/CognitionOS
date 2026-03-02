"""
Event Bus - High-Performance Domain Event System

Provides in-process event publishing with optional persistence,
replay, dead-letter handling, and distributed event forwarding.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import (
    Any, Callable, Coroutine, Dict, List, Optional, Set, Type, Union,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core value objects
# ---------------------------------------------------------------------------


class EventPriority(IntEnum):
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class Event:
    """Base event container."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        if not self.event_type:
            self.event_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "tenant_id": self.tenant_id,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "payload": self.payload,
            "version": self.version,
        }


@dataclass
class DomainEvent(Event):
    """Domain event – stays within the bounded context."""

    aggregate_type: str = ""
    aggregate_id: str = ""


@dataclass
class IntegrationEvent(Event):
    """Integration event – crosses bounded-context boundaries."""

    target_service: str = ""
    retry_count: int = 0
    max_retries: int = 3


# ---------------------------------------------------------------------------
# Event handler protocol
# ---------------------------------------------------------------------------


class EventHandler(ABC):
    """Abstract event handler."""

    @abstractmethod
    async def handle(self, event: Event) -> None: ...

    def can_handle(self, event: Event) -> bool:
        return True


EventCallable = Callable[[Event], Coroutine[Any, Any, None]]


@dataclass
class EventSubscription:
    """Tracks a subscription to an event type."""

    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    handler: Optional[Union[EventHandler, EventCallable]] = None
    priority: EventPriority = EventPriority.NORMAL
    filter_fn: Optional[Callable[[Event], bool]] = None
    max_retries: int = 3
    timeout_seconds: float = 30.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class EventMiddleware(ABC):
    """Pre/post processing for events."""

    @abstractmethod
    async def before_publish(self, event: Event) -> Event: ...

    @abstractmethod
    async def after_publish(self, event: Event, results: List[Any]) -> None: ...


class LoggingMiddleware(EventMiddleware):
    async def before_publish(self, event: Event) -> Event:
        logger.debug("Publishing event %s [%s]", event.event_type, event.event_id)
        return event

    async def after_publish(self, event: Event, results: List[Any]) -> None:
        logger.debug(
            "Event %s processed by %d handlers", event.event_type, len(results)
        )


class MetricsMiddleware(EventMiddleware):
    def __init__(self):
        self.publish_count: int = 0
        self.error_count: int = 0
        self.total_latency_ms: float = 0.0

    async def before_publish(self, event: Event) -> Event:
        event.metadata["_publish_start"] = time.monotonic()
        return event

    async def after_publish(self, event: Event, results: List[Any]) -> None:
        start = event.metadata.pop("_publish_start", None)
        if start:
            elapsed_ms = (time.monotonic() - start) * 1000
            self.total_latency_ms += elapsed_ms
        self.publish_count += 1

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.publish_count, 1)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "publish_count": self.publish_count,
            "error_count": self.error_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class TenantIsolationMiddleware(EventMiddleware):
    """Ensures events carry tenant context."""

    async def before_publish(self, event: Event) -> Event:
        if not event.tenant_id:
            logger.warning("Event %s missing tenant_id", event.event_type)
        return event

    async def after_publish(self, event: Event, results: List[Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Event store (in-memory, pluggable)
# ---------------------------------------------------------------------------


class EventStore:
    """Append-only event store with replay support."""

    def __init__(self, max_events: int = 100_000):
        self._events: List[Event] = []
        self._max_events = max_events
        self._lock = asyncio.Lock()

    async def append(self, event: Event) -> None:
        async with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

    async def get_events(
        self,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        aggregate_id: Optional[str] = None,
    ) -> List[Event]:
        filtered = self._events
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if since:
            filtered = [e for e in filtered if e.timestamp >= since]
        if aggregate_id:
            filtered = [
                e
                for e in filtered
                if isinstance(e, DomainEvent) and e.aggregate_id == aggregate_id
            ]
        return filtered[-limit:]

    async def replay(
        self,
        handler: Union[EventHandler, EventCallable],
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> int:
        events = await self.get_events(event_type=event_type, since=since, limit=10_000)
        count = 0
        for ev in events:
            try:
                if isinstance(handler, EventHandler):
                    await handler.handle(ev)
                else:
                    await handler(ev)
                count += 1
            except Exception:
                logger.exception("Replay error for event %s", ev.event_id)
        return count

    @property
    def event_count(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Dead-letter queue
# ---------------------------------------------------------------------------


@dataclass
class DeadLetterEntry:
    event: Event
    error: str
    handler_name: str
    attempt: int
    failed_at: datetime = field(default_factory=datetime.utcnow)


class DeadLetterQueue:
    def __init__(self, max_size: int = 10_000):
        self._entries: List[DeadLetterEntry] = []
        self._max_size = max_size

    async def add(self, entry: DeadLetterEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size:]

    async def get_entries(
        self, limit: int = 50, event_type: Optional[str] = None
    ) -> List[DeadLetterEntry]:
        entries = self._entries
        if event_type:
            entries = [e for e in entries if e.event.event_type == event_type]
        return entries[-limit:]

    async def retry(self, bus: "EventBus", entry_index: int) -> bool:
        if 0 <= entry_index < len(self._entries):
            entry = self._entries.pop(entry_index)
            await bus.publish(entry.event)
            return True
        return False

    @property
    def size(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------


class EventBus:
    """
    High-performance async event bus.

    Features:
    - Typed subscriptions with priority ordering
    - Middleware pipeline
    - Dead-letter queue
    - Optional event store for replay
    - Correlation / causation tracking
    - Tenant isolation support
    """

    def __init__(
        self,
        store: Optional[EventStore] = None,
        enable_dead_letter: bool = True,
        max_concurrent: int = 50,
    ):
        self._subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._wildcard_subscriptions: List[EventSubscription] = []
        self._middleware: List[EventMiddleware] = []
        self._store = store
        self._dlq = DeadLetterQueue() if enable_dead_letter else None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = True

        # Metrics
        self._published_count = 0
        self._handled_count = 0
        self._error_count = 0

    # -- Middleware ----------------------------------------------------------

    def add_middleware(self, middleware: EventMiddleware) -> None:
        self._middleware.append(middleware)

    # -- Subscribe ----------------------------------------------------------

    def subscribe(
        self,
        event_type: str,
        handler: Union[EventHandler, EventCallable],
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[Event], bool]] = None,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ) -> EventSubscription:
        sub = EventSubscription(
            event_type=event_type,
            handler=handler,
            priority=priority,
            filter_fn=filter_fn,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        if event_type == "*":
            self._wildcard_subscriptions.append(sub)
        else:
            self._subscriptions[event_type].append(sub)
            self._subscriptions[event_type].sort(
                key=lambda s: s.priority, reverse=True
            )
        return sub

    def unsubscribe(self, subscription_id: str) -> bool:
        for subs in self._subscriptions.values():
            for sub in subs:
                if sub.subscription_id == subscription_id:
                    sub.active = False
                    subs.remove(sub)
                    return True
        for sub in self._wildcard_subscriptions:
            if sub.subscription_id == subscription_id:
                sub.active = False
                self._wildcard_subscriptions.remove(sub)
                return True
        return False

    def on(self, event_type: str, **kwargs):
        """Decorator for subscribing to events."""

        def decorator(fn: EventCallable):
            self.subscribe(event_type, fn, **kwargs)
            return fn

        return decorator

    # -- Publish ------------------------------------------------------------

    async def publish(self, event: Event) -> List[Any]:
        if not self._running:
            logger.warning("EventBus is stopped; dropping event %s", event.event_id)
            return []

        # Middleware: before
        for mw in self._middleware:
            event = await mw.before_publish(event)

        # Store event
        if self._store:
            await self._store.append(event)

        self._published_count += 1

        # Gather matching subscriptions
        subs = list(self._subscriptions.get(event.event_type, []))
        subs.extend(self._wildcard_subscriptions)
        subs = [s for s in subs if s.active]

        results: List[Any] = []

        for sub in subs:
            if sub.filter_fn and not sub.filter_fn(event):
                continue

            async with self._semaphore:
                result = await self._invoke_handler(sub, event)
                results.append(result)

        # Middleware: after
        for mw in self._middleware:
            await mw.after_publish(event, results)

        return results

    async def publish_many(self, events: List[Event]) -> None:
        tasks = [self.publish(ev) for ev in events]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _invoke_handler(
        self, sub: EventSubscription, event: Event
    ) -> Optional[Any]:
        handler = sub.handler
        handler_name = (
            handler.__class__.__name__
            if isinstance(handler, EventHandler)
            else getattr(handler, "__name__", "anonymous")
        )

        for attempt in range(1, sub.max_retries + 1):
            try:
                if isinstance(handler, EventHandler):
                    if not handler.can_handle(event):
                        return None
                    result = await asyncio.wait_for(
                        handler.handle(event), timeout=sub.timeout_seconds
                    )
                else:
                    result = await asyncio.wait_for(
                        handler(event), timeout=sub.timeout_seconds
                    )
                self._handled_count += 1
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    "Handler %s timed out for event %s (attempt %d/%d)",
                    handler_name,
                    event.event_id,
                    attempt,
                    sub.max_retries,
                )
            except Exception as exc:
                logger.exception(
                    "Handler %s failed for event %s (attempt %d/%d): %s",
                    handler_name,
                    event.event_id,
                    attempt,
                    sub.max_retries,
                    exc,
                )
                if attempt == sub.max_retries:
                    self._error_count += 1
                    if self._dlq:
                        await self._dlq.add(
                            DeadLetterEntry(
                                event=event,
                                error=str(exc),
                                handler_name=handler_name,
                                attempt=attempt,
                            )
                        )
                    return None
                await asyncio.sleep(0.1 * (2 ** attempt))
        return None

    # -- Management ---------------------------------------------------------

    def stop(self) -> None:
        self._running = False

    def start(self) -> None:
        self._running = True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "published_count": self._published_count,
            "handled_count": self._handled_count,
            "error_count": self._error_count,
            "subscription_count": sum(
                len(s) for s in self._subscriptions.values()
            )
            + len(self._wildcard_subscriptions),
            "dlq_size": self._dlq.size if self._dlq else 0,
            "store_size": self._store.event_count if self._store else 0,
        }

    def get_subscription_info(self) -> List[Dict[str, Any]]:
        result = []
        for event_type, subs in self._subscriptions.items():
            for s in subs:
                result.append(
                    {
                        "subscription_id": s.subscription_id,
                        "event_type": event_type,
                        "priority": s.priority.name,
                        "active": s.active,
                    }
                )
        return result
