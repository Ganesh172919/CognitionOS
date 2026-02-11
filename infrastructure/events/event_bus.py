"""
Event Infrastructure - Event Bus Implementation

Provides event publishing and subscription.
"""

from typing import Any, Callable, Dict, List
from dataclasses import dataclass
from uuid import UUID
import asyncio


@dataclass
class DomainEvent:
    """Base domain event"""
    occurred_at: Any
    event_id: UUID


class EventBus:
    """
    Simple in-memory event bus for domain events.

    In production, this could be replaced with RabbitMQ, Kafka, etc.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type name (e.g., 'WorkflowCreated')
            handler: Async callable to handle event
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def publish(self, event: DomainEvent) -> None:
        """
        Publish domain event to all subscribers.

        Args:
            event: Domain event to publish
        """
        event_type = type(event).__name__

        if event_type not in self._handlers:
            return

        # Call all handlers asynchronously
        tasks = [
            handler(event)
            for handler in self._handlers[event_type]
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def clear(self) -> None:
        """Clear all subscriptions (useful for testing)"""
        self._handlers.clear()


# Global event bus instance
_event_bus: EventBus | None = None


def init_event_bus() -> EventBus:
    """Initialize global event bus"""
    global _event_bus
    _event_bus = EventBus()
    return _event_bus


def get_event_bus() -> EventBus:
    """
    Get global event bus.

    Returns:
        EventBus instance

    Raises:
        RuntimeError: If event bus not initialized
    """
    if _event_bus is None:
        raise RuntimeError("Event bus not initialized. Call init_event_bus() first.")
    return _event_bus
