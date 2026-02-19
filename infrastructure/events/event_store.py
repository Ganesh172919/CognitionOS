"""
Event Sourcing & CQRS Implementation

Production-grade event sourcing with:
- Append-only event store
- Event replay and projection
- Read model generation
- Snapshot management
- Event versioning
- Saga pattern support
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types."""
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_UPDATED = "workflow.updated"
    WORKFLOW_DELETED = "workflow.deleted"
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    USER_REGISTERED = "user.registered"
    SUBSCRIPTION_CREATED = "subscription.created"
    SUBSCRIPTION_UPGRADED = "subscription.upgraded"
    PAYMENT_PROCESSED = "payment.processed"


@dataclass
class Event:
    """Base event class."""
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    timestamp: datetime
    version: int
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "data": self.data,
            "metadata": self.metadata,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            aggregate_id=data["aggregate_id"],
            aggregate_type=data["aggregate_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data["version"],
            data=data["data"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Snapshot:
    """Aggregate snapshot."""
    snapshot_id: str
    aggregate_id: str
    aggregate_type: str
    version: int
    state: Dict[str, Any]
    timestamp: datetime


class EventStore:
    """
    Append-only event store for event sourcing.
    
    Features:
    - Append events atomically
    - Read events by aggregate
    - Event replay
    - Snapshot support
    - Concurrent read/write
    """
    
    def __init__(self):
        # In-memory storage (replace with PostgreSQL in production)
        self.events: List[Event] = []
        self.snapshots: Dict[str, Snapshot] = {}
        self.event_index: Dict[str, List[Event]] = {}
        self.version_index: Dict[str, int] = {}
        self.lock = asyncio.Lock()
        
    async def append_event(self, event: Event) -> bool:
        """Append event to store."""
        async with self.lock:
            # Validate version (optimistic concurrency control)
            aggregate_key = f"{event.aggregate_type}:{event.aggregate_id}"
            current_version = self.version_index.get(aggregate_key, 0)
            
            if event.version != current_version + 1:
                logger.error(
                    f"Version conflict: expected {current_version + 1}, "
                    f"got {event.version}"
                )
                return False
                
            # Append event
            self.events.append(event)
            
            # Update indexes
            if aggregate_key not in self.event_index:
                self.event_index[aggregate_key] = []
            self.event_index[aggregate_key].append(event)
            self.version_index[aggregate_key] = event.version
            
            logger.info(
                f"Appended event: {event.event_type} "
                f"for {aggregate_key} (version {event.version})"
            )
            return True
            
    async def get_events(
        self,
        aggregate_id: str,
        aggregate_type: str,
        from_version: int = 0,
    ) -> List[Event]:
        """Get events for aggregate."""
        aggregate_key = f"{aggregate_type}:{aggregate_id}"
        events = self.event_index.get(aggregate_key, [])
        return [e for e in events if e.version > from_version]
        
    async def get_all_events(
        self,
        event_type: Optional[str] = None,
        from_timestamp: Optional[datetime] = None,
    ) -> List[Event]:
        """Get all events with optional filters."""
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        if from_timestamp:
            events = [e for e in events if e.timestamp >= from_timestamp]
            
        return events
        
    async def save_snapshot(self, snapshot: Snapshot):
        """Save aggregate snapshot."""
        key = f"{snapshot.aggregate_type}:{snapshot.aggregate_id}"
        self.snapshots[key] = snapshot
        logger.info(f"Saved snapshot: {key} (version {snapshot.version})")
        
    async def get_snapshot(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> Optional[Snapshot]:
        """Get latest snapshot."""
        key = f"{aggregate_type}:{aggregate_id}"
        return self.snapshots.get(key)
        
    async def get_current_version(
        self,
        aggregate_id: str,
        aggregate_type: str,
    ) -> int:
        """Get current version of aggregate."""
        key = f"{aggregate_type}:{aggregate_id}"
        return self.version_index.get(key, 0)


class EventProjector:
    """
    Event projector for building read models.
    
    Subscribes to events and updates read models.
    """
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.handlers: Dict[str, List[Callable]] = {}
        self.read_models: Dict[str, Any] = {}
        self.last_processed_event: int = 0
        
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}")
        
    async def project_events(self):
        """Project events to read models."""
        events = await self.event_store.get_all_events()
        
        for event in events[self.last_processed_event:]:
            await self._process_event(event)
            self.last_processed_event += 1
            
        logger.info(f"Processed {len(events)} events")
        
    async def _process_event(self, event: Event):
        """Process single event."""
        handlers = self.handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
                
    def get_read_model(self, model_name: str) -> Optional[Any]:
        """Get read model."""
        return self.read_models.get(model_name)
        
    def update_read_model(self, model_name: str, data: Any):
        """Update read model."""
        self.read_models[model_name] = data


class SagaOrchestrator:
    """
    Saga orchestrator for distributed transactions.
    
    Coordinates multi-step processes across aggregates.
    """
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.sagas: Dict[str, 'Saga'] = {}
        
    async def start_saga(
        self,
        saga_id: str,
        saga_type: str,
        steps: List[Dict[str, Any]],
    ) -> str:
        """Start a saga."""
        saga = Saga(
            saga_id=saga_id,
            saga_type=saga_type,
            steps=steps,
            event_store=self.event_store,
        )
        
        self.sagas[saga_id] = saga
        await saga.execute()
        return saga_id
        
    async def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get saga status."""
        if saga_id not in self.sagas:
            return None
            
        saga = self.sagas[saga_id]
        return {
            "saga_id": saga.saga_id,
            "saga_type": saga.saga_type,
            "status": saga.status,
            "current_step": saga.current_step,
            "total_steps": len(saga.steps),
            "started_at": saga.started_at.isoformat() if saga.started_at else None,
            "completed_at": saga.completed_at.isoformat() if saga.completed_at else None,
        }


@dataclass
class Saga:
    """Saga definition."""
    saga_id: str
    saga_type: str
    steps: List[Dict[str, Any]]
    event_store: EventStore
    
    status: str = "pending"
    current_step: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    compensation_needed: bool = False
    
    async def execute(self):
        """Execute saga steps."""
        self.status = "running"
        self.started_at = datetime.utcnow()
        
        try:
            for i, step in enumerate(self.steps):
                self.current_step = i
                
                # Execute step
                await self._execute_step(step)
                
            self.status = "completed"
            self.completed_at = datetime.utcnow()
            logger.info(f"Saga completed: {self.saga_id}")
            
        except Exception as e:
            logger.error(f"Saga failed: {self.saga_id}: {e}")
            self.status = "failed"
            self.compensation_needed = True
            await self._compensate()
            
    async def _execute_step(self, step: Dict[str, Any]):
        """Execute saga step."""
        # Create and append event
        event = Event(
            event_id=str(uuid4()),
            event_type=step["event_type"],
            aggregate_id=step["aggregate_id"],
            aggregate_type=step["aggregate_type"],
            timestamp=datetime.utcnow(),
            version=step.get("version", 1),
            data=step.get("data", {}),
            metadata={"saga_id": self.saga_id},
        )
        
        await self.event_store.append_event(event)
        
    async def _compensate(self):
        """Compensate failed saga."""
        logger.info(f"Compensating saga: {self.saga_id}")
        
        # Execute compensation steps in reverse order
        for i in range(self.current_step, -1, -1):
            step = self.steps[i]
            
            if "compensation" in step:
                await self._execute_step(step["compensation"])
                
        self.status = "compensated"


class EventReplayer:
    """
    Event replayer for rebuilding state.
    
    Replays events to reconstruct aggregate state.
    """
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        
    async def replay_aggregate(
        self,
        aggregate_id: str,
        aggregate_type: str,
        initial_state: Dict[str, Any],
        event_handlers: Dict[str, Callable],
    ) -> Dict[str, Any]:
        """Replay events to rebuild aggregate state."""
        # Try to load snapshot first
        snapshot = await self.event_store.get_snapshot(aggregate_id, aggregate_type)
        
        if snapshot:
            state = snapshot.state
            from_version = snapshot.version
        else:
            state = initial_state
            from_version = 0
            
        # Replay events from snapshot version
        events = await self.event_store.get_events(
            aggregate_id,
            aggregate_type,
            from_version,
        )
        
        for event in events:
            handler = event_handlers.get(event.event_type)
            if handler:
                state = handler(state, event)
                
        return state
        
    async def replay_all_events(
        self,
        event_type: Optional[str] = None,
    ) -> List[Event]:
        """Replay all events."""
        return await self.event_store.get_all_events(event_type=event_type)
