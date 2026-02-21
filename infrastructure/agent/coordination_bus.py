"""
Multi-Agent Coordination Bus

Enables multiple autonomous agents to coordinate on shared tasks:
- Broadcast and point-to-point messaging between agents
- Resource negotiation (locks, quotas, shared state)
- Task delegation with priority queuing
- Heartbeat and liveness tracking
- Distributed consensus for critical decisions
- Event-driven coordination with typed messages
- Dead-letter queue for undelivered messages
- Agent capability registry for intelligent routing
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4


# ──────────────────────────────────────────────────────────────────────────────
# Message primitives
# ──────────────────────────────────────────────────────────────────────────────


class MessagePriority(int, Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class MessageType(str, Enum):
    TASK_DELEGATION = "task_delegation"    # Delegate a sub-task to another agent
    TASK_RESULT = "task_result"            # Result of a delegated task
    BROADCAST = "broadcast"               # Broadcast to all agents
    RESOURCE_REQUEST = "resource_request" # Request a lock/resource
    RESOURCE_GRANT = "resource_grant"     # Grant a resource
    RESOURCE_DENY = "resource_deny"       # Deny a resource request
    HEARTBEAT = "heartbeat"               # Liveness ping
    CONSENSUS_VOTE = "consensus_vote"     # Participate in consensus
    CONSENSUS_RESULT = "consensus_result" # Final consensus decision
    CAPABILITY_QUERY = "capability_query" # Ask what agents can do
    CAPABILITY_REPLY = "capability_reply" # Respond with capabilities


class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class CoordinationMessage:
    """A message exchanged between agents on the coordination bus"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]       # None = broadcast
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None  # Ties request/response
    reply_to: Optional[str] = None    # Channel for response
    ttl_seconds: float = 30.0
    created_at: float = field(default_factory=time.time)
    delivered_at: Optional[float] = None
    retry_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "type": self.message_type.value,
            "sender": self.sender_id,
            "recipient": self.recipient_id,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "age_seconds": round(self.age_seconds, 2),
        }


@dataclass
class AgentRegistration:
    """An agent registered on the coordination bus"""
    agent_id: str
    capabilities: Set[str]          # Tags like "code_gen", "search", "analysis"
    max_concurrent_tasks: int = 1
    current_task_count: int = 0
    status: AgentStatus = AgentStatus.IDLE
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)

    @property
    def is_available(self) -> bool:
        return (
            self.status == AgentStatus.IDLE
            and self.current_task_count < self.max_concurrent_tasks
        )

    @property
    def is_alive(self, timeout: float = 60.0) -> bool:
        return (time.time() - self.last_heartbeat) < timeout

    def can_handle(self, required_capabilities: Set[str]) -> bool:
        return required_capabilities.issubset(self.capabilities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": list(self.capabilities),
            "status": self.status.value,
            "current_tasks": self.current_task_count,
            "max_tasks": self.max_concurrent_tasks,
            "is_available": self.is_available,
            "last_heartbeat": self.last_heartbeat,
        }


@dataclass
class ResourceLock:
    """A distributed lock on a shared resource"""
    lock_id: str
    resource_id: str
    holder_id: str
    granted_at: float = field(default_factory=time.time)
    ttl_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.granted_at + self.ttl_seconds


@dataclass
class ConsensusProposal:
    """A multi-agent consensus proposal"""
    proposal_id: str
    topic: str
    proposed_value: Any
    proposer_id: str
    votes: Dict[str, bool] = field(default_factory=dict)  # agent_id -> vote
    required_agents: Optional[List[str]] = None
    quorum_fraction: float = 0.5
    deadline: float = field(default_factory=lambda: time.time() + 30.0)
    result: Optional[bool] = None

    @property
    def is_decided(self) -> bool:
        return self.result is not None or time.time() > self.deadline

    @property
    def approval_count(self) -> int:
        return sum(1 for v in self.votes.values() if v)

    @property
    def total_votes(self) -> int:
        return len(self.votes)

    def compute_result(self) -> Optional[bool]:
        if not self.votes:
            return None
        approval_rate = self.approval_count / self.total_votes
        return approval_rate >= self.quorum_fraction


MessageHandler = Callable[[CoordinationMessage], Awaitable[None]]


class AgentCoordinationBus:
    """
    In-process multi-agent coordination bus.

    Features:
    - Priority queuing per agent (higher priority delivered first)
    - Resource lock management (exclusive + shared locks)
    - Consensus voting with configurable quorum
    - Agent capability registry for intelligent delegation
    - Dead-letter queue for undeliverable messages
    - Heartbeat tracking for agent liveness

    Usage::

        bus = AgentCoordinationBus()
        bus.register_agent("agent-1", capabilities={"code_gen"})
        bus.register_agent("agent-2", capabilities={"search", "analysis"})

        # Delegate a task
        msg = bus.create_message(
            MessageType.TASK_DELEGATION,
            sender_id="orchestrator",
            recipient_id="agent-1",
            payload={"task": "generate Python function", "spec": "..."}
        )
        await bus.send(msg)
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentRegistration] = {}
        self._queues: Dict[str, deque] = defaultdict(deque)  # agent_id -> msgs
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self._broadcast_handlers: List[MessageHandler] = []
        self._locks: Dict[str, ResourceLock] = {}
        self._proposals: Dict[str, ConsensusProposal] = {}
        self._dead_letter_queue: List[CoordinationMessage] = []
        self._sent_count: int = 0
        self._delivered_count: int = 0
        self._dlq_count: int = 0
        self._pending_futures: Dict[str, asyncio.Future] = {}

    # ──────────────────────────────────────────────
    # Agent Registry
    # ──────────────────────────────────────────────

    def register_agent(
        self,
        agent_id: str,
        capabilities: Optional[Set[str]] = None,
        max_concurrent_tasks: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentRegistration:
        """Register an agent on the bus"""
        reg = AgentRegistration(
            agent_id=agent_id,
            capabilities=capabilities or set(),
            max_concurrent_tasks=max_concurrent_tasks,
            metadata=metadata or {},
        )
        self._agents[agent_id] = reg
        return reg

    def deregister_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._queues.pop(agent_id, None)
            return True
        return False

    def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            return True
        return False

    def heartbeat(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            self._agents[agent_id].last_heartbeat = time.time()
            self._agents[agent_id].status = AgentStatus.IDLE
            return True
        return False

    def find_capable_agents(
        self,
        required_capabilities: Set[str],
        available_only: bool = True,
    ) -> List[AgentRegistration]:
        """Find agents with matching capabilities"""
        candidates = [
            a for a in self._agents.values()
            if a.can_handle(required_capabilities)
        ]
        if available_only:
            candidates = [a for a in candidates if a.is_available]
        return sorted(candidates, key=lambda a: a.current_task_count)

    def list_agents(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._agents.values()]

    # ──────────────────────────────────────────────
    # Messaging
    # ──────────────────────────────────────────────

    @staticmethod
    def create_message(
        message_type: MessageType,
        sender_id: str,
        payload: Dict[str, Any],
        recipient_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_seconds: float = 30.0,
        correlation_id: Optional[str] = None,
    ) -> CoordinationMessage:
        return CoordinationMessage(
            message_id=str(uuid4()),
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload,
            priority=priority,
            ttl_seconds=ttl_seconds,
            correlation_id=correlation_id or str(uuid4()),
        )

    async def send(self, message: CoordinationMessage) -> bool:
        """
        Send a message. Broadcasts to all agents if recipient_id is None.
        Returns True if successfully enqueued.
        """
        if message.is_expired:
            self._dlq_count += 1
            self._dead_letter_queue.append(message)
            return False

        self._sent_count += 1

        if message.recipient_id is None:
            # Broadcast
            for agent_id in list(self._agents.keys()):
                if agent_id != message.sender_id:
                    await self._enqueue(message, agent_id)
            await self._fire_broadcast_handlers(message)
            return True

        if message.recipient_id not in self._agents:
            self._dlq_count += 1
            self._dead_letter_queue.append(message)
            return False

        await self._enqueue(message, message.recipient_id)
        return True

    def subscribe(
        self,
        agent_id: str,
        handler: MessageHandler,
        message_type: Optional[MessageType] = None,
    ) -> None:
        """Register an async handler to be called when messages arrive"""
        if message_type is None:
            self._handlers[agent_id].append(handler)
        else:
            # Filter wrapper
            async def filtered(msg: CoordinationMessage) -> None:
                if msg.message_type == message_type:
                    await handler(msg)
            self._handlers[agent_id].append(filtered)

    def subscribe_broadcast(self, handler: MessageHandler) -> None:
        self._broadcast_handlers.append(handler)

    async def receive(
        self,
        agent_id: str,
        timeout: float = 0.0,
    ) -> Optional[CoordinationMessage]:
        """
        Receive the next message for an agent.
        timeout=0 is non-blocking; >0 waits up to that many seconds.
        """
        deadline = time.time() + timeout
        while True:
            msg = self._dequeue(agent_id)
            if msg:
                msg.delivered_at = time.time()
                self._delivered_count += 1
                return msg
            if timeout <= 0 or time.time() >= deadline:
                return None
            await asyncio.sleep(0.01)

    async def request_reply(
        self,
        message: CoordinationMessage,
        timeout: float = 10.0,
    ) -> Optional[CoordinationMessage]:
        """
        Send a message and wait for a correlated reply.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_futures[message.correlation_id] = future

        await self.send(message)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_futures.pop(message.correlation_id, None)

    def resolve_pending(self, correlation_id: str, reply: CoordinationMessage) -> bool:
        """Resolve a pending request_reply future"""
        future = self._pending_futures.get(correlation_id)
        if future and not future.done():
            future.set_result(reply)
            return True
        return False

    # ──────────────────────────────────────────────
    # Resource Locking
    # ──────────────────────────────────────────────

    def acquire_lock(
        self,
        resource_id: str,
        holder_id: str,
        ttl_seconds: float = 30.0,
    ) -> Optional[ResourceLock]:
        """
        Attempt to acquire an exclusive lock on a resource.
        Returns None if already locked by another holder.
        """
        existing = self._locks.get(resource_id)
        if existing and not existing.is_expired and existing.holder_id != holder_id:
            return None  # Locked by someone else

        lock = ResourceLock(
            lock_id=str(uuid4()),
            resource_id=resource_id,
            holder_id=holder_id,
            ttl_seconds=ttl_seconds,
        )
        self._locks[resource_id] = lock
        return lock

    def release_lock(self, resource_id: str, holder_id: str) -> bool:
        """Release a lock held by a specific agent"""
        lock = self._locks.get(resource_id)
        if lock and lock.holder_id == holder_id:
            del self._locks[resource_id]
            return True
        return False

    def is_locked(self, resource_id: str) -> bool:
        lock = self._locks.get(resource_id)
        if not lock:
            return False
        if lock.is_expired:
            del self._locks[resource_id]
            return False
        return True

    def list_locks(self) -> List[Dict[str, Any]]:
        # Clean expired locks first
        for rid in list(self._locks.keys()):
            if self._locks[rid].is_expired:
                del self._locks[rid]
        return [
            {
                "resource_id": lk.resource_id,
                "holder_id": lk.holder_id,
                "age_seconds": round(time.time() - lk.granted_at, 2),
                "ttl_seconds": lk.ttl_seconds,
            }
            for lk in self._locks.values()
        ]

    # ──────────────────────────────────────────────
    # Consensus
    # ──────────────────────────────────────────────

    def propose_consensus(
        self,
        topic: str,
        proposed_value: Any,
        proposer_id: str,
        required_agents: Optional[List[str]] = None,
        quorum_fraction: float = 0.5,
        deadline_seconds: float = 30.0,
    ) -> ConsensusProposal:
        """Create a new consensus proposal"""
        proposal = ConsensusProposal(
            proposal_id=str(uuid4()),
            topic=topic,
            proposed_value=proposed_value,
            proposer_id=proposer_id,
            required_agents=required_agents,
            quorum_fraction=quorum_fraction,
            deadline=time.time() + deadline_seconds,
        )
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def cast_vote(
        self,
        proposal_id: str,
        agent_id: str,
        vote: bool,
    ) -> Optional[bool]:
        """
        Cast a vote on a proposal.
        Returns the final result if quorum is reached, else None.
        """
        proposal = self._proposals.get(proposal_id)
        if not proposal or proposal.is_decided:
            return None
        proposal.votes[agent_id] = vote

        # Check if we have all required votes or exceeded quorum
        eligible = (
            set(proposal.required_agents)
            if proposal.required_agents
            else set(self._agents.keys())
        )
        all_voted = eligible.issubset(set(proposal.votes.keys()))
        if all_voted or proposal.total_votes >= max(1, len(eligible)):
            proposal.result = proposal.compute_result()
            return proposal.result
        return None

    def get_consensus_result(self, proposal_id: str) -> Optional[ConsensusProposal]:
        return self._proposals.get(proposal_id)

    # ──────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "registered_agents": len(self._agents),
            "available_agents": sum(1 for a in self._agents.values() if a.is_available),
            "messages_sent": self._sent_count,
            "messages_delivered": self._delivered_count,
            "dead_letter_count": self._dlq_count,
            "active_locks": len(self._locks),
            "active_proposals": len(self._proposals),
            "pending_futures": len(self._pending_futures),
        }

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    async def _enqueue(self, message: CoordinationMessage, agent_id: str) -> None:
        """Enqueue a message in priority order"""
        queue = self._queues[agent_id]
        # Insert in priority order (highest priority first)
        inserted = False
        for i, existing_msg in enumerate(queue):
            if message.priority.value > existing_msg.priority.value:
                # Convert deque to list, insert, convert back
                lst = list(queue)
                lst.insert(i, message)
                self._queues[agent_id] = deque(lst)
                inserted = True
                break
        if not inserted:
            queue.append(message)

        # Fire async handlers
        for handler in self._handlers.get(agent_id, []):
            try:
                asyncio.create_task(handler(message))
            except Exception:  # noqa: BLE001
                pass

    def _dequeue(self, agent_id: str) -> Optional[CoordinationMessage]:
        queue = self._queues.get(agent_id)
        while queue:
            msg = queue.popleft()
            if not msg.is_expired:
                return msg
            # Expired - move to DLQ
            self._dead_letter_queue.append(msg)
            self._dlq_count += 1
        return None

    async def _fire_broadcast_handlers(self, message: CoordinationMessage) -> None:
        for handler in self._broadcast_handlers:
            try:
                asyncio.create_task(handler(message))
            except Exception:  # noqa: BLE001
                pass


# Global singleton
_coordination_bus: Optional[AgentCoordinationBus] = None


def get_coordination_bus() -> AgentCoordinationBus:
    global _coordination_bus
    if _coordination_bus is None:
        _coordination_bus = AgentCoordinationBus()
    return _coordination_bus
