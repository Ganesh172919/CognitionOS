"""
Multi-Agent Coordination System — Orchestrates multiple AI agents with
consensus mechanisms, task delegation, federated execution, and inter-agent
communication protocols.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────── Enums ───────────────────────────────────


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    SPECIALIST = "specialist"
    MONITOR = "monitor"
    PLANNER = "planner"


class AgentStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    INITIALIZING = "initializing"


class MessageType(str, Enum):
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"
    CONSENSUS_VOTE = "consensus_vote"
    DELEGATION = "delegation"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"


class ConsensusAlgorithm(str, Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    RAFT = "raft"
    BYZANTINE_FAULT_TOLERANT = "byzantine_fault_tolerant"
    UNANIMOUS = "unanimous"


class DelegationStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_MATCH = "capability_match"
    PRIORITY_BASED = "priority_based"
    RANDOM = "random"


# ────────────────────────────── Data structures ──────────────────────────────


@dataclass
class AgentMessage:
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    correlation_id: Optional[str]
    priority: int = 5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentCapability:
    capability_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    avg_latency_ms: float
    success_rate: float
    cost_per_call: float
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "cost_per_call": self.cost_per_call,
            "tags": self.tags,
        }


@dataclass
class AgentDescriptor:
    agent_id: str
    name: str
    role: AgentRole
    capabilities: List[AgentCapability]
    status: AgentStatus
    current_load: float
    max_load: float
    trust_score: float
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def available_capacity(self) -> float:
        return max(0.0, self.max_load - self.current_load)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "status": self.status.value,
            "current_load": self.current_load,
            "max_load": self.max_load,
            "available_capacity": self.available_capacity,
            "trust_score": self.trust_score,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "metadata": self.metadata,
        }


@dataclass
class DelegatedTask:
    task_id: str
    description: str
    required_capabilities: List[str]
    delegated_to: Optional[str]
    delegated_by: str
    priority: int
    payload: Dict[str, Any]
    deadline: Optional[datetime]
    status: str = "pending"
    result: Optional[Any] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "required_capabilities": self.required_capabilities,
            "delegated_to": self.delegated_to,
            "delegated_by": self.delegated_by,
            "priority": self.priority,
            "payload": self.payload,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ConsensusRound:
    round_id: str
    proposal: Dict[str, Any]
    proposer_id: str
    algorithm: ConsensusAlgorithm
    votes: Dict[str, Any]
    status: str
    result: Optional[Dict[str, Any]]
    quorum_size: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "proposal": self.proposal,
            "proposer_id": self.proposer_id,
            "algorithm": self.algorithm.value,
            "votes": self.votes,
            "status": self.status,
            "result": self.result,
            "quorum_size": self.quorum_size,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


# ─────────────────────────── Agent Registry ─────────────────────────────────


class AgentRegistry:
    """
    Central registry for all active agents. Manages registration, discovery,
    health tracking, and capability indexing.
    """

    def __init__(self):
        self._agents: Dict[str, AgentDescriptor] = {}
        self._capability_index: Dict[str, List[str]] = defaultdict(list)
        self._role_index: Dict[AgentRole, List[str]] = defaultdict(list)
        self._heartbeat_timeout_s = 30.0

    def register(self, descriptor: AgentDescriptor) -> None:
        self._agents[descriptor.agent_id] = descriptor
        for cap in descriptor.capabilities:
            self._capability_index[cap.name].append(descriptor.agent_id)
            for tag in cap.tags:
                self._capability_index[f"tag:{tag}"].append(descriptor.agent_id)
        self._role_index[descriptor.role].append(descriptor.agent_id)

    def deregister(self, agent_id: str) -> bool:
        descriptor = self._agents.pop(agent_id, None)
        if descriptor is None:
            return False
        for cap in descriptor.capabilities:
            self._capability_index[cap.name] = [
                aid for aid in self._capability_index[cap.name] if aid != agent_id
            ]
        self._role_index[descriptor.role] = [
            aid for aid in self._role_index[descriptor.role] if aid != agent_id
        ]
        return True

    def heartbeat(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        if agent is None:
            return False
        agent.last_heartbeat = datetime.now(timezone.utc)
        if agent.status == AgentStatus.FAILED:
            agent.status = AgentStatus.IDLE
        return True

    def update_load(self, agent_id: str, load_delta: float) -> bool:
        agent = self._agents.get(agent_id)
        if agent is None:
            return False
        agent.current_load = min(
            agent.max_load, max(0.0, agent.current_load + load_delta)
        )
        if agent.current_load >= agent.max_load:
            agent.status = AgentStatus.BUSY
        elif agent.status == AgentStatus.BUSY:
            agent.status = AgentStatus.IDLE
        return True

    def find_by_capability(
        self, capability_name: str, available_only: bool = True
    ) -> List[AgentDescriptor]:
        agent_ids = self._capability_index.get(capability_name, [])
        agents = [self._agents[aid] for aid in agent_ids if aid in self._agents]
        if available_only:
            agents = [
                a
                for a in agents
                if a.status not in (AgentStatus.UNAVAILABLE, AgentStatus.FAILED)
                and a.available_capacity > 0
            ]
        return agents

    def find_by_role(self, role: AgentRole) -> List[AgentDescriptor]:
        agent_ids = self._role_index.get(role, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def get_agent(self, agent_id: str) -> Optional[AgentDescriptor]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentDescriptor]:
        return list(self._agents.values())

    def get_healthy_agents(self) -> List[AgentDescriptor]:
        now_ts = time.monotonic()
        return [
            a
            for a in self._agents.values()
            if a.status not in (AgentStatus.FAILED, AgentStatus.UNAVAILABLE)
        ]

    def get_registry_stats(self) -> Dict[str, Any]:
        agents = list(self._agents.values())
        by_role: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        for a in agents:
            by_role[a.role.value] += 1
            by_status[a.status.value] += 1
        return {
            "total_agents": len(agents),
            "by_role": dict(by_role),
            "by_status": dict(by_status),
            "total_capabilities": sum(len(a.capabilities) for a in agents),
            "avg_trust_score": (
                sum(a.trust_score for a in agents) / len(agents) if agents else 0.0
            ),
        }


# ─────────────────────────── Message Bus ────────────────────────────────────


class AgentMessageBus:
    """
    Asynchronous message bus for inter-agent communication with priority
    queuing, routing, and delivery guarantees.
    """

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._broadcast_subscribers: Set[str] = set()
        self._message_log: List[AgentMessage] = []
        self._max_log_size = 500

    def register_agent(self, agent_id: str) -> None:
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue(maxsize=100)

    def deregister_agent(self, agent_id: str) -> None:
        self._queues.pop(agent_id, None)
        self._broadcast_subscribers.discard(agent_id)

    async def send(self, message: AgentMessage) -> bool:
        queue = self._queues.get(message.receiver_id)
        if queue is None:
            return False
        try:
            queue.put_nowait(message)
            self._log_message(message)
            return True
        except asyncio.QueueFull:
            return False

    async def broadcast(
        self, sender_id: str, message_type: MessageType, payload: Dict[str, Any]
    ) -> int:
        sent = 0
        for agent_id in list(self._broadcast_subscribers):
            if agent_id == sender_id:
                continue
            msg = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id=sender_id,
                receiver_id=agent_id,
                message_type=message_type,
                payload=payload,
                correlation_id=None,
            )
            if await self.send(msg):
                sent += 1
        return sent

    async def receive(self, agent_id: str, timeout_s: float = 1.0) -> Optional[AgentMessage]:
        queue = self._queues.get(agent_id)
        if queue is None:
            return None
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=timeout_s)
            return msg
        except asyncio.TimeoutError:
            return None

    def subscribe_to_broadcast(self, agent_id: str) -> None:
        self._broadcast_subscribers.add(agent_id)

    def get_queue_depth(self, agent_id: str) -> int:
        queue = self._queues.get(agent_id)
        return queue.qsize() if queue else 0

    def get_message_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._message_log[-limit:]]

    def _log_message(self, message: AgentMessage) -> None:
        self._message_log.append(message)
        if len(self._message_log) > self._max_log_size:
            self._message_log = self._message_log[-self._max_log_size :]

    def get_bus_stats(self) -> Dict[str, Any]:
        return {
            "registered_agents": len(self._queues),
            "broadcast_subscribers": len(self._broadcast_subscribers),
            "total_messages_logged": len(self._message_log),
            "queue_depths": {
                aid: q.qsize() for aid, q in self._queues.items()
            },
        }


# ─────────────────────────── Task Delegator ─────────────────────────────────


class TaskDelegator:
    """
    Intelligent task delegation with multiple strategies: round-robin,
    load-balanced, capability-matched, and priority-based.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._round_robin_index: Dict[str, int] = defaultdict(int)
        self._delegated_tasks: Dict[str, DelegatedTask] = {}
        self._completed_count: int = 0
        self._failed_count: int = 0

    def delegate(
        self,
        task: DelegatedTask,
        strategy: DelegationStrategy = DelegationStrategy.CAPABILITY_MATCH,
    ) -> Optional[str]:
        candidates = self._find_candidates(task)
        if not candidates:
            return None

        selected = self._select_agent(candidates, task, strategy)
        if selected is None:
            return None

        task.delegated_to = selected.agent_id
        task.status = "delegated"
        self._delegated_tasks[task.task_id] = task
        self.registry.update_load(selected.agent_id, 0.2)
        return selected.agent_id

    def complete_task(
        self,
        task_id: str,
        result: Any,
        success: bool = True,
    ) -> bool:
        task = self._delegated_tasks.get(task_id)
        if task is None:
            return False
        task.status = "completed" if success else "failed"
        task.result = result
        task.completed_at = datetime.now(timezone.utc)
        if task.delegated_to:
            self.registry.update_load(task.delegated_to, -0.2)
        if success:
            self._completed_count += 1
        else:
            self._failed_count += 1
        return True

    def get_task(self, task_id: str) -> Optional[DelegatedTask]:
        return self._delegated_tasks.get(task_id)

    def list_tasks(self, status_filter: Optional[str] = None) -> List[DelegatedTask]:
        tasks = list(self._delegated_tasks.values())
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        return tasks

    def get_delegation_stats(self) -> Dict[str, Any]:
        tasks = list(self._delegated_tasks.values())
        by_status: Dict[str, int] = defaultdict(int)
        for t in tasks:
            by_status[t.status] += 1
        return {
            "total_delegated": len(tasks),
            "by_status": dict(by_status),
            "completed": self._completed_count,
            "failed": self._failed_count,
            "success_rate": (
                self._completed_count / max(self._completed_count + self._failed_count, 1)
            ),
        }

    def _find_candidates(self, task: DelegatedTask) -> List[AgentDescriptor]:
        if task.required_capabilities:
            # Find agents with all required capabilities
            candidate_sets: List[Set[str]] = []
            for cap in task.required_capabilities:
                agents = self.registry.find_by_capability(cap, available_only=True)
                candidate_sets.append({a.agent_id for a in agents})
            if not candidate_sets:
                return []
            common_ids = candidate_sets[0].intersection(*candidate_sets[1:])
            return [
                a
                for a in self.registry.list_agents()
                if a.agent_id in common_ids
            ]
        # No capability requirement: any idle executor
        return [
            a
            for a in self.registry.find_by_role(AgentRole.EXECUTOR)
            if a.status == AgentStatus.IDLE
        ]

    def _select_agent(
        self,
        candidates: List[AgentDescriptor],
        task: DelegatedTask,
        strategy: DelegationStrategy,
    ) -> Optional[AgentDescriptor]:
        if not candidates:
            return None

        if strategy == DelegationStrategy.LOAD_BALANCED:
            return max(candidates, key=lambda a: a.available_capacity)

        if strategy == DelegationStrategy.PRIORITY_BASED:
            # High-priority tasks go to high-trust agents
            return max(candidates, key=lambda a: a.trust_score)

        if strategy == DelegationStrategy.ROUND_ROBIN:
            cap_key = ",".join(sorted(task.required_capabilities))
            idx = self._round_robin_index[cap_key] % len(candidates)
            self._round_robin_index[cap_key] += 1
            return candidates[idx]

        if strategy == DelegationStrategy.CAPABILITY_MATCH:
            # Score candidates by capability match quality
            scored = []
            for a in candidates:
                agent_cap_names = {c.name for c in a.capabilities}
                match = len(set(task.required_capabilities) & agent_cap_names)
                score = (
                    match * 0.4
                    + a.trust_score * 0.3
                    + a.available_capacity / max(a.max_load, 1) * 0.3
                )
                scored.append((a, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]

        # Default: first available
        return candidates[0]


# ─────────────────────────── Consensus Engine ───────────────────────────────


class ConsensusEngine:
    """
    Implements distributed consensus mechanisms for multi-agent decision making.
    Supports majority vote, weighted vote, and Byzantine fault tolerance.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._rounds: Dict[str, ConsensusRound] = {}

    def initiate_round(
        self,
        proposal: Dict[str, Any],
        proposer_id: str,
        algorithm: ConsensusAlgorithm = ConsensusAlgorithm.MAJORITY_VOTE,
        include_roles: Optional[List[AgentRole]] = None,
    ) -> ConsensusRound:
        participating_agents = self._select_participants(include_roles)
        quorum_size = max(1, int(len(participating_agents) * 0.51))

        round_obj = ConsensusRound(
            round_id=str(uuid.uuid4()),
            proposal=proposal,
            proposer_id=proposer_id,
            algorithm=algorithm,
            votes={},
            status="open",
            result=None,
            quorum_size=quorum_size,
        )
        self._rounds[round_obj.round_id] = round_obj
        return round_obj

    def cast_vote(
        self,
        round_id: str,
        voter_id: str,
        vote: bool,
        confidence: float = 1.0,
        rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        round_obj = self._rounds.get(round_id)
        if round_obj is None:
            return {"success": False, "error": "Round not found"}
        if round_obj.status != "open":
            return {"success": False, "error": f"Round is {round_obj.status}"}

        agent = self.registry.get_agent(voter_id)
        trust_weight = agent.trust_score if agent else 1.0

        round_obj.votes[voter_id] = {
            "vote": vote,
            "confidence": confidence,
            "trust_weight": trust_weight,
            "rationale": rationale,
            "cast_at": datetime.now(timezone.utc).isoformat(),
        }

        # Check if consensus reached
        result = self._check_consensus(round_obj)
        if result is not None:
            round_obj.status = "resolved"
            round_obj.result = result
            round_obj.resolved_at = datetime.now(timezone.utc)

        return {"success": True, "round_id": round_id, "votes_cast": len(round_obj.votes)}

    def get_round(self, round_id: str) -> Optional[ConsensusRound]:
        return self._rounds.get(round_id)

    def list_rounds(self, status_filter: Optional[str] = None) -> List[ConsensusRound]:
        rounds = list(self._rounds.values())
        if status_filter:
            rounds = [r for r in rounds if r.status == status_filter]
        return rounds

    def _select_participants(
        self, roles: Optional[List[AgentRole]]
    ) -> List[AgentDescriptor]:
        if roles:
            agents = []
            for role in roles:
                agents.extend(self.registry.find_by_role(role))
            return agents
        return self.registry.get_healthy_agents()

    def _check_consensus(
        self, round_obj: ConsensusRound
    ) -> Optional[Dict[str, Any]]:
        votes = round_obj.votes
        if len(votes) < round_obj.quorum_size:
            return None

        if round_obj.algorithm == ConsensusAlgorithm.MAJORITY_VOTE:
            yes_count = sum(1 for v in votes.values() if v["vote"])
            no_count = len(votes) - yes_count
            accepted = yes_count > no_count
            return {
                "accepted": accepted,
                "yes_votes": yes_count,
                "no_votes": no_count,
                "total_votes": len(votes),
            }

        if round_obj.algorithm == ConsensusAlgorithm.WEIGHTED_VOTE:
            yes_weight = sum(
                v["trust_weight"] * v["confidence"]
                for v in votes.values()
                if v["vote"]
            )
            no_weight = sum(
                v["trust_weight"] * v["confidence"]
                for v in votes.values()
                if not v["vote"]
            )
            accepted = yes_weight > no_weight
            return {
                "accepted": accepted,
                "yes_weight": round(yes_weight, 4),
                "no_weight": round(no_weight, 4),
                "total_votes": len(votes),
            }

        if round_obj.algorithm == ConsensusAlgorithm.UNANIMOUS:
            all_yes = all(v["vote"] for v in votes.values())
            return {"accepted": all_yes, "total_votes": len(votes)}

        if round_obj.algorithm == ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT:
            # 2/3 majority required for BFT
            yes_count = sum(1 for v in votes.values() if v["vote"])
            bft_threshold = int(len(votes) * 2 / 3) + 1
            accepted = yes_count >= bft_threshold
            return {
                "accepted": accepted,
                "yes_votes": yes_count,
                "bft_threshold": bft_threshold,
                "total_votes": len(votes),
            }

        return None

    def get_consensus_stats(self) -> Dict[str, Any]:
        rounds = list(self._rounds.values())
        by_algorithm: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        accepted_count = 0
        for r in rounds:
            by_algorithm[r.algorithm.value] += 1
            by_status[r.status] += 1
            if r.result and r.result.get("accepted"):
                accepted_count += 1
        resolved = by_status.get("resolved", 0)
        return {
            "total_rounds": len(rounds),
            "by_algorithm": dict(by_algorithm),
            "by_status": dict(by_status),
            "acceptance_rate": accepted_count / max(resolved, 1),
        }


# ─────────────────────────── Performance Tracker ────────────────────────────


class AgentPerformanceTracker:
    """
    Tracks per-agent performance metrics: success rates, latency,
    throughput, and anomaly detection.
    """

    def __init__(self):
        self._metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "total_latency_ms": 0.0,
                "min_latency_ms": float("inf"),
                "max_latency_ms": 0.0,
                "recent_latencies": [],
                "error_types": defaultdict(int),
                "last_activity": None,
            }
        )

    def record_completion(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float,
        error_type: Optional[str] = None,
    ) -> None:
        m = self._metrics[agent_id]
        if success:
            m["tasks_completed"] += 1
        else:
            m["tasks_failed"] += 1
            if error_type:
                m["error_types"][error_type] += 1
        m["total_latency_ms"] += latency_ms
        m["min_latency_ms"] = min(m["min_latency_ms"], latency_ms)
        m["max_latency_ms"] = max(m["max_latency_ms"], latency_ms)
        m["recent_latencies"].append(latency_ms)
        if len(m["recent_latencies"]) > 100:
            m["recent_latencies"] = m["recent_latencies"][-100:]
        m["last_activity"] = datetime.now(timezone.utc).isoformat()

    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        m = self._metrics[agent_id]
        total = m["tasks_completed"] + m["tasks_failed"]
        recent = m["recent_latencies"]
        avg_recent = sum(recent) / len(recent) if recent else 0.0
        return {
            "agent_id": agent_id,
            "tasks_completed": m["tasks_completed"],
            "tasks_failed": m["tasks_failed"],
            "success_rate": m["tasks_completed"] / max(total, 1),
            "avg_latency_ms": m["total_latency_ms"] / max(total, 1),
            "avg_recent_latency_ms": round(avg_recent, 2),
            "min_latency_ms": m["min_latency_ms"] if m["min_latency_ms"] != float("inf") else 0.0,
            "max_latency_ms": m["max_latency_ms"],
            "error_types": dict(m["error_types"]),
            "last_activity": m["last_activity"],
        }

    def get_fleet_metrics(self) -> Dict[str, Any]:
        all_agent_ids = list(self._metrics.keys())
        if not all_agent_ids:
            return {"total_agents_tracked": 0}
        per_agent = [self.get_agent_metrics(aid) for aid in all_agent_ids]
        total_completed = sum(m["tasks_completed"] for m in per_agent)
        total_failed = sum(m["tasks_failed"] for m in per_agent)
        avg_success = (
            sum(m["success_rate"] for m in per_agent) / len(per_agent)
        )
        return {
            "total_agents_tracked": len(all_agent_ids),
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "fleet_success_rate": round(avg_success, 4),
            "per_agent": per_agent,
        }


# ──────────────────────── Multi-Agent Coordinator ───────────────────────────


class MultiAgentCoordinator:
    """
    Master coordinator integrating registry, message bus, delegation,
    consensus, and performance tracking into a unified multi-agent system.
    """

    def __init__(self):
        self.registry = AgentRegistry()
        self.message_bus = AgentMessageBus()
        self.delegator = TaskDelegator(self.registry)
        self.consensus = ConsensusEngine(self.registry)
        self.performance = AgentPerformanceTracker()

    def register_agent(
        self,
        name: str,
        role: AgentRole,
        capabilities: List[Dict[str, Any]],
        max_load: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        agent_id = str(uuid.uuid4())
        cap_objects = [
            AgentCapability(
                capability_id=str(uuid.uuid4()),
                name=c.get("name", "unknown"),
                description=c.get("description", ""),
                input_schema=c.get("input_schema", {}),
                output_schema=c.get("output_schema", {}),
                avg_latency_ms=c.get("avg_latency_ms", 100.0),
                success_rate=c.get("success_rate", 0.95),
                cost_per_call=c.get("cost_per_call", 0.001),
                tags=c.get("tags", []),
            )
            for c in capabilities
        ]
        descriptor = AgentDescriptor(
            agent_id=agent_id,
            name=name,
            role=role,
            capabilities=cap_objects,
            status=AgentStatus.IDLE,
            current_load=0.0,
            max_load=max_load,
            trust_score=0.8,
            metadata=metadata or {},
        )
        self.registry.register(descriptor)
        self.message_bus.register_agent(agent_id)
        self.message_bus.subscribe_to_broadcast(agent_id)
        return agent_id

    def deregister_agent(self, agent_id: str) -> bool:
        self.registry.deregister(agent_id)
        self.message_bus.deregister_agent(agent_id)
        return True

    async def delegate_task(
        self,
        description: str,
        required_capabilities: List[str],
        payload: Dict[str, Any],
        delegated_by: str = "system",
        strategy: DelegationStrategy = DelegationStrategy.CAPABILITY_MATCH,
        priority: int = 5,
    ) -> Dict[str, Any]:
        task = DelegatedTask(
            task_id=str(uuid.uuid4()),
            description=description,
            required_capabilities=required_capabilities,
            delegated_to=None,
            delegated_by=delegated_by,
            priority=priority,
            payload=payload,
            deadline=None,
        )
        assigned_agent_id = self.delegator.delegate(task, strategy)
        if assigned_agent_id is None:
            return {
                "success": False,
                "task_id": task.task_id,
                "error": "No available agent with required capabilities",
            }

        # Send task assignment message
        msg = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=delegated_by,
            receiver_id=assigned_agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={"task": task.to_dict()},
            correlation_id=task.task_id,
            priority=priority,
        )
        await self.message_bus.send(msg)

        return {
            "success": True,
            "task_id": task.task_id,
            "assigned_to": assigned_agent_id,
            "strategy_used": strategy.value,
        }

    async def run_consensus(
        self,
        proposal: Dict[str, Any],
        proposer_id: str,
        algorithm: ConsensusAlgorithm = ConsensusAlgorithm.MAJORITY_VOTE,
        auto_vote: bool = True,
    ) -> Dict[str, Any]:
        round_obj = self.consensus.initiate_round(proposal, proposer_id, algorithm)

        if auto_vote:
            # Simulate agents voting based on trust scores
            healthy_agents = self.registry.get_healthy_agents()
            for agent in healthy_agents[:10]:  # Limit to 10 agents
                vote = agent.trust_score > 0.5
                confidence = agent.trust_score
                self.consensus.cast_vote(
                    round_obj.round_id,
                    agent.agent_id,
                    vote,
                    confidence,
                    f"Agent {agent.name} auto-vote",
                )

        round_obj = self.consensus.get_round(round_obj.round_id)
        return round_obj.to_dict() if round_obj else {}

    async def broadcast_status(
        self,
        sender_id: str,
        status_data: Dict[str, Any],
    ) -> int:
        return await self.message_bus.broadcast(
            sender_id, MessageType.STATUS_UPDATE, status_data
        )

    def get_system_health(self) -> Dict[str, Any]:
        registry_stats = self.registry.get_registry_stats()
        bus_stats = self.message_bus.get_bus_stats()
        delegation_stats = self.delegator.get_delegation_stats()
        consensus_stats = self.consensus.get_consensus_stats()
        fleet_metrics = self.performance.get_fleet_metrics()
        return {
            "registry": registry_stats,
            "message_bus": bus_stats,
            "delegation": delegation_stats,
            "consensus": consensus_stats,
            "fleet_performance": fleet_metrics,
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.registry.list_agents()]

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        a = self.registry.get_agent(agent_id)
        return a.to_dict() if a else None
