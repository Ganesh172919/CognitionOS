"""
Agent Run Domain - Entities

Persisted autonomous single-agent runtime execution model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class RunStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepType(str, Enum):
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    EVALUATION = "evaluation"
    TOOL = "tool"
    CODEGEN = "codegen"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactKind(str, Enum):
    CODE = "code"
    PATCH = "patch"
    TEST = "test"
    LOG = "log"
    REPORT = "report"
    BINARY = "binary"


@dataclass
class AgentRun:
    id: UUID
    requirement: str
    status: RunStatus
    tenant_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    budgets: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @staticmethod
    def create(
        requirement: str,
        tenant_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        budgets: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentRun":
        now = datetime.utcnow()
        return AgentRun(
            id=uuid4(),
            requirement=requirement,
            status=RunStatus.CREATED,
            tenant_id=tenant_id,
            user_id=user_id,
            budgets=budgets or {},
            usage={},
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

    def mark_queued(self) -> None:
        self.status = RunStatus.QUEUED
        self.updated_at = datetime.utcnow()

    def start(self) -> None:
        self.status = RunStatus.RUNNING
        now = datetime.utcnow()
        self.started_at = self.started_at or now
        self.updated_at = now

    def start_validation(self) -> None:
        self.status = RunStatus.VALIDATING
        self.updated_at = datetime.utcnow()

    def complete(self) -> None:
        self.status = RunStatus.COMPLETED
        now = datetime.utcnow()
        self.completed_at = now
        self.updated_at = now

    def fail(self, error: str) -> None:
        self.status = RunStatus.FAILED
        now = datetime.utcnow()
        self.error = error
        self.completed_at = now
        self.updated_at = now

    def cancel(self, reason: str = "cancelled") -> None:
        self.status = RunStatus.CANCELLED
        now = datetime.utcnow()
        self.error = reason
        self.completed_at = now
        self.updated_at = now

    def is_terminal(self) -> bool:
        return self.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}


@dataclass
class AgentRunStep:
    id: UUID
    run_id: UUID
    step_index: int
    step_type: StepType
    status: StepStatus
    input: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    cost_usd: Decimal = Decimal("0")
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def create(
        run_id: UUID,
        step_index: int,
        step_type: StepType,
        input: Optional[Dict[str, Any]] = None,
    ) -> "AgentRunStep":
        return AgentRunStep(
            id=uuid4(),
            run_id=run_id,
            step_index=step_index,
            step_type=step_type,
            status=StepStatus.PENDING,
            input=input or {},
        )

    def start(self) -> None:
        self.status = StepStatus.RUNNING
        self.started_at = datetime.utcnow()

    def complete(
        self,
        output: Optional[Dict[str, Any]] = None,
        tokens_used: int = 0,
        cost_usd: Decimal = Decimal("0"),
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.status = StepStatus.COMPLETED
        self.output = output
        self.tokens_used = tokens_used
        self.cost_usd = cost_usd
        self.tool_calls = tool_calls or self.tool_calls
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)

    def fail(self, error: str) -> None:
        self.status = StepStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)


@dataclass
class AgentRunArtifact:
    id: UUID
    run_id: UUID
    kind: ArtifactKind
    name: str
    content_type: str = "text/plain"
    step_id: Optional[UUID] = None
    content_text: Optional[str] = None
    content_bytes: Optional[bytes] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    storage_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def create_text(
        run_id: UUID,
        kind: ArtifactKind,
        name: str,
        content_text: str,
        *,
        step_id: Optional[UUID] = None,
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentRunArtifact":
        blob = content_text.encode("utf-8")
        import hashlib

        return AgentRunArtifact(
            id=uuid4(),
            run_id=run_id,
            step_id=step_id,
            kind=kind,
            name=name,
            content_type=content_type,
            content_text=content_text,
            content_bytes=None,
            sha256=hashlib.sha256(blob).hexdigest(),
            size_bytes=len(blob),
            metadata=metadata or {},
        )


@dataclass
class AgentRunEvaluation:
    id: UUID
    run_id: UUID
    success: bool
    confidence: float
    quality_scores: Dict[str, Any] = field(default_factory=dict)
    policy_violations: List[Dict[str, Any]] = field(default_factory=list)
    retry_plan: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def create(
        run_id: UUID,
        *,
        success: bool,
        confidence: float,
        quality_scores: Optional[Dict[str, Any]] = None,
        policy_violations: Optional[List[Dict[str, Any]]] = None,
        retry_plan: Optional[Dict[str, Any]] = None,
    ) -> "AgentRunEvaluation":
        return AgentRunEvaluation(
            id=uuid4(),
            run_id=run_id,
            success=success,
            confidence=max(0.0, min(1.0, confidence)),
            quality_scores=quality_scores or {},
            policy_violations=policy_violations or [],
            retry_plan=retry_plan or {},
        )


@dataclass
class AgentRunMemoryLink:
    id: UUID
    run_id: UUID
    memory_id: UUID
    memory_tier: str
    relation: str = "used"
    created_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def create(
        run_id: UUID,
        memory_id: UUID,
        memory_tier: str,
        relation: str = "used",
    ) -> "AgentRunMemoryLink":
        return AgentRunMemoryLink(
            id=uuid4(),
            run_id=run_id,
            memory_id=memory_id,
            memory_tier=memory_tier,
            relation=relation,
        )

