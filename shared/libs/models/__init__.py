"""
Shared data models for CognitionOS.

All models use Pydantic for validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


# ============================================================================
# Enums
# ============================================================================

class AgentRole(str, Enum):
    PLANNER = "planner"
    REASONER = "reasoner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"


class AgentState(str, Enum):
    CREATED = "created"
    IDLE = "idle"
    ASSIGNED = "assigned"
    REASONING = "reasoning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    RETRY = "retry"
    TERMINATED = "terminated"
    FAILED = "failed"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MemoryType(str, Enum):
    FACT = "fact"
    PATTERN = "pattern"
    PREFERENCE = "preference"
    SKILL = "skill"


class MemoryScope(str, Enum):
    USER = "user"
    SESSION = "session"


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ERROR = "error"


class EventType(str, Enum):
    TASK_START = "task_start"
    TASK_END = "task_end"
    TOOL_EXEC = "tool_exec"
    DECISION = "decision"
    ERROR = "error"


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    AGENT = "agent"


# ============================================================================
# Base Models
# ============================================================================

class BaseTimestampModel(BaseModel):
    """Base model with timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class BaseIDModel(BaseTimestampModel):
    """Base model with ID and timestamps."""
    id: UUID = Field(default_factory=uuid4)


# ============================================================================
# User Models
# ============================================================================

class User(BaseIDModel):
    """User account model."""
    email: str
    username: str
    password_hash: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower()


class UserPreferences(BaseModel):
    """User preferences model."""
    user_id: UUID
    language: str = "en"
    timezone: str = "UTC"
    theme: str = "light"
    notification_enabled: bool = True
    preferences: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Agent Models
# ============================================================================

class BudgetLimits(BaseModel):
    """Budget constraints for agent execution."""
    max_tokens: int = 32000
    max_cost_usd: float = 1.0
    max_time_seconds: int = 300
    max_tool_executions: int = 10


class BudgetUsed(BaseModel):
    """Tracking of budget consumption."""
    tokens_used: int = 0
    cost_usd: float = 0.0
    time_seconds: float = 0.0
    tool_executions: int = 0

    def within_limits(self, limits: BudgetLimits) -> bool:
        """Check if usage is within limits."""
        return (
            self.tokens_used <= limits.max_tokens and
            self.cost_usd <= limits.max_cost_usd and
            self.time_seconds <= limits.max_time_seconds and
            self.tool_executions <= limits.max_tool_executions
        )


class ToolDefinition(BaseModel):
    """Definition of a tool that agents can use."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_permissions: List[str] = Field(default_factory=list)
    timeout_seconds: int = 30
    is_destructive: bool = False


class ModelConfig(BaseModel):
    """LLM model configuration."""
    provider: str  # "openai", "anthropic", "local"
    model_name: str  # "gpt-4", "claude-3-opus", etc.
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0


class AgentDefinition(BaseIDModel):
    """Agent type definition registered in the system."""
    name: str
    role: AgentRole
    version: str
    description: str
    capabilities: List[str]
    tools: List[ToolDefinition]
    model_config: ModelConfig
    default_budget: BudgetLimits
    system_prompt: str


class AgentInstance(BaseIDModel):
    """Running instance of an agent."""
    definition_id: UUID
    user_id: UUID
    state: AgentState
    current_task_id: Optional[UUID] = None
    budget_limits: BudgetLimits
    budget_used: BudgetUsed = Field(default_factory=BudgetUsed)
    last_active: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Task Models
# ============================================================================

class Task(BaseIDModel):
    """Individual task in the system."""
    user_id: UUID
    session_id: UUID
    goal_id: UUID
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    required_capabilities: List[str] = Field(default_factory=list)
    dependencies: List[UUID] = Field(default_factory=list)  # Task IDs
    assigned_agent_id: Optional[UUID] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


class Goal(BaseIDModel):
    """High-level user goal."""
    user_id: UUID
    description: str
    task_ids: List[UUID] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    completed_at: Optional[datetime] = None


# ============================================================================
# Memory Models
# ============================================================================

class Memory(BaseIDModel):
    """Long-term memory entry."""
    user_id: UUID
    content: str
    embedding: Optional[List[float]] = None
    memory_type: MemoryType
    scope: MemoryScope = MemoryScope.USER
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str  # Where this memory came from
    confidence: float = 1.0  # 0.0 to 1.0
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    version: int = 1
    is_sensitive: bool = False
    deleted: bool = False

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class ShortTermMemory(BaseModel):
    """Session-level memory."""
    session_id: UUID
    user_id: UUID
    goal: str
    context_summary: str
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime


class EpisodicMemory(BaseIDModel):
    """Execution history entry."""
    user_id: UUID
    session_id: UUID
    task_id: UUID
    agent_id: UUID
    agent_role: AgentRole
    event_type: EventType
    input_summary: str
    output_summary: Optional[str] = None
    reasoning: Optional[str] = None
    success: bool
    cost: float = 0.0
    duration_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Message Models
# ============================================================================

class AgentMessage(BaseIDModel):
    """Message passed between agents."""
    sender_id: UUID
    receiver_id: Optional[UUID] = None  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    correlation_id: Optional[UUID] = None  # For request-response pairing


# ============================================================================
# Audit Models
# ============================================================================

class AuditLog(BaseIDModel):
    """Audit trail entry."""
    user_id: UUID
    agent_id: Optional[UUID] = None
    action: str
    resource_type: str
    resource_id: UUID
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


# ============================================================================
# API Models
# ============================================================================

class APIRequest(BaseModel):
    """API request wrapper."""
    request_id: UUID = Field(default_factory=uuid4)
    user_id: Optional[UUID] = None
    endpoint: str
    method: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class APIResponse(BaseModel):
    """API response wrapper."""
    request_id: UUID
    status_code: int
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
