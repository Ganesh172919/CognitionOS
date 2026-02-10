"""
Workflow Engine - Pydantic Models

Defines the data models for workflows, executions, and steps.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class WorkflowInputType(str, Enum):
    """Input parameter types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    JSON = "json"


class WorkflowOutputType(str, Enum):
    """Output parameter types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    JSON = "json"


class WorkflowStepType(str, Enum):
    """Workflow step types"""
    # Task execution
    EXECUTE_TASK = "execute_task"
    EXECUTE_PYTHON = "execute_python"
    EXECUTE_JAVASCRIPT = "execute_javascript"

    # HTTP operations
    HTTP_REQUEST = "http_request"
    API_CALL = "api_call"

    # Database operations
    QUERY_DATABASE = "query_database"
    UPDATE_DATABASE = "update_database"

    # File operations
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"

    # Git operations
    GIT_CLONE = "git_clone"
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"

    # Docker operations
    DOCKER_BUILD = "docker_build"
    DOCKER_RUN = "docker_run"

    # Kubernetes operations
    KUBERNETES_APPLY = "kubernetes_apply"
    KUBERNETES_DELETE = "kubernetes_delete"

    # AI operations
    AI_GENERATE = "ai_generate"
    AI_EMBEDDING = "ai_embedding"
    AI_VALIDATE = "ai_validate"

    # Memory operations
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"

    # Workflow control
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"

    # Notifications
    SEND_EMAIL = "send_email"
    SEND_SLACK = "send_slack"
    SEND_ALERT = "send_alert"

    # Custom
    CUSTOM = "custom"


class ExecutionStatus(str, Enum):
    """Workflow and step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class AgentRole(str, Enum):
    """Agent roles for step execution"""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"
    CUSTOM = "custom"


# ==================== Workflow Definition Models ====================

class WorkflowInput(BaseModel):
    """Workflow input parameter definition"""
    name: str = Field(..., description="Parameter name")
    type: WorkflowInputType = Field(..., description="Parameter type")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value")
    description: Optional[str] = Field(default=None, description="Parameter description")
    values: Optional[List[Any]] = Field(default=None, description="Allowed values (for enum type)")


class WorkflowOutput(BaseModel):
    """Workflow output definition"""
    name: str = Field(..., description="Output name")
    type: WorkflowOutputType = Field(..., description="Output type")
    description: Optional[str] = Field(default=None, description="Output description")


class WorkflowStep(BaseModel):
    """Workflow step definition"""
    id: str = Field(..., description="Unique step ID within workflow")
    type: WorkflowStepType = Field(..., description="Step type")
    name: Optional[str] = Field(default=None, description="Human-readable step name")
    description: Optional[str] = Field(default=None, description="Step description")

    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Step IDs this step depends on")

    # Parameters (can include template variables like ${{ inputs.repo_url }})
    params: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")

    # Agent assignment
    agent_role: Optional[AgentRole] = Field(default=None, description="Agent role to execute this step")

    # Execution settings
    timeout: Optional[str] = Field(default="300s", description="Timeout (e.g., '60s', '5m')")
    retry: int = Field(default=0, description="Number of retries on failure")
    retry_delay: Optional[str] = Field(default="5s", description="Delay between retries")

    # Conditional execution
    condition: Optional[str] = Field(default=None, description="Condition for execution (template expression)")

    # Approval
    approval_required: bool = Field(default=False, description="Whether this step requires manual approval")


class WorkflowDefinition(BaseModel):
    """Complete workflow definition (DSL)"""
    id: str = Field(..., description="Unique workflow ID")
    version: str = Field(..., description="Workflow version (e.g., '1.0.0')")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(default=None, description="Workflow description")

    # Schedule (cron expression for periodic workflows)
    schedule: Optional[str] = Field(default=None, description="Cron schedule (e.g., '0 2 * * *')")

    # Inputs and outputs
    inputs: List[WorkflowInput] = Field(default_factory=list, description="Input parameters")
    outputs: List[WorkflowOutput] = Field(default_factory=list, description="Output definitions")

    # Steps
    steps: List[WorkflowStep] = Field(..., description="Workflow steps (DAG)")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    created_by: Optional[str] = Field(default=None, description="Creator user ID")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")


# ==================== Workflow Execution Models ====================

class WorkflowExecutionInput(BaseModel):
    """Input values for workflow execution"""
    workflow_id: str = Field(..., description="Workflow ID to execute")
    workflow_version: str = Field(..., description="Workflow version to execute")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameter values")
    user_id: Optional[UUID] = Field(default=None, description="User executing the workflow")


class WorkflowExecution(BaseModel):
    """Workflow execution instance"""
    id: UUID = Field(default_factory=uuid4, description="Unique execution ID")
    workflow_id: str = Field(..., description="Workflow ID")
    workflow_version: str = Field(..., description="Workflow version")

    # Inputs and status
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output values (populated on completion)")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="Execution status")

    # Timing
    started_at: Optional[datetime] = Field(default=None, description="Execution start time")
    completed_at: Optional[datetime] = Field(default=None, description="Execution completion time")

    # Error handling
    error: Optional[str] = Field(default=None, description="Error message if failed")

    # User context
    user_id: Optional[UUID] = Field(default=None, description="User who triggered execution")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class WorkflowExecutionStep(BaseModel):
    """Individual step execution within a workflow execution"""
    id: UUID = Field(default_factory=uuid4, description="Unique step execution ID")
    execution_id: UUID = Field(..., description="Parent workflow execution ID")

    # Step reference
    step_id: str = Field(..., description="Step ID from workflow definition")
    step_type: WorkflowStepType = Field(..., description="Step type")

    # Status
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="Step status")

    # Timing
    started_at: Optional[datetime] = Field(default=None, description="Step start time")
    completed_at: Optional[datetime] = Field(default=None, description="Step completion time")

    # Results
    output: Optional[Dict[str, Any]] = Field(default=None, description="Step output")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    # Agent assignment
    agent_id: Optional[UUID] = Field(default=None, description="Agent that executed this step")

    # Retry tracking
    retry_count: int = Field(default=0, description="Number of retries attempted")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


# ==================== Workflow Response Models ====================

class WorkflowCreateResponse(BaseModel):
    """Response for workflow creation"""
    workflow_id: str
    version: str
    message: str = "Workflow created successfully"


class WorkflowExecuteResponse(BaseModel):
    """Response for workflow execution start"""
    execution_id: UUID
    status: ExecutionStatus
    message: str = "Workflow execution started"

    class Config:
        json_encoders = {UUID: lambda v: str(v)}


class WorkflowStatusResponse(BaseModel):
    """Response for workflow execution status"""
    execution_id: UUID
    workflow_id: str
    workflow_version: str
    status: ExecutionStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[int]
    steps_completed: int
    steps_total: int
    error: Optional[str]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class WorkflowGraphNode(BaseModel):
    """Node in workflow execution graph"""
    step_id: str
    step_name: str
    step_type: WorkflowStepType
    status: ExecutionStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class WorkflowGraphEdge(BaseModel):
    """Edge in workflow execution graph"""
    from_step: str
    to_step: str


class WorkflowExecutionGraph(BaseModel):
    """Complete execution graph for visualization"""
    execution_id: UUID
    nodes: List[WorkflowGraphNode]
    edges: List[WorkflowGraphEdge]

    class Config:
        json_encoders = {UUID: lambda v: str(v)}
