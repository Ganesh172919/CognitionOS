"""
Pydantic schemas for Workflow API endpoints.

These schemas handle request/response validation and serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ==================== Request Schemas ====================

class WorkflowStepSchema(BaseModel):
    """Schema for workflow step"""

    model_config = ConfigDict(extra="ignore")

    # Accept both V3 shapes and legacy test payload shapes.
    step_id: str = Field(
        ...,
        description="Step identifier",
        validation_alias=AliasChoices("step_id", "id"),
    )
    type: str = Field(
        default="task",
        description="Step type (task, http_request, etc.)",
        validation_alias=AliasChoices("type", "agent_capability"),
    )
    name: Optional[str] = Field(default=None, description="Step name")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step parameters/inputs",
        validation_alias=AliasChoices("params", "inputs"),
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Step dependencies",
        validation_alias=AliasChoices("depends_on", "dependencies"),
    )
    timeout_seconds: int = Field(default=300, description="Step timeout in seconds")
    retry_count: int = Field(default=0, description="Retry count")
    condition: Optional[str] = Field(default=None, description="Optional execution condition")
    agent_role: Optional[str] = Field(default=None, description="Optional agent role")

    # Legacy nested config object used by tests (e.g., {"timeout": 300, "retry_count": 3})
    config: Optional[Dict[str, Any]] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_config(cls, data: Any) -> Any:
        """Support legacy step payloads that store timeout/retry in a nested config object."""
        if not isinstance(data, dict):
            return data

        config = data.get("config")
        if not isinstance(config, dict):
            return data

        if "timeout_seconds" not in data and "timeout" in config:
            data["timeout_seconds"] = config.get("timeout")
        if "retry_count" not in data and "retry_count" in config:
            data["retry_count"] = config.get("retry_count")

        return data


class CreateWorkflowRequest(BaseModel):
    """Request to create a new workflow"""

    model_config = ConfigDict(extra="ignore")

    workflow_id: str = Field(..., min_length=1, max_length=255, description="Workflow identifier")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version (e.g., 1.0.0)")
    name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
    description: str = Field(..., description="Workflow description")
    steps: List[WorkflowStepSchema] = Field(..., min_length=1, description="Workflow steps")
    schedule: Optional[str] = Field(default=None, description="Cron schedule expression")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    created_by: Optional[str] = Field(default=None, description="Creator identifier")

    # Legacy envelope used by tests.
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional workflow metadata")

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_metadata(cls, data: Any) -> Any:
        """Support legacy payloads where created_by lives under metadata.created_by."""
        if not isinstance(data, dict):
            return data

        if data.get("created_by") is None and isinstance(data.get("metadata"), dict):
            created_by = data["metadata"].get("created_by")
            if created_by:
                data["created_by"] = created_by

        return data
    
    @field_validator('steps')
    @classmethod
    def validate_steps(cls, steps: List[WorkflowStepSchema]) -> List[WorkflowStepSchema]:
        """Validate steps have unique IDs"""
        step_ids = [step.step_id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Step IDs must be unique")
        return steps


class ExecuteWorkflowRequest(BaseModel):
    """Request to execute a workflow"""
    workflow_id: str = Field(..., description="Workflow identifier")
    workflow_version: Optional[str] = Field(
        default=None,
        description="Workflow version (defaults to latest)",
        validation_alias=AliasChoices("workflow_version", "version"),
    )
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Workflow inputs")
    user_id: Optional[UUID] = Field(default=None, description="User identifier")


class UpdateWorkflowStatusRequest(BaseModel):
    """Request to update workflow status"""
    status: str = Field(..., description="New status (draft, active, deprecated, archived)")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, status: str) -> str:
        """Validate status value"""
        valid_statuses = ['draft', 'active', 'deprecated', 'archived']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return status


# ==================== Response Schemas ====================

class WorkflowStepResponse(BaseModel):
    """Response schema for workflow step"""
    step_id: str
    name: str
    type: str
    params: Dict[str, Any]
    depends_on: List[str]
    timeout_seconds: int
    retry_count: int
    condition: Optional[str] = None
    agent_role: Optional[str] = None


class WorkflowResponse(BaseModel):
    """Response schema for workflow"""
    workflow_id: str
    version: str
    name: str
    description: str
    status: str
    steps: List[WorkflowStepResponse]
    schedule: Optional[str]
    tags: List[str]
    created_by: Optional[str]
    created_at: datetime
    # Domain workflow doesn't currently track updated_at; keep optional for compatibility.
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class WorkflowExecutionResponse(BaseModel):
    """Response schema for workflow execution"""
    execution_id: UUID
    workflow_id: str
    workflow_version: str
    status: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    error: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    user_id: Optional[UUID]
    
    model_config = ConfigDict(from_attributes=True)


class StepExecutionResponse(BaseModel):
    """Response schema for step execution"""
    step_execution_id: UUID
    execution_id: UUID
    step_id: str
    status: str
    agent_id: Optional[UUID]
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    error: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    retry_count: int
    
    model_config = ConfigDict(from_attributes=True)


class WorkflowListResponse(BaseModel):
    """Response schema for workflow list"""
    workflows: List[WorkflowResponse]
    total: int
    page: int
    page_size: int


class WorkflowExecutionListResponse(BaseModel):
    """Response schema for workflow execution list"""
    executions: List[WorkflowExecutionResponse]
    total: int
    page: int
    page_size: int


# ==================== Common Response Schemas ====================

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(..., description="Current timestamp")
    database: str = Field(..., description="Database status")
    redis: str = Field(..., description="Redis status")
    rabbitmq: str = Field(..., description="RabbitMQ status")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for debugging")


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = Field(default=True, description="Success flag")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
