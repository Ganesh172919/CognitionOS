"""
Pydantic schemas for Workflow API endpoints.

These schemas handle request/response validation and serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ==================== Request Schemas ====================

class WorkflowStepSchema(BaseModel):
    """Schema for workflow step"""
    step_id: str = Field(..., description="Step identifier")
    name: str = Field(..., description="Step name")
    agent_capability: str = Field(..., description="Required agent capability")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Step inputs")
    depends_on: List[str] = Field(default_factory=list, description="Step dependencies")
    timeout_seconds: Optional[int] = Field(default=300, description="Step timeout")
    retry_config: Optional[Dict[str, Any]] = Field(default=None, description="Retry configuration")


class CreateWorkflowRequest(BaseModel):
    """Request to create a new workflow"""
    workflow_id: str = Field(..., min_length=1, max_length=255, description="Workflow identifier")
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$', description="Semantic version (e.g., 1.0.0)")
    name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
    description: str = Field(..., description="Workflow description")
    steps: List[WorkflowStepSchema] = Field(..., min_items=1, description="Workflow steps")
    schedule: Optional[str] = Field(default=None, description="Cron schedule expression")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    created_by: Optional[str] = Field(default=None, description="Creator identifier")
    
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
    workflow_version: str = Field(..., description="Workflow version")
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
    agent_capability: str
    inputs: Dict[str, Any]
    depends_on: List[str]
    timeout_seconds: int
    retry_config: Optional[Dict[str, Any]]


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
    updated_at: datetime
    
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
