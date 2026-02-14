"""
Pydantic schemas for Phase 3 API endpoints.

These schemas handle request/response validation and serialization
for Checkpoints, Health Monitoring, and Cost Governance.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ==================== Checkpoint Schemas ====================

class CreateCheckpointRequest(BaseModel):
    """Request to create a checkpoint"""
    workflow_execution_id: UUID = Field(..., description="Workflow execution identifier")
    execution_variables: Dict[str, Any] = Field(default_factory=dict, description="Execution variables")
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    current_step_id: Optional[str] = Field(default=None, description="Current step identifier")
    completed_steps: List[str] = Field(default_factory=list, description="Completed steps")
    pending_steps: List[str] = Field(default_factory=list, description="Pending steps")
    failed_steps: List[str] = Field(default_factory=list, description="Failed steps")
    skipped_steps: List[str] = Field(default_factory=list, description="Skipped steps")
    total_steps: int = Field(..., ge=0, description="Total number of steps")
    allocated_budget: float = Field(..., ge=0.0, description="Allocated budget")
    consumed_budget: float = Field(..., ge=0.0, description="Consumed budget")
    memory_snapshot_ref: Optional[str] = Field(default=None, description="Memory snapshot reference")
    active_tasks: Optional[List[Dict[str, Any]]] = Field(default=None, description="Active tasks")
    compression_enabled: bool = Field(default=True, description="Enable compression")
    error_state: Optional[Dict[str, Any]] = Field(default=None, description="Error state")

    model_config = ConfigDict(from_attributes=True)


class RestoreCheckpointRequest(BaseModel):
    """Request to restore a checkpoint"""
    recovery_reason: str = Field(..., min_length=1, description="Reason for recovery")

    model_config = ConfigDict(from_attributes=True)


class CheckpointResponse(BaseModel):
    """Response schema for checkpoint"""
    checkpoint_id: UUID
    workflow_execution_id: UUID
    checkpoint_number: int
    status: str
    completion_percentage: float
    budget_consumed: float
    checkpoint_size_bytes: Optional[int]
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class CheckpointListResponse(BaseModel):
    """Response schema for checkpoint list"""
    checkpoints: List[CheckpointResponse]
    total: int


# ==================== Health Monitoring Schemas ====================

class RecordHeartbeatRequest(BaseModel):
    """Request to record agent heartbeat"""
    agent_id: UUID = Field(..., description="Agent identifier")
    workflow_execution_id: UUID = Field(..., description="Workflow execution identifier")
    memory_usage_mb: float = Field(..., ge=0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage")
    working_memory_count: int = Field(..., ge=0, description="Working memory count")
    episodic_memory_count: int = Field(..., ge=0, description="Episodic memory count")
    cost_consumed: float = Field(..., ge=0.0, description="Cost consumed")
    budget_remaining: float = Field(..., ge=0.0, description="Budget remaining")
    active_tasks_count: int = Field(..., ge=0, description="Active tasks count")
    completed_tasks_count: int = Field(..., ge=0, description="Completed tasks count")
    failed_tasks_count: int = Field(..., ge=0, description="Failed tasks count")

    model_config = ConfigDict(from_attributes=True)


class CreateHealthIncidentRequest(BaseModel):
    """Request to create health incident"""
    agent_id: UUID = Field(..., description="Agent identifier")
    workflow_execution_id: UUID = Field(..., description="Workflow execution identifier")
    severity: str = Field(..., description="Incident severity (low, medium, high, critical)")
    title: str = Field(..., min_length=1, max_length=255, description="Incident title")
    description: str = Field(..., min_length=1, description="Incident description")
    error_message: Optional[str] = Field(default=None, description="Error message")
    health_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Health score")
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Failure rate")

    @field_validator('severity')
    @classmethod
    def validate_severity(cls, severity: str) -> str:
        """Validate severity value"""
        valid_severities = ['low', 'medium', 'high', 'critical']
        if severity not in valid_severities:
            raise ValueError(f"Severity must be one of: {', '.join(valid_severities)}")
        return severity

    model_config = ConfigDict(from_attributes=True)


class DetectFailuresRequest(BaseModel):
    """Request to detect health failures"""
    threshold_seconds: int = Field(default=30, ge=1, description="Heartbeat staleness threshold in seconds")

    model_config = ConfigDict(from_attributes=True)


class HealthStatusResponse(BaseModel):
    """Response schema for health status"""
    agent_id: UUID
    workflow_execution_id: UUID
    status: str
    health_score: float
    last_heartbeat: str
    memory_usage_mb: float
    cpu_usage_percent: float
    cost_consumed: float
    budget_remaining: float
    active_tasks_count: int
    completed_tasks_count: int
    failed_tasks_count: int
    error_message: Optional[str]
    recovery_attempts: int

    model_config = ConfigDict(from_attributes=True)


class HealthIncidentResponse(BaseModel):
    """Response schema for health incident"""
    incident_id: UUID
    agent_id: UUID
    workflow_execution_id: UUID
    severity: str
    status: str
    title: str
    description: str
    error_message: Optional[str]
    health_score: float
    failure_rate: float
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class HealthStatusListResponse(BaseModel):
    """Response schema for health status list"""
    health_statuses: List[HealthStatusResponse]
    total: int


# ==================== Cost Governance Schemas ====================

class CreateBudgetRequest(BaseModel):
    """Request to create workflow budget"""
    workflow_execution_id: UUID = Field(..., description="Workflow execution identifier")
    allocated_budget: float = Field(..., gt=0.0, description="Allocated budget")
    currency: str = Field(default="USD", pattern=r'^[A-Z]{3}$', description="Currency code (ISO 4217)")
    warning_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Warning threshold (0-1)")
    critical_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Critical threshold (0-1)")

    @field_validator('critical_threshold')
    @classmethod
    def validate_thresholds(cls, critical_threshold: float, info) -> float:
        """Validate critical threshold is greater than warning threshold"""
        warning_threshold = info.data.get('warning_threshold', 0.8)
        if critical_threshold <= warning_threshold:
            raise ValueError("Critical threshold must be greater than warning threshold")
        return critical_threshold

    model_config = ConfigDict(from_attributes=True)


class RecordCostRequest(BaseModel):
    """Request to record cost entry"""
    workflow_execution_id: UUID = Field(..., description="Workflow execution identifier")
    operation_type: str = Field(..., description="Operation type (llm_inference, memory_storage, compute, data_transfer, tool_usage)")
    provider: str = Field(..., min_length=1, description="Provider name")
    cost: float = Field(..., ge=0.0, description="Cost amount")
    agent_id: Optional[UUID] = Field(default=None, description="Agent identifier")
    model: Optional[str] = Field(default=None, description="Model name")
    tokens_used: Optional[int] = Field(default=None, ge=0, description="Tokens used")
    execution_time_ms: Optional[int] = Field(default=None, ge=0, description="Execution time in milliseconds")
    memory_bytes: Optional[int] = Field(default=None, ge=0, description="Memory bytes")
    task_id: Optional[UUID] = Field(default=None, description="Task identifier")
    step_name: Optional[str] = Field(default=None, description="Step name")
    currency: str = Field(default="USD", pattern=r'^[A-Z]{3}$', description="Currency code (ISO 4217)")

    @field_validator('operation_type')
    @classmethod
    def validate_operation_type(cls, operation_type: str) -> str:
        """Validate operation type value"""
        valid_types = ['llm_inference', 'memory_storage', 'compute', 'data_transfer', 'tool_usage']
        if operation_type not in valid_types:
            raise ValueError(f"Operation type must be one of: {', '.join(valid_types)}")
        return operation_type

    model_config = ConfigDict(from_attributes=True)


class BudgetResponse(BaseModel):
    """Response schema for budget"""
    budget_id: UUID
    workflow_execution_id: UUID
    allocated_budget: float
    consumed_budget: float
    remaining_budget: float
    currency: str
    status: str
    usage_percentage: float
    warning_threshold: float
    critical_threshold: float
    is_exhausted: bool
    is_suspended: bool
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class CostEntryResponse(BaseModel):
    """Response schema for cost entry"""
    cost_entry_id: UUID
    workflow_execution_id: UUID
    operation_type: str
    provider: str
    cost: float
    agent_id: Optional[UUID]
    model: Optional[str]
    tokens_used: Optional[int]
    consumed_budget: float
    remaining_budget: float
    usage_percentage: float
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class CostSummaryResponse(BaseModel):
    """Response schema for cost summary"""
    workflow_execution_id: UUID
    total_cost: float
    budget_exists: bool
    allocated_budget: Optional[float]
    consumed_budget: Optional[float]
    remaining_budget: Optional[float]
    usage_percentage: Optional[float]
    status: Optional[str]
    currency: Optional[str]
    is_exhausted: Optional[bool]
    is_suspended: Optional[bool]
    cost_by_operation_type: Optional[Dict[str, float]]
    cost_by_agent: Optional[Dict[str, float]]

    model_config = ConfigDict(from_attributes=True)


class EnforceBudgetRequest(BaseModel):
    """Request to enforce budget limits"""
    workflow_execution_id: UUID = Field(..., description="Workflow execution identifier")
    
    model_config = ConfigDict(from_attributes=True)


class EnforceBudgetResponse(BaseModel):
    """Response schema for budget enforcement"""
    status: str
    events_raised: int
    suspended: bool
    should_halt: bool

    model_config = ConfigDict(from_attributes=True)
