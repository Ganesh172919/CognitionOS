"""
API Schemas Package
"""

from .workflows import (
    CreateWorkflowRequest,
    ExecuteWorkflowRequest,
    UpdateWorkflowStatusRequest,
    WorkflowResponse,
    WorkflowExecutionResponse,
    StepExecutionResponse,
    WorkflowListResponse,
    WorkflowExecutionListResponse,
    HealthCheckResponse,
    ErrorResponse,
    SuccessResponse,
)

from .agents import (
    RegisterAgentDefinitionRequest,
    CreateAgentRequest,
    AssignTaskRequest,
    CompleteTaskRequest,
    AgentDefinitionResponse,
    AgentResponse,
    TaskAssignmentResponse,
    AgentListResponse,
    AgentDefinitionListResponse,
)

__all__ = [
    # Workflow schemas
    "CreateWorkflowRequest",
    "ExecuteWorkflowRequest",
    "UpdateWorkflowStatusRequest",
    "WorkflowResponse",
    "WorkflowExecutionResponse",
    "StepExecutionResponse",
    "WorkflowListResponse",
    "WorkflowExecutionListResponse",
    # Agent schemas
    "RegisterAgentDefinitionRequest",
    "CreateAgentRequest",
    "AssignTaskRequest",
    "CompleteTaskRequest",
    "AgentDefinitionResponse",
    "AgentResponse",
    "TaskAssignmentResponse",
    "AgentListResponse",
    "AgentDefinitionListResponse",
    # Common schemas
    "HealthCheckResponse",
    "ErrorResponse",
    "SuccessResponse",
]
