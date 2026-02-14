"""
Pydantic schemas for Agent API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


# ==================== Request Schemas ====================

class CapabilitySchema(BaseModel):
    """Schema for agent capability"""
    name: str = Field(..., description="Capability name")
    description: Optional[str] = Field(default=None, description="Capability description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Capability parameters")


class ModelConfigSchema(BaseModel):
    """Schema for LLM model configuration"""
    provider: str = Field(..., description="LLM provider (openai, anthropic, etc.)")
    model_name: str = Field(..., description="Model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=4096, ge=1, description="Max tokens")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")


class BudgetLimitsSchema(BaseModel):
    """Schema for agent budget limits"""
    max_cost_usd: float = Field(..., ge=0.0, description="Maximum cost in USD")
    max_tokens: int = Field(..., ge=0, description="Maximum tokens")
    max_execution_time_seconds: int = Field(..., ge=0, description="Maximum execution time")


class RegisterAgentDefinitionRequest(BaseModel):
    """Request to register an agent definition"""
    definition_id: str = Field(..., min_length=1, max_length=255, description="Definition identifier")
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$', description="Semantic version")
    name: str = Field(..., min_length=1, max_length=255, description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: List[CapabilitySchema] = Field(..., min_items=1, description="Agent capabilities")
    model_config: ModelConfigSchema = Field(..., description="Model configuration")
    system_prompt: str = Field(..., description="System prompt template")
    budget_limits: BudgetLimitsSchema = Field(..., description="Budget limits")
    tags: List[str] = Field(default_factory=list, description="Agent tags")


class CreateAgentRequest(BaseModel):
    """Request to create an agent instance"""
    definition_id: str = Field(..., description="Agent definition ID")
    definition_version: str = Field(..., description="Agent definition version")
    user_id: Optional[UUID] = Field(default=None, description="Owner user ID")


class AssignTaskRequest(BaseModel):
    """Request to assign a task to an agent"""
    agent_id: UUID = Field(..., description="Agent ID")
    task_description: str = Field(..., description="Task description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1-10)")


class CompleteTaskRequest(BaseModel):
    """Request to complete an agent task"""
    task_id: UUID = Field(..., description="Task ID")
    outputs: Dict[str, Any] = Field(..., description="Task outputs")
    success: bool = Field(..., description="Whether task completed successfully")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


# ==================== Response Schemas ====================

class CapabilityResponse(BaseModel):
    """Response schema for capability"""
    name: str
    description: Optional[str]
    parameters: Dict[str, Any]


class ModelConfigResponse(BaseModel):
    """Response schema for model configuration"""
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float


class BudgetLimitsResponse(BaseModel):
    """Response schema for budget limits"""
    max_cost_usd: float
    max_tokens: int
    max_execution_time_seconds: int


class AgentDefinitionResponse(BaseModel):
    """Response schema for agent definition"""
    definition_id: str
    version: str
    name: str
    description: str
    capabilities: List[CapabilityResponse]
    model_config: ModelConfigResponse
    system_prompt: str
    budget_limits: BudgetLimitsResponse
    tags: List[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class AgentResponse(BaseModel):
    """Response schema for agent instance"""
    agent_id: UUID
    definition_id: str
    definition_version: str
    status: str
    user_id: Optional[UUID]
    created_at: datetime
    last_active_at: Optional[datetime]
    total_cost_usd: float
    total_tokens_used: int
    
    class Config:
        from_attributes = True


class TaskAssignmentResponse(BaseModel):
    """Response schema for task assignment"""
    task_id: UUID
    agent_id: UUID
    status: str
    assigned_at: datetime
    
    class Config:
        from_attributes = True


class AgentListResponse(BaseModel):
    """Response schema for agent list"""
    agents: List[AgentResponse]
    total: int
    page: int
    page_size: int


class AgentDefinitionListResponse(BaseModel):
    """Response schema for agent definition list"""
    definitions: List[AgentDefinitionResponse]
    total: int
    page: int
    page_size: int
