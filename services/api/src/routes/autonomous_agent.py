"""
API Routes for Autonomous Agent System
Exposes autonomous agent capabilities via REST API.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from infrastructure.autonomous_agent import (
    AutonomousAgentOrchestrator,
    CodeLanguage,
    TaskType
)

router = APIRouter(prefix="/api/v3/autonomous-agent", tags=["Autonomous Agent"])

# Initialize orchestrator (in production, would be dependency injection)
orchestrator = AutonomousAgentOrchestrator()


class ExecuteRequirementRequest(BaseModel):
    """Request to execute a high-level requirement"""
    requirement: str = Field(..., description="High-level requirement description")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class GenerateCodeRequest(BaseModel):
    """Request to generate code"""
    purpose: str = Field(..., description="Purpose of the code")
    language: CodeLanguage = Field(default=CodeLanguage.PYTHON, description="Programming language")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Context for generation")
    template_name: Optional[str] = Field(default=None, description="Template to use")


@router.post("/execute")
async def execute_requirement(request: ExecuteRequirementRequest):
    """
    Execute a high-level requirement autonomously

    The agent will:
    1. Analyze and decompose the requirement
    2. Create an execution plan
    3. Execute tasks autonomously
    4. Validate results
    5. Self-evaluate and iterate if needed
    """
    try:
        result = await orchestrator.execute_requirement(
            requirement=request.requirement,
            context=request.context
        )

        return {
            "success": True,
            "execution_id": result["execution_id"],
            "completed": result["success"],
            "duration_seconds": result["duration_seconds"],
            "results": result["results"],
            "validation": result["validation"],
            "evaluation": result["evaluation"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/generate-code")
async def generate_code(request: GenerateCodeRequest):
    """
    Generate code using the intelligent code generator

    Returns generated code with:
    - Production-grade implementation
    - Automated tests
    - Documentation
    - Validation results
    """
    try:
        generated = await orchestrator.code_generator.generate_code(
            purpose=request.purpose,
            language=request.language,
            context=request.context or {},
            template_name=request.template_name
        )

        return {
            "success": True,
            "code": generated.code,
            "test_code": generated.test_code,
            "documentation": generated.documentation,
            "language": generated.language.value,
            "dependencies": generated.dependencies,
            "validation": {
                "is_valid": generated.validation.is_valid if generated.validation else None,
                "quality": generated.validation.quality.value if generated.validation else None,
                "errors": generated.validation.errors if generated.validation else [],
                "warnings": generated.validation.warnings if generated.validation else [],
                "suggestions": generated.validation.suggestions if generated.validation else []
            } if generated.validation else None
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/execution/{execution_id}/status")
async def get_execution_status(execution_id: str):
    """
    Get status of an active execution
    """
    status_info = await orchestrator.get_execution_status(execution_id)

    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found"
        )

    return status_info


@router.get("/memory/context")
async def get_memory_context():
    """
    Get current memory context of the agent

    Returns statistics about agent's learning and execution history
    """
    context = orchestrator.get_memory_context()

    return {
        "success": True,
        "memory": context
    }


@router.get("/capabilities")
async def get_capabilities():
    """
    Get list of agent capabilities
    """
    return {
        "success": True,
        "capabilities": {
            "task_types": [t.value for t in TaskType],
            "code_languages": [lang.value for lang in CodeLanguage],
            "features": [
                "Autonomous task planning",
                "Multi-step execution",
                "Code generation with validation",
                "Self-evaluation and iteration",
                "Context-aware memory",
                "Hallucination detection",
                "Safety boundaries enforcement"
            ]
        }
    }


@router.get("/templates")
async def get_code_templates():
    """
    Get available code generation templates
    """
    templates = orchestrator.code_generator.templates

    return {
        "success": True,
        "templates": [
            {
                "name": template.name,
                "language": template.language.value,
                "description": template.description,
                "variables": template.variables
            }
            for template in templates.values()
        ]
    }
