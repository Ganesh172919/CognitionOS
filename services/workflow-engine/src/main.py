"""
Workflow Engine Service.

Executes declarative workflows defined in YAML/JSON DSL.
"""

import os

# Add shared libs to path

from datetime import datetime
from typing import List, Dict, Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, status, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shared.libs.config import BaseConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)

from .models import (
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowExecutionStep,
    WorkflowExecutionInput,
    WorkflowCreateResponse,
    WorkflowExecuteResponse,
    WorkflowStatusResponse,
    WorkflowExecutionGraph,
    WorkflowGraphNode,
    WorkflowGraphEdge,
    ExecutionStatus,
    WorkflowStepType
)
from .core.dsl_parser import WorkflowDSLParser
from .core.executor import WorkflowExecutor


# Configuration
config = load_config(BaseConfig)
config.service_name = "workflow-engine"
config.port = 8010
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Workflow Engine",
    version=config.service_version,
    description="Declarative workflow execution engine with YAML/JSON DSL support"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)


# ============================================================================
# In-Memory Storage (to be replaced with database persistence)
# ============================================================================

workflows_db: Dict[str, Dict[str, WorkflowDefinition]] = {}  # {id: {version: definition}}
executions_db: Dict[UUID, WorkflowExecution] = {}
execution_steps_db: Dict[UUID, List[WorkflowExecutionStep]] = {}  # {execution_id: [steps]}

# Initialize parser and executor
parser = WorkflowDSLParser()

# Service clients (to be configured)
service_clients = {
    "tool_runner": None,  # Will be configured with actual HTTP client
    "ai_runtime": None,
    "memory_service": None,
    "agent_orchestrator": None
}

executor = WorkflowExecutor(service_clients)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "workflow-engine",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Workflow Definition Endpoints
# ============================================================================

@app.post("/workflows", response_model=WorkflowCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(workflow_data: Dict):
    """
    Create a new workflow definition from YAML/JSON DSL.

    Args:
        workflow_data: Workflow definition (parsed YAML/JSON)

    Returns:
        WorkflowCreateResponse with workflow ID and version
    """
    log = get_contextual_logger()

    try:
        # Parse workflow definition
        workflow = parser.parse_dict(workflow_data)

        # Store workflow
        if workflow.id not in workflows_db:
            workflows_db[workflow.id] = {}

        workflows_db[workflow.id][workflow.version] = workflow

        log.info(f"Workflow created: {workflow.id} v{workflow.version}")

        return WorkflowCreateResponse(
            workflow_id=workflow.id,
            version=workflow.version,
            message=f"Workflow '{workflow.name}' created successfully"
        )

    except Exception as e:
        log.error(f"Failed to create workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid workflow definition: {str(e)}"
        )


@app.get("/workflows/{workflow_id}", response_model=WorkflowDefinition)
async def get_workflow(workflow_id: str, version: Optional[str] = None):
    """
    Get workflow definition by ID.

    Args:
        workflow_id: Workflow ID
        version: Workflow version (defaults to latest)

    Returns:
        WorkflowDefinition
    """
    log = get_contextual_logger()

    if workflow_id not in workflows_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found"
        )

    versions = workflows_db[workflow_id]

    if version:
        if version not in versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' version '{version}' not found"
            )
        workflow = versions[version]
    else:
        # Get latest version
        latest_version = max(versions.keys())
        workflow = versions[latest_version]

    log.info(f"Retrieved workflow: {workflow_id} v{workflow.version}")
    return workflow


@app.get("/workflows/{workflow_id}/versions", response_model=List[str])
async def list_workflow_versions(workflow_id: str):
    """
    List all versions of a workflow.

    Args:
        workflow_id: Workflow ID

    Returns:
        List of version strings
    """
    if workflow_id not in workflows_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found"
        )

    versions = list(workflows_db[workflow_id].keys())
    return sorted(versions)


@app.get("/workflows", response_model=List[WorkflowDefinition])
async def list_workflows():
    """
    List all workflows (latest versions only).

    Returns:
        List of WorkflowDefinition objects
    """
    workflows = []

    for workflow_id, versions in workflows_db.items():
        # Get latest version
        latest_version = max(versions.keys())
        workflows.append(versions[latest_version])

    return workflows


# ============================================================================
# Workflow Execution Endpoints
# ============================================================================

@app.post("/workflows/{workflow_id}/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(workflow_id: str, execution_input: WorkflowExecutionInput):
    """
    Execute a workflow.

    Args:
        workflow_id: Workflow ID to execute
        execution_input: Execution parameters (version, inputs, user_id)

    Returns:
        WorkflowExecuteResponse with execution ID and status
    """
    log = get_contextual_logger()

    # Get workflow definition
    if workflow_id not in workflows_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' not found"
        )

    versions = workflows_db[workflow_id]
    version = execution_input.workflow_version

    if version not in versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow '{workflow_id}' version '{version}' not found"
        )

    workflow = versions[version]

    try:
        # Validate inputs
        validated_inputs = parser.validate_inputs(workflow, execution_input.inputs)

        # Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_version=version,
            inputs=validated_inputs,
            status=ExecutionStatus.PENDING,
            user_id=execution_input.user_id
        )

        # Store execution
        executions_db[execution.id] = execution
        execution_steps_db[execution.id] = []

        log.info(f"Workflow execution started: {execution.id}")

        # Execute workflow asynchronously (in background)
        # For now, we'll execute synchronously for simplicity
        # In production, this would be a background task
        try:
            execution = await executor.execute(workflow, validated_inputs, execution_input.user_id)
            executions_db[execution.id] = execution
            log.info(f"Workflow execution completed: {execution.id}")
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error = str(e)
            executions_db[execution.id] = execution
            log.error(f"Workflow execution failed: {execution.id} - {e}")

        return WorkflowExecuteResponse(
            execution_id=execution.id,
            status=execution.status,
            message=f"Workflow execution {'completed' if execution.status == ExecutionStatus.COMPLETED else 'failed'}"
        )

    except Exception as e:
        log.error(f"Failed to execute workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workflow execution failed: {str(e)}"
        )


@app.get("/executions/{execution_id}", response_model=WorkflowStatusResponse)
async def get_execution_status(execution_id: UUID):
    """
    Get workflow execution status.

    Args:
        execution_id: Execution ID

    Returns:
        WorkflowStatusResponse with execution details
    """
    if execution_id not in executions_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution '{execution_id}' not found"
        )

    execution = executions_db[execution_id]
    steps = execution_steps_db.get(execution_id, [])

    # Calculate duration
    duration_seconds = None
    if execution.started_at and execution.completed_at:
        duration = execution.completed_at - execution.started_at
        duration_seconds = int(duration.total_seconds())

    # Count completed steps
    workflow_id = execution.workflow_id
    workflow_version = execution.workflow_version
    total_steps = 0

    if workflow_id in workflows_db and workflow_version in workflows_db[workflow_id]:
        workflow = workflows_db[workflow_id][workflow_version]
        total_steps = len(workflow.steps)

    steps_completed = len([s for s in steps if s.status == ExecutionStatus.COMPLETED])

    return WorkflowStatusResponse(
        execution_id=execution.id,
        workflow_id=execution.workflow_id,
        workflow_version=execution.workflow_version,
        status=execution.status,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        duration_seconds=duration_seconds,
        steps_completed=steps_completed,
        steps_total=total_steps,
        error=execution.error
    )


@app.get("/executions/{execution_id}/graph", response_model=WorkflowExecutionGraph)
async def get_execution_graph(execution_id: UUID):
    """
    Get workflow execution graph for visualization.

    Args:
        execution_id: Execution ID

    Returns:
        WorkflowExecutionGraph with nodes and edges
    """
    if execution_id not in executions_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution '{execution_id}' not found"
        )

    execution = executions_db[execution_id]
    workflow_id = execution.workflow_id
    workflow_version = execution.workflow_version

    if workflow_id not in workflows_db or workflow_version not in workflows_db[workflow_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow definition not found"
        )

    workflow = workflows_db[workflow_id][workflow_version]
    steps = execution_steps_db.get(execution_id, [])

    # Build step status map
    step_status_map = {step.step_id: step for step in steps}

    # Build nodes
    nodes = []
    for step_def in workflow.steps:
        step_exec = step_status_map.get(step_def.id)

        node = WorkflowGraphNode(
            step_id=step_def.id,
            step_name=step_def.name or step_def.id,
            step_type=step_def.type,
            status=step_exec.status if step_exec else ExecutionStatus.PENDING,
            started_at=step_exec.started_at if step_exec else None,
            completed_at=step_exec.completed_at if step_exec else None,
            error=step_exec.error if step_exec else None
        )
        nodes.append(node)

    # Build edges (dependencies)
    edges = []
    for step_def in workflow.steps:
        for dep in step_def.depends_on:
            edge = WorkflowGraphEdge(
                from_step=dep,
                to_step=step_def.id
            )
            edges.append(edge)

    return WorkflowExecutionGraph(
        execution_id=execution_id,
        nodes=nodes,
        edges=edges
    )


@app.get("/executions", response_model=List[WorkflowStatusResponse])
async def list_executions(
    workflow_id: Optional[str] = None,
    status: Optional[ExecutionStatus] = None,
    limit: int = 100
):
    """
    List workflow executions.

    Args:
        workflow_id: Filter by workflow ID (optional)
        status: Filter by status (optional)
        limit: Maximum number of results

    Returns:
        List of WorkflowStatusResponse objects
    """
    executions = list(executions_db.values())

    # Filter by workflow_id
    if workflow_id:
        executions = [e for e in executions if e.workflow_id == workflow_id]

    # Filter by status
    if status:
        executions = [e for e in executions if e.status == status]

    # Sort by created_at (newest first)
    executions.sort(key=lambda e: e.created_at, reverse=True)

    # Limit results
    executions = executions[:limit]

    # Convert to status responses
    results = []
    for execution in executions:
        steps = execution_steps_db.get(execution.id, [])

        # Calculate duration
        duration_seconds = None
        if execution.started_at and execution.completed_at:
            duration = execution.completed_at - execution.started_at
            duration_seconds = int(duration.total_seconds())

        # Count steps
        workflow_id = execution.workflow_id
        workflow_version = execution.workflow_version
        total_steps = 0

        if workflow_id in workflows_db and workflow_version in workflows_db[workflow_id]:
            workflow = workflows_db[workflow_id][workflow_version]
            total_steps = len(workflow.steps)

        steps_completed = len([s for s in steps if s.status == ExecutionStatus.COMPLETED])

        results.append(WorkflowStatusResponse(
            execution_id=execution.id,
            workflow_id=execution.workflow_id,
            workflow_version=execution.workflow_version,
            status=execution.status,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_seconds=duration_seconds,
            steps_completed=steps_completed,
            steps_total=total_steps,
            error=execution.error
        ))

    return results


# ============================================================================
# Workflow Replay (Future Enhancement)
# ============================================================================

@app.post("/executions/{execution_id}/replay", response_model=WorkflowExecuteResponse)
async def replay_execution(execution_id: UUID):
    """
    Replay a workflow execution.

    Args:
        execution_id: Execution ID to replay

    Returns:
        WorkflowExecuteResponse with new execution ID
    """
    # TODO: Implement replay functionality
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Workflow replay not yet implemented"
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {config.service_name} on port {config.port}")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.port,
        reload=True,
        log_level=config.log_level.lower()
    )
