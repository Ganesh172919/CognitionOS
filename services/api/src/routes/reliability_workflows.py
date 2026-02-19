"""
API Routes for Chaos Engineering and Workflow Orchestration
Exposes production reliability testing and advanced workflow capabilities.
"""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from infrastructure.reliability.chaos_engineering import (
    ChaosEngineeringFramework,
    ChaosExperimentType,
    ImpactLevel,
    ChaosTarget,
    SteadyStateHypothesis
)
from infrastructure.workflow.orchestrator import (
    WorkflowOrchestrationEngine,
    TaskType,
    TaskDefinition,
    WorkflowDefinition
)

router = APIRouter(prefix="/api/v3/reliability", tags=["Reliability & Workflows"])

# Initialize systems
chaos_framework = ChaosEngineeringFramework()
workflow_engine = WorkflowOrchestrationEngine()


# ========== Chaos Engineering Endpoints ==========

class CreateExperimentRequest(BaseModel):
    """Request to create chaos experiment"""
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Detailed description")
    experiment_type: ChaosExperimentType = Field(..., description="Type of failure to inject")
    targets: List[dict] = Field(..., description="List of targets to affect")
    impact_level: ImpactLevel = Field(..., description="Blast radius control")
    steady_state_hypothesis: List[dict] = Field(..., description="Expected normal behavior")
    duration_seconds: int = Field(default=60, description="Experiment duration")
    parameters: dict = Field(default_factory=dict, description="Experiment parameters")


@router.post("/chaos/experiments")
async def create_chaos_experiment(request: CreateExperimentRequest):
    """
    Create new chaos engineering experiment

    Defines a controlled failure injection test to validate system resilience.
    """
    try:
        # Convert dict targets to ChaosTarget objects
        targets = [
            ChaosTarget(**t) for t in request.targets
        ]

        # Convert dict hypotheses to SteadyStateHypothesis objects
        hypotheses = [
            SteadyStateHypothesis(**h) for h in request.steady_state_hypothesis
        ]

        experiment = await chaos_framework.create_experiment(
            name=request.name,
            description=request.description,
            experiment_type=request.experiment_type,
            targets=targets,
            impact_level=request.impact_level,
            steady_state_hypothesis=hypotheses,
            duration_seconds=request.duration_seconds,
            parameters=request.parameters
        )

        return {
            "success": True,
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "type": experiment.experiment_type.value,
            "status": experiment.status.value
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/chaos/experiments/{experiment_id}/run")
async def run_chaos_experiment(experiment_id: str, dry_run: bool = False):
    """
    Execute chaos experiment

    Runs controlled failure injection and monitors system behavior.
    """
    try:
        result = await chaos_framework.run_experiment(experiment_id, dry_run)

        return {
            "success": True,
            "experiment_id": result.experiment_id,
            "status": result.status.value,
            "duration_seconds": result.duration_seconds,
            "steady_state_maintained": result.steady_state_maintained,
            "recovery_time_seconds": result.recovery_time_seconds,
            "insights": result.insights,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/chaos/experiments/{experiment_id}/history")
async def get_experiment_history(experiment_id: str):
    """
    Get historical results for chaos experiment

    Returns all previous executions with results and insights.
    """
    try:
        history = await chaos_framework.get_experiment_history(experiment_id)

        return {
            "success": True,
            "experiment_id": experiment_id,
            "total_runs": len(history),
            "runs": [
                {
                    "status": r.status.value,
                    "duration_seconds": r.duration_seconds,
                    "steady_state_maintained": r.steady_state_maintained,
                    "recovery_time_seconds": r.recovery_time_seconds,
                    "started_at": r.started_at.isoformat()
                }
                for r in history
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/chaos/resilience-report")
async def get_resilience_report():
    """
    Get overall system resilience report

    Aggregates results from all chaos experiments to assess system reliability.
    """
    try:
        report = await chaos_framework.get_resilience_report()

        return {
            "success": True,
            **report
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ========== Workflow Orchestration Endpoints ==========

class RegisterWorkflowRequest(BaseModel):
    """Request to register workflow definition"""
    workflow_id: str = Field(..., description="Unique workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    tasks: List[dict] = Field(..., description="List of task definitions")
    initial_task_id: str = Field(..., description="Starting task ID")
    variables: dict = Field(default_factory=dict, description="Initial variables")
    max_execution_time_seconds: Optional[int] = Field(None, description="Max execution time")
    enable_compensation: bool = Field(default=True, description="Enable SAGA compensation")


@router.post("/workflows/register")
async def register_workflow(request: RegisterWorkflowRequest):
    """
    Register workflow definition

    Defines a complex workflow with branching, parallel tasks, and compensation.
    """
    try:
        # Convert dict tasks to TaskDefinition objects
        tasks = [
            TaskDefinition(
                task_id=t["task_id"],
                name=t["name"],
                task_type=TaskType(t["task_type"]),
                action=t.get("action"),
                parameters=t.get("parameters", {}),
                retry_policy=t.get("retry_policy"),
                timeout_seconds=t.get("timeout_seconds"),
                compensation_action=t.get("compensation_action"),
                depends_on=t.get("depends_on", []),
                condition=t.get("condition"),
                branches=t.get("branches", {}),
                parallel_tasks=t.get("parallel_tasks", [])
            )
            for t in request.tasks
        ]

        workflow = WorkflowDefinition(
            workflow_id=request.workflow_id,
            name=request.name,
            description=request.description,
            version=request.version,
            tasks=tasks,
            initial_task_id=request.initial_task_id,
            variables=request.variables,
            max_execution_time_seconds=request.max_execution_time_seconds,
            enable_compensation=request.enable_compensation
        )

        await workflow_engine.register_workflow(workflow)

        return {
            "success": True,
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "version": workflow.version,
            "task_count": len(workflow.tasks)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/workflows/{workflow_id}/start")
async def start_workflow(workflow_id: str, input_data: Optional[dict] = None):
    """
    Start workflow execution

    Begins executing workflow with provided input data.
    """
    try:
        execution = await workflow_engine.start_workflow(workflow_id, input_data)

        return {
            "success": True,
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/workflows/executions/{execution_id}/status")
async def get_workflow_status(execution_id: str):
    """
    Get workflow execution status

    Returns current status and progress of workflow execution.
    """
    try:
        status_info = await workflow_engine.get_workflow_status(execution_id)

        return {
            "success": True,
            **status_info
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/workflows/executions/{execution_id}/pause")
async def pause_workflow(execution_id: str):
    """
    Pause workflow execution

    Pauses workflow, allowing it to be resumed later.
    """
    try:
        await workflow_engine.pause_workflow(execution_id)

        return {
            "success": True,
            "execution_id": execution_id,
            "status": "paused"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/workflows/executions/{execution_id}/resume")
async def resume_workflow(execution_id: str):
    """
    Resume paused workflow

    Resumes workflow execution from where it was paused.
    """
    try:
        await workflow_engine.resume_workflow(execution_id)

        return {
            "success": True,
            "execution_id": execution_id,
            "status": "resumed"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/workflows/executions/{execution_id}/cancel")
async def cancel_workflow(execution_id: str):
    """
    Cancel workflow execution

    Cancels running workflow.
    """
    try:
        await workflow_engine.cancel_workflow(execution_id)

        return {
            "success": True,
            "execution_id": execution_id,
            "status": "cancelled"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/workflows/metrics")
async def get_workflow_metrics():
    """
    Get workflow execution metrics

    Returns aggregated metrics across all workflow executions.
    """
    try:
        metrics = await workflow_engine.get_workflow_metrics()

        return {
            "success": True,
            **metrics
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for reliability systems"""
    return {
        "success": True,
        "service": "reliability-workflows",
        "components": {
            "chaos_engineering": "operational",
            "workflow_orchestration": "operational"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
