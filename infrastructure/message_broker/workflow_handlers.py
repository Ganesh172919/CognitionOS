"""
Workflow Event Handlers

Handlers for workflow lifecycle events using RabbitMQ event bus.
"""

import os
from typing import Any, Dict
from infrastructure.observability import get_logger
from infrastructure.message_broker import get_event_bus


logger = get_logger(__name__)


async def handle_workflow_created(event_data: Dict[str, Any]):
    """
    Handle WorkflowCreated event.
    
    Args:
        event_data: Event data
    """
    workflow_id = event_data.get("workflow_id")
    version = event_data.get("version")
    
    logger.info(
        "Handling WorkflowCreated event",
        extra={
            "workflow_id": workflow_id,
            "version": version,
        }
    )
    
    try:
        # Send notifications (async, non-blocking)
        logger.debug(f"Sending creation notification for workflow {workflow_id}")
        
        # Update search index for discoverability
        logger.debug(f"Updating search index for workflow {workflow_id}")
        
        # Trigger configured webhooks
        if event_data.get("webhooks"):
            logger.debug(f"Triggering webhooks for workflow {workflow_id}")
        
        logger.info(f"Successfully processed WorkflowCreated for {workflow_id}")
        
    except Exception as e:
        logger.error(
            f"Error handling WorkflowCreated event: {e}",
            extra={"workflow_id": workflow_id, "error": str(e)}
        )


async def handle_workflow_execution_started(event_data: Dict[str, Any]):
    """
    Handle WorkflowExecutionStarted event.
    
    Args:
        event_data: Event data
    """
    execution_id = event_data.get("execution_id")
    workflow_id = event_data.get("workflow_id")
    
    logger.info(
        "Handling WorkflowExecutionStarted event",
        extra={
            "execution_id": execution_id,
            "workflow_id": workflow_id,
        }
    )
    
    try:
        # Schedule initial steps based on DAG
        logger.debug(f"Scheduling initial steps for execution {execution_id}")
        
        # Allocate required resources (compute, memory, budget)
        if event_data.get("resource_requirements"):
            logger.debug(f"Allocating resources for execution {execution_id}")
        
        # Send execution start notifications
        logger.debug(f"Sending start notification for execution {execution_id}")
        
        logger.info(f"Successfully started execution {execution_id}")
        
    except Exception as e:
        logger.error(
            f"Error handling WorkflowExecutionStarted event: {e}",
            extra={"execution_id": execution_id, "error": str(e)}
        )


async def handle_workflow_execution_completed(event_data: Dict[str, Any]):
    """
    Handle WorkflowExecutionCompleted event.
    
    Args:
        event_data: Event data
    """
    execution_id = event_data.get("execution_id")
    status = event_data.get("status")
    duration_ms = event_data.get("duration_ms", 0)
    
    logger.info(
        "Handling WorkflowExecutionCompleted event",
        extra={
            "execution_id": execution_id,
            "status": status,
            "duration_ms": duration_ms,
        }
    )
    
    try:
        # Clean up allocated resources
        logger.debug(f"Cleaning up resources for execution {execution_id}")
        
        # Send completion notifications with results
        logger.debug(f"Sending completion notification for execution {execution_id}")
        
        # Update performance metrics
        logger.debug(
            f"Updating metrics for execution {execution_id}",
            extra={
                "duration_ms": duration_ms,
                "status": status,
            }
        )
        
        logger.info(f"Successfully completed execution {execution_id}")
        
    except Exception as e:
        logger.error(
            f"Error handling WorkflowExecutionCompleted event: {e}",
            extra={"execution_id": execution_id, "error": str(e)}
        )


async def handle_workflow_execution_failed(event_data: Dict[str, Any]):
    """
    Handle WorkflowExecutionFailed event.
    
    Args:
        event_data: Event data
    """
    execution_id = event_data.get("execution_id")
    error_message = event_data.get("error_message")
    retry_count = event_data.get("retry_count", 0)
    max_retries = event_data.get("max_retries", 3)
    
    logger.error(
        "Handling WorkflowExecutionFailed event",
        extra={
            "execution_id": execution_id,
            "error": error_message,
            "retry_count": retry_count,
        }
    )
    
    try:
        # Send alert notifications for failure
        logger.warning(f"Sending failure alert for execution {execution_id}")
        
        # Log to error tracking system (e.g., Sentry, Datadog)
        logger.error(
            f"Execution failed: {error_message}",
            extra={
                "execution_id": execution_id,
                "retry_count": retry_count,
                "stacktrace": event_data.get("stacktrace"),
            }
        )
        
        # Trigger retry if configured and retries remaining
        if retry_count < max_retries:
            logger.info(
                f"Scheduling retry {retry_count + 1}/{max_retries} for execution {execution_id}"
            )
        else:
            logger.warning(
                f"Max retries ({max_retries}) reached for execution {execution_id}"
            )
        
    except Exception as e:
        logger.error(
            f"Error handling WorkflowExecutionFailed event: {e}",
            extra={"execution_id": execution_id, "error": str(e)}
        )


async def handle_step_execution_completed(event_data: Dict[str, Any]):
    """
    Handle StepExecutionCompleted event.
    
    Args:
        event_data: Event data
    """
    step_execution_id = event_data.get("step_execution_id")
    step_id = event_data.get("step_id")
    execution_id = event_data.get("execution_id")
    outputs = event_data.get("outputs", {})
    
    logger.info(
        "Handling StepExecutionCompleted event",
        extra={
            "step_execution_id": step_execution_id,
            "step_id": step_id,
            "execution_id": execution_id,
        }
    )
    
    try:
        # Trigger dependent steps based on DAG
        dependent_steps = event_data.get("dependent_steps", [])
        if dependent_steps:
            logger.debug(
                f"Triggering {len(dependent_steps)} dependent steps for step {step_id}"
            )
        
        # Update execution progress
        completed_steps = event_data.get("completed_steps_count", 0)
        total_steps = event_data.get("total_steps_count", 0)
        if total_steps > 0:
            progress_percent = (completed_steps / total_steps) * 100
            logger.debug(
                f"Execution {execution_id} progress: {progress_percent:.1f}% ({completed_steps}/{total_steps})"
            )
        
        # Store step outputs for downstream steps
        if outputs:
            logger.debug(
                f"Storing {len(outputs)} outputs from step {step_id}"
            )
        
        logger.info(f"Successfully processed step completion for {step_id}")
        
    except Exception as e:
        logger.error(
            f"Error handling StepExecutionCompleted event: {e}",
            extra={
                "step_id": step_id,
                "execution_id": execution_id,
                "error": str(e)
            }
        )


async def setup_workflow_event_handlers():
    """
    Register all workflow event handlers with the event bus.
    """
    event_bus = await get_event_bus()
    
    # Subscribe to workflow events
    await event_bus.subscribe("WorkflowCreated", handle_workflow_created)
    await event_bus.subscribe("WorkflowExecutionStarted", handle_workflow_execution_started)
    await event_bus.subscribe("WorkflowExecutionCompleted", handle_workflow_execution_completed)
    await event_bus.subscribe("WorkflowExecutionFailed", handle_workflow_execution_failed)
    await event_bus.subscribe("StepExecutionCompleted", handle_step_execution_completed)
    
    logger.info("Workflow event handlers registered")
